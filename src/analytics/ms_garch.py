"""Markov-Switching GARCH with EVT tails.

Top-tier model combining:
  - HMM regime detection (discrete state identification)
  - Per-regime GARCH(1,1) (state-dependent volatility dynamics)
  - Per-regime EVT/GPD tails (state-dependent extreme risk)

Innovation distribution: semi-parametric (Normal bulk + GPD left tail) per regime.
"""

import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm

from src.analytics.regime_hmm import fit_hmm, predict_current_regime
from src.analytics.vol_garch import fit_garch
from src.analytics.evt import fit_gpd


# ---------------------------------------------------------------------------
# EVT shock generation
# ---------------------------------------------------------------------------

def generate_evt_shocks(gpd_params: dict, size: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate standardized shocks with EVT-shaped left tail.

    Semi-parametric approach: Normal bulk + GPD left tail.
    GPD was fitted to standardized residuals, so parameters are in shock-scale.
    """
    xi = gpd_params["shape"]
    beta = gpd_params["scale"]
    threshold = gpd_params["threshold"]
    p_exceed = gpd_params["n_exceedances"] / gpd_params["n_total"]

    u_uniform = rng.uniform(0, 1, size=size)
    shocks = np.empty(size)

    # Bulk: inverse Normal CDF
    bulk_mask = u_uniform >= p_exceed
    shocks[bulk_mask] = norm.ppf(u_uniform[bulk_mask])

    # Left tail: inverse GPD, shifted by threshold, negated (loss direction)
    tail_mask = ~bulk_mask
    if tail_mask.any():
        # Rescale [0, p_exceed) → (0, 1) for GPD quantile function
        u_tail = u_uniform[tail_mask] / p_exceed
        # Clip to avoid 0 and 1 at boundaries
        u_tail = np.clip(u_tail, 1e-10, 1 - 1e-10)
        gpd_quantiles = genpareto.ppf(1 - u_tail, c=xi, loc=0, scale=beta)
        shocks[tail_mask] = -(threshold + gpd_quantiles)

    return shocks


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_ms_garch(
    returns: pd.Series,
    n_regimes: int = 2,
    seed: int = 42,
    threshold_quantile: float = 0.95,
    min_regime_obs: int = 100,
) -> dict:
    """Fit Markov-Switching GARCH with per-regime EVT tails.

    Algorithm:
      1. HMM detects regimes from historical returns
      2. Per regime: fit GARCH(1,1) on regime-filtered returns
      3. Per regime: fit GPD on GARCH standardized residuals (left tail)
      4. Fallbacks for small regimes: constant vol and/or Normal innovations

    Parameters
    ----------
    returns : daily log-return series.
    n_regimes : number of HMM regimes (typically 2).
    seed : random seed for HMM fitting.
    threshold_quantile : GPD threshold quantile (applied to standardized residuals).
    min_regime_obs : minimum observations to fit GARCH per regime.

    Returns dict with hmm_result, per-regime GARCH params, per-regime GPD params.
    """
    hmm_result = fit_hmm(returns, n_regimes=n_regimes, seed=seed)
    current_regime = predict_current_regime(hmm_result, returns)

    # Align returns to HMM labels
    clean_returns = returns.dropna()
    labels = hmm_result["regime_labels"]
    aligned_returns = clean_returns.iloc[:len(labels)]

    regime_garch = []
    regime_gpd = []
    regime_mu = []

    for k in range(n_regimes):
        mask = labels == k
        r_k = aligned_returns[mask]
        n_k = len(r_k)
        mu_k = float(r_k.mean()) if n_k > 0 else 0.0
        regime_mu.append(mu_k)

        # --- GARCH per regime ---
        if n_k >= min_regime_obs:
            try:
                garch_k = fit_garch(r_k)
            except Exception:
                # GARCH failed to converge: fall back to constant vol
                sigma_k = float(r_k.std()) if n_k > 1 else float(clean_returns.std())
                garch_k = _constant_garch_fallback(sigma_k)
        else:
            # Too few observations: degenerate GARCH (constant vol)
            sigma_k = float(r_k.std()) if n_k > 1 else float(clean_returns.std())
            garch_k = _constant_garch_fallback(sigma_k)

        regime_garch.append(garch_k)

        # --- GPD per regime (on standardized residuals) ---
        if n_k >= 50:
            try:
                # Compute standardized residuals for this regime
                resid_k = _compute_standardized_residuals(r_k, garch_k, mu_k)
                # Fit GPD on the loss tail of standardized residuals
                gpd_k = fit_gpd(pd.Series(-resid_k), threshold_quantile)
                regime_gpd.append(gpd_k)
            except Exception:
                regime_gpd.append(None)
        else:
            regime_gpd.append(None)

    return {
        "hmm_result": hmm_result,
        "n_regimes": n_regimes,
        "regime_garch": regime_garch,
        "regime_gpd": regime_gpd,
        "regime_mu": regime_mu,
        "current_regime": current_regime,
        "min_regime_obs": min_regime_obs,
    }


def _constant_garch_fallback(sigma: float) -> dict:
    """Degenerate GARCH params (alpha=0, beta=0) that behave as constant vol."""
    return {
        "omega": sigma**2,
        "alpha": 0.0,
        "beta": 0.0,
        "long_run_vol": float(sigma * np.sqrt(252)),
        "last_variance": sigma**2,
        "last_resid": 0.0,
        "persistence": 0.0,
    }


def _compute_standardized_residuals(
    returns: pd.Series, garch_params: dict, mu: float
) -> np.ndarray:
    """Compute GARCH standardized residuals: (r_t - mu) / sigma_t."""
    omega = garch_params["omega"]
    alpha = garch_params["alpha"]
    beta = garch_params["beta"]
    prev_var = garch_params["last_variance"]

    r = returns.values
    residuals = np.empty(len(r))

    # Run GARCH forward to get conditional variances
    # For regime-filtered (non-contiguous) returns, we treat them as contiguous
    curr_var = prev_var
    for t in range(len(r)):
        sigma_t = np.sqrt(curr_var)
        residuals[t] = (r[t] - mu) / sigma_t if sigma_t > 0 else 0.0
        raw_resid = r[t] - mu
        curr_var = omega + alpha * raw_resid**2 + beta * curr_var

    return residuals


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def generate_log_returns(
    n_days: int,
    n_simulations: int,
    returns: pd.Series,
    seed: int | None = None,
    ms_garch_params: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """Log returns with Markov-Switching GARCH + EVT tails.

    Per-timestep:
      1. GARCH variance update using current regime's (omega, alpha, beta)
      2. Generate shock: EVT if regime has fitted GPD, else Normal
      3. Compute log return: drift + sigma_t * shock
      4. Transition regime via HMM transition matrix

    Variance state (prev_var, prev_resid) carries across regime transitions
    for physical continuity.
    """
    if ms_garch_params is None:
        ms_garch_params = fit_ms_garch(returns, seed=seed or 42)

    n_regimes = ms_garch_params["n_regimes"]
    hmm_result = ms_garch_params["hmm_result"]
    regime_garch = ms_garch_params["regime_garch"]
    regime_gpd = ms_garch_params["regime_gpd"]
    regime_mu = np.array(ms_garch_params["regime_mu"])
    current_regime = ms_garch_params["current_regime"]
    trans_mat = hmm_result["transition_matrix"]

    rng = np.random.RandomState(seed)

    # Initialize state
    regimes = np.full(n_simulations, current_regime, dtype=int)

    # Initialize variance from current regime's GARCH
    init_garch = regime_garch[current_regime]
    prev_var = np.full(n_simulations, init_garch["last_variance"])
    prev_resid = np.full(n_simulations, init_garch["last_resid"])

    # Variance clamps per regime (10x long-run variance)
    max_var = np.array([
        10.0 * gp["omega"] / max(1.0 - gp["alpha"] - gp["beta"], 0.01)
        if gp["alpha"] + gp["beta"] < 1.0 else 10.0 * gp["last_variance"]
        for gp in regime_garch
    ])

    log_returns = np.empty((n_days, n_simulations))

    for t in range(n_days):
        # Regime transition FIRST — so day-1 VaR reflects tomorrow's regime
        u = rng.rand(n_simulations)
        cum_probs = np.cumsum(trans_mat[regimes], axis=1)
        regimes = (u[:, None] > cum_probs).sum(axis=1)
        regimes = np.clip(regimes, 0, n_regimes - 1)

        for k in range(n_regimes):
            mask = regimes == k
            n_k = mask.sum()
            if n_k == 0:
                continue

            gp = regime_garch[k]
            omega_k = gp["omega"]
            alpha_k = gp["alpha"]
            beta_k = gp["beta"]

            # GARCH variance update
            curr_var_k = omega_k + alpha_k * prev_resid[mask]**2 + beta_k * prev_var[mask]
            curr_var_k = np.clip(curr_var_k, 1e-12, max_var[k])
            sigma_k = np.sqrt(curr_var_k)

            # Generate shocks
            if regime_gpd[k] is not None:
                shock_k = generate_evt_shocks(regime_gpd[k], n_k, rng)
            else:
                shock_k = rng.normal(0, 1, n_k)

            # Log returns
            drift_k = regime_mu[k] - 0.5 * curr_var_k
            log_returns[t, mask] = drift_k + sigma_k * shock_k

            # Update GARCH state
            prev_resid[mask] = sigma_k * shock_k
            prev_var[mask] = curr_var_k

    return log_returns
