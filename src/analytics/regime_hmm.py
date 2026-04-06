"""HMM-based regime detection for drift and volatility estimation."""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def fit_hmm(returns: pd.Series, n_regimes: int = 2, n_iter: int = 200,
            seed: int = 42) -> dict:
    """Fit a Gaussian HMM on daily returns.

    Returns dict with:
      - model: fitted GaussianHMM
      - n_regimes: number of regimes
      - regime_params: list of {mu, sigma} per regime, sorted by sigma ascending
      - transition_matrix: (n_regimes, n_regimes) transition probabilities
      - regime_labels: array of regime assignments for each observation
      - sort_order: index mapping from raw HMM states to sorted states
    """
    if n_regimes < 1:
        raise ValueError("n_regimes must be >= 1.")

    clean = returns.dropna().values.reshape(-1, 1)

    if n_regimes == 1:
        mu = float(np.mean(clean))
        sigma = float(np.std(clean, ddof=1))
        return {
            "model": None,
            "n_regimes": 1,
            "regime_params": [{"mu": mu, "sigma": sigma}],
            "transition_matrix": np.array([[1.0]]),
            "regime_labels": np.zeros(len(clean), dtype=int),
            "sort_order": np.array([0]),
        }

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=seed,
    )
    model.fit(clean)

    raw_labels = model.predict(clean)

    # Extract per-regime mu and sigma
    raw_params = []
    for i in range(n_regimes):
        raw_params.append({
            "mu": float(model.means_[i, 0]),
            "sigma": float(np.sqrt(model.covars_[i, 0, 0])),
        })

    # Sort regimes by sigma ascending (regime 0 = calmest)
    sort_order = np.argsort([p["sigma"] for p in raw_params])
    regime_params = [raw_params[i] for i in sort_order]

    # Remap labels and transition matrix to sorted order
    inv_map = np.empty(n_regimes, dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        inv_map[old_idx] = new_idx
    regime_labels = inv_map[raw_labels]

    raw_transmat = model.transmat_
    transition_matrix = raw_transmat[np.ix_(sort_order, sort_order)]

    return {
        "model": model,
        "n_regimes": n_regimes,
        "regime_params": regime_params,
        "transition_matrix": transition_matrix,
        "regime_labels": regime_labels,
        "sort_order": sort_order,
    }


def predict_current_regime(hmm_result: dict, recent_returns: pd.Series) -> int:
    """Predict the current regime from the most recent returns window."""
    if hmm_result["n_regimes"] == 1:
        return 0

    model = hmm_result["model"]
    clean = recent_returns.dropna().values.reshape(-1, 1)
    raw_labels = model.predict(clean)
    raw_current = raw_labels[-1]

    # Remap to sorted order
    sort_order = hmm_result["sort_order"]
    inv_map = np.empty(len(sort_order), dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        inv_map[old_idx] = new_idx

    return int(inv_map[raw_current])


def get_regime_params(hmm_result: dict, regime_idx: int | None = None) -> dict | list:
    """Get drift (mu) and volatility (sigma) for a specific regime or all regimes.

    If regime_idx is None, returns the full list.
    """
    if regime_idx is None:
        return hmm_result["regime_params"]
    return hmm_result["regime_params"][regime_idx]


def generate_log_returns(
    shocks: np.ndarray,
    returns: pd.Series,
    seed: int | None = None,
    hmm_params: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """Log returns using HMM regime-switching drift and volatility."""
    if hmm_params is None:
        hmm_params = fit_hmm(returns, n_regimes=2, seed=seed or 42)

    current_regime = predict_current_regime(hmm_params, returns)
    n_days, n_simulations = shocks.shape

    if hmm_params["n_regimes"] == 1:
        p = hmm_params["regime_params"][0]
        drift = p["mu"] - 0.5 * p["sigma"]**2
        return drift + p["sigma"] * shocks

    trans_mat = hmm_params["transition_matrix"]
    n_regimes = hmm_params["n_regimes"]
    all_mu = np.array([p["mu"] for p in hmm_params["regime_params"]])
    all_sigma = np.array([p["sigma"] for p in hmm_params["regime_params"]])

    rng = np.random.RandomState(seed)
    log_returns = np.empty_like(shocks)
    regimes = np.full(n_simulations, current_regime, dtype=int)

    for t_step in range(n_days):
        sig = all_sigma[regimes]
        mu_r = all_mu[regimes]
        drift_t = mu_r - 0.5 * sig**2
        log_returns[t_step] = drift_t + sig * shocks[t_step]

        u = rng.rand(n_simulations)
        cum_probs = np.cumsum(trans_mat[regimes], axis=1)
        regimes = (u[:, None] > cum_probs).sum(axis=1)
        regimes = np.clip(regimes, 0, n_regimes - 1)

    return log_returns
