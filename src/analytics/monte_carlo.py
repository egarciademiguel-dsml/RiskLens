"""Monte Carlo simulation for forward-looking risk estimation."""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from src.analytics.vol_constant import generate_log_returns as constant_log_returns
from src.analytics.vol_garch import generate_log_returns as garch_log_returns
from src.analytics.vol_garch import fit_garch
from src.analytics.regime_hmm import generate_log_returns as hmm_log_returns
from src.analytics.regime_hmm import fit_hmm, predict_current_regime, get_regime_params
from src.analytics.regime_gmm import generate_log_returns as gmm_log_returns
from src.analytics.regime_gmm import fit_gmm
from src.analytics.regime_gmm import predict_current_regime as gmm_predict_current_regime
from src.analytics.regime_gmm import get_regime_params as gmm_get_regime_params
from src.analytics.vol_rvol import generate_log_returns as rvol_log_returns
from src.analytics.vol_rvol import fit_rvol, predict_current_vol

TRADING_DAYS_PER_YEAR = 252

# Registry: model name → log-return generator
_VOLATILITY_MODELS = {
    "constant": constant_log_returns,
    "garch": garch_log_returns,
    "hmm": hmm_log_returns,
    "gmm": gmm_log_returns,
    "rvol": rvol_log_returns,
}


def simulate_paths(
    close: pd.Series,
    returns: pd.Series,
    n_days: int = 252,
    n_simulations: int = 10_000,
    seed: int | None = None,
    distribution: str = "normal",
    df_t: float | None = None,
    volatility_model: str = "constant",
    **model_kwargs,
) -> pd.DataFrame:
    """Simulate forward price paths.

    Returns DataFrame of shape (n_days, n_simulations) with simulated prices.

    distribution: "normal" (Gaussian shocks) or "t" (Student-t, fat tails).
    df_t: degrees of freedom for Student-t. Auto-fitted from returns if None.
    volatility_model: key into the model registry ("constant", "garch", "hmm", ...).
    **model_kwargs: passed directly to the chosen model's generate_log_returns().
    """
    if distribution not in ("normal", "t"):
        raise ValueError(f"Unknown distribution '{distribution}'. Use 'normal' or 't'.")
    if volatility_model not in _VOLATILITY_MODELS:
        known = ", ".join(sorted(_VOLATILITY_MODELS))
        raise ValueError(f"Unknown volatility_model '{volatility_model}'. Available: {known}.")

    last_price = close.iloc[-1]

    # Generate unit shocks (mean=0, var=1)
    if distribution == "t":
        if df_t is None:
            df_fit, _, _ = t_dist.fit(returns.dropna())
            df_t = max(df_fit, 2.1)
        raw = t_dist.rvs(df=df_t, size=(n_days, n_simulations), random_state=seed)
        shocks = raw / np.sqrt(df_t / (df_t - 2))
    else:
        if seed is not None:
            np.random.seed(seed)
        shocks = np.random.normal(0, 1, size=(n_days, n_simulations))

    # Delegate to the chosen model
    model_fn = _VOLATILITY_MODELS[volatility_model]
    log_returns = model_fn(shocks, returns, seed=seed, **model_kwargs)

    paths = last_price * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(paths)


def fit_t_distribution(returns: pd.Series) -> dict:
    """Fit Student-t distribution to historical returns via MLE."""
    clean = returns.dropna()
    df_fit, loc_fit, scale_fit = t_dist.fit(clean)
    df_clamped = max(df_fit, 2.1)

    if df_clamped < 4:
        tail_desc = "Very heavy tails"
    elif df_clamped < 8:
        tail_desc = "Moderately heavy tails"
    elif df_clamped < 30:
        tail_desc = "Mild fat tails"
    else:
        tail_desc = "Near-normal"

    return {
        "df": float(df_clamped),
        "df_raw": float(df_fit),
        "loc": float(loc_fit),
        "scale": float(scale_fit),
        "n_observations": len(clean),
        "tail_description": tail_desc,
    }


def compute_var(
    final_prices: pd.Series,
    initial_price: float,
    confidence: float = 0.95,
) -> float:
    """Value at Risk as percent loss at given confidence level."""
    returns = (final_prices - initial_price) / initial_price
    return float(np.percentile(returns, (1 - confidence) * 100))


def compute_cvar(
    final_prices: pd.Series,
    initial_price: float,
    confidence: float = 0.95,
) -> float:
    """Conditional VaR (Expected Shortfall) — average loss beyond VaR."""
    returns = (final_prices - initial_price) / initial_price
    var = np.percentile(returns, (1 - confidence) * 100)
    return float(returns[returns <= var].mean())


def prob_target(
    final_prices: pd.Series,
    initial_price: float,
    target_pct: float,
) -> float:
    """Probability of reaching a specific return target (e.g., 0.20 for +20%).

    Uses strict inequality (>) for consistency with prob_gain in simulation_summary.
    """
    returns = (final_prices - initial_price) / initial_price
    return float((returns > target_pct).mean())


def scenario_buckets(
    final_prices: pd.Series,
    initial_price: float,
) -> dict:
    """Bin simulation outcomes into intuitive scenario categories."""
    returns = (final_prices - initial_price) / initial_price
    n = len(returns)
    return {
        "severe_loss (<-30%)": float((returns < -0.30).sum() / n),
        "moderate_loss (-30% to -10%)": float(((returns >= -0.30) & (returns < -0.10)).sum() / n),
        "flat (-10% to +10%)": float(((returns >= -0.10) & (returns < 0.10)).sum() / n),
        "moderate_gain (+10% to +30%)": float(((returns >= 0.10) & (returns < 0.30)).sum() / n),
        "strong_gain (>+30%)": float((returns >= 0.30).sum() / n),
    }


def simulation_summary(
    final_prices: pd.Series,
    initial_price: float,
    confidence: float = 0.95,
) -> dict:
    """Full summary of simulation results."""
    returns = (final_prices - initial_price) / initial_price
    return {
        "initial_price": initial_price,
        "mean_final_price": float(final_prices.mean()),
        "median_final_price": float(final_prices.median()),
        "min_final_price": float(final_prices.min()),
        "max_final_price": float(final_prices.max()),
        "var": compute_var(final_prices, initial_price, confidence),
        "cvar": compute_cvar(final_prices, initial_price, confidence),
        "confidence": confidence,
        "prob_gain": float((returns > 0).mean()),
        "prob_loss": float((returns < 0).mean()),
        "scenarios": scenario_buckets(final_prices, initial_price),
    }
