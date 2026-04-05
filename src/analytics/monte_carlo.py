"""Monte Carlo simulation for forward-looking risk estimation (GBM-based)."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def simulate_paths(
    close: pd.Series,
    returns: pd.Series,
    n_days: int = 252,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate forward price paths using geometric Brownian motion.

    Returns DataFrame of shape (n_days, n_simulations) with simulated prices.
    """
    if seed is not None:
        np.random.seed(seed)

    mu = returns.mean()
    sigma = returns.std()
    last_price = close.iloc[-1]

    # GBM: S(t+1) = S(t) * exp((mu - 0.5*sigma^2) + sigma*Z)
    drift = mu - 0.5 * sigma**2
    shocks = np.random.normal(0, 1, size=(n_days, n_simulations))
    log_returns = drift + sigma * shocks

    # Cumulative sum of log returns, then exponentiate
    paths = last_price * np.exp(np.cumsum(log_returns, axis=0))

    return pd.DataFrame(paths)


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
