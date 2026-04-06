"""Monte Carlo simulation for forward-looking risk estimation."""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t as t_dist

TRADING_DAYS_PER_YEAR = 252


def simulate_paths(
    close: pd.Series,
    returns: pd.Series,
    n_days: int = 252,
    n_simulations: int = 10_000,
    seed: int | None = None,
    distribution: str = "normal",
    df_t: float | None = None,
    volatility_model: str = "constant",
    garch_params: dict | None = None,
) -> pd.DataFrame:
    """Simulate forward price paths.

    Returns DataFrame of shape (n_days, n_simulations) with simulated prices.

    distribution: "normal" (Gaussian shocks) or "t" (Student-t, fat tails).
    df_t: degrees of freedom for Student-t. Auto-fitted from returns if None.
    volatility_model: "constant" (flat sigma) or "garch" (GARCH(1,1) time-varying).
    garch_params: pre-fitted GARCH params dict. Auto-fitted from returns if None.
    """
    if distribution not in ("normal", "t"):
        raise ValueError(f"Unknown distribution '{distribution}'. Use 'normal' or 't'.")
    if volatility_model not in ("constant", "garch"):
        raise ValueError(f"Unknown volatility_model '{volatility_model}'. Use 'constant' or 'garch'.")

    mu = returns.mean()
    sigma = returns.std()
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

    if volatility_model == "garch":
        if garch_params is None:
            garch_params = fit_garch(returns)
        omega = garch_params["omega"]
        alpha = garch_params["alpha"]
        beta = garch_params["beta"]
        prev_var = garch_params["last_variance"]
        prev_resid = garch_params["last_resid"]

        # Day-by-day variance simulation
        log_returns = np.empty_like(shocks)
        for t in range(n_days):
            curr_var = omega + alpha * prev_resid**2 + beta * prev_var
            sigma_t = np.sqrt(curr_var)
            drift_t = mu - 0.5 * curr_var
            log_returns[t] = drift_t + sigma_t * shocks[t]
            prev_resid = sigma_t * shocks[t]  # realized shock for next step
            prev_var = curr_var
    else:
        drift = mu - 0.5 * sigma**2
        log_returns = drift + sigma * shocks

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


def fit_garch(returns: pd.Series) -> dict:
    """Fit GARCH(1,1) to historical returns. Returns params for simulation."""
    clean = (returns.dropna() * 100)  # arch expects percentage returns
    model = arch_model(clean, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    res = model.fit(disp="off")

    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]

    # Long-run variance (annualized vol from it)
    long_run_var = omega / (1 - alpha - beta)
    long_run_vol = np.sqrt(long_run_var * TRADING_DAYS_PER_YEAR) / 100  # back to decimal

    # Last conditional variance and residual for simulation seeding
    last_var = float(res.conditional_volatility.iloc[-1] ** 2)
    last_resid = float(clean.iloc[-1] - res.params["mu"])

    # Convert back to decimal scale for simulation
    return {
        "omega": omega / 1e4,
        "alpha": float(alpha),
        "beta": float(beta),
        "long_run_vol": float(long_run_vol),
        "last_variance": last_var / 1e4,
        "last_resid": last_resid / 100,
        "persistence": float(alpha + beta),
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
