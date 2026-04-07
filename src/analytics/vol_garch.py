"""GARCH(1,1) volatility model."""

import numpy as np
import pandas as pd
from arch import arch_model

from src.config import TRADING_DAYS_PER_YEAR


def fit_garch(returns: pd.Series) -> dict:
    """Fit GARCH(1,1) to historical returns. Returns params for simulation."""
    clean = (returns.dropna() * 100)  # arch expects percentage returns
    model = arch_model(clean, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
    res = model.fit(disp="off")

    omega = res.params["omega"]
    alpha = res.params["alpha[1]"]
    beta = res.params["beta[1]"]

    long_run_var = omega / (1 - alpha - beta)
    long_run_vol = np.sqrt(long_run_var * TRADING_DAYS_PER_YEAR) / 100

    last_var = float(res.conditional_volatility.iloc[-1] ** 2)
    last_resid = float(clean.iloc[-1] - res.params["mu"])

    return {
        "omega": omega / 1e4,
        "alpha": float(alpha),
        "beta": float(beta),
        "long_run_vol": float(long_run_vol),
        "last_variance": last_var / 1e4,
        "last_resid": last_resid / 100,
        "persistence": float(alpha + beta),
    }


def generate_log_returns(
    shocks: np.ndarray,
    returns: pd.Series,
    seed: int | None = None,
    garch_params: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """Log returns with GARCH(1,1) time-varying volatility."""
    if garch_params is None:
        garch_params = fit_garch(returns)

    mu = returns.mean()
    omega = garch_params["omega"]
    alpha = garch_params["alpha"]
    beta = garch_params["beta"]
    prev_var = garch_params["last_variance"]
    prev_resid = garch_params["last_resid"]

    n_days, n_simulations = shocks.shape
    log_returns = np.empty_like(shocks)

    for t in range(n_days):
        curr_var = omega + alpha * prev_resid**2 + beta * prev_var
        sigma_t = np.sqrt(curr_var)
        drift_t = mu - 0.5 * curr_var
        log_returns[t] = drift_t + sigma_t * shocks[t]
        prev_resid = sigma_t * shocks[t]
        prev_var = curr_var

    return log_returns
