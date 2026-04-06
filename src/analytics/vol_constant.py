"""Constant volatility model — baseline GBM."""

import numpy as np
import pandas as pd


def generate_log_returns(
    shocks: np.ndarray,
    returns: pd.Series,
    seed: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Log returns using global mean and standard deviation (flat vol)."""
    mu = returns.mean()
    sigma = returns.std()
    drift = mu - 0.5 * sigma**2
    return drift + sigma * shocks
