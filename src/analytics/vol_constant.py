"""Constant volatility model — baseline GBM.

Innovation distribution: Normal(0,1). This is the null model — symmetric risk,
no volatility clustering, no fat tails.
"""

import numpy as np
import pandas as pd


def generate_log_returns(
    n_days: int,
    n_simulations: int,
    returns: pd.Series,
    seed: int | None = None,
    **kwargs,
) -> np.ndarray:
    """Log returns using global mean and standard deviation (flat vol).

    Generates Normal(0,1) shocks internally — baseline Gaussian assumption.
    """
    mu = returns.mean()
    sigma = returns.std()
    drift = mu - 0.5 * sigma**2
    rng = np.random.RandomState(seed)
    shocks = rng.normal(0, 1, size=(n_days, n_simulations))
    return drift + sigma * shocks
