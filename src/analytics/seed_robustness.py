"""Multi-seed robustness analysis for Monte Carlo risk estimates."""

import numpy as np
import pandas as pd

from src.analytics.monte_carlo import simulate_paths, compute_var, compute_cvar


def run_multi_seed(
    close: pd.Series,
    returns: pd.Series,
    n_days: int = 252,
    n_simulations: int = 10_000,
    seeds: list[int] | None = None,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    volatility_model: str = "constant",
    **model_kwargs,
) -> pd.DataFrame:
    """Run simulations across multiple seeds and collect VaR/CVaR per seed.

    Parameters
    ----------
    close, returns : historical price / return series.
    n_days : simulation horizon.
    n_simulations : paths per seed.
    seeds : list of random seeds (default: 10 seeds from 0-90).
    confidence_levels : confidence levels to compute VaR/CVaR for.
    volatility_model : model name passed to simulate_paths.
    **model_kwargs : extra kwargs passed to simulate_paths.

    Returns
    -------
    DataFrame with one row per seed, columns for each VaR/CVaR metric.
    """
    if seeds is None:
        seeds = list(range(0, 100, 10))

    initial_price = close.iloc[-1]
    rows = []

    for seed in seeds:
        paths = simulate_paths(
            close, returns,
            n_days=n_days, n_simulations=n_simulations,
            seed=seed, volatility_model=volatility_model,
            **model_kwargs,
        )
        fp = paths.iloc[-1]

        row = {"seed": seed}
        for conf in confidence_levels:
            row[f"var_{conf:.0%}"] = compute_var(fp, initial_price, conf)
            row[f"cvar_{conf:.0%}"] = compute_cvar(fp, initial_price, conf)
        rows.append(row)

    return pd.DataFrame(rows).set_index("seed")


def robustness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize multi-seed results: mean, std, cv, min, max per metric.

    Parameters
    ----------
    df : output of run_multi_seed.

    Returns
    -------
    DataFrame with summary statistics per metric.
    """
    summary = pd.DataFrame({
        "mean": df.mean(),
        "std": df.std(),
        "cv": (df.std() / df.mean().abs()),
        "min": df.min(),
        "max": df.max(),
        "range": df.max() - df.min(),
    })
    return summary
