"""VaR backtesting — rolling-window validation with Kupiec and Christoffersen tests."""

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.analytics.monte_carlo import (
    simulate_paths, compute_var, fit_garch, fit_hmm, fit_gmm, fit_rvol,
)


# ---------------------------------------------------------------------------
# Predefined fit functions: returns_window → simulate_paths kwargs
# ---------------------------------------------------------------------------

def constant_fit(returns_window, seed):
    return {}


def garch_fit(returns_window, seed):
    return {"volatility_model": "garch", "garch_params": fit_garch(returns_window)}


def hmm_fit(returns_window, seed, n_regimes=2):
    return {"volatility_model": "hmm", "hmm_params": fit_hmm(returns_window, n_regimes, seed)}


def gmm_fit(returns_window, seed, n_regimes=2):
    return {"volatility_model": "gmm", "gmm_params": fit_gmm(returns_window, n_regimes, seed)}


def rvol_fit(returns_window, seed, horizon=21):
    return {"volatility_model": "rvol", "rvol_params": fit_rvol(returns_window, horizon, seed)}


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------

def backtest_var(
    close: pd.Series,
    returns: pd.Series,
    fit_fn=constant_fit,
    train_window: int = 252,
    confidence: float = 0.95,
    n_simulations: int = 2_000,
    seed: int = 42,
    step: int = 1,
) -> pd.DataFrame:
    """Rolling-window VaR backtest.

    Parameters
    ----------
    close, returns : historical price / return series (same index).
    fit_fn : callable(returns_window, seed) → dict of simulate_paths kwargs.
    train_window : number of days in the training window.
    confidence : VaR confidence level (e.g. 0.95).
    n_simulations : MC paths per window (lower = faster).
    seed : random seed.
    step : test every N-th day (1 = every day, 5 = weekly).

    Returns
    -------
    DataFrame with columns: actual_return, predicted_var, breach.
    """
    results = []
    test_indices = range(train_window, len(returns) - 1, step)

    for t in test_indices:
        close_window = close.iloc[:t + 1]
        returns_window = returns.iloc[:t + 1]

        try:
            model_kwargs = fit_fn(returns_window, seed)
        except Exception:
            continue

        try:
            paths = simulate_paths(
                close_window, returns_window,
                n_days=1, n_simulations=n_simulations, seed=seed,
                **model_kwargs,
            )
            final_prices = paths.iloc[-1]
            initial_price = close_window.iloc[-1]
            var = compute_var(final_prices, initial_price, confidence)
        except Exception:
            continue

        actual = returns.iloc[t + 1]
        results.append({
            "date": returns.index[t + 1],
            "actual_return": actual,
            "predicted_var": var,
            "breach": actual < var,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.set_index("date")
    return df


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def kupiec_test(n_obs: int, n_breaches: int, confidence: float = 0.95) -> dict:
    """Kupiec (1995) unconditional coverage test.

    Returns dict with expected_rate, observed_rate, lr_statistic, p_value, pass.
    """
    p0 = 1 - confidence
    p_hat = n_breaches / n_obs if n_obs > 0 else 0.0

    # Edge cases
    if n_breaches == 0:
        lr = -2 * (n_obs * np.log(1 - p0)) + 2 * (n_obs * np.log(1))
        lr = -2 * n_obs * np.log(1 - p0)
    elif n_breaches == n_obs:
        lr = -2 * n_obs * np.log(p0)
    else:
        log_l0 = (n_obs - n_breaches) * np.log(1 - p0) + n_breaches * np.log(p0)
        log_l1 = (n_obs - n_breaches) * np.log(1 - p_hat) + n_breaches * np.log(p_hat)
        lr = -2 * (log_l0 - log_l1)

    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "expected_rate": p0,
        "observed_rate": p_hat,
        "lr_statistic": float(lr),
        "p_value": float(p_value),
        "pass": bool(p_value >= 0.05),
    }


def christoffersen_test(breaches: np.ndarray) -> dict:
    """Christoffersen (1998) independence test on breach sequence.

    Parameters
    ----------
    breaches : 1-D boolean/int array of breach indicators.

    Returns dict with lr_statistic, p_value, pass.
    """
    b = np.asarray(breaches, dtype=int)
    if len(b) < 2:
        return {"lr_statistic": 0.0, "p_value": 1.0, "pass": True}

    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(len(b) - 1):
        prev, curr = b[i], b[i + 1]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    # Avoid division by zero
    total_0 = n00 + n01
    total_1 = n10 + n11
    total = total_0 + total_1

    if total == 0 or total_0 == 0:
        return {"lr_statistic": 0.0, "p_value": 1.0, "pass": True}

    p_hat = (n01 + n11) / total  # unconditional breach rate
    pi01 = n01 / total_0 if total_0 > 0 else 0.0
    pi11 = n11 / total_1 if total_1 > 0 else 0.0

    # Log-likelihoods
    def safe_log(x):
        return np.log(max(x, 1e-15))

    log_l0 = 0.0
    if n00 > 0:
        log_l0 += n00 * safe_log(1 - p_hat)
    if n01 > 0:
        log_l0 += n01 * safe_log(p_hat)
    if n10 > 0:
        log_l0 += n10 * safe_log(1 - p_hat)
    if n11 > 0:
        log_l0 += n11 * safe_log(p_hat)

    log_l1 = 0.0
    if n00 > 0:
        log_l1 += n00 * safe_log(1 - pi01)
    if n01 > 0:
        log_l1 += n01 * safe_log(pi01)
    if n10 > 0:
        log_l1 += n10 * safe_log(1 - pi11)
    if n11 > 0:
        log_l1 += n11 * safe_log(pi11)

    lr = -2 * (log_l0 - log_l1)
    lr = max(lr, 0.0)  # numerical safety

    p_value = 1 - chi2.cdf(lr, df=1)

    return {
        "lr_statistic": float(lr),
        "p_value": float(p_value),
        "pass": bool(p_value >= 0.05),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def backtest_summary(results: pd.DataFrame, confidence: float = 0.95) -> dict:
    """Compute full backtest summary from backtest_var output."""
    n_obs = len(results)
    n_breaches = int(results["breach"].sum())

    kupiec = kupiec_test(n_obs, n_breaches, confidence)
    christoffersen = christoffersen_test(results["breach"].values)

    return {
        "n_obs": n_obs,
        "n_breaches": n_breaches,
        "breach_rate": n_breaches / n_obs if n_obs > 0 else 0.0,
        "expected_rate": 1 - confidence,
        "kupiec": kupiec,
        "christoffersen": christoffersen,
    }
