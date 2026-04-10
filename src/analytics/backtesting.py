"""VaR backtesting — rolling-window validation with Kupiec and Christoffersen tests."""

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.analytics.monte_carlo import (
    simulate_paths, compute_var, fit_garch, fit_hmm, fit_gmm,
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


def ms_garch_fit(returns_window, seed, n_regimes=2):
    from src.analytics.ms_garch import fit_ms_garch
    return {"volatility_model": "ms_garch",
            "ms_garch_params": fit_ms_garch(returns_window, n_regimes, seed)}


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
    n_skipped = 0
    test_indices = range(train_window, len(returns) - 1, step)

    for t in test_indices:
        close_window = close.iloc[:t + 1]
        returns_window = returns.iloc[:t + 1]

        try:
            model_kwargs = fit_fn(returns_window, seed)
        except Exception:
            n_skipped += 1
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
            n_skipped += 1
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
    df.attrs["n_skipped"] = n_skipped
    return df


# ---------------------------------------------------------------------------
# EVT backtest
# ---------------------------------------------------------------------------

def backtest_evt_var(
    returns: pd.Series,
    confidence: float = 0.95,
    threshold_quantile: float = 0.95,
    train_window: int = 252,
    step: int = 1,
) -> pd.DataFrame:
    """Expanding-window walk-forward backtest for EVT (GPD/POT) VaR.

    At each step: fit GPD on [0:t+1], compute EVT VaR, check against returns[t+1].
    Returns DataFrame compatible with backtest_summary().
    """
    from src.analytics.evt import fit_gpd, evt_var

    results = []
    n_skipped = 0
    test_indices = range(train_window, len(returns) - 1, step)

    for t in test_indices:
        returns_window = returns.iloc[:t + 1]

        try:
            gpd_params = fit_gpd(returns_window, threshold_quantile)
            predicted = evt_var(gpd_params, confidence)
        except Exception:
            n_skipped += 1
            continue

        actual = returns.iloc[t + 1]
        results.append({
            "date": returns.index[t + 1],
            "actual_return": actual,
            "predicted_var": predicted,
            "breach": actual < predicted,
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.set_index("date")
    df.attrs["n_skipped"] = n_skipped
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

def compare_models(
    close: pd.Series,
    returns: pd.Series,
    model_configs: dict,
    train_window: int = 252,
    confidence: float = 0.95,
    n_simulations: int = 2_000,
    seed: int = 42,
    step: int = 5,
) -> dict:
    """Backtest all models and return dual ranking.

    Parameters
    ----------
    close, returns : historical price / return series.
    model_configs : dict mapping model name → fit_fn callable.
    train_window, confidence, n_simulations, seed, step : backtest parameters.

    Returns
    -------
    dict with:
      - "results": DataFrame with per-model stats, calibration_rank, conservative_rank
      - "best_calibrated": name of the best-calibrated model
      - "most_conservative": name of the most risk-conservative model
    """
    rows = []
    for name, fit_fn in model_configs.items():
        bt = backtest_var(
            close, returns, fit_fn=fit_fn, train_window=train_window,
            confidence=confidence, n_simulations=n_simulations, seed=seed, step=step,
        )
        summary = backtest_summary(bt, confidence)
        mean_var = float(bt["predicted_var"].mean()) if len(bt) > 0 else 0.0
        rows.append({
            "model": name,
            "n_obs": summary["n_obs"],
            "n_breaches": summary["n_breaches"],
            "breach_rate": summary["breach_rate"],
            "expected_rate": summary["expected_rate"],
            "kupiec_p": summary["kupiec"]["p_value"],
            "kupiec_pass": summary["kupiec"]["pass"],
            "christoffersen_p": summary["christoffersen"]["p_value"],
            "christoffersen_pass": summary["christoffersen"]["pass"],
            "mean_predicted_var": mean_var,
        })

    df = pd.DataFrame(rows).set_index("model")
    df["breach_error"] = (df["breach_rate"] - df["expected_rate"]).abs()

    # Calibration rank: lowest breach_error wins, tiebreak by highest kupiec_p
    df["calibration_rank"] = (
        df[["breach_error", "kupiec_p"]]
        .apply(lambda r: (r["breach_error"], -r["kupiec_p"]), axis=1)
        .rank(method="min").astype(int)
    )

    # Conservative rank: lowest breach_rate wins (VaR rarely breached = conservative),
    # tiebreak by most negative mean_predicted_var (deeper VaR = more conservative)
    df["conservative_rank"] = (
        df[["breach_rate", "mean_predicted_var"]]
        .apply(lambda r: (r["breach_rate"], r["mean_predicted_var"]), axis=1)
        .rank(method="min").astype(int)
    )

    best_cal = df["calibration_rank"].idxmin()
    most_cons = df["conservative_rank"].idxmin()

    return {
        "results": df,
        "best_calibrated": best_cal,
        "most_conservative": most_cons,
    }


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
        "n_skipped": results.attrs.get("n_skipped", 0),
        "kupiec": kupiec,
        "christoffersen": christoffersen,
    }
