"""Model audit — systematic comparison of parametric vs nonparametric VaR.

The XGBoost conditional quantile model audits parametric VaR estimates
(Normal, GARCH, MS-GARCH) by flagging dates and regimes where they disagree.
Disagreement reveals what the parametric model is missing.
"""

import numpy as np
import pandas as pd


def detect_disagreements(
    parametric_var: pd.Series,
    xgb_var: pd.Series,
    threshold_pct: float = 0.20,
) -> pd.DataFrame:
    """Flag dates where parametric and XGB VaR diverge significantly.

    Parameters
    ----------
    parametric_var : time-indexed series of parametric VaR estimates.
    xgb_var : time-indexed series of XGB VaR estimates (same index).
    threshold_pct : flag dates where relative divergence exceeds this
        fraction of the parametric VaR magnitude.

    Returns DataFrame with columns: parametric_var, xgb_var,
    abs_diff, rel_diff, xgb_deeper (True if XGB predicts a bigger loss).
    Only rows exceeding the threshold are returned.
    """
    aligned = pd.DataFrame({
        "parametric_var": parametric_var,
        "xgb_var": xgb_var,
    }).dropna()

    aligned["abs_diff"] = (aligned["xgb_var"] - aligned["parametric_var"]).abs()
    aligned["rel_diff"] = aligned["abs_diff"] / aligned["parametric_var"].abs().clip(lower=1e-10)
    aligned["xgb_deeper"] = aligned["xgb_var"] < aligned["parametric_var"]

    return aligned[aligned["rel_diff"] > threshold_pct].copy()


def regime_conditional_breach_rates(
    backtest_results: pd.DataFrame,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """Split backtest breach rates by HMM regime.

    Parameters
    ----------
    backtest_results : DataFrame with 'breach' column (bool) and DatetimeIndex.
    regime_labels : Series of integer regime labels aligned to the same dates.

    Returns DataFrame indexed by regime with columns: n_obs, n_breaches,
    breach_rate.
    """
    bt = backtest_results[["breach"]].copy()
    bt["regime"] = regime_labels.reindex(bt.index)
    bt = bt.dropna(subset=["regime"])
    bt["regime"] = bt["regime"].astype(int)

    grouped = bt.groupby("regime")["breach"]
    result = pd.DataFrame({
        "n_obs": grouped.count(),
        "n_breaches": grouped.sum().astype(int),
    })
    result["breach_rate"] = result["n_breaches"] / result["n_obs"]
    return result


def comparative_var_table(
    returns: pd.Series,
    var_estimates: dict[str, pd.Series],
) -> pd.DataFrame:
    """Build a multi-model VaR comparison table.

    Parameters
    ----------
    returns : actual daily returns (DatetimeIndex).
    var_estimates : dict mapping model name → Series of VaR estimates,
        all on the same DatetimeIndex as returns.

    Returns DataFrame with one column per model plus 'actual_return'
    and per-model 'breach_*' columns.
    """
    df = pd.DataFrame({"actual_return": returns})
    for name, var_series in var_estimates.items():
        col = name.lower().replace(" ", "_").replace("+", "_")
        df[f"var_{col}"] = var_series.reindex(returns.index)
        df[f"breach_{col}"] = returns < var_series.reindex(returns.index)

    return df.dropna()
