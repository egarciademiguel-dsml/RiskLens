"""XGBoost conditional quantile regression — nonparametric VaR estimation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def engineer_features(returns: pd.Series) -> pd.DataFrame:
    """Build rolling features for conditional VaR prediction.

    Features: rolling vol (5/10/21/63d), rolling mean return (5/10/21d),
    rolling skew (21d), rolling kurtosis (21d), abs return, squared return.
    """
    df = pd.DataFrame({"returns": returns})

    for w in [5, 10, 21, 63]:
        df[f"vol_{w}d"] = returns.rolling(w).std()

    for w in [5, 10, 21]:
        df[f"mean_ret_{w}d"] = returns.rolling(w).mean()

    df["skew_21d"] = returns.rolling(21).skew()
    df["kurtosis_21d"] = returns.rolling(21).kurt()
    df["abs_ret"] = returns.abs()
    df["sq_ret"] = returns ** 2

    df = df.dropna()
    return df


def fit_quantile_model(
    returns: pd.Series,
    quantile: float = 0.05,
    seed: int = 42,
) -> dict:
    """Fit XGBoost quantile regression to predict conditional VaR.

    Parameters
    ----------
    returns : daily log-return series.
    quantile : target quantile (0.05 = 95% VaR, 0.01 = 99% VaR).
    seed : random seed for reproducibility.

    Returns dict with fitted model, scaler, feature info, and current prediction.
    """
    features = engineer_features(returns)
    feature_cols = [c for c in features.columns if c != "returns"]

    # Target: next-day return (shift -1 aligns today's features with tomorrow's return)
    target = returns.shift(-1)

    # Align features and target
    aligned = features.copy()
    aligned["target"] = target
    aligned = aligned.dropna()

    X = aligned[feature_cols].values
    y = aligned["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_scaled, y)

    # Predict current VaR (latest features → tomorrow's quantile)
    latest_X = features[feature_cols].iloc[-1:].values
    latest_scaled = scaler.transform(latest_X)
    predicted_var = float(model.predict(latest_scaled)[0])

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "quantile": quantile,
        "predicted_var": predicted_var,
    }


def predict_var(model_result: dict, recent_returns: pd.Series) -> float:
    """Predict conditional VaR from the most recent returns."""
    features = engineer_features(recent_returns)
    feature_cols = model_result["feature_cols"]

    X = features[feature_cols].iloc[-1:].values
    X_scaled = model_result["scaler"].transform(X)
    return float(model_result["model"].predict(X_scaled)[0])


def backtest_quantile_var(
    returns: pd.Series,
    quantile: float = 0.05,
    train_window: int = 252,
    seed: int = 42,
    step: int = 1,
) -> pd.DataFrame:
    """Rolling walk-forward backtest for XGB quantile VaR.

    At each step: train on window → predict next-day quantile → check breach.
    Returns DataFrame compatible with backtesting.backtest_summary().

    Parameters
    ----------
    returns : full daily return series.
    quantile : target quantile (0.05 = 95% VaR).
    train_window : training window size in days.
    seed : random seed.
    step : test every N-th day.
    """
    results = []
    n_skipped = 0
    test_indices = range(train_window, len(returns) - 1, step)

    for t in test_indices:
        returns_window = returns.iloc[:t + 1]

        try:
            model_result = fit_quantile_model(returns_window, quantile=quantile, seed=seed)
            predicted = model_result["predicted_var"]
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
