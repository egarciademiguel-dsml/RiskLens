"""Realized volatility regression — XGBoost predicts forward vol."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def engineer_vol_features(returns: pd.Series) -> pd.DataFrame:
    """Build rolling features for volatility prediction.

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


def build_target(returns: pd.Series, horizon: int = 21) -> pd.Series:
    """Forward realized volatility: std of next `horizon` days of returns."""
    return returns.rolling(horizon).std().shift(-horizon)


def fit_rvol(returns: pd.Series, horizon: int = 21, seed: int = 42) -> dict:
    """Fit XGBoost to predict forward realized volatility.

    Returns dict with:
      - model: fitted XGBRegressor
      - scaler: fitted StandardScaler
      - feature_cols: list of feature column names
      - horizon: prediction horizon in days
      - r2_train: R² on training data
      - predicted_vol: predicted vol for the most recent observation
    """
    features = engineer_vol_features(returns)
    feature_cols = [c for c in features.columns if c != "returns"]

    target = build_target(returns, horizon)

    # Align features and target, drop NaNs from target shift
    aligned = features.copy()
    aligned["target"] = target
    aligned = aligned.dropna()

    X = aligned[feature_cols].values
    y = aligned["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
    )
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Predict current vol using latest features (not in training target due to shift)
    latest_features = features[feature_cols].iloc[-1:].values
    latest_scaled = scaler.transform(latest_features)
    predicted_vol = float(model.predict(latest_scaled)[0])
    predicted_vol = max(predicted_vol, 1e-8)  # floor at near-zero

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "horizon": horizon,
        "r2_train": float(r2),
        "predicted_vol": predicted_vol,
    }


def predict_current_vol(rvol_result: dict, recent_returns: pd.Series) -> float:
    """Predict forward realized vol from the most recent returns."""
    features = engineer_vol_features(recent_returns)
    feature_cols = rvol_result["feature_cols"]

    X = features[feature_cols].iloc[-1:].values
    X_scaled = rvol_result["scaler"].transform(X)
    vol = float(rvol_result["model"].predict(X_scaled)[0])
    return max(vol, 1e-8)


def generate_log_returns(
    shocks: np.ndarray,
    returns: pd.Series,
    seed: int | None = None,
    rvol_params: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """Log returns using global drift and ML-predicted volatility."""
    if rvol_params is None:
        rvol_params = fit_rvol(returns, horizon=21, seed=seed or 42)

    mu = returns.mean()
    sigma = predict_current_vol(rvol_params, returns)

    drift = mu - 0.5 * sigma**2
    return drift + sigma * shocks
