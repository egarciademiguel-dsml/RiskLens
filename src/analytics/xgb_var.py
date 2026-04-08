"""XGBoost conditional quantile regression — nonparametric VaR estimation."""

from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Hyperparameter tuning helpers
# ---------------------------------------------------------------------------

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss — lower is better."""
    residual = y_true - y_pred
    return float(np.mean(np.where(residual >= 0, quantile * residual, (quantile - 1) * residual)))


_PARAM_GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [100, 200],
}


def tune_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.05,
    n_splits: int = 3,
    seed: int = 42,
) -> dict:
    """Select XGBoost hyperparameters via TimeSeriesSplit cross-validation.

    Returns dict with best parameters (max_depth, learning_rate, n_estimators).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    keys = list(_PARAM_GRID.keys())
    combos = list(product(*_PARAM_GRID.values()))

    best_loss = np.inf
    best_params = dict(zip(keys, combos[0]))

    for combo in combos:
        params = dict(zip(keys, combo))
        fold_losses = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbosity=0,
                **params,
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            fold_losses.append(pinball_loss(y_val, preds, quantile))

        mean_loss = np.mean(fold_losses)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = params

    return best_params


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
    tune: bool = False,
) -> dict:
    """Fit XGBoost quantile regression to predict conditional VaR.

    Parameters
    ----------
    returns : daily log-return series.
    quantile : target quantile (0.05 = 95% VaR, 0.01 = 99% VaR).
    seed : random seed for reproducibility.
    tune : if True, select hyperparameters via TimeSeriesSplit CV.

    Returns dict with fitted model, scaler, feature info, and current prediction.
    When tune=True, also includes 'best_params'.
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

    # Hyperparameters: tuned or default
    if tune:
        best_params = tune_hyperparameters(X_scaled, y, quantile=quantile, seed=seed)
    else:
        best_params = {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 200}

    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=0,
        **best_params,
    )
    model.fit(X_scaled, y)

    # Predict current VaR (latest features → tomorrow's quantile)
    latest_X = features[feature_cols].iloc[-1:].values
    latest_scaled = scaler.transform(latest_X)
    predicted_var = float(model.predict(latest_scaled)[0])

    result = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "quantile": quantile,
        "predicted_var": predicted_var,
    }
    if tune:
        result["best_params"] = best_params
    return result


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
