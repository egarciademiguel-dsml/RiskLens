"""GMM clustering + classifier regime detection for drift and volatility estimation."""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def engineer_regime_features(returns: pd.Series, dropna: bool = True) -> pd.DataFrame:
    """Build rolling features from returns for regime clustering.

    Features: rolling vol (5/10/21d), rolling mean return (5/10/21d),
    rolling skew (21d), rolling kurtosis (21d).
    """
    df = pd.DataFrame({"returns": returns})

    for w in [5, 10, 21]:
        df[f"vol_{w}d"] = returns.rolling(w).std()
        df[f"mean_ret_{w}d"] = returns.rolling(w).mean()

    df["skew_21d"] = returns.rolling(21).skew()
    df["kurtosis_21d"] = returns.rolling(21).kurt()

    if dropna:
        df = df.dropna()

    return df


def fit_gmm(returns: pd.Series, n_regimes: int = 2, seed: int = 42) -> dict:
    """Fit a GMM on engineered features and train a classifier to predict regimes.

    Returns dict with:
      - gmm: fitted GaussianMixture
      - classifier: fitted RandomForestClassifier
      - scaler: fitted StandardScaler
      - n_regimes: number of regimes
      - regime_params: list of {mu, sigma} per regime, sorted by sigma ascending
      - regime_labels: array of regime assignments for each observation (aligned to features index)
      - features_index: DatetimeIndex of the feature rows (after dropna)
      - sort_order: index mapping from raw GMM states to sorted states
    """
    if n_regimes < 1:
        raise ValueError("n_regimes must be >= 1.")

    clean_returns = returns.dropna()

    if n_regimes == 1:
        mu = float(clean_returns.mean())
        sigma = float(clean_returns.std(ddof=1))
        features = engineer_regime_features(clean_returns)
        return {
            "gmm": None,
            "classifier": None,
            "scaler": None,
            "n_regimes": 1,
            "regime_params": [{"mu": mu, "sigma": sigma}],
            "regime_labels": np.zeros(len(features), dtype=int),
            "features_index": features.index,
            "sort_order": np.array([0]),
        }

    features = engineer_regime_features(clean_returns)
    feature_cols = [c for c in features.columns if c != "returns"]
    X = features[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        n_init=5,
        random_state=seed,
    )
    gmm.fit(X_scaled)
    raw_labels = gmm.predict(X_scaled)

    # Compute per-regime mu/sigma from actual returns
    aligned_returns = clean_returns.loc[features.index]
    raw_params = []
    for i in range(n_regimes):
        mask = raw_labels == i
        r = aligned_returns.values[mask]
        raw_params.append({
            "mu": float(r.mean()) if len(r) > 0 else 0.0,
            "sigma": float(r.std(ddof=1)) if len(r) > 1 else float(clean_returns.std(ddof=1)),
        })

    # Sort regimes by sigma ascending (regime 0 = calmest)
    sort_order = np.argsort([p["sigma"] for p in raw_params])
    regime_params = [raw_params[i] for i in sort_order]

    # Remap labels to sorted order
    inv_map = np.empty(n_regimes, dtype=int)
    for new_idx, old_idx in enumerate(sort_order):
        inv_map[old_idx] = new_idx
    regime_labels = inv_map[raw_labels]

    # Train classifier to predict regime from features
    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1,
    )
    classifier.fit(X_scaled, regime_labels)

    return {
        "gmm": gmm,
        "classifier": classifier,
        "scaler": scaler,
        "n_regimes": n_regimes,
        "regime_params": regime_params,
        "regime_labels": regime_labels,
        "features_index": features.index,
        "sort_order": sort_order,
    }


def predict_current_regime(gmm_result: dict, recent_returns: pd.Series) -> int:
    """Predict the current regime from the most recent returns."""
    if gmm_result["n_regimes"] == 1:
        return 0

    features = engineer_regime_features(recent_returns)
    feature_cols = [c for c in features.columns if c != "returns"]
    X = features[feature_cols].values

    X_scaled = gmm_result["scaler"].transform(X)
    predictions = gmm_result["classifier"].predict(X_scaled)
    return int(predictions[-1])


def get_regime_params(gmm_result: dict, regime_idx: int | None = None) -> dict | list:
    """Get drift (mu) and volatility (sigma) for a specific regime or all regimes."""
    if regime_idx is None:
        return gmm_result["regime_params"]
    return gmm_result["regime_params"][regime_idx]


def generate_log_returns(
    shocks: np.ndarray,
    returns: pd.Series,
    seed: int | None = None,
    gmm_params: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """Log returns using GMM regime-switching drift and volatility.

    Uses a simple Markov transition approximation: transition probabilities
    are estimated from observed regime label sequences.
    """
    if gmm_params is None:
        gmm_params = fit_gmm(returns, n_regimes=2, seed=seed or 42)

    current_regime = predict_current_regime(gmm_params, returns)
    n_days, n_simulations = shocks.shape

    if gmm_params["n_regimes"] == 1:
        p = gmm_params["regime_params"][0]
        drift = p["mu"] - 0.5 * p["sigma"]**2
        return drift + p["sigma"] * shocks

    # Build transition matrix from observed label sequences
    n_regimes = gmm_params["n_regimes"]
    labels = gmm_params["regime_labels"]
    trans_counts = np.zeros((n_regimes, n_regimes))
    for i in range(len(labels) - 1):
        trans_counts[labels[i], labels[i + 1]] += 1
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    trans_mat = trans_counts / row_sums

    all_mu = np.array([p["mu"] for p in gmm_params["regime_params"]])
    all_sigma = np.array([p["sigma"] for p in gmm_params["regime_params"]])

    rng = np.random.RandomState(seed)
    log_returns = np.empty_like(shocks)
    regimes = np.full(n_simulations, current_regime, dtype=int)

    for t_step in range(n_days):
        sig = all_sigma[regimes]
        mu_r = all_mu[regimes]
        drift_t = mu_r - 0.5 * sig**2
        log_returns[t_step] = drift_t + sig * shocks[t_step]

        u = rng.rand(n_simulations)
        cum_probs = np.cumsum(trans_mat[regimes], axis=1)
        regimes = (u[:, None] > cum_probs).sum(axis=1)
        regimes = np.clip(regimes, 0, n_regimes - 1)

    return log_returns
