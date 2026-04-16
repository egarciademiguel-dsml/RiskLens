"""Extreme Value Theory — GPD tail risk estimation (Peaks Over Threshold)."""

import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm, kstest


def fit_gpd(
    returns: pd.Series,
    threshold_quantile: float = 0.95,
) -> dict:
    """Fit Generalized Pareto Distribution to tail losses via POT method.

    Parameters
    ----------
    returns : daily log-return series.
    threshold_quantile : quantile of losses defining the POT threshold (0-1).
        Higher = fewer exceedances, more extreme tail focus.

    Returns dict with fitted GPD parameters and threshold info.
    """
    losses = -returns.dropna().values  # positive = loss
    threshold = np.quantile(losses, threshold_quantile)
    exceedances = losses[losses > threshold] - threshold

    if len(exceedances) < 5:
        raise ValueError(
            f"Only {len(exceedances)} exceedances above threshold. "
            "Need at least 5. Lower threshold_quantile."
        )

    shape, _, scale = genpareto.fit(exceedances, floc=0)

    return {
        "shape": float(shape),
        "scale": float(scale),
        "threshold": float(threshold),
        "n_exceedances": len(exceedances),
        "n_total": len(losses),
        "threshold_quantile": threshold_quantile,
    }


def evt_var(gpd_params: dict, confidence: float = 0.99) -> float:
    """EVT Value-at-Risk using GPD tail estimate.

    Returns VaR as a negative number (loss) for consistency with compute_var().
    """
    xi = gpd_params["shape"]
    beta = gpd_params["scale"]
    u = gpd_params["threshold"]
    n = gpd_params["n_total"]
    nu = gpd_params["n_exceedances"]

    # POT VaR formula
    var_loss = u + (beta / xi) * ((n / nu * (1 - confidence)) ** (-xi) - 1)
    return -float(var_loss)


def evt_cvar(gpd_params: dict, confidence: float = 0.99) -> float:
    """EVT Expected Shortfall (CVaR) using GPD tail estimate.

    Returns CVaR as a negative number (loss) for consistency with compute_cvar().
    """
    xi = gpd_params["shape"]
    beta = gpd_params["scale"]
    u = gpd_params["threshold"]

    var_loss = -evt_var(gpd_params, confidence)  # positive loss magnitude

    if xi >= 1:
        return -np.inf  # infinite expected shortfall for xi >= 1

    cvar_loss = var_loss / (1 - xi) + (beta - xi * u) / (1 - xi)
    return -float(cvar_loss)


def normal_var(returns: pd.Series, confidence: float = 0.99) -> float:
    """Parametric VaR assuming Normal distribution. For comparison."""
    mu = returns.mean()
    sigma = returns.std()
    return float(mu + sigma * norm.ppf(1 - confidence))


def normal_cvar(returns: pd.Series, confidence: float = 0.99) -> float:
    """Parametric CVaR assuming Normal distribution. For comparison."""
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(1 - confidence)
    return float(mu - sigma * norm.pdf(z) / (1 - confidence))


def evt_summary(
    returns: pd.Series,
    confidence: float = 0.99,
    threshold_quantile: float = 0.95,
) -> dict:
    """Full EVT analysis with comparison to Normal VaR.

    Returns dict with GPD params, EVT VaR/CVaR, Normal VaR/CVaR, and diagnostics.
    """
    gpd_params = fit_gpd(returns, threshold_quantile)

    xi = gpd_params["shape"]
    if xi < 0:
        tail_type = "Thin tail (bounded)"
    elif xi == 0:
        tail_type = "Exponential tail"
    elif xi < 0.5:
        tail_type = "Heavy tail (finite variance)"
    else:
        tail_type = "Very heavy tail"

    return {
        **gpd_params,
        "tail_type": tail_type,
        "evt_var": evt_var(gpd_params, confidence),
        "evt_cvar": evt_cvar(gpd_params, confidence),
        "normal_var": normal_var(returns, confidence),
        "normal_cvar": normal_cvar(returns, confidence),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def mean_residual_life(
    returns: pd.Series,
    n_thresholds: int = 50,
) -> pd.DataFrame:
    """Mean residual life plot data.

    For a range of thresholds u, computes E[X - u | X > u]. Linearity in u
    above some u₀ supports the GPD assumption above u₀.

    Returns DataFrame with columns: threshold, mrl, ci_lower, ci_upper.
    """
    losses = -returns.dropna().values
    lo, hi = np.quantile(losses, 0.50), np.quantile(losses, 0.995)
    thresholds = np.linspace(lo, hi, n_thresholds)

    rows = []
    for u in thresholds:
        exceedances = losses[losses > u] - u
        n = len(exceedances)
        if n < 3:
            continue
        mean_excess = float(exceedances.mean())
        se = float(exceedances.std() / np.sqrt(n))
        rows.append({
            "threshold": float(u),
            "mrl": mean_excess,
            "ci_lower": mean_excess - 1.96 * se,
            "ci_upper": mean_excess + 1.96 * se,
        })

    return pd.DataFrame(rows)


def gpd_stability(
    returns: pd.Series,
    quantile_range: np.ndarray | None = None,
) -> pd.DataFrame:
    """GPD parameter stability across thresholds.

    Fits GPD at each threshold quantile and reports shape (ξ) and modified
    scale (σ* = σ - ξu). Both should stabilize above the correct threshold.

    Returns DataFrame with columns: quantile, threshold, shape, scale,
    modified_scale, n_exceedances.
    """
    if quantile_range is None:
        quantile_range = np.linspace(0.80, 0.98, 30)

    rows = []
    for q in quantile_range:
        try:
            params = fit_gpd(returns, threshold_quantile=q)
            rows.append({
                "quantile": float(q),
                "threshold": params["threshold"],
                "shape": params["shape"],
                "scale": params["scale"],
                "modified_scale": params["scale"] - params["shape"] * params["threshold"],
                "n_exceedances": params["n_exceedances"],
            })
        except ValueError:
            continue

    return pd.DataFrame(rows)


def gpd_qq(gpd_params: dict, returns: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """QQ plot data for fitted GPD vs empirical exceedances.

    Returns (theoretical, empirical) quantile arrays. Points on the diagonal
    indicate a good fit; deviations at the upper end mean the GPD
    underestimates the most extreme losses.
    """
    losses = -returns.dropna().values
    threshold = gpd_params["threshold"]
    exceedances = np.sort(losses[losses > threshold] - threshold)
    n = len(exceedances)

    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical = genpareto.ppf(
        plotting_positions, c=gpd_params["shape"], loc=0, scale=gpd_params["scale"],
    )
    return theoretical, exceedances


def gpd_ks_test(gpd_params: dict, returns: pd.Series) -> dict:
    """Kolmogorov-Smirnov goodness-of-fit test for the fitted GPD.

    Returns dict with ks_statistic, p_value, pass (True if p >= 0.05).
    """
    losses = -returns.dropna().values
    threshold = gpd_params["threshold"]
    exceedances = losses[losses > threshold] - threshold

    frozen_gpd = genpareto(c=gpd_params["shape"], loc=0, scale=gpd_params["scale"])
    stat, p_value = kstest(exceedances, frozen_gpd.cdf)

    return {
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "pass": bool(p_value >= 0.05),
    }


def gpd_bootstrap_ci(
    returns: pd.Series,
    threshold_quantile: float = 0.95,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence intervals for GPD shape and scale.

    Resamples exceedances with replacement, refits GPD on each sample.

    Returns dict with shape_ci (lo, hi), scale_ci (lo, hi), shape_samples,
    scale_samples.
    """
    losses = -returns.dropna().values
    threshold = np.quantile(losses, threshold_quantile)
    exceedances = losses[losses > threshold] - threshold

    if len(exceedances) < 5:
        raise ValueError(f"Only {len(exceedances)} exceedances — need ≥5 for bootstrap.")

    rng = np.random.RandomState(seed)
    shapes, scales = [], []

    for _ in range(n_boot):
        sample = rng.choice(exceedances, size=len(exceedances), replace=True)
        try:
            shape, _, scale = genpareto.fit(sample, floc=0)
            shapes.append(shape)
            scales.append(scale)
        except Exception:
            continue

    shapes = np.array(shapes)
    scales = np.array(scales)
    alpha = (1 - ci) / 2

    return {
        "shape_ci": (float(np.quantile(shapes, alpha)), float(np.quantile(shapes, 1 - alpha))),
        "scale_ci": (float(np.quantile(scales, alpha)), float(np.quantile(scales, 1 - alpha))),
        "shape_samples": shapes,
        "scale_samples": scales,
    }


def decluster_pot(
    returns: pd.Series,
    threshold_quantile: float = 0.95,
    run_length: int = 10,
) -> dict:
    """Decluster exceedances using runs method.

    Groups consecutive exceedances separated by fewer than run_length
    non-exceedances into clusters, keeping only the cluster maximum.
    Addresses the iid violation in POT for serially dependent returns.

    Returns dict with declustered_exceedances (array), n_clusters,
    n_raw_exceedances, threshold.
    """
    losses = -returns.dropna().values
    threshold = np.quantile(losses, threshold_quantile)
    exceed_mask = losses > threshold
    n_raw = int(exceed_mask.sum())

    clusters = []
    current_cluster = []
    gap = 0

    for i, is_exceed in enumerate(exceed_mask):
        if is_exceed:
            if current_cluster and gap >= run_length:
                clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append(losses[i] - threshold)
            gap = 0
        else:
            gap += 1

    if current_cluster:
        clusters.append(current_cluster)

    declustered = np.array([max(c) for c in clusters]) if clusters else np.array([])

    return {
        "declustered_exceedances": declustered,
        "n_clusters": len(clusters),
        "n_raw_exceedances": n_raw,
        "threshold": float(threshold),
    }
