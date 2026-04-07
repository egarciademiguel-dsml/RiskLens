"""Extreme Value Theory — GPD tail risk estimation (Peaks Over Threshold)."""

import numpy as np
import pandas as pd
from scipy.stats import genpareto, norm


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
