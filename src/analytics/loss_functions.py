"""Economic loss functions for VaR backtest evaluation.

All functions consume the DataFrame returned by ``backtest_var()``
(columns: ``actual_return``, ``predicted_var``, ``breach``; DatetimeIndex).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lopez (1998) — quadratic loss
# ---------------------------------------------------------------------------

def lopez_loss(bt: pd.DataFrame) -> dict:
    """Quadratic loss penalising breach magnitude.

    On breach days: ``1 + (actual_return - predicted_var) ** 2``.
    Non-breach days: 0.
    """
    breach_mask = bt["breach"]
    excess = bt.loc[breach_mask, "actual_return"] - bt.loc[breach_mask, "predicted_var"]
    per_day = 1.0 + excess ** 2
    total = float(per_day.sum())
    n = len(bt)
    return {
        "total": total,
        "mean": total / n if n > 0 else 0.0,
        "max_single": float(per_day.max()) if len(per_day) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Basel traffic-light zone
# ---------------------------------------------------------------------------

_BASEL_MULTIPLIERS = {
    "green": 3.0,
    "yellow": 3.4,   # midpoint of 3.4–3.85 schedule
    "red": 4.0,
}


def basel_zone(bt: pd.DataFrame, confidence: float = 0.95) -> dict:
    """Basel Committee traffic-light zone classification.

    Standard thresholds are for 250 observations at 99% confidence.
    This implementation scales breach counts proportionally when
    ``n_obs != 250`` so the function works with any backtest length.
    """
    n_obs = len(bt)
    n_breaches = int(bt["breach"].sum())

    # Scale to 250-equivalent breach count
    scale = 250 / n_obs if n_obs > 0 else 1.0
    scaled = n_breaches * scale

    if scaled <= 4:
        zone = "green"
    elif scaled <= 9:
        zone = "yellow"
    else:
        zone = "red"

    return {
        "zone": zone,
        "n_breaches": n_breaches,
        "n_obs": n_obs,
        "scaled_breaches_250": round(scaled, 1),
        "capital_multiplier": _BASEL_MULTIPLIERS[zone],
    }


# ---------------------------------------------------------------------------
# Blanco-Ihle — linear magnitude loss
# ---------------------------------------------------------------------------

def blanco_ihle_loss(bt: pd.DataFrame) -> dict:
    """Linear loss penalising breach magnitude.

    On breach days: ``1 + |actual_return - predicted_var|``.
    Non-breach days: 0.
    """
    breach_mask = bt["breach"]
    excess = (bt.loc[breach_mask, "actual_return"]
              - bt.loc[breach_mask, "predicted_var"]).abs()
    per_day = 1.0 + excess
    total = float(per_day.sum())
    n = len(bt)
    return {
        "total": total,
        "mean": total / n if n > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Conditional exceedance statistics
# ---------------------------------------------------------------------------

def conditional_exceedance(bt: pd.DataFrame) -> dict:
    """Statistics of loss magnitude conditional on VaR breach.

    Returns mean breach return, mean excess over VaR, and worst breach.
    """
    breaches = bt.loc[bt["breach"]]

    if len(breaches) == 0:
        return {
            "mean_breach_return": float("nan"),
            "mean_excess_loss": float("nan"),
            "worst_breach": float("nan"),
        }

    excess = breaches["actual_return"] - breaches["predicted_var"]
    return {
        "mean_breach_return": float(breaches["actual_return"].mean()),
        "mean_excess_loss": float(excess.mean()),
        "worst_breach": float(breaches["actual_return"].min()),
    }
