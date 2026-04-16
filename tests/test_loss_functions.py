"""Tests for economic loss functions."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.loss_functions import (
    lopez_loss,
    basel_zone,
    blanco_ihle_loss,
    conditional_exceedance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bt_with_breaches():
    """Backtest DataFrame with known breach magnitudes."""
    return pd.DataFrame({
        "actual_return": [-0.08, 0.01, -0.12, 0.02, -0.06],
        "predicted_var":  [-0.05, -0.05, -0.05, -0.05, -0.05],
        "breach":         [True,  False, True,  False, True],
    }, index=pd.date_range("2024-01-01", periods=5))


@pytest.fixture
def bt_no_breaches():
    """Backtest DataFrame with zero breaches."""
    return pd.DataFrame({
        "actual_return": [0.01, 0.02, -0.03, 0.00],
        "predicted_var":  [-0.05, -0.05, -0.05, -0.05],
        "breach":         [False, False, False, False],
    }, index=pd.date_range("2024-01-01", periods=4))


# ---------------------------------------------------------------------------
# Lopez loss
# ---------------------------------------------------------------------------

class TestLopezLoss:
    def test_known_values(self, bt_with_breaches):
        result = lopez_loss(bt_with_breaches)
        # Breach excesses: -0.08-(-0.05)=-0.03, -0.12-(-0.05)=-0.07, -0.06-(-0.05)=-0.01
        # Per day: 1+0.0009=1.0009, 1+0.0049=1.0049, 1+0.0001=1.0001
        expected_total = 1.0009 + 1.0049 + 1.0001
        assert result["total"] == pytest.approx(expected_total, rel=1e-6)
        assert result["mean"] == pytest.approx(expected_total / 5, rel=1e-6)
        assert result["max_single"] == pytest.approx(1.0049, rel=1e-6)

    def test_no_breaches(self, bt_no_breaches):
        result = lopez_loss(bt_no_breaches)
        assert result["total"] == 0.0
        assert result["mean"] == 0.0
        assert result["max_single"] == 0.0


# ---------------------------------------------------------------------------
# Basel zone
# ---------------------------------------------------------------------------

class TestBaselZone:
    def _make_bt(self, n_obs, n_breaches):
        breach_flags = [True] * n_breaches + [False] * (n_obs - n_breaches)
        return pd.DataFrame({
            "actual_return": [-0.10] * n_breaches + [0.01] * (n_obs - n_breaches),
            "predicted_var": [-0.05] * n_obs,
            "breach": breach_flags,
        })

    def test_green(self):
        result = basel_zone(self._make_bt(250, 3))
        assert result["zone"] == "green"
        assert result["capital_multiplier"] == 3.0

    def test_yellow(self):
        result = basel_zone(self._make_bt(250, 7))
        assert result["zone"] == "yellow"
        assert result["capital_multiplier"] == 3.4

    def test_red(self):
        result = basel_zone(self._make_bt(250, 11))
        assert result["zone"] == "red"
        assert result["capital_multiplier"] == 4.0

    def test_scaling(self):
        """125 obs with 3 breaches → scaled to 6/250 → yellow."""
        result = basel_zone(self._make_bt(125, 3))
        assert result["zone"] == "yellow"
        assert result["scaled_breaches_250"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Blanco-Ihle loss
# ---------------------------------------------------------------------------

class TestBlancoIhleLoss:
    def test_known_values(self, bt_with_breaches):
        result = blanco_ihle_loss(bt_with_breaches)
        # |excess|: 0.03, 0.07, 0.01 → per day: 1.03, 1.07, 1.01
        expected_total = 1.03 + 1.07 + 1.01
        assert result["total"] == pytest.approx(expected_total, rel=1e-6)
        assert result["mean"] == pytest.approx(expected_total / 5, rel=1e-6)

    def test_no_breaches(self, bt_no_breaches):
        result = blanco_ihle_loss(bt_no_breaches)
        assert result["total"] == 0.0
        assert result["mean"] == 0.0


# ---------------------------------------------------------------------------
# Conditional exceedance
# ---------------------------------------------------------------------------

class TestConditionalExceedance:
    def test_known_values(self, bt_with_breaches):
        result = conditional_exceedance(bt_with_breaches)
        # Breach returns: -0.08, -0.12, -0.06
        assert result["mean_breach_return"] == pytest.approx(np.mean([-0.08, -0.12, -0.06]))
        # Excess: -0.03, -0.07, -0.01
        assert result["mean_excess_loss"] == pytest.approx(np.mean([-0.03, -0.07, -0.01]))
        assert result["worst_breach"] == pytest.approx(-0.12)

    def test_no_breaches(self, bt_no_breaches):
        result = conditional_exceedance(bt_no_breaches)
        assert np.isnan(result["mean_breach_return"])
        assert np.isnan(result["mean_excess_loss"])
        assert np.isnan(result["worst_breach"])
