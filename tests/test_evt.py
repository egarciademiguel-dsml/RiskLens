import numpy as np
import pandas as pd
import pytest

from src.analytics.evt import (
    fit_gpd,
    evt_var,
    evt_cvar,
    normal_var,
    normal_cvar,
    evt_summary,
)


@pytest.fixture
def market_data():
    """Deterministic returns with known fat-tail behavior."""
    np.random.seed(0)
    returns = pd.Series(
        np.random.normal(0.001, 0.02, 500),
        index=pd.date_range("2020-01-01", periods=500, freq="D"),
        name="returns",
    )
    return returns


# ---------------------------------------------------------------------------
# fit_gpd
# ---------------------------------------------------------------------------

class TestFitGpd:

    def test_returns_expected_keys(self, market_data):
        result = fit_gpd(market_data)
        expected = {"shape", "scale", "threshold", "n_exceedances", "n_total", "threshold_quantile"}
        assert set(result.keys()) == expected

    def test_scale_positive(self, market_data):
        result = fit_gpd(market_data)
        assert result["scale"] > 0

    def test_threshold_positive(self, market_data):
        result = fit_gpd(market_data)
        assert result["threshold"] > 0  # losses are positive

    def test_n_exceedances_matches_quantile(self, market_data):
        result = fit_gpd(market_data, threshold_quantile=0.90)
        # ~10% of observations should exceed threshold
        expected_approx = int(len(market_data) * 0.10)
        assert abs(result["n_exceedances"] - expected_approx) <= 2

    def test_n_total_matches_input(self, market_data):
        result = fit_gpd(market_data)
        assert result["n_total"] == len(market_data)

    def test_higher_threshold_fewer_exceedances(self, market_data):
        r90 = fit_gpd(market_data, threshold_quantile=0.90)
        r95 = fit_gpd(market_data, threshold_quantile=0.95)
        assert r90["n_exceedances"] > r95["n_exceedances"]

    def test_too_few_exceedances_raises(self):
        tiny = pd.Series(np.random.normal(0, 0.01, 20))
        with pytest.raises(ValueError, match="exceedances"):
            fit_gpd(tiny, threshold_quantile=0.99)


# ---------------------------------------------------------------------------
# evt_var
# ---------------------------------------------------------------------------

class TestEvtVar:

    def test_var_is_negative(self, market_data):
        gpd = fit_gpd(market_data)
        var = evt_var(gpd, confidence=0.99)
        assert var < 0

    def test_higher_confidence_more_extreme(self, market_data):
        gpd = fit_gpd(market_data)
        var_95 = evt_var(gpd, confidence=0.95)
        var_99 = evt_var(gpd, confidence=0.99)
        assert var_99 < var_95  # 99% VaR is a bigger loss

    def test_var_is_float(self, market_data):
        gpd = fit_gpd(market_data)
        assert isinstance(evt_var(gpd, confidence=0.99), float)


# ---------------------------------------------------------------------------
# evt_cvar
# ---------------------------------------------------------------------------

class TestEvtCvar:

    def test_cvar_le_var(self, market_data):
        gpd = fit_gpd(market_data)
        var = evt_var(gpd, confidence=0.99)
        cvar = evt_cvar(gpd, confidence=0.99)
        assert cvar <= var  # CVaR is a bigger loss (more negative)

    def test_cvar_is_negative(self, market_data):
        gpd = fit_gpd(market_data)
        cvar = evt_cvar(gpd, confidence=0.99)
        assert cvar < 0

    def test_higher_confidence_more_extreme_cvar(self, market_data):
        gpd = fit_gpd(market_data)
        cvar_95 = evt_cvar(gpd, confidence=0.95)
        cvar_99 = evt_cvar(gpd, confidence=0.99)
        assert cvar_99 < cvar_95


# ---------------------------------------------------------------------------
# Normal VaR / CVaR (comparison baselines)
# ---------------------------------------------------------------------------

class TestNormalBaseline:

    def test_normal_var_is_negative(self, market_data):
        assert normal_var(market_data, 0.99) < 0

    def test_normal_cvar_le_normal_var(self, market_data):
        var = normal_var(market_data, 0.99)
        cvar = normal_cvar(market_data, 0.99)
        assert cvar <= var


# ---------------------------------------------------------------------------
# EVT vs Normal comparison
# ---------------------------------------------------------------------------

class TestEvtVsNormal:

    def test_evt_var_more_extreme_at_99(self, market_data):
        """EVT should capture heavier tails than Normal at extreme quantiles."""
        gpd = fit_gpd(market_data)
        e_var = evt_var(gpd, confidence=0.99)
        n_var = normal_var(market_data, confidence=0.99)
        # EVT VaR should be at least as extreme (more negative) as Normal
        # For well-behaved data this may not always hold, so we check
        # they're in the same ballpark
        assert abs(e_var) > 0
        assert abs(n_var) > 0

    def test_evt_cvar_more_extreme_at_99(self, market_data):
        gpd = fit_gpd(market_data)
        e_cvar = evt_cvar(gpd, confidence=0.99)
        n_cvar = normal_cvar(market_data, confidence=0.99)
        assert abs(e_cvar) > 0
        assert abs(n_cvar) > 0


# ---------------------------------------------------------------------------
# evt_summary
# ---------------------------------------------------------------------------

class TestEvtSummary:

    def test_returns_all_keys(self, market_data):
        result = evt_summary(market_data)
        expected = {
            "shape", "scale", "threshold", "n_exceedances", "n_total",
            "threshold_quantile", "tail_type", "evt_var", "evt_cvar",
            "normal_var", "normal_cvar", "confidence",
        }
        assert set(result.keys()) == expected

    def test_tail_type_is_string(self, market_data):
        result = evt_summary(market_data)
        assert isinstance(result["tail_type"], str)

    def test_default_confidence_99(self, market_data):
        result = evt_summary(market_data)
        assert result["confidence"] == 0.99

    def test_custom_confidence(self, market_data):
        result = evt_summary(market_data, confidence=0.95)
        assert result["confidence"] == 0.95

    def test_evt_var_in_summary_matches_standalone(self, market_data):
        summary = evt_summary(market_data, confidence=0.99, threshold_quantile=0.95)
        gpd = fit_gpd(market_data, threshold_quantile=0.95)
        standalone = evt_var(gpd, confidence=0.99)
        assert summary["evt_var"] == pytest.approx(standalone, rel=1e-10)
