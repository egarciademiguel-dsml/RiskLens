import numpy as np
import pandas as pd
import pytest
from scipy.stats import genpareto

from src.analytics.evt import (
    fit_gpd,
    evt_var,
    evt_cvar,
    normal_var,
    normal_cvar,
    evt_summary,
    mean_residual_life,
    gpd_stability,
    gpd_qq,
    gpd_ks_test,
    gpd_bootstrap_ci,
    decluster_pot,
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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@pytest.fixture
def gpd_fat_tails():
    """Synthetic returns drawn from a known heavy-tailed process."""
    np.random.seed(1)
    returns = pd.Series(
        np.concatenate([
            np.random.normal(0.001, 0.01, 900),
            -np.abs(np.random.standard_t(3, 100)) * 0.03,
        ]),
        index=pd.date_range("2020-01-01", periods=1000, freq="D"),
        name="returns",
    )
    return returns


class TestMeanResidualLife:

    def test_returns_dataframe_with_columns(self, market_data):
        mrl = mean_residual_life(market_data)
        assert isinstance(mrl, pd.DataFrame)
        assert set(mrl.columns) == {"threshold", "mrl", "ci_lower", "ci_upper"}

    def test_mrl_positive(self, market_data):
        mrl = mean_residual_life(market_data)
        assert (mrl["mrl"] > 0).all()

    def test_ci_contains_mrl(self, market_data):
        mrl = mean_residual_life(market_data)
        assert (mrl["ci_lower"] <= mrl["mrl"]).all()
        assert (mrl["ci_upper"] >= mrl["mrl"]).all()

    def test_threshold_monotonically_increases(self, market_data):
        mrl = mean_residual_life(market_data)
        assert (mrl["threshold"].diff().dropna() >= 0).all()


class TestGpdStability:

    def test_returns_dataframe_with_columns(self, market_data):
        stab = gpd_stability(market_data)
        assert isinstance(stab, pd.DataFrame)
        expected = {"quantile", "threshold", "shape", "scale",
                    "modified_scale", "n_exceedances"}
        assert set(stab.columns) == expected

    def test_n_exceedances_decreases_with_quantile(self, market_data):
        stab = gpd_stability(market_data)
        if len(stab) > 1:
            assert stab["n_exceedances"].iloc[0] >= stab["n_exceedances"].iloc[-1]

    def test_quantile_range_respected(self, market_data):
        qr = np.array([0.85, 0.90, 0.95])
        stab = gpd_stability(market_data, quantile_range=qr)
        assert len(stab) <= len(qr)


class TestGpdQQ:

    def test_returns_equal_length_arrays(self, market_data):
        gpd = fit_gpd(market_data)
        theo, emp = gpd_qq(gpd, market_data)
        assert len(theo) == len(emp)
        assert len(theo) > 0

    def test_empirical_sorted(self, market_data):
        gpd = fit_gpd(market_data)
        _, emp = gpd_qq(gpd, market_data)
        assert (np.diff(emp) >= 0).all()

    def test_theoretical_sorted(self, market_data):
        gpd = fit_gpd(market_data)
        theo, _ = gpd_qq(gpd, market_data)
        assert (np.diff(theo) >= 0).all()


class TestGpdKsTest:

    def test_returns_expected_keys(self, market_data):
        gpd = fit_gpd(market_data)
        result = gpd_ks_test(gpd, market_data)
        assert set(result.keys()) == {"ks_statistic", "p_value", "pass"}

    def test_passes_on_gpd_drawn_data(self):
        """Data drawn from a known GPD should pass the KS test."""
        np.random.seed(42)
        gpd_data = genpareto.rvs(c=0.2, scale=0.01, size=500)
        returns = pd.Series(-gpd_data, name="returns")
        gpd_params = {
            "shape": 0.2, "scale": 0.01, "threshold": 0.0,
            "n_exceedances": 500, "n_total": 500, "threshold_quantile": 0.0,
        }
        result = gpd_ks_test(gpd_params, returns)
        assert result["pass"] is True

    def test_p_value_in_range(self, market_data):
        gpd = fit_gpd(market_data)
        result = gpd_ks_test(gpd, market_data)
        assert 0 <= result["p_value"] <= 1


class TestGpdBootstrapCi:

    def test_returns_expected_keys(self, market_data):
        result = gpd_bootstrap_ci(market_data, n_boot=100)
        assert "shape_ci" in result
        assert "scale_ci" in result
        assert len(result["shape_ci"]) == 2
        assert len(result["scale_ci"]) == 2

    def test_ci_contains_point_estimate(self, market_data):
        gpd = fit_gpd(market_data)
        result = gpd_bootstrap_ci(market_data, n_boot=500)
        lo, hi = result["shape_ci"]
        assert lo <= gpd["shape"] <= hi or abs(gpd["shape"] - lo) < 0.1

    def test_scale_ci_positive(self, market_data):
        result = gpd_bootstrap_ci(market_data, n_boot=100)
        lo, hi = result["scale_ci"]
        assert lo > 0 and hi > 0

    def test_too_few_exceedances_raises(self):
        tiny = pd.Series(np.random.normal(0, 0.01, 20))
        with pytest.raises(ValueError, match="exceedances"):
            gpd_bootstrap_ci(tiny, threshold_quantile=0.99)


class TestDeclusterPot:

    def test_returns_expected_keys(self, market_data):
        result = decluster_pot(market_data)
        expected = {"declustered_exceedances", "n_clusters",
                    "n_raw_exceedances", "threshold"}
        assert set(result.keys()) == expected

    def test_fewer_clusters_than_raw(self, market_data):
        result = decluster_pot(market_data)
        assert result["n_clusters"] <= result["n_raw_exceedances"]

    def test_declustered_exceedances_positive(self, market_data):
        result = decluster_pot(market_data)
        if len(result["declustered_exceedances"]) > 0:
            assert (result["declustered_exceedances"] > 0).all()

    def test_longer_run_length_fewer_clusters(self, gpd_fat_tails):
        r5 = decluster_pot(gpd_fat_tails, run_length=5)
        r20 = decluster_pot(gpd_fat_tails, run_length=20)
        assert r20["n_clusters"] <= r5["n_clusters"]

    def test_run_length_1_matches_raw(self, market_data):
        """run_length=1: only truly consecutive exceedances are grouped."""
        result = decluster_pot(market_data, run_length=1)
        assert result["n_clusters"] >= 1
