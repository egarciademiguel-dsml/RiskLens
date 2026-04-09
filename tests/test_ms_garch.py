import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis

from src.analytics.ms_garch import (
    fit_ms_garch,
    generate_evt_shocks,
    generate_log_returns,
    _constant_garch_fallback,
    _compute_standardized_residuals,
)
from src.analytics.monte_carlo import simulate_paths, compute_var, compute_cvar


@pytest.fixture
def market_data():
    """Deterministic close and returns series."""
    np.random.seed(0)
    returns = pd.Series(
        np.random.normal(0.001, 0.02, 500),
        index=pd.date_range("2020-01-01", periods=500, freq="D"),
        name="returns",
    )
    close = 100 * (1 + returns).cumprod()
    close.name = "close"
    return close, returns


@pytest.fixture
def ms_garch_params(market_data):
    """Pre-fitted MS-GARCH params for reuse."""
    _, returns = market_data
    return fit_ms_garch(returns, n_regimes=2, seed=42)


# ---------------------------------------------------------------------------
# fit_ms_garch
# ---------------------------------------------------------------------------

class TestFitMsGarch:

    def test_returns_expected_keys(self, ms_garch_params):
        expected = {"hmm_result", "n_regimes", "regime_garch", "regime_gpd",
                    "regime_mu", "current_regime", "min_regime_obs"}
        assert set(ms_garch_params.keys()) == expected

    def test_n_regimes_param_count(self, ms_garch_params):
        n = ms_garch_params["n_regimes"]
        assert len(ms_garch_params["regime_garch"]) == n
        assert len(ms_garch_params["regime_gpd"]) == n
        assert len(ms_garch_params["regime_mu"]) == n

    def test_regime_garch_has_expected_keys(self, ms_garch_params):
        for gp in ms_garch_params["regime_garch"]:
            assert "omega" in gp
            assert "alpha" in gp
            assert "beta" in gp
            assert "last_variance" in gp

    def test_current_regime_in_range(self, ms_garch_params):
        n = ms_garch_params["n_regimes"]
        assert 0 <= ms_garch_params["current_regime"] < n

    def test_hmm_result_present(self, ms_garch_params):
        hmm = ms_garch_params["hmm_result"]
        assert "transition_matrix" in hmm
        assert "regime_params" in hmm

    @pytest.mark.parametrize("n", [2, 3])
    def test_multi_regime(self, market_data, n):
        _, returns = market_data
        result = fit_ms_garch(returns, n_regimes=n, seed=42)
        assert result["n_regimes"] == n
        assert len(result["regime_garch"]) == n


# ---------------------------------------------------------------------------
# generate_evt_shocks
# ---------------------------------------------------------------------------

class TestGenerateEvtShocks:

    def test_output_shape(self):
        gpd_params = {"shape": 0.2, "scale": 0.5, "threshold": 1.5,
                      "n_exceedances": 25, "n_total": 500}
        rng = np.random.RandomState(42)
        shocks = generate_evt_shocks(gpd_params, 1000, rng)
        assert shocks.shape == (1000,)

    def test_no_nans(self):
        gpd_params = {"shape": 0.1, "scale": 0.3, "threshold": 1.2,
                      "n_exceedances": 30, "n_total": 500}
        rng = np.random.RandomState(42)
        shocks = generate_evt_shocks(gpd_params, 5000, rng)
        assert not np.any(np.isnan(shocks))

    def test_heavier_left_tail_than_normal(self):
        """EVT shocks should have heavier left tail than Normal."""
        gpd_params = {"shape": 0.3, "scale": 0.5, "threshold": 1.5,
                      "n_exceedances": 50, "n_total": 500}
        rng = np.random.RandomState(42)
        evt_shocks = generate_evt_shocks(gpd_params, 50000, rng)
        normal_shocks = np.random.RandomState(42).normal(0, 1, 50000)

        # More extreme left tail: lower 1st percentile
        assert np.percentile(evt_shocks, 1) < np.percentile(normal_shocks, 1)

    def test_bulk_is_normal_like(self):
        """Non-tail portion should be approximately Normal."""
        gpd_params = {"shape": 0.1, "scale": 0.3, "threshold": 1.5,
                      "n_exceedances": 25, "n_total": 500}
        rng = np.random.RandomState(42)
        shocks = generate_evt_shocks(gpd_params, 10000, rng)
        # Median should be near 0 (Normal bulk)
        assert abs(np.median(shocks)) < 0.1

    def test_thin_tail_shape(self):
        """Negative shape (thin tail) should still work."""
        gpd_params = {"shape": -0.1, "scale": 0.3, "threshold": 1.2,
                      "n_exceedances": 25, "n_total": 500}
        rng = np.random.RandomState(42)
        shocks = generate_evt_shocks(gpd_params, 1000, rng)
        assert not np.any(np.isnan(shocks))


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

class TestFallbacks:

    def test_constant_garch_fallback_keys(self):
        fb = _constant_garch_fallback(0.02)
        expected = {"omega", "alpha", "beta", "long_run_vol",
                    "last_variance", "last_resid", "persistence"}
        assert set(fb.keys()) == expected

    def test_constant_garch_fallback_degenerate(self):
        fb = _constant_garch_fallback(0.02)
        assert fb["alpha"] == 0.0
        assert fb["beta"] == 0.0
        assert fb["persistence"] == 0.0
        assert abs(fb["omega"] - 0.02**2) < 1e-10

    def test_small_regime_uses_fallback(self, market_data):
        """With min_regime_obs very high, all regimes should fall back."""
        _, returns = market_data
        result = fit_ms_garch(returns, n_regimes=2, seed=42, min_regime_obs=10000)
        for gp in result["regime_garch"]:
            # Fallback: alpha=0, beta=0
            assert gp["alpha"] == 0.0
            assert gp["beta"] == 0.0


# ---------------------------------------------------------------------------
# MC integration
# ---------------------------------------------------------------------------

class TestMsGarchIntegration:

    def test_simulate_paths_shape(self, market_data):
        close, returns = market_data
        paths = simulate_paths(
            close, returns, n_days=60, n_simulations=1000,
            seed=42, volatility_model="ms_garch",
        )
        assert paths.shape == (60, 1000)

    def test_all_prices_positive(self, market_data):
        close, returns = market_data
        paths = simulate_paths(
            close, returns, n_days=100, n_simulations=5000,
            seed=42, volatility_model="ms_garch",
        )
        assert (paths > 0).all().all()

    def test_reproducible_with_seed(self, market_data):
        close, returns = market_data
        params = fit_ms_garch(returns, n_regimes=2, seed=42)
        p1 = simulate_paths(close, returns, n_days=30, n_simulations=100,
                            seed=42, volatility_model="ms_garch",
                            ms_garch_params=params)
        p2 = simulate_paths(close, returns, n_days=30, n_simulations=100,
                            seed=42, volatility_model="ms_garch",
                            ms_garch_params=params)
        pd.testing.assert_frame_equal(p1, p2)

    def test_differs_from_constant(self, market_data):
        close, returns = market_data
        p_const = simulate_paths(close, returns, n_days=30, n_simulations=100,
                                 seed=42, volatility_model="constant")
        p_ms = simulate_paths(close, returns, n_days=30, n_simulations=100,
                              seed=42, volatility_model="ms_garch")
        assert not p_const.equals(p_ms)

    def test_differs_from_garch(self, market_data):
        close, returns = market_data
        p_garch = simulate_paths(close, returns, n_days=30, n_simulations=100,
                                 seed=42, volatility_model="garch")
        p_ms = simulate_paths(close, returns, n_days=30, n_simulations=100,
                              seed=42, volatility_model="ms_garch")
        assert not p_garch.equals(p_ms)

    def test_cvar_le_var(self, market_data):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=100, n_simulations=5000,
                               seed=42, volatility_model="ms_garch")
        fp = paths.iloc[-1]
        ip = close.iloc[-1]
        var = compute_var(fp, ip, 0.95)
        cvar = compute_cvar(fp, ip, 0.95)
        assert cvar <= var

    def test_pre_fitted_params(self, market_data, ms_garch_params):
        close, returns = market_data
        paths = simulate_paths(
            close, returns, n_days=30, n_simulations=100,
            seed=42, volatility_model="ms_garch",
            ms_garch_params=ms_garch_params,
        )
        assert paths.shape == (30, 100)
        assert (paths > 0).all().all()


# ---------------------------------------------------------------------------
# Backtest integration
# ---------------------------------------------------------------------------

class TestMsGarchBacktest:

    def test_backtest_fit_function(self, market_data):
        """ms_garch_fit produces valid simulate_paths kwargs."""
        from src.analytics.backtesting import ms_garch_fit
        _, returns = market_data
        result = ms_garch_fit(returns, seed=42)
        assert result["volatility_model"] == "ms_garch"
        assert "ms_garch_params" in result
