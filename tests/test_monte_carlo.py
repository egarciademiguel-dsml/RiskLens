import pandas as pd
import numpy as np
import pytest
from scipy.stats import kurtosis

from src.analytics.monte_carlo import (
    simulate_paths,
    compute_var,
    compute_cvar,
    prob_target,
    scenario_buckets,
    simulation_summary,
    fit_t_distribution,
    fit_garch,
)


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
def sim_results(market_data):
    """Pre-computed simulation (constant vol, normal shocks) for reuse."""
    close, returns = market_data
    paths = simulate_paths(close, returns, n_days=100, n_simulations=5000, seed=42)
    final_prices = paths.iloc[-1]
    initial_price = close.iloc[-1]
    return paths, final_prices, initial_price


# ---------------------------------------------------------------------------
# Path simulation — shape, positivity, reproducibility
# ---------------------------------------------------------------------------

class TestSimulatePaths:

    def test_shape(self, market_data):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=60, n_simulations=1000, seed=0)
        assert paths.shape == (60, 1000)

    def test_row_zero_is_not_initial_price(self, market_data):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=10, n_simulations=100, seed=0)
        initial = close.iloc[-1]
        assert not np.isclose(paths.iloc[0].mean(), initial, rtol=0.001)

    def test_reproducible_with_seed(self, market_data):
        close, returns = market_data
        p1 = simulate_paths(close, returns, n_days=30, n_simulations=100, seed=42)
        p2 = simulate_paths(close, returns, n_days=30, n_simulations=100, seed=42)
        pd.testing.assert_frame_equal(p1, p2)

    def test_different_seeds_differ(self, market_data):
        close, returns = market_data
        p1 = simulate_paths(close, returns, n_days=30, n_simulations=100, seed=1)
        p2 = simulate_paths(close, returns, n_days=30, n_simulations=100, seed=2)
        assert not p1.equals(p2)

    def test_all_prices_positive(self, sim_results):
        paths, _, _ = sim_results
        assert (paths > 0).all().all()


# ---------------------------------------------------------------------------
# VaR / CVaR — property-based
# ---------------------------------------------------------------------------

class TestVarCvar:

    def test_var_is_negative_at_95(self, sim_results):
        _, final_prices, initial_price = sim_results
        var = compute_var(final_prices, initial_price, 0.95)
        assert var < 0

    def test_cvar_le_var(self, sim_results):
        _, final_prices, initial_price = sim_results
        var = compute_var(final_prices, initial_price, 0.95)
        cvar = compute_cvar(final_prices, initial_price, 0.95)
        assert cvar <= var

    def test_higher_confidence_stricter_var(self, sim_results):
        _, final_prices, initial_price = sim_results
        var_90 = compute_var(final_prices, initial_price, 0.90)
        var_99 = compute_var(final_prices, initial_price, 0.99)
        assert var_99 <= var_90


# ---------------------------------------------------------------------------
# Probability targets — property-based
# ---------------------------------------------------------------------------

class TestProbTarget:

    def test_prob_gain_consistency(self, sim_results):
        _, final_prices, initial_price = sim_results
        pt = prob_target(final_prices, initial_price, 0.0)
        summary = simulation_summary(final_prices, initial_price)
        assert pt == summary["prob_gain"]

    def test_prob_between_0_and_1(self, sim_results):
        _, final_prices, initial_price = sim_results
        p = prob_target(final_prices, initial_price, 0.10)
        assert 0.0 <= p <= 1.0

    def test_prob_decreases_with_higher_target(self, sim_results):
        _, final_prices, initial_price = sim_results
        p10 = prob_target(final_prices, initial_price, 0.10)
        p50 = prob_target(final_prices, initial_price, 0.50)
        assert p50 <= p10

    def test_extreme_target_returns_zero(self, sim_results):
        _, final_prices, initial_price = sim_results
        assert prob_target(final_prices, initial_price, 100.0) == 0.0


# ---------------------------------------------------------------------------
# Scenario buckets — property-based
# ---------------------------------------------------------------------------

class TestScenarioBuckets:

    def test_sums_to_one(self, sim_results):
        _, final_prices, initial_price = sim_results
        buckets = scenario_buckets(final_prices, initial_price)
        assert abs(sum(buckets.values()) - 1.0) < 1e-9

    def test_all_values_between_0_and_1(self, sim_results):
        _, final_prices, initial_price = sim_results
        buckets = scenario_buckets(final_prices, initial_price)
        for v in buckets.values():
            assert 0.0 <= v <= 1.0

    def test_has_five_buckets(self, sim_results):
        _, final_prices, initial_price = sim_results
        buckets = scenario_buckets(final_prices, initial_price)
        assert len(buckets) == 5


# ---------------------------------------------------------------------------
# Simulation summary — property-based
# ---------------------------------------------------------------------------

class TestSimulationSummary:

    def test_has_all_keys(self, sim_results):
        _, final_prices, initial_price = sim_results
        summary = simulation_summary(final_prices, initial_price)
        expected = {"initial_price", "mean_final_price", "median_final_price",
                    "min_final_price", "max_final_price", "var", "cvar",
                    "confidence", "prob_gain", "prob_loss", "scenarios"}
        assert set(summary.keys()) == expected

    def test_prob_gain_plus_loss_eq_one(self, sim_results):
        _, final_prices, initial_price = sim_results
        summary = simulation_summary(final_prices, initial_price)
        assert abs(summary["prob_gain"] + summary["prob_loss"] - 1.0) < 1e-9

    def test_mean_above_min_below_max(self, sim_results):
        _, final_prices, initial_price = sim_results
        summary = simulation_summary(final_prices, initial_price)
        assert summary["min_final_price"] <= summary["mean_final_price"] <= summary["max_final_price"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    def test_empty_series_var(self):
        with pytest.raises(Exception):
            compute_var(pd.Series(dtype=float), 100.0, 0.95)

    def test_empty_series_cvar(self):
        with pytest.raises(Exception):
            compute_cvar(pd.Series(dtype=float), 100.0, 0.95)

    def test_single_simulation(self, market_data):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=10, n_simulations=1, seed=0)
        assert paths.shape == (10, 1)
        assert (paths > 0).all().all()

    def test_invalid_volatility_model_raises(self, market_data):
        close, returns = market_data
        with pytest.raises(ValueError):
            simulate_paths(close, returns, n_days=10, n_simulations=10, volatility_model="stochastic")

    def test_hmm_accepted(self, market_data):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=10, n_simulations=10,
                               seed=0, volatility_model="hmm")
        assert paths.shape == (10, 10)


# ---------------------------------------------------------------------------
# fit_t_distribution utility (still available for display)
# ---------------------------------------------------------------------------

class TestFitTDistribution:

    def test_returns_expected_keys(self, market_data):
        _, returns = market_data
        result = fit_t_distribution(returns)
        expected = {"df", "df_raw", "loc", "scale", "n_observations", "tail_description"}
        assert set(result.keys()) == expected

    def test_df_clamped_above_2(self, market_data):
        _, returns = market_data
        result = fit_t_distribution(returns)
        assert result["df"] >= 2.1

    def test_n_observations_matches(self, market_data):
        _, returns = market_data
        result = fit_t_distribution(returns)
        assert result["n_observations"] == len(returns.dropna())

    def test_tail_description_is_string(self, market_data):
        _, returns = market_data
        result = fit_t_distribution(returns)
        assert isinstance(result["tail_description"], str)
        assert len(result["tail_description"]) > 0


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

class TestGarchSimulation:

    def test_shape_with_garch(self, market_data):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=60, n_simulations=1000,
                               seed=0, volatility_model="garch")
        assert paths.shape == (60, 1000)

    def test_all_prices_positive_garch(self, market_data):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=100, n_simulations=5000,
                               seed=42, volatility_model="garch")
        assert (paths > 0).all().all()

    def test_reproducible_with_seed_garch(self, market_data):
        close, returns = market_data
        p1 = simulate_paths(close, returns, n_days=30, n_simulations=100,
                            seed=42, volatility_model="garch")
        p2 = simulate_paths(close, returns, n_days=30, n_simulations=100,
                            seed=42, volatility_model="garch")
        pd.testing.assert_frame_equal(p1, p2)

    def test_garch_differs_from_constant(self, market_data):
        close, returns = market_data
        p_const = simulate_paths(close, returns, n_days=30, n_simulations=100,
                                 seed=42, volatility_model="constant")
        p_garch = simulate_paths(close, returns, n_days=30, n_simulations=100,
                                 seed=42, volatility_model="garch")
        assert not p_const.equals(p_garch)

    def test_default_is_constant(self, market_data):
        close, returns = market_data
        p_default = simulate_paths(close, returns, n_days=30, n_simulations=100, seed=42)
        p_const = simulate_paths(close, returns, n_days=30, n_simulations=100,
                                 seed=42, volatility_model="constant")
        pd.testing.assert_frame_equal(p_default, p_const)

    def test_garch_heavier_tails_than_constant(self, market_data):
        """GARCH + Student-t should produce heavier tails than Constant + Normal."""
        close, returns = market_data
        p_const = simulate_paths(close, returns, n_days=252, n_simulations=20000,
                                 seed=42, volatility_model="constant")
        p_garch = simulate_paths(close, returns, n_days=252, n_simulations=20000,
                                 seed=42, volatility_model="garch")
        assert kurtosis(p_garch.iloc[-1]) > kurtosis(p_const.iloc[-1])

    def test_garch_normal_innovation(self, market_data):
        """GARCH with forced Normal innovation should still work."""
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=60, n_simulations=1000,
                               seed=42, volatility_model="garch", innovation="normal")
        assert paths.shape == (60, 1000)
        assert (paths > 0).all().all()


class TestFitGarch:

    def test_returns_expected_keys(self, market_data):
        _, returns = market_data
        result = fit_garch(returns)
        expected = {"omega", "alpha", "beta", "long_run_vol", "last_variance",
                    "last_resid", "persistence"}
        assert set(result.keys()) == expected

    def test_params_positive(self, market_data):
        _, returns = market_data
        result = fit_garch(returns)
        assert result["omega"] > 0
        assert result["alpha"] > 0
        assert result["beta"] > 0

    def test_stationarity(self, market_data):
        _, returns = market_data
        result = fit_garch(returns)
        assert result["persistence"] < 1.0

    def test_long_run_vol_reasonable(self, market_data):
        _, returns = market_data
        result = fit_garch(returns)
        assert 0.01 < result["long_run_vol"] < 5.0


# ---------------------------------------------------------------------------
# Cross-model property tests
# ---------------------------------------------------------------------------

class TestCrossModelProperties:

    @pytest.mark.parametrize("vol", ["constant", "garch"])
    def test_all_prices_positive(self, market_data, vol):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=60, n_simulations=1000,
                               seed=42, volatility_model=vol)
        assert (paths > 0).all().all()

    @pytest.mark.parametrize("vol", ["constant", "garch"])
    def test_cvar_le_var_all_models(self, market_data, vol):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=100, n_simulations=5000,
                               seed=42, volatility_model=vol)
        fp = paths.iloc[-1]
        ip = close.iloc[-1]
        var = compute_var(fp, ip, 0.95)
        cvar = compute_cvar(fp, ip, 0.95)
        assert cvar <= var

    @pytest.mark.parametrize("vol", ["constant", "garch"])
    def test_prob_gain_plus_loss_eq_one_all_models(self, market_data, vol):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=100, n_simulations=5000,
                               seed=42, volatility_model=vol)
        fp = paths.iloc[-1]
        ip = close.iloc[-1]
        summary = simulation_summary(fp, ip)
        assert abs(summary["prob_gain"] + summary["prob_loss"] - 1.0) < 1e-9

    @pytest.mark.parametrize("vol", ["constant", "garch"])
    def test_scenarios_sum_to_one_all_models(self, market_data, vol):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=100, n_simulations=5000,
                               seed=42, volatility_model=vol)
        fp = paths.iloc[-1]
        ip = close.iloc[-1]
        buckets = scenario_buckets(fp, ip)
        assert abs(sum(buckets.values()) - 1.0) < 1e-9
