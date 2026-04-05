import pandas as pd
import numpy as np
import pytest

from src.analytics.monte_carlo import (
    simulate_paths,
    compute_var,
    compute_cvar,
    prob_target,
    scenario_buckets,
    simulation_summary,
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
    """Pre-computed simulation for reuse across tests."""
    close, returns = market_data
    paths = simulate_paths(close, returns, n_days=100, n_simulations=5000, seed=42)
    final_prices = paths.iloc[-1]
    initial_price = close.iloc[-1]
    return paths, final_prices, initial_price


class TestSimulatePaths:

    def test_shape(self, market_data):
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=60, n_simulations=1000, seed=0)
        assert paths.shape == (60, 1000)

    def test_row_zero_is_not_initial_price(self, market_data):
        """Current convention: row 0 = day 1 (first simulated day), not initial price."""
        close, returns = market_data
        paths = simulate_paths(close, returns, n_days=10, n_simulations=100, seed=0)
        initial = close.iloc[-1]
        # Row 0 should differ from initial price (it's already one step forward)
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


class TestVarCvar:

    def test_var_exact_value(self, sim_results):
        _, final_prices, initial_price = sim_results
        var = compute_var(final_prices, initial_price, 0.95)
        assert abs(var - (-0.2620556777)) < 1e-4

    def test_cvar_exact_value(self, sim_results):
        _, final_prices, initial_price = sim_results
        cvar = compute_cvar(final_prices, initial_price, 0.95)
        assert abs(cvar - (-0.3168622943)) < 1e-4

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


class TestProbTarget:

    def test_prob_gain_consistency(self, sim_results):
        """prob_target(0.0) must equal prob_gain — both use strict > 0."""
        _, final_prices, initial_price = sim_results
        pt = prob_target(final_prices, initial_price, 0.0)
        summary = simulation_summary(final_prices, initial_price)
        assert pt == summary["prob_gain"]

    def test_exact_prob_gain(self, sim_results):
        _, final_prices, initial_price = sim_results
        assert prob_target(final_prices, initial_price, 0.0) == 0.5550

    def test_prob_decreases_with_higher_target(self, sim_results):
        _, final_prices, initial_price = sim_results
        p10 = prob_target(final_prices, initial_price, 0.10)
        p50 = prob_target(final_prices, initial_price, 0.50)
        assert p50 <= p10

    def test_extreme_target_returns_zero(self, sim_results):
        _, final_prices, initial_price = sim_results
        assert prob_target(final_prices, initial_price, 100.0) == 0.0


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

    def test_exact_prob_values(self, sim_results):
        _, final_prices, initial_price = sim_results
        summary = simulation_summary(final_prices, initial_price)
        assert summary["prob_gain"] == 0.5550
        assert summary["prob_loss"] == 0.4450

    def test_mean_final_price(self, sim_results):
        _, final_prices, initial_price = sim_results
        summary = simulation_summary(final_prices, initial_price)
        assert abs(summary["mean_final_price"] - 121.3655101365) < 0.1


class TestInvalidInputs:

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
