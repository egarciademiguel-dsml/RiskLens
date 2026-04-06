import numpy as np
import pandas as pd
import pytest

from src.analytics.regime_gmm import (
    engineer_regime_features,
    fit_gmm,
    predict_current_regime,
    get_regime_params,
)
from src.analytics.monte_carlo import simulate_paths


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


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestEngineerRegimeFeatures:

    def test_output_is_dataframe(self, market_data):
        _, returns = market_data
        features = engineer_regime_features(returns)
        assert isinstance(features, pd.DataFrame)

    def test_no_nans_after_dropna(self, market_data):
        _, returns = market_data
        features = engineer_regime_features(returns)
        assert not features.isna().any().any()

    def test_expected_columns(self, market_data):
        _, returns = market_data
        features = engineer_regime_features(returns)
        expected = {"returns", "vol_5d", "vol_10d", "vol_21d",
                    "mean_ret_5d", "mean_ret_10d", "mean_ret_21d",
                    "skew_21d", "kurtosis_21d"}
        assert set(features.columns) == expected

    def test_fewer_rows_than_input(self, market_data):
        _, returns = market_data
        features = engineer_regime_features(returns)
        assert len(features) < len(returns)


# ---------------------------------------------------------------------------
# fit_gmm
# ---------------------------------------------------------------------------

class TestFitGmm:

    def test_returns_expected_keys(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=2)
        expected = {"gmm", "classifier", "scaler", "n_regimes", "regime_params",
                    "regime_labels", "features_index", "sort_order"}
        assert set(result.keys()) == expected

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_n_regimes_param_count(self, market_data, n):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=n)
        assert len(result["regime_params"]) == n
        assert result["n_regimes"] == n

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_labels_in_range(self, market_data, n):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=n)
        labels = result["regime_labels"]
        assert labels.min() >= 0
        assert labels.max() <= n - 1

    def test_regimes_sorted_by_sigma(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=3)
        sigmas = [p["sigma"] for p in result["regime_params"]]
        assert sigmas == sorted(sigmas)

    def test_single_regime_matches_global(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=1)
        p = result["regime_params"][0]
        assert abs(p["mu"] - returns.mean()) < 1e-10
        assert abs(p["sigma"] - returns.std(ddof=1)) < 1e-10

    def test_single_regime_no_model(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=1)
        assert result["gmm"] is None
        assert result["classifier"] is None

    def test_invalid_n_regimes(self, market_data):
        _, returns = market_data
        with pytest.raises(ValueError):
            fit_gmm(returns, n_regimes=0)

    def test_each_regime_has_mu_and_sigma(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=2)
        for p in result["regime_params"]:
            assert "mu" in p
            assert "sigma" in p
            assert p["sigma"] > 0

    def test_classifier_exists_for_multi_regime(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=2)
        assert result["classifier"] is not None
        assert result["scaler"] is not None


# ---------------------------------------------------------------------------
# predict_current_regime
# ---------------------------------------------------------------------------

class TestPredictCurrentRegime:

    def test_returns_int(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=2)
        regime = predict_current_regime(result, returns)
        assert isinstance(regime, int)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_regime_in_range(self, market_data, n):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=n)
        regime = predict_current_regime(result, returns)
        assert 0 <= regime < n

    def test_single_regime_always_zero(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=1)
        assert predict_current_regime(result, returns) == 0


# ---------------------------------------------------------------------------
# get_regime_params
# ---------------------------------------------------------------------------

class TestGetRegimeParams:

    def test_returns_all(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=2)
        params = get_regime_params(result)
        assert len(params) == 2

    def test_returns_single(self, market_data):
        _, returns = market_data
        result = fit_gmm(returns, n_regimes=2)
        p = get_regime_params(result, regime_idx=0)
        assert "mu" in p and "sigma" in p


# ---------------------------------------------------------------------------
# MC integration
# ---------------------------------------------------------------------------

class TestGmmMcIntegration:

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_shape(self, market_data, n):
        close, returns = market_data
        gmm_result = fit_gmm(returns, n_regimes=n)
        paths = simulate_paths(
            close, returns, n_days=60, n_simulations=1000,
            seed=42, volatility_model="gmm", gmm_params=gmm_result,
        )
        assert paths.shape == (60, 1000)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_all_prices_positive(self, market_data, n):
        close, returns = market_data
        gmm_result = fit_gmm(returns, n_regimes=n)
        paths = simulate_paths(
            close, returns, n_days=100, n_simulations=5000,
            seed=42, volatility_model="gmm", gmm_params=gmm_result,
        )
        assert (paths > 0).all().all()

    def test_reproducible_with_seed(self, market_data):
        close, returns = market_data
        gmm_result = fit_gmm(returns, n_regimes=2)
        p1 = simulate_paths(close, returns, n_days=30, n_simulations=100,
                            seed=42, volatility_model="gmm", gmm_params=gmm_result)
        p2 = simulate_paths(close, returns, n_days=30, n_simulations=100,
                            seed=42, volatility_model="gmm", gmm_params=gmm_result)
        pd.testing.assert_frame_equal(p1, p2)

    def test_gmm_differs_from_constant(self, market_data):
        close, returns = market_data
        gmm_result = fit_gmm(returns, n_regimes=2)
        p_const = simulate_paths(close, returns, n_days=30, n_simulations=100,
                                 seed=42, volatility_model="constant")
        p_gmm = simulate_paths(close, returns, n_days=30, n_simulations=100,
                               seed=42, volatility_model="gmm", gmm_params=gmm_result)
        assert not p_const.equals(p_gmm)

    def test_gmm_with_t_shocks(self, market_data):
        close, returns = market_data
        gmm_result = fit_gmm(returns, n_regimes=2)
        paths = simulate_paths(
            close, returns, n_days=60, n_simulations=1000,
            seed=42, distribution="t", df_t=5.0,
            volatility_model="gmm", gmm_params=gmm_result,
        )
        assert paths.shape == (60, 1000)
        assert (paths > 0).all().all()

    def test_auto_fit_when_no_params(self, market_data):
        close, returns = market_data
        paths = simulate_paths(
            close, returns, n_days=30, n_simulations=100,
            seed=42, volatility_model="gmm",
        )
        assert paths.shape == (30, 100)
        assert (paths > 0).all().all()
