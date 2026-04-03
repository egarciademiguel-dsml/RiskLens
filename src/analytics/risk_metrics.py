"""Historical risk metrics. All functions take pd.Series, return scalars or Series."""

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(close: pd.Series) -> float:
    return drawdown_series(close).min()


def drawdown_series(close: pd.Series) -> pd.Series:
    cummax = close.cummax()
    return (close - cummax) / cummax


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns.mean() - risk_free_rate / TRADING_DAYS_PER_YEAR
    if returns.std() == 0:
        return 0.0
    return (excess / returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe but only penalizes downside volatility."""
    excess = returns.mean() - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0.0
    return (excess / downside) * np.sqrt(TRADING_DAYS_PER_YEAR)


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """Sum of all returns / sum of absolute losses."""
    total = returns.sum()
    pain = returns[returns < 0].abs().sum()
    if pain == 0:
        return float("inf")
    return total / pain


def best_worst_periods(returns: pd.Series, window: int = 21) -> dict:
    """Best and worst rolling N-day cumulative returns."""
    rolling = returns.rolling(window).sum()
    return {
        "best": float(rolling.max()),
        "worst": float(rolling.min()),
        "window": window,
    }


def return_statistics(returns: pd.Series) -> dict:
    return {
        "mean_daily": returns.mean(),
        "std_daily": returns.std(),
        "annualized_mean": returns.mean() * TRADING_DAYS_PER_YEAR,
        "annualized_std": annualized_volatility(returns),
        "skewness": stats.skew(returns.dropna()),
        "kurtosis": stats.kurtosis(returns.dropna()),
        "min": returns.min(),
        "max": returns.max(),
    }
