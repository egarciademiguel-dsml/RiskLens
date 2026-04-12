# Project Overview

## VaRify — Investor Risk Assistant

Decision-support app for contextual financial asset risk assessment. Built with Python and Streamlit.

## MVP Scope

- **Historical risk analysis** — returns, volatility, drawdowns, rolling metrics.
- **Monte Carlo simulation** — VaR, CVaR, forward-looking risk estimation. Supports Normal/Student-t shocks and Constant/GARCH(1,1) volatility models.
- **Optional: news sentiment** — lightweight contextual risk signals.

## Non-Goals

- Price prediction
- Trading signals
- Portfolio management

## Architecture

- **Frontend**: Streamlit (`app/streamlit_app.py`)
- **Core logic**: Python modules under `src/` (data, analytics)
- **Tests**: `tests/`
- **Docs**: `docs/` — includes [`conclusions.md`](conclusions.md) (full prose synthesis of deep-dive findings) and [`assumptions.md`](assumptions.md) (consolidated modeling assumptions)

## Notebook Structure

- `notebooks/risk_analysis_walkthrough.ipynb` — guided tour through every feature of the app on a single asset.
- `notebooks/model_comparison.ipynb` — controlled comparisons (tail effect, vol effect at 252d and 21d, 3 tiers head-to-head, seed robustness, MS-GARCH internals).
- `notebooks/validation_backtesting.ipynb` — non-MC VaR cross-check (EVT + XGBoost) and rolling-window backtest of the 3 tiers.
- `notebooks/model_diagnosis.ipynb` — why MS-GARCH fails/passes the backtest; failure-mode taxonomy; overall conclusions.
- `notebooks/horizon_crossover.ipynb` — horizon at which each tier's ranking flips; drives the app's horizon-aware recommendation.

The three "model_*" notebooks were split from a single 41-cell `model_deep_dive.ipynb` under [RL-042](tickets/RL-042.md) to improve navigability. Each is self-contained and runs standalone in under ~5 minutes.

## Tooling

- **Language**: Python 3.10+
- **UI**: Streamlit
- **Data**: pandas, NumPy, yfinance
- **Stats**: SciPy, statsmodels, scikit-learn
- **Visualization**: matplotlib, seaborn, Plotly
- **Quality**: Ruff, pre-commit, pytest

## Data Handling Policy

- Fetch only what's needed for current analysis.
- One file per asset maximum, overwrite on refresh.
- No raw data persistence. No growing local store.
- `python main.py --purge` removes all cached data.

## Product Principles

- **Simple inputs** — minimal user effort for meaningful results.
- **Interpretable outputs** — no black-box metrics.
- **Minimal friction** — interface stays out of the way.
- **No unnecessary complexity** — every feature justifies its existence.
