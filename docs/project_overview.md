# Project Overview

## RiskLens — Investor Risk Assistant

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
- **Docs**: `docs/`

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
