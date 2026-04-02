# Project Overview

## Project Name

RiskLens — Investor Risk Assistant

## Purpose

RiskLens is a decision-support application designed to help investors assess financial asset risk through data-driven analysis and clear, interpretable outputs. It provides contextual risk information rather than trading recommendations.

## MVP Scope

- Historical risk analysis (volatility, drawdown, rolling metrics).
- Monte Carlo simulation for forward-looking risk estimation.
- Later phase: news sentiment integration for contextual risk signals.

## Non-Goals

- Price prediction.
- Trading signals.
- Portfolio management.

## Architecture Summary

- **Frontend**: Streamlit application (`app/streamlit_app.py`).
- **Core logic**: Python modules under `src/` organized by domain responsibility (`data`, `analytics`, `pipelines`, `services`, `domain`, `utils`).
- **Tests**: Located in `tests/`.
- **Documentation**: Located in `docs/`.

## Tooling Summary

- **Language**: Python 3.10+
- **UI**: Streamlit
- **Data**: pandas, NumPy, yfinance
- **Visualization**: matplotlib, Plotly
- **NLP**: Transformers, Trafilatura
- **ML**: scikit-learn
- **Quality**: Ruff, pre-commit, pytest

## Product Principles

- **Simple inputs** — The user provides minimal information to get meaningful results.
- **Interpretable outputs** — All results are presented clearly with no black-box metrics.
- **Minimal friction** — The interface stays out of the way and focuses on delivering value.
- **No unnecessary complexity** — Every feature must justify its existence.
