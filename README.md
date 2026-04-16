# RiskLens — Investor Risk Assistant

Decision-support tool for financial asset risk assessment. Combines historical risk analytics, Monte Carlo simulation with multiple volatility models, and statistical backtesting to give investors a clear, quantitative view of downside risk.

Built with Python and Streamlit. Data sourced from Yahoo Finance.

## Quick Start

```bash
git clone <repo-url>
cd RiskLens
python -m venv .venv
source .venv/Scripts/activate    # Windows
# source .venv/bin/activate      # Mac/Linux
pip install -r requirements.txt

streamlit run app/streamlit_app.py   # Launch the app
pytest tests/ -v                     # Run tests
```

## What It Does

**3-tier risk modeling** — each tier adds something the previous one structurally cannot capture:

| Tier | Volatility | Shocks | What it captures |
|---|---|---|---|
| Baseline | Constant | Normal | Textbook GBM (null model) |
| GARCH+t | GARCH(1,1) | Student-t | Vol clustering + fat tails |
| MS-GARCH+EVT | Regime-switching GARCH | Per-regime GPD | Market state changes + extreme tails |

Plus two non-MC modules: **EVT** (GPD/POT tail risk) and **XGBoost conditional quantile regression** (nonparametric VaR that audits parametric models — ML validates, it doesn't replace).

**Key finding:** there is no single best model — ranking depends on horizon and asset. The app recommends the appropriate tier based on configuration.

## Project Structure

```
RiskLens/
├── app/streamlit_app.py          # Interactive dashboard (pure UI)
├── src/
│   ├── data/                     # Fetch, validate, process, store
│   └── analytics/                # Risk metrics, Monte Carlo, GARCH, MS-GARCH,
│                                 # EVT, XGBoost, backtesting
├── tests/                        # 271 deterministic tests
├── notebooks/
│   ├── risk_analysis_walkthrough.ipynb   # Guided tour
│   ├── model_comparison.ipynb           # 3-tier comparison, seed robustness
│   ├── validation_backtesting.ipynb     # EVT/XGBoost audit + rolling backtest
│   ├── model_diagnosis.ipynb            # Failure-mode taxonomy
│   └── horizon_crossover.ipynb          # When each tier wins
├── docs/                         # Documentation (see below)
├── tickets/                      # Work tracking (RL-001..RL-047)
└── reports/                      # Internal notes (gitignored)
```

## Documentation

| Document | What it covers |
|---|---|
| [`docs/project_overview.md`](docs/project_overview.md) | Scope, architecture, notebook structure |
| [`docs/conclusions.md`](docs/conclusions.md) | Full prose synthesis of every empirical finding |
| [`docs/decisions/models.md`](docs/decisions/models.md) | Every model decision, rationale, and trade-offs |
| [`docs/assumptions.md`](docs/assumptions.md) | What the models assume and when they break |
| [`docs/mathematical_reference.md`](docs/mathematical_reference.md) | Equations and formal definitions |

## Tech Stack

Python 3.10+ · Streamlit · pandas · NumPy · SciPy · arch · scikit-learn · XGBoost · hmmlearn · pytest · Ruff

## License

MIT
