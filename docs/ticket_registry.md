# Ticket Registry

| ID     | Title                          | Status | Files                         | Notes |
|--------|--------------------------------|--------|-------------------------------|-------|
| RL-001 | Project bootstrap              | Done   | structure, config, docs       |       |
| RL-002 | Data ingestion layer           | Done   | src/data/, src/config.py, main.py |  |
| RL-003 | Environment & reproducibility  | Done   | requirements.txt, .gitignore, README.md, Makefile | |
| RL-004 | No-accumulation data policy    | Done   | src/data/storage.py, main.py, tests/ | |
| RL-005 | Documentation cleanup          | Done   | docs/                         | Simplified workflow, removed prompts registry |
| RL-006 | Historical risk analytics      | Done   | src/analytics/risk_metrics.py | Series-in API, pure functions |
| RL-007 | Monte Carlo + analysis notebook | Done   | src/analytics/monte_carlo.py, notebooks/ | GBM, VaR/CVaR |
| RL-008 | Streamlit app (full UI)         | Done   | app/streamlit_app.py          | Pure integration, no logic |
| RL-009 | Test suite                      | Done   | tests/                        | 75 tests, exact assertions, edge cases |
| RL-010 | CI + Deployment                  | Done   | .github/workflows/, .streamlit/ | GitHub Actions, Streamlit Cloud config |
| RL-011 | Final presentation package       | Done   | notebooks/, docs/             | Clean walkthrough notebook, removed scratch |
| RL-012 | Repo & README polish             | Done   | README.md, LICENSE, .gitignore, src/ | Removed empty dirs, MIT license, full README |
| RL-013 | Fat-tailed Monte Carlo (Student-t) | Done | src/analytics/monte_carlo.py, app/, tests/ | t-distributed shocks, auto-fit df, UI toggle |
| RL-014 | GARCH(1,1) volatility model       | Done | src/analytics/monte_carlo.py, app/, tests/ | Time-varying vol, arch library, UI toggle |
| RL-015 | Test refactor + math reference + model comparison | Done | tests/, docs/, notebooks/ | Property-based tests, 61 total, math doc, comparison notebook |
| RL-016 | HMM regime detection | Done | src/analytics/regime_hmm.py, vol_constant.py, vol_garch.py, tests/, app/ | hmmlearn, drift+vol per regime, n=1/2/3, model refactor |
| RL-017 | GMM clustering + classifier regime detection | Done | src/analytics/regime_gmm.py, tests/, app/ | sklearn GMM+RF, feature engineering, n=1/2/3 |
| RL-018 | Regime model comparison notebook | Done | notebooks/regime_comparison.ipynb | HMM vs GMM × 1/2/3 regimes, risk impact study |
| RL-019 | Realized volatility regression (XGBoost) | Done | src/analytics/vol_rvol.py, tests/, app/ | 5/10/21d horizons, feature engineering, ML-predicted σ |
| RL-020 | Notebook consolidation (3→2) | Done | notebooks/ | Walkthrough + Deep Dive, deleted 3 stale notebooks |
| RL-021 | VaR Backtesting | Done | src/analytics/backtesting.py, tests/, app/, notebooks/, docs/ | Kupiec + Christoffersen tests, rolling-window engine, Streamlit tab |
| RL-022 | v0.2 wrap: vol cutoff, model comparison, cleanup | Superseded | vol_rvol, backtesting, monte_carlo, risk_metrics, vol_garch, config, README | XGB horizon cutoff, compare_models(), RNG fix, constant consolidation, dead code removal, README rewrite |
| RL-023 | EVT tail risk + XGB model removal | Done | evt.py (new), vol_rvol.py (deleted), monte_carlo, backtesting, streamlit_app, tests | GPD/POT tail risk, removed broken XGB vol model, 22 new tests |
| RL-024 | XGB conditional quantile regression | Done | xgb_var.py (new), tests, notebooks | Nonparametric VaR via quantile regression, 3-way comparison (Normal/EVT/XGB), notebook updates |
| RL-025 | XGBoost temporal CV + EVT backtesting | Done | xgb_var, backtesting, streamlit_app, tests | TimeSeriesSplit CV for XGB hyperparams, EVT walk-forward backtest, 14 new tests |
