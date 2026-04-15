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
| RL-026 | Refactor shock generation: model-owned innovations | Done | monte_carlo, vol_constant, vol_garch, regime_hmm, regime_gmm, streamlit_app, tests | Each model owns its innovation distribution. Constant=Normal, GARCH=Student-t, HMM/GMM=Normal |
| RL-027 | MS-GARCH: Markov-Switching GARCH + EVT tails | Done | ms_garch.py (new), monte_carlo, backtesting, tests | HMM regimes + per-regime GARCH + per-regime GPD tails, 23 new tests |
| RL-028 | Controlled comparison notebook | Done | notebooks/model_deep_dive.ipynb | 3 tiers x 3 assets, isolates tail vs vol effect, backtesting, conclusions |
| RL-029 | App simplification to 3 tiers | Done | app/streamlit_app.py | Baseline/GARCH+t/MS-GARCH+EVT, removed HMM/GMM standalone, tier expander |
| RL-030 | Fix `conservative_rank` bug | Done | src/analytics/backtesting.py | Lowest breach rate now ranks as most conservative |
| RL-031 | Multi-seed stability analysis | Done | src/analytics/seed_robustness.py, notebooks/, reports/seed_robustness_results.md | Reusable module shared by notebook + app; CV per metric; stability ≠ accuracy |
| RL-032 | Diagnose MS-GARCH backtest failure | Done | notebooks/model_deep_dive.ipynb, reports/ms_garch_diagnosis_results.md | Root cause: regime-sliced GARCH collapses alpha to 0 in sparse regimes |
| RL-033 | MS-GARCH stage 1 — regime transition before return generation | Done | src/analytics/ms_garch.py, reports/ms_garch_fix_results.md | 11.7% → 7.0% breach, Kupiec FAIL → PASS |
| RL-034 | MS-GARCH unified — global GARCH + per-regime omega + per-regime GPD | Done | src/analytics/ms_garch.py, docs/decisions/ms_garch_unified.md, reports/ms_garch_unified_results.md | 7.0% → 5.4% breach, Kupiec p=0.74; shared-persistence assumption documented |
| RL-035 | "When Each Model Fails" notebook section | Done | notebooks/model_deep_dive.ipynb, reports/rl035_when_each_model_fails.md | Evidence-led failure-mode taxonomy; Baseline beats GARCH+t overall (counter-intuitive) |
| RL-036 | Conditional backtest — bull vs bear regime | Pending | src/analytics/backtesting.py, notebooks/ | Regime-conditional model performance |
| RL-037 | Update notebook conclusions to match actual results | Done | notebooks/model_deep_dive.ipynb | Evidence-led conclusions citing RL-031..RL-035; duplicate "## 8 Conclusions" cell deleted (RL-036 deferred — does not block) |
| RL-038 | Elevate horizon crossover as headline finding | Done | notebooks/horizon_crossover.ipynb (new), notebooks/model_deep_dive.ipynb, reports/horizon_crossover_results.md | Standalone notebook (Option A split). Crossover BTC≈63d, NVDA≈126d, SPY≈5d (oscillating). Prior intuition was wrong — SPY crosses earliest |
| RL-039 | MS-GARCH fit quality transparency | Done | src/analytics/ms_garch.py, app/streamlit_app.py | Reframed post-RL-034: surface global GARCH fit failure + per-regime GPD fallback (n<50 vs fit failed); info box shows shared α/β/persistence once + per-regime σ_lr & tail |
| RL-040 | Surface Student-t df<3 as stability warning | Done   | app/streamlit_app.py, docs/tickets/RL-040.md | Two-tier UI: df<2 → st.error (infinite variance), 2≤df<3 → st.warning (finite var, infinite kurtosis). Uses df_raw, steers to MS-GARCH+EVT. Simulator's 2.1 numerical floor untouched |
| RL-041 | Model recommendation logic in app (phase 4) | Done | app/streamlit_app.py | Hard-coded horizon×ticker recommendation table from RL-038; success/warning panel; SPY non-monotonic note; "deepest VaR ≠ best calibrated" caveat |
| RL-042 | Polish storytelling: split deep dive notebook + consolidate conclusions & assumptions | Done | notebooks/{model_comparison,validation_backtesting,model_diagnosis}.ipynb (new), docs/{conclusions,assumptions}.md (new), README, project_overview, mathematical_reference | Split 41-cell model_deep_dive.ipynb into 3 focused notebooks (35/12/22 cells); source deleted after verification. Conclusions & assumptions promoted to /docs with actual observed numbers from the run. Verification uncovered that the old source had stale pre-RL-034 outputs; new notebooks match current codebase behavior (MS-GARCH+EVT now best-calibrated: 4.1% breach, Kupiec p=0.464 PASS). RL-040 reframed (no cap, surface df<3 as warning) |
| RL-043 | Retroactive: Streamlit backtest tab stabilization | Done (retro) | app/streamlit_app.py (commits 4031bc3, 41e277c, 66823f4) | Documents the 3 untagged 2026-04-10 commits that unblocked the VaR Backtest tab on Streamlit Cloud: dropped Tail Risk tab (EVT/XGB crashes), removed interactive controls + EVT backtest, and replaced per-window refit with upstream fit-reuse. Introduces lookahead bias in the UI backtest — notebook remains the strict reference. Flags follow-up work (coarse-cadence refit, caching, honest approximation note) |
| RL-044 | Fix UI backtest lookahead bias (coarse-cadence refit + session cache + progress bar) | Done | src/analytics/backtesting.py, tests/test_backtesting.py, app/streamlit_app.py | Added `refit_every` (default 1 — notebook unchanged) and `progress_callback` to `backtest_var`. UI refits every 21 trading days (monthly cadence, no lookahead), caches per `(ticker, tier, last_date, confidence)` in `st.session_state`, shows live per-window progress bar on first run. Honesty caption on the tab. 35/35 tests pass |
| RL-045 | Redefine `data/` as flat yfinance cache + fix CI build backend | Done | src/config.py, src/data/storage.py, app/streamlit_app.py, main.py, tests/test_data_storage.py, .gitignore, pyproject.toml | Collapsed `data/{raw,interim,processed}/` into single gitignored `data/`. `storage.py` rewritten as `get_or_fetch/clear_cache/list_cached` keyed by `(ticker,start,end)`. Fixed `pyproject.toml` build-backend typo (`setuptools.backends._legacy:_Backend` → `setuptools.build_meta`) that was blocking CI `pip install -e .` on every run |
