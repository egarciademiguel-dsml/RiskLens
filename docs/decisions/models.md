# Model & Simulation Decisions

## Context

RiskLens started as a single-model GBM simulator (constant vol + Normal shocks). Over successive tickets the engine evolved into a **three-tier architecture** where each tier owns its volatility dynamics and shock distribution. This document records every model decision, its rationale, and the empirical evidence that confirmed or overturned it.

**Supporting references:**
- Full assumptions register: [`../assumptions.md`](../assumptions.md)
- Empirical results & prose synthesis: [`../conclusions.md`](../conclusions.md)
- Mathematical definitions: [`../mathematical_reference.md`](../mathematical_reference.md)

---

## 1. Shock Distribution (RL-013)

**Options considered:** Normal, Student-t, Skewed-t, Generalized Hyperbolic, EVT/GPD.

**Chosen:** Normal (default) + Student-t (user toggle).

- Student-t captures fat tails (excess kurtosis) observed in real financial returns.
- Degrees of freedom auto-fitted via MLE (`scipy.stats.t.fit`).
- Shocks standardized to unit variance: `Z = T / sqrt(df / (df - 2))`.
- df clamped >= 2.1 to ensure finite variance.
- More advanced distributions (skewed-t, GHD, EVT) deferred at this stage — Student-t is the best signal-to-complexity ratio for an MVP.

## 2. Volatility Model (RL-014)

**Options considered:** Constant sigma, GARCH(1,1), EGARCH, GJR-GARCH, Regime-switching.

**Chosen:** Constant (default) + GARCH(1,1) (user toggle).

- GARCH(1,1) captures volatility clustering: `σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)`.
- Fitted via `arch` library (MLE, `disp="off"`).
- Day-by-day variance simulation (not vectorizable — each day depends on previous).
- Jensen's drift correction uses per-day sigma: `drift_t = μ - 0.5·σ_t²`.
- `arch` expects percentage-scale returns; params converted to decimal for simulation.
- Asymmetric GARCH variants (EGARCH, GJR) deferred — GARCH(1,1) is the industry baseline.

## 3. HMM Regime Detection (RL-016)

**Chosen:** `hmmlearn.GaussianHMM` with n_regimes = 2 (calm / crisis).

- Gaussian emissions on daily returns — mild assumption since HMM only provides regime labels, not return prediction.
- Decoding via Viterbi (hard assignments). Soft posteriors considered but not needed: tail modeling is delegated to per-regime GPD, not the emission distribution.
- Regimes sorted by σ ascending: regime 0 = calm, regime 1 = crisis.
- 3-regime configs exist in codebase but are not used in the production pipeline — 2 regimes capture the dominant bimodality without sparse-regime identification problems on a 252-day window.

## 4. EVT / GPD and XGBoost Vol Removal (RL-023)

**Added:** Extreme Value Theory via Peaks-Over-Threshold (GPD).

- GPD fitted on left-tail exceedances above the 10th percentile threshold (see [`assumptions.md`](../assumptions.md) §4.1).
- Theoretically justified by Pickands–Balkema–de Haan theorem: exceedance distribution converges to GPD for sufficiently high threshold.
- Sits as an independent tail estimator alongside the MC pipeline, later integrated into MS-GARCH as per-regime GPD tails.

**Removed:** XGBoost realized-vol model.

- Evaluation showed it was functionally identical to constant vol — predicted σ was nearly flat across time, adding complexity without dynamic behavior.
- Removal actually strengthens the ML narrative: "I built it, evaluated it, saw it was constant-σ in disguise, killed it."

## 5. Model-Owned Innovations (RL-026)

**Decision:** Each volatility model owns its shock distribution internally.

| Volatility model | Innovation distribution |
|---|---|
| Constant | Normal(0,1) |
| GARCH | Student-t (auto-fitted df) |
| MS-GARCH | EVT/GPD (per-regime) |

Before RL-026 all models shared a single innovation toggle, which meant the 3-tier comparison collapsed to scaled versions of the same underlying process. Model-owned innovations make the tiers genuinely different and enable the pure-tail / pure-vol effect decomposition in `model_comparison.ipynb`.

## 6. Original Model Combinations

The early 2×2 grid (RL-013 × RL-014):

| Configuration | Shocks | Volatility | Role |
|---|---|---|---|
| Constant + Normal | N(0,1) | Fixed σ | **Tier 1 — Baseline** |
| Constant + Student-t | t(df) | Fixed σ | Diagnostic only (pure tail effect) |
| GARCH + Normal | N(0,1) | GARCH(1,1) | Diagnostic only (pure vol effect) |
| GARCH + Student-t | t(df) | GARCH(1,1) | **Tier 2 — GARCH+t** |

All defaults preserve v0.1 behavior (`distribution="normal"`, `volatility_model="constant"`).

## 7. MS-GARCH — Three Iterations (RL-027 → RL-033 → RL-034)

The evolution of the third tier is the central architectural arc of the project.

### 7a. Original MS-GARCH (RL-027)

Per-regime GARCH fitted on regime-filtered slices + per-regime GPD on regime-filtered residuals.

**Failure:** On BTC the crisis regime had only ~44 observations. GARCH on 44 points collapsed α to exactly 0 — turning "regime-switching GARCH" into regime-switching constant-vol. Backtest breach rate: **11.7%** (Kupiec FAIL).

### 7b. Transition-Order Fix (RL-033)

Moved regime transition *before* return generation in the simulation loop. Original code generated the return first, then transitioned — so 1-day simulations always used the current (usually calm) regime.

**Impact:** Breach rate dropped from 11.7% → **7.0%** (Kupiec PASS, p = 0.123). But α = 0 in crisis regime remained.

### 7c. Unified Specification (RL-034) — Current

**Two-layer approach:**

1. **Shared GARCH persistence.** Fit GARCH(1,1) once on the full pooled return series → global (α, β). Reparametrize ω per regime via variance targeting: `ω_k = σ²_lr,k · (1 − α − β)` where `σ²_lr,k` = per-regime sample variance.
2. **Per-regime GPD tail.** Fit GPD on global standardized residuals filtered by regime label. Fallback to Normal when < 50 exceedances (surfaced explicitly in UI per RL-039).

**Literature justification for shared persistence:** Haas, Mittnik & Paolella (2004), Klaassen (2002), Marcucci (2005), Ardia et al. (2018) converge on intercept-only switching as most stable. Fully-switching GARCH suffers path-dependence — σ² at time t would depend on every possible regime history up to t.

**Backtest progression:** breach rate improved from 11.7% (FAIL) → 7.0% → 4.1% (PASS, best-calibrated tier). Full results in [`conclusions.md`](../conclusions.md) §7.

## 8. Three-Tier Architecture — Final

| Tier | Vol model | Shocks | Role |
|---|---|---|---|
| **Baseline** | Constant σ | Normal | Null model / textbook GBM |
| **GARCH+t** | GARCH(1,1) | Student-t | Classical stochastic vol + fat tails |
| **MS-GARCH+EVT** | Global GARCH + per-regime ω | Per-regime GPD | Regime-switching + EVT tails |

**Design principle:** The tiers are not "good → better → best." Each fails in a different, nameable way (see §14). The right tier depends on horizon, asset, and which failure mode the user can tolerate.

## 9. Student-t df < 3 Warning — Surface, Don't Cap (RL-040 / RL-042)

**Decision:** Do not clamp df at 3. Instead surface a two-tier warning in the app.

- df < 2.0 → **error** (infinite variance, model meaningless).
- 2.0 ≤ df < 3.0 → **warning** (finite variance, infinite kurtosis — GARCH's long-run variance theoretically undefined). Steer user toward MS-GARCH+EVT.

**Rationale:** Capping hides information. On this run BTC fits at df = 2.7 — the warning fires and the user sees "your model knows when it's lying." Consistent with the transparency philosophy (RL-039 surfaces GPD fallbacks the same way).

## 10. Horizon-Aware Recommendation (RL-038 / RL-041)

**Finding:** Model ranking flips with horizon. There is no single best tier.

| Asset | Crossover horizon (MS-GARCH+EVT overtakes GARCH+t) |
|---|---|
| SPY | ≈ 5 days (oscillating) |
| BTC | ≈ 63 days |
| NVDA | ≈ 126 days |

**Counter-intuitive:** SPY (thinnest tails) crosses first. Crossover depends on the ratio of regime-mixing volatility to single-regime volatility, not tail heaviness.

**Implementation (RL-041):** Hard-coded recommendation table (asset × horizon → tier) in the Streamlit app. Shows match/mismatch vs user's manual selection.

**Implication:** Vol dynamics dominate short horizons; tail shape dominates long horizons. This split is why the app surfaces different tiers at different horizons rather than a single "best model."

## 11. Backtest Design (RL-033 / RL-042 / RL-044)

| Parameter | Value | Rationale |
|---|---|---|
| Window | 252 trading days | ~1 year — long enough for GARCH identification, short enough to avoid averaging across structural breaks |
| Step | 5 trading days | Weekly granularity — fast enough for interactive use, granular enough for Kupiec/Christoffersen power |
| n_sim | 2,000 per window (notebook) | CV < 5% on all metrics at this count (RL-031) |
| Tests | Kupiec (unconditional coverage) + Christoffersen (breach independence) | Joint pass = well-calibrated |
| Refit | Full refit from scratch every window (notebook); monthly cadence in app (RL-044) | No warm-starting, genuinely out-of-sample |

**RL-044 app refit cadence:** The Streamlit backtest originally reused a single upstream fit across all windows (lookahead bias). Fixed to refit every 21 trading days — between refits the most recent fit is reused (industry-standard practice). Notebook remains the trustworthy reference.

## 12. EVT Diagnostic Suite (RL-046)

**Added:** Formal validation of the GPD fit used in the MS-GARCH+EVT tier.

- Mean residual life plot — visual check for threshold appropriateness.
- GPD parameter stability across thresholds — ξ and σ should stabilize above the chosen threshold.
- QQ plots — fitted GPD quantiles vs empirical.
- Kolmogorov-Smirnov goodness-of-fit test.
- Bootstrap confidence intervals on ξ and σ.
- Declustering for iid violation — vol-clustering causes exceedances to cluster rather than arrive as Poisson. Reduces exceedance count by ~50–65% (BTC 53%, SPY 65%, NVDA 60%).

**Finding:** BTC shape CI [−0.14, 0.27] spans zero — not statistically conclusive at 95%. SPY and NVDA crisis regimes produce clearly positive ξ (> 0.5). The diagnostic suite surfaces these differences rather than hiding them.

## 13. XGBoost Conditional Quantile Regression — Tail Underestimation by Design (RL-047)

**Role:** Non-parametric auditor of parametric VaR models. XGBoost is trained as a conditional quantile regressor (pinball loss at α = 0.05 / 0.01) on engineered lag features (rolling vol, realized variance, downside vol, SHAP-explained). It cross-checks the Monte Carlo tiers without assuming anything about the shock distribution.

**Known limitation: systematic tail underestimation.** XGBoost's predicted VaR at 99% is consistently shallower than the EVT/GPD estimate on heavy-tailed assets (BTC ~34% shallower, e.g. −3.14% vs Normal −4.78%). This is structural, not a tuning failure:

- **Tree-based models partition the feature space into finite leaf nodes.** The predicted quantile for any input is a sample quantile of the training observations that fall in that leaf. Extreme tails (< p1) have very few observations per leaf — the model cannot predict beyond what it has seen.
- **Pinball loss is symmetric in coverage, not severity.** A breach of −8% and a breach of −15% contribute identically to the loss if both are below the predicted quantile. The model has no incentive to distinguish "bad" from "catastrophic."
- **No extrapolation mechanism.** Unlike EVT/GPD, which fits a parametric tail beyond the data, XGBoost interpolates within the convex hull of observed returns. The deepest VaR it can ever predict is bounded by the deepest return in the training window.

**Decision: keep XGBoost as-is.** The tail underestimation is accepted because XGBoost's role is *audit*, not replacement. Its value lies in providing a model-free conditional benchmark — if the parametric tiers disagree with XGBoost at 95%, something is wrong with the parametric model. At 99%, divergence is *expected and informative*: it shows where parametric tail assumptions (Student-t df, GPD ξ) are doing real work that a nonparametric method cannot replicate. The gap between XGBoost and EVT is itself a finding: it quantifies how much work the parametric tail assumption is doing.

**Possible improvements (evaluated, not implemented):**

1. **Asymmetric sample weights.** Upweight tail observations during training (e.g., `weight = 5 if return < p10 else 1`). Forces tree splits to pay more attention to the left tail, improving leaf purity in the extreme region. Trade-off: biases the body of the distribution and reduces accuracy at 95%.

2. **Two-stage model.** First-stage classifier detects "tail regime" (return < p5), second-stage quantile regressor trains only on tail-regime observations for a richer feature partition. Trade-off: hard boundary at the first stage + very few training observations for the second → overfitting risk.

3. **EVT residual overlay.** Use XGBoost to predict conditional location and scale, then fit GPD to the standardized residuals' tail. Combines XGBoost's conditional adaptivity with EVT's extrapolation capability. Trade-off: significant complexity and the residual GPD fit depends on XGBoost's conditional mean/scale estimates, which may not be well-calibrated in the tails.

4. **Deeper trees / more estimators.** Increasing `max_depth` or `n_estimators` creates more leaf nodes, potentially isolating tail observations better. Trade-off: diminishing returns — the fundamental issue is sample scarcity in the tails, not model capacity.

None of these were implemented because they would shift XGBoost from auditor to competitor, conflicting with the project's design principle of keeping ML in a read-only cross-check role.

## 14. Failure-Mode Taxonomy (RL-035)

The headline finding of the project. Each tier fails in a different, nameable way: Baseline suffers **lagged level adjustment**, GARCH+t suffers **post-crisis amnesia**, MS-GARCH+EVT suffers **detection lag** at regime turns. The right tier depends on which failure mode the user can tolerate.

Full per-regime breach tables and failure-mode narratives: [`conclusions.md`](../conclusions.md) §12.

## Trade-offs

**Pros:**
- Realistic tail risk and volatility dynamics across three tiers
- Each tier addresses a different risk regime / horizon combination
- Auto-fitted from data — no manual parameter tuning
- Backward compatible (defaults preserve v0.1 behavior)
- XGBoost auditor provides model-free cross-check without driving decisions
- Transparent about limitations (df warnings, GPD fallbacks, tail underestimation)

**Cons:**
- GARCH day-by-day loop is slower than vectorized constant-vol (acceptable for current scale)
- No asymmetric volatility (leverage effect) — EGARCH would capture this
- Student-t is symmetric — doesn't model left-skew separately from right-skew
- Historical params assumed stationary into the future
- HMM detection lag at regime turns is structural and unsolved
- XGBoost cannot extrapolate beyond observed returns (by construction)
- Backtest validated on BTC only — SPY/NVDA analyzed but not backtested

## Future Considerations

### Regime-Dependent Ensemble Risk Engine

A production-grade extension would be a **regime-aware ensemble or model-selection layer**, where model weights depend on the current market state detected by the HMM. Each regime plays to the strengths of the models that perform best under its conditions:

| Regime | Ensemble weights (illustrative) | Rationale |
|---|---|---|
| **Calm** | 70% Baseline + 30% XGBoost | Vol is stable — Baseline is well-calibrated (0% breach), XGBoost adds conditional adaptivity |
| **Stress** | 60% MS-GARCH+EVT + 40% EVT standalone | Tail shape dominates — both models extrapolate beyond observed data via GPD |
| **Transition** | 50% XGBoost + 50% GARCH+t | Regime uncertain — XGBoost's lag features detect shifts the HMM hasn't confirmed yet; GARCH+t adapts σ in real time |

This was intentionally left out of scope to preserve interpretability and keep validation focused on individual model behavior. The failure-mode taxonomy in §14 exists precisely because each tier is evaluated in isolation — an ensemble would obscure which component is miscalibrated and make the diagnostic story untellable. In a production system where calibration matters more than explainability, regime-dependent weighting is the natural next step.

### Other

- EGARCH / GJR-GARCH for leverage effect (asymmetric vol response to negative shocks)
- Skewed Student-t or GHD for asymmetric tail modeling
- ES backtest (Acerbi–Szekely) and Basel traffic-light test for regulatory framing
- Multi-asset: copulas for dependency modeling, portfolio-level VaR
- Stress testing / scenario shocks (jump-diffusion à la Merton)
- Conditional backtest on bull/bear sub-periods (RL-036, deferred)
- Improved regime detection latency (time-varying transition probabilities, online HMM)
