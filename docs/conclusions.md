# Conclusions

Full prose synthesis of every finding across the deep-dive notebooks. Each section is the expanded version of the short inline conclusion that sits next to the evidence in the notebook. If you want the numbers, go to the notebook section referenced in the heading. If you want the narrative, it is below.

**Source notebooks:**
- [`notebooks/model_comparison.ipynb`](../notebooks/model_comparison.ipynb) — controlled comparisons, 3-tier head-to-head, seed robustness, MS-GARCH internals
- [`notebooks/validation_backtesting.ipynb`](../notebooks/validation_backtesting.ipynb) — EVT/XGBoost audit + rolling backtest
- [`notebooks/model_diagnosis.ipynb`](../notebooks/model_diagnosis.ipynb) — regime stability, per-regime GARCH quality, breach concentration, failure-mode taxonomy
- [`notebooks/horizon_crossover.ipynb`](../notebooks/horizon_crossover.ipynb) — horizon at which each tier's ranking flips, drives the app recommendation ([RL-038](tickets/RL-038.md), [RL-041](tickets/RL-041.md))

**Supporting reports (the raw results backing the claims below):**
- `reports/seed_robustness_results.md`
- `reports/ms_garch_diagnosis_results.md`
- `reports/ms_garch_fix_results.md`
- `reports/ms_garch_unified_results.md`
- `reports/rl035_when_each_model_fails.md`
- `reports/horizon_crossover_results.md`

**Modeling assumptions behind every result below:** [`assumptions.md`](assumptions.md).

---

## §1 — Horizon is the hidden variable

*Source: `horizon_crossover.ipynb`, `reports/horizon_crossover_results.md`.*

The relative ranking of the three tiers depends on **horizon**, not on which model is "more realistic." There is no single best model; there is a best model *for a given horizon on a given asset*. RL-038 mapped the crossover horizons — the horizon at which MS-GARCH+EVT overtakes GARCH+t in VaR depth — for each asset:

- **SPY ≈ 5 days** (oscillating)
- **BTC ≈ 63 days**
- **NVDA ≈ 126 days**

The intuitive prior that tail-heaviness predicts early crossover is wrong: SPY has the thinnest tails of the three and crosses *first*. This headline finding is the thing the app's horizon-aware recommendation in [RL-041](tickets/RL-041.md) consumes. It is also the thing that justifies running the pure-vol-effect comparison (§3 below) at 21 days rather than 252 — 21 days is the horizon at which every asset is still in the "vol-dominant" regime.

## §2 — Pure tail effect (fix vol, vary innovations)

*Source: `model_comparison.ipynb` §2.*

Holding GARCH dynamics fixed and only switching the innovation distribution from Normal to Student-t shifts the deep tail (VaR 99%, CVaR 99%) noticeably but barely moves the 95% quantile. On BTC at 252 days the VaR 95% even moves *up* slightly (−53.11% Normal → −44.99% Student-t) while the CVaR 99% deepens by about 12 percentage points (−72.17% → −84.37%). Student-t's fat tails matter at the tail; at the body of the distribution they are invisible because the two distributions have nearly identical central mass.

**The implication** is that the choice of innovation distribution is a *tail-quantile* decision, not a *whole-distribution* decision. If your use case only cares about 95% VaR at long horizons, Student-t buys you very little over Normal — the visible differences live at 99%+ and in CVaR, not VaR, at 95%. If your use case cares about tail expectations (CVaR) or stress scenarios, Student-t is non-negotiable.

**Fitted df on this run.** BTC df = 2.7, SPY df = 3.8, NVDA df = 5.1. Note that **BTC's fitted df sits below 3.0**, which is the finite-variance threshold flagged in [`assumptions.md`](assumptions.md) §2.4 as a stability warning: the Student-t's theoretical variance is infinite, GARCH's long-run variance is undefined, and the GARCH+t tier's results on BTC should be interpreted with that caveat. This is exactly the "danger-zone" case RL-040 (reframed under RL-042) proposes to surface as a UI warning rather than silently cap.

## §3 — Pure vol effect (fix innovations, vary vol model)

*Source: `model_comparison.ipynb` §3 and §3b.*

This is the comparison where horizon matters most. At the **252-day horizon**, time-varying vol under identical Normal shocks produces a distribution that is nearly indistinguishable from constant-σ. That is not a bug in the comparison; it is the mathematics of GARCH(1,1) mean reversion. With typical persistence α+β ≈ 0.97, the shock half-life is ≈ 23 days. Over 252 days any individual path has mean-reverted roughly 11 half-lives and the simulated distribution is dominated by the unconditional variance — which is exactly what a constant-vol model uses from day one. The two models converge **by construction** at long horizons.

At the **21-day horizon** the story is different. A shock has only decayed to roughly $0.97^{21} \approx 0.52$ of its initial magnitude, so time-varying dynamics are still economically meaningful. The GARCH+Normal and Constant+Normal distributions visibly separate, with the gap largest on BTC (highest persistence, fattest historical tails) and smallest on SPY.

**Why 21 days specifically.** The choice is deliberate:
1. 21 trading days ≈ one calendar month — the standard monthly regulatory analogue (Basel uses 10-day scaling; monthly is its natural extension).
2. Persistence has not yet washed out — a shock retains ~50% magnitude.
3. It is **not cherry-picked**: every asset's horizon-crossover point (§1 above) sits at or above 21 days, which means 21d is in the vol-dominant regime for all three assets.

**The punchline.** *Vol dynamics are a short-horizon story; tail shape is a long-horizon story.* That split is what drives the horizon-dependent tier recommendation in the Streamlit app ([RL-041](tickets/RL-041.md)) and is why the app surfaces different tiers at different horizons rather than a single "best model."

## §4 — 3 tiers head-to-head at 252 days

*Source: `model_comparison.ipynb` §4.*

On BTC and NVDA, MS-GARCH+EVT produces the deepest VaR at 252 days, GARCH+t is middle, Baseline is shallowest. This is the ranking the 3-tier narrative was built on, and at this horizon it holds. On SPY the ranking oscillates across confidence levels — the tail gain from EVT is small when the historical tail is already thin, and the regime structure is weak enough that the MS-GARCH layer adds more variance than signal.

**Critical caveat.** "Deepest VaR" is not the same as "best calibrated." A model that systematically overestimates risk will produce the deepest VaR *and* the lowest breach rate *and* the worst capital efficiency. Depth is a property of the simulated distribution. Calibration is a property of the model's agreement with realized returns. These are independent axes. The backtest in NB2 (§7 below) is the only thing that resolves which depth is right.

## §4b — Seed robustness

*Source: `model_comparison.ipynb` §4b, `reports/seed_robustness_results.md`, [RL-031](tickets/RL-031.md).*

All three tiers are stable across 10 seeds (CV < 5% on every metric — VaR 95%, CVaR 95%, VaR 99%, CVaR 99%). Somewhat counter-intuitively, **MS-GARCH+EVT has the *lowest* CV**, not the highest. The intuition that a more complex model must be noisier is wrong here: the regime layer *averages* across regimes at each simulation step, which actually reduces Monte Carlo variance compared to a pure GARCH+t simulation that depends more heavily on individual shock realizations.

**What this does NOT say.** A low CV only means the Monte Carlo estimator has converged on some number. It says **nothing** about whether that number is the right number. This is the **stability ≠ accuracy** sub-finding, and it is important enough that the notebook says so inline: any risk-model evaluation that stops at "the estimate is stable" is incomplete — the backtest is the only thing that tells you whether the stable answer is also the true answer.

The original MS-GARCH (pre RL-033/RL-034) was the clearest illustration of this. It was *more stable* than GARCH+t and also *more wrong*. Stability is necessary but not sufficient.

## §5 — MS-GARCH internals (unified spec)

*Source: `model_comparison.ipynb` §5, [`docs/decisions/ms_garch_unified.md`](decisions/ms_garch_unified.md).*

After the RL-034 unified refit, the MS-GARCH+EVT model has three layers per regime:

1. **Global GARCH(1,1)** — one α, one β, fit on the full pooled return series. This gives a single persistence (α+β) shared across regimes.
2. **Regime-specific long-run variance** via variance targeting: each regime gets its own ω reparametrized from the per-regime sample variance, so $\sigma^2_{lr,k} = \omega_k / (1 - \alpha - \beta)$ differs per regime while α and β stay shared.
3. **Per-regime GPD tail** fit on regime-filtered exceedances above the per-regime threshold. On sparse regimes (< 50 exceedances) this falls back to a Normal tail, and the fallback is surfaced explicitly in the app ([RL-039](tickets/RL-039.md)).

**Empirical values from this run (shared α, β per asset):**

| Asset | α | β | persistence α+β | calm σ_lr | crisis σ_lr | ratio |
|---|---|---|---|---|---|---|
| BTC-USD | 0.0835 | 0.8814 | 0.9648 | 26.71% | 89.68% | 3.36× |
| SPY | 0.1115 | 0.8567 | 0.9682 | 13.00% | 29.87% | 2.30× |
| NVDA | 0.0608 | 0.9023 | 0.9631 | 34.96% | 67.24% | 1.92× |

Persistence is tight around 0.96–0.97 for all three assets (it should be: it's a single global fit). Crisis-regime σ_lr is 1.9–3.4× calm-regime σ_lr — the regime level differentiation is real but asset-dependent. The old "regime-sliced GARCH" failure mode (α collapsing to 0 in sparse regimes) is gone.

**Per-regime GPD tails on this run:**

| Asset | Calm ξ | Crisis ξ | Crisis exceedances |
|---|---|---|---|
| BTC-USD | −0.33 | −0.32 | 19 |
| SPY | −0.34 | **+0.81** | 11 |
| NVDA | −0.03 | **+0.62** | 28 |

SPY and NVDA crisis regimes produce **clearly positive GPD shape** (ξ > 0.5 — heavy Pareto tail), while their calm regimes are bounded. This is the mechanism by which the MS-GARCH+EVT tier captures the fat-tail structure of crisis periods without imposing it on calm periods. BTC is the exception: both regimes fit bounded tails (ξ ≈ −0.33), likely because the global GARCH already absorbs most of BTC's kurtosis into its innovation scaling, leaving the standardized residuals relatively tame. The small crisis exceedance counts (19/11/28) are the reason GPD fallback is surfaced in the UI — these fits exist but sit near the data-limited edge of what is identifiable.

**Why shared persistence.** The shared-(α, β) assumption is not a modeling compromise, it is the literature-supported best practice: Haas, Mittnik & Paolella (2004), Klaassen (2002), Marcucci (2005), and Ardia et al. (2018) all converge on intercept-only switching as the most stable specification for MS-GARCH, primarily because it avoids path-dependence (each regime's σ² at time $t$ would otherwise depend on every possible regime history up to $t$, which is computationally and statistically intractable). Full derivation in `docs/decisions/ms_garch_unified.md`.

## §6 — EVT/XGBoost audit (non-MC cross-check)

*Source: `validation_backtesting.ipynb` §2.*

Two independent, non-simulation VaR methods cross-checked against the Monte Carlo tiers at 1-day horizon:

- **EVT/GPD** — fits a Generalized Pareto Distribution to peaks over a high threshold. No distributional assumption on the body, only on the tail. Theoretically justified by the Pickands–Balkema–de Haan theorem.
- **XGBoost conditional quantile regression** — nonparametric conditional quantile estimator with engineered lag features. Audits parametric models without assuming anything about the shock distribution.

**Findings:**
- At 95% all three methods (Normal, EVT, XGBoost) roughly agree on thin-tailed assets (SPY) and diverge at 99% on heavy-tailed ones (BTC).
- EVT is systematically deeper at 99% where the GPD shape ξ is clearly positive — the tail is genuinely Pareto, not Gaussian. On BTC ξ > 0 with a tight confidence interval; on SPY ξ is close to zero (tail is effectively exponential).
- XGBoost's nonparametric estimate typically brackets EVT from above or below depending on the sample's most recent regime: it adapts to the current conditional distribution in a way the pooled GPD does not.
- Agreement is a sanity check on the parametric tiers. Divergence is where parametric assumptions stop working. Both outcomes are useful.

**ML role:** audit, not replacement. This is the guiding principle of the project and is enforced here by keeping XGBoost in a read-only "auditor" role rather than letting it drive simulation.

## §7 — Rolling backtest on BTC

*Source: `validation_backtesting.ipynb` §3, [RL-033](tickets/RL-033.md), [RL-034](tickets/RL-034.md).*

Rolling-window (step=5, train=252) 95% VaR backtest of the 3 tiers on BTC-USD over 315 test observations. Kupiec tests unconditional coverage; Christoffersen tests breach independence (clustering).

**Observed results on this run (n_sim = 2,000 per window, seed = 42):**

| Tier | n_breaches | Breach rate | Kupiec p | Kupiec | Christoffersen p | Christoffersen |
|---|---|---|---|---|---|---|
| Baseline | 8 / 315 | 2.5% | 0.027 | **FAIL** (under-breach) | 0.518 | PASS |
| GARCH+t | 19 / 315 | 6.0% | 0.415 | PASS | 0.439 | PASS |
| MS-GARCH+EVT (unified) | 13 / 315 | 4.1% | 0.464 | **PASS (best)** | 0.554 | PASS |
| EVT 1-day (non-MC) | — | 2.2% | — | **FAIL** (under-breach) | — | PASS |

**Calibration ranking:** MS-GARCH+EVT (1) → GARCH+t (2) → Baseline (3).
**Conservatism ranking:** Baseline (1, most conservative) → MS-GARCH+EVT (2) → GARCH+t (3).

**The headline.** MS-GARCH+EVT is the best-calibrated tier — its 4.1% observed breach rate is closest to the expected 5%, and it passes both Kupiec and Christoffersen cleanly. Baseline and EVT-1day both **fail Kupiec by under-breaching** — they are too conservative, not too optimistic. This is not a "safe failure": over-conservative capital allocation still wastes capital, and it also masks the fact that both tiers are catastrophically under-reactive *in crisis periods specifically* (see §12 below). GARCH+t passes Kupiec but sits slightly above the expected rate, consistent with its "adaptive in the wrong direction after a crisis" failure mode.

The MS-GARCH unified spec is the only tier that both *levels* the VaR correctly across regimes (because it has a regime concept at all) and *keeps within-regime vol dynamics* (because the global GARCH still captures clustering inside each regime). It is the only tier where statistical calibration is not accidental, and on this run it is also the only tier where the observed breach rate lands within a comfortable p-value of the expected 5%.

**Note on run-to-run variation.** The specific breach rates above depend on the seed, on n_simulations, and on the data snapshot (one extra BTC day relative to the old source notebook already shifts rates by ~0.1 percentage points). The ranking and pass/fail pattern are stable; the third decimal is not. Any claim that refers to exact numbers in this section is a claim about *this run*, not a universal property of the tier.

## §8 — Regime stability

*Source: `model_diagnosis.ipynb` §3.*

A 2-regime HMM fit on daily returns produces regime labels that persist rather than flipping daily. This is the first necessary condition for a regime-switching model to work at all: if the labels oscillated, there would be no persistent regime for per-regime GARCH or per-regime GPD to estimate.

**Transition matrices from this run:**

| Asset | P(calm → calm) | P(crisis → crisis) |
|---|---|---|
| BTC-USD | 0.825 | 0.620 |
| SPY | 0.994 | 0.974 |
| NVDA | 0.989 | 0.986 |

SPY and NVDA regimes are **very** persistent — roughly 99% self-transition, meaning expected regime dwell times of ~100 days or more. BTC is **markedly less persistent**: calm self-transition 0.825 implies a calm spell lasts about 6 days on average before the HMM flips to crisis, and crisis self-transition 0.620 implies crisis spells of ~2.5 days. BTC is essentially ping-ponging between regimes compared to the two equity names. That matters for §10 below: the MS-GARCH *detection lag* problem is structurally worse when regimes themselves are short-lived.

**Crisis-regime sample size on this run (from backtest observations):** 43 crisis days / 272 calm days = **13.6% crisis coverage on BTC**. The full pre-backtest sample has more crisis days (~50–70 depending on how the rolling window starts), but from the backtest engine's perspective the crisis sample is small — enough for the unified GARCH refit to work, marginal but workable for per-regime GPD identification (which is why RL-039 surfaces GPD fallback explicitly).

## §9 — Per-regime GARCH quality

*Source: `model_diagnosis.ipynb` §4, [RL-032](tickets/RL-032.md), [RL-034](tickets/RL-034.md).*

The original MS-GARCH (RL-027, pre-fix) had a structural bug: per-regime GARCH was fit on regime-filtered slices. On BTC this meant fitting GARCH on 44 crisis observations — not enough to identify the ARCH parameter at all. The result was α collapsing to exactly 0 in the crisis regime, turning the "regime-switching GARCH" into a *regime-switching constant-vol* with a label. The model became one of the cleanest examples of the stability ≠ accuracy problem: it was reproducible across seeds (constant-vol models are inherently stable) and it was wrong.

**RL-034's fix.** Fit GARCH **globally** on the pooled series. Reparametrize ω per regime via variance targeting ($\omega_k = \sigma^2_{lr,k} (1 - \alpha - \beta)$ with $\sigma^2_{lr,k}$ estimated from the regime's own sample variance). Share (α, β) across regimes. The specific numbers observed on this run are in §5 above: persistence 0.9631–0.9682 for every asset, crisis-to-calm σ_lr ratio 1.92–3.36×. No regime, on any asset, shows the degenerate α = 0 pattern. The old failure mode is gone.

The per-regime GPD tail is still fit on regime-filtered exceedances, and it can still fall back to Normal on sparse regimes — that is a data limitation, not a spec bug, and it is surfaced to the user rather than hidden.

## §10 — Breach concentration

*Source: `model_diagnosis.ipynb` §5.*

Where do MS-GARCH+EVT's breaches actually happen? Tagging each backtest day with its HMM regime and splitting the breach rate shows that **breaches concentrate sharply in the crisis regime**. On this run (BTC, using pre-fitted MS-GARCH params without per-window refit to isolate the regime effect):

| Regime | n obs | Breach rate |
|---|---|---|
| Calm | 272 | **2.2%** |
| Crisis | 43 | **34.9%** |
| Overall | 315 | 6.7% |

Calm-regime calibration is excellent (2.2% vs expected 5% — MS-GARCH+EVT is arguably *over*-conservative in calm, which is the correct direction to err). Crisis-regime calibration is dramatically worse: 34.9% is seven times the expected 5% rate. The rolling 63-day breach rate shows visible spikes at regime turns.

**Note on the overall rate.** The 6.7% here is higher than the 4.1% from §7's main backtest — not a contradiction. The §7 number is from a full per-window refit at n_sim = 2,000; the number above is from a fixed-params run used specifically to isolate *where* the breaches occur rather than *how many* per-window refits produce. The §7 number is the headline; this one is the diagnostic.

**This is a detection lag problem, not a wrong-model problem.** The HMM needs ~5–10 days to confirm a regime switch after volatility has already changed. During that lag window the model is still predicting VaR from the old regime's parameters, and the breaches happen then. The BTC case is worse than SPY/NVDA specifically because BTC's regimes are less persistent (see §8 transition matrix: 0.825 calm-self-transition, 0.620 crisis-self-transition) — the detection lag eats a larger fraction of each regime spell. The fix is structural (improve regime detection latency, or widen the transition smoothing) rather than parametric (no amount of tuning α or ξ will help). It is out of scope for RL-042.

## §11 — Cross-tier VaR on the same dates

*Source: `model_diagnosis.ipynb` §6.*

Plotting all three predicted-VaR series on the same time axis makes the tier differences visible in a way the summary tables do not:

- **Baseline** is nearly flat. It doesn't know there are regimes.
- **GARCH+t** oscillates aggressively. Shallow in quiet stretches, deep during vol spikes, then rapidly shallow again as the GARCH state decays back toward σ_lr.
- **MS-GARCH+EVT** is *layered*. It sits at one level during calm regimes and jumps discretely to a deeper level when the HMM confirms a crisis switch. Within each regime the global GARCH still produces within-regime clustering, but the level is anchored by the regime.

The jumpiness of GARCH+t is exactly what the failure-mode table in §12 calls "post-crisis amnesia." The model has no concept of regime, so after a vol spike its σ decays at the persistence half-life — roughly 23 days — and by the time the next vol cluster arrives σ is already back near the long-run level. It under-reacts at the moment that matters most.

## §12 — When each model fails (failure-mode taxonomy)

*Source: `model_diagnosis.ipynb` §7, `reports/rl035_when_each_model_fails.md`, [RL-035](tickets/RL-035.md).*

The headline finding of this project. Every risk-model-marketing slide about "each tier is more realistic than the previous" assumes that realism implies calibration. The backtest says otherwise. Each tier fails in a different, nameable way. **All numbers below are from the RL-035 per-tier-per-regime table on this run (BTC, n_sim = 500 to keep the rolling refit tractable, window = 252, step = 5, seed = 42).**

| Tier | Overall breach | Calm breach (n=272) | Crisis breach (n=43) |
|---|---|---|---|
| Baseline | 2.9% | **0.0%** | **20.9%** |
| GARCH+t | **7.9%** | 3.7% | **34.9%** |
| MS-GARCH+EVT | 5.1% | 1.1% | 30.2% |

### Baseline (Constant + Normal) — *lagged level adjustment*

"Always answering the previous regime's question."

**0.0% breach in calm** (wastes capital — the VaR is so deep that no calm day ever touches it) and **20.9% in crisis** (dangerously under-reactive — one in five crisis days is a breach). The 252-day σ is dominated by *all* historical variance equally, so calm-period VaR is too deep (rejecting profitable opportunities) and crisis-period VaR is too shallow (missing real losses). The two errors *partially* cancel in the overall number (2.9%) — which is exactly why the main backtest in §7 has Baseline **fail** Kupiec at p = 0.027 for under-breaching: the false safety of the calm regime drags the aggregate down below the expected 5%.

This is the uniform-over-conservatism failure mode. It looks safe and isn't.

### GARCH + Student-t — *post-crisis amnesia*

"σ recovers too fast after volatility subsides."

**7.9% breach overall** — highest of the three tiers in the per-regime split. Calibrates reasonably in calm regimes (3.7% — close to the expected 5%) but **breaches 34.9% in crisis**, because the GARCH state forgets the previous crisis level before the next shock arrives. The persistence half-life implied by the fitted α+β = 0.9648 for BTC is $\ln(0.5) / \ln(0.9648) \approx 19$ days — shorter than the empirical clustering structure of crypto returns, which cluster at longer timescales than individual GARCH sessions can remember.

This is the wrong-kind-of-adaptivity failure mode. The model *is* adaptive — it just decays toward the wrong level. Adaptive in the wrong direction produces 34.9% crisis breaches where Baseline's static-but-dominated-by-history approach produces 20.9%. **By this per-regime measure, GARCH+t is strictly worse than Baseline on BTC.** Both overall (7.9% vs 2.9%) and in the crisis regime specifically (34.9% vs 20.9%).

Note that this is a cleaner version of the claim than the overall backtest in §7 supports: at n_sim = 2,000 GARCH+t's aggregate passes Kupiec (6.0%, p = 0.415), but the per-regime decomposition still tells the real story — GARCH+t's aggregate pass is achieved by *averaging a good calm rate with a catastrophic crisis rate*, not by being well-calibrated in either regime.

### MS-GARCH + EVT (unified) — *detection lag + small crisis sample*

"HMM needs time to confirm a regime switch, and there are only 43 crisis backtest days to identify the crisis-regime GPD."

**5.1% overall** (closest to expected 5%). **1.1% in calm regime** (essentially solved — slightly over-conservative, the right direction). **30.2% in crisis regime** — still too high, but better than both GARCH+t's 34.9% and Baseline's 20.9% on the correct-direction metric (MS-GARCH+EVT *acknowledges* the crisis at all, which Baseline structurally cannot). The failure mode is now *structural* (HMM lag at regime turns + small sample for tail identification) rather than *parametric* (wrong level or wrong reactivity). This is the only tier where the remaining gap is about the detection system rather than the model itself.

### Why this matters

The pre-fix narrative was "each tier captures something the previous one missed." The post-fix narrative is **each tier fails in a different way**, and the right model depends on which failure mode you can tolerate. That framing is what the Streamlit app ([RL-041](tickets/RL-041.md)) tries to surface: depending on horizon and asset, it recommends the tier whose failure mode is least damaging for that configuration, rather than pretending one tier dominates.

## §13 — Overall synthesis

Distilling the 12 findings above into a few sentences the reader can walk away with:

1. **Horizon is the hidden variable.** Model ranking flips at different horizons for different assets. The 252-day comparison that the original deep-dive was built on is only one slice of the story. (§1, §3, §4)
2. **Stability is not accuracy.** The pre-fix MS-GARCH was the most stable model and the most wrong. Backtesting is the only accuracy test. This run: every tier has CV < 3.6% on VaR 95%, and the least-stable tier (NVDA GARCH+t at 3.6%) is not the worst-calibrated. (§4b, §9)
3. **The MS-GARCH bug was structural, not parametric.** Per-regime GARCH on regime-filtered slices collapsed α in sparse regimes. The unified refit (global GARCH + per-regime ω via variance targeting) produces sensible shared persistence 0.9631–0.9682 across all assets on this run and is supported by the MS-GARCH literature. (§5, §9, [RL-034](tickets/RL-034.md))
4. **MS-GARCH+EVT is the best-calibrated tier in this run's main backtest.** Observed 4.1% breach rate vs expected 5%, Kupiec p = 0.464, Christoffersen p = 0.554, both PASS. Baseline **fails** Kupiec at p = 0.027 by under-breaching (2.5%). GARCH+t passes but sits above expected (6.0%). The pre-fix narrative "MS-GARCH fails the backtest" is no longer true post RL-033/RL-034. (§7)
5. **Per-regime, GARCH+t is strictly worse than Baseline on BTC.** 7.9% overall vs 2.9%; 34.9% crisis breach vs 20.9%. The aggregate Kupiec test in §7 hides this because GARCH+t's good calm rate (3.7%) partially offsets its terrible crisis rate (34.9%). Looking only at the aggregate miscalibrates the comparison. (§12, [RL-035](tickets/RL-035.md))
6. **Each tier fails differently.** Baseline: lagged level adjustment (0.0% calm, 20.9% crisis). GARCH+t: post-crisis amnesia (3.7% calm, 34.9% crisis). MS-GARCH+EVT: detection lag at regime turns (1.1% calm, 30.2% crisis). The "best" tier depends on which failure mode you can afford. (§12)
7. **The app's recommendation is horizon-aware for a reason.** Long horizons on heavy-tail assets → MS-GARCH+EVT. Short horizons → sometimes GARCH+t, sometimes even Baseline. The non-monotonicity on SPY is real, not a bug. ([RL-038](tickets/RL-038.md), [RL-041](tickets/RL-041.md))
8. **BTC sits in the Student-t df < 3 danger zone.** Fitted df = 2.7 on this run. Per [`assumptions.md`](assumptions.md) §2.4 this means the GARCH+t tier's long-run variance is theoretically undefined for BTC. The MS-GARCH+EVT tier, which replaces the Student-t assumption with a per-regime GPD tail, is the architecturally correct answer for assets in this zone — independently of the backtest results above. (§2, `assumptions.md` §2.4)

## Footnote — why the comparison required RL-026 first

Before [RL-026](tickets/RL-026.md), all volatility models in this project shared the same shock distribution toggle (Normal or Student-t). That meant the models only differed in how they scaled σ, and the differences collapsed over 252 days because every model converged to its long-run vol with the same shocks. RL-026 made each model own its innovation distribution (Constant → Normal, GARCH → Student-t, MS-GARCH → EVT/GPD). Without that refactor the §2/§3/§4 comparisons in `model_comparison.ipynb` would not have been meaningful — each model would have been a rescaling of the same underlying process.
