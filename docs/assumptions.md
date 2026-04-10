# Modeling Assumptions

Every quantitative model rests on assumptions, and every finding in [`conclusions.md`](conclusions.md) is conditional on the assumptions in this document. This file consolidates them in one place so a reader can check "does this model apply to my situation?" without having to read the source code.

> **Note on overlap with `docs/decisions/`.** Some assumptions listed here — particularly around MS-GARCH specification — are stated in more depth in `docs/decisions/ms_garch_unified.md`. The overlap is intentional for now: this file is the flat summary, `decisions/` holds the per-decision context. Reconciling the two is explicitly deferred and is not part of [RL-042](tickets/RL-042.md). If you want the derivation, go to `decisions/`. If you want the list, stay here.

---

## 1. Data assumptions

### 1.1 Log-returns
All volatility and simulation modeling operates on **daily log-returns** $r_t = \ln(P_t / P_{t-1})$, not simple returns. This matters because:
- Log-returns are approximately symmetric for small moves (unlike simple returns, which are bounded below at −100% but unbounded above).
- Log-returns are additive over time: $r_{t,T} = \sum_{s=t}^{T} r_s$, which is what makes GARCH's time-scaling clean.
- The normal approximation to the central-limit behavior of log-returns is better than to simple returns.

The app converts back to simple returns for user-facing VaR/CVaR numbers because that is what the user expects to see.

### 1.2 Daily frequency
Every model assumes the data is sampled at **trading-day frequency** (not calendar days, not intraday). 252 trading days ≈ one calendar year is used throughout. This matters for:
- The annualization factor $\sqrt{252}$ in volatility reporting.
- The fit-window default of 252 days ≈ one year of history.
- The Basel monthly analogue of 21 days used for the pure-vol-effect horizon in §3b of `model_comparison.ipynb`.

Using the app on weekly or intraday data without changing these constants would produce silently wrong annualizations.

### 1.3 Fit window
The default fit window is **252 trading days**. This is a compromise:
- Long enough to fit GARCH(1,1) stably (roughly 100 observations is the minimum for α and β to separate from zero; 252 gives comfortable identification).
- Long enough to include at least one moderate vol episode in most samples.
- Short enough that the model does not average across structurally different regimes — e.g., pre-2020 vs post-2020 crypto volatility are different enough that a 10-year window would blend them into something representing neither.

Changing the fit window is a config decision that propagates through the backtest rolling window; it is not a free parameter.

### 1.4 Survivorship bias (acknowledged, not corrected)
The app uses Yahoo Finance data via `yfinance`. Delisted tickers are not in the sample. For individual-stock analysis this biases historical risk estimates *downward* (survivors look safer than the population of all stocks did ex-ante). For the three default assets (BTC-USD, SPY, NVDA) this is not a concern — none are at risk of delisting — but it is a real limitation for arbitrary user-chosen tickers and should be kept in mind when interpreting historical risk metrics on small-cap or emerging-market names.

No correction is applied. The bias is acknowledged and left to the user to account for.

---

## 2. GARCH(1,1) assumptions

### 2.1 Stationarity: α + β < 1
The GARCH(1,1) variance equation $\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$ has a finite unconditional variance $\sigma_{lr}^2 = \omega / (1 - \alpha - \beta)$ **only when α + β < 1**. Every formula in the project that uses $\sigma_{lr}$ (long-run vol reporting, variance-targeted ω reparametrization in MS-GARCH, mean-reversion half-life) assumes this holds. In practice the `arch` library's MLE almost always converges to a stationary solution; on the rare occasion it does not, the behavior is undefined and the app should treat it as a fit failure.

### 2.2 Constant unconditional variance
Under α + β < 1 the model assumes the *unconditional* variance $\sigma_{lr}^2$ is constant over the fit window. If the true DGP has a structural break (pre-/post-COVID, pre-/post-regulatory-change), GARCH will average across it and produce a long-run vol that matches neither sub-regime. **The MS-GARCH tier is the project's explicit answer to this assumption violation**: it replaces one constant $\sigma_{lr}$ with per-regime $\sigma_{lr,k}$.

### 2.3 No leverage effect
GARCH(1,1) is **symmetric** in the sense that positive and negative shocks of the same magnitude have the same effect on future variance. The well-known *leverage effect* — bad news raises future vol more than good news of the same magnitude — is not modeled. Asymmetric alternatives (GJR-GARCH, EGARCH, APARCH) exist and would capture this, but they add parameters and are not used in this project. The resulting bias is that post-crash vol is slightly underestimated.

### 2.4 Student-t innovations for the GARCH+t tier
The GARCH+t tier uses **Student-t innovations with a fitted degrees-of-freedom parameter**. The fit is done via `scipy.stats.t.fit` on the standardized residuals from the GARCH pre-fit. No prior or regularization is applied to df.

**df ≥ 3 as a stability indicator (not a cap — see [RL-040](tickets/RL-040.md) reframe).** When the fitted df is below 3, the tail is heavier than a finite-variance Student-t can express: df < 2 gives infinite variance, and 2 ≤ df < 3 gives finite variance but infinite kurtosis. In either case GARCH's long-run variance $\sigma_{lr}^2$ is theoretically undefined and the GARCH+t tier's results should be treated as unreliable for that asset.

The project's original RL-040 proposed hard-capping df at 3.0 to avoid pathological tail estimates. **Reframed under RL-042**: do NOT cap. Capping hides information — the user's data is telling you that the tail is genuinely heavier than the GARCH+t model can represent, and that fact should surface as a *warning* rather than being silently fixed. A "model instability" indicator at df < 3 is a better design than a cap because:

1. It preserves the raw fit for diagnostics.
2. It is consistent with the project's transparency philosophy ([RL-039](tickets/RL-039.md) surfaces GPD fallbacks rather than hiding them).
3. It tells the user to use MS-GARCH+EVT instead, where the per-regime GPD can represent heavier tails theoretically.

Implementation of the warning is deferred; the reframe is noted in the ticket registry under RL-040.

---

## 3. MS-GARCH unified-spec assumptions

See [`decisions/ms_garch_unified.md`](decisions/ms_garch_unified.md) for the full derivation. Flat list:

### 3.1 Two regimes, fixed
The HMM is fit with **n_regimes = 2** (calm and crisis). 3-regime configurations exist in the codebase (RL-016, RL-017) but are not used in the 3-tier simulation pipeline. Two regimes are enough to capture the dominant bimodality of crypto and equity-index returns without running into the sparse-regime identification problems a 3-regime HMM would face on 252 days.

### 3.2 Gaussian HMM emissions
The HMM uses **Gaussian emissions on daily returns** (via `hmmlearn.GaussianHMM`). This is a mild assumption because the HMM is only being used for *regime labeling*, not for return prediction — the tail modeling is delegated to per-regime GPD, and the conditional volatility is delegated to the global GARCH. The Gaussian emission is essentially doing threshold detection on the level of daily returns, and that works even when the true conditional distribution is not Gaussian.

### 3.3 Shared GARCH persistence across regimes
The unified spec fits **one global GARCH(1,1)** on the full pooled return series, so α and β are shared across regimes. Per-regime variation enters only through ω (reparametrized via variance targeting). This is an *assumption*, and a material one: the ARCH/GARCH reactivity is assumed to be the same in calm and crisis regimes, even though one could argue crisis regimes should have higher α (shock impact) and lower β (persistence decay).

**Why shared persistence.** The MS-GARCH literature (Haas/Mittnik/Paolella 2004, Klaassen 2002, Marcucci 2005, Ardia et al. 2018) converges on intercept-only switching as the most stable specification, primarily because allowing per-regime α and β creates *path-dependence*: each regime's σ² at time $t$ would depend on every possible regime history up to $t$, which is computationally and statistically intractable. Shared persistence avoids this at the cost of the modeling concession above.

### 3.4 Per-regime GPD tails
Each regime gets its own Generalized Pareto Distribution fit on the regime-filtered **standardized residuals** from the global GARCH fit. The GPD shape parameter ξ is estimated per regime; the threshold is per regime.

**Fallback rule.** When a regime has fewer than ~50 threshold exceedances (typically the crisis regime on short samples), the GPD fit is unreliable and the code falls back to a Normal tail for that regime. The fallback is **surfaced explicitly in the Streamlit app** ([RL-039](tickets/RL-039.md)), not hidden. The user sees which regimes are on real GPD and which are on Normal fallback.

### 3.5 Variance-targeted ω per regime
Under the unified spec, $\omega_k = \sigma_{lr,k}^2 (1 - \alpha - \beta)$ where $\sigma_{lr,k}^2$ is estimated from the regime's own sample variance. This is [Engle & Mezrich (1996)](https://www.researchgate.net/publication/243758022_GARCH_for_Groups) variance targeting applied per regime. The assumption is that the **empirical regime-conditional variance is a good estimator of the regime's true unconditional variance** — reasonable for the calm regime (many observations), weaker for the crisis regime (sparse), and is one of the known sources of residual miscalibration.

---

## 4. Extreme Value Theory (GPD/POT) assumptions

### 4.1 Threshold selection
The threshold for Peaks Over Threshold (POT) is set at the **10th percentile of the loss distribution** (i.e. the 90th percentile of positive losses). This is a classic rule-of-thumb threshold that balances:
- Bias (lower threshold → more data, but the GPD approximation starts to fail because you are no longer in the tail)
- Variance (higher threshold → fewer exceedances, tighter to the true tail, but less precise shape/scale estimates)

No formal threshold selection procedure (mean residual life plot, parameter stability plot) is implemented. 10% is a defensible default, not an optimum.

### 4.2 IID exceedances
The GPD is fit under the assumption that **exceedances are independent and identically distributed** conditional on being above the threshold. In financial returns this is violated during vol clusters (exceedances cluster in time rather than being Poisson-distributed). Declustering methods exist (runs declustering, intervals estimator) but are not applied. The practical effect is that the standard errors on ξ and σ are understated — the point estimate of tail VaR is approximately unbiased, but the confidence interval around it is tighter than it should be.

### 4.3 Tail index stability
The Pickands–Balkema–de Haan theorem says that for sufficiently high threshold, the exceedance distribution converges to a GPD regardless of the underlying DGP. This is an *asymptotic* result. In finite samples the approximation quality depends on the threshold being "high enough," and "high enough" is asset-dependent. For BTC (heavy tails, many exceedances) the 10% threshold is clearly in the asymptotic regime. For SPY (thin tails, few exceedances) the 10% threshold may not be.

---

## 5. Monte Carlo assumptions

### 5.1 Default simulation count
**`N_SIMS = 10,000`** paths is the default across the deep-dive notebooks. The backtest uses a lower `n_simulations = 2,000` per window to keep the rolling-window refit tractable.

### 5.2 Seed sensitivity and multi-seed stability
Single-seed estimates can be misleading for heavy-tailed distributions. [RL-031](tickets/RL-031.md) added a reusable `run_multi_seed` / `robustness_summary` module and the deep-dive notebook runs every tier across 10 seeds to report a coefficient of variation per metric. Seed sensitivity is **stability, not accuracy**. A low CV means the Monte Carlo estimator has converged; it does not mean the converged number is right. This is a deliberately repeated caveat because it is the most common misreading of robustness results.

### 5.3 Antithetic variates / variance reduction
**Not used.** The project does not apply antithetic variates, control variates, or importance sampling to reduce Monte Carlo variance. At 10,000 paths the residual MC noise is small enough that variance reduction is not critical for the project's use cases, and adding it would complicate the model-owned innovation architecture (RL-026) that makes the tiers comparable in the first place.

### 5.4 Model-owned innovations
**Each volatility model owns its own innovation distribution.** Constant → Normal, GARCH → Student-t, MS-GARCH → EVT/GPD. This is [RL-026](tickets/RL-026.md)'s refactor and is a structural design decision, not a tuning parameter. Before RL-026, all models shared one innovation toggle, which made the pure-vol and pure-tail comparisons in `model_comparison.ipynb` impossible — the models would collapse to scaled versions of the same underlying process. Model-owned innovations are what make the 3-tier comparison meaningful at all.

---

## 6. Backtesting assumptions

### 6.1 Rolling-window refit
Every backtest step **refits all model parameters from scratch** on the most recent 252-day window. No parameter smoothing, no exponential re-weighting of old windows, no warm-starting the optimizer from the previous window's fit. This is slow but reproducible and makes the backtest genuinely out-of-sample.

### 6.2 Step size
Default step is **5 trading days** (roughly weekly). The trade-off is speed vs granularity: step=1 would produce the most granular breach series at the cost of 5× runtime; step=5 is fast enough for interactive use while still producing enough observations for Kupiec and Christoffersen to have power.

### 6.3 Kupiec unconditional coverage
The Kupiec test is the primary calibration test: it asks whether the empirical breach rate matches the expected breach rate, using a likelihood ratio statistic. It is an *unconditional* test — it does not care whether breaches are clustered in time or spread out uniformly, only whether the total count matches expectation.

### 6.4 Christoffersen independence
The Christoffersen test complements Kupiec by testing **independence of consecutive breaches** (no clustering). A model can pass Kupiec by having the right breach count and still fail Christoffersen by having all the breaches bunched into a few crisis windows. A well-calibrated model passes both. Only MS-GARCH+EVT (unified, post RL-034) passes both cleanly on BTC.

### 6.5 Breach indicator
A breach is defined as **actual return < predicted VaR** on a given day, where "predicted VaR" is a negative number (e.g. −3%) and "actual return" is also signed. The indicator is day-level, not path-level.

---

## 7. HMM assumptions

### 7.1 Two regimes (fixed)
See §3.1 above. Two regimes is a modeling choice, not an output of model selection — no BIC/AIC search is run.

### 7.2 Gaussian emissions
See §3.2 above.

### 7.3 Viterbi decoding for regime labels
Regime labels used downstream (for per-regime GARCH refit, per-regime GPD fit, and the breach concentration analysis in `model_diagnosis.ipynb` §5) come from the **Viterbi most-likely-path** algorithm, not from the forward-backward posteriors. This means the labels are hard assignments, not soft probabilities. The model_diagnosis.ipynb breach-concentration-by-regime splits are therefore "breach rate conditional on the HMM's most-likely regime assignment" rather than "expected breach rate under the posterior regime distribution."

### 7.4 Stationary transition matrix
The HMM transition matrix is assumed **constant over the fit window**. No time-varying transition probabilities. If the underlying regime-switching behavior is itself evolving (regimes become more persistent, or crisis regimes become more frequent), the HMM will average across that and produce a transition matrix that matches neither the early nor late behavior of the window.

---

## Things that are NOT assumed

Sometimes it is clearer to list what is not being assumed:

- **No Gaussianity of returns.** The GARCH+t tier explicitly uses Student-t innovations; the MS-GARCH+EVT tier explicitly uses GPD tails.
- **No independence of returns.** All GARCH models assume conditional heteroskedasticity; this is the opposite of assuming iid returns.
- **No symmetry between tiers.** The 3 tiers are not meant to be "equivalent under different distributional assumptions" — they deliberately differ in how they model both the volatility process and the shock distribution.
- **No direction of return (drift).** All tiers use the historical mean return as drift. No predictive drift model, no ML return forecast. This is a deliberate non-goal ([`project_overview.md`](project_overview.md)): the project does not attempt price prediction or trading signal generation.
- **No portfolio effects.** All analysis is single-asset. Correlation, copulas, and portfolio aggregation are out of scope.
