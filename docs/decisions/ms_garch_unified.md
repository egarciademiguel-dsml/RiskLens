# MS-GARCH: Unified GARCH with Regime-Parametrized Omega

## Context

The original MS-GARCH (RL-027) fit a separate GARCH(1,1) inside each HMM regime
by slicing the return series by regime label and calling `arch_model` on each
slice independently. The diagnosis in RL-032
(`reports/ms_garch_diagnosis_results.md`) showed two structural failures of
this approach:

1. **BTC crisis regime: alpha = 0.0000.** With only ~44 crisis days and the
   serial dependence destroyed by slicing, MLE collapsed the ARCH coefficient
   to zero. The "crisis GARCH" became constant volatility at a fixed level,
   unable to react to shock magnitude.
2. **NVDA both regimes: persistence < 0.12.** Both regime-local fits degenerated
   into near-constant-vol GARCHes with different levels but no clustering.

The result: 11.7% breach rate on BTC (Kupiec FAIL), with 34.1% breaches
concentrated inside crisis windows where the model was supposed to be at its
most reactive.

The first fix (RL-033, `reports/ms_garch_fix_results.md`) reordered the
simulation so the regime transition happens *before* return generation. This
brought the breach rate to 7.0% (Kupiec PASS) but did **not** address the
underlying alpha = 0 issue — it only let calm-day forecasts mix in some crisis
paths via the transition matrix.

## Decision

Fit GARCH(1,1) **once on the full return series** (all regimes pooled). Use the
regime label only to:

1. **Reparametrize the unconditional variance level via omega.**
   For each regime k, compute `sigma²_k = Var(r_t | label_t = k)` from the
   regime-filtered returns, then set
   `omega_k = sigma²_k * (1 - alpha - beta)`
   where `alpha`, `beta` are the global estimates. This is variance targeting
   (Engle & Mezrich, 1996) applied per regime.

2. **Discretize the innovation tail via per-regime GPD.**
   Compute global standardized residuals `z_t = (r_t - mu) / sigma_t` from the
   global GARCH fit, partition them by regime label, and fit a Generalized
   Pareto Distribution to each partition's left tail. Crisis-regime z_t carry
   the heavy-tailed shape; calm-regime z_t carry the bulk shape.

The simulation engine (`generate_log_returns`) is unchanged: it already reads
`(omega_k, alpha_k, beta_k)` from `regime_garch[k]` and per-regime GPD from
`regime_gpd[k]`. Only `fit_ms_garch` was modified.

## The Shared-Persistence Assumption

The unified approach forces `alpha + beta` (volatility persistence) to be
**identical across regimes**. Only the unconditional level differs. This
deserves explicit justification because regime-switching GARCH literature
typically allows persistence to vary by state.

### Theoretical Justification

- **Volatility clustering is a property of the return-generating process,
  not of regime labels.** In a Markov-switching framework, the regime variable
  governs the level (or distribution parameters), while the conditional
  variance dynamics encode short-memory dependence in squared innovations.
  These are conceptually orthogonal: regimes capture structural shifts;
  GARCH captures local clustering within whichever structural state holds.

- **Identification under sparsity.** Hamilton & Susmel (1994) and Cai (1994)
  introduced SWARCH precisely because GARCH parameters were not identifiable
  inside short regime episodes. They circumvent the issue by switching ARCH
  scale factors, not the ARCH coefficients themselves. Our per-regime omega
  is the discrete-time analogue: a regime-dependent scale on a globally
  estimated dynamic.

- **Variance targeting (Engle & Mezrich, 1996).** Reparametrizing
  `omega = sigma²_uncond * (1 - alpha - beta)` is a standard, theoretically
  consistent way to impose an exogenous level on a GARCH process while
  preserving its dynamic structure. Applying it per regime simply swaps the
  exogenous level when the regime label flips.

- **Avoiding the path-dependence problem of MS-GARCH.** Gray (1996),
  Klaassen (2002), and Haas, Mittnik & Paolella (2004) document that fully
  regime-switching GARCH suffers from path-dependence: the conditional
  variance becomes a function of the entire latent regime history, making
  estimation either intractable or reliant on approximations. Sharing
  `(alpha, beta)` across regimes eliminates path-dependence cleanly — the
  recursion `sigma²_t = omega_{s_t} + alpha * eps²_{t-1} + beta * sigma²_{t-1}`
  is well-defined regardless of the regime path.

### Empirical Justification from the Literature

- **Haas, Mittnik & Paolella (2004), "A New Approach to Markov-Switching GARCH",
  *Journal of Financial Econometrics* 2(4):493–530.** Their MSGARCH formulation
  with regime-specific omega and shared (alpha, beta) is shown to dominate
  fully-switching variants on S&P 500 and DAX in out-of-sample density
  forecasts. The "alpha varies by state" specifications add parameters without
  improving log-likelihood once omega is allowed to switch.

- **Klaassen (2002), "Improving GARCH Volatility Forecasts with
  Regime-Switching GARCH", *Empirical Economics* 27(2):363–394.** Reports that
  the persistence-shared specification produces more stable forecasts and
  avoids the explosive behavior that plagues regime-specific persistence in
  small crisis subsamples.

- **Marcucci (2005), "Forecasting Stock Market Volatility with
  Regime-Switching GARCH Models", *Studies in Nonlinear Dynamics &
  Econometrics* 9(4).** Finds that for S&P 500 daily returns, regime-shared
  ARCH/GARCH coefficients with switching intercepts beat fully-switching
  models on RMSE and QLIKE at 1- and 5-day horizons.

- **Ardia, Bluteau, Boudt & Catania (2018),
  *MSGARCH: Markov-Switching GARCH Models in R*.** Their package documentation
  notes that the "intercept-only switching" specification is the most
  numerically stable in practice and is the recommended starting point when
  any regime contains fewer than ~200 observations — exactly our BTC crisis
  case (44 observations).

### Empirical Justification from Our Data

- BTC global GARCH: persistence ≈ 0.99 (typical for daily crypto). This
  long-memory dynamic is **observable in the full series**, but disappears
  when we isolate the 44 crisis days. The shared-persistence assumption
  recovers it.
- The two regimes still differ by ~3x in unconditional volatility
  (calm ≈ 27% annualized, crisis ≈ 88% on BTC), captured by omega.
- Per-regime tail shape still differs: GPD fitted on crisis-regime
  standardized residuals will pick up heavier xi than the calm subset.

### What We Lose

We give up the (theoretical) ability to model regimes where the *speed* of
volatility mean-reversion differs. In practice, the literature finds this
is a small effect that is dominated by sample noise in any regime smaller
than a few hundred observations, and our data (BTC: 44 crisis days)
cannot identify it anyway.

## Implementation Footprint

- `src/analytics/ms_garch.py::fit_ms_garch` — refactored to fit one global
  GARCH and derive `omega_k`, plus per-regime GPD on global residuals
  filtered by regime.
- `src/analytics/ms_garch.py::generate_log_returns` — **unchanged**. The
  simulation contract is identical; only the per-regime parameter values
  differ.
- `_compute_standardized_residuals` and `_constant_garch_fallback` — left
  in place but no longer called from the main path. Retained for any
  external callers and as documentation of the original approach.

## Validation

See `reports/ms_garch_unified_results.md` for the BTC backtest comparison
across the three states:

| State                          | Approach                                 |
|--------------------------------|------------------------------------------|
| Original (RL-027)              | Per-regime GARCH on filtered slices      |
| Transition-fix (RL-033)        | Same fit, regime transition reordered    |
| Unified (RL-033b, this doc)    | Global GARCH, per-regime omega + GPD     |
