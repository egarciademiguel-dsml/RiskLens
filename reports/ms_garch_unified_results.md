# MS-GARCH Unified — Global GARCH + Per-Regime Omega + Per-Regime GPD

## The Change

Refactored `fit_ms_garch` to fit **one** GARCH(1,1) on the full return series
(all regimes pooled). The HMM regime label now controls only:
1. The unconditional level via variance targeting:
   `omega_k = sigma²_k * (1 - alpha - beta)` with global `(alpha, beta)`.
2. The tail shape via per-regime GPD fitted on global standardized residuals
   filtered by regime.

The simulation engine `generate_log_returns` was not modified — it already
reads `(omega, alpha, beta)` from `regime_garch[k]`.

Theoretical and empirical justification for the shared-persistence assumption
is in `docs/decisions/ms_garch_unified.md`.

## Global GARCH Fit (BTC-USD, full 5-year window)

| Parameter      | Value     |
|----------------|-----------|
| alpha          | 0.0872 |
| beta           | 0.8774 |
| persistence    | 0.9646 |

Compare to the original per-regime fits from `ms_garch_diagnosis_results.md`:
- BTC Calm: alpha=0.0117, beta=0.9774, persistence=0.9891
- BTC Crisis: alpha=0.0000, beta=0.9519, persistence=0.9519 ← the bug

The unified fit recovers a single sensible (alpha, beta) on the pooled series.

## Per-Regime Parametrization

| Regime  | omega         | long_run_vol (annualized) | GPD xi      |
|---------|---------------|---------------------------|-------------|
| Calm    | 1.0320e-05 | 0.2711                | -0.2419 |
| Crisis  | 1.1458e-04 | 0.9032                | -0.5209 |

## Backtest Results (BTC-USD, 95% confidence, window=252, step=5, n_sim=500)

| State                          | Breach Rate | Kupiec p | Kupiec | Christoffersen p | Christoffersen |
|--------------------------------|------------|---------|--------|-----------------|----------------|
| MS-GARCH original (RL-027)     | 11.7%      | 0.000   | FAIL   | 0.733           | PASS           |
| MS-GARCH transition-fix (RL-033)| 7.0%      | 0.123   | PASS   | 0.706           | PASS           |
| MS-GARCH unified (RL-033b)     | 5.4%     | 0.740   | PASS   | 0.934           | PASS           |

n_obs=314, n_breaches=17, mean predicted VaR = -0.0437

## Breach Concentration by Regime

| Regime  | Observations | Breach Rate |
|---------|-------------|-------------|
| Calm    | 261          | 0.8%       |
| Crisis  | 53          | 28.3%       |

Compare to the original (from diagnosis report):
- Calm: 270 obs, 3.3%
- Crisis: 44 obs, 34.1% ← the structural failure

## Interpretation

The unified GARCH approach addresses the structural alpha=0 problem in the
crisis regime. Volatility now reacts to shock magnitude in *both* regimes
using the same global ARCH coefficient, while the per-regime omega keeps the
unconditional levels distinct. The crisis regime can finally cluster.

Whether this translates into an improved breach rate vs. the transition-fix
alone depends on whether the crisis-window misses came from *level* or
*reactivity* — the table above is the answer.
