# MS-GARCH Fix — Regime Transition Before Return Generation

## The Bug

In the original `generate_log_returns`, the regime transition happened *after* generating returns at each timestep. For 1-day simulations (n_days=1), all paths used the current regime (calm) — no mixing with crisis regime occurred. The daily VaR was purely calm-regime VaR.

## The Fix

Move regime transition to *before* return generation. Now 1-day paths sample tomorrow's regime first (using transition matrix probabilities), then generate returns under that regime. ~17.5% of BTC paths switch to crisis before generating shocks.

**Change:** 2 lines moved in `src/analytics/ms_garch.py:generate_log_returns`.

## Backtest Results (BTC-USD, 95% confidence, window=252, step=5)

| Model                      | Breach Rate | Kupiec p | Kupiec | Christoffersen p | Christoffersen |
|---------------------------|------------|---------|--------|-----------------|----------------|
| MS-GARCH (original)        | 11.7%      | 0.000   | FAIL   | 0.733           | PASS           |
| MS-GARCH (transition-first)| 7.0%       | 0.123   | PASS   | 0.706           | PASS           |

## Impact

- Breach rate: 11.7% → 7.0% (target 5.0%)
- Kupiec: FAIL → PASS
- Christoffersen: unchanged (PASS)
- Mean predicted VaR: -0.0318 → -0.0323 (slightly deeper)

The remaining gap from 5% is expected — the crisis regime still has alpha=0 (no ARCH reactivity), so crisis VaR is fixed at one level rather than adapting to shock magnitude. But the model is now statistically calibrated.
