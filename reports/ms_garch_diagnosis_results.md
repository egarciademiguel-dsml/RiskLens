# MS-GARCH Diagnosis — Why Does It Fail the Backtest?

MS-GARCH+EVT produces the deepest 252-day VaR (-70.79%) and the most stable estimates across seeds (CV 0.5%), yet fails the daily backtest with 11.7% breach rate. This report diagnoses why.

---

## 1. Per-Regime GARCH Quality

| Asset   | Regime | alpha  | beta   | persistence | long_run_vol | GPD fitted | Fallback? |
|---------|--------|--------|--------|-------------|-------------|-----------|-----------|
| BTC-USD | Calm   | 0.0117 | 0.9774 | 0.9891      | 26.77%      | Yes       | No        |
| BTC-USD | Crisis | 0.0000 | 0.9519 | 0.9519      | 87.98%      | Yes       | No        |
| SPY     | Calm   | 0.0675 | 0.8521 | 0.9196      | 13.08%      | Yes       | No        |
| SPY     | Crisis | 0.1284 | 0.7347 | 0.8631      | 30.64%      | Yes       | No        |
| NVDA    | Calm   | 0.0474 | 0.0695 | 0.1169      | 34.93%      | Yes       | **YES**   |
| NVDA    | Crisis | 0.0866 | 0.0000 | 0.0866      | 66.96%      | Yes       | **YES**   |

**Findings:**
- BTC crisis regime: alpha=0.0000 — no ARCH effect. Volatility doesn't react to shocks. It's constant vol at 87.98%, labeled as GARCH.
- NVDA both regimes: persistence < 0.12 — GARCH is barely doing anything. Both regimes are effectively constant volatility with different levels.
- SPY is the only asset where both regimes have genuine GARCH dynamics (persistence > 0.86).

---

## 2. Breach Concentration by Regime (BTC-USD)

| Regime | Observations | Breach Rate |
|--------|-------------|-------------|
| Calm   | 270         | **3.3%**    |
| Crisis | 44          | **34.1%**   |

**Overall: 24 breaches / 314 obs = 7.6%**

**This is the root cause.** MS-GARCH performs well in the calm regime (3.3% < 5% expected — slightly conservative). But during crisis periods, it breaches 34.1% of the time — the model's VaR is catastrophically too optimistic when it matters most.

---

## 3. Mean Predicted VaR (1-day, BTC-USD)

| Model        | Mean Predicted VaR |
|-------------|-------------------|
| Baseline     | -0.0514           |
| GARCH+t      | -0.0323           |
| MS-GARCH+EVT | -0.0318           |

**MS-GARCH+EVT produces the shallowest daily VaR** — even shallower than GARCH+t. Despite being the "most complex" model, it predicts less daily risk than the others. This is the opposite of what happens at 252d, where MS-GARCH produces the deepest VaR.

---

## 4. Diagnosis

The contradiction (deepest annual VaR, shallowest daily VaR) has a clear mechanism:

1. **The calm regime dominates** (~86% of days for BTC). When in calm, the model uses calm-regime GARCH with long_run_vol=26.77% — lower than the unconditional vol. This produces shallow daily VaR.

2. **During crisis, the model uses crisis-regime params** (87.98% vol), but the crisis regime has alpha=0.0 — vol doesn't react to shocks. It sits at a fixed high level that was calibrated to historical crisis periods, not the current shock magnitude. The VaR jumps up but not enough to match actual crisis losses.

3. **At 252d, regime transitions accumulate.** The simulation samples regime paths over a year, so the probability of visiting crisis at least once is high. This deepens the annual VaR. But on any single day, the model is usually in calm mode with shallow VaR.

4. **The time-scaling mismatch:** MS-GARCH is designed for regime dynamics over time, not for single-day predictions. Its daily VaR is the VaR of "what happens tomorrow given the current regime" — and the current regime is almost always calm.

---

## 5. Implications for RL-033 (Tune or Refute)

The failure is **structural, not parametric**. Tuning hyperparameters won't fix it because:
- The calm regime will always dominate daily predictions
- Alpha=0 in crisis is a data issue (too few crisis observations for GARCH to converge), not a tuning issue
- The HMM transition matrix correctly identifies regimes — it's the per-regime VaR that's miscalibrated

**Recommendation: Refute MS-GARCH for daily VaR. Reposition as scenario/stress-testing tool for longer horizons where regime dynamics add genuine value.**
