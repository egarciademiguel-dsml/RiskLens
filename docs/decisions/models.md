# Model & Simulation Decisions

## Context

VaRify v0.1 used a single simulation model: Geometric Brownian Motion (GBM) with constant volatility and normal shocks. Starting with v0.2, the simulation engine was upgraded to support multiple model configurations while keeping the GBM framework as the base.

## Decisions

### 1. Shock Distribution (RL-013)

**Options considered:** Normal, Student-t, Skewed-t, Generalized Hyperbolic, EVT/GPD.

**Chosen:** Normal (default) + Student-t (user toggle).

- Student-t captures fat tails (excess kurtosis) observed in real financial returns.
- Degrees of freedom auto-fitted via MLE (`scipy.stats.t.fit`).
- Shocks standardized to unit variance: `Z = T / sqrt(df / (df - 2))`.
- df clamped >= 2.1 to ensure finite variance.
- More advanced distributions (skewed-t, GHD, EVT) deferred — Student-t is the best signal-to-complexity ratio for an MVP.

### 2. Volatility Model (RL-014)

**Options considered:** Constant sigma, GARCH(1,1), EGARCH, GJR-GARCH, Regime-switching.

**Chosen:** Constant (default) + GARCH(1,1) (user toggle).

- GARCH(1,1) captures volatility clustering: `sigma²(t) = omega + alpha * resid²(t-1) + beta * sigma²(t-1)`.
- Fitted via `arch` library (MLE, `disp="off"`).
- Day-by-day variance simulation (not vectorizable — each day depends on previous).
- Jensen's drift correction uses per-day sigma: `drift_t = mu - 0.5 * sigma_t²`.
- arch expects percentage-scale returns; params converted to decimal for simulation.
- Asymmetric GARCH variants (EGARCH, GJR) deferred — GARCH(1,1) is the industry baseline.

### 3. Model Combinations

The two dimensions (distribution x volatility) are independent and composable:

| Configuration | Shocks | Volatility | Use Case |
|---------------|--------|------------|----------|
| Constant + Normal | N(0,1) | Fixed sigma | Baseline / textbook GBM |
| Constant + Student-t | t(df) | Fixed sigma | Fat tails, constant vol |
| GARCH + Normal | N(0,1) | GARCH(1,1) | Vol clustering, thin tails |
| GARCH + Student-t | t(df) | GARCH(1,1) | Vol clustering + fat tails (recommended) |

### 4. Backward Compatibility

All defaults preserve v0.1 behavior: `distribution="normal"`, `volatility_model="constant"`. Existing tests with hardcoded exact values remain valid.

## Trade-offs

**Pros:**
- Realistic tail risk and volatility dynamics
- Composable model options (2x2 grid)
- Auto-fitted from data — no manual parameter tuning needed
- Backward compatible

**Cons:**
- GARCH day-by-day loop is slower than vectorized constant-vol (acceptable for current scale)
- No asymmetric volatility (leverage effect) — EGARCH would capture this
- Student-t is symmetric — doesn't model left-skew separately from right-skew
- Historical params assumed stationary into the future

## Future Considerations

- Skewed Student-t or GHD for asymmetric tail modeling
- EGARCH / GJR-GARCH for leverage effect
- Extreme Value Theory (GPD) for tail-only VaR/CVaR
- Regime-switching (HMM) for structural market state changes
- Copulas for multi-asset dependency modeling
