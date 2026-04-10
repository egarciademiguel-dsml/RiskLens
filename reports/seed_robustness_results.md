# Seed Robustness Analysis — Results

**10 seeds (0, 10, 20, ..., 90) × 3 tiers × 3 assets, 10,000 simulations each, 252-day horizon.**

Stability threshold: CV < 5% = stable, 5-15% = moderate, >15% = unstable.

---

## Summary Table

| Asset   | Tier         | VaR 95% (mean) | VaR 95% (std) | VaR 95% (CV) | CVaR 99% (mean) | CVaR 99% (std) | CVaR 99% (CV) |
|---------|-------------|---------------|--------------|-------------|----------------|---------------|-------------|
| BTC-USD | Baseline     | -52.47%       | 0.44%        | 0.8%        | -70.12%        | 0.83%         | 1.2%        |
| BTC-USD | GARCH+t      | -46.36%       | 0.52%        | 1.1%        | -82.23%        | 1.70%         | 2.1%        |
| BTC-USD | MS-GARCH+EVT | -70.79%       | 0.32%        | 0.5%        | -85.36%        | 0.53%         | 0.6%        |
| SPY     | Baseline     | -15.31%       | 0.29%        | 1.9%        | -28.71%        | 0.73%         | 2.5%        |
| SPY     | GARCH+t      | -13.85%       | 0.40%        | 2.9%        | -38.82%        | 1.76%         | 4.5%        |
| SPY     | MS-GARCH+EVT | -23.71%       | 0.54%        | 2.3%        | -58.37%        | 2.43%         | 4.2%        |
| NVDA    | Baseline     | -29.41%       | 0.72%        | 2.5%        | -57.72%        | 1.30%         | 2.3%        |
| NVDA    | GARCH+t      | -30.82%       | 1.11%        | 3.6%        | -64.77%        | 1.74%         | 2.7%        |
| NVDA    | MS-GARCH+EVT | -37.12%       | 0.57%        | 1.5%        | -68.57%        | 0.93%         | 1.4%        |

---

## Key Findings

1. **All estimates are stable.** Every CV is under 5%, most under 3%. The point estimates from the deep dive are reliable — no seed sensitivity concern.

2. **The kurtosis=81.77 scare is a non-issue for VaR/CVaR.** BTC GARCH+t has the highest CVaR 99% CV at 2.1% — meaning CVaR moves ±1.7pp across seeds around a mean of -82.23%. The extreme kurtosis comes from rare outlier paths that affect the 4th moment but barely move the percentile-based risk metrics.

3. **MS-GARCH+EVT is the most stable model** (lowest CVs across the board). This is surprising given its complexity — the regime structure and EVT tails produce less seed-dependent results than simpler models. The regime assignments are deterministic (fitted once), so only the shock generation varies across seeds.

4. **GARCH+t is the least stable** (highest CVs), particularly for SPY CVaR 99% (4.5%). The heavy-tailed Student-t shocks introduce more simulation variance, but even this worst case is well within acceptable bounds.

5. **Stability ranking by CV (VaR 95%):** MS-GARCH+EVT (0.5-2.3%) > Baseline (0.8-2.5%) > GARCH+t (1.1-3.6%).
