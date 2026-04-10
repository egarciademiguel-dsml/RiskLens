# Horizon Crossover — Results

Per-tier VaR 95% and CVaR 99% across 7 horizons × 3 assets.
Sims: 10,000 paths per (asset, tier, horizon), seed=42.

## BTC-USD — VaR 95% by horizon

| Horizon | Baseline | GARCH+t | MS-GARCH+EVT | Deepest |
|---|---|---|---|---|
| 1d | -4.7% | -3.0% | -3.3% | Baseline |
| 5d | -10.5% | -7.0% | -8.7% | Baseline |
| 10d | -14.1% | -10.0% | -12.6% | Baseline |
| 21d | -20.0% | -14.5% | -19.5% | Baseline |
| 63d | -31.5% | -24.6% | -36.1% | MS-GARCH+EVT |
| 126d | -41.7% | -33.9% | -49.8% | MS-GARCH+EVT |
| 252d | -52.4% | -45.0% | -62.5% | MS-GARCH+EVT |

## SPY — VaR 95% by horizon

| Horizon | Baseline | GARCH+t | MS-GARCH+EVT | Deepest |
|---|---|---|---|---|
| 1d | -1.7% | -1.8% | -1.8% | MS-GARCH+EVT |
| 5d | -3.8% | -4.1% | -3.7% | GARCH+t |
| 10d | -5.0% | -5.4% | -4.8% | GARCH+t |
| 21d | -7.0% | -7.5% | -6.3% | GARCH+t |
| 63d | -10.6% | -10.4% | -10.5% | Baseline |
| 126d | -13.4% | -12.3% | -15.4% | MS-GARCH+EVT |
| 252d | -15.3% | -13.7% | -22.0% | MS-GARCH+EVT |

## NVDA — VaR 95% by horizon

| Horizon | Baseline | GARCH+t | MS-GARCH+EVT | Deepest |
|---|---|---|---|---|
| 1d | -5.1% | -3.9% | -3.6% | Baseline |
| 5d | -10.7% | -8.4% | -7.3% | Baseline |
| 10d | -13.9% | -11.7% | -9.3% | Baseline |
| 21d | -18.7% | -15.8% | -11.7% | Baseline |
| 63d | -25.9% | -24.4% | -20.9% | Baseline |
| 126d | -30.2% | -29.0% | -41.0% | MS-GARCH+EVT |
| 252d | -29.5% | -31.8% | -61.7% | MS-GARCH+EVT |

## Crossover summary

| Asset | Short-horizon winner | Long-horizon winner | Crossover horizon | Sequence |
|---|---|---|---|---|
| BTC-USD | Baseline | MS-GARCH+EVT | 63d | Baseline → Baseline → Baseline → Baseline → MS-GARCH+EVT → MS-GARCH+EVT → MS-GARCH+EVT |
| SPY | MS-GARCH+EVT | MS-GARCH+EVT | 5d | MS-GARCH+EVT → GARCH+t → GARCH+t → GARCH+t → Baseline → MS-GARCH+EVT → MS-GARCH+EVT |
| NVDA | Baseline | MS-GARCH+EVT | 126d | Baseline → Baseline → Baseline → Baseline → Baseline → MS-GARCH+EVT → MS-GARCH+EVT |

## Recommendation table — tier with deepest VaR 95%

This is the table consumed by RL-041 (model recommendation logic in the app).

| Asset | 1d | 5d | 10d | 21d | 63d | 126d | 252d |
|---|---|---|---|---|---|---|---|
| BTC-USD | Baseline | Baseline | Baseline | Baseline | MS-GARCH+EVT | MS-GARCH+EVT | MS-GARCH+EVT |
| SPY | MS-GARCH+EVT | GARCH+t | GARCH+t | GARCH+t | Baseline | MS-GARCH+EVT | MS-GARCH+EVT |
| NVDA | Baseline | Baseline | Baseline | Baseline | Baseline | MS-GARCH+EVT | MS-GARCH+EVT |

## Headline finding

The deepest-VaR tier **changes with horizon** on every asset, and the
sequence is **not monotonic on SPY**. There is no single best model — only a
horizon-conditional best. This is the practical reason the project ships
three tiers and the empirical justification for the horizon-aware
recommendation logic in RL-041.

### What the numbers actually show

- **BTC**: Baseline wins from 1d through 21d, then MS-GARCH+EVT takes over at
  63d and stays deepest through 252d. Crossover at **63d**. The MS-GARCH
  long-horizon dominance is large (-62.5% vs -45.0% at 252d) — this is the
  classic "regime mixing accumulates over long horizons" effect.
- **NVDA**: Baseline wins all the way out to 63d. MS-GARCH+EVT only takes
  over at **126d**, the **latest** crossover of the three assets. Once it
  flips, the gap is enormous (-61.7% vs -29.5% at 252d).
- **SPY**: **Non-monotonic.** MS-GARCH+EVT (1d) → GARCH+t (5-21d) → Baseline
  (63d) → MS-GARCH+EVT (126-252d). The earliest "crossover" is at 5d but the
  ranking flips three more times. SPY is the only asset where every tier wins
  at some horizon.

### Why the prior intuition was wrong

Before running this, the expected ordering was *"crypto crosses over earliest
because tails are heaviest; broad-market ETF crosses latest because tails are
thinnest."* The actual ordering (earliest to latest crossover) is:

| Rank | Asset | First crossover |
|---|---|---|
| 1 | SPY  | 5d (then oscillates) |
| 2 | BTC-USD | 63d |
| 3 | NVDA | 126d |

**SPY crosses earliest, not latest.** The reason is that SPY has the *lowest
absolute volatility*, so the gap between tier estimates at short horizons is
small in absolute terms and small noise/regime effects flip the ranking
easily. NVDA is the heaviest single-stock tail in the sample but its
single-regime σ stays high enough that Baseline keeps winning until the
regime-mixing effect catches up at 126d.

The lesson: tail-heaviness alone does not predict crossover horizon — what
matters is the *ratio* of long-run regime-mixing volatility to single-regime
σ, which can go either way depending on how persistent the asset's calm
regime is relative to its crisis frequency.

### Caveats when consuming the recommendation table

1. **Deepest VaR ≠ best calibrated.** Section 7 of `model_deep_dive` shows
   that on the daily backtest GARCH+t over-breaches and Baseline can look
   over-conservative. The recommendation here is "the most conservative
   estimate at horizon H," not "the best-calibrated model at horizon H."
   Calibration must be checked separately for the chosen tier at the chosen
   horizon.
2. **SPY's non-monotonic behaviour means the recommendation flips back and
   forth.** A user picking 21d on SPY gets GARCH+t; 63d gets Baseline; 126d
   gets MS-GARCH+EVT again. RL-041 should surface this rather than hide it.
