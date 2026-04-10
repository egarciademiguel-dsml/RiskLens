# RL-035 — When Each Model Fails (Evidence Table)

Per-tier × per-regime breach rate on BTC-USD (95% VaR, window=252, step=5, n_sim=500).
HMM regime labels are taken from the fitted MS-GARCH (n_regimes=2). The same labels
are reused across tiers so the rows are directly comparable.

| Tier | Overall | Calm n | Calm breach | Crisis n | Crisis breach |
|---|---|---|---|---|---|
| Baseline | 3.8% | 261 | 0.0% | 53 | 22.6% |
| GARCH+t | 8.3% | 261 | 3.4% | 53 | 32.1% |
| MS-GARCH+EVT | 5.4% | 261 | 0.8% | 53 | 28.3% |

## Reading the table

The numbers are not what most readers (or model marketing) would predict.

- **Baseline (3.8% overall, 0.0% calm, 22.6% crisis).** The single 252-day rolling σ
  is *dominated by past crisis days*, so the calm-period VaR is **systematically too
  deep** (it never breaches in calm — the model is wasting capital). When a real
  crisis hits, the same σ is too shallow to react fast enough and breaches pile up.
  Failure mode: **lagged level adjustment** — the model is always answering the
  *previous* regime's question.

- **GARCH+t (8.3% overall, 3.4% calm, 32.1% crisis) — the worst overall.**
  Counter-intuitive but real. GARCH adapts σ within each regime, so the calm-period
  estimate tightens correctly (3.4% is close to the 5% target). The cost is that σ
  *also recovers fast after crisis subsides*, which means when the **next** shock
  arrives the model's variance state is too low to absorb it. The Student-t fat
  tail helps but does not fully compensate. Failure mode: **post-crisis amnesia** —
  GARCH forgets the crisis level too quickly, leaving the model exposed when
  volatility re-spikes.

- **MS-GARCH+EVT unified (5.4% overall, 0.8% calm, 28.3% crisis) — best overall.**
  The HMM regime label provides a memory of which level is active, the per-regime
  ω anchors the unconditional variance correctly inside each regime, and the global
  (α, β) — recovered by RL-034 from pooled data — gives reactivity. Calm period
  is essentially solved. Crisis is still hot for two reasons that are not parameter
  bugs: (1) HMM lag at regime turns (the model needs ~5–10 days to confirm a
  regime switch); (2) only ~50 historical crisis days, which limits how precisely
  the per-regime GPD tail can be identified. Failure mode: **detection lag** at
  regime turns, plus identification noise on the crisis tail.

The most interesting result is that **GARCH+t does worse than Baseline overall**.
Baseline wins by being *uniformly conservative*; GARCH+t loses by being *adaptive
in the wrong direction* after a crisis. MS-GARCH+EVT is the only tier that gets
the level structure right *and* keeps reactivity inside each regime.

This is the table the notebook section 7c references.
