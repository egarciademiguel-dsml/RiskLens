# Mathematical Reference

A concise summary of the models, metrics, and assumptions behind RiskLens.

---

## 1. Return Types

**Simple return:**
```
R(t) = (P(t) - P(t-1)) / P(t-1)
```
Used in: Sharpe, Sortino, gain-to-pain, rolling volatility.

**Log return:**
```
r(t) = ln(P(t) / P(t-1))
```
Used in: Monte Carlo simulation (GBM). Additive over time: `r(1..T) = sum(r(t))`.

For small values, `R(t) ≈ r(t)`. They diverge for large moves.

---

## 2. Risk Metrics

### Annualized Volatility
```
σ_annual = σ_daily × √252
```
Square-root-of-time rule. Assumes i.i.d. daily returns. 252 = trading days/year.

### Sharpe Ratio
```
SR = (μ - Rf/252) / σ × √252
```
Excess return per unit of total risk. Optimal interpretation under normality; misleading for skewed distributions.

### Sortino Ratio
```
So = (μ - Rf/252) / σ_down × √252
σ_down = std(returns where returns < 0)
```
Like Sharpe but only penalizes downside volatility.

### Maximum Drawdown
```
DD(t) = (P(t) - max(P(0..t))) / max(P(0..t))
MaxDD = min(DD(t)) for all t
```
Worst peak-to-trough decline. No distributional assumptions.

### Value at Risk (VaR)
```
VaR(α) = percentile(returns, (1-α) × 100)
```
Example: VaR(95%) = 5th percentile of return distribution. "With 95% confidence, losses will not exceed |VaR|."

### Conditional VaR (CVaR / Expected Shortfall)
```
CVaR(α) = mean(returns where returns ≤ VaR(α))
```
Average loss in the worst (1-α) scenarios. Always CVaR ≤ VaR. Captures tail severity, not just threshold.

### Skewness (Fisher-Pearson)
```
γ = E[(X - μ)³] / σ³
```
γ < 0: left-skewed (fat left tail, crashes). γ > 0: right-skewed. γ = 0: symmetric.

### Excess Kurtosis (Fisher)
```
κ = E[(X - μ)⁴] / σ⁴ - 3
```
κ > 0: heavier tails than normal (leptokurtic). Typical for financial returns: κ ≈ 3-10+.

---

## 3. Simulation Models

### 3.1 Geometric Brownian Motion (GBM)

The base framework for all simulation in RiskLens.

```
S(t) = S(0) × exp(Σ log_return(i) for i=1..t)
log_return(t) = drift + σ × Z(t)
drift = μ - 0.5 × σ²
```

The `-0.5σ²` term is Jensen's inequality correction: ensures `E[S(t)]` equals the arithmetic mean return, not the geometric.

`Z(t)` are i.i.d. unit shocks (mean=0, variance=1). The choice of distribution for Z defines the model variant.

### 3.2 Shock Distributions

**Normal (Gaussian):**
```
Z ~ N(0, 1)
```
Thin tails. Underestimates extreme events. Baseline/textbook model.

**Student-t (implemented):**
```
T ~ t(df)
Z = T / √(df / (df - 2))
```
The scaling ensures Var(Z) = 1 so drift and σ retain their meaning. Lower df = heavier tails. Typical for financial data: df ≈ 3-8. df > 30 ≈ normal.

Degrees of freedom fitted via MLE: `scipy.stats.t.fit(returns)`. Clamped df ≥ 2.1 (variance undefined for df ≤ 2).

**Not yet implemented:**
- Skewed Student-t (Hansen's): adds asymmetry parameter. Captures left-skew in crashes.
- Generalized Hyperbolic: 4-parameter family (includes NIG, Variance-Gamma). Independent control of skew and tail weight.
- Extreme Value Theory (GPD): models only the tails. Gold standard for VaR/CVaR.

### 3.3 Volatility Models

**Constant volatility:**
```
σ(t) = σ = std(historical returns)    for all t
```
Simple. Ignores volatility clustering.

**GARCH(1,1) (implemented):**
```
σ²(t) = ω + α × ε²(t-1) + β × σ²(t-1)
```
Where:
- ω (omega): baseline variance floor
- α (alpha): reaction to recent shocks — how much yesterday's surprise matters
- β (beta): persistence — how slowly volatility decays
- ε(t-1) = σ(t-1) × Z(t-1): realized shock

**Key properties:**
- Persistence = α + β. Must be < 1 for stationarity.
- Long-run variance = ω / (1 - α - β)
- High persistence (α + β ≈ 0.99): volatility shocks take months to decay
- High α, low β: reactive (jumps quickly, decays quickly)
- Low α, high β: sluggish (slow to react, slow to revert)

In simulation, the drift correction uses the current day's variance:
```
drift(t) = μ - 0.5 × σ²(t)
log_return(t) = drift(t) + σ(t) × Z(t)
```

**HMM Regime Detection (implemented):**

Markets alternate between latent states (regimes), each with distinct drift and volatility. A Hidden Markov Model learns these states from observed returns.

```
Hidden states: S(t) ∈ {1, ..., K}
Transition matrix: A(i,j) = P(S(t)=j | S(t-1)=i)
Emission per regime k: r(t) | S(t)=k ~ N(μ(k), σ(k)²)
```

- **Fitting**: Baum-Welch algorithm (EM) maximizes P(observations | model)
- **Decoding**: Viterbi algorithm finds most likely state sequence
- **Regimes sorted by σ ascending**: regime 0 = calmest, regime K-1 = most volatile
- **Simulation**: At each step, transition to next regime via A, then draw from that regime's N(μ(k), σ(k))

Typical setup: K=2 (calm/crisis) or K=3 (calm/moderate/crisis). K=1 reduces to constant model.

**GMM Regime Detection (implemented):**

Alternative to HMM: cluster returns using feature engineering + Gaussian Mixture Model, then classify with Random Forest.

```
Features: rolling_vol(5,10,21,63d), rolling_mean(5,10,21d), skew, kurtosis
GMM: fit K Gaussians to feature space → cluster labels
RF classifier: features → regime label (for real-time prediction)
Transition matrix: estimated from observed label sequences (empirical Markov)
```

Same simulation logic as HMM: Markov transitions + per-regime μ(k), σ(k). The difference is in fitting — GMM uses richer features but doesn't learn transitions natively.

**XGBoost Realized Volatility (implemented):**

ML-driven volatility prediction. Instead of parametric models, use gradient-boosted trees to predict forward realized vol.

```
Target: σ_realized(t, h) = std(r(t+1), ..., r(t+h))    for horizon h
Features (12): rolling_vol(5,10,21,63d), rolling_mean(5,10,21d),
               rolling_skew(21d), rolling_kurtosis(21d), |r(t)|, r(t)²
```

- Trained on historical (features, target) pairs with `XGBRegressor`
- Predicted σ replaces constant/GARCH σ in GBM: `log_return(t) = (μ - 0.5σ_pred²) + σ_pred × Z(t)`
- Horizons: 5d (weekly), 10d (biweekly), 21d (monthly)
- Evaluation: R² on training set (no forward leakage — target is future vol)

**Not yet implemented:**
- EGARCH: allows asymmetric response (negative shocks increase vol more than positive — leverage effect).
- GJR-GARCH: similar asymmetry via indicator function for negative returns.
- Stochastic volatility: volatility itself follows a random process (Heston model).

---

## 4. Model Combinations

The volatility and shock dimensions are independent. RiskLens implements 5 volatility models × 2 shock distributions:

| | Constant σ | GARCH(1,1) σ(t) | HMM regimes | GMM regimes | XGBoost σ_pred |
|---|---|---|---|---|---|
| **Normal Z** | Textbook GBM | Vol clustering | Regime-switching drift+vol | Feature-based regimes | ML-predicted vol |
| **Student-t Z** | Fat tails | **Clustering + fat tails** | Regimes + fat tails | Regimes + fat tails | ML vol + fat tails |

Additionally, regime models (HMM, GMM) support K=1,2,3 regimes, and XGBoost supports 5/10/21-day horizons — yielding 13 configurations in total.

---

## 5. Key Assumptions

| Assumption | Impact | Mitigation |
|---|---|---|
| Returns are i.i.d. (constant model) | Ignores autocorrelation and clustering | GARCH partially addresses clustering |
| Log-normal prices (GBM) | Prices always positive | Correct by construction |
| Historical params = future params | Stationarity assumption | No mitigation; fundamental limitation |
| No jumps | Misses earnings gaps, flash crashes | Would need jump-diffusion (Merton) |
| Symmetric shocks (Student-t) | Left/right tails equal | Skewed-t would address this |
| No transaction costs | Gross returns | Acceptable for risk estimation |
| 252 trading days/year | Industry standard | Approximate; varies by market |
| Stationarity within training window (backtest) | Model params fixed per window | Rolling refit; shorter windows adapt faster but noisier |

---

## 6. What the Numbers Mean (Interpretation Guide)

- **VaR = -15% at 95%**: In 95 out of 100 scenarios, you lose less than 15%. In the worst 5, you lose more.
- **CVaR = -22% at 95%**: When things go bad (worst 5%), the average loss is 22%.
- **Sharpe = 0.5**: For every unit of risk, you get 0.5 units of excess return. >1 is strong.
- **GARCH persistence = 0.97**: After a volatility spike, it takes ~33 days to halve (`ln(2)/ln(0.97) ≈ 23`, but roughly: `1/(1-0.97) ≈ 33` day half-life).
- **Student-t df = 4**: Tails are ~3x heavier than normal at the 1% level. Extreme events are much more likely.

---

## 7. VaR Backtesting

A model that produces VaR numbers is useless unless those numbers are validated. Backtesting checks whether predicted VaR is consistent with observed losses.

### Rolling Window Approach

```
For each day t in test period:
  1. Fit model on returns[t-W : t]     (training window of W days)
  2. Run MC simulation → compute 1-day VaR(α)
  3. Observe actual return r(t+1)
  4. Record breach: I(t) = 1 if r(t+1) < VaR(α)
```

If the model is correctly calibrated, the breach rate should converge to `(1 - α)`. A 95% VaR should be breached ~5% of the time.

### Kupiec Test (1995) — Unconditional Coverage

Tests whether the observed breach rate differs from the expected rate.

```
H₀: p = p₀ = (1 - α)
H₁: p ≠ p₀

LR_uc = -2 ln[L(p₀) / L(p̂)]
      = -2 ln[(1-p₀)^(n-x) × p₀^x] + 2 ln[(1-p̂)^(n-x) × p̂^x]

where n = total observations, x = total breaches, p̂ = x/n
LR_uc ~ χ²(1) under H₀
```

Reject H₀ if p-value < 0.05 → the model's VaR is miscalibrated.

**Edge cases**: If x=0 (no breaches) or x=n (all breaches), the model is clearly wrong. The LR formula handles x=0 via `p̂^0 = 1`.

### Christoffersen Test (1998) — Independence

A model can have the right breach rate but cluster all breaches together (e.g., 10 consecutive breaches in a crash, then none for a year). This test checks that breaches are i.i.d.

```
Build 2×2 transition matrix from breach sequence:
  n₀₀ = no-breach → no-breach
  n₀₁ = no-breach → breach
  n₁₀ = breach → no-breach
  n₁₁ = breach → breach

π₀₁ = n₀₁ / (n₀₀ + n₀₁)    (prob of breach after no-breach)
π₁₁ = n₁₁ / (n₁₀ + n₁₁)    (prob of breach after breach)
p̂   = (n₀₁ + n₁₁) / n       (unconditional breach rate)

H₀: π₀₁ = π₁₁ (breaches are independent of prior state)

LR_ind = -2 ln[L_independent / L_markov]
LR_ind ~ χ²(1) under H₀
```

Reject H₀ if p-value < 0.05 → breaches are clustered, the model fails to capture volatility dynamics.

### Conditional Coverage (Joint Test)

The combined test checks both coverage and independence simultaneously:

```
LR_cc = LR_uc + LR_ind ~ χ²(2)
```

A model passes if it has both: (1) correct breach rate and (2) independent breaches.

### Interpretation

| Result | Meaning |
|--------|---------|
| Kupiec pass, Christoffersen pass | Model is well-calibrated |
| Kupiec fail (too many breaches) | Model underestimates risk |
| Kupiec fail (too few breaches) | Model is too conservative |
| Christoffersen fail | Breaches cluster — model misses vol dynamics |
