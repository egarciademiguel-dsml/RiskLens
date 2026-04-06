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

**Not yet implemented:**
- EGARCH: allows asymmetric response (negative shocks increase vol more than positive — leverage effect).
- GJR-GARCH: similar asymmetry via indicator function for negative returns.
- Regime-switching (HMM): discrete market states (calm/crisis) each with own μ, σ.
- Stochastic volatility: volatility itself follows a random process (Heston model).

---

## 4. Model Combinations

The two dimensions are independent:

| | Constant σ | GARCH(1,1) σ(t) |
|---|---|---|
| **Normal Z** | Textbook GBM | Vol clustering, thin tails |
| **Student-t Z** | Fat tails, flat vol | **Vol clustering + fat tails** |

The bottom-right cell (GARCH + Student-t) is the most realistic for financial data.

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

---

## 6. What the Numbers Mean (Interpretation Guide)

- **VaR = -15% at 95%**: In 95 out of 100 scenarios, you lose less than 15%. In the worst 5, you lose more.
- **CVaR = -22% at 95%**: When things go bad (worst 5%), the average loss is 22%.
- **Sharpe = 0.5**: For every unit of risk, you get 0.5 units of excess return. >1 is strong.
- **GARCH persistence = 0.97**: After a volatility spike, it takes ~33 days to halve (`ln(2)/ln(0.97) ≈ 23`, but roughly: `1/(1-0.97) ≈ 33` day half-life).
- **Student-t df = 4**: Tails are ~3x heavier than normal at the 1% level. Extreme events are much more likely.
