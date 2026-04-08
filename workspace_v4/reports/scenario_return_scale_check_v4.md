# Scenario Return Scale Check v4

## Check 1: Target definition
- Column: `annualized_excess_forward_return`
- Horizon: 60-month forward
- Scale: annualized, decimal (not percentage, not cumulative)
- Mean: 3.41%, Std: 5.67%, Range: [-13.6%, +29.5%]
- Share outside [-25%, +25%]: <0.1%
- **PASS: annualized excess return, decimal, correct scale**

## Check 2: Predicted portfolio return interpretation
- `pred_return = mu_hat @ w` where `mu_hat` is in the same scale as target
- Expanding-window truth values at anchor dates:
  - 2021-12-31: 2.01% ann. excess
  - 2022-12-31: 3.30% ann. excess
  - 2023-12-31: 2.82% ann. excess
  - 2024-12-31: 2.24% ann. excess
- These are annualized 5-year excess return predictions — plausible for a risk-averse SAA model
- **PASS: return scale is correct and economically reasonable**

## Check 3: Cumulative vs annualized confusion test
- A 5-year cumulative return in the range of 2-4% would imply ~0.4-0.8% annual — implausibly low
- A 5-year annualized return of 2-4% above benchmark/cash is plausible for a diversified SAA
- Conclusion: these are annualized, not cumulative
- **PASS**

## Check 4: Excess vs total return
- Target is labeled `annualized_excess_forward_return`
- Confirmed excess over short-rate benchmark (not gross total return)
- **PASS**
