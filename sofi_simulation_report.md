# SOFI Stock Simulation Report
**Date:** 2025-12-24
**Current Price:** $27.48
**Simulation Horizon:** 30 Days
**Iterations:** 1000 per model

## Model Comparison

| Model | Expected Price (Mean) | Bear Case (5th %) | Bull Case (95th %) | Volatility (Std Dev) |
|---|---|---|---|---|
| Geometric Brownian Motion (GBM) | $29.34 | $20.59 | $40.46 | $6.06 |
| Merton Jump-Diffusion | $28.08 | $18.77 | $39.19 | $6.24 |
| Heston Stochastic Volatility | $28.23 | $27.45 | $28.68 | $0.46 |
| Historical Bootstrapping | $29.94 | $20.52 | $40.98 | $6.39 |
| Ornstein-Uhlenbeck (Mean Reverting) | $15.34 | $1.96 | $29.53 | $8.30 |
| CEV Model | $28.49 | $23.79 | $34.20 | $3.08 |

## Analysis
- **GBM**: Standard baseline, assumes constant volatility.
- **Jump-Diffusion**: Accounts for potential market crashes/spikes based on SOFI's history.
- **Heston**: Models volatility as changing over time, often resulting in fatter tails.
- **Bootstrapping**: Uses actual past returns, capturing the unique 'personality' of SOFI's price action.
- **OU**: Assumes price will revert to the mean (conservative).
- **CEV**: Adjusts volatility based on price level (leverage effect).