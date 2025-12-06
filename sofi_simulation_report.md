# SOFI Stock Simulation Report
**Date:** 2025-12-02
**Current Price:** $29.51
**Simulation Horizon:** 30 Days
**Iterations:** 1000 per model

## Model Comparison

| Model | Expected Price (Mean) | Bear Case (5th %) | Bull Case (95th %) | Volatility (Std Dev) |
|---|---|---|---|---|
| Geometric Brownian Motion (GBM) | $31.88 | $21.69 | $44.09 | $6.90 |
| Merton Jump-Diffusion | $30.68 | $20.52 | $43.52 | $7.16 |
| Heston Stochastic Volatility | $30.86 | $30.09 | $31.28 | $0.41 |
| Historical Bootstrapping | $32.54 | $22.54 | $45.09 | $7.18 |
| Ornstein-Uhlenbeck (Mean Reverting) | $14.74 | $2.18 | $28.75 | $7.82 |
| CEV Model | $30.95 | $25.74 | $36.63 | $3.26 |

## Analysis
- **GBM**: Standard baseline, assumes constant volatility.
- **Jump-Diffusion**: Accounts for potential market crashes/spikes based on SOFI's history.
- **Heston**: Models volatility as changing over time, often resulting in fatter tails.
- **Bootstrapping**: Uses actual past returns, capturing the unique 'personality' of SOFI's price action.
- **OU**: Assumes price will revert to the mean (conservative).
- **CEV**: Adjusts volatility based on price level (leverage effect).