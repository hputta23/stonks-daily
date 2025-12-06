import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Add current directory to path to import backend
sys.path.append(os.getcwd())

from backend.model import MonteCarloPredictor

def run_sofi_report():
    ticker = "SOFI"
    print(f"Fetching data for {ticker}...")
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2) # 2 years data
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print("No data found for SOFI")
        return

    # Reset index to make Date a column if it's the index
    data = data.reset_index()
    
    # Ensure columns are correct (yfinance might return MultiIndex)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Predictor
    mc = MonteCarloPredictor()
    # Train (calculate stats)
    mc.train(data)
    
    methods = [
        ("Geometric Brownian Motion (GBM)", "gbm"),
        ("Merton Jump-Diffusion", "jump_diffusion"),
        ("Heston Stochastic Volatility", "heston"),
        ("Historical Bootstrapping", "bootstrapping"),
        ("Ornstein-Uhlenbeck (Mean Reverting)", "ou"),
        ("CEV Model", "cev")
    ]
    
    report_lines = []
    report_lines.append(f"# SOFI Stock Simulation Report")
    report_lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    report_lines.append(f"**Current Price:** ${data['Close'].iloc[-1]:.2f}")
    report_lines.append(f"**Simulation Horizon:** 30 Days")
    report_lines.append(f"**Iterations:** 1000 per model")
    report_lines.append("\n## Model Comparison\n")
    report_lines.append("| Model | Expected Price (Mean) | Bear Case (5th %) | Bull Case (95th %) | Volatility (Std Dev) |")
    report_lines.append("|---|---|---|---|---|")
    
    print("Running simulations...")
    
    for name, method in methods:
        print(f"Simulating {name}...")
        try:
            dates, mean_path, paths_list = mc.predict_paths(data, days=30, iterations=1000, method=method)
            
            # paths_list is list of lists (iterations x days)
            # Get final prices (last day of each path)
            final_prices = [path[-1] for path in paths_list]
            
            mean_price = np.mean(final_prices)
            bear_price = np.percentile(final_prices, 5)
            bull_price = np.percentile(final_prices, 95)
            std_dev = np.std(final_prices)
            
            report_lines.append(f"| {name} | ${mean_price:.2f} | ${bear_price:.2f} | ${bull_price:.2f} | ${std_dev:.2f} |")
            
        except Exception as e:
            print(f"Error running {name}: {e}")
            report_lines.append(f"| {name} | Error | - | - | - |")

    report_lines.append("\n## Analysis")
    report_lines.append("- **GBM**: Standard baseline, assumes constant volatility.")
    report_lines.append("- **Jump-Diffusion**: Accounts for potential market crashes/spikes based on SOFI's history.")
    report_lines.append("- **Heston**: Models volatility as changing over time, often resulting in fatter tails.")
    report_lines.append("- **Bootstrapping**: Uses actual past returns, capturing the unique 'personality' of SOFI's price action.")
    report_lines.append("- **OU**: Assumes price will revert to the mean (conservative).")
    report_lines.append("- **CEV**: Adjusts volatility based on price level (leverage effect).")

    report_content = "\n".join(report_lines)
    
    with open("sofi_simulation_report.md", "w") as f:
        f.write(report_content)
        
    print("Report generated: sofi_simulation_report.md")

if __name__ == "__main__":
    run_sofi_report()
