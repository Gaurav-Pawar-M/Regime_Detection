import json
import os

def create_notebook(filename, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split('\n')]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split('\n')]
    }

cells = [
    md_cell("# Day 1: Data Acquisition\nThis notebook downloads 5 years of daily OHLCV data for 18 stocks and the NIFTY50 index, computes log returns and rolling 10-day volatility, and saves the cleaned data to parquet files."),
    code_cell("""import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')"""),
    md_cell("## 1. Stock Universe and Configuration"),
    code_cell("""STOCKS = {
    "financials":  ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "it":          ["TCS.NS", "INFY.NS", "WIPRO.NS"],
    "energy":      ["RELIANCE.NS", "ONGC.NS"],
    "auto":        ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS"],
    "consumer":    ["HINDUNILVR.NS", "NESTLEIND.NS", "ASIANPAINT.NS"],
    "pharma":      ["SUNPHARMA.NS", "DRREDDY.NS"],
    "industrial":  ["LT.NS"],
}

# Flatten the stocks list
TICKERS = [ticker for sector_stocks in STOCKS.values() for ticker in sector_stocks]
INDEX = "^NSEI"
ALL_TICKERS = TICKERS + [INDEX]

START = "2019-01-01"
END   = "2024-06-30"

print(f"Total specific stocks: {len(TICKERS)}")
print(f"Index: {INDEX}")"""),
    md_cell("## 2. Download Data via yfinance"),
    code_cell("""# Using auto_adjust=True to get split/dividend adjusted prices
raw_data = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True)
raw_data.head()"""),
    md_cell("## 3. Data Cleaning and Flattening Columns"),
    code_cell("""# Flatten MultiIndex columns immediately
raw_data.columns = ['_'.join(col).strip() for col in raw_data.columns]

# Extract only the Close columns (which are already adjusted due to auto_adjust=True)
close_cols = [col for col in raw_data.columns if col.startswith("Close_")]
prices = raw_data[close_cols].copy()

# Rename columns to just the ticker name
prices.rename(columns=lambda x: x.replace("Close_", ""), inplace=True)
prices.head()"""),
    md_cell("## 4. Master Trading Calendar and Missing Value Imputation"),
    code_cell("""# Build master trading calendar (union of all non-NaN dates across all tickers)
# yf.download already aligns dates, but to be sure:
master_calendar = prices.dropna(how='all').index
prices = prices.reindex(master_calendar)

# Forward-fill at most 2 consecutive NaN days (exchange holidays)
prices = prices.ffill(limit=2)

# Drop any stock with > 5% missing rows after forward-filling
missing_pct = prices.isnull().sum() / len(prices)
print("Missing percentage per column after ffill:")
print(missing_pct)

cols_to_drop = missing_pct[missing_pct > 0.05].index
if len(cols_to_drop) > 0:
    print(f"Dropping {list(cols_to_drop)} due to >5% missing data.")
    prices.drop(columns=cols_to_drop, inplace=True)
else:
    print("No stocks dropped due to missing data.")

# Final dropna for any remaining NaNs at the very beginning of the series
prices.dropna(inplace=True)
print(f"Final shape of prices dataframe: {prices.shape}")"""),
    md_cell("## 5. Compute Log Returns and Realized Volatility"),
    code_cell("""# Calculate log returns
log_ret = np.log(prices / prices.shift(1)).dropna()

# Rolling 10-day realized volatility
vol = log_ret.rolling(10).std()

# Align returns and volatility to the same index (dropping the 9 rows of warmup)
# Using intersection of valid indices
valid_idx = vol.dropna(how='all').index
log_returns = log_ret.loc[valid_idx]
volatility = vol.loc[valid_idx]
prices_aligned = prices.loc[valid_idx]

print(f"Shape of log_returns: {log_returns.shape}")
print(f"Shape of volatility: {volatility.shape}")"""),
    md_cell("## 6. Sanity Checks and Visualizations"),
    code_cell("""print("Date range:")
print(f"Start: {log_returns.index.min()}")
print(f"End: {log_returns.index.max()}")
print(f"Total Trading Days: {len(log_returns)}")
print("\\nMissing values in log_returns:", log_returns.isna().sum().sum())
print("Missing values in volatility:", volatility.isna().sum().sum())"""),
    code_cell("""# Plot: NIFTY50 price series with log-scale y-axis
plt.figure(figsize=(12, 5))
plt.plot(prices_aligned[INDEX], label='NIFTY50 Price')
plt.yscale('log')
plt.title("NIFTY50 Price (Log Scale) 2019-2024")
plt.xlabel("Date")
plt.ylabel("Price (log scale)")
plt.legend()
plt.show()"""),
    code_cell("""# Plot: NIFTY50 log returns as a time series
plt.figure(figsize=(12, 5))
plt.plot(log_returns[INDEX], color='red', alpha=0.7, label='NIFTY50 Log Returns')
plt.title("NIFTY50 Log Returns (Note COVID crash in March 2020)")
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.axhline(0, color='black', lw=1)
plt.legend()
plt.show()"""),
    code_cell("""# Descriptive stats table (mean, std, skew, kurtosis)
def get_stats(df):
    stats = pd.DataFrame({
        'Mean (bps)': df.mean() * 10000,
        'Std Dev (%)': df.std() * 100,
        'Skewness': df.skew(),
        'Kurtosis': df.kurtosis()
    })
    return stats

ret_stats = get_stats(log_returns)
print("Descriptive Statistics for Log Returns:")
display(ret_stats)"""),
    md_cell("## 7. Save Parquet Files"),
    code_cell("""os.makedirs('../data', exist_ok=True)
log_returns.to_parquet('../data/log_returns.parquet')
volatility.to_parquet('../data/volatility.parquet')
prices_aligned.to_parquet('../data/price_levels.parquet')

print("Saved log_returns.parquet, volatility.parquet, and price_levels.parquet to /data/")""")
]

create_notebook('d:/Regime_Detection/notebooks/01_data_acquisition.ipynb', cells)
