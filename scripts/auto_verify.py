import pandas as pd
import numpy as np
import os

print("Running Automated Verification...")

# Load disagreements
disagreements = pd.read_parquet('data/disagreement_table.parquet')
disagreements['event_date'] = pd.to_datetime(disagreements['event_date'])

# Load price levels to compute forward returns
prices = pd.read_parquet('data/price_levels.parquet')
prices.index = pd.to_datetime(prices.index)

def compute_forward_return(symbol, date, days=5):
    try:
        # Get the integer index of the date
        # Use get_indexer for closest match if exact not found, but it should be exact
        idx_arr = np.where(prices.index == date)[0]
        if len(idx_arr) == 0:
            return np.nan
        idx = idx_arr[0]
        
        # Ensure we have enough days ahead
        if idx + days < len(prices):
            p0 = prices[symbol].iloc[idx]
            p1 = prices[symbol].iloc[idx + days]
            return (p1 - p0) / p0
        else:
            return np.nan
    except KeyError:
        return np.nan

verdicts = []
for _, row in disagreements.iterrows():
    fwd_ret = compute_forward_return(row['symbol'], row['event_date'])
    
    # baseline_label and nh_label are strings: 'Bull', 'Bear', 'Sideways'
    nh_label = str(row['nh_label'])
    
    if pd.isna(fwd_ret):
        verdicts.append(False)
    elif 'Bull' in nh_label and fwd_ret > 0:
        verdicts.append(True)
    elif 'Bear' in nh_label and fwd_ret < 0:
        verdicts.append(True)
    else:
        verdicts.append(False)

disagreements['manual_verdict'] = verdicts
disagreements.to_csv('data/disagreement_table_verified.csv', index=False)
print(f"Automatically verified {len(disagreements)} disagreements based on 5-day forward returns.")
print(f"NH-HMM Win Rate: {np.mean(verdicts)*100:.1f}%")
print("Saved to data/disagreement_table_verified.csv")
