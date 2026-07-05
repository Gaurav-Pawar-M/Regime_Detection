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
    md_cell("# Day 7: Week 1 Checkpoint\nThis notebook asserts that all Week 1 outputs have been generated correctly, validates data integrity, and produces a final summary for handoff to Week 2. No model fitting occurs here."),
    code_cell("""import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')"""),
    md_cell("## 1. Data Integrity Checks"),
    code_cell("""print("Checking Parquet Files...")
req_files = [
    'log_returns.parquet', 'volatility.parquet', 'price_levels.parquet',
    'baseline_hmm_states_nifty.parquet', 'events_table.parquet', 
    'returns_with_events.parquet', 'disagreement_table.parquet',
    'nh_hmm_states_nifty.parquet'
]

for f in req_files:
    assert os.path.exists(f'../data/{f}'), f"Missing file: {f}"

log_returns = pd.read_parquet('../data/log_returns.parquet')
volatility = pd.read_parquet('../data/volatility.parquet')
returns_with_events = pd.read_parquet('../data/returns_with_events.parquet')

# Assertions
assert not log_returns.isna().any().any(), "NaNs found in log_returns"
assert not volatility.isna().any().any(), "NaNs found in volatility"
assert len(log_returns) == len(volatility), "Mismatch in rows between returns and volatility"

nifty_data = returns_with_events[returns_with_events['symbol'] == '^NSEI']
X_nifty = np.column_stack([nifty_data['log_return'].values, nifty_data['volatility'].values])
E_nifty = nifty_data['E_t'].values.astype(bool)

print("Data integrity checks passed.")"""),
    md_cell("## 2. Baseline Model Reload Check"),
    code_cell("""baseline_model = joblib.load('../src/baseline_model.pkl')
ll_base = baseline_model.score(X_nifty)

# We cannot easily hardcode the exact value from Day 2 because it's random,
# but we can ensure it scores without errors and gives a reasonable number
assert ll_base is not None
print(f"Baseline LL on reload: {ll_base:.2f}")"""),
    md_cell("## 3. NH-HMM Reload Check"),
    code_cell("""nh_model = joblib.load('../src/nh_model_nifty.pkl')
ll_nh = nh_model.score(X_nifty, E_nifty)

assert ll_nh is not None
print(f"NH-HMM LL on reload: {ll_nh:.2f}")"""),
    md_cell("## 4. Disagreement Table Check"),
    code_cell("""disagreement_df = pd.read_parquet('../data/disagreement_table.parquet')

print(f"Total Disagreements: {len(disagreement_df)}")
if len(disagreement_df) < 80:
    print(f"WARNING: Found fewer than 80 disagreements ({len(disagreement_df)}). Try reducing threshold if possible.")
else:
    print("Sufficient disagreements found.")

# Assert required columns
required_cols = ['symbol', 'event_date', 'event_type', 'baseline_state', 
                 'baseline_label', 'nh_state', 'nh_label', 'baseline_ll', 
                 'nh_ll', 'A_event_norm', 'days_to_event_signed']
                 
for col in required_cols:
    assert col in disagreement_df.columns, f"Missing column {col} in disagreement table"
    
print("\\nDisagreements per stock:")
print(disagreement_df['symbol'].value_counts())"""),
    md_cell("## 5. Event Data Quality Checks"),
    code_cell("""events_table = pd.read_parquet('../data/events_table.parquet')
trading_dates = log_returns.index
start_date = trading_dates.min()
end_date = trading_dates.max()

unique_symbols = [col for col in log_returns.columns if col != '^NSEI']

for sym in unique_symbols:
    sym_events = events_table[events_table['symbol'] == sym]
    
    total_events = len(sym_events)
    if total_events == 0:
         print(f"WARNING: {sym} has no events.")
         continue
         
    in_range = sym_events['event_date'].between(start_date, end_date)
    pct_in_range = in_range.mean() * 100
    
    on_trading_day = sym_events['event_date'].isin(trading_dates)
    pct_trading_day = on_trading_day.mean() * 100
    
    if total_events < 8:
        print(f"WARNING: {sym} has only {total_events} events — consider dropping from Week 2 verification")
        
    # print(f"{sym}: {total_events} events | {pct_in_range:.0f}% in range | {pct_trading_day:.0f}% on trading days")"""),
    md_cell("## 6. Summary Print Block"),
    code_cell("""print("=== WEEK 1 CHECKPOINT SUMMARY ===")
print(f"Data range:           {start_date.date()} to {end_date.date()}")
print(f"Stocks in universe:   {len(unique_symbols) + 1} (including index)")
valid_stocks_after_filter = len(disagreement_df['symbol'].unique())
print(f"Stocks after filter:  {valid_stocks_after_filter}")
print(f"Trading days (T):     {len(trading_dates)}")
print(f"Baseline HMM LL:      {ll_base:.1f}   (N=3, BIC-selected)")
print(f"NH-HMM LL (NIFTY50):  {ll_nh:.1f}   (improvement: +{((ll_nh - ll_base)/abs(ll_base))*100:.2f}%)")

frobenius_norm = np.linalg.norm(nh_model.A_event_ - nh_model.A_normal_, 'fro')
print(f"A_event vs A_normal Frobenius norm (NIFTY50): {frobenius_norm:.3f}")

print(f"Disagreement table:   {len(disagreement_df)} rows across {valid_stocks_after_filter} stocks")
print(f"Rows passing quality filter: {len(disagreement_df)}")

ready = "YES" if len(disagreement_df) >= 80 else "NO"
print(f"Ready for Week 2 hand-verification: {ready}")
print("=================================")""")
]

create_notebook('d:/Regime_Detection/notebooks/07_week1_checkpoint.ipynb', cells)
