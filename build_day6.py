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
    md_cell("# Day 6: Per-Stock Fitting and Disagreement Detection\nWe fit a separate NH-HMM and Baseline HMM for each stock, map states to consistent labels, and detect days where the models disagree exactly on an event date."),
    code_cell("""import sys
sys.path.append('../')

import os
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from src.nonhomogeneous_hmm import NonHomogeneousGaussianHMM
from src.regime_utils import label_states_by_mean
import warnings
warnings.filterwarnings('ignore')"""),
    md_cell("## 1. Load Data"),
    code_cell("""returns_with_events = pd.read_parquet('../data/returns_with_events.parquet')
events_table = pd.read_parquet('../data/events_table.parquet')

# Get unique stock symbols (exclude NIFTY50)
symbols = [sym for sym in returns_with_events['symbol'].unique() if sym != '^NSEI']
print(f"Fitting models for {len(symbols)} stocks.")

os.makedirs('../data/per_stock_states', exist_ok=True)"""),
    md_cell("## 2. Fit Models and Decode States for Each Stock"),
    code_cell("""all_disagreements = []

for sym in symbols:
    # 1. Get stock data
    stock_data = returns_with_events[returns_with_events['symbol'] == sym].copy()
    X = np.column_stack([stock_data['log_return'].values, stock_data['volatility'].values])
    E = stock_data['E_t'].values.astype(bool)
    dates = stock_data['date'].values
    
    # 2. Fit Baseline HMM
    best_base_ll = -np.inf
    best_base = None
    for i in range(3):
        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=i)
        model.fit(X)
        ll = model.score(X)
        if ll > best_base_ll:
            best_base_ll = ll
            best_base = model
            
    base_states = best_base.predict(X)
    
    # 3. Fit NH-HMM
    nh = NonHomogeneousGaussianHMM(n_components=3, n_iter=100, n_restarts=3, random_state=42)
    nh.fit(X, E)
    nh_ll = nh.score(X, E)
    nh_states = nh.decode(X, E)
    
    # 4. Standardize Labels
    base_labels_map, base_std_map = label_states_by_mean(best_base.means_, best_base.covars_)
    nh_labels_map, nh_std_map = label_states_by_mean(nh.means_, nh.covars_)
    
    stock_data['baseline_state'] = [base_std_map[s] for s in base_states]
    stock_data['baseline_label'] = [base_labels_map[s] for s in base_states]
    stock_data['nh_state'] = [nh_std_map[s] for s in nh_states]
    stock_data['nh_label'] = [nh_labels_map[s] for s in nh_states]
    
    # 5. Save per-stock states
    out_cols = ['date', 'baseline_state', 'baseline_label', 'nh_state', 'nh_label']
    stock_data[out_cols].to_parquet(f'../data/per_stock_states/{sym}.parquet')
    
    # 6. Find disagreements on EVENT DATES
    A_event_norm = np.linalg.norm(nh.A_event_ - nh.A_normal_, 'fro')
    
    sym_events = events_table[events_table['symbol'] == sym]
    
    for _, ev in sym_events.iterrows():
        ev_date = ev['event_date']
        # Check if event_date is a trading day
        day_data = stock_data[stock_data['date'] == ev_date]
        if len(day_data) == 0:
            continue
            
        day_data = day_data.iloc[0]
        
        if day_data['baseline_state'] != day_data['nh_state']:
            disagreement = {
                'symbol': sym,
                'event_date': ev_date,
                'event_type': ev['event_type'],
                'baseline_state': day_data['baseline_state'],
                'baseline_label': day_data['baseline_label'],
                'nh_state': day_data['nh_state'],
                'nh_label': day_data['nh_label'],
                'baseline_ll': best_base_ll,
                'nh_ll': nh_ll,
                'A_event_norm': A_event_norm,
                'days_to_event_signed': 0  # It's exactly on the event date
            }
            all_disagreements.append(disagreement)

disagreement_df = pd.DataFrame(all_disagreements)
print(f"Total raw disagreements found on event dates: {len(disagreement_df)}")"""),
    md_cell("## 3. Quality Filters"),
    code_cell("""if len(disagreement_df) > 0:
    # Filter 1: A_event_norm threshold
    threshold = 0.01
    filtered_df = disagreement_df[disagreement_df['A_event_norm'] >= threshold]
    
    if len(filtered_df) < 80:
        print(f"Only {len(filtered_df)} rows with threshold 0.01. Reducing to 0.005.")
        threshold = 0.005
        filtered_df = disagreement_df[disagreement_df['A_event_norm'] >= threshold]
        
    print(f"Rows after A_event_norm >= {threshold}: {len(filtered_df)}")
    
    # Filter 2: Drop stocks with < 10 disagreement rows
    counts = filtered_df['symbol'].value_counts()
    valid_stocks = counts[counts >= 10].index
    dropped_stocks = counts[counts < 10].index
    
    print(f"Stocks dropped due to <10 rows: {list(dropped_stocks)}")
    
    filtered_df = filtered_df[filtered_df['symbol'].isin(valid_stocks)]
    print(f"Rows after dropping sparse stocks: {len(filtered_df)}")
    
    # Filter 3: Target 80-100 rows
    if len(filtered_df) > 100:
        filtered_df['ll_diff'] = np.abs(filtered_df['nh_ll'] - filtered_df['baseline_ll'])
        filtered_df = filtered_df.sort_values('ll_diff', ascending=False).head(100)
        filtered_df = filtered_df.drop(columns=['ll_diff'])
        print(f"Capped at top 100 rows by log-likelihood difference.")
        
    disagreement_df = filtered_df
    
print("Final disagreement count per stock:")
if len(disagreement_df) > 0:
    print(disagreement_df['symbol'].value_counts())
else:
    print("WARNING: No disagreements found matching criteria!")"""),
    md_cell("## 4. Save Disagreement Table"),
    code_cell("""if len(disagreement_df) > 0:
    disagreement_df.to_parquet('../data/disagreement_table.parquet')
    print("Saved disagreement_table.parquet to /data/")
else:
    # Create empty dataframe with schema just in case
    empty_df = pd.DataFrame(columns=[
        'symbol', 'event_date', 'event_type', 'baseline_state', 
        'baseline_label', 'nh_state', 'nh_label', 'baseline_ll', 
        'nh_ll', 'A_event_norm', 'days_to_event_signed'
    ])
    empty_df.to_parquet('../data/disagreement_table.parquet')
    print("Saved empty disagreement_table.parquet to /data/")""")
]

create_notebook('d:/Regime_Detection/notebooks/06_per_stock_fitting.ipynb', cells)
