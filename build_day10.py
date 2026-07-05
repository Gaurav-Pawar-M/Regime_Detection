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
    md_cell("# Day 10: Quantitative Validation\nWe compute the precision of the new model based on hand-verified labels, then perform a formal backtest comparing the Baseline HMM against the Non-Homogeneous HMM."),
    code_cell("""import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')"""),
    md_cell("## 1. Hand-Verification Precision Analysis"),
    code_cell("""verified_df = pd.read_csv('../data/disagreement_table_verified.csv')

def parse_verdict(v):
    if pd.isna(v): return 0.0
    v_str = str(v).strip().lower()
    if v_str == 'nh-hmm correct': return 1.0
    if v_str == 'partial': return 0.5
    return 0.0

verified_df['nh_score'] = verified_df['manual_verdict'].apply(parse_verdict)
total_cases = len(verified_df)
nh_correct_cases = verified_df['nh_score'].sum()

print(f"Total Hand-Verified Disagreements: {total_cases}")
print(f"NH-HMM Precision (Justified Calls): {nh_correct_cases / total_cases * 100:.1f}%")"""),
    md_cell("## 2. Prepare Backtest Data"),
    code_cell("""returns_with_events = pd.read_parquet('../data/returns_with_events.parquet')

# Merge states for each stock
state_dfs = []
symbols = [s for s in returns_with_events['symbol'].unique() if s != '^NSEI']

for sym in symbols:
    state_file = f'../data/per_stock_states/{sym}.parquet'
    if os.path.exists(state_file):
        df_sym = pd.read_parquet(state_file)
        # We need the forward return to evaluate the state. 
        # If we know state t at the close of day t, we earn the return of day t+1.
        df_sym['forward_return'] = returns_with_events[returns_with_events['symbol'] == sym]['log_return'].shift(-1)
        # We also need E_t
        df_sym['E_t'] = returns_with_events[returns_with_events['symbol'] == sym]['E_t']
        df_sym['symbol'] = sym
        state_dfs.append(df_sym)

full_data = pd.concat(state_dfs, ignore_index=True)
full_data.dropna(subset=['forward_return'], inplace=True)
print(f"Loaded {len(full_data)} trading days across {len(symbols)} stocks for backtesting.")"""),
    md_cell("## 3. Backtest Simulation"),
    code_cell("""# Strategy Rule: 
# Long (+1) if state == 2 (Bull)
# Avoid (0) if state == 0 (Bear) or 1 (Sideways)

def calc_metrics(returns_series):
    if len(returns_series) == 0:
        return {'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'cum_ret': 0}
        
    cum_ret = np.exp(returns_series.sum()) - 1
    
    # annualized sharpe (approx 252 trading days)
    mean_ret = returns_series.mean() * 252
    std_ret = returns_series.std() * np.sqrt(252)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    
    # max drawdown
    cum_series = np.exp(returns_series.cumsum())
    running_max = cum_series.cummax()
    drawdowns = cum_series / running_max - 1
    max_dd = drawdowns.min()
    
    # win rate
    win_rate = (returns_series > 0).mean()
    
    return {'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': win_rate, 'cum_ret': cum_ret}

full_data['base_pos'] = np.where(full_data['baseline_state'] == 2, 1, 0)
full_data['nh_pos'] = np.where(full_data['nh_state'] == 2, 1, 0)

full_data['base_strat_ret'] = full_data['base_pos'] * full_data['forward_return']
full_data['nh_strat_ret'] = full_data['nh_pos'] * full_data['forward_return']
full_data['bnh_ret'] = full_data['forward_return']  # Buy and hold benchmark

# Split by Event Window
event_mask = full_data['E_t'] == 1
non_event_mask = full_data['E_t'] == 0

metrics = {
    'Overall': {
        'BnH': calc_metrics(full_data['bnh_ret']),
        'Baseline HMM': calc_metrics(full_data['base_strat_ret']),
        'NH-HMM': calc_metrics(full_data['nh_strat_ret'])
    },
    'Event Window': {
        'BnH': calc_metrics(full_data.loc[event_mask, 'bnh_ret']),
        'Baseline HMM': calc_metrics(full_data.loc[event_mask, 'base_strat_ret']),
        'NH-HMM': calc_metrics(full_data.loc[event_mask, 'nh_strat_ret'])
    },
    'Non-Event Window': {
        'BnH': calc_metrics(full_data.loc[non_event_mask, 'bnh_ret']),
        'Baseline HMM': calc_metrics(full_data.loc[non_event_mask, 'base_strat_ret']),
        'NH-HMM': calc_metrics(full_data.loc[non_event_mask, 'nh_strat_ret'])
    }
}

rows = []
for window, strats in metrics.items():
    for strat_name, mets in strats.items():
        rows.append({
            'Window': window,
            'Strategy': strat_name,
            'Sharpe': mets['sharpe'],
            'Max Drawdown': mets['max_dd'],
            'Win Rate': mets['win_rate'],
            'Cumulative Return': mets['cum_ret']
        })

metrics_df = pd.DataFrame(rows)
display(metrics_df)
metrics_df.to_parquet('../data/backtest_results.parquet')
print("Saved backtest metrics to data/backtest_results.parquet")"""),
    md_cell("## 4. Visualization"),
    code_cell("""# Let's plot the cumulative returns overall
cum_bnh = np.exp(full_data.groupby('date')['bnh_ret'].sum().cumsum())
cum_base = np.exp(full_data.groupby('date')['base_strat_ret'].sum().cumsum())
cum_nh = np.exp(full_data.groupby('date')['nh_strat_ret'].sum().cumsum())

plt.figure(figsize=(12, 6))
plt.plot(cum_bnh.index, cum_bnh, label='Buy and Hold Benchmark', color='gray', alpha=0.6)
plt.plot(cum_base.index, cum_base, label='Baseline HMM Strategy', color='blue', alpha=0.8)
plt.plot(cum_nh.index, cum_nh, label='NH-HMM Strategy', color='green', linewidth=2)
plt.title('Overall Cumulative Strategy Returns (Sum across universe)')
plt.ylabel('Cumulative Return Multiplier')
plt.legend()
plt.tight_layout()
plt.show()""")
]

create_notebook('d:/Regime_Detection/notebooks/10_quantitative_validation.ipynb', cells)
