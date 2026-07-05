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
    md_cell("# Day 11: SHAP Feature Attribution & Layered Classifier\nWe train an XGBoost classifier over the regime states to produce directional signals, then interpret feature importance using SHAP to confirm the impact of `days_to_event`."),
    code_cell("""import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import joblib
from src.classifier import LayeredRegimeClassifier
import warnings
warnings.filterwarnings('ignore')"""),
    md_cell("## 1. Feature Engineering"),
    code_cell("""returns_with_events = pd.read_parquet('../data/returns_with_events.parquet')
symbols = [s for s in returns_with_events['symbol'].unique() if s != '^NSEI']

classifier = LayeredRegimeClassifier(random_state=42)
all_data = []

# Merge in the NH states and build features
for sym in symbols:
    state_file = f'../data/per_stock_states/{sym}.parquet'
    if os.path.exists(state_file):
        df_states = pd.read_parquet(state_file)
        df_stock = returns_with_events[returns_with_events['symbol'] == sym].copy()
        
        # Merge nh_state
        df_stock = pd.merge(df_stock, df_states[['date', 'nh_state', 'baseline_state']], on='date', how='inner')
        
        # Create features (adds momentum_20d, vol_ratio, target)
        df_feat = classifier.create_features(df_stock)
        all_data.append(df_feat)

full_df = pd.concat(all_data, ignore_index=True)
full_df = full_df.dropna(subset=['momentum_20d', 'vol_ratio', 'target', 'days_to_event'])
print(f"Total samples for training: {len(full_df)}")"""),
    md_cell("## 2. Train XGBoost Classifier"),
    code_cell("""# Split chronologically (last year as test set)
test_start = full_df['date'].max() - pd.DateOffset(days=365)
train_df = full_df[full_df['date'] < test_start]
test_df = full_df[full_df['date'] >= test_start]

print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

# Fit the classifier
classifier.fit(train_df, train_df['target'])

# Predict
test_df['xgb_pred'] = classifier.predict(test_df)

# Accuracy on test set
accuracy = (test_df['xgb_pred'] == test_df['target']).mean()
print(f"Test Set Accuracy (3-class): {accuracy*100:.1f}%")"""),
    md_cell("## 3. SHAP Feature Importance"),
    code_cell("""# We use the TreeExplainer from SHAP
explainer = shap.TreeExplainer(classifier.model)
X_test = test_df[classifier.features]
shap_values = explainer.shap_values(X_test)

# SHAP values for a multi-class model returns a list of arrays (one for each class).
# Class 2 is the 'Buy/Up' class.
shap.summary_plot(shap_values[2], X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance for BUY signal")
plt.show()"""),
    md_cell("## 4. Backtest the Layered Signal"),
    code_cell("""# Let's compare NH-HMM pure strategy vs XGBoost layered strategy on the TEST set
# Base Strategy: Long (+1) if NH-HMM == Bull (2), else Avoid (0)
# XGB Strategy: Long (+1) if XGB == Buy (2), Short (-1) if XGB == Avoid (0), else Flat (0)

# We use the true forward 1-day return for the backtest
test_df['fwd_1d'] = test_df.groupby('symbol')['log_return'].shift(-1)
test_df = test_df.dropna(subset=['fwd_1d'])

test_df['nh_pos'] = np.where(test_df['nh_state'] == 2, 1, 0)
test_df['xgb_pos'] = np.where(test_df['xgb_pred'] == 2, 1, np.where(test_df['xgb_pred'] == 0, -1, 0))

test_df['nh_ret'] = test_df['nh_pos'] * test_df['fwd_1d']
test_df['xgb_ret'] = test_df['xgb_pos'] * test_df['fwd_1d']
test_df['bnh_ret'] = test_df['fwd_1d']

# Cumulative returns sum across all stocks in the test set
cum_bnh = np.exp(test_df.groupby('date')['bnh_ret'].sum().cumsum())
cum_nh = np.exp(test_df.groupby('date')['nh_ret'].sum().cumsum())
cum_xgb = np.exp(test_df.groupby('date')['xgb_ret'].sum().cumsum())

plt.figure(figsize=(12, 6))
plt.plot(cum_bnh.index, cum_bnh, label='Buy and Hold', color='gray', alpha=0.6)
plt.plot(cum_nh.index, cum_nh, label='Pure NH-HMM (Regime Only)', color='blue', alpha=0.8)
plt.plot(cum_xgb.index, cum_xgb, label='XGBoost Layered (Regime + Features)', color='orange', linewidth=2)
plt.title('Out-of-Sample Backtest: Regime Only vs XGBoost Layered')
plt.ylabel('Cumulative Return Multiplier')
plt.legend()
plt.tight_layout()
plt.show()"""),
    md_cell("## 5. Save Artifacts for Streamlit"),
    code_cell("""joblib.dump(classifier, '../src/xgboost_classifier.pkl')
full_df.to_parquet('../data/xgb_features_full.parquet')
print("Saved XGBoost model to /src/ and features to /data/")""")
]

create_notebook('d:/Regime_Detection/notebooks/11_shap_feature_attribution.ipynb', cells)
