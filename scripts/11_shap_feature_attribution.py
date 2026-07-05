"""
11_shap_feature_attribution.py
================================
Trains an XGBoost classifier to predict 5-day forward return direction,
using NH-HMM regime states + advanced technical indicators as features.
Produces SHAP explainability and saves model + feature tables.

Run: python scripts/11_shap_feature_attribution.py

Outputs:
  ../src/xgboost_classifier.pkl
  ../data/xgb_features_full.parquet
  ../data/shap_values_test.parquet
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SRC_DIR  = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score, f1_score)
from sklearn.model_selection import cross_val_score

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Run: pip install shap")

# ── Config ───────────────────────────────────────────────────────────
FORWARD_DAYS = 5          # predict 5-day return direction
SPLIT_DATE   = "2022-06-01"

# Full feature list — base + advanced
FEATURES_BASE = [
    'log_return', 'volatility',
    'lag_return_1', 'lag_return_2', 'lag_return_5',
    'momentum_5d', 'momentum_20d',
    'vol_ratio', 'lag_vol_1',
    'nh_state', 'E_t', 'days_to_event',
]

FEATURES_ADVANCED = [
    # RSI
    'rsi_14', 'rsi_7',
    # MACD
    'macd_hist', 'macd_cross',
    # Bollinger
    'bb_pct_b', 'bb_bandwidth', 'bb_squeeze',
    # ATR
    'atr_pct',
    # Stochastic
    'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought',
    # MA
    'dist_ma20', 'dist_ma50', 'ma_trend',
    # Momentum
    'momentum_60d', 'realised_vol_5d', 'vol_regime',
    # Market context
    'nifty_momentum_5d', 'nifty_momentum_20d',
    # Volume
    'volume_ratio', 'hl_range',
    # Gaps
    'overnight_gap',
]

ALL_FEATURES = FEATURES_BASE + FEATURES_ADVANCED

# ── Load Data ────────────────────────────────────────────────────────

def load_data():
    print("Loading data...")

    rwe = pd.read_parquet(DATA_DIR / "returns_with_events.parquet")
    rwe['date'] = pd.to_datetime(rwe['date'])

    # Load extra features if available
    extra_path = DATA_DIR / "extra_features.parquet"
    has_extra  = extra_path.exists()
    if has_extra:
        extra = pd.read_parquet(extra_path)
        print(f"  Extra features loaded: {extra.shape}")
    else:
        extra = None
        print("  WARNING: extra_features.parquet not found.")
        print("  Run scripts/01b_fetch_extra.py first for advanced features.")
        print("  Continuing with base features only...")

    # Load per-stock NH-HMM states
    per_stock_dir = DATA_DIR / "per_stock_states"
    states_cache  = {}
    if per_stock_dir.exists():
        for f in per_stock_dir.glob("*.parquet"):
            sym = f.stem
            df  = pd.read_parquet(f)
            df['date'] = pd.to_datetime(df['date'])
            states_cache[sym] = df

    print(f"  NH-HMM states loaded for {len(states_cache)} stocks")
    return rwe, extra, states_cache, has_extra


def build_features(rwe, extra, states_cache, has_extra):
    print("\nBuilding feature matrix...")
    symbols  = [s for s in rwe['symbol'].unique() if s != '^NSEI']
    frames   = []

    for sym in symbols:
        df = rwe[rwe['symbol'] == sym].copy()
        df = df.sort_values('date').reset_index(drop=True)

        # ── NH-HMM state ────────────────────────────────
        if sym in states_cache:
            states = states_cache[sym][['date', 'nh_state', 'nh_label']].copy()
            df = df.merge(states, on='date', how='left')
        else:
            df['nh_state'] = 1
            df['nh_label'] = 'Sideways/Neutral'

        # ── Base lag features (from returns_with_events) ─
        df['lag_return_1'] = df['log_return'].shift(1)
        df['lag_return_2'] = df['log_return'].shift(2)
        df['lag_return_5'] = df['log_return'].shift(5)
        df['momentum_5d']  = df['log_return'].rolling(5).sum().shift(1)
        df['momentum_20d'] = df['log_return'].rolling(20).sum().shift(1)
        df['vol_ratio']    = df['volatility'] / df['volatility'].rolling(20).mean()
        df['lag_vol_1']    = df['volatility'].shift(1)

        # ── Advanced features from extra_features.parquet ─
        if has_extra and (sym, 'rsi_14') in extra.columns:
            sym_extra = extra[sym].copy()
            sym_extra.index.name = 'date'
            sym_extra = sym_extra.reset_index()
            sym_extra['date'] = pd.to_datetime(sym_extra['date'])
            sym_extra = sym_extra.drop(columns=['lag_return_1', 'lag_return_2', 'lag_return_5', 'momentum_5d', 'momentum_20d', 'lag_vol_1', 'vol_ratio'], errors='ignore')
            df = df.merge(sym_extra, on='date', how='left')
        elif has_extra:
            # Try flat column access (sym is top-level key)
            try:
                sym_extra = extra[sym].copy()
                sym_extra.index.name = 'date'
                sym_extra = sym_extra.reset_index()
                sym_extra['date'] = pd.to_datetime(sym_extra['date'])
                sym_extra = sym_extra.drop(columns=['lag_return_1', 'lag_return_2', 'lag_return_5', 'momentum_5d', 'momentum_20d', 'lag_vol_1', 'vol_ratio'], errors='ignore')
                df = df.merge(sym_extra, on='date', how='left')
            except Exception:
                pass

        # ── Target: 5-day forward return direction ───────
        # Will cumulative return over next FORWARD_DAYS be positive?
        future_return = df['log_return'].shift(-1).rolling(FORWARD_DAYS).sum().shift(-(FORWARD_DAYS - 1))
        df['target'] = (future_return > 0).astype(int)

        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)

    # Determine which features are actually available
    available = [f for f in ALL_FEATURES if f in full_df.columns]
    missing   = [f for f in ALL_FEATURES if f not in full_df.columns]

    if missing:
        print(f"  Features not available (run 01b_fetch_extra.py): {missing[:5]}{'...' if len(missing)>5 else ''}")

    full_df = full_df.dropna(subset=available + ['target'])

    print(f"  Total rows: {len(full_df)}")
    print(f"  Features available: {len(available)}/{len(ALL_FEATURES)}")
    print(f"  Target distribution: {full_df['target'].value_counts(normalize=True).round(3).to_dict()}")

    return full_df, available


# ── Train/Test Split ─────────────────────────────────────────────────

def split_data(full_df, features):
    split = pd.Timestamp(SPLIT_DATE)
    train = full_df[full_df['date'] < split]
    test  = full_df[full_df['date'] >= split]

    X_train = train[features].astype(float)
    y_train = train['target']
    X_test  = test[features].astype(float)
    y_test  = test['target']

    print(f"\nTrain: {len(train):,} rows ({train['date'].min().date()} -> {train['date'].max().date()})")
    print(f"Test:  {len(test):,} rows  ({test['date'].min().date()} -> {test['date'].max().date()})")

    return X_train, y_train, X_test, y_test, train, test


# ── Model ────────────────────────────────────────────────────────────

def build_model():
    """
    XGBoost with:
    - Lower max_depth=3 to prevent memorising noise
    - Higher n_estimators with early stopping
    - L1 (reg_alpha) + L2 (reg_lambda) regularisation
    - Lower learning_rate for more robust convergence
    - min_child_weight=15 to prevent splits on tiny event subsets
    """
    return XGBClassifier(
        n_estimators      = 1000,      # high — early stopping will cut it
        max_depth         = 3,         # shallow trees = less overfitting
        learning_rate     = 0.01,      # slow learning = more robust
        subsample         = 0.75,      # row subsampling
        colsample_bytree  = 0.75,      # feature subsampling
        colsample_bylevel = 0.75,      # per-level feature subsampling
        min_child_weight  = 15,        # minimum samples per leaf
        gamma             = 0.1,       # minimum loss reduction for split
        reg_alpha         = 0.5,       # L1 — kills irrelevant features
        reg_lambda        = 2.0,       # L2 — shrinks all weights
        scale_pos_weight  = 1.0,       # class balance (set if imbalanced)
        random_state      = 42,
        eval_metric       = 'logloss',
        early_stopping_rounds = 30,    # stop if no improvement for 30 rounds
        n_jobs            = -1,
    )


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, features):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, preds)
    auc  = roc_auc_score(y_test, probs)
    f1   = f1_score(y_test, preds)

    print("\n" + "=" * 50)
    print(f"TEST SET RESULTS ({FORWARD_DAYS}-day forward prediction)")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Best iter: {model.best_iteration}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['Down/Flat', 'Up']))

    print("\nTop 15 Features by Importance:")
    imp = pd.Series(model.feature_importances_, index=features)
    imp = imp.sort_values(ascending=False)
    for feat, val in imp.head(15).items():
        bar = '#' * int(val * 200)
        print(f"  {feat:25s}: {val:.4f}  {bar}")

    return acc, auc, f1


# ── SHAP ─────────────────────────────────────────────────────────────

def compute_shap(model, X_test, features):
    if not HAS_SHAP:
        print("\nSHAP not available — skipping.")
        return None

    print("\nComputing SHAP values (this may take 1-2 minutes)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary
    print("\nMean |SHAP| per feature (global importance):")
    mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=features)
    mean_shap = mean_shap.sort_values(ascending=False)
    for feat, val in mean_shap.head(10).items():
        bar = '#' * int(val * 500)
        print(f"  {feat:25s}: {val:.4f}  {bar}")

    return shap_values


# ── Save ─────────────────────────────────────────────────────────────

def save_outputs(model, full_df, features, shap_values, X_test):
    print("\nSaving outputs...")

    # 1. Model
    joblib.dump(model, SRC_DIR / "xgboost_classifier.pkl")
    print(f"  Saved: {SRC_DIR / 'xgboost_classifier.pkl'}")

    # 2. Full feature table with predictions
    X_full = full_df[features].astype(float)
    full_df = full_df.copy()
    full_df['xgb_prob_up']    = model.predict_proba(X_full)[:, 1]
    full_df['xgb_pred']       = model.predict(X_full)
    full_df['xgb_signal']     = pd.cut(
        full_df['xgb_prob_up'],
        bins=[0, 0.42, 0.58, 1.0],
        labels=['SELL', 'HOLD', 'BUY']
    )

    save_cols = ['date', 'symbol', 'target', 'xgb_prob_up', 'xgb_pred', 'xgb_signal'] + features
    save_cols = [c for c in save_cols if c in full_df.columns]
    full_df[save_cols].to_parquet(DATA_DIR / "xgb_features_full.parquet")
    print(f"  Saved: {DATA_DIR / 'xgb_features_full.parquet'}")

    # 3. SHAP values
    if shap_values is not None:
        shap_df = pd.DataFrame(shap_values, columns=features)
        shap_df.to_parquet(DATA_DIR / "shap_values_test.parquet")
        print(f"  Saved: {DATA_DIR / 'shap_values_test.parquet'}")

    # 4. Save feature list (so app.py always uses the right features)
    import json
    with open(SRC_DIR / "xgb_features.json", "w") as f:
        json.dump(features, f)
    print(f"  Saved: {SRC_DIR / 'xgb_features.json'}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"11_shap_feature_attribution.py")
    print(f"Target: {FORWARD_DAYS}-day forward return direction")
    print(f"Split: {SPLIT_DATE}")
    print("=" * 60)

    # Load
    rwe, extra, states_cache, has_extra = load_data()

    # Build features
    full_df, features = build_features(rwe, extra, states_cache, has_extra)

    # Split
    X_train, y_train, X_test, y_test, train, test = split_data(full_df, features)

    # Train
    print("\nTraining XGBoost...")
    model = build_model()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100
    )

    # Evaluate
    acc, auc, f1 = evaluate(model, X_test, y_test, features)

    # SHAP
    shap_values = compute_shap(model, X_test, features)

    # Save
    save_outputs(model, full_df, features, shap_values, X_test)

    # Final verdict
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if acc > 0.55:
        print(f"✓ STRONG: {acc:.1%} accuracy (>55% threshold)")
    elif acc > 0.52:
        print(f"~ MODERATE: {acc:.1%} accuracy (52-55% — acceptable for finance)")
    else:
        print(f"✗ WEAK: {acc:.1%} accuracy — consider more features or longer horizon")

    print(f"ROC-AUC: {auc:.4f} (>0.55 = model has genuine signal)")
    print(f"Model uses {len(features)} features, best at iteration {model.best_iteration}")
    print("=" * 60)


if __name__ == "__main__":
    main()
