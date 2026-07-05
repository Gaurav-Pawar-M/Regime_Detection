import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class LayeredRegimeClassifier:
    def __init__(self, random_state=42):
        # We predict a 3-class target: 0 (Avoid/Down), 1 (Hold/Neutral), 2 (Buy/Up)
        self.model = XGBClassifier(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.05, 
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        self.features = ['nh_state', 'momentum_20d', 'vol_ratio', 'days_to_event']

    def create_features(self, df):
        """
        Calculates features required for XGBoost on a per-stock basis.
        Assumes df is sorted by date and belongs to a single stock.
        """
        df = df.copy()
        
        # 1. NH-HMM state (already present as nh_state)
        # 2. Momentum 20-day (sum of log returns over last 20 days)
        df['momentum_20d'] = df['log_return'].rolling(window=20).sum()
        
        # 3. Volatility ratio (5-day std dev of log return / 20-day std dev of log return)
        vol_5 = df['log_return'].rolling(window=5).std()
        vol_20 = df['log_return'].rolling(window=20).std()
        df['vol_ratio'] = vol_5 / (vol_20 + 1e-6)
        
        # 4. Target variable: Forward 5-day return
        df['fwd_5d_ret'] = df['log_return'].shift(-5).rolling(window=5).sum().shift(5) # actually just shift(-5).rolling(5) is backwards, we want future 5 days:
        # Proper way to get forward 5 day sum:
        df['fwd_5d_ret'] = df['log_return'].iloc[::-1].rolling(window=5).sum().iloc[::-1].shift(-1)
        
        # Create Classes: Buy (2) if return > 1%, Avoid (0) if return < -1%, else Hold (1)
        df['target'] = 1
        df.loc[df['fwd_5d_ret'] > 0.01, 'target'] = 2
        df.loc[df['fwd_5d_ret'] < -0.01, 'target'] = 0
        
        return df

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self

    def predict(self, X):
        return self.model.predict(X[self.features])
        
    def predict_proba(self, X):
        return self.model.predict_proba(X[self.features])
