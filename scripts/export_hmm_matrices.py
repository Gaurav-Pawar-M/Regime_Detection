import joblib
import json
from pathlib import Path
import numpy as np

SRC_DIR = Path("src")
DATA_DIR = Path("data")

baseline_model = joblib.load(SRC_DIR / "baseline_model.pkl")
nh_model = joblib.load(SRC_DIR / "nh_model_nifty.pkl")

import pandas as pd
log_returns = pd.read_parquet(DATA_DIR / "log_returns.parquet")
nifty_returns = log_returns[['^NSEI']].dropna().values

baseline_ll = baseline_model.score(nifty_returns)

# E must be provided for nh_model. For NIFTY, there are no corporate events, so E is all zeros.
E = np.zeros(len(nifty_returns))
nh_ll = nh_model.score(nifty_returns, E)

metrics = {
    "baseline_ll": float(baseline_ll),
    "nh_ll": float(nh_ll),
    "A_normal": nh_model.A_normal_.tolist() if hasattr(nh_model, 'A_normal_') else None,
    "A_event": nh_model.A_event_.tolist() if hasattr(nh_model, 'A_event_') else None
}

with open(DATA_DIR / "hmm_metrics.json", "w") as f:
    json.dump(metrics, f)

print("Exported HMM metrics successfully.")
