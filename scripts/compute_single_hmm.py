import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.nonhomogeneous_hmm import NonHomogeneousGaussianHMM
from src.regime_utils import label_states_by_mean

sym = sys.argv[1]
DATA_DIR = Path("data")

returns_with_events = pd.read_parquet(DATA_DIR / 'returns_with_events.parquet')
stock_data = returns_with_events[returns_with_events['symbol'] == sym].copy()

X = np.column_stack([stock_data['log_return'].values, stock_data['volatility'].values])
E = stock_data['E_t'].values.astype(bool)

nh = NonHomogeneousGaussianHMM(n_components=3, n_iter=100, n_restarts=3, random_state=42)
nh.fit(X, E)

nh_labels_map, nh_std_map = label_states_by_mean(nh.means_, nh.covars_)

A_normal = nh.A_normal_
A_event = nh.A_event_

A_normal_perm = np.zeros((3, 3))
A_event_perm = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        std_i = nh_std_map[i]
        std_j = nh_std_map[j]
        A_normal_perm[std_i, std_j] = A_normal[i, j]
        A_event_perm[std_i, std_j] = A_event[i, j]

metrics = {
    "A_normal": A_normal_perm.tolist(),
    "A_event": A_event_perm.tolist()
}

with open(DATA_DIR / f"matrices_{sym}.json", "w") as f:
    json.dump(metrics, f)
