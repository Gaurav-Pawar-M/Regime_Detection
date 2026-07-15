import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.nonhomogeneous_hmm import NonHomogeneousGaussianHMM
from src.regime_utils import label_states_by_mean

DATA_DIR = Path("data")

returns_with_events = pd.read_parquet(DATA_DIR / 'returns_with_events.parquet')
symbols = sorted(returns_with_events['symbol'].unique())
if 'TATAMOTORS.NS' in symbols:
    symbols.remove('TATAMOTORS.NS')

all_A_normal = []
all_A_event = []

print(f"Fitting NH-HMM for {len(symbols)} stocks to extract matrices...")

for sym in symbols:
    stock_data = returns_with_events[returns_with_events['symbol'] == sym].copy()
    X = np.column_stack([stock_data['log_return'].values, stock_data['volatility'].values])
    E = stock_data['E_t'].values.astype(bool)
    
    # 3. Fit NH-HMM
    nh = NonHomogeneousGaussianHMM(n_components=3, n_iter=100, n_restarts=3, random_state=42)
    nh.fit(X, E)
    
    # 4. Standardize Labels
    nh_labels_map, nh_std_map = label_states_by_mean(nh.means_, nh.covars_)
    # nh_std_map: dict mapping model_state -> standardized_state (0: Bear, 1: Sideways, 2: Bull)
    
    A_normal = nh.A_normal_
    A_event = nh.A_event_
    
    # Permute matrices to standardized order
    A_normal_perm = np.zeros((3, 3))
    A_event_perm = np.zeros((3, 3))
    
    for i in range(3):
        for j in range(3):
            std_i = nh_std_map[i]
            std_j = nh_std_map[j]
            A_normal_perm[std_i, std_j] = A_normal[i, j]
            A_event_perm[std_i, std_j] = A_event[i, j]
            
    all_A_normal.append(A_normal_perm)
    all_A_event.append(A_event_perm)
    
    # Save individual stock matrices to disk
    with open(DATA_DIR / f"matrices_{sym}.json", "w") as f:
        json.dump({
            "A_normal": A_normal_perm.tolist(),
            "A_event": A_event_perm.tolist()
        }, f)
        
    print(f"Fitted and saved {sym}")

# Average across all stocks
avg_A_normal = np.mean(all_A_normal, axis=0)
avg_A_event = np.mean(all_A_event, axis=0)

print("\n--- RESULTS ---")
print("Average A_normal:")
print(np.round(avg_A_normal, 4))
print("\nAverage A_event:")
print(np.round(avg_A_event, 4))

fro_norm = np.linalg.norm(avg_A_event - avg_A_normal, 'fro')
print(f"\nFrobenius Norm: {fro_norm:.4f}")

# Bear transition probabilities
print("\nTransitions into Bear (State 0):")
print(f"From Neutral (State 1) Normal: {avg_A_normal[1, 0]:.4f} -> Event: {avg_A_event[1, 0]:.4f}")
print(f"From Bull (State 2) Normal: {avg_A_normal[2, 0]:.4f} -> Event: {avg_A_event[2, 0]:.4f}")

metrics = {
    "A_normal_avg": avg_A_normal.tolist(),
    "A_event_avg": avg_A_event.tolist(),
    "frobenius_norm": float(fro_norm)
}

with open(DATA_DIR / "hmm_metrics_per_stock_avg.json", "w") as f:
    json.dump(metrics, f)
print("\nSaved to data/hmm_metrics_per_stock_avg.json")
