import numpy as np

def label_states_by_mean(means, covars):
    """
    Given means and covariances from an HMM, assigns standard labels:
    Lowest mean -> Bear/Crash
    Highest mean -> Bull/Calm
    Middle -> Sideways/Neutral
    
    Returns a dictionary mapping state index to label string,
    and a dictionary mapping state index to a standard integer:
    0: Bear
    1: Sideways
    2: Bull
    """
    # means is shape (N, n_features). We care about the first feature (returns).
    # If means is 1D, just use it.
    if means.ndim == 2:
        m = means[:, 0]
    else:
        m = means
        
    sorted_idx = np.argsort(m)
    bear_state = sorted_idx[0]
    sideways_state = sorted_idx[1]
    bull_state = sorted_idx[2]
    
    labels_map = {
        bear_state: "Bear",
        sideways_state: "Sideways",
        bull_state: "Bull"
    }
    
    std_map = {
        bear_state: 0,
        sideways_state: 1,
        bull_state: 2
    }
    
    return labels_map, std_map
