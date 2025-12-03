"""
feature_map.py
--------------
Step 6: Build feature vectors f(s,a) for each (A_t, S_{t-l}) pair.

Inputs (from Step 5 - pairs_df):
    - action_ts_ns
    - state_ts_ns
    - lag
    - action (categorical)
    - price_feature
    - size
    - <all state features>

Output:
    fmap_df:
        - action_ts_ns
        - state_ts_ns
        - lag
        - f_0, f_1, ..., f_{d-1}   (full feature vector f(s,a))
"""

import pandas as pd
import numpy as np

# These are your discrete action categories
ACTION_LIST = ["MB", "MS", "JQ_B", "JQ_A", "IQ_B", "IQ_A", "CR", "DN"]


def build_feature_map(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete feature map f(s,a) for each (A_t, S_{t-l}) pair.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Must include:
          - 'action_ts_ns', 'state_ts_ns', 'lag'
          - 'action'
          - 'price_feature', 'size'
          - state features (all numerical columns except the above)

    Returns
    -------
    fmap_df : pd.DataFrame
        Columns:
            - action_ts_ns
            - state_ts_ns
            - lag
            - f_0, f_1, ..., f_{d-1}
    """

    # Make a copy
    df = pairs_df.copy()

    # ----------------------------------------------------------
    # 1. One-hot encode the action label (MB/MS/JQ_B/...)
    # ----------------------------------------------------------
    for act in ACTION_LIST:
        df[f"act_{act}"] = (df["action"] == act).astype(float)

    # ----------------------------------------------------------
    # 2. Identify action numeric features
    # ----------------------------------------------------------
    action_numeric_cols = ["price_feature", "size"]

    # ----------------------------------------------------------
    # 3. Identify state feature columns
    # ----------------------------------------------------------
    skip_cols = {
        "action_ts_ns",
        "state_ts_ns",
        "lag",
        "action",
        "price_feature",
        "size",
    }.union({f"act_{a}" for a in ACTION_LIST})

    state_feature_cols = [c for c in df.columns if c not in skip_cols]

    # ----------------------------------------------------------
    # 4. Build the full feature vector f(s,a)
    # ----------------------------------------------------------
    f_cols = (
        [f"act_{a}" for a in ACTION_LIST] +  # action one-hot
        action_numeric_cols +               # action numeric features
        state_feature_cols                  # state features
    )

    f_matrix = df[f_cols].to_numpy(dtype=float)

    # ----------------------------------------------------------
    # 5. Construct final output DataFrame
    # ----------------------------------------------------------
    
    fmap_df = pd.DataFrame({
        "action_ts_ns": df["action_ts_ns"].astype("int64"),
        "state_ts_ns": df["state_ts_ns"].astype("int64"),
        "lag": df["lag"].astype(int)
    })

    for j in range(f_matrix.shape[1]):
        fmap_df[f"f_{j}"] = f_matrix[:, j]

    # 🔹 add a stable row id for EM (0..N-1)
    fmap_df = fmap_df.reset_index(drop=True)
    fmap_df["row_id"] = np.arange(len(fmap_df), dtype=int)

    # put row_id first for convenience
    cols = ["row_id", "action_ts_ns", "state_ts_ns", "lag"] + [c for c in fmap_df.columns if c.startswith("f_")]
    fmap_df = fmap_df[cols]

    return fmap_df