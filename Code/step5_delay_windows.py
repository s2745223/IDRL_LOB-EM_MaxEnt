"""
delay_windows.py
----------------
Step 5 (long-form): Build (A_t, S_{t-l}) pairs for l = 1..n.

For each action at time t, we create n rows:
    (A_t, S_{t-1}, lag=1),
    (A_t, S_{t-2}, lag=2),
    ...,
    (A_t, S_{t-n}, lag=n).

This is the format you need to attach soft responsibilities γ_{t,l}
in the EM–MaxEnt IRL algorithm.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List

import pandas as pd
import numpy as np

def collapse_actions(actions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate ts_ns rows in actions_df:
    - action: take the FIRST action
    - price_feature: average over duplicates
    - size: sum of all sizes
    """

    # First action per timestamp
    first_actions = actions_df.sort_values("ts_ns").groupby("ts_ns")["action"].first()

    # Average price_feature per timestamp
    avg_price = actions_df.groupby("ts_ns")["price_feature"].mean()

    # Cumulative size per timestamp
    sum_size = actions_df.groupby("ts_ns")["size"].sum()

    # Combine
    out = pd.DataFrame({
        "ts_ns": first_actions.index,
        "action": first_actions.values,
        "price_feature": avg_price.values,
        "size": sum_size.values,
    })

    return out.sort_values("ts_ns").reset_index(drop=True)


def collapse_states(states_scaled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate ts_ns rows in states_scaled_df:
    - All state features are taken from the LAST row with that timestamp.
    """

    # Identify all state feature columns
    state_cols = [c for c in states_scaled_df.columns if c != "ts_ns"]

    # Take the *last* state for each timestamp
    last_states = (
        states_scaled_df
        .sort_values("ts_ns")
        .groupby("ts_ns")
        .last()
    )

    # Restore ts_ns as a column
    last_states = last_states.reset_index()[["ts_ns"] + state_cols]

    return last_states.sort_values("ts_ns").reset_index(drop=True)


def build_action_state_pairs(
    states_scaled_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """
    Construct long-form (A_t, S_{t-l}) pairs for l = 1..n.

    Parameters
    ----------
    states_scaled_df : pd.DataFrame
        Must contain 'ts_ns' and state feature columns (already z-scored).
        Row order is event-time (will be sorted here).

    actions_df : pd.DataFrame
        Must contain at least: 'ts_ns', 'action', 'price_feature', 'size'.

    n : int
        Lookback window size (number of previous states).
        Only actions with index >= n in the state stream are kept.

    Returns
    -------
    pairs_df : pd.DataFrame
        Long-form table with columns:
            - 'action_ts_ns'  (time of A_t)
            - 'state_ts_ns'   (time of S_{t-l})
            - 'lag'           (integer l in [1..n])
            - 'action', 'price_feature', 'size'
            - all state feature columns (same names as in states_scaled_df)
    """
    
    # ----------------------------------------------------------
    # 1) Sort both inputs by time
    # ----------------------------------------------------------
    sdf = states_scaled_df.sort_values("ts_ns").reset_index(drop=True)
    adf = actions_df.sort_values("ts_ns").reset_index(drop=True)

    # State feature names (everything except ts_ns)
    state_feats: List[str] = [c for c in sdf.columns if c != "ts_ns"]
    if len(state_feats) == 0:
        raise ValueError("states_scaled_df must have state feature columns besides 'ts_ns'.")

    # Add row index to states so we can map ts_ns -> position
    sdf = sdf.copy()
    sdf["row_idx"] = np.arange(len(sdf), dtype=int)
    
    # ----------------------------------------------------------
    # 2) Attach row index to each action
    # ----------------------------------------------------------
    adf = adf.merge(sdf[["ts_ns", "row_idx"]], on="ts_ns", how="inner")
    
    # Keep only actions that have at least n past states
    adf_valid = adf[adf["row_idx"] >= n].reset_index(drop=True)
    if adf_valid.empty:
        raise ValueError("No actions have at least n past states. Decrease n or check data.")
    
    action_indices = adf_valid["row_idx"].to_numpy(dtype=int)  # t-indices in state stream
    K = len(adf_valid)  # number of valid actions
    F = len(state_feats)  # number of state features

    # ----------------------------------------------------------
    # 3) Precompute state matrix and ts_ns
    # ----------------------------------------------------------
    state_matrix = sdf[state_feats].to_numpy(dtype=float)
    state_ts_ns = sdf["ts_ns"].to_numpy(dtype="int64")

    # ----------------------------------------------------------
    # 4) Build all (t, l) combinations as arrays
    # ----------------------------------------------------------
    # lags: 1..n
    lags = np.arange(1, n + 1, dtype=int)  # shape (n,)

    # For each action index t, we want previous indices t - l for l in [1..n]
    # action_indices shape: (K,)
    # state_indices shape: (K, n)
    state_indices = action_indices[:, None] - lags[None, :]  # broadcast subtraction

    # Flatten to 1D (K*n,)
    state_indices_flat = state_indices.reshape(-1)

    # -------------- Action side --------------
    # Repeat action information n times (once for each lag)
    action_ts_ns_flat = np.repeat(adf_valid["ts_ns"].to_numpy(dtype="int64"), n)
    action_label_flat = np.repeat(adf_valid["action"].to_numpy(dtype=object), n)
    price_feature_flat = np.repeat(adf_valid["price_feature"].to_numpy(dtype=float), n)
    size_flat = np.repeat(adf_valid["size"].to_numpy(dtype=float), n)

    # -------------- State side --------------
    # State timestamps for the lagged states
    state_ts_ns_flat = state_ts_ns[state_indices_flat]

    # State features: select rows via state_indices_flat
    # state_matrix[state_indices_flat] has shape (K*n, F)
    state_feats_matrix = state_matrix[state_indices_flat, :]

    # -------------- Lag column --------------
    lag_flat = np.tile(lags, K)  # [1,2,...,n, 1,2,...,n, ...] length = K*n

    # ----------------------------------------------------------
    # 5) Build final long-form DataFrame
    # ----------------------------------------------------------
    data = {
        "action_ts_ns": action_ts_ns_flat,
        "state_ts_ns": state_ts_ns_flat,
        "lag": lag_flat,
        "action": action_label_flat,
        "price_feature": price_feature_flat,
        "size": size_flat,
    }

    # Add each state feature column from the matrix
    for j, feat_name in enumerate(state_feats):
        data[feat_name] = state_feats_matrix[:, j]

    pairs_df = pd.DataFrame(data)

    # Ensure integer type for timestamps
    pairs_df["action_ts_ns"] = pairs_df["action_ts_ns"].astype("int64")
    pairs_df["state_ts_ns"] = pairs_df["state_ts_ns"].astype("int64")

    return pairs_df
