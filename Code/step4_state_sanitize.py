"""
state_sanitize.py
-----------------
Step 4: NaN handling and z-score normalization for state features.
Outputs to CSV if desired.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Tuple, Optional

def sanitize_and_standardize_states(states_df: pd.DataFrame,
                                    train_ratio: float = 0.8,
                                    scaler_out: Optional[str] = None
                                    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Clean NaNs / infs and z-score normalize all state features.

    Parameters
    ----------
    states_df : pd.DataFrame
        Output from Step 3. Must contain ts_ns and numeric feature columns.
    train_ratio : float, optional
        Portion of early rows used to compute normalization stats (default 0.8).
    scaler_out : str, optional
        Path to save scaler JSON (means and stds).

    Returns
    -------
    states_scaled_df : pd.DataFrame
        Cleaned, standardized state matrix.
    scaler : dict
        {"feature": {"mean": float, "std": float}, ...}
    """
    df = states_df.copy().reset_index(drop=True)
    n_train = int(len(df) * train_ratio)
    train_df = df.iloc[:n_train]

    feature_cols = [c for c in df.columns if c != "ts_ns"]

    # Replace inf/nan with column medians
    for col in feature_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan)
        med = series.median()
        df[col] = series.fillna(med)

    # Compute z-score stats
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std(ddof=0).replace(0, 1.0)
    scaler = {col: {"mean": float(means[col]), "std": float(stds[col])}
              for col in feature_cols}

    # Apply normalization
    df[feature_cols] = (df[feature_cols] - means) / stds

    # Save scaler JSON
    if scaler_out:
        with open(scaler_out, "w") as f:
            json.dump(scaler, f, indent=2)
        print(f"📁 Saved scaler parameters → {scaler_out}")

    return df, scaler
