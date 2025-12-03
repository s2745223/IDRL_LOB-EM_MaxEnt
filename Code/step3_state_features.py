"""
state_features.py
-----------------
Step 3: Construct state features φ(s_t) from aligned LOBSTER data.
"""

import pandas as pd
import numpy as np

def build_state_features(events_df: pd.DataFrame, levels: int = 5,
                         trade_window: int = 20) -> pd.DataFrame:
    """
    Compute core state features describing the order book around each event.

    Parameters
    ----------
    events_df : pd.DataFrame
        Output of Step 1 (aligned order book + events).
        Must include bid/ask prices and sizes up to given levels.
    levels : int, optional
        Number of LOB levels to include (default 5).
    trade_window : int, optional
        Rolling window (in events) for short-term flow features.

    Returns
    -------
    states_df : pd.DataFrame
        Columns: ['ts_ns','CVI_Depth1','CVI_Depth5','Spread','Microprice',
                  'WAP_Diff_Depth5','Imb1','Imb5',
                  'Recent_Signed_Trade_Volume','Recent_LiqAbs_Cancel_Imb']
    """

    df = events_df.copy()

    # --------------------------------------------------------------
    # Basic sanity
    # --------------------------------------------------------------
    for L in range(1, levels + 1):
        for col in [f"bid_price_{L}", f"ask_price_{L}", f"bid_size_{L}", f"ask_size_{L}"]:
            if col not in df.columns:
                raise ValueError(f"Missing column {col} in events_df")

    # --------------------------------------------------------------
    # Core LOB measures
    # --------------------------------------------------------------
    df["Spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["Microprice"] = (
        df["ask_price_1"] * df["bid_size_1"] +
        df["bid_price_1"] * df["ask_size_1"]
    ) / (df["ask_size_1"] + df["bid_size_1"]).replace(0, np.nan)

    # CVI Depth1 and Depth5
    df["CVI_Depth1"] = np.log1p(df["bid_size_1"] + df["ask_size_1"])

    bid_depth_sum = sum(df[f"bid_size_{l}"] for l in range(1, min(5, levels) + 1))
    ask_depth_sum = sum(df[f"ask_size_{l}"] for l in range(1, min(5, levels) + 1))
    df["CVI_Depth5"] = np.log1p(bid_depth_sum + ask_depth_sum)

    # Imbalances
    df["Imb1"] = (df["bid_size_1"] - df["ask_size_1"]) / \
                 (df["bid_size_1"] + df["ask_size_1"]).replace(0, np.nan)
    df["Imb5"] = (bid_depth_sum - ask_depth_sum) / \
                 (bid_depth_sum + ask_depth_sum).replace(0, np.nan)

    # WAP difference Depth5
    bid_wap = sum(df[f"bid_price_{l}"] * df[f"bid_size_{l}"] for l in range(1, min(5, levels) + 1)) / \
              bid_depth_sum.replace(0, np.nan)
    ask_wap = sum(df[f"ask_price_{l}"] * df[f"ask_size_{l}"] for l in range(1, min(5, levels) + 1)) / \
              ask_depth_sum.replace(0, np.nan)
    df["WAP_Diff_Depth5"] = ask_wap - bid_wap

    # --------------------------------------------------------------
    # Rolling / short-term flow features
    # --------------------------------------------------------------
    # Signed trade volume: direction * size for trade-type events
    trade_mask = df["type"].isin([4, 5])
    signed_vol = np.where(trade_mask, df["direction"] * df["size"], 0)
    df["Recent_Signed_Trade_Volume"] = (
        pd.Series(signed_vol).rolling(trade_window, min_periods=1).sum().to_numpy()
    )

    # Liquidity absorb/cancel imbalance
    add_mask = df["type"] == 1
    cancel_mask = df["type"].isin([2, 3])
    liq_delta = np.where(add_mask, df["size"], np.where(cancel_mask, -df["size"], 0))
    df["Recent_LiqAbs_Cancel_Imb"] = (
        pd.Series(liq_delta).rolling(trade_window, min_periods=1).sum().to_numpy()
    )

    # --------------------------------------------------------------
    # Select and clean
    # --------------------------------------------------------------
    cols = ["ts_ns","CVI_Depth1","CVI_Depth5","Spread","Microprice",
            "WAP_Diff_Depth5","Imb1","Imb5",
            "Recent_Signed_Trade_Volume","Recent_LiqAbs_Cancel_Imb"]

    states_df = df[cols].copy()
    states_df = states_df.replace([np.inf, -np.inf], np.nan).dropna()

    return states_df
