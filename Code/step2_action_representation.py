"""
action_representation.py
------------------------
Step 2 (final): Map LOBSTER events to discrete actions
and compute minimal numeric features for EM–MaxEnt IRL.
"""

import pandas as pd
import numpy as np

def build_actions(events_df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    """
    Convert aligned LOBSTER events to discrete actions with minimal numeric info.

    Parameters
    ----------
    events_df : pd.DataFrame
        Must include ['type','direction','price','bid_price_1','ask_price_1','size','ts_ns'].
    levels : int
        Depth levels available (unused, kept for API consistency).

    Returns
    -------
    actions_df : pd.DataFrame
        Columns: ['ts_ns','action','price_feature','size']
    """
    df = events_df.copy()

    # ------------------------------------------------------------------
    # 1. Midprice & spread
    # ------------------------------------------------------------------
    df["midprice"] = (df["ask_price_1"] + df["bid_price_1"]) / 2
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    df["direction"] = df["direction"].fillna(0)

    # ------------------------------------------------------------------
    # 2. Discrete action mapping
    # ------------------------------------------------------------------
    def classify(row):
        t = int(row["type"])
        side = "B" if row["direction"] > 0 else "A"

        # Market executions
        if t in (4, 5):
            return "MB" if side == "B" else "MS"

        # New limit order
        elif t == 1:
            if side == "B":
                if row["price"] > row["bid_price_1"]:
                    return "IQ_B"   # inside quote bid (price improving)
                elif np.isclose(row["price"], row["bid_price_1"]):
                    return "JQ_B"   # join queue bid
                else:
                    return "JQ_B"
            else:
                if row["price"] < row["ask_price_1"]:
                    return "IQ_A"   # inside quote ask (price improving)
                elif np.isclose(row["price"], row["ask_price_1"]):
                    return "JQ_A"
                else:
                    return "JQ_A"

        # Cancellation / deletion
        elif t in (2, 3):
            return "CR"

        # Everything else
        else:
            return "DN"

    df["action"] = df.apply(classify, axis=1)

    # ------------------------------------------------------------------
    # 3. Relative price feature (normalized by spread)
    # ------------------------------------------------------------------
    df["price_feature"] = (df["price"] - df["midprice"]) / df["spread"].replace(0, np.nan)

    # ------------------------------------------------------------------
    # 4. Keep minimal columns
    # ------------------------------------------------------------------
    actions_df = df[["ts_ns", "action", "price_feature", "size"]].copy()

    # Clean invalid values
    actions_df = actions_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["price_feature"])

    return actions_df
