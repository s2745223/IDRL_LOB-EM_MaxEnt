"""
step1_ingest.py
---------
Module for reading and aligning LOBSTER-formatted message and orderbook files
into a single, event-synchronized DataFrame suitable for further feature and
state construction.

Each row in the output corresponds to a LOBSTER event:
- Pre-event order book snapshot (up to N levels)
- Event metadata (type, direction, price, size, order ID)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
PRICE_SCALE = 10_000  # LOBSTER prices are stored as int(price * 10,000)

# ----------------------------------------------------------------------
# Data container
# ----------------------------------------------------------------------
@dataclass
class LOBSTERFiles:
    """Filepaths for one trading day's LOBSTER message and orderbook data."""
    message_path: str
    orderbook_path: str
    levels: int

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _message_colnames() -> List[str]:
    """Canonical LOBSTER message column names."""
    return ["time_s", "type", "order_id", "size", "price_int", "direction"]

def _orderbook_colnames(levels: int) -> List[str]:
    """Canonical LOBSTER orderbook column names given depth levels."""
    cols = []
    for L in range(1, levels + 1):
        cols += [f"ask_price_int_{L}", f"ask_size_{L}",
                 f"bid_price_int_{L}", f"bid_size_{L}"]
    return cols

def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add nanosecond and datetime timestamp columns derived from time_s."""
    df = df.copy()
    ts_ns = (df["time_s"].astype(float) * 1_000_000_000).round().astype("int64")
    df["ts_ns"] = ts_ns
    df["ts"] = pd.to_datetime(ts_ns, unit="ns", origin="1970-01-01")
    return df

def _add_float_prices(df: pd.DataFrame, levels: int) -> pd.DataFrame:
    """Convert integer prices to float-dollar prices."""
    df = df.copy()
    df["price"] = df["price_int"] / PRICE_SCALE
    for L in range(1, levels + 1):
        df[f"ask_price_{L}"] = df[f"ask_price_int_{L}"] / PRICE_SCALE
        df[f"bid_price_{L}"] = df[f"bid_price_int_{L}"] / PRICE_SCALE
    return df

# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def read_lobster(files: LOBSTERFiles) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read LOBSTER message and orderbook files.

    Returns:
        message_df: DataFrame with message data.
        orderbook_df: DataFrame with order book snapshots.
    """
    msg_cols = _message_colnames()
    ob_cols = _orderbook_colnames(files.levels)

    message_df = pd.read_csv(files.message_path, header=None, names=msg_cols)
    orderbook_df = pd.read_csv(files.orderbook_path, header=None, names=ob_cols)

    # Type coercion and cleanup
    numeric_cols = ["time_s", "type", "order_id", "size", "price_int", "direction"]
    message_df[numeric_cols] = message_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    for L in range(1, files.levels + 1):
        subset = [f"ask_price_int_{L}", f"ask_size_{L}", f"bid_price_int_{L}", f"bid_size_{L}"]
        orderbook_df[subset] = orderbook_df[subset].apply(pd.to_numeric, errors="coerce")

    return message_df, orderbook_df

def align_lobster(message_df: pd.DataFrame,
                  orderbook_df: pd.DataFrame,
                  levels: int) -> pd.DataFrame:
    """
    Validate and align message and order book data row-wise.

    Ensures both files have the same number of rows (as LOBSTER guarantees),
    appends timestamps and float-price columns, and returns a unified DataFrame.
    """
    if len(message_df) != len(orderbook_df):
        raise ValueError("Mismatch in row counts between message and orderbook files.")
    if not np.all(np.diff(message_df["time_s"].to_numpy()) >= 0):
        raise ValueError("time_s column is not monotonic in message file.")

    events_df = pd.concat([message_df.reset_index(drop=True),
                           orderbook_df.reset_index(drop=True)], axis=1)
    events_df = _add_time_columns(events_df)
    events_df = _add_float_prices(events_df, levels)

    # Column ordering
    base_cols = ["ts", "ts_ns", "time_s", "type", "direction", "size",
                 "order_id", "price_int", "price"]
    ob_cols = []
    for L in range(1, levels + 1):
        ob_cols += [f"ask_price_int_{L}", f"ask_price_{L}", f"ask_size_{L}",
                    f"bid_price_int_{L}", f"bid_price_{L}", f"bid_size_{L}"]
    ordered_cols = base_cols + ob_cols
    events_df = events_df[[c for c in ordered_cols if c in events_df.columns]]

    return events_df

def load_and_align(files: LOBSTERFiles) -> pd.DataFrame:
    """Wrapper to read, validate, and merge LOBSTER message + orderbook files."""
    message_df, orderbook_df = read_lobster(files)
    events_df = align_lobster(message_df, orderbook_df, files.levels)
    return events_df
