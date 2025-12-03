"""
step14_delay_analysis.py
-------------------------
Analyze:
    1) Overall delay distribution P(l)
    2) Strategy-specific delay profiles P(l | k)

Inputs:
    gamma_df : DataFrame with ['row_id', 'action_ts_ns', 'lag', 'gamma']
    r_df     : DataFrame with ['row_id', 'action_ts_ns', 'lag', 'strategy', 'r']
    K        : number of strategies
    n        : number of delays

Outputs:
    overall_delay : DataFrame with ['lag', 'p_delay']
    per_strategy_delay : DataFrame with ['strategy', 'lag', 'p_delay_given_strategy']
"""

import pandas as pd
import numpy as np


def compute_overall_delay_distribution(gamma_df: pd.DataFrame, n: int):
    """
    Compute P(lag = l) = ( Σ_t γ_{t,l} )  /  Σ_{t,l} γ_{t,l}
    """
    delay_sums = (
        gamma_df.groupby("lag")["gamma"]
        .sum()
        .reindex(range(1, n + 1), fill_value=0)
        .reset_index()
    )
    delay_sums.columns = ["lag", "total_gamma"]

    total = delay_sums["total_gamma"].sum()
    if total <= 0:
        delay_sums["p_delay"] = 1.0 / n
    else:
        delay_sums["p_delay"] = delay_sums["total_gamma"] / total

    return delay_sums[["lag", "p_delay"]]


def compute_strategy_delay_profiles(r_df: pd.DataFrame, K: int, n: int):
    """
    Compute P(l | strategy = k)

    For each strategy k:
        numerator   = Σ_t r_{t,l}^{(k)}
        denominator = Σ_{t,l} r_{t,l}^{(k)}
    """

    # sum over row_id for each (strategy, lag)
    sums = (
        r_df.groupby(["strategy", "lag"])["r"]
        .sum()
        .reset_index()
    )

    # Ensure all combinations (k, l) exist
    idx = pd.MultiIndex.from_product(
        [range(K), range(1, n + 1)],
        names=["strategy", "lag"]
    )
    sums = sums.set_index(["strategy", "lag"]).reindex(idx, fill_value=0).reset_index()

    # Normalize per strategy
    denom = sums.groupby("strategy")["r"].transform("sum")
    sums["p_delay_given_strategy"] = np.where(
        denom > 0,
        sums["r"] / denom,
        1.0 / n
    )

    return sums[["strategy", "lag", "p_delay_given_strategy"]]


def analyze_delays(gamma_df: pd.DataFrame, r_df: pd.DataFrame, K: int, n: int):
    """
    Full Step 14 pipeline:
      - Compute overall delay P(l)
      - Compute per-strategy delay P(l | k)
    """
    overall_delay = compute_overall_delay_distribution(gamma_df, n)
    per_strategy_delay = compute_strategy_delay_profiles(r_df, K, n)
    return overall_delay, per_strategy_delay

