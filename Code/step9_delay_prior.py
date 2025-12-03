"""
step9_delay_prior.py
--------------------
Proper M-step for delay prior π using joint responsibilities r_{t,l}^{(k)}.
"""

import numpy as np
import pandas as pd


def update_delay_prior_from_r(r_df: pd.DataFrame, n: int) -> np.ndarray:
    """
    Parameters
    ----------
    r_df : DataFrame
        Must contain:
            - 'lag'
            - 'r'  (responsibility r_{t,l}^{(k)})
    n : int
        Number of delay levels (max lag).

    Returns
    -------
    pi_new : np.ndarray
        Updated delay prior π, shape (n,), with pi_new[l-1] = π_l.
    """

    # Sum responsibilities over t,k for each lag ℓ
    lag_sums = (
        r_df.groupby("lag")["r"]
        .sum()
        .reindex(range(1, n + 1), fill_value=0.0)
        .to_numpy(dtype=float)
    )

    total = lag_sums.sum()
    if total <= 0:
        # degenerate case: fallback to uniform
        pi_new = np.ones(n, dtype=float) / n
    else:
        pi_new = lag_sums / total

    return pi_new
