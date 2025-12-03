"""
step10_update_psi_gating.py
---------------------------
Exact M-step for gating parameters ψ (Theorem B).

We treat gating as an action-level softmax regression:

    For each action t:
        h_t       = gating feature vector (from GATING_FEATURES)
        ω_t^(k)   = softmax( ψ_k^T h_t ) over k
        R_tk      = Σ_l r_{t,l}^{(k)}       (sum over delays)

Objective:
    L(ψ) = Σ_t Σ_k R_tk * log ω_t^(k)

Gradient:
    ∂L/∂ψ_k = Σ_t (R_tk - ω_tk) * h_t  - λ ψ_k

This code:
    - aggregates r over lags to get R_tk per action-strategy
    - computes ω_t using ψ and h_t
    - performs one gradient-ascent step on ψ
"""

import numpy as np
import pandas as pd

from step8_joint_responsibilities import GATING_FEATURES


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable softmax over the specified axis.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / s


def update_psi_gating(
    fmap_df: pd.DataFrame,
    r_df: pd.DataFrame,
    psi: np.ndarray,
    learning_rate: float = 1e-5,
    l2_reg: float = 1e-2,
    clip_value: float = 3.0,
) -> np.ndarray:
    """
    One M-step update for gating parameters ψ.

    Parameters
    ----------
    fmap_df : DataFrame
        Must contain:
            - 'action_ts_ns'
            - GATING_FEATURES (e.g., f_0, f_5, f_10, f_17)
        Can contain multiple rows per action (one per lag/state),
        but gating is action-level.
    r_df : DataFrame
        Responsibilities with columns:
            - 'action_ts_ns'
            - 'strategy'
            - 'lag'
            - 'r' (r_{t,l}^{(k)})
    psi : np.ndarray
        Current gating parameters, shape (K, Dg).
    learning_rate : float
        Gradient ascent step size.
    l2_reg : float
        L2 regularisation weight λ on ψ.
    clip_value : float or None
        If not None, ψ is clipped elementwise to [-clip_value, clip_value].

    Returns
    -------
    psi_new : np.ndarray
        Updated gating parameters, same shape as psi.
    """

    # Ensure consistent ordering by action
    df = fmap_df.sort_values(["action_ts_ns", "lag"]).reset_index(drop=True)
    gating_cols = GATING_FEATURES

    K, Dg = psi.shape

    # -------------------------------------------------------
    # 1) Build action-level gating features h_t
    # -------------------------------------------------------
    # Group rows by action, pick FIRST row's gating features as h_t
    action_groups = df.groupby("action_ts_ns")
    action_keys = list(action_groups.groups.keys())  # list of unique action_ts_ns
    T = len(action_keys)

    H_t = np.zeros((T, Dg), dtype=float)  # (T, Dg)

    for t_idx, ts in enumerate(action_keys):
        group_rows = action_groups.get_group(ts)
        h_vec = group_rows[gating_cols].iloc[0].to_numpy(dtype=float)
        H_t[t_idx] = h_vec

    # -------------------------------------------------------
    # 2) Aggregate responsibilities r_{t,l}^{(k)} → R_tk
    # -------------------------------------------------------
    # R_tk = Σ_l r_{t,l}^{(k)}  (sum over lags for each action-strategy)
    # r_df: columns ['row_id','action_ts_ns','lag','strategy','r']
    agg = (
        r_df
        .groupby(["action_ts_ns", "strategy"])["r"]
        .sum()
        .reset_index()
    )

    # Pivot to matrix with index=action, columns=strategy
    R_tk_df = agg.pivot(
        index="action_ts_ns",
        columns="strategy",
        values="r"
    )

    # Reindex to match action_keys order and fill missing with 0
    R_tk_df = R_tk_df.reindex(index=action_keys, fill_value=0.0)

    R = R_tk_df.to_numpy(dtype=float)  # shape (T, K)
    if R.shape[1] != K:
        raise ValueError(
            f"Responsibilities have {R.shape[1]} strategies, "
            f"but psi has K={K}"
        )

    # Sanity: each row's mass over k should be ≈1 (due to Step 8)
    # but we don't enforce; it's fine if slightly off numerically.

    # -------------------------------------------------------
    # 3) Compute gating probabilities ω_t^(k)
    # -------------------------------------------------------
    # logits_tk = ψ_k^T h_t
    logits = H_t @ psi.T          # (T, K)
    omega = _softmax(logits, axis=1)  # (T, K)

    # -------------------------------------------------------
    # 4) Gradient for ψ_k
    #     ∂L/∂ψ_k = Σ_t (R_tk - ω_tk) h_t - λ ψ_k
    # -------------------------------------------------------
    residuals = R - omega         # (T, K)
    psi_new = psi.copy()

    for k in range(K):
        # residuals[:, k] has shape (T,)
        # multiply each residual_tk by h_t (T, Dg) and sum over t
        grad_k = (residuals[:, k][:, None] * H_t).sum(axis=0) - l2_reg * psi[k]
        psi_new[k] += learning_rate * grad_k

    # Optional clipping for numerical stability
    if clip_value is not None:
        psi_new = np.clip(psi_new, -clip_value, clip_value)

    return psi_new
