"""
step10_update_theta_local_maxent.py
-----------------------------------
Full and exact θ-update matching Theorem A & B.

This implementation is mathematically IDENTICAL to Step 8
except it uses the CURRENT θ, π, ψ to compute model expectations.

For each strategy k:

    μ_data[k]  = Σ_{t,l} r_{t,l}^{(k)} f_{t,l}
    μ_model[k] = Σ_{t,l} r̃_{t,l}^{(k)}(θ,π,ψ) f_{t,l}

    θ_k ← θ_k + η ( μ_data[k] - μ_model[k] - λ θ_k )

Everything is grouped by action, and normalization is over
(k,l) for each action — no exceptions.
"""

import numpy as np
import pandas as pd

from step8_joint_responsibilities import GATING_FEATURES


# ============================================================
# INTERNAL: model responsibilities r̃_{t,l}^{(k)}
# ============================================================

def _compute_model_joint_probs(fmap_df, thetas, pi, psi):
    """
    EXACT SAME DISTRIBUTION AS STEP 8, but using the current (θ, π, ψ)
    and WITHOUT using r_df (which is from previous E-step).

    Output:
        model_r_mat[k, i] giving r̃_{t,l}^{(k)} for row i.

    This MUST match Equation (E-step) exactly but with parameters fixed.
    """

    df = fmap_df.sort_values(["action_ts_ns", "lag"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    gating_cols  = GATING_FEATURES

    F = df[feature_cols].to_numpy()       # (N,D)
    H = df[gating_cols].to_numpy()        # (N,Dg)
    lags = df["lag"].to_numpy()           # (N,)
    row_ids = df["row_id"].to_numpy()
    action_ts = df["action_ts_ns"].to_numpy()

    N = len(df)
    K = len(thetas)
    D = F.shape[1]
    Dg = len(gating_cols)

    theta_mat = np.stack(thetas)          # (K,D)
    log_pi = np.log(pi + 1e-12)

    model_r_mat = np.zeros((K, N))        # (K,N)
    groups = df.groupby("action_ts_ns").groups

    # ---------------------------------------------------------
    # PROCESS EACH ACTION t
    # ---------------------------------------------------------
    for ts, idx in groups.items():
        idx = np.asarray(list(idx), int)
        F_g = F[idx]                      # (m,D)
        H_g = H[idx]                      # (m,Dg)
        lags_g = lags[idx]                # (m,)

        # gating feature = action-level vector h_t
        h_t = H_g[0]                      # (Dg,)
        logits = psi @ h_t                # (K,)
        log_omega = logits - np.log(np.exp(logits).sum())

        # log u_{t,l}^{(k)}
        m = len(idx)
        log_u = np.zeros((K, m))

        for k in range(K):
            score = F_g @ theta_mat[k]    # reward term
            score = np.clip(score, -50, 50)
            log_u[k] = log_pi[lags_g - 1] + log_omega[k] + score

        # NORMALIZE OVER ALL (k,l) FOR THIS ACTION — Theorem A/B
        mx = np.max(log_u)
        exps = np.exp(log_u - mx)
        Z = exps.sum()
        if Z <= 0:
            exps[:] = 1.0
            Z = exps.sum()
        probs = exps / Z                  # (K,m)

        for j, i in enumerate(idx):
            model_r_mat[:, i] = probs[:, j]

    return model_r_mat   # (K, N)



# ============================================================
# PUBLIC: θ UPDATE
# ============================================================

def update_thetas_local_maxent(
    fmap_df,
    r_df,
    thetas,
    pi,
    psi,
    learning_rate=1e-4,
    l2_reg=1e-2,
    clip_value=10.0
):
    """
    Perform full MaxEnt θ-update.

    EXACT GRADIENT:
        grad_k = μ_data[k] - μ_model[k] - λ θ_k

    with:
        μ_data[k]  = Σ_i r_{i,k} f_i
        μ_model[k] = Σ_i r̃_{i,k} f_i
    """

    df = fmap_df.sort_values(["action_ts_ns", "lag"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    F = df[feature_cols].to_numpy()
    N, D = F.shape
    K = len(thetas)

    # ------------------------------------------
    # Build matrix R_data[i,k] = r_i^(k)
    # ------------------------------------------
    R_data = np.zeros((N, K))
    for row in r_df.itertuples(index=False):
        # row.row_id corresponds 1-to-1 with df index
        R_data[row.row_id, row.strategy] = row.r

    # μ_data[k] = Σ_i R_data[i,k] * F[i]
    mu_data = R_data.T @ F               # (K,D)

    # ------------------------------------------
    # Compute model responsibilities r̃
    # ------------------------------------------
    model_r = _compute_model_joint_probs(df, thetas, pi, psi)   # (K,N)

    # μ_model[k] = Σ_i r̃_{i,k} f_i
    mu_model = model_r @ F               # (K,D)

    # ------------------------------------------
    # θ Update
    # ------------------------------------------
    new_thetas = []

    for k in range(K):
        grad_k = mu_data[k] - mu_model[k] - l2_reg * thetas[k]
        theta_new = thetas[k] + learning_rate * grad_k
        theta_new = np.clip(theta_new, -clip_value, clip_value)
        new_thetas.append(theta_new)

    return new_thetas
