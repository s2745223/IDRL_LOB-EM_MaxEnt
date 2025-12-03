# THIS IS THE CORRECT STEP 8 IMPLEMENTATION (MATCHES THEOREM B EXACTLY)

import numpy as np
import pandas as pd

GATING_FEATURES = ["f_0", "f_5", "f_10", "f_17"]


def compute_joint_responsibilities(fmap_df, thetas, pi, psi):
    fmap_df = fmap_df.sort_values(["action_ts_ns", "lag"]).reset_index(drop=True)

    feature_cols = [c for c in fmap_df.columns if c.startswith("f_")]
    gating_cols = GATING_FEATURES

    F = fmap_df[feature_cols].to_numpy()     # (N, D)
    H = fmap_df[gating_cols].to_numpy()      # (N, Dg)

    lags = fmap_df["lag"].to_numpy()
    row_ids = fmap_df["row_id"].to_numpy()
    action_ts = fmap_df["action_ts_ns"].to_numpy()

    N = len(fmap_df)
    K = len(thetas)
    n = len(pi)

    D = F.shape[1]
    Dg = len(gating_cols)

    theta_mat = np.stack(thetas)             # (K, D)
    log_pi = np.log(pi + 1e-12)

    r_mat = np.zeros((K, N))
    groups = fmap_df.groupby("action_ts_ns").groups

    for ts, idx in groups.items():
        idx = np.asarray(list(idx), int)
        F_g = F[idx]                       # (m, D)
        H_g = H[idx]                       # (m, Dg)
        lags_g = lags[idx]                 # (m,)

        # gating uses ONE h_t (action-level)
        h_t = H_g[0]                       # (Dg,)
        logits_k = psi @ h_t               # (K,)
        log_omega_t = logits_k - np.log(np.exp(logits_k).sum())

        # log u_{k,l}
        m = len(idx)
        log_u = np.zeros((K, m))

        for k in range(K):
            score = F_g @ theta_mat[k]     # reward score
            score = np.clip(score, -50, 50)
            log_u[k] = log_pi[lags_g - 1] + log_omega_t[k] + score

        # normalise jointly over k,l
        mx = np.max(log_u)
        exps = np.exp(log_u - mx)
        Z = exps.sum()
        probs = exps / Z                   # (K, m)

        for j, i in enumerate(idx):
            r_mat[:, i] = probs[:, j]

    # gamma
    gamma = r_mat.sum(axis=0)

    # r_df
    r_rows = []
    for i in range(N):
        for k in range(K):
            r_rows.append({
                "row_id": int(row_ids[i]),
                "action_ts_ns": int(action_ts[i]),
                "lag": int(lags[i]),
                "strategy": int(k),
                "r": float(r_mat[k,i])
            })
    r_df = pd.DataFrame(r_rows)

    gamma_df = pd.DataFrame({
        "row_id": row_ids,
        "action_ts_ns": action_ts,
        "lag": lags,
        "gamma": gamma
    })

    return r_df, gamma_df
