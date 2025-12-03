"""
step11_convergence.py
---------------------

Convergence diagnostics for EM with gating.

Computes:
    - delta_theta
    - delta_pi
    - delta_psi
    - delta_r   (optional)
    - delta_ll  (not used for now, set to 0)

Returned as a dict for user logging.
"""

import numpy as np
import pandas as pd


def _theta_distance(t_old, t_new):
    """Compute max L2 distance across θ vectors."""
    dists = []
    for o, n in zip(t_old, t_new):
        d = np.linalg.norm(n - o)
        dists.append(d)
    return float(np.max(dists))


def _psi_distance(psi_old, psi_new):
    """Compute L2 distance for gating parameters."""
    return float(np.linalg.norm(psi_new - psi_old))


def _pi_distance(pi_old, pi_new):
    """L2 distance for delay priors."""
    return float(np.linalg.norm(pi_new - pi_old))


def _r_distance(r_old_df, r_new_df):
    """Average absolute difference in responsibilities."""
    # align rows
    merged = r_old_df.merge(
        r_new_df, 
        on=["row_id", "strategy", "lag"],
        suffixes=("_old", "_new")
    )
    diff = np.abs(merged["r_old"] - merged["r_new"]).mean()
    return float(diff)


def check_em_convergence(
    thetas_old,
    thetas_new,
    pi_old,
    pi_new,
    psi_old,
    psi_new,
    r_old,
    r_new,
):
    """
    Compute convergence deltas between old and new EM parameters.
    """

    # ============= θ-distance ============
    delta_theta = _theta_distance(thetas_old, thetas_new)

    # ============= π-distance ============
    delta_pi = _pi_distance(pi_old, pi_new)

    # ============= ψ-distance ============
    if psi_old is None:
        delta_psi = float("inf")
    else:
        delta_psi = _psi_distance(psi_old, psi_new)

    # ============= r-distance ============
    if r_old is None:
        delta_r = float("inf")
    else:
        delta_r = _r_distance(r_old, r_new)

    # log-likelihood (optional)
    delta_ll = 0.0

    return {
        "delta_theta": delta_theta,
        "delta_pi": delta_pi,
        "delta_psi": delta_psi,
        "delta_r": delta_r,
        "delta_ll": delta_ll,
    }
