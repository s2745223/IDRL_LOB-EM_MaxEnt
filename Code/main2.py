"""
main_em.py
==========
EM–MaxEnt IRDL with Mixture-of-Experts Gating (Theorem A & B)

Pipeline (Steps 6–15):
    Step 6   – Load feature_map.csv
    Step 7   – Initialise parameters (θ, π, ψ)
    Step 8   – E-step: joint responsibilities r_{t,l}^{(k)} (with gating)
    Step 9   – M-step: update delay prior π
    Step 10a – M-step: update θ (reward parameters, local MaxEnt)
    Step 10b – M-step: update ψ (gating parameters)
    Step 11  – Convergence check
    Step 12  – Save θ (and ψ)
    Step 13  – Save r and γ
    Step 14  – Delay analysis
    Step 15  – Plots of delays & strategies
"""

import os
import numpy as np
import pandas as pd

# ====== Step imports ======
from step8_joint_responsibilities import compute_joint_responsibilities, GATING_FEATURES  # Step 8
from step9_delay_prior import update_delay_prior_from_r                                   # Step 9
from step10_update_theta_local_maxent import update_thetas_local_maxent                   # Step 10a
from step10_update_psi_gating import update_psi_gating                                    # Step 10b
from step11_convergence import check_em_convergence                                       # Step 11
from step12_save_thetas import save_thetas                                               # Step 12
from step13_save_responsibilities import save_responsibilities                           # Step 13
from step14_delay_analysis import analyze_delays                                          # Step 14
from step15_strategy_delay_plots import plot_strategy_delay_profiles                      # Step 15


# ============================================================
# Helper: Save ψ gating parameters
# ============================================================

def save_psi(psi: np.ndarray, output_dir: str) -> None:
    """
    Save gating parameters ψ to CSV.

    psi shape: (K, Dg), where Dg = len(GATING_FEATURES).
    """
    K, Dg = psi.shape
    df = pd.DataFrame(psi, columns=[f"h_{j}" for j in range(Dg)])
    df.insert(0, "strategy", np.arange(K, dtype=int))
    out_path = os.path.join(output_dir, "psi_gating.csv")
    df.to_csv(out_path, index=False)


# ============================================================
# Main EM Pipeline
# ============================================================

def run_em_pipeline(
    feature_map_file: str,
    output_dir: str,
    K: int = 4,
    n: int = 5,
    max_iter: int = 50,
    learning_rate_theta: float = 1e-4,
    learning_rate_psi: float = 1e-5,
    l2_reg_theta: float = 1e-2,
    l2_reg_psi: float = 1e-2,
    tol: float = 1e-4,
) -> None:
    """
    Execute the full EM–MaxEnt IRDL algorithm with mixture-of-experts gating.

    Parameters
    ----------
    feature_map_file : str
        Path to feature_map.csv produced by earlier steps.
        Must contain:
            - 'row_id'
            - 'action_ts_ns'
            - 'state_ts_ns'
            - 'lag'
            - f_0 ... f_{D-1}
    output_dir : str
        Directory where all EM results (θ, ψ, r, γ, delay profiles, plots) are saved.
    K : int
        Number of latent strategies (clusters).
    n : int
        Number of delay levels (max lag).
    max_iter : int
        Maximum number of EM iterations.
    learning_rate_theta : float
        Gradient ascent step size for θ update.
    learning_rate_psi : float
        Gradient ascent step size for ψ update (gating).
    l2_reg_theta : float
        L2 regularisation coefficient for θ.
    l2_reg_psi : float
        L2 regularisation coefficient for ψ.
    tol : float
        Convergence tolerance on (θ, π, ψ).
    """

    os.makedirs(output_dir, exist_ok=True)

    # =======================================================
    # STEP 6 – Load feature map (sorted)
    # =======================================================
    print("\n📥 Loading feature map...")
    fmap_df = pd.read_csv(feature_map_file)
    # Sort to ensure row_id, lag, and action order are consistent everywhere
    fmap_df = fmap_df.sort_values(["action_ts_ns", "lag"]).reset_index(drop=True)

    feature_cols = [c for c in fmap_df.columns if c.startswith("f_")]
    D = len(feature_cols)
    Dg = len(GATING_FEATURES)

    if Dg == 0:
        raise ValueError("GATING_FEATURES is empty – cannot define gating ψ.")

    print(f"  -> Loaded {len(fmap_df)} pair-rows, D={D} features, Dg={Dg} gating dims.")

    # =======================================================
    # STEP 7 – Initialise parameters (θ, π, ψ)
    # =======================================================
    print("🔧 Initializing parameters (θ, π, ψ)...")

    # θ^(k) ~ N(0, 0.1^2)
    thetas = [np.random.normal(0.0, 0.1, size=D) for _ in range(K)]

    # Delay prior π: start uniform over lags
    pi = np.ones(n, dtype=float) / float(n)

    # Gating parameters ψ: shape (K, Dg)
    psi = np.random.normal(0.0, 0.1, size=(K, Dg))

    # For convergence checking
    prev_thetas = None
    prev_pi = None
    prev_psi = None
    prev_r_df = None

    # =======================================================
    # EM ITERATIONS
    # =======================================================
    for it in range(1, max_iter + 1):
        print("\n============================")
        print(f"🚀 EM Iteration {it}/{max_iter}")
        print("============================")

        # ----------------------------------------------
        # STEP 8 – E-step: joint responsibilities r, γ
        # ----------------------------------------------
        print("  Step 8: Joint responsibilities r (with gating ψ)...")
        r_df, gamma_df = compute_joint_responsibilities(
            fmap_df=fmap_df,
            thetas=thetas,
            pi=pi,
            psi=psi,
        )

        # ----------------------------------------------
        # STEP 9 – Delay prior update π
        # ----------------------------------------------
        print("  Step 9: Updating delay prior π...")
        pi_new = update_delay_prior_from_r(r_df, n=n)

        # ----------------------------------------------
        # STEP 10a – θ update (local MaxEnt)
        # ----------------------------------------------
        print("  Step 10a: Updating θ (reward parameters)...")
        thetas_new = update_thetas_local_maxent(
            fmap_df=fmap_df,
            r_df=r_df,
            thetas=thetas,
            pi=pi_new,
            psi=psi,
            learning_rate=learning_rate_theta,
            l2_reg=l2_reg_theta,
            clip_value=10.0,
        )

        # ----------------------------------------------
        # STEP 10b – ψ update (gating)
        # ----------------------------------------------
        print("  Step 10b: Updating ψ (gating parameters)...")
        psi_new = update_psi_gating(
            fmap_df=fmap_df,
            r_df=r_df,
            psi=psi,
            learning_rate=learning_rate_psi,
            l2_reg=l2_reg_psi,
            clip_value=3.0,
        )

        # ----------------------------------------------
        # STEP 11 – Convergence check
        # ----------------------------------------------
        print("  Step 11: Checking convergence...")
        if it > 1:
            diagnostics = check_em_convergence(
                thetas_old=prev_thetas,
                thetas_new=thetas_new,
                pi_old=prev_pi,
                pi_new=pi_new,
                psi_old=prev_psi,
                psi_new=psi_new,
                r_old=prev_r_df,
                r_new=r_df,
            )
            print("  Diagnostics:", diagnostics)

            if (
                diagnostics["delta_theta"] < tol
                and diagnostics["delta_pi"]   < tol
                and diagnostics["delta_psi"]  < tol
            ):
                print("🎉 Converged by parameter deltas.")
                thetas = thetas_new
                pi = pi_new
                psi = psi_new
                break

        # Update previous values for next iteration
        prev_thetas = thetas
        prev_pi = pi
        prev_psi = psi
        prev_r_df = r_df

        thetas = thetas_new
        pi = pi_new
        psi = psi_new

    # =======================================================
    # STEP 12 – Save θ and ψ
    # =======================================================
    print("\n📦 Saving θ (Step 12a)...")
    save_thetas(thetas, output_dir)

    print("📦 Saving ψ gating parameters (Step 12b)...")
    save_psi(psi, output_dir)

    # =======================================================
    # STEP 13 – Save responsibilities r and γ
    # =======================================================
    print("📦 Saving r and γ (Step 13)...")
    save_responsibilities(r_df, gamma_df, output_dir)

    # =======================================================
    # STEP 14 – Delay analysis
    # =======================================================
    print("📊 Running delay analysis (Step 14)...")
    overall_delay, per_strategy_delay = analyze_delays(
        gamma_df=gamma_df,
        r_df=r_df,
        K=K,
        n=n,
    )

    overall_delay.to_csv(os.path.join(output_dir, "overall_delay.csv"), index=False)
    per_strategy_delay.to_csv(
        os.path.join(output_dir, "per_strategy_delay.csv"), index=False
    )

    # =======================================================
    # STEP 15 – Plots
    # =======================================================
    print("📈 Generating plots (Step 15)...")
    plot_strategy_delay_profiles(
        overall_delay=overall_delay,
        per_strategy_delay=per_strategy_delay,
        output_dir=output_dir,
    )

    print("\n✨ EM–MaxEnt IRDL with gating completed successfully!\n")


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run EM–MaxEnt IRDL with mixture-of-experts gating (Theorem B)"
    )
    parser.add_argument(
        "--fmap",
        type=str,
        required=True,
        help="Path to feature_map.csv (Step 6 output)",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for EM results",
    )
    parser.add_argument(
        "--strategies",
        type=int,
        default=4,
        help="Number of latent strategies K",
    )
    parser.add_argument(
        "--lags",
        type=int,
        default=5,
        help="Number of delay levels n",
    )

    args = parser.parse_args()

    run_em_pipeline(
        feature_map_file=args.fmap,
        output_dir=args.out,
        K=args.strategies,
        n=args.lags,
    )
