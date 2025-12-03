"""
step17_behavioral_plots.py
--------------------------
Additional post-EM plots for behavioural interpretation:

1. Reward Weights per Strategy (theta bar plots)
2. Strategy Responsibilities Over Time (w_k(t))
3. Strategy × Action-Type Heatmap (P(action | strategy))
4. Strategy × Feature Conditioning Heatmap (E[f_i | strategy])

Inputs required:
    - r_df (responsibilities_r.csv)
    - fmap_df (feature_map.csv)
    - thetas (theta_k*.csv or thetas_all.csv)
    - action.csv for action_type field
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# LOAD ALL MATERIAL (r, fmap, theta, action)
# ============================================================

def load_files(fmap_file, r_file, theta_file, action_file):
    fmap = pd.read_csv(fmap_file)
    r_df = pd.read_csv(r_file)
    action_df = pd.read_csv(action_file)

    # rename for merge
    action_df.rename(columns={"ts_ns": "action_ts_ns"}, inplace=True)

    # add action column to fmap
    fmap = fmap.merge(action_df[["action_ts_ns", "action"]], 
                      on="action_ts_ns", how="left")

    # load theta (combined)
    thetas = pd.read_csv(theta_file)
    thetas = thetas.drop(columns=["strategy"]).values  # shape (K, D)

    # Join fmap & responsibilities
    merged = fmap.merge(
        r_df[["row_id", "strategy", "lag", "r"]],
        on=["row_id", "lag"],
        how="inner"
    )

    return merged, thetas



# ============================================================
# 1. Theta Bar Charts
# ============================================================

def plot_theta_bars(thetas, outdir):
    K, D = thetas.shape

    plt.figure(figsize=(16, 10))
    for k in range(K):
        plt.subplot(K, 1, k+1)
        plt.bar(range(D), thetas[k])
        plt.title(f"Reward Weights θ for Strategy {k}")
        plt.ylabel("θ value")
        plt.xlabel("Feature index")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "theta_weights_per_strategy.png"))
    plt.close()



# ============================================================
# 2. Strategy Responsibility Over Time
# ============================================================

def plot_strategy_over_time(df, outdir, K):
    """
    w_k(t) = sum_l r_{t,l}^{(k)}
    We use action_ts_ns as 'time'.
    """

    df_sorted = df.sort_values("action_ts_ns")

    rows = []
    for t_ns, grp in df_sorted.groupby("action_ts_ns"):
        for k in range(K):
            w_k = grp[grp["strategy"] == k]["r"].sum()
            rows.append({"action_ts_ns": t_ns, "strategy": k, "weight": w_k})

    w_df = pd.DataFrame(rows)

    plt.figure(figsize=(12, 6))
    for k in range(K):
        sub = w_df[w_df["strategy"] == k]
        plt.plot(sub["action_ts_ns"], sub["weight"], label=f"Strategy {k}")

    plt.title("Strategy Responsibility Over Time")
    plt.xlabel("Timestamp (ns)")
    plt.ylabel("Σₗ r_{t,l}^{(k)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "strategy_over_time.png"))
    plt.close()



# ============================================================
# 3. Strategy × Action-Type Heatmap
# ============================================================

def plot_strategy_action_heatmap(df, outdir, K):
    """
    Compute P(action | strategy).
    """

    rows = []

    for k in range(K):
        dfk = df[df["strategy"] == k]
        counts = dfk.groupby("action")["r"].sum()
        total = counts.sum()

        for a in counts.index:
            rows.append({
                "strategy": k,
                "action": a,
                "prob": counts[a] / total if total > 0 else 0
            })

    mat = pd.DataFrame(rows)

    pivot = mat.pivot(index="strategy", columns="action", values="prob")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm")
    plt.title("Strategy × Action-Type Probability Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "strategy_action_heatmap.png"))
    plt.close()



# ============================================================
# 4. Strategy-Conditioned Feature Means
# ============================================================

def plot_strategy_feature_heatmap(df, outdir, K):
    features = [c for c in df.columns if c.startswith("f_")]

    rows = []
    for k in range(K):
        dfk = df[df["strategy"] == k]
        for f in features:
            val = np.average(dfk[f], weights=dfk["r"]) if len(dfk) else 0
            rows.append({"strategy": k, "feature": f, "mean_value": val})

    mat = pd.DataFrame(rows)

    pivot = mat.pivot(index="strategy", columns="feature", values="mean_value")

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="RdBu_r", center=0)
    plt.title("Mean Feature Values Conditioned on Strategy")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "strategy_feature_conditioning.png"))
    plt.close()



# ============================================================
# MAIN FUNCTION
# ============================================================

def run_behavioral_plots(
    fmap_file,
    r_file,
    theta_file,
    action_file,
    output_dir,
    K
):
    os.makedirs(output_dir, exist_ok=True)

    print("📥 Loading all data for behavioral plots...")
    df, thetas = load_files(fmap_file, r_file, theta_file, action_file)

    print("🎨 Plotting θ per strategy...")
    plot_theta_bars(thetas, output_dir)

    print("📈 Plotting strategy evolution over time...")
    plot_strategy_over_time(df, output_dir, K)

    print("🔥 Plotting strategy–action heatmap...")
    plot_strategy_action_heatmap(df, output_dir, K)

    print("🌈 Plotting strategy-conditioned feature heatmap...")
    plot_strategy_feature_heatmap(df, output_dir, K)

    print("✅ Step 17 behavioral plots complete!")



run_behavioral_plots(
    fmap_file="fmap.csv",
    r_file="Output/responsibilities_r.csv",
    theta_file="Output/thetas_all.csv",
    action_file="actions.csv",
    output_dir="Output/behavioral_plots/",
    K=4
)
