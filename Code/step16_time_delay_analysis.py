
"""
step16_time_delay_analysis.py
-----------------------------
Post-EM real-time delay analysis.

Computes and plots:

1. Actual Observed Time Differences:
        Δt = action_ts_ns - state_ts_ns

2. Strategy-specific Expected Time Delay:
        E[Δt | strategy]

3. Delay → Feature Sensitivity:
        Correlation between Δt and features f_*

4. Strategy × Delay Joint Heatmap:
        Mean Δt per (strategy k, lag ℓ)

5. Action-Type Delay Profiles:
        Mean Δt per action_type (MB, MS, JQ_B, IQ_A, CR, DN)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================================
# 0. Load + Merge All Files
# =====================================================================

def load_and_prepare(fmap_file, r_file, gamma_file, action_file):
    """
    Load all required files and merge into a single DataFrame.
    """

    fmap = pd.read_csv(fmap_file)
    r_df = pd.read_csv(r_file)
    gamma_df = pd.read_csv(gamma_file)
    action_df = pd.read_csv(action_file)

    # Rename action column
    action_df.rename(columns={"ts_ns": "action_ts_ns"}, inplace=True)

    # Merge action type into feature map
    fmap = fmap.merge(
        action_df[["action_ts_ns", "action"]],
        on="action_ts_ns",
        how="left"
    )

    # Merge responsibilities_r
    df = fmap.merge(
        r_df[["row_id", "strategy", "lag", "r"]],
        on=["row_id", "lag"],
        how="inner"
    )

    # Merge gamma
    df = df.merge(
        gamma_df[["row_id", "lag", "gamma"]],
        on=["row_id", "lag"],
        how="inner"
    )

    return df



# =====================================================================
# 1. Actual Δt = action_ts_ns - state_ts_ns
# =====================================================================

def compute_time_differences(df):
    df["delta_t_ns"] = df["action_ts_ns"] - df["state_ts_ns"]
    return df



# =====================================================================
# 2. Strategy-specific Expected Delay E[Δt | k]
# =====================================================================

def compute_strategy_expected_delay(df, K):
    out = []

    for k in range(K):
        sub = df[df["strategy"] == k]
        num = np.sum(sub["r"] * sub["delta_t_ns"])
        den = np.sum(sub["r"])
        exp_k = num / den if den > 0 else 0
        out.append({"strategy": k, "expected_delay_ns": exp_k})

    return pd.DataFrame(out)



# =====================================================================
# 3. Delay → Feature Sensitivity (Correlation)
# =====================================================================

def compute_feature_delay_correlation(df):
    features = [c for c in df.columns if c.startswith("f_")]
    rows = []

    for f in features:
        corr = df["delta_t_ns"].corr(df[f])
        rows.append({"feature": f, "corr": corr})

    corr_df = pd.DataFrame(rows)
    corr_df["abs_corr"] = corr_df["corr"].abs()
    return corr_df.sort_values("abs_corr", ascending=False)



# =====================================================================
# 4. Strategy × Lag Heatmap (Mean Δt)
# =====================================================================

def compute_strategy_lag_heatmap_data(df, K, n):
    rows = []
    for k in range(K):
        for l in range(1, n+1):
            sub = df[(df["strategy"] == k) & (df["lag"] == l)]
            if len(sub) == 0:
                mean_dt = 0
            else:
                mean_dt = np.average(sub["delta_t_ns"], weights=sub["r"])
            rows.append({"strategy": k, "lag": l, "mean_delta_t_ns": mean_dt})

    return pd.DataFrame(rows)



# =====================================================================
# 5. Action-Type Delay Profiles
# =====================================================================

def compute_action_type_delay(df):
    return (
        df.groupby("action")["delta_t_ns"]
        .mean()
        .reset_index()
        .sort_values("delta_t_ns")
    )



# =====================================================================
# Plotting Routines
# =====================================================================

def plot_strategy_expected_delay(df, outdir):
    plt.figure(figsize=(6,4))
    sns.barplot(x="strategy", y="expected_delay_ns", data=df)
    plt.title("Expected Time Delay per Strategy E[Δt | k]")
    plt.ylabel("Delay (ns)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "strategy_expected_delay.png"))
    plt.close()


def plot_feature_delay_correlation(df, outdir):
    plt.figure(figsize=(7,6))
    sns.barplot(data=df, x="corr", y="feature")
    plt.title("Feature–Delay Correlation")
    plt.xlabel("Correlation with Δt")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_delay_correlation.png"))
    plt.close()


def plot_strategy_lag_heatmap(df, outdir):
    pivot = df.pivot(index="strategy", columns="lag", values="mean_delta_t_ns")
    plt.figure(figsize=(8,4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    plt.title("Mean Δt (ns) for Strategy × Lag")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "strategy_lag_heatmap.png"))
    plt.close()


def plot_action_type_delay(df, outdir):
    plt.figure(figsize=(6,4))
    sns.barplot(x="action", y="delta_t_ns", data=df)
    plt.title("Mean Δt by Action Type")
    plt.ylabel("Delay (ns)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "action_type_delay.png"))
    plt.close()



# =====================================================================
# MAIN PIPELINE FUNCTION
# =====================================================================

def run_time_delay_analysis(
    fmap_file,
    r_file,
    gamma_file,
    action_file,
    output_dir,
    K,
    n
):
    os.makedirs(output_dir, exist_ok=True)

    print("📥 Loading & merging all data...")
    df = load_and_prepare(fmap_file, r_file, gamma_file, action_file)

    print("⏱ Computing Δt...")
    df = compute_time_differences(df)

    print("📊 Strategy Expected Delay...")
    sed = compute_strategy_expected_delay(df, K)
    sed.to_csv(os.path.join(output_dir, "expected_delay_per_strategy.csv"), index=False)
    plot_strategy_expected_delay(sed, output_dir)

    print("🔍 Delay–Feature Correlation...")
    corr = compute_feature_delay_correlation(df)
    corr.to_csv(os.path.join(output_dir, "delay_feature_correlation.csv"), index=False)
    plot_feature_delay_correlation(corr, output_dir)

    print("🌡 Strategy × Lag Δt Heatmap...")
    sl = compute_strategy_lag_heatmap_data(df, K, n)
    sl.to_csv(os.path.join(output_dir, "strategy_lag_mean_delay.csv"), index=False)
    plot_strategy_lag_heatmap(sl, output_dir)

    print("🎯 Action-Type Delay...")
    at = compute_action_type_delay(df)
    at.to_csv(os.path.join(output_dir, "action_type_delay.csv"), index=False)
    plot_action_type_delay(at, output_dir)

    print("✅ Step 16 Delay Analysis Complete!")




run_time_delay_analysis(
    fmap_file="fmap.csv",
    r_file="Output/responsibilities_r.csv",
    gamma_file="Output/responsibilities_gamma.csv",
    action_file="actions.csv",
    output_dir="Output/delay_analysis_output/",
    K=4,
    n=5
)
