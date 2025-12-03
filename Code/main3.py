"""
paper_plots_1_to_7.py
=====================

Generates research-paper-ready plots for:

1. Global delay distribution p(lag)
2. Strategy-specific delay profiles p(lag | k)
3. Expected time delay per strategy (in ms)
4. θ-weight profiles per strategy (feature importance)
5. Action-type profiles per strategy p(action | k)
6. Strategy × delay joint heatmap
7. Feature–delay (Δt) correlation

Inputs
------
- feature_map.csv:
    row_id, action_ts_ns, state_ts_ns, lag, f_0, ..., f_18
- responsibilities_r.csv:
    row_id, action_ts_ns, lag, strategy, r
- actions.csv:
    ts_ns, action, price_feature, size
- theta_k0.csv, theta_k1.csv, ...:
    single column: theta

Outputs (PNG)
-------------
- global_delay_distribution.png
- strategy_delay_profiles.png
- expected_delay_per_strategy_ms.png
- theta_weights_per_strategy.png
- strategy_action_profile_heatmap.png
- strategy_lag_heatmap.png
- feature_delay_correlation.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Feature name mapping (as given)
# ------------------------------------------------------
FEATURE_NAME_MAP = {
    "f_0":  "MB",
    "f_1":  "MS",
    "f_2":  "JQ_B",
    "f_3":  "JQ_A",
    "f_4":  "IQ_B",
    "f_5":  "IQ_A",
    "f_6":  "CR",
    "f_7":  "DN",
    "f_8":  "price_feature",
    "f_9":  "size",
    "f_10": "CVI_Depth1",
    "f_11": "CVI_Depth5",
    "f_12": "Spread",
    "f_13": "Microprice",
    "f_14": "WAP_Diff_Depth5",
    "f_15": "Imb1",
    "f_16": "Imb5",
    "f_17": "Recent_Signed_Trade_Volume",
    "f_18": "Recent_LiqAbs_Cancel_Imb",
}

ACTION_ORDER = ["MB", "MS", "JQ_B", "JQ_A", "IQ_B", "IQ_A", "CR", "DN"]


# ------------------------------------------------------
# Utility plotting helpers
# ------------------------------------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_global_delay_distribution(r_df: pd.DataFrame,
                                   out_path: str):
    """
    1. Global delay distribution p(lag).
    """
    lag_mass = r_df.groupby("lag")["r"].sum()
    lag_mass = lag_mass.sort_index()
    p_delay = lag_mass / lag_mass.sum()

    plt.figure(figsize=(6, 4))
    plt.bar(p_delay.index.astype(int), p_delay.values)
    plt.xlabel("Lag (number of ticks back)")
    plt.ylabel("Probability p(lag)")
    plt.title("Global Inferred Delay Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_strategy_delay_profiles(r_df: pd.DataFrame,
                                 out_path: str):
    """
    2. Strategy-specific delay profiles p(lag | k).
    Also used as basis for heatmap.
    """
    grp = r_df.groupby(["strategy", "lag"])["r"].sum().reset_index()

    # Normalize within each strategy
    grp["p"] = grp.groupby("strategy")["r"].transform(
        lambda x: x / x.sum()
    )

    strategies = sorted(grp["strategy"].unique())
    lags = sorted(grp["lag"].unique())

    plt.figure(figsize=(7, 5))
    for k in strategies:
        sub = grp[grp["strategy"] == k].set_index("lag").reindex(lags, fill_value=0.0)
        plt.plot(lags, sub["p"].values, marker="o", label=f"Strategy {k}")

    plt.xlabel("Lag")
    plt.ylabel("p(lag | strategy)")
    plt.title("Strategy-specific Delay Profiles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return grp  # useful later for heatmap


def plot_expected_delay_per_strategy_ms(fmap_df: pd.DataFrame,
                                        r_df: pd.DataFrame,
                                        out_path: str):
    """
    3. Expected time delay per strategy (ms).
    """
    # Compute Δt per row_id (in ms)
    delays = fmap_df[["row_id", "action_ts_ns", "state_ts_ns"]].drop_duplicates()
    delays["delta_ms"] = (delays["action_ts_ns"] - delays["state_ts_ns"]) / 1e6

    # Merge into r_df
    r_with_dt = r_df.merge(delays, on="row_id", how="left")

    # E[Δt | k] = sum_i r_{i,k} * Δt_i / sum_i r_{i,k}
    grp = r_with_dt.groupby("strategy").apply(
        lambda df: np.average(df["delta_ms"], weights=df["r"])
    )
    strategies = grp.index.to_numpy()
    mean_delta_ms = grp.values

    plt.figure(figsize=(6, 4))
    plt.bar(strategies, mean_delta_ms)
    plt.xlabel("Strategy")
    plt.ylabel("Expected delay E[Δt | strategy] (ms)")
    plt.title("Expected Time Delay per Strategy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_theta_weights_per_strategy(theta_dir: str,
                                    feature_cols: list,
                                    K: int,
                                    out_path: str):
    """
    4. θ-weight profiles per strategy.
       Visualised as heatmap (strategies × features).
    """
    # Build matrix of shape (K, D)
    thetas = []
    for k in range(K):
        theta_file = os.path.join(theta_dir, f"theta_k{k}.csv")
        df_theta = pd.read_csv(theta_file)
        theta_vec = df_theta["theta"].to_numpy()
        thetas.append(theta_vec)

    theta_mat = np.vstack(thetas)  # (K, D)

    # Map feature names
    pretty_feature_names = [FEATURE_NAME_MAP.get(f, f) for f in feature_cols]

    plt.figure(figsize=(max(8, len(feature_cols) * 0.6), 4 + K * 0.3))
    plt.imshow(theta_mat, aspect="auto")
    plt.colorbar(label="θ weight")

    plt.yticks(range(K), [f"Strategy {k}" for k in range(K)])
    plt.xticks(range(len(feature_cols)), pretty_feature_names, rotation=60, ha="right")
    plt.title("θ Weights per Strategy (Feature Importance)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_strategy_action_profile_heatmap(fmap_df: pd.DataFrame,
                                         r_df: pd.DataFrame,
                                         actions_df: pd.DataFrame,
                                         out_path: str):
    """
    5. Action-type profiles per strategy p(action | k),
       shown as a heatmap (strategy × action-type).
    """
    # Merge action type into r_df via action_ts_ns
    # actions.csv: ts_ns, action, ...
    actions_df = actions_df.rename(columns={"ts_ns": "action_ts_ns"})
    r_with_action = r_df.merge(
        actions_df[["action_ts_ns", "action"]],
        on="action_ts_ns",
        how="left"
    )

    # Aggregate mass per (strategy, action)
    mass = (
        r_with_action.groupby(["strategy", "action"])["r"]
        .sum()
        .reset_index()
    )

    # Ensure consistent action ordering
    strategies = sorted(mass["strategy"].unique())
    # If some action types not present, include them with zero
    action_types = ACTION_ORDER

    # Build matrix (K, A)
    K = len(strategies)
    A = len(action_types)
    mat = np.zeros((K, A), dtype=float)

    for i, k in enumerate(strategies):
        sub = mass[mass["strategy"] == k]
        # Normalize within strategy to get p(action | k)
        total = sub["r"].sum()
        if total > 0:
            sub["p"] = sub["r"] / total
        else:
            sub["p"] = 0.0
        for j, a in enumerate(action_types):
            val = sub.loc[sub["action"] == a, "p"]
            if not val.empty:
                mat[i, j] = val.iloc[0]

    plt.figure(figsize=(8, 4 + 0.3 * K))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="p(action | strategy)")

    plt.yticks(range(K), [f"Strategy {k}" for k in strategies])
    plt.xticks(range(A), action_types, rotation=45, ha="right")
    plt.title("Strategy-specific Action Profiles")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_strategy_lag_heatmap(strategy_delay_df: pd.DataFrame,
                              out_path: str):
    """
    6. Strategy × delay heatmap using p(lag | k).
       Uses the dataframe returned by plot_strategy_delay_profiles().
    """
    # strategy_delay_df has columns: strategy, lag, r, p
    strategies = sorted(strategy_delay_df["strategy"].unique())
    lags = sorted(strategy_delay_df["lag"].unique())

    # Build matrix (K, L) of p(lag | k)
    K = len(strategies)
    L = len(lags)
    mat = np.zeros((K, L), dtype=float)

    for i, k in enumerate(strategies):
        sub = strategy_delay_df[strategy_delay_df["strategy"] == k]
        sub = sub.set_index("lag").reindex(lags, fill_value=0.0)
        mat[i, :] = sub["p"].values

    plt.figure(figsize=(6 + 0.4 * L, 4 + 0.3 * K))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="p(lag | strategy)")
    plt.yticks(range(K), [f"Strategy {k}" for k in strategies])
    plt.xticks(range(L), lags)
    plt.xlabel("Lag")
    plt.title("Strategy × Delay Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_feature_delay_correlation(fmap_df: pd.DataFrame,
                                   feature_cols: list,
                                   out_path: str):
    """
    7. Feature–delay correlation using Δt in ms.

    For each feature f_j, compute Pearson correlation between f_j and Δt.
    """
    fmap_df = fmap_df.copy()

    # Δt in ms
    fmap_df["delta_ms"] = (fmap_df["action_ts_ns"] - fmap_df["state_ts_ns"]) / 1e6
    y = fmap_df["delta_ms"].to_numpy()

    corrs = []
    names = []

    for f in feature_cols:
        x = fmap_df[f].to_numpy()
        if np.std(x) == 0 or np.std(y) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(x, y)[0, 1]
        corrs.append(corr)
        names.append(FEATURE_NAME_MAP.get(f, f))

    # Sort by absolute correlation for nicer plotting
    order = np.argsort(np.abs(corrs))[::-1]
    corrs_sorted = np.array(corrs)[order]
    names_sorted = np.array(names)[order]

    plt.figure(figsize=(max(8, len(feature_cols) * 0.5), 5))
    positions = np.arange(len(names_sorted))
    plt.bar(positions, corrs_sorted)
    plt.xticks(positions, names_sorted, rotation=60, ha="right")
    plt.ylabel("Correlation with Δt (ms)")
    plt.title("Feature–Delay Correlation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ------------------------------------------------------
# Main driver
# ------------------------------------------------------
def main(
    feature_map_file: str,
    responsibilities_r_file: str,
    actions_file: str,
    theta_dir: str,
    K: int,
    output_dir: str,
):
    _ensure_dir(output_dir)

    print("📥 Loading feature_map.csv ...")
    fmap_df = pd.read_csv(feature_map_file)
    # Ensure sorted for row_id consistency
    fmap_df = fmap_df.sort_values(["row_id", "lag"]).reset_index(drop=True)

    feature_cols = [c for c in fmap_df.columns if c.startswith("f_")]
    print(f"  -> Loaded {len(fmap_df)} rows, D={len(feature_cols)} features.")

    print("📥 Loading responsibilities_r.csv ...")
    r_df = pd.read_csv(responsibilities_r_file)
    # Ensure float type for r
    r_df["r"] = r_df["r"].astype(float)

    print("📥 Loading actions.csv ...")
    actions_df = pd.read_csv(actions_file)

    # 1. Global delay distribution
    print("📊 Plot 1: Global delay distribution p(lag)...")
    plot_global_delay_distribution(
        r_df=r_df,
        out_path=os.path.join(output_dir, "global_delay_distribution.png"),
    )

    # 2. Strategy-specific delay profiles
    print("📊 Plot 2: Strategy-specific delay profiles p(lag | k)...")
    strat_delay_df = plot_strategy_delay_profiles(
        r_df=r_df,
        out_path=os.path.join(output_dir, "strategy_delay_profiles.png"),
    )

    # 3. Expected time delay per strategy (ms)
    print("📊 Plot 3: Expected delay per strategy (ms)...")
    plot_expected_delay_per_strategy_ms(
        fmap_df=fmap_df,
        r_df=r_df,
        out_path=os.path.join(output_dir, "expected_delay_per_strategy_ms.png"),
    )

    # 4. θ weights per strategy
    print("📊 Plot 4: θ weights per strategy (feature importance)...")
    plot_theta_weights_per_strategy(
        theta_dir=theta_dir,
        feature_cols=feature_cols,
        K=K,
        out_path=os.path.join(output_dir, "theta_weights_per_strategy.png"),
    )

    # 5. Strategy × action profiles
    print("📊 Plot 5: Strategy-specific action profiles...")
    plot_strategy_action_profile_heatmap(
        fmap_df=fmap_df,
        r_df=r_df,
        actions_df=actions_df,
        out_path=os.path.join(output_dir, "strategy_action_profile_heatmap.png"),
    )

    # 6. Strategy × delay heatmap
    print("📊 Plot 6: Strategy × delay heatmap...")
    plot_strategy_lag_heatmap(
        strategy_delay_df=strat_delay_df,
        out_path=os.path.join(output_dir, "strategy_lag_heatmap.png"),
    )

    # 7. Feature–delay correlation
    print("📊 Plot 7: Feature–delay correlation...")
    plot_feature_delay_correlation(
        fmap_df=fmap_df,
        feature_cols=feature_cols,
        out_path=os.path.join(output_dir, "feature_delay_correlation.png"),
    )

    print("\n✅ All plots generated in:", output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate research-paper plots (1–7) for EM–MaxEnt IRDL."
    )
    parser.add_argument("--fmap", type=str, default="feature_map.csv",
                        help="Path to feature_map.csv")
    parser.add_argument("--r", type=str, default="responsibilities_r.csv",
                        help="Path to responsibilities_r.csv")
    parser.add_argument("--actions", type=str, default="actions.csv",
                        help="Path to actions.csv (ts_ns, action, ...)")
    parser.add_argument("--theta_dir", type=str, default=".",
                        help="Directory containing theta_k*.csv files")
    parser.add_argument("--K", type=int, default=4,
                        help="Number of strategies (K)")
    parser.add_argument("--out", type=str, default="paper_plots",
                        help="Output directory for PNG plots")

    args = parser.parse_args()

    main(
        feature_map_file=args.fmap,
        responsibilities_r_file=args.r,
        actions_file=args.actions,
        theta_dir=args.theta_dir,
        K=args.K,
        output_dir=args.out,
    )
