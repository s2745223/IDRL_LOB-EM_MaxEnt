"""
step15_strategy_delay_plots.py
-------------------------------
Generate publication-grade plots for:
    1) Overall delay distribution
    2) Strategy-specific delay profiles (heatmap + bar plots)

Inputs:
    overall_delay: DataFrame ['lag', 'p_delay']
    per_strategy_delay: DataFrame ['strategy', 'lag', 'p_delay_given_strategy']
    output_dir: folder where plots will be saved

Outputs:
    plots/
        delay_overall.png
        delay_per_strategy.png
        delay_heatmap.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_strategy_delay_profiles(
    overall_delay: pd.DataFrame,
    per_strategy_delay: pd.DataFrame,
    output_dir: str
):
    """
    Produce Step 15 plots:
        - Overall delay probabilities
        - Delay heatmap (strategy × lag)
        - Per-strategy delay bar subplots

    Parameters
    ----------
    overall_delay : DataFrame
        Columns: ['lag', 'p_delay']

    per_strategy_delay : DataFrame
        Columns: ['strategy', 'lag', 'p_delay_given_strategy']

    output_dir : str
        Folder where plots will be stored.
    """

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ========================================
    # 1) OVERALL DELAY DISTRIBUTION
    # ========================================

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x="lag", y="p_delay",
        data=overall_delay,
        palette="Blues_d"
    )
    plt.title("Overall Delay Distribution P(lag)")
    plt.xlabel("Lag (ℓ)")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "delay_overall.png"), dpi=200)
    plt.close()

    # ========================================
    # 2) PER-STRATEGY DELAY HEATMAP
    # ========================================

    heat_df = per_strategy_delay.pivot(
        index="strategy",
        columns="lag",
        values="p_delay_given_strategy"
    )

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={'label': 'P(lag | strategy)'}
    )
    plt.title("Strategy-Specific Delay Profile Heatmap")
    plt.xlabel("Lag (ℓ)")
    plt.ylabel("Strategy")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "delay_heatmap.png"), dpi=200)
    plt.close()

    # ========================================
    # 3) PER-STRATEGY BAR PLOTS
    # ========================================

    K = per_strategy_delay["strategy"].nunique()
    n = per_strategy_delay["lag"].nunique()

    fig, axes = plt.subplots(K, 1, figsize=(6, 3*K), sharex=True)

    if K == 1:
        axes = [axes]  # ensure list

    for k in range(K):
        ax = axes[k]
        df_k = per_strategy_delay[per_strategy_delay["strategy"] == k]

        sns.barplot(
            x="lag",
            y="p_delay_given_strategy",
            data=df_k,
            palette="Purples",
            ax=ax
        )
        ax.set_title(f"Strategy {k} — Delay Distribution")
        ax.set_ylabel("P(lag | k)")
        ax.set_ylim(0, df_k["p_delay_given_strategy"].max() * 1.1)

    axes[-1].set_xlabel("Lag (ℓ)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "delay_per_strategy.png"), dpi=200)
    plt.close()

    print(f"  ✔ Step 15 plots saved to: {plot_dir}")
