"""
step12_save_thetas.py
---------------------
Save reward parameter vectors θ^(k) for each strategy k.

Format:
    output_dir/
        theta_k0.csv
        theta_k1.csv
        ...
        theta_k(K-1).csv
        thetas_all.csv   (combined file for convenience)
"""

import os
import numpy as np
import pandas as pd
from typing import List


def save_thetas(
    thetas: List[np.ndarray],
    output_dir: str,
):
    """
    Save θ^(k) for all strategies.

    Parameters
    ----------
    thetas : list of np.ndarray
        thetas[k] is θ^(k), shape (D,)
    output_dir : str
        Folder where theta files will be written

    Outputs
    -------
    Saves:
        theta_k{i}.csv   (one per strategy)
        thetas_all.csv   (stacked version, K × D)
    """

    os.makedirs(output_dir, exist_ok=True)

    # Save each θ^(k) separately
    for k, theta_k in enumerate(thetas):
        df_k = pd.DataFrame(theta_k, columns=["theta"])
        file_k = os.path.join(output_dir, f"theta_k{k}.csv")
        df_k.to_csv(file_k, index=False)

    # Save a combined matrix file
    theta_stack = np.vstack(thetas)  # shape (K, D)
    df_all = pd.DataFrame(theta_stack)
    df_all.columns = [f"f_{i}" for i in range(theta_stack.shape[1])]
    df_all.insert(0, "strategy", list(range(len(thetas))))

    combined_file = os.path.join(output_dir, "thetas_all.csv")
    df_all.to_csv(combined_file, index=False)

    print(f"  ✔ Saved θ vectors to: {output_dir}")
