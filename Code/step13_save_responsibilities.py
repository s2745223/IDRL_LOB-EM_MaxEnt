"""
step13_save_responsibilities.py
-------------------------------
Save joint responsibilities r_{t,l}^{(k)} and delay responsibilities γ_{t,l}
after EM has converged.

Outputs:
    output_dir/
        responsibilities_r.csv
        responsibilities_gamma.csv
"""

import os
import pandas as pd


def save_responsibilities(
    r_df: pd.DataFrame,
    gamma_df: pd.DataFrame,
    output_dir: str,
):
    """
    Save responsibilities r and gamma.

    Parameters
    ----------
    r_df : DataFrame
        Joint responsibilities r_{t,l}^{(k)} for all rows.
        Required columns:
            - row_id
            - action_ts_ns
            - lag
            - strategy
            - r

    gamma_df : DataFrame
        Delay responsibilities γ_{t,l} = Σ_k r_{t,l}^{(k)}.
        Required columns:
            - row_id
            - action_ts_ns
            - lag
            - gamma

    output_dir : str
        Directory to write CSV files.

    Outputs
    -------
    responsibilities_r.csv
    responsibilities_gamma.csv
    """

    os.makedirs(output_dir, exist_ok=True)

    # Save joint responsibilities r_{t,l}^{(k)}
    r_file = os.path.join(output_dir, "responsibilities_r.csv")
    r_df.to_csv(r_file, index=False)

    # Save delay responsibilities γ_{t,l}
    gamma_file = os.path.join(output_dir, "responsibilities_gamma.csv")
    gamma_df.to_csv(gamma_file, index=False)

    print(f"  ✔ Saved responsibilities (r, γ) to: {output_dir}")
