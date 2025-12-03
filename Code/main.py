"""
main.py
-------
Run Steps 1–5: ingest → actions → states → sanitize → delay windows (CSV).
"""
import gc 
import argparse
from ingest import LOBSTERFiles, load_and_align
from action_representation import build_actions
from state_features import build_state_features
from state_sanitize import sanitize_and_standardize_states
from delay_windows import build_action_state_pairs
from delay_windows import collapse_actions,collapse_states 

def parse_args():
    p = argparse.ArgumentParser(description="Run Steps 1–5 for LOBSTER preprocessing (CSV outputs).")
    p.add_argument("--message", "-m", required=True, help="LOBSTER message CSV path")
    p.add_argument("--orderbook", "-o", required=True, help="LOBSTER orderbook CSV path")
    p.add_argument("--levels", "-l", type=int, required=True, help="Orderbook depth levels")
    p.add_argument("--lookback", "-n", type=int, required=True, help="Lookback n (previous states)")
    p.add_argument("--out_action", "-oa", default="actions.csv", help="Output CSV for actions")
    p.add_argument("--out_state_raw", "-osr", default="states_raw.csv", help="Output CSV for raw states")
    p.add_argument("--out_state", "-os", default="states_scaled.csv", help="Output CSV for scaled states")
    p.add_argument("--out_scaler", "-sc", default="state_scaler.json", help="Scaler JSON path")
    p.add_argument("--out_windows", "-ow", default="pairs.csv", help="Output CSV for (A_t, S_{t-l}) pairs")

    return p.parse_args()

def main():
    args = parse_args()
    files = LOBSTERFiles(args.message, args.orderbook, args.levels)

    print("📥 Step 1: Loading LOBSTER files...")
    events_df = load_and_align(files)

    print("⚙️ Step 2: Building discrete actions...")
    actions_df = build_actions(events_df, levels=args.levels)
    actions_df.to_csv(args.out_action, index=False)

    print("🧩 Step 3: Building state features...")
    states_df = build_state_features(events_df, levels=args.levels)
    states_df.to_csv(args.out_state_raw, index=False)

    print("🧼 Step 4: Sanitizing + z-scoring states...")
    states_scaled_df, scaler = sanitize_and_standardize_states(
        states_df, train_ratio=0.8, scaler_out=args.out_scaler
    )
    states_scaled_df.to_csv(args.out_state, index=False)

    print(f"🧱 Step 5: Building (A_t, S_(t-l)) pairs with n={args.lookback}...")
    actions_df = collapse_actions(actions_df)
    states_scaled_df = collapse_states(states_scaled_df)

    pairs_df = build_action_state_pairs(states_scaled_df, actions_df, n=args.lookback)
    pairs_df.to_csv(args.out_windows, index=False)

    print(f"Pairs  → {args.out_windows} ({len(pairs_df):,} rows)")

    print("\n✅ Done.")
    print(f"Actions  → {args.out_action}  ({len(actions_df):,})")
    print(f"States   → {args.out_state}   ({len(states_scaled_df):,})  (scaler: {args.out_scaler})")
    print(f"Windows  → {args.out_windows} ({len(pairs_df):,})")

if __name__ == "__main__":
    main()
