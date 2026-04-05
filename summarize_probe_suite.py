from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a suite of probe summary CSV files")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--pattern", default="**/risk_controller_summary.csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.suite_dir, args.pattern), recursive=True))
    if not paths:
        print("[SUITE] No summary files found.")
        return

    rows: list[dict] = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        config_name = Path(path).parent.name
        rows.append(
            {
                "config": config_name,
                "folds": int(len(df)),
                "test_alpha_pt_mean": float(df["test_alpha_pt"].mean()),
                "test_alpha_pt_min": float(df["test_alpha_pt"].min()),
                "test_sharpe_delta_mean": float(df["test_sharpe_delta"].mean()),
                "test_maxdd_delta_pt_mean": float(df["test_maxdd_delta_pt"].mean()),
                "test_win_rate_mean": float(df["test_win_rate_vs_bh"].mean()),
                "m2_pass_count": int(df["test_m2_pass"].sum()),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        by=["m2_pass_count", "test_alpha_pt_mean", "test_sharpe_delta_mean"],
        ascending=[False, False, False],
    )
    out_path = os.path.join(args.suite_dir, "suite_summary.csv")
    summary.to_csv(out_path, index=False)
    print("[SUITE] Summary:")
    print(summary.to_string(index=False))
    print(f"[SUITE] Saved summary -> {out_path}")


if __name__ == "__main__":
    main()
