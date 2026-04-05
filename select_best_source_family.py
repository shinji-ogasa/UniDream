from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def select_best_family(summary_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError("suite summary is empty")

    ranked = df.copy()
    ranked["m2_ready"] = (
        (ranked["test_alpha_pt_mean"] >= 5.0)
        & (ranked["test_sharpe_delta_mean"] >= 0.20)
        & (ranked["test_maxdd_delta_pt_mean"] <= -10.0)
        & (ranked["test_win_rate_mean"] >= 0.60)
    )
    ranked["selection_score"] = (
        200.0 * ranked["m2_pass_count"].astype(float)
        + 10.0 * ranked["m2_ready"].astype(float)
        + 2.0 * ranked["test_alpha_pt_mean"]
        + 5.0 * ranked["test_sharpe_delta_mean"]
        + 25.0 * (ranked["test_win_rate_mean"] - 0.5)
        + 0.5 * (-ranked["test_maxdd_delta_pt_mean"])
    )
    ranked = ranked.sort_values(
        by=[
            "m2_pass_count",
            "m2_ready",
            "selection_score",
            "test_alpha_pt_mean",
            "test_sharpe_delta_mean",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    return ranked, ranked.iloc[0]


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best source family from a probe suite summary")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    args = parser.parse_args()

    ranked, best = select_best_family(args.summary)
    print("[SELECT] Ranked source families:")
    print(
        ranked[
            [
                "config",
                "m2_pass_count",
                "m2_ready",
                "selection_score",
                "test_alpha_pt_mean",
                "test_sharpe_delta_mean",
                "test_maxdd_delta_pt_mean",
                "test_win_rate_mean",
            ]
        ].to_string(index=False)
    )
    print(f"[SELECT] Best source family -> {best['config']}")

    if args.output:
        lines = [
            "# Source Family Selection",
            "",
            f"- Summary: `{args.summary}`",
            f"- Best config: `{best['config']}`",
            f"- M2 pass count: `{int(best['m2_pass_count'])}`",
            f"- Alpha mean: `{float(best['test_alpha_pt_mean']):+.2f} pt`",
            f"- Sharpe delta mean: `{float(best['test_sharpe_delta_mean']):+.3f}`",
            f"- MaxDD delta mean: `{float(best['test_maxdd_delta_pt_mean']):+.2f} pt`",
            f"- Win rate mean: `{float(best['test_win_rate_mean']):.1%}`",
            "",
            _to_markdown_table(ranked),
            "",
        ]
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[SELECT] Wrote selection report -> {out_path}")


if __name__ == "__main__":
    main()
