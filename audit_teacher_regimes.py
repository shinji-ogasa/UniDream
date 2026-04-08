from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from unidream.experiments.runtime import load_config, resolve_costs, set_seed
from unidream.experiments.teacher_audit import load_audit_features, run_teacher_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit teacher/oracle action distributions by regime.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--start", required=True, help="Start date, e.g. 2020-01-01")
    parser.add_argument("--end", required=True, help="End date, e.g. 2024-01-01")
    parser.add_argument("--cache-dir", default="checkpoints/data_cache", help="Primary cache dir.")
    parser.add_argument("--raw-cache-dir", default=None, help="Optional secondary cache dir for raw source parquet files.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/teacher_audit", help="Output directory.")
    parser.add_argument("--folds", default=None, help="Comma-separated fold indices.")
    parser.add_argument("--oracle-min-holds", default="64", help="Comma-separated oracle min_hold values to audit.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_int_list(raw: str) -> list[int]:
    values = sorted({int(token.strip()) for token in raw.split(",") if token.strip()})
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _write_markdown_report(summary_df: pd.DataFrame, report_path: Path) -> None:
    if summary_df.empty:
        report_path.write_text("# Teacher Audit\n\nNo rows generated.\n", encoding="utf-8")
        return

    overall = summary_df[(summary_df["split"] == "train") & (summary_df["regime"] == "all")].copy()
    bearish = summary_df[
        summary_df["regime_expected_return"].notna() & (summary_df["regime_expected_return"] <= 0.0)
    ].copy()
    bearish = bearish.groupby(["config", "oracle_min_hold", "split"], dropna=False).agg(
        bearish_long_ratio=("long_ratio", "mean"),
        bearish_short_ratio=("short_ratio", "mean"),
        bearish_flat_ratio=("flat_ratio", "mean"),
        bearish_avg_hold=("avg_hold", "mean"),
    ).reset_index()

    merged = overall.merge(bearish, on=["config", "oracle_min_hold", "split"], how="left")
    lines = ["# Teacher Audit", ""]
    for _, row in merged.sort_values(["config", "oracle_min_hold", "split"]).iterrows():
        lines.append(
            f"- `{row['config']}` hold={int(row['oracle_min_hold'])} split={row['split']}: "
            f"all(long={row['long_ratio']:.1%}, short={row['short_ratio']:.1%}, flat={row['flat_ratio']:.1%}, avg_hold={row['avg_hold']:.1f}) "
            f"| bearish(long={row.get('bearish_long_ratio', float('nan')):.1%}, "
            f"short={row.get('bearish_short_ratio', float('nan')):.1%}, "
            f"flat={row.get('bearish_flat_ratio', float('nan')):.1%}, "
            f"avg_hold={row.get('bearish_avg_hold', float('nan')):.1f})"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _ = resolve_costs(cfg)

    symbol = cfg["data"]["symbol"]
    interval = cfg["data"]["interval"]
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_audit_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=args.cache_dir,
        cache_tag=cache_tag,
        raw_cache_dir=args.raw_cache_dir,
    )

    config_name = Path(args.config).stem
    out_dir = Path(args.checkpoint_dir) / config_name
    summary_df, detail_df = run_teacher_audit(
        cfg=cfg,
        config_name=config_name,
        features_df=features_df,
        raw_returns=raw_returns,
        min_hold_values=_parse_int_list(args.oracle_min_holds),
        folds_arg=args.folds,
        checkpoint_dir=str(out_dir),
    )
    _write_markdown_report(summary_df, out_dir / f"{config_name}_teacher_audit.md")

    print(f"[AUDIT] wrote {out_dir / f'{config_name}_teacher_audit_summary.csv'}")
    if summary_df.empty:
        print("[AUDIT] no rows")
        return
    print(summary_df.to_string(index=False))
    if not detail_df.empty:
        bearish = detail_df[
            detail_df["regime"].str.startswith("regime_")
            & detail_df["regime_expected_return"].notna()
            & (detail_df["regime_expected_return"] <= 0.0)
        ]
        if not bearish.empty:
            grouped = bearish.groupby(["oracle_min_hold", "split"], dropna=False).agg(
                long_ratio=("long_ratio", "mean"),
                short_ratio=("short_ratio", "mean"),
                flat_ratio=("flat_ratio", "mean"),
                avg_hold=("avg_hold", "mean"),
            )
            print("\n[AUDIT] bearish regimes")
            print(grouped.to_string())


if __name__ == "__main__":
    main()
