from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def _feature_to_requirement(feature_name: str) -> str | None:
    if feature_name.startswith("basis"):
        return "mark"
    if feature_name.startswith("fund_") or feature_name == "funding_rate":
        return "funding"
    if feature_name.startswith("oi_") or feature_name == "oi_change":
        return "oi"

    suffixes = (
        "_delta1",
        "_abs",
        "_mean_16", "_mean_96", "_mean_288",
        "_std_16", "_std_96", "_std_288",
        "_z_16", "_z_96", "_z_288",
        "_impulse_16", "_impulse_96", "_impulse_288",
    )
    base = feature_name
    for suffix in suffixes:
        if feature_name.endswith(suffix):
            base = feature_name[: -len(suffix)]
            break
    if base in {
        "signed_order_flow",
        "taker_imbalance",
        "buy_sell_ratio",
        "exchange_netflow",
        "stablecoin_inflow",
        "active_address_growth",
    }:
        return f"series:{base}"
    return None


def _load_feature_subset(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return ((cfg.get("risk_controller") or {}).get("feature_subset")) or []


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize raw source requirements for probe configs")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    for config_path in args.config:
        reqs = sorted({req for feat in _load_feature_subset(config_path) if (req := _feature_to_requirement(feat))})
        rows.append(
            {
                "config": Path(config_path).name,
                "needs_mark": "mark" in reqs,
                "needs_funding": "funding" in reqs,
                "needs_oi": "oi" in reqs,
                "extra_series": ", ".join(r.split(":", 1)[1] for r in reqs if r.startswith("series:")),
            }
        )

    df = pd.DataFrame(rows).sort_values("config")
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in df.iterrows()
    ]
    markdown = "\n".join([header, sep, *body])
    print(markdown)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown + "\n", encoding="utf-8")
        print(f"[REQ] Saved markdown -> {out_path}")


if __name__ == "__main__":
    main()
