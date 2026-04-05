from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    out: list[pd.Timestamp] = []
    while cur < end:
        out.append(cur)
        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1, tz="UTC")
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1, tz="UTC")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local manual inputs for free source rollout")
    parser.add_argument("--zip-dir", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--coinmetrics-json", default=None)
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    zip_dir = Path(args.zip_dir)
    months = [m.strftime("%Y-%m") for m in _month_starts(start, end)]
    expected = [f"{args.symbol}-{args.interval}-{m}.zip" for m in months]
    missing = [name for name in expected if not (zip_dir / name).exists()]
    present = [name for name in expected if (zip_dir / name).exists()]

    report: dict[str, object] = {
        "zip_dir": str(zip_dir),
        "expected_months": months,
        "present_zip_count": len(present),
        "missing_zips": missing,
    }

    if args.coinmetrics_json:
        cm_path = Path(args.coinmetrics_json)
        cm_info: dict[str, object] = {"path": str(cm_path), "exists": cm_path.exists()}
        if cm_path.exists():
            try:
                raw = pd.read_json(cm_path)
                if "data" in raw.columns and len(raw.columns) == 1:
                    raw = pd.DataFrame(raw["data"].tolist())
                cm_info["rows"] = int(len(raw))
                cm_info["columns"] = list(raw.columns)
                if "time" in raw.columns:
                    times = pd.to_datetime(raw["time"], utc=False)
                    cm_info["first_time"] = str(times.min())
                    cm_info["last_time"] = str(times.max())
            except Exception as e:
                cm_info["error"] = str(e)
        report["coinmetrics"] = cm_info

    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
