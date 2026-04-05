from __future__ import annotations

import argparse
import json
from pathlib import Path

from source_rollout_plan import build_rollout_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a JSON snapshot for current source rollout status")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()

    snapshot = build_rollout_snapshot(args.cache_dir, args.cache_tag, args.config)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"[SNAPSHOT] Wrote JSON snapshot -> {out_path}")


if __name__ == "__main__":
    main()
