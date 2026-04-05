from __future__ import annotations

import argparse
from pathlib import Path

from check_config_source_requirements import collect_missing_requirements
from source_rollout_plan import dedupe_missing_targets, fetch_command_hint


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend concrete fetch commands for the next blocked source stage")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()

    for config_path in args.config:
        missing = collect_missing_requirements(args.cache_dir, args.cache_tag, config_path)
        if not missing:
            continue
        config_name = Path(config_path).name
        print(f"[FETCH] Next blocked config -> {config_name}")
        for target in dedupe_missing_targets(missing):
            print(f"[FETCH] {target}")
            print(f"  {fetch_command_hint(args.cache_dir, args.cache_tag, target)}")
        break
    else:
        print("[FETCH] All provided configs are unlocked.")


if __name__ == "__main__":
    main()
