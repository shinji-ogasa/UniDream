from __future__ import annotations

import argparse
from pathlib import Path

from check_config_source_requirements import collect_missing_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description="Report which source stages/configs are unlocked by current cache")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()

    print("| config | unlocked | missing_raw_dependencies |")
    print("| --- | --- | --- |")
    for config_path in args.config:
        missing = collect_missing_requirements(args.cache_dir, args.cache_tag, config_path)
        unlocked = len(missing) == 0
        missing_text = "; ".join(missing) if missing else ""
        print(f"| {Path(config_path).name} | {unlocked} | {missing_text} |")


if __name__ == "__main__":
    main()
