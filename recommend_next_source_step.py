from __future__ import annotations

import argparse
from pathlib import Path

from check_config_source_requirements import collect_missing_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend the next raw sources to fetch based on stage order")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--config", action="append", required=True)
    args = parser.parse_args()

    unlocked: list[str] = []
    blocked: list[tuple[str, list[str]]] = []
    for config_path in args.config:
        missing = collect_missing_requirements(args.cache_dir, args.cache_tag, config_path)
        name = Path(config_path).name
        if missing:
            blocked.append((name, missing))
        else:
            unlocked.append(name)

    print("[NEXT] Unlocked configs:")
    for name in unlocked:
        print(f"  {name}")

    if not blocked:
        print("[NEXT] All provided configs are unlocked.")
        return

    next_name, next_missing = blocked[0]
    print(f"[NEXT] Next blocked config -> {next_name}")
    print("[NEXT] Fetch these raw sources next:")
    seen: set[str] = set()
    for item in next_missing:
        target = item.split("missing ", 1)[-1]
        if target not in seen:
            print(f"  {target}")
            seen.add(target)


if __name__ == "__main__":
    main()
