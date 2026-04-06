from __future__ import annotations

import argparse

from unidream.source_rollout.requirements import collect_missing_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether source cache files satisfy a config's raw dependencies")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    missing = collect_missing_requirements(args.cache_dir, args.cache_tag, args.config)
    if missing:
        print("[REQ] Missing raw dependencies:")
        for item in missing:
            print(f"  {item}")
        raise SystemExit(1)
    print("[REQ] Raw dependencies satisfied.")


if __name__ == "__main__":
    main()
