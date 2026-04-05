from __future__ import annotations

import argparse
import json

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a YAML file and emit JSON")
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    print(json.dumps(data))


if __name__ == "__main__":
    main()
