from __future__ import annotations

import argparse
import os

import yaml


def validate_suite(suite_path: str) -> list[str]:
    with open(suite_path, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f) or {}

    ordered_configs = list(suite.get("ordered_configs") or [])
    stages = list(suite.get("stages") or [])
    errors: list[str] = []

    if not ordered_configs:
        errors.append("ordered_configs is empty")
    if not stages:
        errors.append("stages is empty")

    suite_dir = os.path.dirname(os.path.abspath(suite_path))
    repo_root = os.path.dirname(suite_dir)

    seen_stage_names: set[str] = set()
    stage_configs: list[str] = []
    for stage in stages:
        name = str(stage.get("name") or "").strip()
        configs = list(stage.get("configs") or [])
        if not name:
            errors.append("stage without name")
            continue
        if name in seen_stage_names:
            errors.append(f"duplicate stage name: {name}")
        seen_stage_names.add(name)
        if not configs:
            errors.append(f"stage '{name}' has no configs")
        stage_configs.extend(configs)

    ordered_set = set(ordered_configs)
    stage_set = set(stage_configs)

    missing_from_stages = [cfg for cfg in ordered_configs if cfg not in stage_set]
    extra_in_stages = [cfg for cfg in stage_configs if cfg not in ordered_set]
    duplicate_stage_configs = sorted({cfg for cfg in stage_configs if stage_configs.count(cfg) > 1})

    for cfg in missing_from_stages:
        errors.append(f"ordered config missing from stages: {cfg}")
    for cfg in extra_in_stages:
        errors.append(f"stage config missing from ordered_configs: {cfg}")
    for cfg in duplicate_stage_configs:
        errors.append(f"config appears in multiple stages: {cfg}")

    for cfg in ordered_configs:
        cfg_path = os.path.join(repo_root, cfg)
        if not os.path.exists(cfg_path):
            errors.append(f"config path does not exist: {cfg}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate source rollout suite YAML")
    parser.add_argument("--suite-config", required=True)
    args = parser.parse_args()

    errors = validate_suite(args.suite_config)
    if errors:
        print("[SUITE] Invalid rollout suite:")
        for err in errors:
            print(f"  {err}")
        raise SystemExit(1)

    print("[SUITE] Rollout suite is valid.")


if __name__ == "__main__":
    main()
