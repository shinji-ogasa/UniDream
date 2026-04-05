from __future__ import annotations

import argparse
from pathlib import Path

from check_config_source_requirements import collect_missing_requirements
from source_rollout_plan import dedupe_missing_targets, fetch_command_hint


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a markdown report for current source rollout status")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--output", required=True)
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

    lines: list[str] = []
    lines.append("# Source Rollout Report")
    lines.append("")
    lines.append(f"- Cache dir: `{args.cache_dir}`")
    lines.append(f"- Cache tag: `{args.cache_tag}`")
    lines.append("")
    lines.append("## Unlocked Configs")
    if unlocked:
        for name in unlocked:
            lines.append(f"- `{name}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Blocked Configs")
    if blocked:
        for name, missing in blocked:
            lines.append(f"- `{name}`")
            for item in missing:
                lines.append(f"  - {item}")
    else:
        lines.append("- none")
    lines.append("")

    if blocked:
        next_name, next_missing = blocked[0]
        lines.append("## Next Stage")
        lines.append(f"- Next blocked config: `{next_name}`")
        for target in dedupe_missing_targets(next_missing):
            lines.append(f"- Needed raw source: `{target}`")
            hint = fetch_command_hint(args.cache_dir, args.cache_tag, target)
            if hint:
                lines.append(f"  - Fetch hint: `{hint}`")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[REPORT] Wrote markdown report -> {out_path}")


if __name__ == "__main__":
    main()
