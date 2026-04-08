from __future__ import annotations

import argparse
import json
from pathlib import Path


ISSUES = [
    ("issue1_teacher_audit", "checkpoints/teacher_audit"),
    ("issue2_bc_prior", "checkpoints/bc_prior_audit"),
    ("issue3_ac_support", "checkpoints/ac_support_audit"),
    ("issue4_wm_regime", "checkpoints/wm_regime_audit"),
    ("issue5_conservative_ac", "checkpoints/medium_l1_ac_klbudget_fold4"),
    ("issue6_external_sources", "checkpoints/source_family_suite_free"),
]


def build_status(root: Path) -> dict:
    issues: list[dict] = []
    for name, rel in ISSUES:
        path = root / rel
        exists = path.exists()
        has_files = exists and any(path.rglob("*"))
        issues.append(
            {
                "issue": name,
                "path": rel.replace("\\", "/"),
                "exists": exists,
                "has_files": has_files,
                "status": "done" if has_files else ("ready" if exists else "pending"),
            }
        )
    return {"issues": issues}


def write_markdown(status: dict, output: Path) -> None:
    lines = ["# Optimization Status", "", "| issue | status | path |", "| --- | --- | --- |"]
    for row in status["issues"]:
        lines.append(f"| {row['issue']} | {row['status']} | {row['path']} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Write optimization loop status snapshot")
    p.add_argument("--root", default=".")
    p.add_argument("--json-output", default="checkpoints/optimization_status.json")
    p.add_argument("--md-output", default="checkpoints/optimization_status.md")
    args = p.parse_args()

    root = Path(args.root).resolve()
    status = build_status(root)

    json_output = root / args.json_output
    md_output = root / args.md_output
    json_output.parent.mkdir(parents=True, exist_ok=True)
    md_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(status, md_output)


if __name__ == "__main__":
    main()
