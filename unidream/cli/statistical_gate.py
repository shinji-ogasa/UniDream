"""Run the development-only statistical robustness gate on JSON input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from unidream.eval.statistical_gate import evaluate_json_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unidream.cli.statistical_gate",
        description=(
            "Evaluate bootstrap, DSR, CSCV/PBO, fold-sign, and stress gates "
            "for explicitly supplied development-fold paths."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--input", required=True, help="JSON input matching docs/statistical_gate_contract.md")
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument(
        "--allow-reject",
        action="store_true",
        help="return zero even when the machine-readable gate rejects",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = evaluate_json_input(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if bool(result["gate"]["passed"]) or args.allow_reject else 2


if __name__ == "__main__":
    raise SystemExit(main())
