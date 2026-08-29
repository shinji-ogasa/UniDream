"""Run the Wave3D returns-only constant-exposure diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

from unidream.eval.constant_exposure_diagnostic import (
    WAVE_FOLDS,
    load_constant_exposure_data,
    run_constant_exposure_diagnostic,
    validate_wave3d_folds,
)
from unidream.experiments.runtime import load_config, resolve_costs


def _parse_folds(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers")
    try:
        # Preserve order and duplicates so the exact-set validator can fail
        # closed instead of silently sorting/deduplicating a request.
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers") from exc


def _default_returns_path(config: dict) -> Path:
    run_cfg = config["run"]
    data_cfg = config["data"]
    zscore_window = int(config["normalization"]["zscore_window_days"])
    tag = (
        f"{data_cfg['symbol']}_{data_cfg['interval']}_{run_cfg['start']}_{run_cfg['end']}"
        f"_z{zscore_window}_v3"
    )
    return Path(config["logging"]["cache_dir"]) / f"{tag}_returns.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unidream.cli.evaluate_constant_exposure_diagnostic",
        description=(
            "Run the Wave3D fold-0..11 validation-selected constant-exposure "
            "baseline diagnostic (no forecast features or holdout data)."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        default="configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml",
    )
    parser.add_argument(
        "--folds",
        default=",".join(str(value) for value in WAVE_FOLDS),
        type=_parse_folds,
        help="must be exactly 0,1,...,11 once each",
    )
    parser.add_argument("--returns-cache", default=None)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument(
        "--output-dir",
        default="docs/constant_exposure_plan011_dev",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        requested_folds = validate_wave3d_folds(args.folds)
        cfg, _ = resolve_costs(load_config(str(args.config)))
        run_cfg = cfg.get("run", {})
        data_cfg = cfg.get("data", {})
        train_years = int(data_cfg.get("train_years", run_cfg.get("train_years", 2)))
        val_months = int(data_cfg.get("val_months", run_cfg.get("val_months", 3)))
        test_months = int(data_cfg.get("test_months", run_cfg.get("test_months", 3)))
        returns_path = Path(args.returns_cache) if args.returns_cache else _default_returns_path(cfg)
        data = load_constant_exposure_data(
            returns_path,
            folds=requested_folds,
            train_years=train_years,
            val_months=val_months,
            test_months=test_months,
        )
        result = run_constant_exposure_diagnostic(
            data=data,
            cfg=cfg,
            config_path=str(args.config),
            seed=int(args.seed),
            output_dir=args.output_dir,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"[constant-exposure] report: {result['report_path']}")
    print(f"[constant-exposure] ledger: {result['ledger_path']}")
    print(f"[constant-exposure] paths: {result['path_artifacts']['npz_path']}")
    print(f"[constant-exposure] folds: {result['folds']}")
    print(f"[constant-exposure] gate: {result['gate']['status']}")
    print(f"[constant-exposure] next_wave: {result['next_wave_candidates']}")
    print(f"[constant-exposure] runtime_seconds: {result['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
