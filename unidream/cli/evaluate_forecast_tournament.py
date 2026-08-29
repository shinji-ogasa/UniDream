"""CLI for the development-only direct forecast/timing tournament."""
from __future__ import annotations

import argparse
from pathlib import Path

from unidream.eval.forecast_tournament import (
    DEFAULT_HORIZONS,
    DEV_FOLDS,
    load_development_data,
    run_tournament,
    validate_requested_folds,
)
from unidream.experiments.runtime import load_config, resolve_costs


def _parse_int_list(value: str) -> list[int]:
    try:
        parsed = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("folds/horizons must be comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return sorted(set(parsed))


def _default_cache_paths(config: dict) -> tuple[Path, Path]:
    run_cfg = config["run"]
    data_cfg = config["data"]
    zscore_window = int(config["normalization"]["zscore_window_days"])
    tag = (
        f"{data_cfg['symbol']}_{data_cfg['interval']}_{run_cfg['start']}_{run_cfg['end']}"
        f"_z{zscore_window}_v3"
    )
    root = Path(config["logging"]["cache_dir"])
    return root / f"{tag}_features.parquet", root / f"{tag}_returns.parquet"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the leak-safe Plan011 development forecast/timing tournament",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        default="configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml",
    )
    parser.add_argument(
        "--folds",
        default=",".join(str(value) for value in DEV_FOLDS),
        type=_parse_int_list,
        help="development folds only; default 0,2,8",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(value) for value in DEFAULT_HORIZONS),
        type=_parse_int_list,
    )
    parser.add_argument("--features-cache", default=None)
    parser.add_argument("--returns-cache", default=None)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--hist-max-iter", default=60, type=int)
    parser.add_argument("--max-fit-rows", default=40_000, type=int)
    parser.add_argument(
        "--output-dir",
        default="docs/forecast_tournament_plan011_dev",
    )
    args = parser.parse_args(argv)
    requested_folds = validate_requested_folds(args.folds)
    cfg, _ = resolve_costs(load_config(str(args.config)))
    run_cfg = cfg.get("run", {})
    data_cfg = cfg.get("data", {})
    train_years = int(data_cfg.get("train_years", run_cfg.get("train_years", 2)))
    val_months = int(data_cfg.get("val_months", run_cfg.get("val_months", 3)))
    test_months = int(data_cfg.get("test_months", run_cfg.get("test_months", 3)))
    default_features, default_returns = _default_cache_paths(cfg)
    feature_path = Path(args.features_cache) if args.features_cache else default_features
    returns_path = Path(args.returns_cache) if args.returns_cache else default_returns
    data = load_development_data(
        feature_path,
        returns_path,
        folds=requested_folds,
        train_years=train_years,
        val_months=val_months,
        test_months=test_months,
    )
    result = run_tournament(
        data=data,
        cfg=cfg,
        config_path=str(args.config),
        seed=int(args.seed),
        horizons=args.horizons,
        hist_max_iter=int(args.hist_max_iter),
        max_fit_rows=int(args.max_fit_rows),
        output_dir=args.output_dir,
    )
    print(f"[forecast-tournament] report: {result['report_path']}")
    print(f"[forecast-tournament] ledger: {result['ledger_path']}")
    print(f"[forecast-tournament] folds: {result['folds']}")
    print(f"[forecast-tournament] next_wave: {result['next_wave_candidates']}")
    print(f"[forecast-tournament] runtime_seconds: {result['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
