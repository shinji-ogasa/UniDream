"""CLI for the fixed Wave3C causal-context forecast screen."""
from __future__ import annotations

import argparse
from pathlib import Path

from unidream.eval.forecast_context_tournament import (
    HORIZON_GRID,
    WAVE_FOLDS,
    WAVE_SEED,
    load_development_data,
    run_context_tournament,
)
from unidream.experiments.runtime import load_config, resolve_costs


def _parse_fold_list(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers")
    try:
        parsed = [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("folds must be comma-separated integers") from exc
    if any(str(parsed_value) != token for parsed_value, token in zip(parsed, tokens)):
        # Reject values such as ``1.5`` and retain a visible duplicate rather
        # than silently normalizing a fail-closed fold request.
        raise argparse.ArgumentTypeError("folds must contain integral tokens")
    return parsed


def _validate_exact_folds(folds: list[int]) -> tuple[int, ...]:
    if len(folds) != len(WAVE_FOLDS) or len(set(folds)) != len(folds) or set(folds) != set(WAVE_FOLDS):
        raise argparse.ArgumentTypeError(
            f"Wave3C requires exactly one each of folds {list(WAVE_FOLDS)}"
        )
    return tuple(sorted(folds))


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
        description="Run the fixed, development-only Plan011 Wave3C context tournament",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        default="configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml",
    )
    parser.add_argument(
        "--folds",
        default=",".join(str(value) for value in WAVE_FOLDS),
        type=_parse_fold_list,
        help="exactly 0,2,8; no holdout or future folds",
    )
    parser.add_argument("--features-cache", default=None)
    parser.add_argument("--returns-cache", default=None)
    parser.add_argument("--wave3a-result", default="docs/forecast_tournament_plan011_dev/result.json")
    parser.add_argument("--max-fit-rows", default=20_000, type=int)
    parser.add_argument(
        "--output-dir",
        default="docs/forecast_context_tournament_plan011_dev",
    )
    args = parser.parse_args(argv)
    requested_folds = _validate_exact_folds(args.folds)
    if int(args.max_fit_rows) <= 0:
        parser.error("--max-fit-rows must be positive")
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
    result = run_context_tournament(
        data=data,
        cfg=cfg,
        config_path=str(args.config),
        seed=WAVE_SEED,
        horizons=HORIZON_GRID,
        max_fit_rows=int(args.max_fit_rows),
        wave3a_result_path=args.wave3a_result,
        output_dir=args.output_dir,
    )
    print(f"[forecast-context-tournament] report: {result['report_path']}")
    print(f"[forecast-context-tournament] ledger: {result['ledger_path']}")
    print(f"[forecast-context-tournament] folds: {result['folds']}")
    print(f"[forecast-context-tournament] next_wave: {result['next_wave_candidates']}")
    print(f"[forecast-context-tournament] runtime_seconds: {result['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
