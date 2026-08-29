"""CLI for leak-aware alpha attribution and saved-bundle diagnostics.

The command consumes persisted position paths rather than retraining a model.
It writes a JSONL trial ledger, a JSON result payload, and a Markdown report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from unidream.eval.alpha_attribution import (
    DEFAULT_FIXED_EXPOSURES,
    DEFAULT_LAGS,
    DEFAULT_NULL_SHIFTS,
    load_timeseries_artifact,
    run_attribution,
)
from unidream.experiments.runtime import load_config, resolve_costs


def _parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid descending range: {token}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(token))
    if not result:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return sorted(set(result))


def _parse_float_list(value: str) -> list[float]:
    try:
        result = [float(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("list must contain floats") from exc
    if not result:
        raise argparse.ArgumentTypeError("list must contain at least one float")
    return result


def _default_feature_cache(config: dict) -> Path:
    run_cfg = config["run"]
    data_cfg = config["data"]
    zscore_window = int(config["normalization"]["zscore_window_days"])
    tag = (
        f"{data_cfg['symbol']}_{data_cfg['interval']}_{run_cfg['start']}_{run_cfg['end']}"
        f"_z{zscore_window}_v3"
    )
    return Path(config["logging"]["cache_dir"]) / f"{tag}_features.parquet"


def _default_bundle_path(name: str) -> Path:
    # The saved bundle is in the sibling demo repository when both repos are
    # checked out under the standard UniDream workspace.
    research_root = Path(__file__).resolve().parents[2]
    return research_root.parent / "unidream-space" / "bundles" / "current" / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate persisted Plan011 position paths and predictive heads",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--timeseries",
        default="docs/figures/plan011_v31_folds0_12/timeseries.npz",
        help="saved fold_XX time/return/position artifact",
    )
    parser.add_argument("--validation-timeseries", default=None)
    parser.add_argument(
        "--config",
        default="configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml",
    )
    parser.add_argument("--folds", default=None, type=_parse_int_list)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--execution-delay", default=0, type=int)
    parser.add_argument(
        "--fixed-exposures",
        default=",".join(str(value) for value in DEFAULT_FIXED_EXPOSURES),
        type=_parse_float_list,
    )
    parser.add_argument(
        "--lags",
        default=",".join(str(value) for value in DEFAULT_LAGS),
        type=_parse_int_list,
    )
    parser.add_argument(
        "--null-shifts",
        default=",".join(str(value) for value in DEFAULT_NULL_SHIFTS),
        type=_parse_int_list,
    )
    parser.add_argument("--features-cache", default=None)
    parser.add_argument("--sample-input", default=None)
    parser.add_argument("--predictive-state", default=None)
    parser.add_argument(
        "--output-dir",
        default="docs/alpha_attribution_plan011_v31_dev",
    )
    parser.add_argument(
        "--holdout-reference",
        action="store_true",
        help="allow folds >=15 for reference-only reporting; never enables selection",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    cfg, _ = resolve_costs(load_config(str(config_path)))
    timeseries_path = Path(args.timeseries)
    series = load_timeseries_artifact(timeseries_path, folds=args.folds)
    validation_series = None
    validation_path = None
    if args.validation_timeseries:
        validation_path = Path(args.validation_timeseries)
        validation_series = load_timeseries_artifact(validation_path, folds=args.folds)

    feature_path = Path(args.features_cache) if args.features_cache else _default_feature_cache(cfg)
    if not feature_path.exists():
        print(f"[alpha-attribution] feature cache not found; coverage will be N/A: {feature_path}")
        feature_path = None
    sample_input = Path(args.sample_input) if args.sample_input else _default_bundle_path("sample_input.npz")
    predictive_state = (
        Path(args.predictive_state)
        if args.predictive_state
        else _default_bundle_path("predictive_state.npz")
    )
    if not sample_input.exists():
        sample_input = None
    if not predictive_state.exists():
        predictive_state = None

    payload = run_attribution(
        series=series,
        cfg=cfg,
        config_path=str(config_path),
        artifact_path=timeseries_path,
        seed=args.seed,
        fixed_exposures=args.fixed_exposures,
        lags=args.lags,
        null_shifts=args.null_shifts,
        execution_delay_bars=args.execution_delay,
        sample_input_path=sample_input,
        predictive_state_path=predictive_state,
        output_dir=args.output_dir,
        holdout_reference=args.holdout_reference,
        feature_artifact_path=feature_path,
        validation_series=validation_series,
        validation_artifact_path=validation_path,
    )
    print(f"[alpha-attribution] report: {payload['report_path']}")
    print(f"[alpha-attribution] ledger: {payload['ledger_path']}")
    print(f"[alpha-attribution] folds: {payload['folds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
