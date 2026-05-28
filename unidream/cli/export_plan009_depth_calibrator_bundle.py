from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.research.plan005_meta_guard_bc_ac import (
    _build_past_features,
    guard_positions_from_features,
    select_meta_guard_mode,
)


def _bars_per_day(interval: str) -> int:
    return {
        "1m": 1440,
        "5m": 288,
        "15m": 96,
        "30m": 48,
        "1h": 24,
        "4h": 6,
        "1d": 1,
    }.get(str(interval), 96)


def _mode_depths_from_eval(eval_payload: dict[str, Any], fallback: dict[str, float]) -> dict[str, float]:
    by_mode: dict[str, list[float]] = {}
    for row in eval_payload.get("rows", []):
        if row.get("group") != "plan009_depth_calibrator":
            continue
        diag = row.get("diag", {})
        mode = str(diag.get("test_mode", "core_pair"))
        depth = float(diag.get("selected_depth", fallback.get(mode, 0.94)))
        by_mode.setdefault(mode, []).append(depth)
    out = dict(fallback)
    for mode, values in by_mode.items():
        if values:
            out[mode] = float(np.median(np.asarray(values, dtype=np.float64)))
    return out


def _apply_plan009_positions(
    current_returns: np.ndarray,
    *,
    history_returns: np.ndarray,
    benchmark_position: float,
    depth_by_mode: dict[str, float],
    default_depth: float,
    min_position: float,
    max_position: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = np.asarray(current_returns, dtype=np.float64)
    history = np.asarray(history_returns, dtype=np.float64)
    full_returns = np.concatenate([history, current]) if len(history) else current
    offset = int(len(history))
    features = _build_past_features(full_returns)
    mode, mode_diag = select_meta_guard_mode(features, offset)
    guard = guard_positions_from_features(features, mode=mode)[offset : offset + len(current)]
    depth = float(depth_by_mode.get(mode, default_depth))
    bench = float(benchmark_position)
    positions = bench - depth * (bench - guard)
    positions = np.clip(positions, float(min_position), float(max_position))
    overlay = positions - bench
    diag = {
        "plan009_depth_calibrator_version": "dev_f0_12_m48_x2_cap094",
        "plan009_mode": mode,
        "plan009_depth": depth,
        "plan009_history_bars": offset,
        "plan009_current_bars": int(len(current)),
        "plan009_guard_mean": float(np.mean(guard)) if len(guard) else 0.0,
        "plan009_underweight_rate": float(np.mean(guard < bench - 1e-12)) if len(guard) else 0.0,
        "plan009_turnover": float(np.abs(np.diff(overlay)).sum()) if len(overlay) > 1 else 0.0,
        "plan009_active_rate": float(np.mean(np.abs(overlay) > 0.05)) if len(overlay) else 0.0,
        **mode_diag,
    }
    return positions.astype(np.float32), diag


def export_plan009_depth_calibrator_bundle(
    *,
    config_path: str,
    eval_json: str,
    eval_md: str,
    output_dir: str,
    start: str,
    end: str,
    seed: int,
    current_window_days: int,
    min_live_lookback_days: int,
) -> dict[str, Any]:
    set_seed(seed)
    cfg = load_config(config_path)
    cfg, profile = resolve_costs(cfg, None)
    data_cfg = cfg.get("data", {})
    symbol = str(data_cfg.get("symbol", "BTCUSDT"))
    interval = str(data_cfg.get("interval", "15m"))
    zscore_window = int(cfg.get("normalization", {}).get("zscore_window_days", 60))
    cache_tag = f"{symbol}_{interval}_{start}_{end}_z{zscore_window}_v2_plan008_latest"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        zscore_window=zscore_window,
        cache_dir="checkpoints/data_cache",
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", False)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    eval_payload = json.loads(Path(eval_json).read_text(encoding="utf-8"))
    fallback_depths = {
        "core_pair": 0.94,
        "pre_halving_rebound": 0.62,
        "deep_bear_recovery": 0.94,
    }
    depth_by_mode = _mode_depths_from_eval(eval_payload, fallback_depths)
    default_depth = float(depth_by_mode.get("core_pair", 0.94))
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    min_position = 0.0
    max_position = max(benchmark_position, 1.0)

    sample_bars = max(1, int(current_window_days) * _bars_per_day(interval) + 1)
    features_arr = features_df.to_numpy(dtype=np.float32)
    returns_arr = np.asarray(raw_returns, dtype=np.float32)
    t = min(len(features_arr), len(returns_arr), len(features_df.index))
    current_start = max(0, t - sample_bars)
    sample_features = features_arr[current_start:t]
    sample_returns = returns_arr[current_start:t]
    history_returns = returns_arr[:current_start]
    timestamps = np.asarray([str(x) for x in features_df.index[current_start:t]], dtype="<U64")
    expected_positions, sample_diag = _apply_plan009_positions(
        sample_returns,
        history_returns=history_returns,
        benchmark_position=benchmark_position,
        depth_by_mode=depth_by_mode,
        default_depth=default_depth,
        min_position=min_position,
        max_position=max_position,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "model_config.yaml", "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    np.savez_compressed(
        output / "sample_input.npz",
        features=sample_features,
        returns=sample_returns,
        history_returns=history_returns,
        expected_positions=expected_positions,
        timestamps=timestamps,
    )
    sample_output = {
        "last_position": float(expected_positions[-1]) if len(expected_positions) else None,
        "n_positions": int(len(expected_positions)),
        "signal": (
            "underweight"
            if len(expected_positions) and float(expected_positions[-1]) < benchmark_position - 0.05
            else "benchmark"
        ),
        "diag": sample_diag,
    }
    with open(output / "sample_output.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(sample_output), f, ensure_ascii=False, indent=2)
    shutil.copy2(eval_json, output / "plan009_eval_folds0_12.json")
    shutil.copy2(eval_md, output / "plan009_eval_folds0_12.md")

    summary = {
        "experiment": "plan009_depth_calibrator_bundle",
        "source_eval": eval_json,
        "depth_by_mode": depth_by_mode,
        "default_depth": default_depth,
        "dev_aggregate": eval_payload.get("aggregate", {}),
        "dev_stress_aggregate": eval_payload.get("stress_aggregate", {}),
        "sample": sample_output,
    }
    with open(output / "plan009_summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(summary), f, ensure_ascii=False, indent=2)

    manifest = {
        "bundle_version": 9,
        "bundle_type": "plan009_depth_calibrator",
        "created_by": "unidream.cli.export_plan009_depth_calibrator_bundle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "seed": int(seed),
            "status": "dev_candidate",
            "source": "plan004_base_plus_plan005_past_guard",
            "spec": "plan009_depth_calibrator_dev_f0_12_m48_x2_cap094",
            "no_leak_scope": (
                "Dev fold0-12 depth selection used validation split only; runtime signals use shifted trailing-return "
                "features only. Fold0-12 remains a development set, not a pristine holdout claim."
            ),
        },
        "data": {
            "symbol": symbol,
            "interval": interval,
            "start": start,
            "end": end,
            "zscore_window_days": int(zscore_window),
            "feature_columns": list(features_df.columns),
            "obs_dim": int(sample_features.shape[1]) if sample_features.ndim == 2 else 0,
            "seq_len": int(data_cfg.get("seq_len", 64)),
            "cost_profile": profile,
            "include_oi": bool(data_cfg.get("include_oi", False)),
        },
        "split": {
            "dev_folds": eval_payload.get("folds", list(range(13))),
            "dev_eval": "0-12 walk-forward folds",
            "serving_span": f"trailing {int(current_window_days)} day window with prior returns as guard history",
        },
        "policy": {
            "benchmark_position": benchmark_position,
            "target_alpha_excess_pt_min": 3.0,
            "target_maxdd_delta_pt_max": -3.0,
            "current_window_days": int(current_window_days),
            "min_live_lookback_days": int(min_live_lookback_days),
            "sample_bars": int(len(expected_positions)),
            "last_sample_position": sample_output["last_position"],
            "last_sample_signal": sample_output["signal"],
        },
        "artifacts": {
            "sample_input": "sample_input.npz",
            "sample_output": "sample_output.json",
            "plan009_summary": "plan009_summary.json",
            "plan009_eval_json": "plan009_eval_folds0_12.json",
            "plan009_eval_md": "plan009_eval_folds0_12.md",
            "config": "model_config.yaml",
        },
        "plan009_depth_calibrator": {
            "val_dd_target": float(eval_payload.get("val_dd_target", -4.8)),
            "safety_multiplier": float(eval_payload.get("safety_multiplier", 2.0)),
            "min_depth_floor": float(eval_payload.get("min_depth_floor", 0.30)),
            "max_depth_cap": float(eval_payload.get("max_depth_cap", 0.94)),
            "default_depth": default_depth,
            "depth_by_mode": depth_by_mode,
            "min_position": min_position,
            "max_position": max_position,
        },
    }
    with open(output / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(manifest), f, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.export_plan009_depth_calibrator_bundle")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--eval-json", default="docs_local/20260528_plan009_depth_calibrator_f0_12_m48_x2_cap094.json")
    parser.add_argument("--eval-md", default="docs_local/20260528_plan009_depth_calibrator_f0_12_m48_x2_cap094.md")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-05-21")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--current-window-days", type=int, default=60)
    parser.add_argument("--min-live-lookback-days", type=int, default=365)
    args = parser.parse_args()
    manifest = export_plan009_depth_calibrator_bundle(
        config_path=args.config,
        eval_json=args.eval_json,
        eval_md=args.eval_md,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        seed=int(args.seed),
        current_window_days=int(args.current_window_days),
        min_live_lookback_days=int(args.min_live_lookback_days),
    )
    print(f"[export] wrote Plan009 bundle: {args.output_dir}")
    print(json.dumps(_json_sanitize(manifest["run"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
