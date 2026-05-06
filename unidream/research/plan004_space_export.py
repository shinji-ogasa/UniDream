from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.plan003_policy_blend_probe import _stress_metrics
from unidream.research.plan004_residual_bc_ac import (
    SPECS,
    _append_base_features,
    _fit_ridge_multi,
    _positions_from_residual_prediction,
    _residual_utilities,
    _stress_selection_score,
    _threshold_grid,
)
from unidream.cli.exploration_board_probe import _state_features, _unit_cost
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "plan004_residual_bc_ac_space_bundle"


def _benchmark_positions(n: int, benchmark_position: float) -> np.ndarray:
    return np.full(int(n), float(benchmark_position), dtype=np.float64)


def _slice_index(features_df, start, end, *, right_inclusive: bool = False) -> list[str]:
    if right_inclusive:
        mask = (features_df.index >= start) & (features_df.index <= end)
    else:
        mask = (features_df.index >= start) & (features_df.index < end)
    return [str(x) for x in features_df.index[mask]]


def _fit_benchmark_residual(
    *,
    ds: WFODataset,
    x_train_state: np.ndarray,
    x_val_state: np.ndarray,
    x_test_state: np.ndarray,
    cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    spec_name: str,
    benchmark_position: float,
    unit_cost: float,
    min_position: float,
    max_position: float,
    selection_stress_mode: str,
) -> dict[str, Any]:
    spec = next((s for s in SPECS if s.name == spec_name), None)
    if spec is None:
        raise ValueError(f"unknown residual spec: {spec_name}")

    base_train = _benchmark_positions(len(ds.train_returns), benchmark_position)
    base_val = _benchmark_positions(len(ds.val_returns), benchmark_position)
    base_test = _benchmark_positions(len(ds.test_returns), benchmark_position)
    x_train = _append_base_features(x_train_state, base_train, benchmark_position)
    x_val = _append_base_features(x_val_state, base_val, benchmark_position)
    x_test = _append_base_features(x_test_state, base_test, benchmark_position)

    y_train, valid_train = _residual_utilities(
        ds.train_returns,
        base_train,
        spec=spec,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        min_position=min_position,
        max_position=max_position,
    )
    model = _fit_ridge_multi(x_train[valid_train], y_train[valid_train], l2=spec.l2)
    if model is None:
        raise RuntimeError("failed to fit benchmark residual ridge model")

    pred_val = model.predict(x_val)
    zero_idx = int(np.argmin(np.abs(np.asarray(spec.deltas, dtype=np.float64))))
    best_val = np.argmax(pred_val, axis=1)
    improve_val = pred_val[np.arange(len(pred_val)), best_val] - pred_val[:, zero_idx]
    thresholds = _threshold_grid(improve_val, active_cap=spec.active_cap)

    best: dict[str, Any] | None = None
    for hold in spec.hold_grid:
        for cooldown in spec.cooldown_grid:
            for threshold in thresholds:
                val_positions, diag = _positions_from_residual_prediction(
                    pred_val,
                    base_val,
                    spec=spec,
                    threshold=threshold,
                    hold_bars=hold,
                    cooldown_bars=cooldown,
                    benchmark_position=benchmark_position,
                    min_position=min_position,
                    max_position=max_position,
                    max_total_turnover=spec.max_turnover,
                )
                val_stress = _stress_metrics(
                    returns=ds.val_returns,
                    positions=val_positions,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=benchmark_position,
                )
                score = _stress_selection_score(
                    val_stress,
                    spec,
                    "benchmark",
                    selection_stress_mode=selection_stress_mode,
                )
                rec = {
                    "source": "benchmark",
                    "spec": spec.name,
                    "threshold": "inf" if not math.isfinite(float(threshold)) else float(threshold),
                    "hold_bars": int(hold),
                    "cooldown_bars": int(cooldown),
                    "score": float(score),
                    "val": val_stress["cost_x1"],
                    "val_stress": val_stress,
                    "diag": diag,
                }
                if best is None or score > float(best["score"]):
                    best = rec
    if best is None:
        raise RuntimeError("failed to select benchmark residual extraction settings")

    pred_test = model.predict(x_test)
    threshold = float("inf") if best["threshold"] == "inf" else float(best["threshold"])
    test_positions, test_diag = _positions_from_residual_prediction(
        pred_test,
        base_test,
        spec=spec,
        threshold=threshold,
        hold_bars=int(best["hold_bars"]),
        cooldown_bars=int(best["cooldown_bars"]),
        benchmark_position=benchmark_position,
        min_position=min_position,
        max_position=max_position,
        max_total_turnover=spec.max_turnover,
    )
    test_stress = _stress_metrics(
        returns=ds.test_returns,
        positions=test_positions,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )
    return {
        "model": model,
        "spec": spec,
        "selection": best,
        "test_positions": test_positions,
        "test_diag": test_diag,
        "test_stress": test_stress,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.export_plan004_space_bundle")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--fold", type=int, default=13)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--spec", default="bc_resid_wide_riskoff_h16")
    parser.add_argument("--selection-stress-mode", choices=("primary", "include_costx3"), default="primary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg, None)
    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir="checkpoints/data_cache",
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, selected_folds = select_wfo_splits(build_wfo_splits(features_df, data_cfg), str(args.fold))
    if selected_folds != [args.fold] or len(splits) != 1:
        raise RuntimeError(f"failed to select fold {args.fold}: got {selected_folds}")
    split = splits[0]
    ds = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
    x_train_state = _state_features(ds.train_features, ds.train_returns)
    x_val_state = _state_features(ds.val_features, ds.val_returns)
    x_test_state = _state_features(ds.test_features, ds.test_returns)
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    min_position = float(cfg.get("ac", {}).get("abs_min_position", 0.0))
    max_position = float(cfg.get("ac", {}).get("abs_max_position", 1.25))
    fit = _fit_benchmark_residual(
        ds=ds,
        x_train_state=x_train_state,
        x_val_state=x_val_state,
        x_test_state=x_test_state,
        cfg=cfg,
        costs_cfg=costs_cfg,
        spec_name=args.spec,
        benchmark_position=benchmark_position,
        unit_cost=_unit_cost(costs_cfg),
        min_position=min_position,
        max_position=max_position,
        selection_stress_mode=args.selection_stress_mode,
    )
    model = fit["model"]
    spec = fit["spec"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_npz = output_dir / "policy_model.npz"
    np.savez(
        policy_npz,
        model_mean=np.asarray(model.mean, dtype=np.float64),
        model_std=np.asarray(model.std, dtype=np.float64),
        model_coef=np.asarray(model.coef, dtype=np.float64),
        deltas=np.asarray(spec.deltas, dtype=np.float64),
        threshold=np.asarray([float("inf") if fit["selection"]["threshold"] == "inf" else float(fit["selection"]["threshold"])], dtype=np.float64),
        hold_bars=np.asarray([int(fit["selection"]["hold_bars"])], dtype=np.int64),
        cooldown_bars=np.asarray([int(fit["selection"]["cooldown_bars"])], dtype=np.int64),
        benchmark_position=np.asarray([benchmark_position], dtype=np.float64),
        min_position=np.asarray([min_position], dtype=np.float64),
        max_position=np.asarray([max_position], dtype=np.float64),
        max_total_turnover=np.asarray([float(spec.max_turnover)], dtype=np.float64),
    )

    test_timestamps = _slice_index(features_df, split.test_start, split.test_end, right_inclusive=True)
    np.savez(
        output_dir / "sample_input.npz",
        features=np.asarray(ds.test_features, dtype=np.float32),
        returns=np.asarray(ds.test_returns, dtype=np.float32),
        timestamps=np.asarray(test_timestamps, dtype=object),
        expected_positions=np.asarray(fit["test_positions"], dtype=np.float32),
    )
    sample_last = float(fit["test_positions"][-1]) if len(fit["test_positions"]) else float("nan")
    sample_output = {
        "position": sample_last,
        "signal": "overweight" if sample_last > benchmark_position + 0.05 else ("underweight" if sample_last < benchmark_position - 0.05 else "benchmark"),
        "n_bars": int(len(fit["test_positions"])),
        "fold": int(args.fold),
        "source": "benchmark",
        "spec": spec.name,
    }
    with open(output_dir / "sample_output.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(sample_output, f, ensure_ascii=False, indent=2)

    shutil.copyfile(args.config, output_dir / "model_config.yaml")
    manifest = {
        "bundle_version": 2,
        "bundle_type": "plan004_residual_bc_ac",
        "created_by": "unidream.cli.export_plan004_space_bundle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "experiment": EXPERIMENT_NAME,
            "fold": int(args.fold),
            "seed": int(args.seed),
            "source": "benchmark",
            "spec": spec.name,
            "selection_stress_mode": args.selection_stress_mode,
            "no_leak_scope": "model fit uses train split; extraction settings use validation split; test split is sample verification only",
        },
        "data": {
            "symbol": symbol,
            "interval": interval,
            "zscore_window_days": int(zscore_window),
            "seq_len": int(cfg.get("data", {}).get("seq_len", 64)),
            "feature_columns": list(features_df.columns),
            "obs_dim": int(ds.obs_dim),
            "train_start": str(split.train_start),
            "train_end": str(split.train_end),
            "val_start": str(split.val_start),
            "val_end": str(split.val_end),
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
        },
        "policy": {
            "benchmark_position": benchmark_position,
            "min_position": min_position,
            "max_position": max_position,
            "deltas": list(map(float, spec.deltas)),
            "selection": fit["selection"],
        },
        "artifacts": {
            "policy_model": "policy_model.npz",
            "sample_input": "sample_input.npz",
            "sample_output": "sample_output.json",
        },
        "metrics": {
            "test_stress": fit["test_stress"],
            "test_diag": fit["test_diag"],
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(manifest), f, ensure_ascii=False, indent=2)

    report = {
        "manifest": manifest,
        "sample_output": sample_output,
    }
    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(_json_sanitize(report), f, ensure_ascii=False, indent=2)
    print(f"[export] wrote Plan004 space bundle to {output_dir}")
    print(json.dumps(_json_sanitize(fit["test_stress"]["cost_x1"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

