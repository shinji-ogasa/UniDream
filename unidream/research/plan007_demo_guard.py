from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.plan003_policy_blend_probe import _stress_metrics
from unidream.data.dataset import WFODataset, WFOSplit
from unidream.experiments.runtime import load_config, resolve_costs, set_seed
from unidream.research.plan005_meta_guard_bc_ac import _build_past_features, guard_positions_from_features
from unidream.research.plan006_guard_redesign import _apply_relative_loss_stop, _combine_oos_data, _start_diag


EXPERIMENT_NAME = "plan007_demo_guard_probe"


@dataclass(frozen=True)
class Plan007GuardConfig:
    arm_mom24576_min: float = -0.25
    arm_mom24576_max: float = 0.15
    arm_dd12288_max: float = -0.30
    val_alpha_min_pt: float = -15.0
    val_turnover_max: float = 45.0
    relative_loss_stop: float = -0.03
    riskoff_scale: float = 3.0
    guard_mode: str = "core_pair"


def _scaled_guard(guard: np.ndarray, *, benchmark_position: float, riskoff_scale: float) -> np.ndarray:
    guard_arr = np.asarray(guard, dtype=np.float64)
    bench = float(benchmark_position)
    scaled = bench - float(riskoff_scale) * (bench - guard_arr)
    return np.clip(scaled, 0.0, bench)


def apply_plan007_guard(
    *,
    full_returns: np.ndarray,
    base_positions: np.ndarray,
    segment_mask: np.ndarray,
    val_mask: np.ndarray,
    cfg: dict[str, Any],
    guard_cfg: Plan007GuardConfig = Plan007GuardConfig(),
    min_position: float = 0.0,
    max_position: float = 1.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the Plan007 demo guard using only past features and prior validation returns."""
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    returns_arr = np.asarray(full_returns, dtype=np.float64)
    features = _build_past_features(returns_arr)
    seg_mask = np.asarray(segment_mask, dtype=bool)
    val_mask_arr = np.asarray(val_mask, dtype=bool)
    seg_idx = np.flatnonzero(seg_mask)
    if len(seg_idx) == 0:
        raise ValueError("segment_mask is empty")

    start_idx = int(seg_idx[0])
    diag = _start_diag(features, start_idx)
    raw_armed = bool(
        diag["start_mom24576"] < float(guard_cfg.arm_mom24576_max)
        or diag["start_dd12288"] < float(guard_cfg.arm_dd12288_max)
    )
    late_bear_blocked = bool(diag["start_mom24576"] < float(guard_cfg.arm_mom24576_min))
    armed = bool(raw_armed and not late_bear_blocked)

    guard_full = guard_positions_from_features(features, mode=guard_cfg.guard_mode)
    val_guard = guard_full[val_mask_arr]
    val_returns = returns_arr[val_mask_arr]
    val_metrics = _stress_metrics(
        returns=val_returns,
        positions=val_guard,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )["cost_x1"]
    val_allowed = bool(
        float(val_metrics["alpha_excess_pt"]) >= float(guard_cfg.val_alpha_min_pt)
        and float(val_metrics["turnover"]) <= float(guard_cfg.val_turnover_max)
    )

    base = np.asarray(base_positions, dtype=np.float64)
    raw_guard = guard_full[seg_mask][: len(base)]
    scaled_guard = _scaled_guard(
        raw_guard,
        benchmark_position=benchmark_position,
        riskoff_scale=guard_cfg.riskoff_scale,
    )
    desired = np.clip(np.minimum(base[: len(scaled_guard)], scaled_guard), float(min_position), float(max_position))
    used = bool(armed and val_allowed)
    if used:
        positions, stop_diag = _apply_relative_loss_stop(
            desired,
            base,
            returns_arr[seg_mask][: len(desired)],
            costs_cfg=costs_cfg,
            stop=float(guard_cfg.relative_loss_stop),
        )
    else:
        positions = base[: len(scaled_guard)].copy()
        stop_diag = {"relative_loss_end": 0.0, "relative_loss_stopped_at": None}

    diag.update(
        {
            "mode": "plan007_demo_guard" if used else "base_passthrough",
            "raw_armed": raw_armed,
            "late_bear_blocked": late_bear_blocked,
            "armed": armed,
            "val_allowed": val_allowed,
            "val_alpha_excess_pt": float(val_metrics["alpha_excess_pt"]),
            "val_maxdd_delta_pt": float(val_metrics["maxdd_delta_pt"]),
            "val_turnover": float(val_metrics["turnover"]),
            "raw_guard_mean": float(np.mean(raw_guard)) if len(raw_guard) else 0.0,
            "scaled_guard_mean": float(np.mean(scaled_guard)) if len(scaled_guard) else 0.0,
            "scaled_guard_underweight_rate": float(np.mean(scaled_guard < benchmark_position - 1e-9))
            if len(scaled_guard)
            else 0.0,
            **stop_diag,
        }
    )
    return positions, diag


def run_oos_quarterly_probe(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg, None)
    data_cfg = cfg.get("data", {})
    features, returns = _combine_oos_data(
        old_features=args.old_features,
        old_returns=args.old_returns,
        new_features=args.new_features,
        new_returns=args.new_returns,
        eval_start=args.eval_start,
    )
    eval_start = pd.Timestamp(args.eval_start)
    eval_end = pd.Timestamp(args.eval_end)
    starts = list(pd.date_range(eval_start, pd.Timestamp(args.last_test_start), freq="3MS"))
    full_returns = returns.to_numpy(dtype=np.float64)
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    min_position = float(cfg.get("ac", {}).get("abs_min_position", 0.0))
    max_position = float(cfg.get("ac", {}).get("abs_max_position", 1.25))
    guard_cfg = Plan007GuardConfig()
    stitched_idx = features.index[(features.index >= eval_start) & (features.index < eval_end)]
    stitched_returns = returns.loc[stitched_idx].to_numpy(dtype=np.float64)
    stitched_positions = np.full(len(stitched_idx), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for fold_offset, test_start in enumerate(starts):
        test_end = min(test_start + pd.DateOffset(months=3), eval_end)
        split = WFOSplit(
            fold_idx=200 + fold_offset,
            train_start=test_start - pd.DateOffset(years=2, months=3),
            train_end=test_start - pd.DateOffset(months=3),
            val_start=test_start - pd.DateOffset(months=3),
            val_end=test_start,
            test_start=test_start,
            test_end=test_end,
        )
        ds = WFODataset(features, returns, split, seq_len=data_cfg.get("seq_len", 64))
        base_positions = np.full(len(ds.test_returns), benchmark_position, dtype=np.float64)
        segment_mask = (features.index >= split.test_start) & (features.index <= split.test_end)
        val_mask = (features.index >= split.val_start) & (features.index < split.val_end)
        positions, diag = apply_plan007_guard(
            full_returns=full_returns,
            base_positions=base_positions,
            segment_mask=np.asarray(segment_mask, dtype=bool),
            val_mask=np.asarray(val_mask, dtype=bool),
            cfg=cfg,
            guard_cfg=guard_cfg,
            min_position=min_position,
            max_position=max_position,
        )
        stress = _stress_metrics(
            returns=ds.test_returns[: len(positions)],
            positions=positions,
            cfg=cfg,
            costs_cfg=cfg.get("costs", {}),
            benchmark_position=benchmark_position,
        )
        out_mask = (stitched_idx >= split.test_start) & (stitched_idx < split.test_end)
        stitched_positions[out_mask] = positions[: int(out_mask.sum())]
        rows.append(
            {
                "fold": int(split.fold_idx),
                "source": "benchmark",
                "split": {
                    key: str(getattr(split, key))
                    for key in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")
                },
                "stress": stress,
                "diag": diag,
            }
        )
        m = stress["cost_x1"]
        print(
            f"[Plan007] {test_start.date()} use={diag['mode']} "
            f"alpha={m['alpha_excess_pt']:+.2f} dd={m['maxdd_delta_pt']:+.2f} "
            f"val_alpha={diag['val_alpha_excess_pt']:+.2f} blocked={diag['late_bear_blocked']}"
        )

    if np.isnan(stitched_positions).any():
        raise RuntimeError(f"stitched positions have {int(np.isnan(stitched_positions).sum())} NaN rows")
    stitched_stress = _stress_metrics(
        returns=stitched_returns,
        positions=stitched_positions,
        cfg=cfg,
        costs_cfg=cfg.get("costs", {}),
        benchmark_position=benchmark_position,
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "base_mode": "benchmark",
        "guard_config": asdict(guard_cfg),
        "period": {
            "start": str(stitched_idx.min()),
            "end": str(stitched_idx.max()),
            "n_bars": int(len(stitched_idx)),
        },
        "rows": rows,
        "stitched_stress": stitched_stress,
        "position_summary": {
            "mean": float(np.mean(stitched_positions)),
            "min": float(np.min(stitched_positions)),
            "max": float(np.max(stitched_positions)),
            "underweight_rate": float(np.mean(stitched_positions < benchmark_position - 1e-9)),
            "benchmark_rate": float(np.mean(np.abs(stitched_positions - benchmark_position) <= 1e-9)),
            "turnover": float(np.abs(np.diff(stitched_positions, prepend=benchmark_position)).sum()),
        },
        "leak_discipline": {
            "guard_features": "shifted rolling returns only",
            "selection": "calendar segment start regime plus immediately preceding validation segment metrics",
            "test": "test metrics are report-only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan007_demo_guard_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--old-features", default="checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v2_features.parquet")
    parser.add_argument("--old-returns", default="checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v2_returns.parquet")
    parser.add_argument("--new-features", default="checkpoints/data_cache/BTCUSDT_15m_2023-09-01_2026-05-18_z60_v2_oos_latest_features.parquet")
    parser.add_argument("--new-returns", default="checkpoints/data_cache/BTCUSDT_15m_2023-09-01_2026-05-18_z60_v2_oos_latest_returns.parquet")
    parser.add_argument("--eval-start", default="2024-01-01")
    parser.add_argument("--eval-end", default="2026-05-18")
    parser.add_argument("--last-test-start", default="2026-04-01")
    parser.add_argument("--output-json", default="docs_local/20260518_plan007_demo_guard_oos_quarterly_to_20260518.json")
    args = parser.parse_args()
    payload = run_oos_quarterly_probe(args)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    print(f"[Plan007] wrote {args.output_json}")
    print(json.dumps(_json_sanitize(payload["stitched_stress"]["cost_x1"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
