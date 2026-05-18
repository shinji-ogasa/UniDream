from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.plan003_policy_blend_probe import _stress_metrics
from unidream.data.dataset import WFODataset, WFOSplit
from unidream.experiments.runtime import load_config, resolve_costs, set_seed
from unidream.research.plan004_residual_bc_ac import run_plan004_fold_policy
from unidream.research.plan005_meta_guard_bc_ac import _build_past_features, guard_positions_from_features


EXPERIMENT_NAME = "plan006_guard_redesign_probe"


@dataclass(frozen=True)
class Plan006GuardConfig:
    arm_mom24576_max: float = 0.15
    arm_dd12288_max: float = -0.30
    val_alpha_min_pt: float = -15.0
    val_turnover_max: float = 45.0
    relative_loss_stop: float = -0.03
    guard_mode: str = "core_pair"


def _cost_per_step(delta_position: float, costs_cfg: dict[str, Any]) -> float:
    spread_bps = float(costs_cfg.get("spread_bps", 3.0))
    fee_rate = float(costs_cfg.get("fee_rate", 0.0003))
    slippage_bps = float(costs_cfg.get("slippage_bps", 1.0))
    delta = abs(float(delta_position))
    return (spread_bps / 10000.0) / 2.0 * delta + fee_rate * delta + (slippage_bps / 10000.0) * delta


def _apply_relative_loss_stop(
    desired: np.ndarray,
    base: np.ndarray,
    returns: np.ndarray,
    *,
    costs_cfg: dict[str, Any],
    stop: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    desired_arr = np.asarray(desired, dtype=np.float64)
    base_arr = np.asarray(base, dtype=np.float64)[: len(desired_arr)]
    ret = np.asarray(returns, dtype=np.float64)[: len(desired_arr)]
    out = np.empty(len(desired_arr), dtype=np.float64)
    rel = 0.0
    stopped_at: int | None = None
    active = True
    prev = float(base_arr[0]) if len(base_arr) else 1.0
    base_prev = prev
    for i, target in enumerate(desired_arr):
        chosen = float(target) if active else float(base_arr[i])
        out[i] = chosen
        rel += (
            chosen * ret[i]
            - _cost_per_step(chosen - prev, costs_cfg)
            - (float(base_arr[i]) * ret[i] - _cost_per_step(float(base_arr[i]) - base_prev, costs_cfg))
        )
        if active and rel < float(stop):
            active = False
            stopped_at = i
        prev = chosen
        base_prev = float(base_arr[i])
    return out, {"relative_loss_end": float(rel), "relative_loss_stopped_at": stopped_at}


def _start_diag(features: dict[str, np.ndarray], start_idx: int) -> dict[str, float]:
    idx = int(start_idx)

    def value(name: str) -> float:
        arr = features[name]
        return float(arr[min(max(idx, 0), len(arr) - 1)]) if len(arr) else 0.0

    return {
        "start_mom24576": value("mom24576"),
        "start_dd12288": value("dd12288"),
        "start_mom6144": value("mom6144"),
        "start_mom3072": value("mom3072"),
    }


def apply_plan006_guard(
    *,
    full_returns: np.ndarray,
    base_positions: np.ndarray,
    segment_mask: np.ndarray,
    val_mask: np.ndarray,
    cfg: dict[str, Any],
    guard_cfg: Plan006GuardConfig = Plan006GuardConfig(),
    min_position: float = 0.0,
    max_position: float = 1.25,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the Plan006 guard without using future segment returns for selection."""
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    features = _build_past_features(full_returns)
    seg_mask = np.asarray(segment_mask, dtype=bool)
    val_mask_arr = np.asarray(val_mask, dtype=bool)
    seg_idx = np.flatnonzero(seg_mask)
    if len(seg_idx) == 0:
        raise ValueError("segment_mask is empty")
    start_idx = int(seg_idx[0])
    diag = _start_diag(features, start_idx)
    armed = bool(
        diag["start_mom24576"] < float(guard_cfg.arm_mom24576_max)
        or diag["start_dd12288"] < float(guard_cfg.arm_dd12288_max)
    )

    guard_full = guard_positions_from_features(features, mode=guard_cfg.guard_mode)
    val_guard = guard_full[val_mask_arr]
    val_returns = np.asarray(full_returns, dtype=np.float64)[val_mask_arr]
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
    guard = guard_full[seg_mask][: len(base)]
    desired = np.clip(np.minimum(base[: len(guard)], guard), float(min_position), float(max_position))
    used = bool(armed and val_allowed)
    if used:
        positions, stop_diag = _apply_relative_loss_stop(
            desired,
            base,
            np.asarray(full_returns, dtype=np.float64)[seg_mask][: len(desired)],
            costs_cfg=costs_cfg,
            stop=float(guard_cfg.relative_loss_stop),
        )
    else:
        positions = base[: len(guard)].copy()
        stop_diag = {"relative_loss_end": 0.0, "relative_loss_stopped_at": None}
    diag.update(
        {
            "mode": "plan006_guard" if used else "base_passthrough",
            "armed": armed,
            "val_allowed": val_allowed,
            "val_alpha_excess_pt": float(val_metrics["alpha_excess_pt"]),
            "val_maxdd_delta_pt": float(val_metrics["maxdd_delta_pt"]),
            "val_turnover": float(val_metrics["turnover"]),
            "guard_mean": float(np.mean(guard)) if len(guard) else 0.0,
            "guard_underweight_rate": float(np.mean(guard < benchmark_position - 1e-9)) if len(guard) else 0.0,
            **stop_diag,
        }
    )
    return positions, diag


def _read_series(path: str) -> pd.Series:
    data = pd.read_parquet(path).squeeze()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(f"{path} must have a DatetimeIndex")
    return data.sort_index()


def _read_features(path: str) -> pd.DataFrame:
    data = pd.read_parquet(path)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(f"{path} must have a DatetimeIndex")
    return data.sort_index()


def _combine_oos_data(
    *,
    old_features: str,
    old_returns: str,
    new_features: str,
    new_returns: str,
    eval_start: str,
) -> tuple[pd.DataFrame, pd.Series]:
    start = pd.Timestamp(eval_start)
    old_feat = _read_features(old_features).loc[lambda x: x.index < start]
    old_ret = _read_series(old_returns).loc[lambda x: x.index < start]
    new_feat = _read_features(new_features).loc[lambda x: x.index >= start]
    new_ret = _read_series(new_returns).loc[lambda x: x.index >= start]
    features = pd.concat([old_feat, new_feat]).sort_index()
    returns = pd.concat([old_ret, new_ret]).sort_index()
    features = features[~features.index.duplicated(keep="last")]
    returns = returns[~returns.index.duplicated(keep="last")]
    returns = returns.loc[features.index]
    return features, returns


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
    guard_cfg = Plan006GuardConfig()
    stitched_idx = features.index[(features.index >= eval_start) & (features.index < eval_end)]
    stitched_returns = returns.loc[stitched_idx].to_numpy(dtype=np.float64)
    stitched_positions = np.full(len(stitched_idx), np.nan, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for fold_offset, test_start in enumerate(starts):
        test_end = min(test_start + pd.DateOffset(months=3), eval_end)
        split = WFOSplit(
            fold_idx=100 + fold_offset,
            train_start=test_start - pd.DateOffset(years=2, months=3),
            train_end=test_start - pd.DateOffset(months=3),
            val_start=test_start - pd.DateOffset(months=3),
            val_end=test_start,
            test_start=test_start,
            test_end=test_end,
        )
        ds = WFODataset(features, returns, split, seq_len=data_cfg.get("seq_len", 64))
        if args.base_mode == "plan004_retrain":
            base_rec = run_plan004_fold_policy(
                ds=ds,
                cfg=cfg,
                costs_cfg=cfg.get("costs", {}),
                fold_idx=split.fold_idx,
                seed=int(args.seed),
                ridge_l2=1.0,
                max_train_samples=50000,
                selection_stress_mode="primary",
            )
            base_positions = np.asarray(base_rec["positions"], dtype=np.float64)[: len(ds.test_returns)]
            source = f"{base_rec['selected_row'].get('source')}:{base_rec['selected_row'].get('spec')}"
        else:
            base_positions = np.full(len(ds.test_returns), benchmark_position, dtype=np.float64)
            source = "benchmark"
        segment_mask = (features.index >= split.test_start) & (features.index <= split.test_end)
        val_mask = (features.index >= split.val_start) & (features.index < split.val_end)
        positions, diag = apply_plan006_guard(
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
        row = {
            "fold": int(split.fold_idx),
            "source": source,
            "split": {k: str(getattr(split, k)) for k in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")},
            "stress": stress,
            "diag": diag,
        }
        rows.append(row)
        m = stress["cost_x1"]
        print(
            f"[Plan006] {test_start.date()} use={diag['mode']} "
            f"alpha={m['alpha_excess_pt']:+.2f} dd={m['maxdd_delta_pt']:+.2f} "
            f"val_alpha={diag['val_alpha_excess_pt']:+.2f}"
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
        "base_mode": args.base_mode,
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
            "selection": "segment start regime plus immediately preceding validation segment metrics",
            "test": "test metrics are report-only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan006_guard_redesign_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--old-features", default="checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v2_features.parquet")
    parser.add_argument("--old-returns", default="checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v2_returns.parquet")
    parser.add_argument("--new-features", default="checkpoints/data_cache/BTCUSDT_15m_2023-09-01_2026-03-18_z60_v2_oos_features.parquet")
    parser.add_argument("--new-returns", default="checkpoints/data_cache/BTCUSDT_15m_2023-09-01_2026-03-18_z60_v2_oos_returns.parquet")
    parser.add_argument("--eval-start", default="2024-01-01")
    parser.add_argument("--eval-end", default="2026-03-18")
    parser.add_argument("--last-test-start", default="2026-01-01")
    parser.add_argument("--base-mode", choices=("benchmark", "plan004_retrain"), default="benchmark")
    parser.add_argument("--output-json", default="docs_local/20260518_plan006_guard_redesign_oos_quarterly.json")
    args = parser.parse_args()
    payload = run_oos_quarterly_probe(args)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    print(f"[Plan006] wrote {args.output_json}")
    print(json.dumps(_json_sanitize(payload["stitched_stress"]["cost_x1"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
