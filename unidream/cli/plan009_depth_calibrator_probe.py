from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.plan003_policy_blend_probe import _stress_metrics
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.research.plan004_residual_bc_ac import run_plan004_fold_policy
from unidream.research.plan005_meta_guard_bc_ac import (
    _build_past_features,
    guard_positions_from_features,
    select_meta_guard_mode,
)


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _mask_for(index, start, end, *, inclusive_end: bool) -> np.ndarray:
    if inclusive_end:
        return np.asarray((index >= start) & (index <= end), dtype=bool)
    return np.asarray((index >= start) & (index < end), dtype=bool)


def _segment_start_idx(mask: np.ndarray) -> int:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(idx) == 0:
        raise ValueError("empty segment mask")
    return int(idx[0])


def _scaled_guard(guard: np.ndarray, *, depth: float, benchmark_position: float) -> np.ndarray:
    bench = float(benchmark_position)
    return bench - float(depth) * (bench - np.asarray(guard, dtype=np.float64))


def _load_base_cache(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == group]
        metrics = [r["stress"]["cost_x1"] for r in rs]
        alpha = np.asarray([float(m["alpha_excess_pt"]) for m in metrics], dtype=np.float64)
        dd = np.asarray([float(m["maxdd_delta_pt"]) for m in metrics], dtype=np.float64)
        turnover = np.asarray([float(m["turnover"]) for m in metrics], dtype=np.float64)
        out[group] = {
            "folds": int(len(rs)),
            "pass_alpha_ge3_dd_le_neg3": int(np.sum((alpha >= 3.0) & (dd <= -3.0))),
            "pass_alpha_ge10_dd_le_neg5": int(np.sum((alpha >= 10.0) & (dd <= -5.0))),
            "alpha_mean": float(np.mean(alpha)) if len(alpha) else float("nan"),
            "alpha_median": float(np.median(alpha)) if len(alpha) else float("nan"),
            "alpha_worst": float(np.min(alpha)) if len(alpha) else float("nan"),
            "maxdd_mean": float(np.mean(dd)) if len(dd) else float("nan"),
            "maxdd_worst": float(np.max(dd)) if len(dd) else float("nan"),
            "turnover_mean": float(np.mean(turnover)) if len(turnover) else float("nan"),
            "turnover_max": float(np.max(turnover)) if len(turnover) else float("nan"),
        }
    return out


def _stress_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    stresses = sorted({s for r in rows for s in r["stress"].keys()})
    groups = sorted({r["group"] for r in rows})
    for stress in stresses:
        out[stress] = {}
        for group in groups:
            rs = [r for r in rows if r["group"] == group]
            metrics = [r["stress"][stress] for r in rs]
            alpha = np.asarray([float(m["alpha_excess_pt"]) for m in metrics], dtype=np.float64)
            dd = np.asarray([float(m["maxdd_delta_pt"]) for m in metrics], dtype=np.float64)
            turnover = np.asarray([float(m["turnover"]) for m in metrics], dtype=np.float64)
            out[stress][group] = {
                "folds": int(len(rs)),
                "pass_alpha_ge3_dd_le_neg3": int(np.sum((alpha >= 3.0) & (dd <= -3.0))),
                "alpha_median": float(np.median(alpha)) if len(alpha) else float("nan"),
                "alpha_worst": float(np.min(alpha)) if len(alpha) else float("nan"),
                "maxdd_worst": float(np.max(dd)) if len(dd) else float("nan"),
                "turnover_max": float(np.max(turnover)) if len(turnover) else float("nan"),
            }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan009 Depth Calibrator Probe",
        "",
        "Validation-calibrated scalar depth over the Plan005 past-only guard.",
        "",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Seed: `{payload['seed']}`",
        f"Val DD target: `{payload['val_dd_target']}`",
        f"Safety multiplier: `{payload['safety_multiplier']}`",
        f"Max depth cap: `{payload['max_depth_cap']}`",
        "",
        "## Aggregate: cost_x1",
        "",
        "| group | folds | pass +3/-3 | pass +10/-5 | Alpha mean | Alpha median | Alpha worst | MaxDD mean | MaxDD worst | TO mean | TO max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in payload["aggregate"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(row["folds"]),
                    str(row["pass_alpha_ge3_dd_le_neg3"]),
                    str(row["pass_alpha_ge10_dd_le_neg5"]),
                    _fmt(row["alpha_mean"]),
                    _fmt(row["alpha_median"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["maxdd_mean"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_mean"]),
                    _fmt(row["turnover_max"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Fold Detail: cost_x1", ""])
    lines.append("| fold | group | AlphaEx | MaxDDDelta | TO | depth | val required | val Alpha | val MaxDD | val TO |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["rows"]:
        m = row["stress"]["cost_x1"]
        d = row.get("diag", {})
        v = d.get("val", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["group"],
                    _fmt(m["alpha_excess_pt"]),
                    _fmt(m["maxdd_delta_pt"]),
                    _fmt(m["turnover"]),
                    _fmt(d.get("selected_depth")),
                    _fmt(d.get("val_required_depth")),
                    _fmt(v.get("alpha_excess_pt")),
                    _fmt(v.get("maxdd_delta_pt")),
                    _fmt(v.get("turnover")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Stress Aggregate", ""])
    for stress, groups in payload["stress_aggregate"].items():
        lines.extend(
            [
                f"### {stress}",
                "",
                "| group | folds | pass +3/-3 | Alpha median | Alpha worst | MaxDD worst | TO max |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group, row in groups.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        str(row["folds"]),
                        str(row["pass_alpha_ge3_dd_le_neg3"]),
                        _fmt(row["alpha_median"]),
                        _fmt(row["alpha_worst"]),
                        _fmt(row["maxdd_worst"]),
                        _fmt(row["turnover_max"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Leak Discipline",
            "",
            "- Base positions are Plan004 fold-local policies.",
            "- Guard modes are selected from shifted past features at segment start.",
            "- Required depth is selected on validation only, then multiplied and capped before test.",
            "- Test metrics are report-only; fold0-12 is still a development set.",
        ]
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan009_depth_calibrator_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--base-positions-cache", default="docs_local/20260527_plan005_meta_guard_f0_12_positions.npz")
    parser.add_argument("--depth-grid", default="0.25:1.0:0.01")
    parser.add_argument("--val-dd-target", type=float, default=-5.0)
    parser.add_argument("--safety-multiplier", type=float, default=2.0)
    parser.add_argument("--min-depth-floor", type=float, default=0.30)
    parser.add_argument("--max-depth-cap", type=float, default=0.94)
    parser.add_argument("--selection-stress-mode", choices=("primary", "include_costx3"), default="primary")
    parser.add_argument("--output-json", default="docs_local/20260528_plan009_depth_calibrator.json")
    parser.add_argument("--output-md", default="docs_local/20260528_plan009_depth_calibrator.md")
    args = parser.parse_args()
    set_seed(int(args.seed))

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
    splits, selected_folds = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    full_returns = np.asarray(raw_returns, dtype=np.float64)
    past_features = _build_past_features(full_returns)
    base_cache = _load_base_cache(args.base_positions_cache)
    if ":" in str(args.depth_grid):
        start_s, end_s, step_s = str(args.depth_grid).split(":")
        depths = np.round(np.arange(float(start_s), float(end_s) + 1e-12, float(step_s)), 6)
    else:
        depths = np.asarray([float(x) for x in str(args.depth_grid).split(",") if x.strip()], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for cache_idx, split in enumerate(splits):
        fid = int(split.fold_idx)
        print(f"[Plan009Calib] fold={fid} start")
        ds = WFODataset(features_df, raw_returns, split, seq_len=data_cfg.get("seq_len", 64))
        val_mask = _mask_for(features_df.index, split.val_start, split.val_end, inclusive_end=False)
        test_mask = _mask_for(features_df.index, split.test_start, split.test_end, inclusive_end=True)
        val_mode, _val_diag = select_meta_guard_mode(past_features, _segment_start_idx(val_mask))
        test_mode, test_diag = select_meta_guard_mode(past_features, _segment_start_idx(test_mask))
        guard_val = guard_positions_from_features(past_features, mode=val_mode)[val_mask][: len(ds.val_returns)]
        guard_test = guard_positions_from_features(past_features, mode=test_mode)[test_mask][: len(ds.test_returns)]
        if base_cache is not None:
            base_test = np.asarray(base_cache["base_positions"][cache_idx], dtype=np.float64)[: len(ds.test_returns)]
            base_source = str(base_cache["selected"][cache_idx]) if "selected" in base_cache else "cache"
        else:
            rec = run_plan004_fold_policy(
                ds=ds,
                cfg=cfg,
                costs_cfg=costs_cfg,
                fold_idx=fid,
                seed=int(args.seed),
                ridge_l2=float(args.ridge_l2),
                max_train_samples=int(args.max_train_samples),
                selection_stress_mode=str(args.selection_stress_mode),
            )
            base_test = np.asarray(rec["positions"], dtype=np.float64)[: len(ds.test_returns)]
            base_source = f"{rec['selected_row'].get('source')}:{rec['selected_row'].get('spec')}"

        base_stress = _stress_metrics(
            returns=ds.test_returns[: len(base_test)],
            positions=base_test,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        rows.append(
            {
                "fold": fid,
                "group": "plan004_base",
                "source": base_source,
                "stress": base_stress,
                "diag": {},
            }
        )

        required_depth = float(depths[-1])
        required_val: dict[str, Any] | None = None
        for depth in depths:
            val_pos = _scaled_guard(guard_val, depth=float(depth), benchmark_position=benchmark_position)
            val_stress = _stress_metrics(
                returns=ds.val_returns[: len(val_pos)],
                positions=val_pos,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )
            val_metrics = val_stress["cost_x1"]
            if float(val_metrics["maxdd_delta_pt"]) <= float(args.val_dd_target):
                required_depth = float(depth)
                required_val = val_metrics
                break
        if required_val is None:
            val_pos = _scaled_guard(guard_val, depth=required_depth, benchmark_position=benchmark_position)
            required_val = _stress_metrics(
                returns=ds.val_returns[: len(val_pos)],
                positions=val_pos,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )["cost_x1"]
        selected_depth = min(
            float(args.max_depth_cap),
            max(float(args.min_depth_floor), required_depth * float(args.safety_multiplier)),
        )
        guard_scaled = _scaled_guard(guard_test, depth=selected_depth, benchmark_position=benchmark_position)
        positions = np.minimum(base_test[: len(guard_scaled)], guard_scaled)
        stress = _stress_metrics(
            returns=ds.test_returns[: len(positions)],
            positions=positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        m = stress["cost_x1"]
        print(
            f"[Plan009Calib] fold={fid} depth={selected_depth:.2f} "
            f"alpha={m['alpha_excess_pt']:+.2f} dd={m['maxdd_delta_pt']:+.2f} to={m['turnover']:.2f}"
        )
        rows.append(
            {
                "fold": fid,
                "group": "plan009_depth_calibrator",
                "source": base_source,
                "stress": stress,
                "diag": {
                    "val_mode": val_mode,
                    "test_mode": test_mode,
                    "val_required_depth": required_depth,
                    "selected_depth": selected_depth,
                    "val": required_val,
                    **test_diag,
                },
            }
        )

    payload = {
        "experiment": "plan009_depth_calibrator_probe",
        "seed": int(args.seed),
        "config": args.config,
        "folds": selected_folds,
        "base_positions_cache": args.base_positions_cache,
        "val_dd_target": float(args.val_dd_target),
        "safety_multiplier": float(args.safety_multiplier),
        "min_depth_floor": float(args.min_depth_floor),
        "max_depth_cap": float(args.max_depth_cap),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "stress_aggregate": _stress_aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[Plan009Calib] wrote {args.output_json}")
    print(f"[Plan009Calib] wrote {args.output_md}")


if __name__ == "__main__":
    main()
