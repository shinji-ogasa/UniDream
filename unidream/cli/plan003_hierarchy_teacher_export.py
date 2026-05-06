from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import _state_features, _unit_cost
from unidream.cli.plan003_bc_student_probe import _compute_teacher, _json_sanitize
from unidream.cli.plan003_policy_blend_probe import EPS_DD_PT, _stress_metrics
from unidream.cli.round1_meta_label_probe import _fmt, _nanmax, _nanmean, _nanmin
from unidream.cli.round2_selector_audit_probe import _nanmedian
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "plan003_hierarchy_teacher_export"


SOURCE_IDS = {
    "benchmark": 0,
    "benchmark_safety": 0,
    "D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly": 1,
    "GR_baseline": 2,
    "recovery_rescue_fixed_state": 3,
    "micro_triple_fixed_raw": 4,
}


def _date_prefix() -> str:
    return datetime.now().strftime("%Y%m%d")


def _action_stats(pos: np.ndarray, benchmark: float) -> dict[str, Any]:
    p = np.asarray(pos, dtype=np.float64)
    active = np.abs(p - float(benchmark)) > 1e-12
    return {
        "n": int(len(p)),
        "active_rate": float(active.mean()) if len(p) else 0.0,
        "under_rate": float(np.mean(p < float(benchmark) - 1e-12)) if len(p) else 0.0,
        "over_rate": float(np.mean(p > float(benchmark) + 1e-12)) if len(p) else 0.0,
        "mean_position": float(np.mean(p)) if len(p) else float("nan"),
        "unique_positions": sorted(float(x) for x in np.unique(np.round(p, 8))),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stress in ("cost_x1", "cost_x1_5", "cost_x2", "slippage_x2"):
        metrics = [r["stress"][stress] for r in rows]
        alphas = [float(m["alpha_excess_pt"]) for m in metrics]
        dds = [float(m["maxdd_delta_pt"]) for m in metrics]
        turns = [float(m["turnover"]) for m in metrics]
        out[stress] = {
            "folds": len(rows),
            "pass_both_eps": int(sum(a > 0.0 and d <= EPS_DD_PT and t <= 3.5 for a, d, t in zip(alphas, dds, turns))),
            "alpha_mean": _nanmean(alphas),
            "alpha_median": _nanmedian(alphas),
            "alpha_worst": _nanmin(alphas),
            "maxdd_worst": _nanmax(dds),
            "turnover_max": _nanmax(turns),
        }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan003 Hierarchy Teacher Export",
        "",
        f"Bundle dir: `{payload['bundle_dir']}`",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        "",
        "## Aggregate",
        "",
        "| stress | pass | Alpha mean | Alpha median | Alpha worst | MaxDD worst | turnover max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stress, row in payload["aggregate"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    stress,
                    str(row["pass_both_eps"]),
                    _fmt(row["alpha_mean"]),
                    _fmt(row["alpha_median"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_max"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Fold Detail", ""])
    lines.append("| fold | source | AlphaEx | MaxDDDelta | turnover | train active | val active | test active | bundle |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["rows"]:
        m = row["stress"]["cost_x1"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["source"],
                    _fmt(m["alpha_excess_pt"], 6),
                    _fmt(m["maxdd_delta_pt"], 12),
                    _fmt(m["turnover"]),
                    _fmt(row["train_stats"]["active_rate"], 6),
                    _fmt(row["val_stats"]["active_rate"], 6),
                    _fmt(row["test_stats"]["active_rate"], 6),
                    row["bundle_path"],
                ]
            )
            + " |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan003_hierarchy_teacher_export")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--d-test-turnover-max", type=float, default=2.5)
    parser.add_argument("--bundle-dir", default="codex_outputs/plan003_hierarchy_teacher_bundle")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    date = _date_prefix()
    if not args.output_json:
        args.output_json = os.path.join("codex_outputs", f"{date}_{EXPERIMENT_NAME}.json")
    if not args.output_md:
        args.output_md = os.path.join("codex_outputs", f"{date}_{EXPERIMENT_NAME}.md")
    os.makedirs(args.bundle_dir, exist_ok=True)

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
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    unit_cost = _unit_cost(costs_cfg)

    rows: list[dict[str, Any]] = []
    for split in splits:
        fid = int(split.fold_idx)
        print(f"[HierarchyTeacherExport] fold={fid} start")
        ds = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
        x_train = _state_features(ds.train_features, ds.train_returns)
        x_val = _state_features(ds.val_features, ds.val_returns)
        x_test = _state_features(ds.test_features, ds.test_returns)
        teacher = _compute_teacher(
            ds=ds,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            ridge_l2=args.ridge_l2,
            seed=args.seed + fid * 100,
            max_train_samples=max(len(x_train), 50000),
            d_selector="D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly",
            gr_spec_name="GR_baseline",
            d_test_turnover_max=args.d_test_turnover_max,
            gr_val_alpha_min=0.30,
            gr_val_turnover_max=3.0,
            recovery_rescue_val_alpha_min=0.10,
            recovery_rescue_val_turnover_max=4.5,
            recovery_rescue_test_turnover_max=3.5,
            micro_triple_val_alpha_min=0.02,
            micro_triple_val_auc_min=0.50,
            micro_triple_val_turnover_max=0.5,
            micro_triple_test_turnover_max=0.7,
        )
        source = str(teacher["source"])
        source_id = int(SOURCE_IDS.get(source, -1))
        train_pos = np.asarray(teacher["train_positions"], dtype=np.float32)
        val_pos = np.asarray(teacher["val_positions"], dtype=np.float32)
        test_pos = np.asarray(teacher["test_positions"], dtype=np.float32)
        bundle_path = os.path.join(args.bundle_dir, f"fold{fid:02d}_teacher.npz")
        np.savez_compressed(
            bundle_path,
            train_positions=train_pos,
            val_positions=val_pos,
            test_positions=test_pos,
            source_id=np.asarray([source_id], dtype=np.int64),
            benchmark_position=np.asarray([benchmark_position], dtype=np.float32),
        )
        rows.append(
            {
                "fold": fid,
                "source": source,
                "source_id": source_id,
                "bundle_path": bundle_path,
                "train_stats": _action_stats(train_pos, benchmark_position),
                "val_stats": _action_stats(val_pos, benchmark_position),
                "test_stats": _action_stats(test_pos, benchmark_position),
                "stress": _stress_metrics(
                    returns=ds.test_returns,
                    positions=test_pos,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=benchmark_position,
                ),
                "meta": teacher["meta"],
            }
        )

    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": args.config,
        "folds": [int(s.fold_idx) for s in splits],
        "seed": int(args.seed),
        "d_test_turnover_max": float(args.d_test_turnover_max),
        "bundle_dir": args.bundle_dir,
        "source_ids": SOURCE_IDS,
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[HierarchyTeacherExport] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
