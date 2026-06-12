from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from unidream.cli.plan011_lowfreq_overlay_probe import (
    OverlaySpec,
    _fmt,
    _run_fold,
    _spec_from_config,
)
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.train_app import resolve_cache_dir
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


def _strict_grid(quick: bool) -> list[OverlaySpec]:
    specs: list[OverlaySpec] = [_noop_spec()]
    base_overlays = (0.0, 0.01, 0.015, 0.02, 0.03) if not quick else (0.0, 0.02)
    base_safe_modes = (False, True)
    ema_spans = (32, 64, 128, 256) if not quick else (64, 128)
    risk_his = (0.0, 0.5, 1.0, 1.5) if not quick else (0.5, 1.0)
    risk_los = (-0.75, -0.25, 0.0) if not quick else (-0.5,)
    edge_his = (-0.25, 0.0, 0.25, 99.0) if not quick else (0.0, 99.0)
    edge_los = (-99.0, -0.5, -0.25, 0.0, 0.25) if not quick else (-99.0, 0.0)
    trend_lookbacks = (64, 128, 256, 512) if not quick else (128, 256)
    trend_maxes = (-0.08, -0.04, -0.02, 0.0, 0.02) if not quick else (-0.04, 0.0)
    trend_mins = (-0.02, 0.0, 0.02) if not quick else (0.0,)
    dd_lookbacks = (0, 64, 128, 256, 512) if not quick else (0, 128, 256)
    dd_mins = (0.0, 0.015, 0.03, 0.05, 0.08) if not quick else (0.0, 0.03)
    dd_maxes = (0.04, 0.08, 0.12, np.inf) if not quick else (0.08,)
    down_overlays = (0.04, 0.08, 0.15, 0.30, 0.50, 0.70) if not quick else (0.15, 0.50)
    up_overlays = (0.0, 0.02, 0.04, 0.08) if not quick else (0.04,)
    max_steps = (0.01, 0.02, 0.04) if not quick else (0.02,)
    min_holds = (16, 32, 64) if not quick else (32,)
    for base_overlay in base_overlays:
        for base_safe_only in base_safe_modes:
            for ema_span in ema_spans:
                for risk_hi in risk_his:
                    for risk_lo in risk_los:
                        for edge_hi in edge_his:
                            for edge_lo in edge_los:
                                if edge_hi < edge_lo and edge_hi < 90.0 and edge_lo > -90.0:
                                    continue
                                for trend_lookback in trend_lookbacks:
                                    for trend_max in trend_maxes:
                                        for trend_min in trend_mins:
                                            for dd_lookback in dd_lookbacks:
                                                for dd_min in dd_mins:
                                                    if dd_lookback <= 0 and dd_min > 0.0:
                                                        continue
                                                    for dd_max in dd_maxes:
                                                        for down_overlay in down_overlays:
                                                            for up_overlay in up_overlays:
                                                                for max_step in max_steps:
                                                                    for min_hold in min_holds:
                                                                        specs.append(
                                                                            OverlaySpec(
                                                                                base_overlay=base_overlay,
                                                                                base_safe_only=base_safe_only,
                                                                                ema_span=ema_span,
                                                                                risk_hi=risk_hi,
                                                                                risk_lo=risk_lo,
                                                                                edge_hi=edge_hi,
                                                                                edge_lo=edge_lo,
                                                                                trend_lookback=trend_lookback,
                                                                                trend_max=trend_max,
                                                                                trend_min=trend_min,
                                                                                dd_lookback=dd_lookback,
                                                                                dd_min=dd_min,
                                                                                dd_max=dd_max,
                                                                                down_overlay=down_overlay,
                                                                                up_overlay=up_overlay,
                                                                                max_step=max_step,
                                                                                min_hold=min_hold,
                                                                                deadzone=0.01,
                                                                            )
                                                                        )
    dedup: dict[str, OverlaySpec] = {}
    for spec in specs:
        dedup[json.dumps(asdict(spec), sort_keys=True)] = spec
    return list(dedup.values())


def _noop_spec() -> OverlaySpec:
    return OverlaySpec(
        base_overlay=0.0,
        base_safe_only=False,
        ema_span=128,
        risk_hi=99.0,
        risk_lo=-99.0,
        edge_hi=99.0,
        edge_lo=-99.0,
        trend_lookback=128,
        trend_max=0.0,
        trend_min=0.0,
        dd_lookback=0,
        dd_min=0.0,
        dd_max=np.inf,
        down_overlay=0.0,
        up_overlay=0.0,
        max_step=0.01,
        min_hold=32,
        deadzone=0.01,
    )


def _spec_key(spec: dict[str, Any]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def _candidate_score(cand: dict[str, Any], *, alpha_floor: float, dd_target: float, max_turnover: float) -> float:
    vals = [cand["val"], cand["val_first"], cand["val_second"]]
    alpha_full = float(cand["val"]["alpha_excess_pt"])
    alpha_min = min(float(v["alpha_excess_pt"]) for v in vals)
    dd_full = float(cand["val"]["maxdd_delta_pt"])
    dd_worst = max(float(v["maxdd_delta_pt"]) for v in vals)
    dd_best = min(float(v["maxdd_delta_pt"]) for v in vals)
    turnover = float(cand["val"]["turnover"])
    score = 3.0 * alpha_full + 3.0 * alpha_min + 80.0 * max(0.0, -dd_best) - 80.0 * max(0.0, dd_worst)
    score -= 3.0 * turnover
    if alpha_min < alpha_floor:
        score -= 250.0 + 35.0 * (alpha_floor - alpha_min)
    if dd_worst > dd_target:
        score -= 250.0 + 60.0 * (dd_worst - dd_target)
    if turnover > max_turnover:
        score -= 200.0 + 25.0 * (turnover - max_turnover)
    if abs(float(cand["spec"]["base_overlay"])) < 1e-9 and float(cand["spec"]["down_overlay"]) < 1e-9 and float(cand["spec"]["up_overlay"]) < 1e-9:
        score -= 25.0
    return float(score)


def _rank_global(
    results: list[dict[str, Any]], *, alpha_floor: float, dd_target: float, max_turnover: float
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in results:
        for cand in row["candidates"]:
            key = _spec_key(cand["spec"])
            item = grouped.setdefault(
                key,
                {"spec": cand["spec"], "folds": [], "global_score": 0.0, "alpha_min": float("inf"), "dd_worst": -float("inf")},
            )
            score = _candidate_score(cand, alpha_floor=alpha_floor, dd_target=dd_target, max_turnover=max_turnover)
            vals = [cand["val"], cand["val_first"], cand["val_second"]]
            item["folds"].append({"fold": row["fold"], "val": cand["val"], "val_first": cand["val_first"], "val_second": cand["val_second"], "score": score})
            item["global_score"] = float(item["global_score"]) + score
            item["alpha_min"] = min(float(item["alpha_min"]), *(float(v["alpha_excess_pt"]) for v in vals))
            item["dd_worst"] = max(float(item["dd_worst"]), *(float(v["maxdd_delta_pt"]) for v in vals))
    if not grouped:
        raise RuntimeError("no candidates for global selector")
    ranked = sorted(grouped.values(), key=lambda x: float(x["global_score"]), reverse=True)
    counts = {
        "total": len(ranked),
        "alpha_floor_all": sum(1 for x in ranked if float(x["alpha_min"]) >= alpha_floor),
        "dd_target_all": sum(1 for x in ranked if float(x["dd_worst"]) <= dd_target),
        "alpha_and_dd_all": sum(
            1 for x in ranked if float(x["alpha_min"]) >= alpha_floor and float(x["dd_worst"]) <= dd_target
        ),
        "alpha_nonnegative_dd_nonpositive": sum(
            1 for x in ranked if float(x["alpha_min"]) >= 0.0 and float(x["dd_worst"]) <= 0.0
        ),
    }
    return ranked, counts


def _best_per_fold(results: list[dict[str, Any]], *, alpha_floor: float, dd_target: float, max_turnover: float) -> list[dict[str, Any]]:
    out = []
    for row in results:
        best = max(
            row["candidates"],
            key=lambda c: _candidate_score(c, alpha_floor=alpha_floor, dd_target=dd_target, max_turnover=max_turnover),
        )
        out.append({"fold": row["fold"], "selected": best})
    return out


def _per_fold_counts(results: list[dict[str, Any]], *, alpha_floor: float, dd_target: float) -> list[dict[str, Any]]:
    rows = []
    for row in results:
        counts = {
            "fold": row["fold"],
            "total": len(row["candidates"]),
            "alpha_floor": 0,
            "dd_target": 0,
            "alpha_and_dd": 0,
            "alpha_nonnegative_dd_nonpositive": 0,
            "top_dd": [],
        }
        ranked_dd = []
        for cand in row["candidates"]:
            vals = [cand["val"], cand["val_first"], cand["val_second"]]
            alpha_min = min(float(v["alpha_excess_pt"]) for v in vals)
            dd_worst = max(float(v["maxdd_delta_pt"]) for v in vals)
            if alpha_min >= alpha_floor:
                counts["alpha_floor"] += 1
            if dd_worst <= dd_target:
                counts["dd_target"] += 1
            if alpha_min >= alpha_floor and dd_worst <= dd_target:
                counts["alpha_and_dd"] += 1
            if alpha_min >= 0.0 and dd_worst <= 0.0:
                counts["alpha_nonnegative_dd_nonpositive"] += 1
            ranked_dd.append({"spec": cand["spec"], "val": cand["val"], "val_first": cand["val_first"], "val_second": cand["val_second"], "alpha_min": alpha_min, "dd_worst": dd_worst})
        counts["top_dd"] = sorted(ranked_dd, key=lambda x: (x["dd_worst"], -x["alpha_min"]))[:10]
        rows.append(counts)
    return rows


def _write_report(payload: dict[str, Any], output_md: str) -> None:
    lines = [
        "# Plan011 Alpha-Floor Overlay Selector Probe",
        "",
        f"- config: `{payload['config']}`",
        f"- checkpoint_dir: `{payload['checkpoint_dir']}`",
        f"- folds: `{', '.join(map(str, payload['folds']))}`",
        f"- alpha_floor: `{payload['alpha_floor']}`",
        f"- dd_target: `{payload['dd_target']}`",
        "",
        "## Global Candidate Counts",
        "",
        "| total | alpha floor all | dd target all | alpha+dd all | alpha>=0 & dd<=0 |",
        "|---:|---:|---:|---:|---:|",
        f"| {payload['global_counts']['total']} | {payload['global_counts']['alpha_floor_all']} | "
        f"{payload['global_counts']['dd_target_all']} | {payload['global_counts']['alpha_and_dd_all']} | "
        f"{payload['global_counts']['alpha_nonnegative_dd_nonpositive']} |",
        "",
        "## Global Selected Test",
        "",
        "| fold | val AlphaEx | val MaxDDDelta | test AlphaEx | test MaxDDDelta | test TO |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["global_test"]:
        lines.append(
            f"| {row['fold']} | {_fmt(row['selected']['val']['alpha_excess_pt'])} | "
            f"{_fmt(row['selected']['val']['maxdd_delta_pt'])} | {_fmt(row['test']['alpha_excess_pt'])} | "
            f"{_fmt(row['test']['maxdd_delta_pt'])} | {row['test']['turnover']:.2f} |"
        )
    lines.extend([
        "",
        "## Per-Fold Strict Val Winners",
        "",
        "| fold | val AlphaEx | val MaxDDDelta | val split AlphaEx | spec |",
        "|---:|---:|---:|---:|---|",
    ])
    for row in payload["per_fold_val"]:
        spec = row["selected"]["spec"]
        lines.append(
            f"| {row['fold']} | {_fmt(row['selected']['val']['alpha_excess_pt'])} | "
            f"{_fmt(row['selected']['val']['maxdd_delta_pt'])} | "
            f"{_fmt(row['selected']['val_first']['alpha_excess_pt'])}/{_fmt(row['selected']['val_second']['alpha_excess_pt'])} | "
            f"base={spec['base_overlay']} safe={spec['base_safe_only']} ema={spec['ema_span']} "
            f"risk={spec['risk_hi']} edge={spec['edge_lo']} trend={spec['trend_max']} "
            f"dd={spec['dd_lookback']}:{spec['dd_min']} down={spec['down_overlay']} up={spec['up_overlay']} |"
        )
    Path(output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/plan011_overlay_actor_v14_edgewm_bconly.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/plan011_overlay_actor_v14_edgewm_bconly_s007")
    parser.add_argument("--folds", default="3,4,5")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--quick-grid", action="store_true")
    parser.add_argument("--include-config-spec", action="store_true")
    parser.add_argument("--alpha-floor", type=float, default=3.0)
    parser.add_argument("--dd-target", type=float, default=-3.0)
    parser.add_argument("--max-turnover", type=float, default=6.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg)
    set_seed(args.seed)
    symbol = cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_dir = resolve_cache_dir(args.checkpoint_dir, cfg)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=cfg.get("data", {}).get("extra_series_mode", "derived"),
        extra_series_include=cfg.get("data", {}).get("extra_series_include"),
        include_funding=bool(cfg.get("data", {}).get("include_funding", True)),
        include_oi=bool(cfg.get("data", {}).get("include_oi", True)),
        include_mark=bool(cfg.get("data", {}).get("include_mark", True)),
    )
    splits, selected = select_wfo_splits(build_wfo_splits(features_df, cfg.get("data", {})), args.folds)
    specs = _strict_grid(quick=bool(args.quick_grid))
    if args.include_config_spec:
        specs.append(_spec_from_config(cfg))
    risk_indices = tuple(int(i) for i in cfg.get("bc", {}).get("benchmark_overlay_risk_gate_indices", [10, 11, 12, 13, 15, 16, 17, 18]))
    edge_source = cfg.get("bc", {}).get("benchmark_overlay_edge_indices")
    if not edge_source:
        edge_source = cfg.get("bc", {}).get("benchmark_overlay_edge_protect_indices", [])
    edge_indices = tuple(int(i) for i in edge_source or [])
    val_results = [
        _run_fold(
            split,
            features_df,
            raw_returns,
            cfg,
            args.device,
            args.checkpoint_dir,
            specs,
            risk_indices,
            edge_indices,
            include_candidates=True,
        )
        for split in splits
    ]
    ranked_global, global_counts = _rank_global(
        val_results,
        alpha_floor=float(args.alpha_floor),
        dd_target=float(args.dd_target),
        max_turnover=float(args.max_turnover),
    )
    selected_global = ranked_global[0]
    selected_spec = OverlaySpec(**selected_global["spec"])
    global_test = [
        _run_fold(
            split,
            features_df,
            raw_returns,
            cfg,
            args.device,
            args.checkpoint_dir,
            [selected_spec],
            risk_indices,
            edge_indices,
            include_candidates=False,
        )
        for split in splits
    ]
    per_fold_val = _best_per_fold(
        val_results,
        alpha_floor=float(args.alpha_floor),
        dd_target=float(args.dd_target),
        max_turnover=float(args.max_turnover),
    )
    output_base = args.output or f"codex_outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_plan011_alpha_floor_f{args.folds.replace(',', '')}"
    payload = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "folds": selected if selected is not None else [int(s.fold_idx) for s in splits],
        "alpha_floor": float(args.alpha_floor),
        "dd_target": float(args.dd_target),
        "global_selected": selected_global,
        "global_counts": global_counts,
        "top_global": ranked_global[:20],
        "global_test": global_test,
        "per_fold_val": per_fold_val,
        "per_fold_counts": _per_fold_counts(
            val_results,
            alpha_floor=float(args.alpha_floor),
            dd_target=float(args.dd_target),
        ),
    }
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    Path(output_base + ".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload, output_base + ".md")
    print(f"[Plan011AlphaFloor] wrote {output_base}.json")
    print(f"[Plan011AlphaFloor] wrote {output_base}.md")


if __name__ == "__main__":
    main()
