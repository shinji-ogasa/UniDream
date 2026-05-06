from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import (
    SELECTOR_SPECS,
    _backtest_positions,
    _evaluate_selector,
    _state_features,
)
from unidream.cli.round1_guarded_recovery_probe import GUARD_SPECS, _evaluate_guarded
from unidream.cli.round1_meta_label_probe import _fmt, _nanmax, _nanmean, _nanmin
from unidream.cli.round1_meta_label_probe import DEFAULT_CANDIDATES, _event_masks, _make_label_bundle
from unidream.cli.round2_selector_audit_probe import _boundary_masks, _evaluate_combo_positions, _nanmedian
from unidream.cli.exploration_board_probe import _unit_cost
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "plan003_policy_blend_probe"
EPS_DD_PT = 1e-6


def _date_prefix() -> str:
    return datetime.now().strftime("%Y%m%d")


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _stress_costs(costs_cfg: dict[str, Any], *, cost_mult: float = 1.0, slippage_mult: float = 1.0) -> dict[str, Any]:
    out = dict(costs_cfg)
    for key in ("spread_bps", "fee_rate"):
        if key in out:
            out[key] = float(out[key]) * float(cost_mult)
    if "slippage_bps" in out:
        out["slippage_bps"] = float(out["slippage_bps"]) * float(cost_mult) * float(slippage_mult)
    return out


def _stress_grid() -> dict[str, dict[str, float]]:
    return {
        "cost_x1": {"cost_mult": 1.0, "slippage_mult": 1.0},
        "cost_x1_5": {"cost_mult": 1.5, "slippage_mult": 1.0},
        "cost_x2": {"cost_mult": 2.0, "slippage_mult": 1.0},
        "slippage_x2": {"cost_mult": 1.0, "slippage_mult": 2.0},
    }


def _stress_metrics(
    *,
    returns: np.ndarray,
    positions: np.ndarray,
    cfg: dict,
    costs_cfg: dict[str, Any],
    benchmark_position: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, spec in _stress_grid().items():
        stress_costs = _stress_costs(costs_cfg, cost_mult=spec["cost_mult"], slippage_mult=spec["slippage_mult"])
        metrics, _pnl = _backtest_positions(
            returns,
            positions,
            cfg=cfg,
            costs_cfg=stress_costs,
            benchmark_position=benchmark_position,
        )
        out[name] = metrics
    return out


def _is_active(positions: np.ndarray, benchmark_position: float) -> bool:
    pos = np.asarray(positions, dtype=np.float64)
    return bool(np.any(np.abs(pos - float(benchmark_position)) > 1e-12))


def _evaluate_recovery_rescue(
    *,
    ds: WFODataset,
    combo: tuple[str, str, str],
    x_train_state: np.ndarray,
    x_val_state: np.ndarray,
    x_test_state: np.ndarray,
    cfg: dict,
    costs_cfg: dict[str, Any],
    benchmark_position: float,
    unit_cost: float,
    ridge_l2: float,
    seed: int,
    max_train_samples: int,
    false_active_cap: float,
    pred_rate_cap: float,
) -> dict[str, Any]:
    train_sets = {"raw": np.asarray(ds.train_features), "state": x_train_state}
    val_sets = {"raw": np.asarray(ds.val_features), "state": x_val_state}
    test_sets = {"raw": np.asarray(ds.test_features), "state": x_test_state}
    events = _event_masks(
        x_train_state=x_train_state,
        x_val_state=x_val_state,
        x_test_state=x_test_state,
        train_returns=ds.train_returns,
        val_returns=ds.val_returns,
        test_returns=ds.test_returns,
        candidates=DEFAULT_CANDIDATES,
        horizon=32,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        dd_penalty=1.0,
        vol_penalty=0.10,
        ridge_l2=ridge_l2,
        ridge_event_rate=0.02,
    )
    labels = {
        "train": _make_label_bundle(
            ds.train_returns,
            candidates=DEFAULT_CANDIDATES,
            horizon=32,
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            dd_penalty=1.0,
            vol_penalty=0.10,
        ),
        "val": _make_label_bundle(
            ds.val_returns,
            candidates=DEFAULT_CANDIDATES,
            horizon=32,
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            dd_penalty=1.0,
            vol_penalty=0.10,
        ),
        "test": _make_label_bundle(
            ds.test_returns,
            candidates=DEFAULT_CANDIDATES,
            horizon=32,
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            dd_penalty=1.0,
            vol_penalty=0.10,
        ),
    }
    boundaries = _boundary_masks(
        train_len=len(ds.train_returns),
        val_len=len(ds.val_returns),
        test_len=len(ds.test_returns),
        mode="purged",
        horizon=32,
        purge_bars=32,
        embargo_bars=128,
        lookback_bars=256,
    )
    row = _evaluate_combo_positions(
        combo=combo,
        train_sets=train_sets,
        val_sets=val_sets,
        test_sets=test_sets,
        events=events,
        labels=labels,
        val_returns=ds.val_returns,
        test_returns=ds.test_returns,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
        max_train_samples=max_train_samples,
        seed=seed,
        false_active_cap=false_active_cap,
        pred_rate_cap=pred_rate_cap,
        audit_mode="normal",
        shift_bars=128,
        boundary_masks=boundaries,
    )
    return row


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in ("blend", "d_safe", "gr_baseline", "recovery_rescue", "micro_triple"):
        group = [r for r in rows if r["source_group"] == source]
        if not group:
            continue
        out[source] = {}
        for stress in _stress_grid():
            metrics = [r["stress"][stress] for r in group]
            alphas = [float(m["alpha_excess_pt"]) for m in metrics]
            dds = [float(m["maxdd_delta_pt"]) for m in metrics]
            turns = [float(m["turnover"]) for m in metrics]
            out[source][stress] = {
                "folds": len(group),
                "pass_alpha": int(sum(a > 0.0 for a in alphas)),
                "pass_both_eps": int(sum(a > 0.0 and d <= EPS_DD_PT for a, d in zip(alphas, dds))),
                "alpha_mean": _nanmean(alphas),
                "alpha_median": _nanmedian(alphas),
                "alpha_worst": _nanmin(alphas),
                "alpha_median_no_best": _nanmedian([a for a in alphas if a != max(alphas)]),
                "maxdd_worst": _nanmax(dds),
                "turnover_mean": _nanmean(turns),
                "turnover_max": _nanmax(turns),
            }
    return out


def _selected_row(
    *,
    fold: int,
    source_group: str,
    source: str,
    positions: np.ndarray,
    returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict[str, Any],
    benchmark_position: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "fold": int(fold),
        "source_group": source_group,
        "source": source,
        "active": _is_active(positions, benchmark_position),
        "stress": _stress_metrics(
            returns=returns,
            positions=positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        ),
        "meta": meta,
    }


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan003 Policy Blend Probe",
        "",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"D source: `{payload['d_selector']}`",
        f"GR source: `{payload['gr_spec']}`",
        "",
        "## Selection Rule",
        "",
        "Use D safe overlay by default. If D generates no test-side active positions and validation supports both D and GR, use GR baseline. If D safe violates the hard turnover guard, fall back to benchmark exposure.",
        "If D and GR are inactive/invalid, optionally use a purged fixed recovery-state rescue only when validation and runtime turnover guards pass.",
        "",
        "```text",
        f"D active == false",
        f"D val Alpha > {payload['d_val_alpha_min']}",
        f"GR val Alpha >= {payload['gr_val_alpha_min']}",
        f"GR val turnover <= {payload['gr_val_turnover_max']}",
        f"D test turnover <= {payload['d_test_turnover_max']} else benchmark safety fallback",
        f"Recovery rescue val Alpha >= {payload['recovery_rescue_val_alpha_min']}",
        f"Recovery rescue val turnover <= {payload['recovery_rescue_val_turnover_max']}",
        f"Recovery rescue test turnover <= {payload['recovery_rescue_test_turnover_max']}",
        f"Micro triple rescue val Alpha >= {payload['micro_triple_val_alpha_min']}",
        f"Micro triple rescue val AUC >= {payload['micro_triple_val_auc_min']}",
        f"Micro triple rescue val turnover <= {payload['micro_triple_val_turnover_max']}",
        f"Micro triple rescue test turnover <= {payload['micro_triple_test_turnover_max']}",
        "```",
        "",
        "## Aggregate",
        "",
        "| group | stress | folds | Alpha pass | both eps | Alpha mean | Alpha median | Alpha worst | Alpha median no best | MaxDD worst | turnover mean | turnover max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, stresses in payload["aggregate"].items():
        for stress, row in stresses.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        stress,
                        str(row["folds"]),
                        str(row["pass_alpha"]),
                        str(row["pass_both_eps"]),
                        _fmt(row["alpha_mean"]),
                        _fmt(row["alpha_median"]),
                        _fmt(row["alpha_worst"]),
                        _fmt(row["alpha_median_no_best"]),
                        _fmt(row["maxdd_worst"]),
                        _fmt(row["turnover_mean"]),
                        _fmt(row["turnover_max"]),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Fold Detail: blend / cost_x1", ""])
    lines.append("| fold | source | active | AlphaEx | MaxDDDelta | turnover | pass | D test turnover | D safety fail | D val alpha | GR val alpha | GR val turnover | Recovery val alpha | Micro val alpha | Micro AUC | Micro turnover |")
    lines.append("|---:|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["rows"]:
        if row["source_group"] != "blend":
            continue
        metrics = row["stress"]["cost_x1"]
        a = float(metrics["alpha_excess_pt"])
        d = float(metrics["maxdd_delta_pt"])
        verdict = "pass" if a > 0.0 and d <= EPS_DD_PT else "fail"
        meta = row.get("meta", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["source"],
                    str(row["active"]),
                    _fmt(a, 6),
                    _fmt(d, 12),
                    _fmt(metrics["turnover"]),
                    verdict,
                    _fmt(meta.get("d_test_turnover")),
                    str(meta.get("d_safety_fail")),
                    _fmt(meta.get("d_val_alpha")),
                    _fmt(meta.get("gr_val_alpha")),
                    _fmt(meta.get("gr_val_turnover")),
                    _fmt(meta.get("recovery_rescue_val_alpha")),
                    _fmt(meta.get("micro_triple_val_alpha")),
                    _fmt(meta.get("micro_triple_val_auc")),
                    _fmt(meta.get("micro_triple_test_turnover")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Judgment Rule",
            "",
            "- This is still experimental. It is a policy hierarchy, not a mainline change.",
            "- Adoption requires cost stress and null/alignment follow-up because GR uses the inflated all-bar pullback label.",
            "- If blend holds under cost_x1.5 and does not rely only on the best fold, it becomes a BC candidate policy source.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan003_policy_blend_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--d-selector", default="D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly")
    parser.add_argument("--gr-spec", default="GR_baseline")
    parser.add_argument("--d-val-alpha-min", type=float, default=0.0)
    parser.add_argument("--gr-val-alpha-min", type=float, default=0.30)
    parser.add_argument("--gr-val-turnover-max", type=float, default=3.0)
    parser.add_argument("--d-test-turnover-max", type=float, default=2.5)
    parser.add_argument("--recovery-rescue-val-alpha-min", type=float, default=0.10)
    parser.add_argument("--recovery-rescue-val-turnover-max", type=float, default=4.5)
    parser.add_argument("--recovery-rescue-test-turnover-max", type=float, default=3.5)
    parser.add_argument("--micro-triple-val-alpha-min", type=float, default=0.02)
    parser.add_argument("--micro-triple-val-auc-min", type=float, default=0.50)
    parser.add_argument("--micro-triple-val-turnover-max", type=float, default=0.5)
    parser.add_argument("--micro-triple-test-turnover-max", type=float, default=0.7)
    parser.add_argument("--disable-micro-triple", action="store_true")
    parser.add_argument("--disable-recovery-rescue", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    date = _date_prefix()
    if not args.output_json:
        args.output_json = os.path.join("codex_outputs", f"{date}_{EXPERIMENT_NAME}.json")
    if not args.output_md:
        args.output_md = os.path.join("codex_outputs", f"{date}_{EXPERIMENT_NAME}.md")

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
    d_spec = next((s for s in SELECTOR_SPECS if s.name == args.d_selector), None)
    if d_spec is None:
        raise ValueError(f"unknown --d-selector={args.d_selector}")
    gr_spec = next((s for s in GUARD_SPECS if s["name"] == args.gr_spec), None)
    if gr_spec is None:
        raise ValueError(f"unknown --gr-spec={args.gr_spec}")

    rows: list[dict[str, Any]] = []
    for split in splits:
        fid = int(split.fold_idx)
        print(f"[PolicyBlend] fold={fid} start")
        ds = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
        x_train = _state_features(ds.train_features, ds.train_returns)
        x_val = _state_features(ds.val_features, ds.val_returns)
        x_test = _state_features(ds.test_features, ds.test_returns)

        d_row = _evaluate_selector(
            spec=d_spec,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            train_returns=ds.train_returns,
            val_returns=ds.val_returns,
            test_returns=ds.test_returns,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
            l2=args.ridge_l2,
            seed=args.seed + fid * 100,
        )
        gr_row = _evaluate_guarded(
            spec=gr_spec,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            train_returns=ds.train_returns,
            val_returns=ds.val_returns,
            test_returns=ds.test_returns,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
            unit_cost=_unit_cost(costs_cfg),
            seed=args.seed + fid * 100,
            max_train_samples=args.max_train_samples,
        )
        if d_row.get("status") != "ok" or gr_row.get("status") != "ok":
            raise RuntimeError(f"fold {fid}: d_status={d_row.get('status')} gr_status={gr_row.get('status')}")
        recovery_rescue_row: dict[str, Any] = {"status": "disabled"}
        if not args.disable_recovery_rescue:
            recovery_rescue_row = _evaluate_recovery_rescue(
                ds=ds,
                combo=("vol_shock", "recovery", "state"),
                x_train_state=x_train,
                x_val_state=x_val,
                x_test_state=x_test,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                unit_cost=unit_cost,
                ridge_l2=args.ridge_l2,
                seed=args.seed + fid * 17 + len("normal"),
                max_train_samples=max(len(x_train), int(args.max_train_samples)),
                false_active_cap=0.03,
                pred_rate_cap=0.05,
            )
        micro_triple_row: dict[str, Any] = {"status": "disabled"}
        if not args.disable_micro_triple:
            micro_triple_row = _evaluate_recovery_rescue(
                ds=ds,
                combo=("vol_shock", "triple_barrier", "raw"),
                x_train_state=x_train,
                x_val_state=x_val,
                x_test_state=x_test,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                unit_cost=unit_cost,
                ridge_l2=args.ridge_l2,
                seed=args.seed + fid * 17 + len("normal"),
                max_train_samples=max(len(x_train), int(args.max_train_samples)),
                false_active_cap=0.005,
                pred_rate_cap=0.005,
            )
        d_positions = np.asarray(d_row["_test_positions"], dtype=np.float64)
        gr_positions = np.asarray(gr_row["_test_positions"], dtype=np.float64)
        benchmark_positions = np.full_like(d_positions, benchmark_position, dtype=np.float64)
        recovery_rescue_positions = (
            np.asarray(recovery_rescue_row["_test_positions"], dtype=np.float64)
            if recovery_rescue_row.get("status") == "ok"
            else benchmark_positions
        )
        micro_triple_positions = (
            np.asarray(micro_triple_row["_test_positions"], dtype=np.float64)
            if micro_triple_row.get("status") == "ok"
            else benchmark_positions
        )
        d_val = d_row["selection"]["val"]
        gr_val = gr_row["val_selection"]
        recovery_rescue_val = recovery_rescue_row.get("val", {})
        micro_triple_val = micro_triple_row.get("val", {})
        d_active = _is_active(d_positions, benchmark_position)
        recovery_rescue_active = _is_active(recovery_rescue_positions, benchmark_position)
        micro_triple_active = _is_active(micro_triple_positions, benchmark_position)
        d_cost_x1 = _stress_metrics(
            returns=ds.test_returns,
            positions=d_positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )["cost_x1"]
        recovery_rescue_cost_x1 = _stress_metrics(
            returns=ds.test_returns,
            positions=recovery_rescue_positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )["cost_x1"]
        micro_triple_cost_x1 = _stress_metrics(
            returns=ds.test_returns,
            positions=micro_triple_positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )["cost_x1"]
        d_safety_fail = float(d_cost_x1.get("turnover", 0.0)) > float(args.d_test_turnover_max)
        use_gr = (
            (not d_active)
            and (not d_safety_fail)
            and float(d_val.get("alpha_excess_pt", 0.0)) > float(args.d_val_alpha_min)
            and float(gr_val.get("val_alpha", 0.0)) >= float(args.gr_val_alpha_min)
            and float(gr_val.get("val_turnover", 999.0)) <= float(args.gr_val_turnover_max)
        )
        use_recovery_rescue = (
            (not d_active)
            and (not use_gr)
            and (not d_safety_fail)
            and recovery_rescue_row.get("status") == "ok"
            and recovery_rescue_active
            and float(recovery_rescue_val.get("alpha_excess_pt", 0.0)) >= float(args.recovery_rescue_val_alpha_min)
            and float(recovery_rescue_val.get("maxdd_delta_pt", 999.0)) <= EPS_DD_PT
            and float(recovery_rescue_val.get("turnover", 999.0)) <= float(args.recovery_rescue_val_turnover_max)
            and float(recovery_rescue_cost_x1.get("turnover", 999.0)) <= float(args.recovery_rescue_test_turnover_max)
        )
        use_micro_triple = (
            ((not d_active) or d_safety_fail)
            and (not use_gr)
            and (not use_recovery_rescue)
            and micro_triple_row.get("status") == "ok"
            and micro_triple_active
            and float(micro_triple_val.get("alpha_excess_pt", 0.0)) >= float(args.micro_triple_val_alpha_min)
            and float(micro_triple_row.get("val_auc") or 0.0) >= float(args.micro_triple_val_auc_min)
            and float(micro_triple_val.get("maxdd_delta_pt", 999.0)) <= EPS_DD_PT
            and float(micro_triple_val.get("turnover", 999.0)) <= float(args.micro_triple_val_turnover_max)
            and float(micro_triple_cost_x1.get("turnover", 999.0)) <= float(args.micro_triple_test_turnover_max)
        )
        if use_gr:
            blend_source = gr_spec["name"]
            blend_positions = gr_positions
        elif use_recovery_rescue:
            blend_source = "recovery_rescue_fixed_state"
            blend_positions = recovery_rescue_positions
        elif use_micro_triple:
            blend_source = "micro_triple_fixed_raw"
            blend_positions = micro_triple_positions
        elif d_safety_fail:
            blend_source = "benchmark_safety"
            blend_positions = benchmark_positions
        else:
            blend_source = d_spec.name
            blend_positions = d_positions
        meta = {
            "d_active": d_active,
            "d_test_alpha": float(d_cost_x1.get("alpha_excess_pt", 0.0)),
            "d_test_maxdd": float(d_cost_x1.get("maxdd_delta_pt", 0.0)),
            "d_test_turnover": float(d_cost_x1.get("turnover", 0.0)),
            "d_test_turnover_max": float(args.d_test_turnover_max),
            "d_safety_fail": bool(d_safety_fail),
            "d_val_alpha": float(d_val.get("alpha_excess_pt", 0.0)),
            "d_val_maxdd": float(d_val.get("maxdd_delta_pt", 0.0)),
            "d_val_turnover": float(d_val.get("turnover", 0.0)),
            "gr_val_alpha": float(gr_val.get("val_alpha", 0.0)),
            "gr_val_maxdd": float(gr_val.get("val_maxdd", 0.0)),
            "gr_val_turnover": float(gr_val.get("val_turnover", 0.0)),
            "use_gr": bool(use_gr),
            "recovery_rescue_status": recovery_rescue_row.get("status"),
            "recovery_rescue_active": bool(recovery_rescue_active),
            "recovery_rescue_val_alpha": float(recovery_rescue_val.get("alpha_excess_pt", 0.0)),
            "recovery_rescue_val_maxdd": float(recovery_rescue_val.get("maxdd_delta_pt", 0.0)),
            "recovery_rescue_val_turnover": float(recovery_rescue_val.get("turnover", 0.0)),
            "recovery_rescue_test_alpha": float(recovery_rescue_cost_x1.get("alpha_excess_pt", 0.0)),
            "recovery_rescue_test_maxdd": float(recovery_rescue_cost_x1.get("maxdd_delta_pt", 0.0)),
            "recovery_rescue_test_turnover": float(recovery_rescue_cost_x1.get("turnover", 0.0)),
            "use_recovery_rescue": bool(use_recovery_rescue),
            "micro_triple_status": micro_triple_row.get("status"),
            "micro_triple_active": bool(micro_triple_active),
            "micro_triple_val_alpha": float(micro_triple_val.get("alpha_excess_pt", 0.0)),
            "micro_triple_val_auc": float(micro_triple_row.get("val_auc") or 0.0),
            "micro_triple_val_maxdd": float(micro_triple_val.get("maxdd_delta_pt", 0.0)),
            "micro_triple_val_turnover": float(micro_triple_val.get("turnover", 0.0)),
            "micro_triple_test_alpha": float(micro_triple_cost_x1.get("alpha_excess_pt", 0.0)),
            "micro_triple_test_maxdd": float(micro_triple_cost_x1.get("maxdd_delta_pt", 0.0)),
            "micro_triple_test_turnover": float(micro_triple_cost_x1.get("turnover", 0.0)),
            "use_micro_triple": bool(use_micro_triple),
        }
        rows.append(
            _selected_row(
                fold=fid,
                source_group="d_safe",
                source=d_spec.name,
                positions=d_positions,
                returns=ds.test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                meta=meta,
            )
        )
        rows.append(
            _selected_row(
                fold=fid,
                source_group="gr_baseline",
                source=gr_spec["name"],
                positions=gr_positions,
                returns=ds.test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                meta=meta,
            )
        )
        rows.append(
            _selected_row(
                fold=fid,
                source_group="recovery_rescue",
                source="fixed_recovery_state",
                positions=recovery_rescue_positions,
                returns=ds.test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                meta=meta,
            )
        )
        rows.append(
            _selected_row(
                fold=fid,
                source_group="micro_triple",
                source="fixed_triple_raw_tight",
                positions=micro_triple_positions,
                returns=ds.test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                meta=meta,
            )
        )
        rows.append(
            _selected_row(
                fold=fid,
                source_group="blend",
                source=blend_source,
                positions=blend_positions,
                returns=ds.test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
                meta=meta,
            )
        )

    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": args.config,
        "folds": [int(s.fold_idx) for s in splits],
        "seed": int(args.seed),
        "d_selector": args.d_selector,
        "gr_spec": args.gr_spec,
        "d_val_alpha_min": float(args.d_val_alpha_min),
        "gr_val_alpha_min": float(args.gr_val_alpha_min),
        "gr_val_turnover_max": float(args.gr_val_turnover_max),
        "d_test_turnover_max": float(args.d_test_turnover_max),
        "recovery_rescue_val_alpha_min": float(args.recovery_rescue_val_alpha_min),
        "recovery_rescue_val_turnover_max": float(args.recovery_rescue_val_turnover_max),
        "recovery_rescue_test_turnover_max": float(args.recovery_rescue_test_turnover_max),
        "recovery_rescue_enabled": not bool(args.disable_recovery_rescue),
        "micro_triple_val_alpha_min": float(args.micro_triple_val_alpha_min),
        "micro_triple_val_auc_min": float(args.micro_triple_val_auc_min),
        "micro_triple_val_turnover_max": float(args.micro_triple_val_turnover_max),
        "micro_triple_test_turnover_max": float(args.micro_triple_test_turnover_max),
        "micro_triple_enabled": not bool(args.disable_micro_triple),
        "stress_grid": _stress_grid(),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[PolicyBlend] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
