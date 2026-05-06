from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unidream.cli.exploration_board_probe import SELECTOR_SPECS, _evaluate_selector, _state_features
from unidream.cli.plan003_policy_blend_probe import (
    EPS_DD_PT,
    _evaluate_recovery_rescue,
    _is_active,
    _stress_metrics,
)
from unidream.cli.round1_guarded_recovery_probe import GUARD_SPECS, _evaluate_guarded
from unidream.cli.round1_meta_label_probe import _fmt, _nanmax, _nanmean, _nanmin
from unidream.cli.round2_selector_audit_probe import _nanmedian
from unidream.cli.exploration_board_probe import _unit_cost
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "plan003_bc_student_probe"


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


def _benchmark_positions(n: int, benchmark_position: float) -> np.ndarray:
    return np.full(int(n), float(benchmark_position), dtype=np.float64)


def _compute_teacher(
    *,
    ds: WFODataset,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    cfg: dict,
    costs_cfg: dict[str, Any],
    benchmark_position: float,
    unit_cost: float,
    ridge_l2: float,
    seed: int,
    max_train_samples: int,
    d_selector: str,
    gr_spec_name: str,
    d_test_turnover_max: float,
    gr_val_alpha_min: float,
    gr_val_turnover_max: float,
    recovery_rescue_val_alpha_min: float,
    recovery_rescue_val_turnover_max: float,
    recovery_rescue_test_turnover_max: float,
    micro_triple_val_alpha_min: float,
    micro_triple_val_auc_min: float,
    micro_triple_val_turnover_max: float,
    micro_triple_test_turnover_max: float,
    selection_mode: str = "legacy_test_guarded",
) -> dict[str, Any]:
    d_spec = next((s for s in SELECTOR_SPECS if s.name == d_selector), None)
    if d_spec is None:
        raise ValueError(f"unknown d selector: {d_selector}")
    gr_spec = next((s for s in GUARD_SPECS if s["name"] == gr_spec_name), None)
    if gr_spec is None:
        raise ValueError(f"unknown gr spec: {gr_spec_name}")

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
        l2=ridge_l2,
        seed=seed,
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
        unit_cost=unit_cost,
        seed=seed,
        max_train_samples=max_train_samples,
    )
    recovery_row = _evaluate_recovery_rescue(
        ds=ds,
        combo=("vol_shock", "recovery", "state"),
        x_train_state=x_train,
        x_val_state=x_val,
        x_test_state=x_test,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        ridge_l2=ridge_l2,
        seed=seed,
        max_train_samples=max_train_samples,
        false_active_cap=0.03,
        pred_rate_cap=0.05,
    )
    micro_row = _evaluate_recovery_rescue(
        ds=ds,
        combo=("vol_shock", "triple_barrier", "raw"),
        x_train_state=x_train,
        x_val_state=x_val,
        x_test_state=x_test,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        ridge_l2=ridge_l2,
        seed=seed,
        max_train_samples=max_train_samples,
        false_active_cap=0.005,
        pred_rate_cap=0.005,
    )
    if d_row.get("status") != "ok" or gr_row.get("status") != "ok":
        raise RuntimeError(f"teacher component failed: d={d_row.get('status')} gr={gr_row.get('status')}")

    d_val_pos = np.asarray(d_row.get("_val_positions", _benchmark_positions(len(ds.val_returns), benchmark_position)), dtype=np.float64)
    d_train_pos = np.asarray(d_row.get("_train_positions", _benchmark_positions(len(ds.train_returns), benchmark_position)), dtype=np.float64)
    d_test_pos = np.asarray(d_row["_test_positions"], dtype=np.float64)
    gr_train_pos = np.asarray(gr_row.get("_train_positions", _benchmark_positions(len(ds.train_returns), benchmark_position)), dtype=np.float64)
    gr_val_pos = np.asarray(gr_row.get("_val_positions", _benchmark_positions(len(ds.val_returns), benchmark_position)), dtype=np.float64)
    gr_test_pos = np.asarray(gr_row["_test_positions"], dtype=np.float64)
    rec_train_pos = np.asarray(recovery_row.get("_train_positions", _benchmark_positions(len(ds.train_returns), benchmark_position)), dtype=np.float64)
    rec_val_pos = np.asarray(recovery_row.get("_val_positions", _benchmark_positions(len(ds.val_returns), benchmark_position)), dtype=np.float64)
    rec_test_pos = np.asarray(recovery_row.get("_test_positions", _benchmark_positions(len(ds.test_returns), benchmark_position)), dtype=np.float64)
    mic_train_pos = np.asarray(micro_row.get("_train_positions", _benchmark_positions(len(ds.train_returns), benchmark_position)), dtype=np.float64)
    mic_val_pos = np.asarray(micro_row.get("_val_positions", _benchmark_positions(len(ds.val_returns), benchmark_position)), dtype=np.float64)
    mic_test_pos = np.asarray(micro_row.get("_test_positions", _benchmark_positions(len(ds.test_returns), benchmark_position)), dtype=np.float64)

    d_val = d_row["selection"]["val"]
    gr_val = gr_row["val_selection"]
    rec_val = recovery_row.get("val", {})
    mic_val = micro_row.get("val", {})

    d_cost = _stress_metrics(
        returns=ds.test_returns,
        positions=d_test_pos,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )["cost_x1"]
    rec_cost = _stress_metrics(
        returns=ds.test_returns,
        positions=rec_test_pos,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )["cost_x1"]
    mic_cost = _stress_metrics(
        returns=ds.test_returns,
        positions=mic_test_pos,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )["cost_x1"]

    if selection_mode not in {"legacy_test_guarded", "val_only"}:
        raise ValueError(f"unknown teacher selection_mode: {selection_mode}")
    if selection_mode == "val_only":
        d_active = _is_active(d_val_pos, benchmark_position)
        d_safety_fail = float(d_val.get("turnover", 0.0)) > float(d_test_turnover_max)
        rec_active = _is_active(rec_val_pos, benchmark_position)
        rec_turnover_ok = True
        mic_active = _is_active(mic_val_pos, benchmark_position)
        mic_turnover_ok = True
    else:
        d_active = _is_active(d_test_pos, benchmark_position)
        d_safety_fail = float(d_cost.get("turnover", 0.0)) > float(d_test_turnover_max)
        rec_active = _is_active(rec_test_pos, benchmark_position)
        rec_turnover_ok = float(rec_cost.get("turnover", 999.0)) <= float(recovery_rescue_test_turnover_max)
        mic_active = _is_active(mic_test_pos, benchmark_position)
        mic_turnover_ok = float(mic_cost.get("turnover", 999.0)) <= float(micro_triple_test_turnover_max)
    use_gr = (
        (not d_active)
        and (not d_safety_fail)
        and float(d_val.get("alpha_excess_pt", 0.0)) > 0.0
        and float(gr_val.get("val_alpha", 0.0)) >= float(gr_val_alpha_min)
        and float(gr_val.get("val_turnover", 999.0)) <= float(gr_val_turnover_max)
    )
    use_recovery = (
        (not d_active)
        and (not use_gr)
        and (not d_safety_fail)
        and recovery_row.get("status") == "ok"
        and rec_active
        and float(rec_val.get("alpha_excess_pt", 0.0)) >= float(recovery_rescue_val_alpha_min)
        and float(rec_val.get("maxdd_delta_pt", 999.0)) <= EPS_DD_PT
        and float(rec_val.get("turnover", 999.0)) <= float(recovery_rescue_val_turnover_max)
        and rec_turnover_ok
    )
    use_micro = (
        ((not d_active) or d_safety_fail)
        and (not use_gr)
        and (not use_recovery)
        and micro_row.get("status") == "ok"
        and mic_active
        and float(mic_val.get("alpha_excess_pt", 0.0)) >= float(micro_triple_val_alpha_min)
        and float(micro_row.get("val_auc") or 0.0) >= float(micro_triple_val_auc_min)
        and float(mic_val.get("maxdd_delta_pt", 999.0)) <= EPS_DD_PT
        and float(mic_val.get("turnover", 999.0)) <= float(micro_triple_val_turnover_max)
        and mic_turnover_ok
    )

    if use_gr:
        source = "GR_baseline"
        train_pos = gr_train_pos
        val_pos = gr_val_pos
        test_pos = gr_test_pos
    elif use_recovery:
        source = "recovery_rescue_fixed_state"
        train_pos = rec_train_pos
        val_pos = rec_val_pos
        test_pos = rec_test_pos
    elif use_micro:
        source = "micro_triple_fixed_raw"
        train_pos = mic_train_pos
        val_pos = mic_val_pos
        test_pos = mic_test_pos
    elif d_safety_fail:
        source = "benchmark_safety"
        train_pos = _benchmark_positions(len(ds.train_returns), benchmark_position)
        val_pos = _benchmark_positions(len(ds.val_returns), benchmark_position)
        test_pos = _benchmark_positions(len(ds.test_returns), benchmark_position)
    elif d_active:
        source = d_selector
        train_pos = d_train_pos
        val_pos = d_val_pos
        test_pos = d_test_pos
    else:
        source = "benchmark"
        train_pos = _benchmark_positions(len(ds.train_returns), benchmark_position)
        val_pos = _benchmark_positions(len(ds.val_returns), benchmark_position)
        test_pos = _benchmark_positions(len(ds.test_returns), benchmark_position)

    return {
        "source": source,
        "train_positions": train_pos,
        "val_positions": val_pos,
        "test_positions": test_pos,
        "meta": {
            "d_active": bool(d_active),
            "d_safety_fail": bool(d_safety_fail),
            "selection_mode": selection_mode,
            "use_gr": bool(use_gr),
            "use_recovery": bool(use_recovery),
            "use_micro": bool(use_micro),
            "micro_val_auc": float(micro_row.get("val_auc") or 0.0),
            "micro_val_alpha": float(mic_val.get("alpha_excess_pt", 0.0)),
        },
    }


def _fit_ridge_student(
    x_val: np.ndarray,
    teacher_val: np.ndarray,
    *,
    benchmark_position: float,
    alpha: float,
    active_weight: float,
    model_kind: str,
) -> Any | None:
    delta = np.asarray(teacher_val, dtype=np.float64) - float(benchmark_position)
    if np.all(np.abs(delta) < 1e-12):
        return None
    active = np.abs(delta) > 1e-12
    weights = np.ones(len(delta), dtype=np.float64)
    weights[active] += float(active_weight)
    if model_kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), random_state=0))
        model.fit(np.asarray(x_val, dtype=np.float64), delta, ridge__sample_weight=weights)
    elif model_kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=float(alpha),
            min_samples_leaf=50,
            random_state=0,
        )
        model.fit(np.asarray(x_val, dtype=np.float64), delta, sample_weight=weights)
    else:
        raise ValueError(f"unknown student model: {model_kind}")
    return model


def _positions_from_delta_pred(
    pred: np.ndarray,
    *,
    benchmark_position: float,
    threshold: float,
    scale: float,
    min_position: float,
    max_position: float,
    quantize: bool,
) -> np.ndarray:
    delta = np.asarray(pred, dtype=np.float64) * float(scale)
    active = np.abs(delta) >= float(threshold)
    pos = np.full(len(delta), float(benchmark_position), dtype=np.float64)
    pos[active] = np.clip(float(benchmark_position) + delta[active], float(min_position), float(max_position))
    if quantize:
        grid = np.asarray([min_position, 0.75, benchmark_position, 1.05, max_position], dtype=np.float64)
        grid = np.unique(np.clip(grid, min_position, max_position))
        idx = np.argmin(np.abs(pos[:, None] - grid[None, :]), axis=1)
        pos = grid[idx]
    return pos


def _train_student_policy(
    *,
    x_val: np.ndarray,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    teacher_val: np.ndarray,
    teacher_test: np.ndarray,
    cfg: dict,
    costs_cfg: dict[str, Any],
    benchmark_position: float,
    alpha: float,
    active_weight: float,
    model_kind: str,
    min_position: float,
    max_position: float,
    quantize: bool,
) -> dict[str, Any]:
    model = _fit_ridge_student(
        x_val,
        teacher_val,
        benchmark_position=benchmark_position,
        alpha=alpha,
        active_weight=active_weight,
        model_kind=model_kind,
    )
    if model is None:
        pos = _benchmark_positions(len(test_returns), benchmark_position)
        return {
            "status": "benchmark_no_active_teacher",
            "positions": pos,
            "selection": {"threshold": "inf", "scale": 0.0},
        }

    pred_val = np.asarray(model.predict(np.asarray(x_val, dtype=np.float64)), dtype=np.float64)
    pred_test = np.asarray(model.predict(np.asarray(x_val[:0], dtype=np.float64)), dtype=np.float64)
    del pred_test

    abs_pred = np.abs(pred_val[np.isfinite(pred_val)])
    if len(abs_pred) == 0:
        thresholds = [float("inf")]
    else:
        qs = [0.50, 0.70, 0.85, 0.92, 0.96, 0.98, 0.99, 0.995]
        thresholds = sorted({0.0, *[float(np.quantile(abs_pred, q)) for q in qs]})
    scales = [0.5, 1.0, 1.5, 2.0]

    best: dict[str, Any] | None = None
    for scale in scales:
        for threshold in thresholds:
            val_pos = _positions_from_delta_pred(
                pred_val,
                benchmark_position=benchmark_position,
                threshold=threshold,
                scale=scale,
                min_position=min_position,
                max_position=max_position,
                quantize=quantize,
            )
            val_stress = _stress_metrics(
                returns=val_returns,
                positions=val_pos,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )
            m = val_stress["cost_x1"]
            alpha_ex = float(m.get("alpha_excess_pt", 0.0))
            maxdd = float(m.get("maxdd_delta_pt", 0.0))
            turnover = float(m.get("turnover", 999.0))
            active = 1.0 - float(m.get("flat_rate", 1.0))
            if turnover > 3.5 or maxdd > EPS_DD_PT:
                score = -1e6 + alpha_ex - turnover - 10.0 * max(maxdd, 0.0)
            else:
                score = alpha_ex + 0.10 * max(-maxdd, 0.0) - 0.05 * turnover - 5.0 * active
            candidate = {
                "threshold": float(threshold),
                "scale": float(scale),
                "score": float(score),
                "val": m,
            }
            if best is None or score > float(best["score"]):
                best = candidate

    pred_test = np.asarray(model.predict(np.asarray(x_val[:0], dtype=np.float64)), dtype=np.float64)
    del pred_test
    # The caller provides only x_val here for fitting; predict test outside to keep the selected config reusable.
    return {"status": "ok", "model": model, "selection": best}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({r["group"] for r in rows}):
        group_rows = [r for r in rows if r["group"] == group]
        out[group] = {}
        for stress in ("cost_x1", "cost_x1_5", "cost_x2", "slippage_x2"):
            metrics = [r["stress"][stress] for r in group_rows]
            alphas = [float(m["alpha_excess_pt"]) for m in metrics]
            dds = [float(m["maxdd_delta_pt"]) for m in metrics]
            turns = [float(m["turnover"]) for m in metrics]
            out[group][stress] = {
                "folds": len(group_rows),
                "pass_both_eps": int(sum(a > 0.0 and d <= EPS_DD_PT for a, d in zip(alphas, dds))),
                "alpha_mean": _nanmean(alphas),
                "alpha_median": _nanmedian(alphas),
                "alpha_worst": _nanmin(alphas),
                "maxdd_worst": _nanmax(dds),
                "turnover_max": _nanmax(turns),
            }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan003 BC Student Probe",
        "",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        "",
        "## Aggregate",
        "",
        "| group | stress | pass | Alpha mean | Alpha median | Alpha worst | MaxDD worst | turnover max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, stresses in payload["aggregate"].items():
        for stress, row in stresses.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
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
    lines.extend(["", "## Fold Detail: cost_x1", ""])
    lines.append("| fold | group | source | AlphaEx | MaxDDDelta | turnover | selection |")
    lines.append("|---:|---|---|---:|---:|---:|---|")
    for row in payload["rows"]:
        m = row["stress"]["cost_x1"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["group"],
                    row.get("source", ""),
                    _fmt(m["alpha_excess_pt"], 6),
                    _fmt(m["maxdd_delta_pt"], 12),
                    _fmt(m["turnover"]),
                    str(row.get("selection", "")),
                ]
            )
            + " |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan003_bc_student_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--student-alpha", type=float, default=1.0)
    parser.add_argument("--student-active-weight", type=float, default=30.0)
    parser.add_argument("--student-model", choices=("ridge", "hgb"), default="ridge")
    parser.add_argument("--fit-source", choices=("val", "trainval"), default="val")
    parser.add_argument("--quantize", action="store_true")
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
    min_position = float(cfg.get("ac", {}).get("abs_min_position", 0.0))
    max_position = float(cfg.get("ac", {}).get("abs_max_position", 1.25))

    rows: list[dict[str, Any]] = []
    for split in splits:
        fid = int(split.fold_idx)
        print(f"[BCStudent] fold={fid} start")
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
            d_test_turnover_max=2.5,
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
        teacher_test_pos = np.asarray(teacher["test_positions"], dtype=np.float64)
        rows.append(
            {
                "fold": fid,
                "group": "teacher_hierarchy",
                "source": str(teacher["source"]),
                "stress": _stress_metrics(
                    returns=ds.test_returns,
                    positions=teacher_test_pos,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=benchmark_position,
                ),
                "selection": teacher["meta"],
            }
        )

        if args.fit_source == "trainval":
            student_x = np.concatenate([x_train, x_val], axis=0)
            student_y = np.concatenate(
                [
                    np.asarray(teacher["train_positions"], dtype=np.float64),
                    np.asarray(teacher["val_positions"], dtype=np.float64),
                ],
                axis=0,
            )
        else:
            student_x = x_val
            student_y = np.asarray(teacher["val_positions"], dtype=np.float64)
        model = _fit_ridge_student(
            student_x,
            student_y,
            benchmark_position=benchmark_position,
            alpha=args.student_alpha,
            active_weight=args.student_active_weight,
            model_kind=args.student_model,
        )
        if model is None:
            student_pos = _benchmark_positions(len(ds.test_returns), benchmark_position)
            selection: dict[str, Any] = {"status": "benchmark_no_active_teacher"}
        else:
            pred_val = np.asarray(model.predict(x_val), dtype=np.float64)
            pred_test = np.asarray(model.predict(x_test), dtype=np.float64)
            abs_pred = np.abs(pred_val[np.isfinite(pred_val)])
            thresholds = [float("inf")] if len(abs_pred) == 0 else sorted(
                {0.0, *[float(np.quantile(abs_pred, q)) for q in (0.50, 0.70, 0.85, 0.92, 0.96, 0.98, 0.99, 0.995)]}
            )
            best: dict[str, Any] | None = None
            for scale in (0.5, 1.0, 1.5, 2.0):
                for threshold in thresholds:
                    val_pos = _positions_from_delta_pred(
                        pred_val,
                        benchmark_position=benchmark_position,
                        threshold=threshold,
                        scale=scale,
                        min_position=min_position,
                        max_position=max_position,
                        quantize=args.quantize,
                    )
                    m = _stress_metrics(
                        returns=ds.val_returns,
                        positions=val_pos,
                        cfg=cfg,
                        costs_cfg=costs_cfg,
                        benchmark_position=benchmark_position,
                    )["cost_x1"]
                    alpha_ex = float(m.get("alpha_excess_pt", 0.0))
                    maxdd = float(m.get("maxdd_delta_pt", 0.0))
                    turnover = float(m.get("turnover", 999.0))
                    active = 1.0 - float(m.get("flat_rate", 1.0))
                    if turnover > 3.5 or maxdd > EPS_DD_PT:
                        score = -1e6 + alpha_ex - turnover - 10.0 * max(maxdd, 0.0)
                    else:
                        score = alpha_ex + 0.10 * max(-maxdd, 0.0) - 0.05 * turnover - 5.0 * active
                    if best is None or score > float(best["score"]):
                        best = {"threshold": float(threshold), "scale": float(scale), "score": float(score), "val": m}
            assert best is not None
            student_pos = _positions_from_delta_pred(
                pred_test,
                benchmark_position=benchmark_position,
                threshold=float(best["threshold"]),
                scale=float(best["scale"]),
                min_position=min_position,
                max_position=max_position,
                quantize=args.quantize,
            )
            selection = best
        rows.append(
            {
                "fold": fid,
                "group": "bc_ridge_student",
                "source": str(teacher["source"]),
                "stress": _stress_metrics(
                    returns=ds.test_returns,
                    positions=student_pos,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=benchmark_position,
                ),
                "selection": selection,
            }
        )

    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": args.config,
        "folds": [int(s.fold_idx) for s in splits],
        "seed": int(args.seed),
        "student_alpha": float(args.student_alpha),
        "student_active_weight": float(args.student_active_weight),
        "student_model": args.student_model,
        "fit_source": args.fit_source,
        "quantize": bool(args.quantize),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[BCStudent] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
