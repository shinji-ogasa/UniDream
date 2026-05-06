from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import _backtest_positions, _state_features, _unit_cost
from unidream.cli.round1_meta_label_probe import (
    DEFAULT_CANDIDATES,
    _event_masks,
    _fmt,
    _json_sanitize,
    _make_label_bundle,
    _nanmax,
    _nanmean,
    _nanmin,
    _valid_eval_mask,
)
from unidream.cli.round1_period_selector_probe import _active_rate, _passes_val
from unidream.cli.route_separability_probe import _fit_binary_model, _safe_ap, _safe_auc, _score_binary, _select_threshold
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "round2_selector_audit_probe"
EPS_DD_PT = 1e-6

FIXED_SELECTORS: dict[str, tuple[str, str, str]] = {
    "fixed_recovery_state": ("vol_shock", "recovery", "state"),
    "fixed_recovery_raw": ("vol_shock", "recovery", "raw"),
    "fixed_triple_state": ("vol_shock", "triple_barrier", "state"),
    "fixed_triple_raw": ("vol_shock", "triple_barrier", "raw"),
}


def _date_prefix() -> str:
    return datetime.now().strftime("%Y%m%d")


def _nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


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


def _boundary_eval_mask(
    n: int,
    *,
    role: str,
    mode: str,
    horizon: int,
    purge_bars: int,
    embargo_bars: int,
    lookback_bars: int,
) -> np.ndarray:
    """Mask samples close to WFO split boundaries for leakage/overfit audits."""

    mask = np.ones(int(n), dtype=bool)
    if mode == "none":
        return mask
    if mode != "purged":
        raise ValueError(f"unknown boundary mode: {mode}")

    left = max(int(lookback_bars), 0)
    if role in {"val", "test"}:
        left = max(left, int(embargo_bars))
    right_drop = max(int(purge_bars), 0)
    if role in {"train", "val"}:
        right_drop += max(int(horizon), 0)

    if left > 0:
        mask[: min(left, len(mask))] = False
    if right_drop > 0:
        mask[max(0, len(mask) - right_drop) :] = False
    return mask


def _boundary_masks(
    *,
    train_len: int,
    val_len: int,
    test_len: int,
    mode: str,
    horizon: int,
    purge_bars: int,
    embargo_bars: int,
    lookback_bars: int,
) -> dict[str, np.ndarray]:
    return {
        "train": _boundary_eval_mask(
            train_len,
            role="train",
            mode=mode,
            horizon=horizon,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            lookback_bars=lookback_bars,
        ),
        "val": _boundary_eval_mask(
            val_len,
            role="val",
            mode=mode,
            horizon=horizon,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            lookback_bars=lookback_bars,
        ),
        "test": _boundary_eval_mask(
            test_len,
            role="test",
            mode=mode,
            horizon=horizon,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            lookback_bars=lookback_bars,
        ),
    }


def _positions_from_event_score(
    *,
    n: int,
    eval_mask: np.ndarray,
    event_score: np.ndarray,
    threshold: float,
    overlay_position: float,
    benchmark_position: float,
) -> np.ndarray:
    from unidream.cli.exploration_board_probe import _shift_for_execution

    selected = np.full(int(n), float(benchmark_position), dtype=np.float64)
    pred = np.asarray(event_score, dtype=np.float64) >= float(threshold)
    idx = np.flatnonzero(eval_mask)[pred]
    selected[idx] = float(overlay_position)
    return _shift_for_execution(selected, benchmark_position)


def _perturb_y(y_raw: np.ndarray, valid_mask: np.ndarray, *, mode: str, seed: int, shift_bars: int) -> np.ndarray:
    y = np.asarray(y_raw, dtype=np.int64).copy()
    idx = np.flatnonzero(valid_mask)
    if mode == "normal":
        return y
    if mode in {"shuffle", "shuffle_all"}:
        rng = np.random.default_rng(int(seed))
        y[idx] = rng.permutation(y[idx])
        return y
    if mode == "time_shift":
        shift = max(int(shift_bars), 1)
        src = idx + shift
        valid = src < len(y)
        y[idx[valid]] = y[src[valid]]
        if np.any(~valid):
            y[idx[~valid]] = y[idx[~valid][0]]
        return y
    raise ValueError(f"unknown audit mode: {mode}")


def _evaluate_combo_positions(
    *,
    combo: tuple[str, str, str],
    train_sets: dict[str, np.ndarray],
    val_sets: dict[str, np.ndarray],
    test_sets: dict[str, np.ndarray],
    events: dict[str, dict[str, Any]],
    labels: dict[str, dict[str, dict[str, np.ndarray | float]]],
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    max_train_samples: int,
    seed: int,
    false_active_cap: float,
    pred_rate_cap: float,
    audit_mode: str,
    shift_bars: int,
    boundary_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    event_name, label_name, feature_set = combo
    event = events[event_name]
    train_label = labels["train"][label_name]
    val_label = labels["val"][label_name]
    test_label = labels["test"][label_name]
    x_train = train_sets[feature_set]
    x_val = val_sets[feature_set]
    x_test = test_sets[feature_set]

    train_y_raw = np.asarray(train_label["y"], dtype=np.int64)
    val_y = np.asarray(val_label["y"], dtype=np.int64)
    test_y = np.asarray(test_label["y"], dtype=np.int64)
    train_utility = np.asarray(train_label["utility"], dtype=np.float64)
    val_utility = np.asarray(val_label["utility"], dtype=np.float64)
    test_utility = np.asarray(test_label["utility"], dtype=np.float64)
    train_mask = _valid_eval_mask(x_train, train_y_raw, train_utility, event["masks"]["train"]) & np.asarray(
        boundary_masks["train"][: len(x_train)], dtype=bool
    )
    val_mask = _valid_eval_mask(x_val, val_y, val_utility, event["masks"]["val"]) & np.asarray(
        boundary_masks["val"][: len(x_val)], dtype=bool
    )
    test_mask = _valid_eval_mask(x_test, test_y, test_utility, event["masks"]["test"]) & np.asarray(
        boundary_masks["test"][: len(x_test)], dtype=bool
    )
    train_y = _perturb_y(train_y_raw, train_mask, mode=audit_mode, seed=seed, shift_bars=shift_bars)
    if audit_mode in {"shuffle_all", "time_shift"}:
        val_y = _perturb_y(val_y, val_mask, mode=audit_mode, seed=seed + 991, shift_bars=shift_bars)

    out: dict[str, Any] = {
        "status": "ok",
        "event": event_name,
        "label": label_name,
        "feature_set": feature_set,
        "combo": f"{event_name}/{label_name}/{feature_set}",
        "audit_mode": audit_mode,
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "test_count": int(test_mask.sum()),
    }
    if int(train_mask.sum()) < 100 or int(val_mask.sum()) < 20 or int(test_mask.sum()) < 20:
        out["status"] = "insufficient_events"
        return out
    if len(np.unique(train_y[train_mask])) < 2:
        out["status"] = "one_class_train"
        return out

    model = _fit_binary_model(
        x_train[train_mask],
        train_y[train_mask],
        max_train_samples=max_train_samples,
        seed=seed,
    )
    if model is None:
        out["status"] = "no_model"
        return out
    val_score_event = _score_binary(model, x_val[val_mask])
    threshold, val_rates = _select_threshold(
        val_y[val_mask],
        val_score_event,
        false_active_cap=false_active_cap,
        pred_rate_cap=pred_rate_cap,
    )
    train_score_event = _score_binary(model, x_train[train_mask])
    test_score_event = _score_binary(model, x_test[test_mask])
    train_positions = _positions_from_event_score(
        n=len(x_train),
        eval_mask=train_mask,
        event_score=train_score_event,
        threshold=threshold,
        overlay_position=float(train_label["overlay_position"]),
        benchmark_position=benchmark_position,
    )
    val_positions = _positions_from_event_score(
        n=len(val_returns),
        eval_mask=val_mask,
        event_score=val_score_event,
        threshold=threshold,
        overlay_position=float(val_label["overlay_position"]),
        benchmark_position=benchmark_position,
    )
    test_positions = _positions_from_event_score(
        n=len(test_returns),
        eval_mask=test_mask,
        event_score=test_score_event,
        threshold=threshold,
        overlay_position=float(test_label["overlay_position"]),
        benchmark_position=benchmark_position,
    )
    val_bt, _ = _backtest_positions(val_returns, val_positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=benchmark_position)
    test_bt, _ = _backtest_positions(test_returns, test_positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=benchmark_position)
    out.update(
        {
            "threshold": float(threshold) if math.isfinite(float(threshold)) else str(threshold),
            "threshold_selected_on_val": val_rates,
            "val": val_bt,
            "test": test_bt,
            "val_auc": _safe_auc(val_y[val_mask], val_score_event),
            "val_ap": _safe_ap(val_y[val_mask], val_score_event),
            "test_auc": _safe_auc(test_y[test_mask], test_score_event),
            "test_ap": _safe_ap(test_y[test_mask], test_score_event),
            "_train_positions": train_positions,
            "_val_positions": val_positions,
            "_test_positions": test_positions,
        }
    )
    return out


def _priority_score(row: dict[str, Any], bonus: float) -> float:
    val = row["val"]
    return (
        float(bonus)
        + float(val.get("alpha_excess_pt", 0.0))
        + 0.10 * max(-float(val.get("maxdd_delta_pt", 0.0)), 0.0)
        - 0.02 * float(val.get("turnover", 0.0))
        - 2.0 * _active_rate(val)
    )


def _selector_score(metrics: dict[str, Any], *, maxdd_cap: float, turnover_cap: float, active_cap: float) -> float:
    alpha = float(metrics.get("alpha_excess_pt", 0.0))
    maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
    turnover = float(metrics.get("turnover", 999.0))
    active = _active_rate(metrics)
    if alpha <= 0.0:
        return -1_000_000.0 + alpha
    if maxdd > maxdd_cap or turnover > turnover_cap or active > active_cap:
        return -1_000_000.0 + alpha - 10.0 * max(maxdd - maxdd_cap, 0.0) - turnover
    return alpha + 0.10 * max(-maxdd, 0.0) - 0.03 * turnover - 2.0 * active


def _select_priority(
    rows: list[dict[str, Any]],
    *,
    use_veto: bool,
    use_triple: bool,
) -> tuple[dict[str, Any] | None, float]:
    by_combo = {str(r["combo"]): r for r in rows if r.get("status") == "ok"}
    if use_veto:
        veto_state = by_combo.get("vol_shock/veto/state")
        if veto_state is not None and _passes_val(veto_state, min_alpha=2.0, maxdd_cap=0.0, turnover_cap=3.5, active_cap=0.05):
            return veto_state, _priority_score(veto_state, 100.0)

    recovery_state = by_combo.get("vol_shock/recovery/state")
    if recovery_state is not None and _passes_val(recovery_state, min_alpha=-0.5, maxdd_cap=0.10, turnover_cap=5.0, active_cap=0.05):
        return recovery_state, _priority_score(recovery_state, 50.0)

    ordered = []
    if use_triple:
        ordered.append("vol_shock/triple_barrier/raw")
    ordered.extend(["vol_shock/recovery/raw"])
    if use_triple:
        ordered.append("vol_shock/triple_barrier/state")
    ordered.append("vol_shock/take/state")

    candidates: list[tuple[float, dict[str, Any]]] = []
    for combo in ordered:
        row = by_combo.get(combo)
        if row is None:
            continue
        if _passes_val(row, min_alpha=0.0, maxdd_cap=0.0, turnover_cap=5.0, active_cap=0.05):
            candidates.append((_priority_score(row, 0.0), row))
    if not candidates:
        return None, 0.0
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][0]


def _select_simple(rows: list[dict[str, Any]], *, volshock_only: bool, maxdd_cap: float, turnover_cap: float) -> tuple[dict[str, Any] | None, float]:
    candidates = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        if volshock_only and row.get("event") != "vol_shock":
            continue
        score = _selector_score(row["val"], maxdd_cap=maxdd_cap, turnover_cap=turnover_cap, active_cap=0.08)
        candidates.append((score, row))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates or candidates[0][0] <= 0.0:
        return None, 0.0
    return candidates[0][1], candidates[0][0]


def _benchmark_metrics() -> dict[str, Any]:
    return {
        "alpha_excess_pt": 0.0,
        "sharpe_delta": 0.0,
        "maxdd_delta_pt": 0.0,
        "period_win_rate": 0.0,
        "bar_win_rate": 0.0,
        "turnover": 0.0,
        "long_rate": 0.0,
        "short_rate": 0.0,
        "flat_rate": 1.0,
        "mean_position": 0.0,
        "n_trades": 0,
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
        metrics, _ = _backtest_positions(returns, positions, cfg=cfg, costs_cfg=stress_costs, benchmark_position=benchmark_position)
        out[name] = metrics
    return out


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _selected_row(
    *,
    fold: int,
    selector: str,
    row: dict[str, Any] | None,
    score: float,
    test_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict[str, Any],
    benchmark_position: float,
) -> dict[str, Any]:
    if row is None:
        return {
            "fold": int(fold),
            "selector": selector,
            "selected_combo": "benchmark",
            "val_score": 0.0,
            "status": "benchmark_fallback",
            "val": _benchmark_metrics(),
            "stress": {name: _benchmark_metrics() for name in _stress_grid()},
        }
    stress = _stress_metrics(
        returns=test_returns,
        positions=np.asarray(row["_test_positions"], dtype=np.float64),
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )
    return {
        "fold": int(fold),
        "selector": selector,
        "selected_combo": str(row["combo"]),
        "val_score": float(score),
        "status": "selected",
        "val": row["val"],
        "stress": stress,
        "val_auc": row.get("val_auc"),
        "val_ap": row.get("val_ap"),
        "test_auc": row.get("test_auc"),
        "test_ap": row.get("test_ap"),
    }


def _select_rows(
    *,
    rows: list[dict[str, Any]],
    fold: int,
    test_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict[str, Any],
    benchmark_position: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_combo = {(r.get("event"), r.get("label"), r.get("feature_set")): r for r in rows if r.get("status") == "ok"}
    for selector, combo in FIXED_SELECTORS.items():
        out.append(
            _selected_row(
                fold=fold,
                selector=selector,
                row=by_combo.get(combo),
                score=0.0,
                test_returns=test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )
        )
    for selector, kwargs in {
        "priority_recovery_veto": {"use_veto": True, "use_triple": True},
        "priority_no_veto": {"use_veto": False, "use_triple": True},
        "priority_no_triple": {"use_veto": True, "use_triple": False},
    }.items():
        row, score = _select_priority(rows, **kwargs)
        out.append(
            _selected_row(
                fold=fold,
                selector=selector,
                row=row,
                score=score,
                test_returns=test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )
        )
    for selector, kwargs in {
        "simple_best_t35": {"volshock_only": False, "maxdd_cap": 0.0, "turnover_cap": 3.5},
        "simple_volshock_t35": {"volshock_only": True, "maxdd_cap": 0.0, "turnover_cap": 3.5},
        "simple_volshock_t8": {"volshock_only": True, "maxdd_cap": 0.0, "turnover_cap": 8.0},
    }.items():
        row, score = _select_simple(rows, **kwargs)
        out.append(
            _selected_row(
                fold=fold,
                selector=selector,
                row=row,
                score=score,
                test_returns=test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )
        )
    return out


def _aggregate(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    modes = sorted({str(r["audit_mode"]) for r in selected_rows})
    for mode in modes:
        rows_mode = [r for r in selected_rows if str(r["audit_mode"]) == mode]
        out[mode] = {}
        selectors = sorted({str(r["selector"]) for r in rows_mode})
        for selector in selectors:
            rows_sel = [r for r in rows_mode if str(r["selector"]) == selector]
            out[mode][selector] = {}
            for stress in sorted(_stress_grid()):
                metrics = [r["stress"][stress] for r in rows_sel]
                alphas = [float(m["alpha_excess_pt"]) for m in metrics]
                dds = [float(m["maxdd_delta_pt"]) for m in metrics]
                turns = [float(m["turnover"]) for m in metrics]
                folds = [int(r["fold"]) for r in rows_sel]
                no_fold2 = [(a, d, t) for f, a, d, t in zip(folds, alphas, dds, turns) if f != 2]
                out[mode][selector][stress] = {
                    "folds": len(metrics),
                    "pass_alpha": int(sum(a > 0.0 for a in alphas)),
                    "pass_maxdd_strict": int(sum(d <= 0.0 for d in dds)),
                    "pass_both_strict": int(sum(a > 0.0 and d <= 0.0 for a, d in zip(alphas, dds))),
                    "pass_both_eps": int(sum(a > 0.0 and d <= EPS_DD_PT for a, d in zip(alphas, dds))),
                    "alpha_mean": _nanmean(alphas),
                    "alpha_median": _nanmedian(alphas),
                    "alpha_worst": _nanmin(alphas),
                    "alpha_median_no_fold2": _nanmedian([x[0] for x in no_fold2]),
                    "alpha_worst_no_fold2": _nanmin([x[0] for x in no_fold2]),
                    "maxdd_worst": _nanmax(dds),
                    "turnover_mean": _nanmean(turns),
                    "turnover_max": _nanmax(turns),
                }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "# Round2 Selector Audit Probe",
        "",
        f"Config: `{payload['config']}`",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Audit modes: `{', '.join(payload['audit_modes'])}`",
        f"Boundary mode: `{payload.get('boundary_mode', 'none')}`",
        f"Purge/embargo/lookback bars: `{payload.get('purge_bars', 0)}/{payload.get('embargo_bars', 0)}/{payload.get('lookback_bars', 0)}`",
        "",
        "## Aggregate: normal / cost_x1",
        "",
        "| selector | folds | Alpha pass | both strict | both eps | Alpha median | Alpha worst | Alpha median no fold2 | MaxDD worst | turnover mean | turnover max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    normal = aggregate.get("normal", {})
    for selector, stress_rows in normal.items():
        row = stress_rows["cost_x1"]
        lines.append(
            "| "
            + " | ".join(
                [
                    selector,
                    str(row["folds"]),
                    str(row["pass_alpha"]),
                    str(row["pass_both_strict"]),
                    str(row["pass_both_eps"]),
                    _fmt(row["alpha_median"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["alpha_median_no_fold2"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_mean"]),
                    _fmt(row["turnover_max"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Cost Stress: priority_recovery_veto", ""])
    lines.append("| stress | Alpha pass | both strict | both eps | Alpha median | Alpha worst | Alpha median no fold2 | MaxDD worst | turnover mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for stress, row in normal.get("priority_recovery_veto", {}).items():
        lines.append(
            "| "
            + " | ".join(
                [
                    stress,
                    str(row["pass_alpha"]),
                    str(row["pass_both_strict"]),
                    str(row["pass_both_eps"]),
                    _fmt(row["alpha_median"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["alpha_median_no_fold2"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Leakage/Null Audit", ""])
    lines.append("| mode | selector | stress | Alpha pass | both eps | Alpha median | Alpha worst | MaxDD worst |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for mode in [m for m in payload["audit_modes"] if m != "normal"]:
        for selector in ("priority_recovery_veto", "simple_volshock_t35"):
            row = aggregate.get(mode, {}).get(selector, {}).get("cost_x1")
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        mode,
                        selector,
                        "cost_x1",
                        str(row["pass_alpha"]),
                        str(row["pass_both_eps"]),
                        _fmt(row["alpha_median"]),
                        _fmt(row["alpha_worst"]),
                        _fmt(row["maxdd_worst"]),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Fold Detail: priority_recovery_veto / normal / cost_x1", ""])
    lines.append("| fold | selected combo | AlphaEx | MaxDDDelta | turnover | verdict eps |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for row in payload["selected"]:
        if row["audit_mode"] != "normal" or row["selector"] != "priority_recovery_veto":
            continue
        metrics = row["stress"]["cost_x1"]
        verdict = "pass" if float(metrics["alpha_excess_pt"]) > 0.0 and float(metrics["maxdd_delta_pt"]) <= EPS_DD_PT else "fail"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["selected_combo"],
                    _fmt(metrics["alpha_excess_pt"], 6),
                    _fmt(metrics["maxdd_delta_pt"], 12),
                    _fmt(metrics["turnover"]),
                    verdict,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Selector rules are frozen inside this probe; cost stress reuses the selected test positions and only changes execution costs.",
            "- `both eps` treats MaxDDDelta <= 1e-6 pt as numerical zero.",
            "- Shuffle and time_shift are null audits. If they remain strong, leakage or selector overfit is likely.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.round2_selector_audit_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,4,5,6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--ridge-event-rate", type=float, default=0.02)
    parser.add_argument("--dd-penalty", type=float, default=1.0)
    parser.add_argument("--vol-penalty", type=float, default=0.10)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--false-active-cap", type=float, default=0.03)
    parser.add_argument("--pred-rate-cap", type=float, default=0.05)
    parser.add_argument("--audit-modes", default="normal,shuffle,shuffle_all,time_shift")
    parser.add_argument("--shift-bars", type=int, default=128)
    parser.add_argument("--boundary-mode", choices=("none", "purged"), default="none")
    parser.add_argument("--purge-bars", type=int, default=0)
    parser.add_argument("--embargo-bars", type=int, default=0)
    parser.add_argument("--lookback-bars", type=int, default=0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    date = _date_prefix()
    if not args.output_json:
        args.output_json = os.path.join("docs_local", f"{date}_{EXPERIMENT_NAME}.json")
    if not args.output_md:
        args.output_md = os.path.join("docs_local", f"{date}_{EXPERIMENT_NAME}.md")

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
    audit_modes = [x.strip() for x in str(args.audit_modes).split(",") if x.strip()]
    feature_sets = ("raw", "state")

    selected_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for split in splits:
        print(f"[Round2SelectorAudit] fold={split.fold_idx} start")
        dataset = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
        x_train_state = _state_features(dataset.train_features, dataset.train_returns)
        x_val_state = _state_features(dataset.val_features, dataset.val_returns)
        x_test_state = _state_features(dataset.test_features, dataset.test_returns)
        train_sets = {"raw": np.asarray(dataset.train_features), "state": x_train_state}
        val_sets = {"raw": np.asarray(dataset.val_features), "state": x_val_state}
        test_sets = {"raw": np.asarray(dataset.test_features), "state": x_test_state}
        events = _event_masks(
            x_train_state=x_train_state,
            x_val_state=x_val_state,
            x_test_state=x_test_state,
            train_returns=dataset.train_returns,
            val_returns=dataset.val_returns,
            test_returns=dataset.test_returns,
            candidates=DEFAULT_CANDIDATES,
            horizon=args.horizon,
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            dd_penalty=args.dd_penalty,
            vol_penalty=args.vol_penalty,
            ridge_l2=args.ridge_l2,
            ridge_event_rate=args.ridge_event_rate,
        )
        labels = {
            "train": _make_label_bundle(
                dataset.train_returns,
                candidates=DEFAULT_CANDIDATES,
                horizon=args.horizon,
                benchmark_position=benchmark_position,
                unit_cost=unit_cost,
                dd_penalty=args.dd_penalty,
                vol_penalty=args.vol_penalty,
            ),
            "val": _make_label_bundle(
                dataset.val_returns,
                candidates=DEFAULT_CANDIDATES,
                horizon=args.horizon,
                benchmark_position=benchmark_position,
                unit_cost=unit_cost,
                dd_penalty=args.dd_penalty,
                vol_penalty=args.vol_penalty,
            ),
            "test": _make_label_bundle(
                dataset.test_returns,
                candidates=DEFAULT_CANDIDATES,
                horizon=args.horizon,
                benchmark_position=benchmark_position,
                unit_cost=unit_cost,
                dd_penalty=args.dd_penalty,
                vol_penalty=args.vol_penalty,
            ),
        }
        boundary_masks = _boundary_masks(
            train_len=len(dataset.train_returns),
            val_len=len(dataset.val_returns),
            test_len=len(dataset.test_returns),
            mode=args.boundary_mode,
            horizon=args.horizon,
            purge_bars=args.purge_bars,
            embargo_bars=args.embargo_bars,
            lookback_bars=args.lookback_bars,
        )
        combos = [
            (event_name, label_name, feature_set)
            for event_name in events.keys()
            for label_name in labels["train"].keys()
            for feature_set in feature_sets
        ]
        diagnostics[str(split.fold_idx)] = {
            "boundary_counts": {name: int(mask.sum()) for name, mask in boundary_masks.items()},
            "rows": {},
        }
        for audit_mode in audit_modes:
            print(f"[Round2SelectorAudit] fold={split.fold_idx} audit_mode={audit_mode}")
            fold_rows: list[dict[str, Any]] = []
            for combo in combos:
                row = _evaluate_combo_positions(
                    combo=combo,
                    train_sets=train_sets,
                    val_sets=val_sets,
                    test_sets=test_sets,
                    events=events,
                    labels=labels,
                    val_returns=dataset.val_returns,
                    test_returns=dataset.test_returns,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=benchmark_position,
                    max_train_samples=args.max_train_samples,
                    seed=args.seed + int(split.fold_idx) * 17 + len(audit_mode),
                    false_active_cap=args.false_active_cap,
                    pred_rate_cap=args.pred_rate_cap,
                    audit_mode=audit_mode,
                    shift_bars=args.shift_bars,
                    boundary_masks=boundary_masks,
                )
                row["fold"] = int(split.fold_idx)
                fold_rows.append(row)
            selected = _select_rows(
                rows=fold_rows,
                fold=int(split.fold_idx),
                test_returns=dataset.test_returns,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )
            for row in selected:
                row["audit_mode"] = audit_mode
            selected_rows.extend(selected)
            diagnostics[str(split.fold_idx)]["rows"][audit_mode] = [_public_row(r) for r in fold_rows]

    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": args.config,
        "start": args.start,
        "end": args.end,
        "folds": [int(split.fold_idx) for split in splits],
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "ridge_event_rate": float(args.ridge_event_rate),
        "false_active_cap": float(args.false_active_cap),
        "pred_rate_cap": float(args.pred_rate_cap),
        "audit_modes": audit_modes,
        "boundary_mode": str(args.boundary_mode),
        "purge_bars": int(args.purge_bars),
        "embargo_bars": int(args.embargo_bars),
        "lookback_bars": int(args.lookback_bars),
        "eps_dd_pt": EPS_DD_PT,
        "stress_grid": _stress_grid(),
        "selected": selected_rows,
        "aggregate": _aggregate(selected_rows),
        "diagnostics": diagnostics,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[Round2SelectorAudit] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
