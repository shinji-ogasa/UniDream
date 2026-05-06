from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import _backtest_positions, _shift_for_execution, _state_features
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
from unidream.cli.route_separability_probe import _fit_binary_model, _safe_ap, _safe_auc, _score_binary, _select_threshold
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "round1_period_selector_probe"


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    maxdd_cap_pt: float
    turnover_cap: float
    min_alpha_pt: float
    max_active_rate: float
    dd_reward: float
    turnover_penalty: float
    allow_zero_alpha: bool = False


SPECS: tuple[SelectorSpec, ...] = (
    SelectorSpec("strict_t35", maxdd_cap_pt=0.0, turnover_cap=3.5, min_alpha_pt=0.0, max_active_rate=0.05, dd_reward=0.20, turnover_penalty=0.04),
    SelectorSpec("strict_t8", maxdd_cap_pt=0.0, turnover_cap=8.0, min_alpha_pt=0.0, max_active_rate=0.08, dd_reward=0.15, turnover_penalty=0.03),
    SelectorSpec("alpha_safe_t35", maxdd_cap_pt=0.10, turnover_cap=3.5, min_alpha_pt=0.0, max_active_rate=0.05, dd_reward=0.05, turnover_penalty=0.04),
    SelectorSpec("dd_first_t35", maxdd_cap_pt=0.0, turnover_cap=3.5, min_alpha_pt=-0.10, max_active_rate=0.05, dd_reward=0.50, turnover_penalty=0.04),
)


FIXED_COMBOS: tuple[tuple[str, str, str], ...] = (
    ("vol_shock", "recovery", "state"),
    ("vol_shock", "recovery", "raw"),
    ("vol_shock", "triple_barrier", "raw"),
    ("vol_shock", "veto", "state"),
)


def _date_prefix() -> str:
    return datetime.now().strftime("%Y%m%d")


def _active_rate(metrics: dict[str, Any]) -> float:
    return max(0.0, 1.0 - float(metrics.get("flat_rate", 1.0)))


def _selector_score(metrics: dict[str, Any], spec: SelectorSpec) -> float:
    alpha = float(metrics.get("alpha_excess_pt", 0.0))
    maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
    turnover = float(metrics.get("turnover", 999.0))
    active = _active_rate(metrics)
    if not spec.allow_zero_alpha and alpha <= spec.min_alpha_pt:
        return -1_000_000.0 + alpha
    if spec.allow_zero_alpha and alpha < spec.min_alpha_pt:
        return -1_000_000.0 + alpha
    if maxdd > spec.maxdd_cap_pt or turnover > spec.turnover_cap or active > spec.max_active_rate:
        return -1_000_000.0 + alpha - 10.0 * max(maxdd - spec.maxdd_cap_pt, 0.0) - turnover
    return alpha + spec.dd_reward * max(-maxdd, 0.0) - spec.turnover_penalty * turnover - 2.0 * active


def _positions_from_score(
    *,
    n: int,
    eval_mask: np.ndarray,
    score: np.ndarray,
    threshold: float,
    overlay_position: float,
    benchmark_position: float,
) -> np.ndarray:
    selected = np.full(int(n), float(benchmark_position), dtype=np.float64)
    pred = np.asarray(score, dtype=np.float64) >= float(threshold)
    idx = np.flatnonzero(eval_mask)[pred]
    selected[idx] = float(overlay_position)
    return _shift_for_execution(selected, benchmark_position)


def _combo_allowed(combo: tuple[str, str, str], scope: str) -> bool:
    event, label, features = combo
    if scope == "all":
        return True
    if scope == "vol_shock":
        return event == "vol_shock"
    if scope == "vol_shock_state":
        return event == "vol_shock" and features == "state"
    if scope == "vol_shock_recovery":
        return event == "vol_shock" and label == "recovery"
    raise ValueError(f"unknown combo scope: {scope}")


def _evaluate_combo_with_val(
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
) -> dict[str, Any]:
    event_name, label_name, feature_set = combo
    event = events[event_name]
    train_label = labels["train"][label_name]
    val_label = labels["val"][label_name]
    test_label = labels["test"][label_name]

    train_y = np.asarray(train_label["y"], dtype=np.int64)
    val_y = np.asarray(val_label["y"], dtype=np.int64)
    test_y = np.asarray(test_label["y"], dtype=np.int64)
    train_utility = np.asarray(train_label["utility"], dtype=np.float64)
    val_utility = np.asarray(val_label["utility"], dtype=np.float64)
    test_utility = np.asarray(test_label["utility"], dtype=np.float64)
    x_train = train_sets[feature_set]
    x_val = val_sets[feature_set]
    x_test = test_sets[feature_set]
    train_mask = _valid_eval_mask(x_train, train_y, train_utility, event["masks"]["train"])
    val_mask = _valid_eval_mask(x_val, val_y, val_utility, event["masks"]["val"])
    test_mask = _valid_eval_mask(x_test, test_y, test_utility, event["masks"]["test"])

    out: dict[str, Any] = {
        "status": "ok",
        "event": event_name,
        "label": label_name,
        "feature_set": feature_set,
        "combo": f"{event_name}/{label_name}/{feature_set}",
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
    val_pos = _positions_from_score(
        n=len(val_returns),
        eval_mask=val_mask,
        score=val_score_event,
        threshold=threshold,
        overlay_position=float(val_label["overlay_position"]),
        benchmark_position=benchmark_position,
    )
    test_score_event = _score_binary(model, x_test[test_mask])
    test_pos = _positions_from_score(
        n=len(test_returns),
        eval_mask=test_mask,
        score=test_score_event,
        threshold=threshold,
        overlay_position=float(test_label["overlay_position"]),
        benchmark_position=benchmark_position,
    )
    val_bt, _ = _backtest_positions(val_returns, val_pos, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=benchmark_position)
    test_bt, _ = _backtest_positions(test_returns, test_pos, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=benchmark_position)
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
        }
    )
    return out


def _benchmark_row(split_idx: int, spec_name: str) -> dict[str, Any]:
    metrics = {
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
    return {
        "fold": int(split_idx),
        "selector": spec_name,
        "selected_combo": "benchmark",
        "val_score": 0.0,
        "test": metrics,
        "val": metrics,
        "status": "benchmark_fallback",
    }


def _passes_val(row: dict[str, Any], *, min_alpha: float, maxdd_cap: float, turnover_cap: float, active_cap: float) -> bool:
    val = row["val"]
    return (
        float(val.get("alpha_excess_pt", 0.0)) >= min_alpha
        and float(val.get("maxdd_delta_pt", 0.0)) <= maxdd_cap
        and float(val.get("turnover", 999.0)) <= turnover_cap
        and _active_rate(val) <= active_cap
    )


def _priority_recovery_veto_select(ok_rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    """Validation-only heuristic: recovery is the default edge, veto is a guarded fold-level rescue."""
    by_combo = {str(r["combo"]): r for r in ok_rows}

    def priority_score(row: dict[str, Any], bonus: float) -> float:
        val = row["val"]
        return (
            bonus
            + float(val.get("alpha_excess_pt", 0.0))
            + 0.10 * max(-float(val.get("maxdd_delta_pt", 0.0)), 0.0)
            - 0.02 * float(val.get("turnover", 0.0))
            - 2.0 * _active_rate(val)
        )

    # Veto is allowed only when validation strongly says it is safe and not churn-heavy.
    veto_state = by_combo.get("vol_shock/veto/state")
    if veto_state is not None and _passes_val(veto_state, min_alpha=2.0, maxdd_cap=0.0, turnover_cap=3.5, active_cap=0.05):
        return veto_state, priority_score(veto_state, 100.0)

    # Recovery/state is the preferred edge. Permit mild negative validation alpha because it was
    # designed as a sparse recovery classifier and validation windows can miss sparse test events.
    recovery_state = by_combo.get("vol_shock/recovery/state")
    if recovery_state is not None and _passes_val(recovery_state, min_alpha=-0.5, maxdd_cap=0.10, turnover_cap=5.0, active_cap=0.05):
        return recovery_state, priority_score(recovery_state, 50.0)

    ordered_fallbacks = [
        "vol_shock/triple_barrier/raw",
        "vol_shock/recovery/raw",
        "vol_shock/triple_barrier/state",
        "vol_shock/take/state",
    ]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for combo in ordered_fallbacks:
        row = by_combo.get(combo)
        if row is None:
            continue
        if _passes_val(row, min_alpha=0.0, maxdd_cap=0.0, turnover_cap=5.0, active_cap=0.05):
            val = row["val"]
            score = float(val["alpha_excess_pt"]) + 0.10 * max(-float(val["maxdd_delta_pt"]), 0.0) - 0.03 * float(val["turnover"])
            candidates.append((score, row))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1], candidates[0][0]
    return None, 0.0


def _aggregate(selected_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    selectors = sorted({str(r["selector"]) for r in selected_rows})
    for selector in selectors:
        group = [r for r in selected_rows if str(r["selector"]) == selector]
        tests = [r["test"] for r in group]
        out[selector] = {
            "folds": len(group),
            "pass_alpha": int(sum(float(t["alpha_excess_pt"]) > 0.0 for t in tests)),
            "pass_maxdd": int(sum(float(t["maxdd_delta_pt"]) <= 0.0 for t in tests)),
            "pass_both": int(sum(float(t["alpha_excess_pt"]) > 0.0 and float(t["maxdd_delta_pt"]) <= 0.0 for t in tests)),
            "pass_both_eps": int(sum(float(t["alpha_excess_pt"]) > 0.0 and float(t["maxdd_delta_pt"]) <= 1e-6 for t in tests)),
            "alpha_mean": _nanmean([t["alpha_excess_pt"] for t in tests]),
            "alpha_worst": _nanmin([t["alpha_excess_pt"] for t in tests]),
            "maxdd_worst": _nanmax([t["maxdd_delta_pt"] for t in tests]),
            "turnover_mean": _nanmean([t["turnover"] for t in tests]),
        }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 1 Period Selector Probe",
        "",
        f"Config: `{payload['config']}`",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Combo scope: `{payload['combo_scope']}`",
        "",
        "## Aggregate",
        "",
        "| selector | folds | Alpha pass | MaxDD pass | both strict | both eps | Alpha mean | Alpha worst | MaxDD worst | turnover |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for selector, row in payload["aggregate"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    selector,
                    str(row["folds"]),
                    str(row["pass_alpha"]),
                    str(row["pass_maxdd"]),
                    str(row["pass_both"]),
                    str(row["pass_both_eps"]),
                    _fmt(row["alpha_mean"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Selected Fold Detail", ""])
    lines.append("| fold | selector | selected combo | val score | val Alpha | val MaxDD | test Alpha | test MaxDD | test turnover | verdict |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["selected"]:
        test = row["test"]
        val = row["val"]
        verdict = "pass" if float(test["alpha_excess_pt"]) > 0.0 and float(test["maxdd_delta_pt"]) <= 0.0 else "fail"
        if float(test["alpha_excess_pt"]) > 0.0 and float(test["maxdd_delta_pt"]) <= 1e-6 and verdict == "fail":
            verdict = "eps-pass"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    str(row["selector"]),
                    str(row["selected_combo"]),
                    _fmt(row["val_score"]),
                    _fmt(val.get("alpha_excess_pt")),
                    _fmt(val.get("maxdd_delta_pt")),
                    _fmt(test.get("alpha_excess_pt")),
                    _fmt(test.get("maxdd_delta_pt")),
                    _fmt(test.get("turnover")),
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
            "- This is an experimental period-level selector. It selects the combo on validation only, then applies the same trained model and threshold to test.",
            "- Threshold selection is still validation-based, so selector results are model-selection probes rather than final production evidence.",
            "- `both eps` treats tiny positive MaxDDDelta up to 1e-6 pt as numerical zero.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.round1_period_selector_probe")
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
    parser.add_argument("--combo-scope", default="vol_shock", choices=["all", "vol_shock", "vol_shock_state", "vol_shock_recovery"])
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
    unit_cost = float(costs_cfg.get("_unit_cost_override", 0.0))
    if unit_cost <= 0.0:
        from unidream.cli.exploration_board_probe import _unit_cost

        unit_cost = _unit_cost(costs_cfg)

    selected_rows: list[dict[str, Any]] = []
    fold_results: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        print(f"[Round1Selector] fold={split.fold_idx} start")
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
        fold_rows: list[dict[str, Any]] = []
        combos = [
            (event_name, label_name, feature_set)
            for event_name in events.keys()
            for label_name in labels["train"].keys()
            for feature_set in ("raw", "state")
            if _combo_allowed((event_name, label_name, feature_set), args.combo_scope)
        ]
        for combo in combos:
            row = _evaluate_combo_with_val(
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
                seed=args.seed + int(split.fold_idx),
                false_active_cap=args.false_active_cap,
                pred_rate_cap=args.pred_rate_cap,
            )
            row["fold"] = int(split.fold_idx)
            fold_rows.append(row)
        ok_rows = [r for r in fold_rows if r.get("status") == "ok"]
        for combo in FIXED_COMBOS:
            name = "fixed_" + "_".join(combo)
            match = next((r for r in ok_rows if (r["event"], r["label"], r["feature_set"]) == combo), None)
            if match is None:
                selected_rows.append(_benchmark_row(int(split.fold_idx), name))
            else:
                selected_rows.append(
                    {
                        "fold": int(split.fold_idx),
                        "selector": name,
                        "selected_combo": match["combo"],
                        "val_score": 0.0,
                        "val": match["val"],
                        "test": match["test"],
                        "status": "fixed",
                    }
                )
        for spec in SPECS:
            scored = [(float(_selector_score(r["val"], spec)), r) for r in ok_rows]
            scored.sort(key=lambda x: x[0], reverse=True)
            if not scored or scored[0][0] <= 0.0:
                selected_rows.append(_benchmark_row(int(split.fold_idx), spec.name))
                continue
            best_score, best = scored[0]
            selected_rows.append(
                {
                    "fold": int(split.fold_idx),
                    "selector": spec.name,
                    "selected_combo": best["combo"],
                    "val_score": best_score,
                    "val": best["val"],
                    "test": best["test"],
                        "status": "selected",
                    }
                )
        priority_best, priority_score = _priority_recovery_veto_select(ok_rows)
        if priority_best is None:
            selected_rows.append(_benchmark_row(int(split.fold_idx), "priority_recovery_veto"))
        else:
            selected_rows.append(
                {
                    "fold": int(split.fold_idx),
                    "selector": "priority_recovery_veto",
                    "selected_combo": priority_best["combo"],
                    "val_score": priority_score,
                    "val": priority_best["val"],
                    "test": priority_best["test"],
                    "status": "selected",
                }
            )
        fold_results[str(split.fold_idx)] = fold_rows

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
        "combo_scope": args.combo_scope,
        "selected": selected_rows,
        "results": fold_results,
        "aggregate": _aggregate(selected_rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[Round1Selector] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
