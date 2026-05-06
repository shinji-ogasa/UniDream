from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import (
    _backtest_positions,
    _candidate_utilities,
    _fit_ridge_multi,
    _future_windows,
    _online_drawdown,
    _pullback_no_fire_mask,
    _rolling_past_sum,
    _rolling_past_vol,
    _shift_for_execution,
    _state_features,
    _top_fraction_mean,
    _triple_barrier_labels,
    _unit_cost,
)
from unidream.cli.plan5_laneF import make_pullback_recovery_label
from unidream.cli.route_separability_probe import (
    _fit_binary_model,
    _safe_ap,
    _safe_auc,
    _score_binary,
    _select_threshold,
)
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "round1_meta_label_probe"
DEFAULT_CANDIDATES = (0.75, 1.0, 1.05)


def _date_prefix() -> str:
    return datetime.now().strftime("%Y%m%d")


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
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


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanmin(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmin(arr))


def _nanmax(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _future_sum(returns: np.ndarray, horizon: int) -> np.ndarray:
    windows, valid = _future_windows(returns, horizon)
    out = np.full(len(returns), np.nan, dtype=np.float64)
    if len(windows):
        out[: windows.shape[0]] = np.sum(windows, axis=1)
    out[~valid] = np.nan
    return out


def _candidate_utility_column(
    returns: np.ndarray,
    *,
    candidates: tuple[float, ...],
    position: float,
    horizon: int,
    benchmark_position: float,
    unit_cost: float,
    dd_penalty: float,
    vol_penalty: float,
) -> np.ndarray:
    values, valid = _candidate_utilities(
        returns,
        candidates=candidates,
        horizon=horizon,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        dd_penalty=dd_penalty,
        vol_penalty=vol_penalty,
    )
    idx = int(np.argmin(np.abs(np.asarray(candidates, dtype=np.float64) - float(position))))
    out = values[:, idx].copy()
    out[~valid] = np.nan
    return out


def _make_label_bundle(
    returns: np.ndarray,
    *,
    candidates: tuple[float, ...],
    horizon: int,
    benchmark_position: float,
    unit_cost: float,
    dd_penalty: float,
    vol_penalty: float,
) -> dict[str, dict[str, np.ndarray | float]]:
    take_utility = _candidate_utility_column(
        returns,
        candidates=candidates,
        position=1.05,
        horizon=horizon,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        dd_penalty=dd_penalty,
        vol_penalty=vol_penalty,
    )
    veto_utility = _candidate_utility_column(
        returns,
        candidates=candidates,
        position=0.75,
        horizon=horizon,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        dd_penalty=dd_penalty,
        vol_penalty=vol_penalty,
    )
    tb = _triple_barrier_labels(returns, horizon=horizon, vol_window=64, barrier_k=1.25)
    recovery = make_pullback_recovery_label(returns, horizon=horizon)
    future = _future_sum(returns, horizon)
    return {
        "take": {
            "y": (take_utility > 0.0).astype(np.int64),
            "utility": take_utility,
            "overlay_position": 1.05,
        },
        "veto": {
            "y": (veto_utility > 0.0).astype(np.int64),
            "utility": veto_utility,
            "overlay_position": 0.75,
        },
        "triple_barrier": {
            "y": np.asarray(tb["tb_up_safe"], dtype=np.int64),
            "utility": take_utility,
            "overlay_position": 1.05,
        },
        "recovery": {
            "y": np.asarray(recovery, dtype=np.int64),
            "utility": future,
            "overlay_position": 1.05,
        },
    }


def _vol_shock_mask(returns: np.ndarray, threshold: float) -> np.ndarray:
    ret = np.asarray(returns, dtype=np.float64)
    vol = _rolling_past_vol(ret, 64)
    return (np.abs(ret) > np.maximum(threshold, 1e-12)) | (vol > threshold)


def _ridge_candidate_masks(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_returns: np.ndarray,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    candidates: tuple[float, ...],
    horizon: int,
    benchmark_position: float,
    unit_cost: float,
    dd_penalty: float,
    vol_penalty: float,
    l2: float,
    event_rate: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    y_train, train_valid = _candidate_utilities(
        train_returns,
        candidates=candidates,
        horizon=horizon,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        dd_penalty=dd_penalty,
        vol_penalty=vol_penalty,
    )
    model = _fit_ridge_multi(x_train[train_valid], y_train[train_valid], l2=l2)
    if model is None:
        empty = {
            "train": np.zeros(len(train_returns), dtype=bool),
            "val": np.zeros(len(val_returns), dtype=bool),
            "test": np.zeros(len(test_returns), dtype=bool),
        }
        return empty, {"status": "no_ridge_model"}

    candidates_arr = np.asarray(candidates, dtype=np.float64)
    bench_idx = int(np.argmin(np.abs(candidates_arr - float(benchmark_position))))

    def score(x: np.ndarray) -> np.ndarray:
        pred = model.predict(x)
        best = np.max(pred, axis=1)
        return best - pred[:, bench_idx]

    score_train = score(x_train)
    score_val = score(x_val)
    score_test = score(x_test)
    finite_val = score_val[np.isfinite(score_val)]
    threshold = float(np.quantile(finite_val, max(0.0, min(1.0, 1.0 - event_rate)))) if len(finite_val) else float("inf")
    return (
        {
            "train": score_train >= threshold,
            "val": score_val >= threshold,
            "test": score_test >= threshold,
        },
        {
            "status": "ok",
            "threshold": threshold,
            "val_event_rate": float(np.mean(score_val >= threshold)) if len(score_val) else float("nan"),
        },
    )


def _event_masks(
    *,
    x_train_state: np.ndarray,
    x_val_state: np.ndarray,
    x_test_state: np.ndarray,
    train_returns: np.ndarray,
    val_returns: np.ndarray,
    test_returns: np.ndarray,
    candidates: tuple[float, ...],
    horizon: int,
    benchmark_position: float,
    unit_cost: float,
    dd_penalty: float,
    vol_penalty: float,
    ridge_l2: float,
    ridge_event_rate: float,
) -> dict[str, dict[str, Any]]:
    train_vol = _rolling_past_vol(train_returns, 64)
    vol_threshold = float(np.quantile(train_vol[np.isfinite(train_vol)], 0.90)) if len(train_vol) else float("inf")
    ridge_masks, ridge_meta = _ridge_candidate_masks(
        x_train=x_train_state,
        x_val=x_val_state,
        x_test=x_test_state,
        train_returns=train_returns,
        val_returns=val_returns,
        test_returns=test_returns,
        candidates=candidates,
        horizon=horizon,
        benchmark_position=benchmark_position,
        unit_cost=unit_cost,
        dd_penalty=dd_penalty,
        vol_penalty=vol_penalty,
        l2=ridge_l2,
        event_rate=ridge_event_rate,
    )
    return {
        "pullback": {
            "masks": {
                "train": _pullback_no_fire_mask(train_returns),
                "val": _pullback_no_fire_mask(val_returns),
                "test": _pullback_no_fire_mask(test_returns),
            },
            "meta": {"status": "ok"},
        },
        "vol_shock": {
            "masks": {
                "train": _vol_shock_mask(train_returns, vol_threshold),
                "val": _vol_shock_mask(val_returns, vol_threshold),
                "test": _vol_shock_mask(test_returns, vol_threshold),
            },
            "meta": {"status": "ok", "train_vol_threshold": vol_threshold},
        },
        "ridge_candidate": {
            "masks": ridge_masks,
            "meta": ridge_meta,
        },
    }


def _valid_eval_mask(x: np.ndarray, y: np.ndarray, utility: np.ndarray, event_mask: np.ndarray) -> np.ndarray:
    n = min(len(x), len(y), len(utility), len(event_mask))
    return (
        np.asarray(event_mask[:n], dtype=bool)
        & np.isfinite(np.asarray(y[:n], dtype=np.float64))
        & np.isfinite(np.asarray(utility[:n], dtype=np.float64))
        & np.all(np.isfinite(np.asarray(x[:n], dtype=np.float64)), axis=1)
    )


def _evaluate_combo(
    *,
    feature_set: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    event_name: str,
    event: dict[str, Any],
    label_name: str,
    train_label: dict[str, np.ndarray | float],
    val_label: dict[str, np.ndarray | float],
    test_label: dict[str, np.ndarray | float],
    test_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    benchmark_position: float,
    max_train_samples: int,
    seed: int,
    false_active_cap: float,
    pred_rate_cap: float,
) -> dict[str, Any]:
    train_y = np.asarray(train_label["y"], dtype=np.int64)
    val_y = np.asarray(val_label["y"], dtype=np.int64)
    test_y = np.asarray(test_label["y"], dtype=np.int64)
    train_utility = np.asarray(train_label["utility"], dtype=np.float64)
    val_utility = np.asarray(val_label["utility"], dtype=np.float64)
    test_utility = np.asarray(test_label["utility"], dtype=np.float64)
    train_mask = _valid_eval_mask(x_train, train_y, train_utility, event["masks"]["train"])
    val_mask = _valid_eval_mask(x_val, val_y, val_utility, event["masks"]["val"])
    test_mask = _valid_eval_mask(x_test, test_y, test_utility, event["masks"]["test"])

    candidate_count = int(test_mask.sum())
    out: dict[str, Any] = {
        "status": "ok",
        "event": event_name,
        "label": label_name,
        "feature_set": feature_set,
        "candidate_count": candidate_count,
        "positive_rate": float(np.mean(test_y[test_mask])) if candidate_count else float("nan"),
        "event_meta": event.get("meta", {}),
    }
    if int(train_mask.sum()) < 100 or int(val_mask.sum()) < 20 or candidate_count < 20:
        out["status"] = "insufficient_events"
        out["train_count"] = int(train_mask.sum())
        out["val_count"] = int(val_mask.sum())
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
    val_score = _score_binary(model, x_val[val_mask])
    threshold, val_rates = _select_threshold(
        val_y[val_mask],
        val_score,
        false_active_cap=false_active_cap,
        pred_rate_cap=pred_rate_cap,
    )
    test_score_full = _score_binary(model, x_test)
    test_score = test_score_full[test_mask]
    test_y_event = test_y[test_mask]
    test_utility_event = test_utility[test_mask]
    pred_event = test_score >= threshold

    selected = np.full(len(test_returns), float(benchmark_position), dtype=np.float64)
    selected_idx = np.flatnonzero(test_mask)[pred_event]
    selected[selected_idx] = float(test_label["overlay_position"])
    shifted = _shift_for_execution(selected, benchmark_position)
    bt, _pnl = _backtest_positions(
        test_returns,
        shifted,
        cfg=cfg,
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
    )

    out.update(
        {
            "train_count": int(train_mask.sum()),
            "val_count": int(val_mask.sum()),
            "auc": _safe_auc(test_y_event, test_score),
            "ap": _safe_ap(test_y_event, test_score),
            "top10_precision": _top_fraction_mean(test_score, test_y_event, 0.10),
            "top10_utility": _top_fraction_mean(test_score, test_utility_event, 0.10),
            "threshold_selected_on_val": val_rates,
            "threshold": float(threshold) if math.isfinite(float(threshold)) else str(threshold),
            "overlay": bt,
        }
    )
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    keys = sorted({(r["event"], r["label"], r["feature_set"]) for r in rows if r.get("status") == "ok"})
    for event_name, label_name, feature_set in keys:
        group = [
            r
            for r in rows
            if r.get("status") == "ok"
            and r["event"] == event_name
            and r["label"] == label_name
            and r["feature_set"] == feature_set
        ]
        overlay = [r["overlay"] for r in group]
        name = f"{event_name}/{label_name}/{feature_set}"
        out[name] = {
            "folds": int(len(group)),
            "candidate_count_mean": _nanmean([r["candidate_count"] for r in group]),
            "positive_rate_mean": _nanmean([r["positive_rate"] for r in group]),
            "auc_mean": _nanmean([r["auc"] for r in group]),
            "auc_worst": _nanmin([r["auc"] for r in group]),
            "ap_mean": _nanmean([r["ap"] for r in group]),
            "top10_precision_mean": _nanmean([r["top10_precision"] for r in group]),
            "top10_utility_mean": _nanmean([r["top10_utility"] for r in group]),
            "alpha_mean": _nanmean([o["alpha_excess_pt"] for o in overlay]),
            "alpha_worst": _nanmin([o["alpha_excess_pt"] for o in overlay]),
            "maxdd_worst": _nanmax([o["maxdd_delta_pt"] for o in overlay]),
            "turnover_mean": _nanmean([o["turnover"] for o in overlay]),
        }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Round 1 Meta-Label Probe",
        "",
        f"Config: `{payload['config']}`",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Feature sets: `{', '.join(payload['feature_sets'])}`",
        f"WM status: `{payload['wm_status']}`",
        "",
        "## Aggregate",
        "",
        "| combo | folds | candidates | positive | AUC mean | AUC worst | AP mean | top10 precision | top10 utility | AlphaEx mean | AlphaEx worst | MaxDDDelta worst | turnover |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["aggregate"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["folds"]),
                    _fmt(row["candidate_count_mean"], 1),
                    _fmt(row["positive_rate_mean"]),
                    _fmt(row["auc_mean"]),
                    _fmt(row["auc_worst"]),
                    _fmt(row["ap_mean"]),
                    _fmt(row["top10_precision_mean"]),
                    _fmt(row["top10_utility_mean"]),
                    _fmt(row["alpha_mean"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_mean"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Fold Detail", ""])
    for fold in payload["folds"]:
        lines.extend([f"### Fold {fold}", ""])
        lines.append("| event | label | features | status | candidates | positive | AUC | AP | top10 precision | top10 utility | AlphaEx | MaxDDDelta | turnover |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in payload["results"].get(str(fold), []):
            overlay = row.get("overlay", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.get("event", ""),
                        row.get("label", ""),
                        row.get("feature_set", ""),
                        row.get("status", ""),
                        str(row.get("candidate_count", 0)),
                        _fmt(row.get("positive_rate")),
                        _fmt(row.get("auc")),
                        _fmt(row.get("ap")),
                        _fmt(row.get("top10_precision")),
                        _fmt(row.get("top10_utility")),
                        _fmt(overlay.get("alpha_excess_pt")),
                        _fmt(overlay.get("maxdd_delta_pt")),
                        _fmt(overlay.get("turnover")),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- Scope: experimental CLI only; no config or production behavior changes.",
            "- Thresholds are selected on validation and applied once to test.",
            "- WM feature-set support is intentionally an extension point in this first lightweight version.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.round1_meta_label_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,4,5")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--ridge-event-rate", type=float, default=0.10)
    parser.add_argument("--dd-penalty", type=float, default=1.0)
    parser.add_argument("--vol-penalty", type=float, default=0.10)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--false-active-cap", type=float, default=0.15)
    parser.add_argument("--pred-rate-cap", type=float, default=0.25)
    parser.add_argument("--feature-sets", default="raw,state")
    parser.add_argument("--include-wm", action="store_true", help="Reserved extension point; skipped unless implemented.")
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
    requested_features = [x.strip() for x in str(args.feature_sets).split(",") if x.strip()]
    feature_sets = [x for x in requested_features if x in {"raw", "state"}]
    wm_status = "not_requested"
    if args.include_wm or "wm" in requested_features:
        wm_status = "skipped_no_loader"

    all_results: dict[str, list[dict[str, Any]]] = {}
    flat_rows: list[dict[str, Any]] = []
    for split in splits:
        print(f"[Round1Meta] fold={split.fold_idx} start")
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
        for event_name, event in events.items():
            for label_name in labels["train"].keys():
                for feature_set in feature_sets:
                    print(f"[Round1Meta] fold={split.fold_idx} event={event_name} label={label_name} features={feature_set}")
                    row = _evaluate_combo(
                        feature_set=feature_set,
                        x_train=train_sets[feature_set],
                        x_val=val_sets[feature_set],
                        x_test=test_sets[feature_set],
                        event_name=event_name,
                        event=event,
                        label_name=label_name,
                        train_label=labels["train"][label_name],
                        val_label=labels["val"][label_name],
                        test_label=labels["test"][label_name],
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
                    flat_rows.append(row)
        all_results[str(split.fold_idx)] = fold_rows

    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": args.config,
        "start": args.start,
        "end": args.end,
        "folds": [int(split.fold_idx) for split in splits],
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "candidates": list(DEFAULT_CANDIDATES),
        "feature_sets": feature_sets,
        "requested_feature_sets": requested_features,
        "wm_status": wm_status,
        "results": all_results,
        "aggregate": _aggregate(flat_rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[Round1Meta] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
