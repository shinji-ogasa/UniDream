from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from unidream.actor_critic.imagination_ac import _action_stats
from unidream.cli.route_separability_probe import (
    _as_2d,
    _binary_eval,
    _concat_features,
    _finite_rows,
    _fit_binary_model,
    _fmt,
    _sample_indices,
    _score_binary,
    _select_threshold,
)
from unidream.data.dataset import WFODataset
from unidream.data.oracle import _forward_window_stats
from unidream.device import add_device_argument, resolve_device
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.experiments.wm_stage import prepare_world_model_stage


TARGET_NAMES = ("active", "risk_off", "recovery", "overweight")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_action_stats(stats: dict) -> str:
    return (
        f"long={stats['long']:.0%} short={stats['short']:.0%} flat={stats['flat']:.0%} "
        f"mean={stats['mean']:+.3f} switches={stats['switches']} "
        f"avg_hold={stats['avg_hold']:.1f}b turnover={stats['turnover']:.2f}"
    )


def _benchmark_position_value(cfg: dict) -> float:
    return float(cfg.get("reward", {}).get("benchmark_position", 1.0))


def _unit_cost(costs_cfg: dict) -> float:
    return (
        float(costs_cfg.get("spread_bps", 3.0)) / 10000.0 / 2.0
        + float(costs_cfg.get("fee_rate", 0.0003))
        + float(costs_cfg.get("slippage_bps", 1.0)) / 10000.0
    )


def _future_windows(returns: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(returns, dtype=np.float64)
    h = int(max(horizon, 1))
    if len(r) <= h:
        return np.empty((0, h), dtype=np.float64), np.zeros(len(r), dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(r[1:], h)
    valid_n = len(r) - h
    valid = np.zeros(len(r), dtype=bool)
    valid[:valid_n] = True
    return windows[:valid_n].copy(), valid


def _path_max_drawdown(windows: np.ndarray, position: float) -> np.ndarray:
    if windows.size == 0:
        return np.zeros(0, dtype=np.float64)
    path = np.cumsum(windows * float(position), axis=1)
    path = np.concatenate([np.zeros((path.shape[0], 1), dtype=np.float64), path], axis=1)
    peak = np.maximum.accumulate(path, axis=1)
    return np.max(peak - path, axis=1)


def _running_underwater(returns: np.ndarray) -> np.ndarray:
    r = np.asarray(returns, dtype=np.float64)
    equity = np.cumsum(r, dtype=np.float64)
    peak = np.maximum.accumulate(np.maximum(equity, 0.0))
    return peak - equity


def _event_labels(
    returns: np.ndarray,
    *,
    horizon: int,
    benchmark_position: float,
    unit_cost: float,
    risk_off_candidates: tuple[float, ...],
    overweight_position: float,
    risk_off_min_dd: float,
    risk_off_margin: float,
    recovery_min_underwater: float,
    recovery_score_margin: float,
    recovery_max_future_dd: float,
    overweight_margin: float,
    overweight_max_dd_worsen: float,
) -> dict[str, np.ndarray]:
    r = np.asarray(returns, dtype=np.float64)
    windows, valid = _future_windows(r, horizon)
    n = len(r)
    score_arrays = {
        "risk_off_score": np.full(n, np.nan, dtype=np.float64),
        "recovery_score": np.full(n, np.nan, dtype=np.float64),
        "overweight_score": np.full(n, np.nan, dtype=np.float64),
        "future_dd": np.full(n, np.nan, dtype=np.float64),
        "future_sum": np.full(n, np.nan, dtype=np.float64),
    }
    label_arrays = {
        "risk_off": np.zeros(n, dtype=np.int64),
        "recovery": np.zeros(n, dtype=np.int64),
        "overweight": np.zeros(n, dtype=np.int64),
    }
    if len(windows) == 0:
        label_arrays["active"] = np.zeros(n, dtype=np.int64)
        return {**label_arrays, **score_arrays, "valid": valid.astype(np.int64)}

    valid_n = windows.shape[0]
    bench = float(benchmark_position)
    bench_dd = _path_max_drawdown(windows, bench)
    future_sum = np.sum(windows * bench, axis=1)
    score_arrays["future_dd"][:valid_n] = bench_dd
    score_arrays["future_sum"][:valid_n] = future_sum

    risk_scores = []
    for candidate in risk_off_candidates:
        candidate = float(candidate)
        candidate_dd = _path_max_drawdown(windows, candidate)
        trade_cost = abs(candidate - bench) * unit_cost
        risk_scores.append(bench_dd - candidate_dd - trade_cost)
    risk_score = np.max(np.stack(risk_scores, axis=1), axis=1) if risk_scores else np.zeros(valid_n)
    risk_label = (bench_dd >= risk_off_min_dd) & (risk_score > risk_off_margin)
    label_arrays["risk_off"][:valid_n] = risk_label.astype(np.int64)
    score_arrays["risk_off_score"][:valid_n] = risk_score

    underwater = _running_underwater(r)
    future_dd = bench_dd
    recovery_score = future_sum - 0.5 * future_dd
    recovery_label = (
        (underwater[:valid_n] >= recovery_min_underwater)
        & (future_dd <= recovery_max_future_dd)
        & (recovery_score > recovery_score_margin)
    )
    label_arrays["recovery"][:valid_n] = recovery_label.astype(np.int64)
    score_arrays["recovery_score"][:valid_n] = recovery_score

    ow = float(overweight_position)
    ow_dd = _path_max_drawdown(windows, ow)
    ow_trade_cost = abs(ow - bench) * unit_cost
    ow_advantage = np.sum(windows * (ow - bench), axis=1) - ow_trade_cost
    ow_dd_worsen = ow_dd - bench_dd
    ow_score = ow_advantage - np.maximum(ow_dd_worsen, 0.0)
    ow_label = (ow_advantage > overweight_margin) & (ow_dd_worsen <= overweight_max_dd_worsen)
    label_arrays["overweight"][:valid_n] = ow_label.astype(np.int64)
    score_arrays["overweight_score"][:valid_n] = ow_score

    active = (
        (label_arrays["risk_off"] == 1)
        | (label_arrays["recovery"] == 1)
        | (label_arrays["overweight"] == 1)
    )
    label_arrays["active"] = active.astype(np.int64)
    return {**label_arrays, **score_arrays, "valid": valid.astype(np.int64)}


def _build_feature_sets(
    *,
    raw: np.ndarray,
    enc: dict,
    regime: np.ndarray | None,
    predictive: np.ndarray | None,
) -> dict[str, np.ndarray]:
    z = _as_2d(enc.get("z"))
    h = _as_2d(enc.get("h"))
    pred = _as_2d(predictive)
    reg = _as_2d(regime)
    out = {
        "raw": _concat_features(raw),
        "wm": _concat_features(z, h),
        "raw_wm": _concat_features(raw, z, h),
    }
    if pred is not None:
        out["predictive"] = _concat_features(pred)
        out["raw_predictive"] = _concat_features(raw, pred)
        out["wm_predictive"] = _concat_features(z, h, pred)
        out["raw_wm_predictive"] = _concat_features(raw, z, h, pred)
    if reg is not None:
        out["regime"] = _concat_features(reg)
        out["raw_regime"] = _concat_features(raw, reg)
    if pred is not None and reg is not None:
        out["context_no_position"] = _concat_features(reg, pred)
        out["raw_context_no_position"] = _concat_features(raw, reg, pred)
        out["wm_context_no_position"] = _concat_features(z, h, reg, pred)
        out["raw_wm_context_no_position"] = _concat_features(raw, z, h, reg, pred)
    return out


def _density(labels: np.ndarray, valid: np.ndarray) -> float:
    mask = np.asarray(valid, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.asarray(labels)[mask]))


def _top_decile_score(score: np.ndarray, valid: np.ndarray) -> float:
    s = np.asarray(score, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(s)
    if mask.sum() < 20:
        return float("nan")
    vals = s[mask]
    threshold = np.quantile(vals, 0.90)
    return float(np.mean(vals[vals >= threshold]))


def _fit_probe_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_train_samples: int,
    seed: int,
    model_type: str,
) -> Any | None:
    if model_type == "logistic":
        return _fit_binary_model(x, y, max_train_samples=max_train_samples, seed=seed)
    if model_type != "hgb":
        raise ValueError(f"Unknown probe model: {model_type}")

    mask = _finite_rows(x, y)
    idx = np.flatnonzero(mask)
    idx = _sample_indices(idx, max_train_samples, seed)
    if len(idx) < 100 or len(np.unique(y[idx])) < 2:
        return None

    y_fit = np.asarray(y[idx], dtype=np.int64)
    pos = max(int((y_fit == 1).sum()), 1)
    neg = max(int((y_fit == 0).sum()), 1)
    sample_weight = np.where(y_fit == 1, len(y_fit) / (2.0 * pos), len(y_fit) / (2.0 * neg))
    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=12,
        random_state=seed,
    )
    model.fit(x[idx], y_fit, sample_weight=sample_weight)
    return model


def _evaluate_target(
    *,
    target_name: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_label: np.ndarray,
    val_label: np.ndarray,
    test_label: np.ndarray,
    train_valid: np.ndarray,
    val_valid: np.ndarray,
    test_valid: np.ndarray,
    max_train_samples: int,
    seed: int,
    model_type: str,
    false_active_cap: float,
    pred_rate_cap: float,
) -> dict:
    n_train = min(len(train_x), len(train_label), len(train_valid))
    n_val = min(len(val_x), len(val_label), len(val_valid))
    n_test = min(len(test_x), len(test_label), len(test_valid))
    train_x = train_x[:n_train]
    val_x = val_x[:n_val]
    test_x = test_x[:n_test]
    train_y = np.asarray(train_label[:n_train], dtype=np.int64)
    val_y = np.asarray(val_label[:n_val], dtype=np.int64)
    test_y = np.asarray(test_label[:n_test], dtype=np.int64)
    train_mask = np.asarray(train_valid[:n_train], dtype=bool)
    val_mask = np.asarray(val_valid[:n_val], dtype=bool)
    test_mask = np.asarray(test_valid[:n_test], dtype=bool)

    model = _fit_probe_model(
        train_x[train_mask],
        train_y[train_mask],
        max_train_samples=max_train_samples,
        seed=seed,
        model_type=model_type,
    )
    val_score = _score_binary(model, val_x[val_mask])
    threshold, val_rates = _select_threshold(
        val_y[val_mask],
        val_score,
        false_active_cap=false_active_cap,
        pred_rate_cap=pred_rate_cap,
    )
    return {
        "target": target_name,
        "density": {
            "train": _density(train_y, train_mask),
            "val": _density(val_y, val_mask),
            "test": _density(test_y, test_mask),
        },
        "threshold_selected_on_val": val_rates,
        "train": _binary_eval(model=model, x=train_x[train_mask], y=train_y[train_mask], threshold=threshold),
        "val": _binary_eval(model=model, x=val_x[val_mask], y=val_y[val_mask], threshold=threshold),
        "test": _binary_eval(model=model, x=test_x[test_mask], y=test_y[test_mask], threshold=threshold),
    }


def _evaluate_feature_set(
    *,
    feature_name: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_labels: dict[str, np.ndarray],
    val_labels: dict[str, np.ndarray],
    test_labels: dict[str, np.ndarray],
    max_train_samples: int,
    seed: int,
    model_type: str,
    false_active_cap: float,
    pred_rate_cap: float,
) -> dict:
    out: dict[str, Any] = {"feature_set": feature_name, "targets": {}}
    for idx, target in enumerate(TARGET_NAMES):
        out["targets"][target] = _evaluate_target(
            target_name=target,
            train_x=train_x,
            val_x=val_x,
            test_x=test_x,
            train_label=train_labels[target],
            val_label=val_labels[target],
            test_label=test_labels[target],
            train_valid=train_labels["valid"],
            val_valid=val_labels["valid"],
            test_valid=test_labels["valid"],
            max_train_samples=max_train_samples,
            seed=seed + idx,
            model_type=model_type,
            false_active_cap=false_active_cap,
            pred_rate_cap=pred_rate_cap,
        )
    out["score_top_decile"] = {
        "risk_off": _top_decile_score(test_labels["risk_off_score"], test_labels["valid"]),
        "recovery": _top_decile_score(test_labels["recovery_score"], test_labels["valid"]),
        "overweight": _top_decile_score(test_labels["overweight_score"], test_labels["valid"]),
        "future_dd": _top_decile_score(test_labels["future_dd"], test_labels["valid"]),
        "future_sum": _top_decile_score(test_labels["future_sum"], test_labels["valid"]),
    }
    return out


def _finite_values(rows: list[float]) -> list[float]:
    return [float(v) for v in rows if math.isfinite(float(v))]


def _mean(rows: list[float]) -> float:
    vals = _finite_values(rows)
    return float(np.mean(vals)) if vals else float("nan")


def _min(rows: list[float]) -> float:
    vals = _finite_values(rows)
    return float(np.min(vals)) if vals else float("nan")


def _max(rows: list[float]) -> float:
    vals = _finite_values(rows)
    return float(np.max(vals)) if vals else float("nan")


def _aggregate(results: dict[str, dict]) -> dict:
    feature_sets = sorted({name for fold in results.values() for name in fold.keys()})
    out: dict[str, dict] = {}
    for feature in feature_sets:
        out[feature] = {}
        for target in TARGET_NAMES:
            rows = [
                fold[feature]["targets"][target]["test"]
                for fold in results.values()
                if feature in fold and target in fold[feature]["targets"]
            ]
            densities = [
                fold[feature]["targets"][target]["density"]["test"]
                for fold in results.values()
                if feature in fold and target in fold[feature]["targets"]
            ]
            out[feature][target] = {
                "folds": int(len(rows)),
                "auc_mean": _mean([r.get("auc", float("nan")) for r in rows]),
                "auc_worst": _min([r.get("auc", float("nan")) for r in rows]),
                "ap_mean": _mean([r.get("ap", float("nan")) for r in rows]),
                "recall_mean": _mean([r.get("recall", float("nan")) for r in rows]),
                "recall_worst": _min([r.get("recall", float("nan")) for r in rows]),
                "false_active_mean": _mean([r.get("false_active_rate", float("nan")) for r in rows]),
                "false_active_worst": _max([r.get("false_active_rate", float("nan")) for r in rows]),
                "pred_rate_mean": _mean([r.get("pred_rate", float("nan")) for r in rows]),
                "density_mean": _mean(densities),
            }
    return out


def _write_md(path: str, payload: dict) -> None:
    lines = [
        "# Market Event Label Probe",
        "",
        f"Config: `{payload['config']}`",
        f"Checkpoint dir: `{payload['checkpoint_dir']}`",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Horizon: `{payload['label_params']['horizon']}`",
        f"Probe model: `{payload['model']}`",
        "",
        "## Aggregate",
        "",
    ]
    for target in TARGET_NAMES:
        lines.extend(
            [
                f"### {target}",
                "",
                "| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for feature, feature_rows in payload["aggregate"].items():
            row = feature_rows[target]
            lines.append(
                "| "
                + " | ".join(
                    [
                        feature,
                        str(row["folds"]),
                        _fmt(row["density_mean"]),
                        _fmt(row["auc_mean"]),
                        _fmt(row["auc_worst"]),
                        _fmt(row["ap_mean"]),
                        _fmt(row["recall_mean"]),
                        _fmt(row["recall_worst"]),
                        _fmt(row["false_active_mean"]),
                        _fmt(row["false_active_worst"]),
                        _fmt(row["pred_rate_mean"]),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(["## Fold Detail", ""])
    for fold, fold_rows in payload["results"].items():
        lines.extend([f"### Fold {fold}", ""])
        lines.append("| feature set | target | density | AUC | AP | recall | false-active | pred-active |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for feature, feature_result in fold_rows.items():
            for target in TARGET_NAMES:
                row = feature_result["targets"][target]
                test = row["test"]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            feature,
                            target,
                            _fmt(row["density"]["test"]),
                            _fmt(test.get("auc")),
                            _fmt(test.get("ap")),
                            _fmt(test.get("recall")),
                            _fmt(test.get("false_active_rate")),
                            _fmt(test.get("pred_rate")),
                        ]
                    )
                    + " |"
                )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.market_event_label_probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--model", choices=["logistic", "hgb"], default="logistic")
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--false-active-cap", type=float, default=0.15)
    parser.add_argument("--pred-rate-cap", type=float, default=0.25)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--risk-off-min-dd", type=float, default=0.006)
    parser.add_argument("--risk-off-margin", type=float, default=0.0015)
    parser.add_argument("--recovery-min-underwater", type=float, default=0.03)
    parser.add_argument("--recovery-score-margin", type=float, default=0.001)
    parser.add_argument("--recovery-max-future-dd", type=float, default=0.035)
    parser.add_argument("--overweight-margin", type=float, default=0.001)
    parser.add_argument("--overweight-max-dd-worsen", type=float, default=0.002)
    parser.add_argument("--overweight-position", type=float, default=1.25)
    parser.add_argument("--risk-off-candidates", default="0.0,0.5")
    parser.add_argument("--feature-sets", default="")
    parser.add_argument("--output-json", default="documents/market_event_label_probe.json")
    parser.add_argument("--output-md", default="documents/market_event_label_probe.md")
    add_device_argument(parser)
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _active_profile = resolve_costs(cfg, None)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = cfg.get("logging", {}).get("checkpoint_dir", "checkpoints")

    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_dir = "checkpoints/data_cache"
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    costs_cfg = cfg.get("costs", {})
    reward_cfg = cfg.get("reward", {})
    unit_cost = _unit_cost(costs_cfg)
    benchmark_position = _benchmark_position_value(cfg)
    risk_off_candidates = tuple(
        float(x.strip()) for x in str(args.risk_off_candidates).split(",") if x.strip()
    )
    feature_filter = {x.strip() for x in str(args.feature_sets or "").split(",") if x.strip()}
    label_params = {
        "horizon": int(args.horizon),
        "risk_off_min_dd": float(args.risk_off_min_dd),
        "risk_off_margin": float(args.risk_off_margin),
        "recovery_min_underwater": float(args.recovery_min_underwater),
        "recovery_score_margin": float(args.recovery_score_margin),
        "recovery_max_future_dd": float(args.recovery_max_future_dd),
        "overweight_margin": float(args.overweight_margin),
        "overweight_max_dd_worsen": float(args.overweight_max_dd_worsen),
        "overweight_position": float(args.overweight_position),
        "risk_off_candidates": list(risk_off_candidates),
    }

    all_results: dict[str, dict] = {}
    for split in splits:
        print(f"[MarketEventProbe] fold={split.fold_idx} start")
        wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
        fold_runtime = prepare_fold_runtime(
            fold_idx=split.fold_idx,
            checkpoint_dir=args.checkpoint_dir,
            ac_cfg=cfg.get("ac", {}),
            resume=False,
            start_from="test",
            stop_after="test",
        )
        if not fold_runtime["has_wm_ckpt"]:
            raise FileNotFoundError(f"Missing WM checkpoint for fold {split.fold_idx}: {fold_runtime['wm_path']}")

        fold_inputs = prepare_fold_inputs(
            wfo_dataset=wfo_dataset,
            cfg=cfg,
            costs_cfg=costs_cfg,
            ac_cfg=cfg.get("ac", {}),
            bc_cfg=cfg.get("bc", {}),
            reward_cfg=reward_cfg,
            action_stats_fn=_action_stats,
            format_action_stats_fn=_fmt_action_stats,
            benchmark_position=benchmark_position,
            forward_window_stats_fn=_forward_window_stats,
            log_ts=_ts,
        )
        ensemble, wm_trainer = prepare_world_model_stage(
            fold_idx=split.fold_idx,
            obs_dim=wfo_dataset.obs_dim,
            cfg=cfg,
            device=args.device,
            has_wm=True,
            wm_path=fold_runtime["wm_path"],
            wfo_dataset=wfo_dataset,
            oracle_positions=fold_inputs["oracle_positions"],
            val_oracle_positions=fold_inputs["val_oracle_positions"],
            train_returns=fold_inputs["train_returns"],
            train_regime_probs=fold_inputs["train_regime_probs"],
            val_regime_probs=fold_inputs["val_regime_probs"],
            log_ts=_ts,
        )
        seq_len = cfg.get("data", {}).get("seq_len", 64)
        enc_train = wm_trainer.encode_sequence(wfo_dataset.train_features, actions=None, seq_len=seq_len)
        enc_val = wm_trainer.encode_sequence(wfo_dataset.val_features, actions=None, seq_len=seq_len)
        enc_test = wm_trainer.encode_sequence(wfo_dataset.test_features, actions=None, seq_len=seq_len)
        predictive_bundle = build_wm_predictive_state_bundle(
            wm_trainer=wm_trainer,
            wfo_dataset=wfo_dataset,
            z_train=enc_train["z"],
            h_train=enc_train["h"],
            seq_len=seq_len,
            ac_cfg=cfg.get("ac", {}),
            log_ts=_ts,
        )
        train_predictive = predictive_bundle["train"] if predictive_bundle is not None else None
        val_predictive = predictive_bundle["val"] if predictive_bundle is not None else None
        test_predictive = predictive_bundle["test"] if predictive_bundle is not None else None

        train_labels = _event_labels(
            wfo_dataset.train_returns,
            horizon=args.horizon,
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            risk_off_candidates=risk_off_candidates,
            overweight_position=args.overweight_position,
            risk_off_min_dd=args.risk_off_min_dd,
            risk_off_margin=args.risk_off_margin,
            recovery_min_underwater=args.recovery_min_underwater,
            recovery_score_margin=args.recovery_score_margin,
            recovery_max_future_dd=args.recovery_max_future_dd,
            overweight_margin=args.overweight_margin,
            overweight_max_dd_worsen=args.overweight_max_dd_worsen,
        )
        val_labels = _event_labels(
            wfo_dataset.val_returns,
            horizon=args.horizon,
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            risk_off_candidates=risk_off_candidates,
            overweight_position=args.overweight_position,
            risk_off_min_dd=args.risk_off_min_dd,
            risk_off_margin=args.risk_off_margin,
            recovery_min_underwater=args.recovery_min_underwater,
            recovery_score_margin=args.recovery_score_margin,
            recovery_max_future_dd=args.recovery_max_future_dd,
            overweight_margin=args.overweight_margin,
            overweight_max_dd_worsen=args.overweight_max_dd_worsen,
        )
        test_labels = _event_labels(
            wfo_dataset.test_returns,
            horizon=args.horizon,
            benchmark_position=benchmark_position,
            unit_cost=unit_cost,
            risk_off_candidates=risk_off_candidates,
            overweight_position=args.overweight_position,
            risk_off_min_dd=args.risk_off_min_dd,
            risk_off_margin=args.risk_off_margin,
            recovery_min_underwater=args.recovery_min_underwater,
            recovery_score_margin=args.recovery_score_margin,
            recovery_max_future_dd=args.recovery_max_future_dd,
            overweight_margin=args.overweight_margin,
            overweight_max_dd_worsen=args.overweight_max_dd_worsen,
        )

        train_sets = _build_feature_sets(
            raw=wfo_dataset.train_features,
            enc=enc_train,
            regime=fold_inputs["train_regime_probs"],
            predictive=train_predictive,
        )
        val_sets = _build_feature_sets(
            raw=wfo_dataset.val_features,
            enc=enc_val,
            regime=fold_inputs["val_regime_probs"],
            predictive=val_predictive,
        )
        test_sets = _build_feature_sets(
            raw=wfo_dataset.test_features,
            enc=enc_test,
            regime=fold_inputs["test_regime_probs"],
            predictive=test_predictive,
        )

        fold_results = {}
        for name in train_sets:
            if feature_filter and name not in feature_filter:
                continue
            print(f"[MarketEventProbe] fold={split.fold_idx} feature_set={name}")
            fold_results[name] = _evaluate_feature_set(
                feature_name=name,
                train_x=train_sets[name],
                val_x=val_sets[name],
                test_x=test_sets[name],
                train_labels=train_labels,
                val_labels=val_labels,
                test_labels=test_labels,
                max_train_samples=args.max_train_samples,
                seed=args.seed + int(split.fold_idx),
                model_type=args.model,
                false_active_cap=args.false_active_cap,
                pred_rate_cap=args.pred_rate_cap,
            )
        all_results[str(split.fold_idx)] = fold_results

    payload = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "folds": [int(split.fold_idx) for split in splits],
        "label_params": label_params,
        "false_active_cap": float(args.false_active_cap),
        "pred_rate_cap": float(args.pred_rate_cap),
        "max_train_samples": int(args.max_train_samples),
        "model": args.model,
        "results": all_results,
        "aggregate": _aggregate(all_results),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[MarketEventProbe] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
