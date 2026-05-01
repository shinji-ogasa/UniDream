from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unidream.actor_critic.imagination_ac import _action_stats
from unidream.data.dataset import WFODataset
from unidream.data.oracle import _forward_window_stats
from unidream.device import add_device_argument, resolve_device
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.experiments.wm_stage import prepare_world_model_stage
from unidream.cli.route_probe import _benchmark_position_value, _test_route_targets


ROUTE_NAMES = ("neutral", "de_risk", "recovery", "overweight")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _fmt_action_stats(stats: dict) -> str:
    return (
        f"long={stats['long']:.0%} short={stats['short']:.0%} flat={stats['flat']:.0%} "
        f"mean={stats['mean']:+.3f} switches={stats['switches']} "
        f"avg_hold={stats['avg_hold']:.1f}b turnover={stats['turnover']:.2f}"
    )


def _as_2d(x: np.ndarray | None, n: int | None = None) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if n is not None:
        arr = arr[:n]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _concat_features(*parts: np.ndarray | None) -> np.ndarray:
    clean = [p for p in (_as_2d(part) for part in parts) if p is not None]
    if not clean:
        raise ValueError("at least one feature array is required")
    n = min(len(part) for part in clean)
    return np.concatenate([part[:n] for part in clean], axis=1).astype(np.float32)


def _finite_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.all(np.isfinite(x), axis=1)


def _sample_indices(idx: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or len(idx) <= max_samples:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_samples, replace=False))


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    if len(y) < 20 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def _fit_binary_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_train_samples: int,
    seed: int,
) -> Any | None:
    mask = _finite_rows(x, y)
    idx = np.flatnonzero(mask)
    idx = _sample_indices(idx, max_train_samples, seed)
    if len(idx) < 100 or len(np.unique(y[idx])) < 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    model.fit(x[idx], y[idx].astype(np.int64))
    return model


def _fit_multiclass_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_train_samples: int,
    seed: int,
) -> Any | None:
    mask = _finite_rows(x, y)
    idx = np.flatnonzero(mask)
    idx = _sample_indices(idx, max_train_samples, seed)
    if len(idx) < 100 or len(np.unique(y[idx])) < 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        ),
    )
    model.fit(x[idx], y[idx].astype(np.int64))
    return model


def _score_binary(model: Any | None, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(x), np.nan, dtype=np.float64)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        return proba[:, -1].astype(np.float64)
    decision = model.decision_function(x)
    return np.asarray(decision, dtype=np.float64)


def _threshold_candidates(score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, dtype=np.float64)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.asarray([float("inf")], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, 401)
    values = np.unique(np.quantile(s, qs))
    return np.concatenate([[float("inf")], values[::-1], [float("-inf")]]).astype(np.float64)


def _binary_rates(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y_i = np.asarray(y, dtype=np.int64)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s)
    y_i = y_i[mask]
    s = s[mask]
    if len(y_i) == 0:
        return {
            "n": 0,
            "positive_rate": float("nan"),
            "pred_rate": float("nan"),
            "recall": float("nan"),
            "precision": float("nan"),
            "false_active_rate": float("nan"),
            "f1": float("nan"),
        }
    pred = s >= threshold
    pos = y_i == 1
    neg = ~pos
    recall = float((pred & pos).sum() / max(pos.sum(), 1))
    precision = float((pred & pos).sum() / max(pred.sum(), 1))
    false_active = float((pred & neg).sum() / max(neg.sum(), 1))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "n": int(len(y_i)),
        "positive_rate": float(pos.mean()),
        "pred_rate": float(pred.mean()),
        "recall": recall,
        "precision": precision,
        "false_active_rate": false_active,
        "f1": float(f1),
    }


def _select_threshold(
    y: np.ndarray,
    score: np.ndarray,
    *,
    false_active_cap: float,
    pred_rate_cap: float,
) -> tuple[float, dict]:
    best_threshold = float("inf")
    best_rates = _binary_rates(y, score, best_threshold)
    best_key = (-1.0, -1.0, 0.0)
    for threshold in _threshold_candidates(score):
        rates = _binary_rates(y, score, float(threshold))
        if rates["n"] == 0:
            continue
        if rates["false_active_rate"] > false_active_cap:
            continue
        if rates["pred_rate"] > pred_rate_cap:
            continue
        key = (
            rates["recall"],
            rates["precision"],
            -abs(rates["pred_rate"] - min(rates["positive_rate"], pred_rate_cap)),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_rates = rates
    return best_threshold, best_rates


def _binary_eval(
    *,
    model: Any | None,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> dict:
    score = _score_binary(model, x)
    out = _binary_rates(y, score, threshold)
    out["auc"] = _safe_auc(y, score)
    out["ap"] = _safe_ap(y, score)
    out["threshold"] = float(threshold) if math.isfinite(float(threshold)) else str(threshold)
    return out


def _multiclass_eval(model: Any | None, x: np.ndarray, y: np.ndarray) -> dict:
    mask = _finite_rows(x, y)
    y_i = np.asarray(y[mask], dtype=np.int64)
    if model is None or len(y_i) < 20:
        return {"n": int(len(y_i))}
    pred = model.predict(x[mask]).astype(np.int64)
    active_true = y_i != 0
    active_pred = pred != 0
    neutral_true = y_i == 0
    return {
        "n": int(len(y_i)),
        "accuracy": float(accuracy_score(y_i, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_i, pred)),
        "macro_f1": float(f1_score(y_i, pred, average="macro", zero_division=0)),
        "active_recall": float((active_true & active_pred).sum() / max(active_true.sum(), 1)),
        "false_active_rate": float((neutral_true & active_pred).sum() / max(neutral_true.sum(), 1)),
        "pred_active_rate": float(active_pred.mean()),
    }


def _label_distribution(labels: np.ndarray) -> dict:
    labels_i = np.asarray(labels, dtype=np.int64)
    total = max(len(labels_i), 1)
    return {
        ROUTE_NAMES[i]: float((labels_i == i).sum() / total)
        for i in range(len(ROUTE_NAMES))
    }


def _build_feature_sets(
    *,
    raw: np.ndarray,
    enc: dict,
    positions: np.ndarray,
    regime: np.ndarray | None,
    advantage: np.ndarray | None,
) -> dict[str, np.ndarray]:
    z = _as_2d(enc.get("z"))
    h = _as_2d(enc.get("h"))
    pos = _as_2d(np.asarray(positions, dtype=np.float32))
    return {
        "raw": _concat_features(raw),
        "wm": _concat_features(z, h),
        "context": _concat_features(pos, regime, advantage),
        "wm_position": _concat_features(z, h, pos),
        "wm_regime": _concat_features(z, h, regime),
        "wm_advantage": _concat_features(z, h, advantage),
        "wm_position_advantage": _concat_features(z, h, pos, advantage),
        "wm_context": _concat_features(z, h, pos, regime, advantage),
        "raw_wm_context": _concat_features(raw, z, h, pos, regime, advantage),
    }


def _evaluate_feature_set(
    *,
    feature_name: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    max_train_samples: int,
    seed: int,
    false_active_cap: float,
    pred_rate_cap: float,
) -> dict:
    n_train = min(len(train_x), len(train_labels))
    n_val = min(len(val_x), len(val_labels))
    n_test = min(len(test_x), len(test_labels))
    train_x = train_x[:n_train]
    val_x = val_x[:n_val]
    test_x = test_x[:n_test]
    train_labels = np.asarray(train_labels[:n_train], dtype=np.int64)
    val_labels = np.asarray(val_labels[:n_val], dtype=np.int64)
    test_labels = np.asarray(test_labels[:n_test], dtype=np.int64)

    train_active = (train_labels != 0).astype(np.int64)
    val_active = (val_labels != 0).astype(np.int64)
    test_active = (test_labels != 0).astype(np.int64)
    binary_model = _fit_binary_model(
        train_x,
        train_active,
        max_train_samples=max_train_samples,
        seed=seed,
    )
    val_score = _score_binary(binary_model, val_x)
    threshold, threshold_val_rates = _select_threshold(
        val_active,
        val_score,
        false_active_cap=false_active_cap,
        pred_rate_cap=pred_rate_cap,
    )
    multi_model = _fit_multiclass_model(
        train_x,
        train_labels,
        max_train_samples=max_train_samples,
        seed=seed,
    )

    one_vs_rest = {}
    for idx, route in enumerate(ROUTE_NAMES[1:], start=1):
        y_train = (train_labels == idx).astype(np.int64)
        y_val = (val_labels == idx).astype(np.int64)
        y_test = (test_labels == idx).astype(np.int64)
        model = _fit_binary_model(
            train_x,
            y_train,
            max_train_samples=max_train_samples,
            seed=seed + idx,
        )
        one_vs_rest[route] = {
            "val_auc": _safe_auc(y_val, _score_binary(model, val_x)),
            "test_auc": _safe_auc(y_test, _score_binary(model, test_x)),
            "val_ap": _safe_ap(y_val, _score_binary(model, val_x)),
            "test_ap": _safe_ap(y_test, _score_binary(model, test_x)),
        }

    return {
        "feature_set": feature_name,
        "dims": {
            "train": int(train_x.shape[1]),
            "val": int(val_x.shape[1]),
            "test": int(test_x.shape[1]),
        },
        "label_distribution": {
            "train": _label_distribution(train_labels),
            "val": _label_distribution(val_labels),
            "test": _label_distribution(test_labels),
        },
        "active_binary": {
            "selected_threshold": float(threshold) if math.isfinite(float(threshold)) else str(threshold),
            "threshold_selected_on_val": threshold_val_rates,
            "train": _binary_eval(model=binary_model, x=train_x, y=train_active, threshold=threshold),
            "val": _binary_eval(model=binary_model, x=val_x, y=val_active, threshold=threshold),
            "test": _binary_eval(model=binary_model, x=test_x, y=test_active, threshold=threshold),
        },
        "one_vs_rest": one_vs_rest,
        "multiclass": {
            "val": _multiclass_eval(multi_model, val_x, val_labels),
            "test": _multiclass_eval(multi_model, test_x, test_labels),
        },
    }


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


def _aggregate(results: dict[str, dict]) -> dict:
    feature_names = sorted({name for fold in results.values() for name in fold.keys()})
    out = {}
    for name in feature_names:
        rows = [fold[name] for fold in results.values() if name in fold]
        test_active = [row["active_binary"]["test"] for row in rows]
        val_active = [row["active_binary"]["val"] for row in rows]
        out[name] = {
            "folds": int(len(rows)),
            "test_auc_mean": _nanmean([r.get("auc", float("nan")) for r in test_active]),
            "test_auc_worst": _nanmin([r.get("auc", float("nan")) for r in test_active]),
            "test_ap_mean": _nanmean([r.get("ap", float("nan")) for r in test_active]),
            "test_recall_mean": _nanmean([r.get("recall", float("nan")) for r in test_active]),
            "test_recall_worst": _nanmin([r.get("recall", float("nan")) for r in test_active]),
            "test_false_active_mean": _nanmean([r.get("false_active_rate", float("nan")) for r in test_active]),
            "test_false_active_worst": max(
                [r.get("false_active_rate", float("nan")) for r in test_active if math.isfinite(r.get("false_active_rate", float("nan")))]
                or [float("nan")]
            ),
            "test_pred_rate_mean": _nanmean([r.get("pred_rate", float("nan")) for r in test_active]),
            "val_auc_mean": _nanmean([r.get("auc", float("nan")) for r in val_active]),
        }
    return out


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


def _write_md(path: str, payload: dict) -> None:
    lines = [
        "# Route Separability Probe",
        "",
        f"Config: `{payload['config']}`",
        f"Checkpoint dir: `{payload['checkpoint_dir']}`",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        "",
        "## Aggregate Active/No-Active Separability",
        "",
        "| feature set | folds | test AUC mean | test AUC worst | test AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in payload["aggregate"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(row["folds"]),
                    _fmt(row["test_auc_mean"]),
                    _fmt(row["test_auc_worst"]),
                    _fmt(row["test_ap_mean"]),
                    _fmt(row["test_recall_mean"]),
                    _fmt(row["test_recall_worst"]),
                    _fmt(row["test_false_active_mean"]),
                    _fmt(row["test_false_active_worst"]),
                    _fmt(row["test_pred_rate_mean"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Fold Detail",
            "",
            "Thresholds are selected on validation with the configured false-active and predicted-active caps, then applied to test.",
        ]
    )
    for fold, fold_rows in payload["results"].items():
        lines.extend(["", f"### Fold {fold}", ""])
        lines.append("| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, row in fold_rows.items():
            active = row["active_binary"]
            multi = row["multiclass"]["test"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(active["val"].get("auc")),
                        _fmt(active["test"].get("auc")),
                        _fmt(active["test"].get("ap")),
                        _fmt(active["test"].get("recall")),
                        _fmt(active["test"].get("false_active_rate")),
                        _fmt(active["test"].get("pred_rate")),
                        _fmt(multi.get("macro_f1")),
                        _fmt(multi.get("false_active_rate")),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## One-Vs-Rest Test AUC", ""])
    for fold, fold_rows in payload["results"].items():
        lines.extend(["", f"### Fold {fold}", ""])
        lines.append("| feature set | de_risk | recovery | overweight |")
        lines.append("|---|---:|---:|---:|")
        for name, row in fold_rows.items():
            ovr = row["one_vs_rest"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(ovr["de_risk"].get("test_auc")),
                        _fmt(ovr["recovery"].get("test_auc")),
                        _fmt(ovr["overweight"].get("test_auc")),
                    ]
                )
                + " |"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.route_separability_probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--folds", default="0,4,5")
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--false-active-cap", type=float, default=0.15)
    parser.add_argument("--pred-rate-cap", type=float, default=0.25)
    parser.add_argument(
        "--feature-sets",
        default="",
        help="Optional comma-separated subset of feature sets to evaluate.",
    )
    parser.add_argument("--output-json", default="documents/route_separability_probe.json")
    parser.add_argument("--output-md", default="documents/route_separability_probe.md")
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
    cache_dir = "checkpoints/data_cache"
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
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

    all_results: dict[str, dict] = {}
    feature_filter = {
        item.strip()
        for item in str(args.feature_sets or "").split(",")
        if item.strip()
    }
    for split in splits:
        print(f"[RouteSep] fold={split.fold_idx} start")
        wfo_dataset = WFODataset(
            features_df,
            raw_returns,
            split,
            seq_len=cfg.get("data", {}).get("seq_len", 64),
        )
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
            benchmark_position=_benchmark_position_value(cfg),
            forward_window_stats_fn=_forward_window_stats,
            log_ts=_ts,
        )
        train_returns = fold_inputs["train_returns"]
        oracle_bundle = fold_inputs["oracle_bundle"]
        oracle_positions = fold_inputs["oracle_positions"]
        val_oracle_positions = fold_inputs["val_oracle_positions"]
        train_regime_probs = fold_inputs["train_regime_probs"]
        val_regime_probs = fold_inputs["val_regime_probs"]
        test_regime_probs = fold_inputs["test_regime_probs"]

        ensemble, wm_trainer = prepare_world_model_stage(
            fold_idx=split.fold_idx,
            obs_dim=wfo_dataset.obs_dim,
            cfg=cfg,
            device=args.device,
            has_wm=True,
            wm_path=fold_runtime["wm_path"],
            wfo_dataset=wfo_dataset,
            oracle_positions=oracle_positions,
            val_oracle_positions=val_oracle_positions,
            train_returns=train_returns,
            train_regime_probs=train_regime_probs,
            val_regime_probs=val_regime_probs,
            log_ts=_ts,
        )
        seq_len = cfg.get("data", {}).get("seq_len", 64)
        enc_train = wm_trainer.encode_sequence(wfo_dataset.train_features, actions=None, seq_len=seq_len)
        enc_val = wm_trainer.encode_sequence(wfo_dataset.val_features, actions=None, seq_len=seq_len)
        enc_test = wm_trainer.encode_sequence(wfo_dataset.test_features, actions=None, seq_len=seq_len)

        train_advantage_values = fold_inputs.get("train_advantage_values")
        val_advantage_values = fold_inputs.get("val_advantage_values")
        test_advantage_values = fold_inputs.get("test_advantage_values")
        predictive_bundle = build_wm_predictive_state_bundle(
            wm_trainer=wm_trainer,
            wfo_dataset=wfo_dataset,
            z_train=enc_train["z"],
            h_train=enc_train["h"],
            seq_len=seq_len,
            ac_cfg=cfg.get("ac", {}),
            log_ts=_ts,
        )
        if predictive_bundle is not None:
            train_advantage_values = predictive_bundle["train"]
            val_advantage_values = predictive_bundle["val"]
            test_advantage_values = predictive_bundle["test"]

        test_positions, test_route_labels, _test_route_soft, _test_route_adv = _test_route_targets(
            wfo_dataset=wfo_dataset,
            cfg=cfg,
            costs_cfg=costs_cfg,
            bc_cfg=cfg.get("bc", {}),
            ac_cfg=cfg.get("ac", {}),
            reward_cfg=reward_cfg,
            train_returns=train_returns,
            oracle_action_values=oracle_bundle["oracle_action_values"],
        )

        train_labels = np.asarray(fold_inputs["train_route_labels"], dtype=np.int64)
        val_labels = np.asarray(fold_inputs["val_route_labels"], dtype=np.int64)
        test_labels = np.asarray(test_route_labels, dtype=np.int64)

        train_sets = _build_feature_sets(
            raw=wfo_dataset.train_features,
            enc=enc_train,
            positions=oracle_positions,
            regime=train_regime_probs,
            advantage=train_advantage_values,
        )
        val_sets = _build_feature_sets(
            raw=wfo_dataset.val_features,
            enc=enc_val,
            positions=val_oracle_positions,
            regime=val_regime_probs,
            advantage=val_advantage_values,
        )
        test_sets = _build_feature_sets(
            raw=wfo_dataset.test_features,
            enc=enc_test,
            positions=test_positions,
            regime=test_regime_probs,
            advantage=test_advantage_values,
        )

        fold_results = {}
        for name in train_sets.keys():
            if feature_filter and name not in feature_filter:
                continue
            print(f"[RouteSep] fold={split.fold_idx} feature_set={name}")
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
                false_active_cap=args.false_active_cap,
                pred_rate_cap=args.pred_rate_cap,
            )
        all_results[str(split.fold_idx)] = fold_results

    payload = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "folds": [int(split.fold_idx) for split in splits],
        "false_active_cap": float(args.false_active_cap),
        "pred_rate_cap": float(args.pred_rate_cap),
        "max_train_samples": int(args.max_train_samples),
        "results": all_results,
        "aggregate": _aggregate(all_results),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[RouteSep] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
