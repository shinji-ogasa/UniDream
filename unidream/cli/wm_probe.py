from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unidream.data.dataset import WFODataset
from unidream.data.oracle import (
    feature_dual_teacher,
    feature_stress_teacher,
    hindsight_oracle_dp,
    hindsight_signal_teacher,
    smooth_aim_positions,
)
from unidream.device import add_device_argument, resolve_device
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.train_app import resolve_cache_dir
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble


@dataclass
class FoldProbeData:
    fold: int
    train_raw: np.ndarray
    val_raw: np.ndarray
    test_raw: np.ndarray
    train_latent: np.ndarray
    val_latent: np.ndarray
    test_latent: np.ndarray
    train_returns: np.ndarray
    val_returns: np.ndarray
    test_returns: np.ndarray
    train_teacher: np.ndarray
    val_teacher: np.ndarray
    test_teacher: np.ndarray


def _cost_rate(cfg: dict) -> float:
    costs = cfg.get("costs", {})
    return (
        (float(costs.get("spread_bps", 5.0)) / 10000.0) / 2.0
        + float(costs.get("fee_rate", 0.0004))
        + float(costs.get("slippage_bps", 2.0)) / 10000.0
    )


def _future_sum(returns: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    out = np.full(len(returns), np.nan, dtype=np.float64)
    if len(returns) <= horizon:
        return out
    csum = np.concatenate([[0.0], np.cumsum(returns, dtype=np.float64)])
    idx = np.arange(0, len(returns) - horizon)
    out[idx] = csum[idx + 1 + horizon] - csum[idx + 1]
    return out


def _future_vol(returns: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    out = np.full(len(returns), np.nan, dtype=np.float64)
    if len(returns) <= horizon:
        return out
    csum = np.concatenate([[0.0], np.cumsum(returns, dtype=np.float64)])
    cs2 = np.concatenate([[0.0], np.cumsum(np.square(returns), dtype=np.float64)])
    idx = np.arange(0, len(returns) - horizon)
    s = csum[idx + 1 + horizon] - csum[idx + 1]
    s2 = cs2[idx + 1 + horizon] - cs2[idx + 1]
    mean = s / horizon
    var = np.maximum(s2 / horizon - np.square(mean), 0.0)
    out[idx] = np.sqrt(var)
    return out


def _future_drawdown_risk(returns: np.ndarray, horizon: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    out = np.full(len(returns), np.nan, dtype=np.float64)
    for t in range(0, max(0, len(returns) - horizon)):
        path = np.cumsum(returns[t + 1 : t + 1 + horizon], dtype=np.float64)
        path_with_origin = np.concatenate([[0.0], path])
        running_max = np.maximum.accumulate(path_with_origin)
        drawdown = path_with_origin - running_max
        out[t] = -float(drawdown.min())
    return out


def _rank_ic(pred: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 20 or np.nanstd(pred[mask]) < 1e-12 or np.nanstd(y[mask]) < 1e-12:
        return float("nan")
    corr = spearmanr(pred[mask], y[mask]).correlation
    return float(corr) if corr is not None else float("nan")


def _direction_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(y) & (np.abs(y) > 1e-12)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(pred[mask]) == np.sign(y[mask])).mean())


def _decile_spread(pred: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(y)
    if mask.sum() < 40:
        return float("nan")
    pred_m = pred[mask]
    y_m = y[mask]
    lo = np.quantile(pred_m, 0.10)
    hi = np.quantile(pred_m, 0.90)
    low_y = y_m[pred_m <= lo]
    high_y = y_m[pred_m >= hi]
    if len(low_y) == 0 or len(high_y) == 0:
        return float("nan")
    return float(high_y.mean() - low_y.mean())


def _safe_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() < 20:
        return float("nan")
    labels_i = labels[mask].astype(int)
    if len(np.unique(labels_i)) < 2:
        return float("nan")
    return float(roc_auc_score(labels_i, scores[mask]))


def _sample_indices(n: int, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_samples, replace=False))


def _fit_ridge(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    idx = np.where(mask)[0]
    idx = idx[_sample_indices(len(idx), max_samples, seed)]
    if len(idx) < 50:
        return None
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(X[idx], y[idx])
    return model


def _fit_logistic(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int):
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    idx = np.where(mask)[0]
    if len(np.unique(y[idx])) < 2:
        return None
    idx = idx[_sample_indices(len(idx), max_samples, seed)]
    if len(idx) < 100 or len(np.unique(y[idx])) < 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X[idx], y[idx])
    return model


def _regression_eval(model, X: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if model is None or mask.sum() < 20:
        return {"n": int(mask.sum())}
    pred = model.predict(X[mask])
    y_m = y[mask]
    return {
        "n": int(mask.sum()),
        "r2": float(r2_score(y_m, pred)),
        "rank_ic": _rank_ic(pred, y_m),
        "direction_acc": _direction_accuracy(pred, y_m),
        "decile_spread": _decile_spread(pred, y_m),
        "target_mean": float(np.mean(y_m)),
        "pred_mean": float(np.mean(pred)),
    }


def _risk_eval(model, X: np.ndarray, y: np.ndarray, event_quantile: float = 0.80) -> dict:
    base = _regression_eval(model, X, y)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if model is None or mask.sum() < 20:
        return base
    pred = model.predict(X[mask])
    y_m = y[mask]
    event = y_m >= np.quantile(y_m, event_quantile)
    base["event_auc"] = _safe_auc(pred, event.astype(int))
    base["top_decile_realized"] = float(y_m[pred >= np.quantile(pred, 0.90)].mean())
    base["bottom_decile_realized"] = float(y_m[pred <= np.quantile(pred, 0.10)].mean())
    return base


def _classification_eval(model, X: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if model is None or mask.sum() < 20:
        return {"n": int(mask.sum())}
    pred = model.predict(X[mask])
    y_m = y[mask].astype(int)
    out = {
        "n": int(mask.sum()),
        "accuracy": float(accuracy_score(y_m, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_m, pred)),
        "classes": sorted(int(c) for c in np.unique(y_m)),
    }
    if len(np.unique(y_m)) == 2 and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X[mask])
        out["auc"] = _safe_auc(proba[:, 1], y_m)
    return out


def _teacher_positions(
    *,
    returns: np.ndarray,
    features: np.ndarray,
    feature_columns: list[str],
    cfg: dict,
) -> np.ndarray:
    oracle_cfg = cfg.get("oracle", {})
    ac_cfg = cfg.get("ac", {})
    reward_cfg = cfg.get("reward", {})
    costs_cfg = cfg.get("costs", {})
    benchmark = float(reward_cfg.get("benchmark_position", 1.0))
    action_values = np.asarray(
        oracle_cfg.get("action_values", cfg.get("actions", {}).get("values", [0.0, 0.5, 1.0])),
        dtype=np.float32,
    )
    abs_min = float(ac_cfg.get("abs_min_position", float(np.min(action_values))))
    abs_max = float(ac_cfg.get("abs_max_position", float(np.max(action_values))))
    mode = str(oracle_cfg.get("teacher_mode", "dp")).lower()

    if mode == "signal_aim":
        positions, _signal = hindsight_signal_teacher(
            returns,
            benchmark_position=benchmark,
            min_position=oracle_cfg.get("signal_floor_position", abs_min),
            max_position=oracle_cfg.get("signal_ceiling_position", abs_max),
            horizons=tuple(oracle_cfg.get("signal_horizons", [4, 16, 64])),
            horizon_weights=tuple(oracle_cfg.get("signal_horizon_weights", [0.2, 0.3, 0.5])),
            signal_scale=oracle_cfg.get("signal_scale", 1.5),
            signal_deadzone=oracle_cfg.get("signal_deadzone", 0.1),
            signal_clip=oracle_cfg.get("signal_clip", 4.0),
            downside_horizon=oracle_cfg.get("signal_downside_horizon", 16),
            downside_weight=oracle_cfg.get("signal_downside_weight", 0.0),
        )
    elif mode == "feature_stress":
        positions, _signal = feature_stress_teacher(
            features,
            feature_columns=feature_columns,
            benchmark_position=benchmark,
            min_position=oracle_cfg.get("stress_floor_position", abs_min),
            max_position=oracle_cfg.get("stress_ceiling_position", abs_max),
        )
    elif mode == "feature_dual":
        positions, _signal = feature_dual_teacher(
            features,
            feature_columns=feature_columns,
            benchmark_position=benchmark,
            min_position=oracle_cfg.get("dual_floor_position", abs_min),
            max_position=oracle_cfg.get("dual_ceiling_position", abs_max),
        )
    else:
        actions, _values, _soft = hindsight_oracle_dp(
            returns,
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            fee_rate=costs_cfg.get("fee_rate", 0.0004),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            discount=oracle_cfg.get("discount", 1.0),
            min_hold=oracle_cfg.get("min_hold", 0),
            soft_label_temp=0.0,
            reward_mode=reward_cfg.get("mode", "absolute"),
            benchmark_position=benchmark,
            action_values=action_values,
        )
        positions = action_values[actions]

    if oracle_cfg.get("use_aim_targets", False):
        positions = smooth_aim_positions(
            positions,
            max_step=oracle_cfg.get("aim_max_step", 0.25),
            band=oracle_cfg.get("aim_band", 0.0),
            initial_position=benchmark if reward_cfg.get("mode") == "excess_bh" else 0.0,
            min_position=abs_min,
            max_position=abs_max,
            benchmark_position=benchmark if reward_cfg.get("mode") == "excess_bh" else 0.0,
            underweight_confirm_bars=oracle_cfg.get("aim_underweight_confirm_bars", 0),
            underweight_min_scale=oracle_cfg.get("aim_underweight_min_scale", 0.0),
            underweight_floor_position=oracle_cfg.get("aim_underweight_floor_position"),
            underweight_step_scale=oracle_cfg.get("aim_underweight_step_scale", 1.0),
        )
    return np.asarray(positions, dtype=np.float32)


def _teacher_class(positions: np.ndarray, benchmark: float) -> np.ndarray:
    overlay = positions - float(benchmark)
    labels = np.full(len(positions), 1, dtype=np.int64)
    labels[overlay < -0.05] = 0
    labels[overlay > 0.05] = 2
    return labels


def _recovery_label(positions: np.ndarray, benchmark: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float64)
    under = positions < benchmark - 0.05
    labels = np.full(len(positions), np.nan, dtype=np.float64)
    for t in range(0, max(0, len(positions) - horizon)):
        if under[t]:
            labels[t] = float(np.any(positions[t + 1 : t + 1 + horizon] >= benchmark - 0.05))
    return labels, under


def _action_advantages(returns: np.ndarray, actions: np.ndarray, benchmark: float, cost_rate: float, horizon: int) -> np.ndarray:
    fwd = _future_sum(returns, horizon)
    adv = np.full((len(returns), len(actions)), np.nan, dtype=np.float64)
    for i, pos in enumerate(actions):
        adv[:, i] = (float(pos) - benchmark) * fwd - cost_rate * abs(float(pos) - benchmark)
    return adv


def _action_advantage_eval(model, X: np.ndarray, y_adv: np.ndarray, benchmark_idx: int) -> dict:
    mask = np.all(np.isfinite(y_adv), axis=1) & np.all(np.isfinite(X), axis=1)
    if model is None or mask.sum() < 20:
        return {"n": int(mask.sum())}
    pred = model.predict(X[mask])
    actual = y_adv[mask]
    best_actual = np.argmax(actual, axis=1)
    best_pred = np.argmax(pred, axis=1)
    top2_pred = np.argsort(pred, axis=1)[:, -2:]
    top1 = float(np.mean(best_pred == best_actual))
    top2 = float(np.mean([best_actual[i] in top2_pred[i] for i in range(len(best_actual))]))
    chosen_adv = actual[np.arange(len(actual)), best_pred]
    bench_adv = actual[:, benchmark_idx]
    return {
        "n": int(mask.sum()),
        "top1_accuracy": top1,
        "top2_accuracy": top2,
        "chosen_minus_benchmark_adv": float(np.mean(chosen_adv - bench_adv)),
        "oracle_best_minus_benchmark_adv": float(np.mean(np.max(actual, axis=1) - bench_adv)),
        "pred_long_rate": float(np.mean(best_pred > benchmark_idx)),
        "pred_underweight_rate": float(np.mean(best_pred < benchmark_idx)),
    }


def _train_multioutput_ridge(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int):
    mask = np.all(np.isfinite(y), axis=1) & np.all(np.isfinite(X), axis=1)
    idx = np.where(mask)[0]
    idx = idx[_sample_indices(len(idx), max_samples, seed)]
    if len(idx) < 50:
        return None
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(X[idx], y[idx])
    return model


def _build_labels(returns: np.ndarray, teacher: np.ndarray, cfg: dict, horizons: list[int]) -> dict:
    benchmark = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    actions = np.asarray(cfg.get("oracle", {}).get("action_values", cfg.get("actions", {}).get("values", [0.0, 0.5, 1.0])), dtype=np.float64)
    benchmark_idx = int(np.argmin(np.abs(actions - benchmark)))
    labels: dict[str, np.ndarray] = {}
    for h in horizons:
        labels[f"return_h{h}"] = _future_sum(returns, h)
        labels[f"vol_h{h}"] = _future_vol(returns, h)
        labels[f"drawdown_risk_h{h}"] = _future_drawdown_risk(returns, h)
    teacher_cls = _teacher_class(teacher, benchmark)
    labels["teacher_underweight"] = (teacher_cls == 0).astype(np.float64)
    labels["teacher_class"] = teacher_cls.astype(np.float64)
    rec, _under = _recovery_label(teacher, benchmark, horizon=16)
    labels["recovery_h16"] = rec
    labels["action_adv_h16"] = _action_advantages(returns, actions, benchmark, _cost_rate(cfg), horizon=16)
    labels["action_values"] = actions
    labels["benchmark_action_index"] = benchmark_idx
    return labels


def _evaluate_feature_set(
    *,
    name: str,
    X_train: np.ndarray,
    X_eval: np.ndarray,
    train_labels: dict,
    eval_labels: dict,
    horizons: list[int],
    max_train_samples: int | None,
    seed: int,
) -> dict:
    out: dict[str, dict] = {}
    for h in horizons:
        model = _fit_ridge(X_train, train_labels[f"return_h{h}"], max_train_samples, seed)
        out[f"return_h{h}"] = _regression_eval(model, X_eval, eval_labels[f"return_h{h}"])
        vol_model = _fit_ridge(X_train, train_labels[f"vol_h{h}"], max_train_samples, seed)
        out[f"vol_h{h}"] = _risk_eval(vol_model, X_eval, eval_labels[f"vol_h{h}"])
        dd_model = _fit_ridge(X_train, train_labels[f"drawdown_risk_h{h}"], max_train_samples, seed)
        out[f"drawdown_risk_h{h}"] = _risk_eval(dd_model, X_eval, eval_labels[f"drawdown_risk_h{h}"])

    under_model = _fit_logistic(
        X_train,
        train_labels["teacher_underweight"].astype(int),
        max_train_samples,
        seed,
    )
    out["teacher_underweight"] = _classification_eval(
        under_model,
        X_eval,
        eval_labels["teacher_underweight"].astype(int),
    )

    class_model = _fit_logistic(
        X_train,
        train_labels["teacher_class"].astype(int),
        max_train_samples,
        seed,
    )
    out["teacher_class"] = _classification_eval(
        class_model,
        X_eval,
        eval_labels["teacher_class"].astype(int),
    )

    rec_mask_train = np.isfinite(train_labels["recovery_h16"])
    if rec_mask_train.sum() >= 100 and len(np.unique(train_labels["recovery_h16"][rec_mask_train])) >= 2:
        rec_model = _fit_logistic(
            X_train,
            train_labels["recovery_h16"].astype(float),
            max_train_samples,
            seed,
        )
    else:
        rec_model = None
    rec_eval = eval_labels["recovery_h16"]
    out["recovery_h16"] = _classification_eval(rec_model, X_eval, rec_eval)

    adv_model = _train_multioutput_ridge(X_train, train_labels["action_adv_h16"], max_train_samples, seed)
    out["action_advantage_h16"] = _action_advantage_eval(
        adv_model,
        X_eval,
        eval_labels["action_adv_h16"],
        int(train_labels["benchmark_action_index"]),
    )
    return {name: out}


def _load_fold_data(args, cfg: dict, split) -> FoldProbeData:
    data_cfg = cfg.get("data", {})
    seq_len = int(data_cfg.get("seq_len", 64))
    symbol = args.symbol or data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
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
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
    wm_path = os.path.join(args.checkpoint_dir, f"fold_{split.fold_idx}", "world_model.pt")
    if not os.path.exists(wm_path):
        raise FileNotFoundError(f"world model checkpoint not found: {wm_path}")
    ensemble = build_ensemble(wfo_dataset.obs_dim, cfg)
    trainer = WorldModelTrainer(ensemble, cfg, device=args.device)
    trainer.load(wm_path)

    def enc(feat: np.ndarray) -> np.ndarray:
        encoded = trainer.encode_sequence(feat, seq_len=seq_len)
        return np.concatenate([encoded["z"], encoded["h"]], axis=1).astype(np.float32)

    feature_columns = wfo_dataset.feature_columns
    return FoldProbeData(
        fold=split.fold_idx,
        train_raw=wfo_dataset.train_features.astype(np.float32),
        val_raw=wfo_dataset.val_features.astype(np.float32),
        test_raw=wfo_dataset.test_features.astype(np.float32),
        train_latent=enc(wfo_dataset.train_features),
        val_latent=enc(wfo_dataset.val_features),
        test_latent=enc(wfo_dataset.test_features),
        train_returns=wfo_dataset.train_returns,
        val_returns=wfo_dataset.val_returns,
        test_returns=wfo_dataset.test_returns,
        train_teacher=_teacher_positions(
            returns=wfo_dataset.train_returns,
            features=wfo_dataset.train_features,
            feature_columns=feature_columns,
            cfg=cfg,
        ),
        val_teacher=_teacher_positions(
            returns=wfo_dataset.val_returns,
            features=wfo_dataset.val_features,
            feature_columns=feature_columns,
            cfg=cfg,
        ),
        test_teacher=_teacher_positions(
            returns=wfo_dataset.test_returns,
            features=wfo_dataset.test_features,
            feature_columns=feature_columns,
            cfg=cfg,
        ),
    )


def _markdown_report(results: dict, sources: list[str]) -> str:
    lines = [
        "# TransformerWM Probe Report",
        "",
        f"Config: `{results['config']}`",
        f"Checkpoint dir: `{results['checkpoint_dir']}`",
        f"Folds: `{', '.join(map(str, results['folds']))}`",
        "",
        "## 判定基準",
        "",
        "- return は `rank_ic`, `direction_acc`, `decile_spread` を重視する。",
        "- risk/vol は `event_auc` と decile bucket の単調性を重視する。",
        "- teacher/action は AUC / balanced accuracy / action advantage top-k を見る。",
        "",
        "## Results",
    ]
    for fold, fold_res in results["results"].items():
        lines.extend(["", f"### Fold {fold}"])
        for split_name, split_res in fold_res.items():
            lines.extend(["", f"#### {split_name}"])
            for feature_set, metrics in split_res.items():
                lines.extend(["", f"##### {feature_set}"])
                rows = []
                for target, vals in metrics.items():
                    compact = []
                    for key in [
                        "rank_ic",
                        "direction_acc",
                        "decile_spread",
                        "event_auc",
                        "accuracy",
                        "balanced_accuracy",
                        "auc",
                        "top1_accuracy",
                        "top2_accuracy",
                        "chosen_minus_benchmark_adv",
                        "pred_long_rate",
                        "pred_underweight_rate",
                    ]:
                        if key in vals and vals[key] == vals[key]:
                            compact.append(f"{key}={vals[key]:+.4f}")
                    rows.append(f"- `{target}`: " + (", ".join(compact) if compact else f"n={vals.get('n', 0)}"))
                lines.extend(rows)
    if sources:
        lines.extend(["", "## 参照", ""])
        lines.extend(f"- {src}" for src in sources)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.wm_probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--folds", default="4")
    parser.add_argument("--cost-profile", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", default="1,4,8,16,32")
    parser.add_argument("--max-train-samples", type=int, default=30000)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    add_device_argument(parser)
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg, args.cost_profile)
    if args.checkpoint_dir is None:
        args.checkpoint_dir = cfg.get("logging", {}).get("checkpoint_dir", "checkpoints")
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    data_cfg = cfg.get("data", {})
    symbol = args.symbol or data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
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
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    selected_folds = selected or [s.fold_idx for s in splits]

    all_results: dict[str, dict] = {}
    for split in splits:
        fold_data = _load_fold_data(args, cfg, split)
        train_labels = _build_labels(fold_data.train_returns, fold_data.train_teacher, cfg, horizons)
        eval_sets = {
            "val": (
                fold_data.val_raw,
                fold_data.val_latent,
                _build_labels(fold_data.val_returns, fold_data.val_teacher, cfg, horizons),
            ),
            "test": (
                fold_data.test_raw,
                fold_data.test_latent,
                _build_labels(fold_data.test_returns, fold_data.test_teacher, cfg, horizons),
            ),
        }
        fold_results: dict[str, dict] = {}
        train_feature_sets = {
            "raw": fold_data.train_raw,
            "latent_zh": fold_data.train_latent,
            "raw_plus_latent": np.concatenate([fold_data.train_raw, fold_data.train_latent], axis=1),
        }
        for split_name, (raw_eval, latent_eval, eval_labels) in eval_sets.items():
            eval_feature_sets = {
                "raw": raw_eval,
                "latent_zh": latent_eval,
                "raw_plus_latent": np.concatenate([raw_eval, latent_eval], axis=1),
            }
            split_results: dict[str, dict] = {}
            for feature_name, X_train in train_feature_sets.items():
                split_results.update(
                    _evaluate_feature_set(
                        name=feature_name,
                        X_train=X_train,
                        X_eval=eval_feature_sets[feature_name],
                        train_labels=train_labels,
                        eval_labels=eval_labels,
                        horizons=horizons,
                        max_train_samples=args.max_train_samples,
                        seed=args.seed,
                    )
                )
            fold_results[split_name] = split_results
        all_results[str(split.fold_idx)] = fold_results

    result_payload = {
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "folds": selected_folds,
        "horizons": horizons,
        "max_train_samples": args.max_train_samples,
        "results": all_results,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        md = _markdown_report(result_payload, sources=[])
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md)
    print(json.dumps(result_payload, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
