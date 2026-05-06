from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import _online_drawdown, _rolling_past_sum, _state_features
from unidream.cli.route_separability_probe import (
    _binary_eval,
    _fit_binary_model,
    _score_binary,
    _select_threshold,
)
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


EXPERIMENT_NAME = "plan003_pullback_label_audit"


def _date_prefix() -> str:
    return datetime.now().strftime("%Y%m%d")


def _fmt(value: Any, digits: int = 3) -> str:
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


def _future_sum(returns: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    ret = np.asarray(returns, dtype=np.float64)
    h = max(int(horizon), 1)
    out = np.full(len(ret), np.nan, dtype=np.float64)
    valid = np.zeros(len(ret), dtype=bool)
    if len(ret) <= h:
        return out, valid
    csum = np.concatenate([[0.0], np.cumsum(ret)])
    n = len(ret) - h
    idx = np.arange(n)
    out[:n] = csum[idx + 1 + h] - csum[idx + 1]
    valid[:n] = True
    return out, valid


def _pullback_context(
    returns: np.ndarray,
    *,
    min_past_dd: float,
) -> np.ndarray:
    past_ret32 = _rolling_past_sum(returns, 32)
    past_ret64 = _rolling_past_sum(returns, 64)
    dd, _underwater = _online_drawdown(returns)
    return (past_ret32 < -0.005) & (past_ret64 > 0.005) & (dd < -float(min_past_dd))


def _labels(
    returns: np.ndarray,
    *,
    horizon: int,
    min_past_dd: float,
    min_recovery: float,
) -> dict[str, np.ndarray]:
    future, valid = _future_sum(returns, horizon)
    context = _pullback_context(returns, min_past_dd=min_past_dd)
    future_recovery = (future > float(min_recovery)) & valid
    all_bar = context & future_recovery
    return {
        "valid": valid,
        "context": context & valid,
        "future_recovery": future_recovery.astype(np.int64),
        "all_bar": all_bar.astype(np.int64),
    }


def _boundary_mask(
    n: int,
    *,
    role: str,
    horizon: int,
    purge_bars: int,
    embargo_bars: int,
    lookback_bars: int,
) -> np.ndarray:
    mask = np.ones(int(n), dtype=bool)
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


def _perturb_y(y_raw: np.ndarray, valid_mask: np.ndarray, *, mode: str, seed: int, shift_bars: int) -> np.ndarray:
    y = np.asarray(y_raw, dtype=np.int64).copy()
    idx = np.flatnonzero(valid_mask)
    if mode == "normal":
        return y
    if mode == "shuffle_all":
        rng = np.random.default_rng(int(seed))
        y[idx] = rng.permutation(y[idx])
        return y
    if mode == "time_shift":
        shift = max(int(shift_bars), 1)
        src = idx + shift
        ok = src < len(y)
        y[idx[ok]] = y[src[ok]]
        if np.any(~ok):
            y[idx[~ok]] = y[idx[~ok][0]]
        return y
    raise ValueError(f"unknown audit mode: {mode}")


def _finite_mask(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = min(len(x), len(y), len(valid))
    return (
        np.asarray(valid[:n], dtype=bool)
        & np.isfinite(np.asarray(y[:n], dtype=np.float64))
        & np.all(np.isfinite(np.asarray(x[:n], dtype=np.float64)), axis=1)
    )


def _evaluate_task(
    *,
    task: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    train_labels: dict[str, np.ndarray],
    val_labels: dict[str, np.ndarray],
    test_labels: dict[str, np.ndarray],
    boundary: dict[str, np.ndarray],
    audit_mode: str,
    seed: int,
    shift_bars: int,
    max_train_samples: int,
    false_active_cap: float,
    pred_rate_cap: float,
) -> dict[str, Any]:
    if task == "all_bar":
        train_y_raw = train_labels["all_bar"]
        val_y_raw = val_labels["all_bar"]
        test_y = test_labels["all_bar"]
        train_base = train_labels["valid"]
        val_base = val_labels["valid"]
        test_base = test_labels["valid"]
    elif task == "conditional_context":
        train_y_raw = train_labels["future_recovery"]
        val_y_raw = val_labels["future_recovery"]
        test_y = test_labels["future_recovery"]
        train_base = train_labels["context"]
        val_base = val_labels["context"]
        test_base = test_labels["context"]
    else:
        raise ValueError(f"unknown task: {task}")

    train_mask = _finite_mask(x_train, train_y_raw, train_base) & boundary["train"][: len(x_train)]
    val_mask = _finite_mask(x_val, val_y_raw, val_base) & boundary["val"][: len(x_val)]
    test_mask = _finite_mask(x_test, test_y, test_base) & boundary["test"][: len(x_test)]
    train_y = _perturb_y(train_y_raw, train_mask, mode=audit_mode, seed=seed, shift_bars=shift_bars)
    val_y = _perturb_y(val_y_raw, val_mask, mode=audit_mode, seed=seed + 1009, shift_bars=shift_bars)

    out: dict[str, Any] = {
        "task": task,
        "audit_mode": audit_mode,
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "test_count": int(test_mask.sum()),
        "train_positive_rate": float(np.mean(train_y[train_mask])) if int(train_mask.sum()) else float("nan"),
        "val_positive_rate": float(np.mean(val_y[val_mask])) if int(val_mask.sum()) else float("nan"),
        "test_positive_rate": float(np.mean(test_y[test_mask])) if int(test_mask.sum()) else float("nan"),
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
    val_score = _score_binary(model, x_val[val_mask])
    threshold, val_rates = _select_threshold(
        val_y[val_mask],
        val_score,
        false_active_cap=false_active_cap,
        pred_rate_cap=pred_rate_cap,
    )
    out["status"] = "ok"
    out["threshold_selected_on_val"] = val_rates
    out["val"] = _binary_eval(model=model, x=x_val[val_mask], y=val_y[val_mask], threshold=threshold)
    out["test"] = _binary_eval(model=model, x=x_test[test_mask], y=test_y[test_mask], threshold=threshold)
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in sorted({str(r["task"]) for r in rows}):
        out[task] = {}
        for mode in sorted({str(r["audit_mode"]) for r in rows if str(r["task"]) == task}):
            group = [r for r in rows if r.get("status") == "ok" and r["task"] == task and r["audit_mode"] == mode]
            tests = [r["test"] for r in group]
            aucs = [float(t.get("auc", float("nan"))) for t in tests]
            aps = [float(t.get("ap", float("nan"))) for t in tests]
            recalls = [float(t.get("recall", float("nan"))) for t in tests]
            false_active = [float(t.get("false_active_rate", float("nan"))) for t in tests]
            out[task][mode] = {
                "folds_ok": len(group),
                "auc_mean": float(np.nanmean(aucs)) if aucs else float("nan"),
                "auc_worst": float(np.nanmin(aucs)) if aucs else float("nan"),
                "ap_mean": float(np.nanmean(aps)) if aps else float("nan"),
                "recall_mean": float(np.nanmean(recalls)) if recalls else float("nan"),
                "false_active_worst": float(np.nanmax(false_active)) if false_active else float("nan"),
                "test_positive_rate_mean": float(np.nanmean([r["test_positive_rate"] for r in group])) if group else float("nan"),
            }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan003 Pullback Label Audit",
        "",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Audit modes: `{', '.join(payload['audit_modes'])}`",
        f"Horizon: `{payload['horizon']}`",
        f"Purge/embargo/lookback bars: `{payload['purge_bars']}/{payload['embargo_bars']}/{payload['lookback_bars']}`",
        "",
        "## Aggregate",
        "",
        "| task | mode | folds ok | AUC mean | AUC worst | AP mean | recall mean | false-active worst | positive rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, modes in payload["aggregate"].items():
        for mode, row in modes.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        task,
                        mode,
                        str(row["folds_ok"]),
                        _fmt(row["auc_mean"]),
                        _fmt(row["auc_worst"]),
                        _fmt(row["ap_mean"]),
                        _fmt(row["recall_mean"]),
                        _fmt(row["false_active_worst"]),
                        _fmt(row["test_positive_rate_mean"]),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Fold Detail", ""])
    lines.append("| fold | task | mode | status | train n | val n | test n | test pos | AUC | AP | recall | false active |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in payload["rows"]:
        test = r.get("test", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["fold"]),
                    r["task"],
                    r["audit_mode"],
                    r.get("status", ""),
                    str(r.get("train_count", 0)),
                    str(r.get("val_count", 0)),
                    str(r.get("test_count", 0)),
                    _fmt(r.get("test_positive_rate")),
                    _fmt(test.get("auc")),
                    _fmt(test.get("ap")),
                    _fmt(test.get("recall")),
                    _fmt(test.get("false_active_rate")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- `all_bar` can be inflated because the label includes the past pullback condition.",
            "- `conditional_context` is the real test: only bars already in pullback context are evaluated.",
            "- If `conditional_context` collapses or null modes remain similar, do not use this as an activation signal.",
        ]
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan003_pullback_label_audit")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--min-past-dd", type=float, default=0.01)
    parser.add_argument("--min-recovery", type=float, default=0.003)
    parser.add_argument("--audit-modes", default="normal,shuffle_all,time_shift")
    parser.add_argument("--shift-bars", type=int, default=128)
    parser.add_argument("--purge-bars", type=int, default=32)
    parser.add_argument("--embargo-bars", type=int, default=128)
    parser.add_argument("--lookback-bars", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--false-active-cap", type=float, default=0.15)
    parser.add_argument("--pred-rate-cap", type=float, default=0.25)
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
    audit_modes = [x.strip() for x in str(args.audit_modes).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for split in splits:
        fid = int(split.fold_idx)
        print(f"[PullbackAudit] fold={fid} start")
        ds = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
        x_train = _state_features(ds.train_features, ds.train_returns)
        x_val = _state_features(ds.val_features, ds.val_returns)
        x_test = _state_features(ds.test_features, ds.test_returns)
        labels = {
            "train": _labels(ds.train_returns, horizon=args.horizon, min_past_dd=args.min_past_dd, min_recovery=args.min_recovery),
            "val": _labels(ds.val_returns, horizon=args.horizon, min_past_dd=args.min_past_dd, min_recovery=args.min_recovery),
            "test": _labels(ds.test_returns, horizon=args.horizon, min_past_dd=args.min_past_dd, min_recovery=args.min_recovery),
        }
        boundary = {
            "train": _boundary_mask(
                len(ds.train_returns),
                role="train",
                horizon=args.horizon,
                purge_bars=args.purge_bars,
                embargo_bars=args.embargo_bars,
                lookback_bars=args.lookback_bars,
            ),
            "val": _boundary_mask(
                len(ds.val_returns),
                role="val",
                horizon=args.horizon,
                purge_bars=args.purge_bars,
                embargo_bars=args.embargo_bars,
                lookback_bars=args.lookback_bars,
            ),
            "test": _boundary_mask(
                len(ds.test_returns),
                role="test",
                horizon=args.horizon,
                purge_bars=args.purge_bars,
                embargo_bars=args.embargo_bars,
                lookback_bars=args.lookback_bars,
            ),
        }
        for audit_mode in audit_modes:
            for task in ("all_bar", "conditional_context"):
                row = _evaluate_task(
                    task=task,
                    x_train=x_train,
                    x_val=x_val,
                    x_test=x_test,
                    train_labels=labels["train"],
                    val_labels=labels["val"],
                    test_labels=labels["test"],
                    boundary=boundary,
                    audit_mode=audit_mode,
                    seed=args.seed + fid * 101 + len(task) + len(audit_mode),
                    shift_bars=args.shift_bars,
                    max_train_samples=args.max_train_samples,
                    false_active_cap=args.false_active_cap,
                    pred_rate_cap=args.pred_rate_cap,
                )
                row["fold"] = fid
                rows.append(row)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "config": args.config,
        "folds": [int(s.fold_idx) for s in splits],
        "seed": int(args.seed),
        "horizon": int(args.horizon),
        "min_past_dd": float(args.min_past_dd),
        "min_recovery": float(args.min_recovery),
        "audit_modes": audit_modes,
        "purge_bars": int(args.purge_bars),
        "embargo_bars": int(args.embargo_bars),
        "lookback_bars": int(args.lookback_bars),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[PullbackAudit] wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
