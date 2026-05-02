"""
Plan 5 Lane F: Ridge expansion — false-de-risk / pullback recovery labels.
Evaluates separability (AUC) of guard-improvement labels on f456.
No WM needed. No config changes.
"""
from __future__ import annotations

import argparse, json, math, os
import numpy as np

from unidream.cli.exploration_board_probe import (
    _state_features, _pullback_no_fire_mask, _rolling_past_sum, _rolling_past_vol,
)
from unidream.cli.route_separability_probe import _fit_binary_model, _binary_eval, _score_binary, _select_threshold
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


HORIZON = 32


def _online_drawdown(returns):
    eq = np.exp(np.cumsum(np.asarray(returns, dtype=np.float64)))
    peak = np.maximum.accumulate(np.maximum(eq, 1.0))
    return eq / np.maximum(peak, 1e-12) - 1.0


def _future_ret_dd(returns, horizon):
    r = np.asarray(returns, dtype=np.float64); h = max(1, int(horizon)); n = len(r)
    ret_sum = np.full(n, np.nan); dd = np.full(n, np.nan)
    for i in range(n - h):
        w = r[i + 1: i + 1 + h]
        ret_sum[i] = np.sum(w); dd[i] = max(0.0, -np.min(np.cumsum(w)))
    return ret_sum, dd


def make_false_derisk_label(returns, horizon=32, min_dd=0.005, min_recovery=0.002):
    """Label=1: de-risk would miss a recovery (false alarm)."""
    ret_sum, future_dd = _future_ret_dd(returns, horizon)
    dd = _online_drawdown(returns)
    n = len(returns)
    label = np.zeros(n, dtype=np.int64)
    for i in range(n - horizon):
        if dd[i] < -min_dd and ret_sum[i] > min_recovery and future_dd[i] < min_dd * 1.5:
            label[i] = 1
    return label


def make_pullback_recovery_label(returns, horizon=32, min_past_dd=0.01, min_recovery=0.003):
    """Label=1: recent pullback, de-risk would miss the rebound."""
    ret_sum, _ = _future_ret_dd(returns, horizon)
    past_ret32 = _rolling_past_sum(returns, 32)
    past_ret64 = _rolling_past_sum(returns, 64)
    dd = _online_drawdown(returns)
    n = len(returns)
    label = np.zeros(n, dtype=np.int64)
    for i in range(n - horizon):
        is_pullback = (past_ret32[i] < -0.005) & (past_ret64[i] > 0.005) & (dd[i] < -min_past_dd)
        if is_pullback and ret_sum[i] > min_recovery:
            label[i] = 1
    return label


def _fmt(v, d=3):
    try: x = float(v)
    except: return "NA"
    return f"{x:.{d}f}" if math.isfinite(x) else "NA"


def eval_label(name, label, x_train, x_val, x_test, seed):
    """Evaluate label separability across folds."""
    valid_tr = np.isfinite(label["train"]) & np.all(np.isfinite(x_train), axis=1)
    valid_val = np.isfinite(label["val"]) & np.all(np.isfinite(x_val), axis=1)
    valid_tst = np.isfinite(label["test"]) & np.all(np.isfinite(x_test), axis=1)

    if label["train"][valid_tr].sum() < 20:
        return None

    model = _fit_binary_model(x_train[valid_tr], label["train"][valid_tr].astype(np.int64),
                              max_train_samples=50000, seed=seed)
    score_val = _score_binary(model, x_val[valid_val])
    thr, _ = _select_threshold(label["val"][valid_val].astype(np.int64), score_val,
                               false_active_cap=0.15, pred_rate_cap=0.25)
    return {
        "name": name,
        "density": float(np.mean(label["test"][valid_tst])),
        "eval": _binary_eval(model=model, x=x_test[valid_tst], y=label["test"][valid_tst].astype(np.int64), threshold=thr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-md", default="documents/20260502_plan5_laneF.md")
    parser.add_argument("--output-json", default="documents/20260502_plan5_laneF.json")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config); cfg, _ = resolve_costs(cfg, None)
    dc = cfg.get("data", {})
    zw = cfg.get("normalization", {}).get("zscore_window_days", 60)
    ct = f"{dc['symbol']}_{dc['interval']}_2018-01-01_2024-01-01_z{zw}_v2"
    fdf, rr = load_training_features(
        symbol=dc["symbol"], interval=dc["interval"], start="2018-01-01", end="2024-01-01",
        zscore_window=zw, cache_dir="checkpoints/data_cache", cache_tag=ct,
    )
    splits, _ = select_wfo_splits(build_wfo_splits(fdf, dc), args.folds)

    all_labels = []
    for s in splits:
        print(f"[LaneF] fold={s.fold_idx}")
        ds = WFODataset(fdf, rr, s, seq_len=64)
        x_tr = _state_features(ds.train_features, ds.train_returns)
        x_val = _state_features(ds.val_features, ds.val_returns)
        x_tst = _state_features(ds.test_features, ds.test_returns)

        labels = {
            "false_derisk": {
                "train": make_false_derisk_label(ds.train_returns),
                "val": make_false_derisk_label(ds.val_returns),
                "test": make_false_derisk_label(ds.test_returns),
            },
            "pullback_recovery": {
                "train": make_pullback_recovery_label(ds.train_returns),
                "val": make_pullback_recovery_label(ds.val_returns),
                "test": make_pullback_recovery_label(ds.test_returns),
            },
            "pullback_mask": {
                "train": np.zeros(len(ds.train_returns), dtype=np.int64),
                "val": np.zeros(len(ds.val_returns), dtype=np.int64),
                "test": _pullback_no_fire_mask(ds.test_returns).astype(np.int64),
            },
        }

        for name, lab in labels.items():
            r = eval_label(name, lab, x_tr, x_val, x_tst, args.seed + s.fold_idx)
            if r:
                r["fold"] = s.fold_idx
                all_labels.append(r)

    # Aggregation
    lines = [
        "# Plan 5 Lane F: Ridge Label Expansion",
        f"Folds: `{', '.join(str(s.fold_idx) for s in splits)}`",
        "",
        "## Label Separability (logistic probe over state features)",
        "",
        "| label | fold | density | AUC | AP | recall | false_active | pred_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in all_labels:
        e = r["eval"]
        lines.append(f"| {r['name']} | {r['fold']} | {_fmt(r['density'])} | {_fmt(e.get('auc'))} | "
                     f"{_fmt(e.get('average_precision'))} | {_fmt(e.get('recall'))} | "
                     f"{_fmt(e.get('false_active_rate'))} | {_fmt(e.get('pred_rate'))} |")

    # Cross-fold means
    lines.extend(["", "## Cross-Fold Mean", "",
        "| label | density | AUC mean | false_active worst |",
        "|---|---:|---:|---:|"])
    for name in ["false_derisk", "pullback_recovery", "pullback_mask"]:
        rows = [r for r in all_labels if r["name"] == name]
        if rows:
            e = [r["eval"] for r in rows]
            lines.append(f"| {name} | {_fmt(np.mean([r['density'] for r in rows]))} | "
                         f"{_fmt(np.mean([x.get('auc', float('nan')) for x in e]))} | "
                         f"{_fmt(np.max([x.get('false_active_rate', float('nan')) for x in e]))} |")

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, default=str)
    print(f"[LaneF] Done -> {args.output_md}")


if __name__ == "__main__":
    main()
