"""Plan010 position-utility WM probe.

This evaluates the Transformer world model's position_utility head without
using test-period labels for selection. Thresholds and throttle settings are
selected on validation only, then applied once to test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import (
    _apply_event_throttle,
    _backtest_positions,
    _candidate_utilities,
    _shift_for_execution,
    _unit_cost,
)
from unidream.data.dataset import WFODataset
from unidream.device import resolve_device
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    return f"{x:.{digits}f}" if math.isfinite(x) else "NA"


def _threshold_grid(improve: np.ndarray, active_cap: float) -> list[float]:
    vals = np.asarray(improve, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [float("inf")]
    qs = (0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995)
    out = [float(np.quantile(vals, q)) for q in qs]
    cap_q = max(0.0, min(0.995, 1.0 - float(active_cap)))
    out.append(float(np.quantile(vals, cap_q)))
    out.extend([0.0, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, float("inf")])
    return sorted(set(out))


def _metric_score(metrics: dict[str, Any], *, active_cap: float, turnover_cap: float) -> float:
    alpha = float(metrics.get("alpha_excess_pt", 0.0))
    sharpe = float(metrics.get("sharpe_delta", 0.0))
    maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
    turnover = float(metrics.get("turnover", 999.0))
    active = 1.0 - float(metrics.get("flat_rate", 1.0))
    score = alpha + 6.0 * sharpe + 2.0 * max(0.0, -maxdd) - 8.0 * max(0.0, maxdd)
    score -= 0.08 * turnover
    score -= 20.0 * max(0.0, active - float(active_cap))
    score -= 2.0 * max(0.0, turnover - float(turnover_cap))
    if alpha >= 3.0 and maxdd <= -3.0:
        score += 100.0
    if alpha < 0.0:
        score -= 15.0 + 0.5 * abs(alpha)
    if turnover <= 1e-12 and active <= 1e-12:
        score -= 2.0
    return float(score)


def _positions_from_pred(
    pred: np.ndarray,
    *,
    candidates: tuple[float, ...],
    allowed_mask: np.ndarray,
    threshold: float,
    benchmark_position: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    cands = np.asarray(candidates, dtype=np.float64)
    bench = float(benchmark_position)
    bench_idx = int(np.argmin(np.abs(cands - bench)))
    best_idx = np.argmax(np.where(allowed_mask.reshape(1, -1), pred_arr, -1e30), axis=1)
    improve = pred_arr[np.arange(len(pred_arr)), best_idx] - pred_arr[:, bench_idx]
    choose = np.isfinite(improve) & (improve > float(threshold))
    selected = np.full(len(pred_arr), bench, dtype=np.float64)
    selected[choose] = cands[best_idx[choose]]
    return selected, {
        "raw_active_rate": float(np.mean(choose)),
        "pred_improve_mean": float(np.nanmean(improve)),
        "pred_improve_top10": _top_fraction_mean(improve, improve, 0.10),
    }


def _top_fraction_mean(score: np.ndarray, value: np.ndarray, frac: float) -> float:
    s = np.asarray(score, dtype=np.float64)
    v = np.asarray(value, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(v)
    if int(mask.sum()) == 0:
        return float("nan")
    n = max(1, int(mask.sum() * float(frac)))
    idx = np.flatnonzero(mask)
    top_idx = idx[np.argsort(s[mask])[-n:]]
    return float(np.mean(v[top_idx]))


def _utility_diag(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    candidates: tuple[float, ...],
    benchmark_position: float,
) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    cands = np.asarray(candidates, dtype=np.float64)
    bench_idx = int(np.argmin(np.abs(cands - float(benchmark_position))))
    best_idx = np.argmax(p, axis=1)
    pred_improve = p[np.arange(len(p)), best_idx] - p[:, bench_idx]
    truth_improve = y[np.arange(len(y)), best_idx] - y[:, bench_idx]
    oracle_best = np.argmax(y, axis=1)
    mask = np.isfinite(pred_improve) & np.isfinite(truth_improve)
    if int(mask.sum()) < 20:
        return {
            "rank_hit_rate": float("nan"),
            "top10_realized_improve": float("nan"),
            "selected_realized_mean": float("nan"),
            "pred_positive_rate": float("nan"),
            "pearson": float("nan"),
        }
    pp = pred_improve[mask]
    tt = truth_improve[mask]
    corr = float(np.corrcoef(pp, tt)[0, 1]) if np.std(pp) > 1e-12 and np.std(tt) > 1e-12 else float("nan")
    return {
        "rank_hit_rate": float(np.mean(best_idx[mask] == oracle_best[mask])),
        "top10_realized_improve": _top_fraction_mean(pp, tt, 0.10),
        "selected_realized_mean": float(np.mean(tt)),
        "pred_positive_rate": float(np.mean(pp > 0.0)),
        "pearson": corr,
    }


def _select_on_val(
    *,
    pred_val: np.ndarray,
    val_returns: np.ndarray,
    cfg: dict,
    costs_cfg: dict,
    candidates: tuple[float, ...],
    allowed_mask: np.ndarray,
    benchmark_position: float,
    hold_grid: tuple[int, ...],
    cooldown_grid: tuple[int, ...],
    active_cap: float,
    turnover_cap: float,
) -> dict[str, Any]:
    cands = np.asarray(candidates, dtype=np.float64)
    bench = float(benchmark_position)
    bench_idx = int(np.argmin(np.abs(cands - bench)))
    masked = np.where(allowed_mask.reshape(1, -1), pred_val, -1e30)
    best_idx = np.argmax(masked, axis=1)
    improve = pred_val[np.arange(len(pred_val)), best_idx] - pred_val[:, bench_idx]
    thresholds = _threshold_grid(improve, active_cap)

    best: dict[str, Any] | None = None
    for threshold in thresholds:
        if not math.isfinite(float(threshold)):
            selected = np.full(len(pred_val), bench, dtype=np.float64)
        else:
            selected, _diag = _positions_from_pred(
                pred_val,
                candidates=candidates,
                allowed_mask=allowed_mask,
                threshold=float(threshold),
                benchmark_position=bench,
            )
        for hold in hold_grid:
            for cooldown in cooldown_grid:
                throttled = _apply_event_throttle(
                    selected,
                    benchmark_position=bench,
                    hold_bars=int(hold),
                    cooldown_bars=int(cooldown),
                )
                positions = _shift_for_execution(throttled, bench)
                metrics, _ = _backtest_positions(
                    val_returns,
                    positions,
                    cfg=cfg,
                    costs_cfg=costs_cfg,
                    benchmark_position=bench,
                )
                score = _metric_score(metrics, active_cap=active_cap, turnover_cap=turnover_cap)
                row = {
                    "threshold": float(threshold),
                    "hold_bars": int(hold),
                    "cooldown_bars": int(cooldown),
                    "score": float(score),
                    "val_metrics": metrics,
                }
                if best is None or row["score"] > float(best["score"]):
                    best = row
    assert best is not None
    return best


def _run_fold(
    *,
    split,
    features_df,
    raw_returns,
    cfg: dict,
    costs_cfg: dict,
    checkpoint_dir: str,
    device: str,
    args,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seq_len = int(cfg.get("data", {}).get("seq_len", 64))
    ds = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
    fr = prepare_fold_runtime(
        fold_idx=split.fold_idx,
        checkpoint_dir=checkpoint_dir,
        ac_cfg=cfg.get("ac", {}),
        resume=False,
        start_from="test",
        stop_after="test",
    )
    if not fr["has_wm_ckpt"]:
        print(f"[Plan010PosU] fold={split.fold_idx} skipped: no WM checkpoint")
        return [], []

    ensemble = build_ensemble(ds.obs_dim, cfg)
    wm = WorldModelTrainer(ensemble, cfg, device=device)
    wm.load(fr["wm_path"])

    enc_val = wm.encode_sequence(ds.val_features, seq_len=seq_len)
    enc_test = wm.encode_sequence(ds.test_features, seq_len=seq_len)
    aux_val = wm.predict_auxiliary_from_encoded(enc_val["z"], enc_val["h"], features=ds.val_features)
    aux_test = wm.predict_auxiliary_from_encoded(enc_test["z"], enc_test["h"], features=ds.test_features)
    if "position_utility" not in aux_val or "position_utility" not in aux_test:
        raise RuntimeError("position_utility head is not available in this checkpoint/config")

    pred_val = np.asarray(aux_val["position_utility"], dtype=np.float64)
    pred_test = np.asarray(aux_test["position_utility"], dtype=np.float64)
    candidates = tuple(float(x) for x in cfg.get("world_model", {}).get("position_utility_positions", []))
    if not candidates:
        raise RuntimeError("world_model.position_utility_positions is empty")
    if pred_val.shape[1] != len(candidates):
        raise RuntimeError(f"position_utility width {pred_val.shape[1]} != candidates {len(candidates)}")

    bench = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    unit_cost = _unit_cost(costs_cfg)
    truth_val, _ = _candidate_utilities(
        ds.val_returns,
        candidates=candidates,
        horizon=int(args.horizon),
        benchmark_position=bench,
        unit_cost=unit_cost,
        dd_penalty=float(args.dd_penalty),
        vol_penalty=float(args.vol_penalty),
    )
    truth_test, _ = _candidate_utilities(
        ds.test_returns,
        candidates=candidates,
        horizon=int(args.horizon),
        benchmark_position=bench,
        unit_cost=unit_cost,
        dd_penalty=float(args.dd_penalty),
        vol_penalty=float(args.vol_penalty),
    )

    hold_grid = tuple(int(x) for x in str(args.hold_grid).split(",") if str(x).strip())
    cooldown_grid = tuple(int(x) for x in str(args.cooldown_grid).split(",") if str(x).strip())

    variant_masks: dict[str, np.ndarray] = {
        "all": np.ones(len(candidates), dtype=bool),
        "under_only": np.asarray(candidates, dtype=np.float64) <= bench + 1e-12,
        "small_overlay": np.asarray(candidates, dtype=np.float64) >= bench - 0.30,
    }
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for variant, mask in variant_masks.items():
        if not np.any(mask):
            continue
        val_diag = _utility_diag(pred_val, truth_val, candidates=candidates, benchmark_position=bench)
        test_diag = _utility_diag(pred_test, truth_test, candidates=candidates, benchmark_position=bench)
        diagnostics.append(
            {
                "fold": split.fold_idx,
                "variant": variant,
                "val_utility": val_diag,
                "test_utility": test_diag,
            }
        )
        best = _select_on_val(
            pred_val=pred_val,
            val_returns=ds.val_returns,
            cfg=cfg,
            costs_cfg=costs_cfg,
            candidates=candidates,
            allowed_mask=mask,
            benchmark_position=bench,
            hold_grid=hold_grid,
            cooldown_grid=cooldown_grid,
            active_cap=float(args.active_cap),
            turnover_cap=float(args.turnover_cap),
        )
        selected, pred_diag = _positions_from_pred(
            pred_test,
            candidates=candidates,
            allowed_mask=mask,
            threshold=float(best["threshold"]),
            benchmark_position=bench,
        )
        throttled = _apply_event_throttle(
            selected,
            benchmark_position=bench,
            hold_bars=int(best["hold_bars"]),
            cooldown_bars=int(best["cooldown_bars"]),
        )
        positions = _shift_for_execution(throttled, bench)
        test_metrics, pnl = _backtest_positions(
            ds.test_returns,
            positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=bench,
        )
        rows.append(
            {
                "fold": split.fold_idx,
                "variant": variant,
                "test": test_metrics,
                "val": best["val_metrics"],
                "selection": {
                    "threshold": float(best["threshold"]),
                    "hold_bars": int(best["hold_bars"]),
                    "cooldown_bars": int(best["cooldown_bars"]),
                    "score": float(best["score"]),
                    **pred_diag,
                },
                "pnl": pnl,
            }
        )
    return rows, diagnostics


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in sorted({str(r["variant"]) for r in rows}):
        subset = [r for r in rows if r["variant"] == variant]
        tests = [r["test"] for r in subset]
        if not tests:
            continue
        alpha = np.asarray([float(t.get("alpha_excess_pt", 0.0)) for t in tests], dtype=np.float64)
        maxdd = np.asarray([float(t.get("maxdd_delta_pt", 0.0)) for t in tests], dtype=np.float64)
        turnover = np.asarray([float(t.get("turnover", 0.0)) for t in tests], dtype=np.float64)
        out.append(
            {
                "variant": variant,
                "folds": len(subset),
                "pass_3m3": int(np.sum((alpha >= 3.0) & (maxdd <= -3.0))),
                "alpha_mean": float(np.mean(alpha)),
                "alpha_worst": float(np.min(alpha)),
                "maxdd_mean": float(np.mean(maxdd)),
                "maxdd_worst": float(np.max(maxdd)),
                "turnover_mean": float(np.mean(turnover)),
                "turnover_max": float(np.max(turnover)),
            }
        )
    return out


def _write_outputs(path_json: str, path_md: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(path_md) or ".", exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = [
        "# Plan010 Position Utility WM Probe",
        "",
        "Validation selects threshold/hold/cooldown only; test is report-only.",
        "",
        "## Aggregate",
        "",
        "| variant | folds | pass +3/-3 | Alpha mean | Alpha worst | MaxDD mean | MaxDD worst | TO mean | TO max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['variant']} | {row['folds']} | {row['pass_3m3']}/{row['folds']} | "
            f"{_fmt(row['alpha_mean'])} | {_fmt(row['alpha_worst'])} | "
            f"{_fmt(row['maxdd_mean'])} | {_fmt(row['maxdd_worst'])} | "
            f"{_fmt(row['turnover_mean'])} | {_fmt(row['turnover_max'])} |"
        )
    lines.extend(
        [
            "",
            "## Fold Detail",
            "",
            "| fold | variant | AlphaEx | SharpeD | MaxDDD | TO | active | selection |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["rows"]:
        t = row["test"]
        active = 1.0 - float(t.get("flat_rate", 1.0))
        sel = row.get("selection", {})
        sel_short = {
            k: sel.get(k)
            for k in ("threshold", "hold_bars", "cooldown_bars", "raw_active_rate", "score")
            if k in sel
        }
        lines.append(
            f"| {row['fold']} | {row['variant']} | {_fmt(t.get('alpha_excess_pt'))} | "
            f"{_fmt(t.get('sharpe_delta'))} | {_fmt(t.get('maxdd_delta_pt'))} | "
            f"{_fmt(t.get('turnover'))} | {_fmt(active)} | `{sel_short}` |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| fold | variant | val signal | test signal |",
            "|---:|---|---|---|",
        ]
    )
    for d in payload["diagnostics"]:
        lines.append(
            f"| {d['fold']} | {d['variant']} | `{d['val_utility']}` | `{d['test_utility']}` |"
        )
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan010_position_utility_probe")
    parser.add_argument("--config", default="configs/plan010_position_utility_wm.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="3,4,5")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--dd-penalty", type=float, default=1.0)
    parser.add_argument("--vol-penalty", type=float, default=0.25)
    parser.add_argument("--hold-grid", default="1,4,16,32,64")
    parser.add_argument("--cooldown-grid", default="0,16,32,64")
    parser.add_argument("--active-cap", type=float, default=0.55)
    parser.add_argument("--turnover-cap", type=float, default=180.0)
    parser.add_argument("--output-json", default="docs_local/plan010_position_utility_probe.json")
    parser.add_argument("--output-md", default="docs_local/plan010_position_utility_probe.md")
    args = parser.parse_args()

    device = resolve_device(args.device)
    set_seed(int(args.seed))
    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg, None)
    checkpoint_dir = args.checkpoint_dir or cfg.get("logging", {}).get("checkpoint_dir", "checkpoints")
    costs_cfg = cfg.get("costs", {})
    data_cfg = cfg.get("data", {})
    zwin = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{data_cfg['symbol']}_{data_cfg['interval']}_{args.start}_{args.end}_z{zwin}_v2"
    features_df, raw_returns = load_training_features(
        symbol=data_cfg["symbol"],
        interval=data_cfg["interval"],
        start=args.start,
        end=args.end,
        zscore_window=zwin,
        cache_dir="checkpoints/data_cache",
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    print(f"[Plan010PosU] device={device} checkpoint_dir={checkpoint_dir} folds={selected}")

    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for split in splits:
        print(f"[Plan010PosU] fold={split.fold_idx}")
        fold_rows, fold_diag = _run_fold(
            split=split,
            features_df=features_df,
            raw_returns=raw_returns,
            cfg=cfg,
            costs_cfg=costs_cfg,
            checkpoint_dir=checkpoint_dir,
            device=device,
            args=args,
        )
        rows.extend(fold_rows)
        diagnostics.extend(fold_diag)

    payload = {
        "config": args.config,
        "checkpoint_dir": checkpoint_dir,
        "folds": selected,
        "rows": rows,
        "diagnostics": diagnostics,
        "aggregate": _aggregate(rows),
    }
    _write_outputs(args.output_json, args.output_md, payload)
    for row in payload["aggregate"]:
        print(
            f"[Plan010PosU] {row['variant']}: pass={row['pass_3m3']}/{row['folds']} "
            f"alpha_mean={row['alpha_mean']:+.2f} maxdd_mean={row['maxdd_mean']:+.2f} "
            f"to_mean={row['turnover_mean']:.2f}"
        )
    print(f"[Plan010PosU] wrote {args.output_md}")


if __name__ == "__main__":
    main()
