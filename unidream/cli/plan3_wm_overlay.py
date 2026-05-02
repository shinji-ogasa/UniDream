"""
Plan 3: WM-based standalone overlay. GPU inference.
Uses trained WM predictive heads to compute D risk-sensitive utility
+ triple-barrier guard + pullback block. No ridge regression.
"""
from __future__ import annotations

import argparse, json, math, os, traceback
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import (
    _apply_event_throttle, _backtest_positions, _pullback_no_fire_mask,
    _shift_for_execution, _unit_cost,
)
from unidream.data.dataset import WFODataset
from unidream.eval.pbo import compute_pbo
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.world_model.train_wm import build_ensemble, WorldModelTrainer
from unidream.device import resolve_device


CANDIDATES = (0.75, 1.0, 1.05, 1.10)
HORIZON = 32
DD_PENALTY = 1.50
VOL_PENALTY = 0.15
MIN_THRESHOLD = 0.001
COOLDOWN_GRID = (0, 32)
ACTIVE_CAP = 0.25


def _future_horizon_stats(returns: np.ndarray, horizon: int):
    r = np.asarray(returns, dtype=np.float64)
    h = max(1, int(horizon))
    n = len(r)
    ret_sum = np.zeros(n, dtype=np.float64)
    dd = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i in range(n - h):
        w = r[i + 1: i + 1 + h]
        ret_sum[i] = np.sum(w)
        dd[i] = max(0.0, -np.min(np.cumsum(w)))
        valid[i] = True
    return ret_sum, dd, valid


def _actual_candidate_utilities(returns, candidates, horizon, benchmark_pos, cost, dd_penalty):
    ret_sum, bench_dd, valid = _future_horizon_stats(returns, horizon)
    bench_dd_clipped = np.maximum(bench_dd, 0.0)
    n, k = len(returns), len(candidates)
    vals = np.full((n, k), np.nan, dtype=np.float64)
    for ci, pos in enumerate(candidates):
        pos = float(pos)
        pos_dd = np.maximum(np.abs(pos) * bench_dd_clipped, 0.0)
        dd_worsen = np.maximum(pos_dd - float(benchmark_pos) * bench_dd_clipped, 0.0)
        overlay = pos - float(benchmark_pos)
        vals[valid, ci] = overlay * ret_sum[valid] - abs(overlay) * cost - dd_penalty * dd_worsen[valid]
    return vals, valid


def _wm_utility(wm_trainer, z, h, candidates, benchmark_pos, unit_cost, dd_penalty, vol_penalty):
    aux = wm_trainer.predict_auxiliary_from_encoded(z, h)
    ret = aux.get("return")
    dd  = aux.get("drawdown")
    vol = aux.get("vol")
    if ret is None and dd is None and vol is None:
        return np.zeros((len(z), len(candidates)), dtype=np.float64)

    # Horizon-32 (last column)
    def _h32(x):
        if x is None:
            return np.zeros(len(z), dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        return x[:, -1] if x.ndim > 1 and x.shape[1] >= 5 else x.squeeze(-1)

    ret_h32 = _h32(ret)
    dd_h32 = np.maximum(_h32(dd), 0.0)
    vol_h32 = np.maximum(_h32(vol), 0.0)

    n, k = len(ret_h32), len(candidates)
    vals = np.full((n, k), np.nan, dtype=np.float64)
    for ci, pos in enumerate(candidates):
        pos = float(pos)
        pos_dd = np.maximum(np.abs(pos) * dd_h32, 0.0)
        dd_worsen = np.maximum(pos_dd - float(benchmark_pos) * dd_h32, 0.0)
        overlay = pos - float(benchmark_pos)
        vals[:, ci] = overlay * ret_h32 - abs(overlay) * unit_cost - dd_penalty * dd_worsen - vol_penalty * abs(overlay) * vol_h32
    return vals


def _threshold_grid(improve: np.ndarray, active_cap: float = 0.25) -> list[float]:
    vals = np.asarray(np.abs(improve), dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return [float("inf")]
    qs = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
    raw = list(vals)
    raw.extend([float(np.quantile(vals, q)) for q in qs])
    raw.extend([0.0])
    cap_q = max(0.0, min(0.995, 1.0 - float(active_cap)))
    if len(vals):
        raw.append(float(np.quantile(vals, cap_q)))
    return sorted(set([float("inf") if not math.isfinite(t) else t for t in raw]))


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "NA"
    return f"{x:.{digits}f}" if math.isfinite(x) else "NA"


def _mean(vals): return float(np.nanmean([float(x) for x in vals])) if vals else float("nan")
def _min(vals): return float(np.nanmin([float(x) for x in vals])) if vals else float("nan")
def _max(vals): return float(np.nanmax([float(x) for x in vals])) if vals else float("nan")


def _val_select(
    wm_val_util, actual_val_util, val_valid, candidates, benchmark_pos,
    cooldown_grid, active_cap, min_threshold,
):
    bench_idx = int(np.argmin(np.abs(np.asarray(candidates) - float(benchmark_pos))))
    best_idx = np.argmax(wm_val_util, axis=1)
    improve = wm_val_util[np.arange(len(wm_val_util)), best_idx] - wm_val_util[:, bench_idx]

    thresholds = _threshold_grid(improve[val_valid], active_cap)
    thresholds = sorted(set([max(t, min_threshold) for t in thresholds if math.isfinite(t)] + [float("inf")]))

    best_score = -1e9
    best = {"threshold": float("inf"), "cooldown": 0}
    for cd in cooldown_grid:
        for thr in thresholds:
            if thr == float("inf"):
                continue
            sel = np.full(len(wm_val_util), float(benchmark_pos), dtype=np.float64)
            mask = val_valid & (improve > thr)
            sel[mask] = np.asarray(candidates, dtype=np.float64)[best_idx[mask]]
            sel = _apply_event_throttle(sel, benchmark_position=float(benchmark_pos), cooldown_bars=int(cd), hold_bars=1)
            sel_idx = np.argmin(np.abs(np.asarray(candidates)[None, :] - sel[:, None]), axis=1)
            util = actual_val_util[np.arange(len(actual_val_util)), sel_idx]
            ok = val_valid & np.isfinite(util)
            if not np.any(ok):
                continue
            active_rate = float(np.mean(np.abs(sel[val_valid] - float(benchmark_pos)) > 1e-12))
            mean_util = float(np.mean(util[ok]))
            if active_rate <= active_cap and mean_util > best_score:
                best_score = mean_util
                best = {"threshold": thr, "cooldown": int(cd), "score": mean_util}
    return best


def _run_fold(
    split, features_df, raw_returns, cfg, costs_cfg,
    benchmark_pos, checkpoint_dir, device, seed, seq_len,
):
    try:
        dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
        fr = prepare_fold_runtime(
            fold_idx=split.fold_idx, checkpoint_dir=checkpoint_dir, ac_cfg=cfg.get("ac", {}),
            resume=False, start_from="test", stop_after="test")
        if not fr["has_wm_ckpt"]:
            print(f"  [SKIP fold={split.fold_idx}] no WM checkpoint")
            return None

        ensemble = build_ensemble(dataset.obs_dim, cfg)
        wm = WorldModelTrainer(ensemble, cfg, device=device)
        wm.load(fr["wm_path"])

        enc_tr = wm.encode_sequence(dataset.train_features, seq_len=seq_len)
        enc_val = wm.encode_sequence(dataset.val_features, seq_len=seq_len)
        enc_test = wm.encode_sequence(dataset.test_features, seq_len=seq_len)

        uc = _unit_cost(costs_cfg)

        wm_tr  = _wm_utility(wm, enc_tr["z"], enc_tr["h"], CANDIDATES, benchmark_pos, uc, DD_PENALTY, VOL_PENALTY)
        wm_val = _wm_utility(wm, enc_val["z"], enc_val["h"], CANDIDATES, benchmark_pos, uc, DD_PENALTY, VOL_PENALTY)
        wm_tst = _wm_utility(wm, enc_test["z"], enc_test["h"], CANDIDATES, benchmark_pos, uc, DD_PENALTY, VOL_PENALTY)

        act_val, val_valid = _actual_candidate_utilities(
            dataset.val_returns, CANDIDATES, HORIZON, benchmark_pos, uc, DD_PENALTY)
        act_tst, tst_valid = _actual_candidate_utilities(
            dataset.test_returns, CANDIDATES, HORIZON, benchmark_pos, uc, DD_PENALTY)

        # Triple-barrier guard: use WM drawdown prediction
        aux_tst = wm.predict_auxiliary_from_encoded(enc_test["z"], enc_test["h"])
        dd_tst = aux_tst.get("drawdown")
        if dd_tst is not None:
            dd_tst = np.asarray(dd_tst, dtype=np.float64)
            dd_h32 = dd_tst[:, -1] if dd_tst.ndim > 1 and dd_tst.shape[1] >= 5 else dd_tst.squeeze(-1)
            dd_h32 = np.maximum(dd_h32, 0.0)
            dd_med = float(np.median(dd_h32[np.isfinite(dd_h32)])) if np.any(np.isfinite(dd_h32)) else 0.0
            danger_allow = dd_h32 >= dd_med * 0.5  # allow de-risk when DD prediction is meaningful
        else:
            danger_allow = np.ones(len(enc_test["z"]), dtype=bool)

        pullback = _pullback_no_fire_mask(dataset.test_returns)

        best = _val_select(wm_val, act_val, val_valid, CANDIDATES, benchmark_pos,
                           COOLDOWN_GRID, ACTIVE_CAP, MIN_THRESHOLD)

        thr = float(best["threshold"]) if math.isfinite(float(best["threshold"])) else float("inf")
        cd = int(best["cooldown"])

        bench_idx = int(np.argmin(np.abs(np.asarray(CANDIDATES) - float(benchmark_pos))))
        best_idx = np.argmax(wm_tst, axis=1)
        improve_tst = wm_tst[np.arange(len(wm_tst)), best_idx] - wm_tst[:, bench_idx]

        sel = np.full(len(wm_tst), float(benchmark_pos), dtype=np.float64)
        active = tst_valid & (improve_tst > thr) & danger_allow & (~pullback)
        sel[active] = np.asarray(CANDIDATES, dtype=np.float64)[best_idx[active]]
        sel = _apply_event_throttle(sel, benchmark_position=float(benchmark_pos), cooldown_bars=cd, hold_bars=1)
        positions = _shift_for_execution(sel, benchmark_pos)
        metrics, pnl = _backtest_positions(dataset.test_returns, positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=benchmark_pos)

        return {
            "fold": split.fold_idx,
            "alpha_excess_pt": float(metrics["alpha_excess_pt"]),
            "sharpe_delta": float(metrics["sharpe_delta"]),
            "maxdd_delta_pt": float(metrics["maxdd_delta_pt"]),
            "turnover": float(metrics["turnover"]),
            "flat_rate": float(metrics["flat_rate"]),
            "long_rate": float(metrics["long_rate"]),
            "active_events": int(np.sum(active)),
            "pullback_blocked": int(np.sum(pullback)),
            "danger_blocked": int(np.sum(~danger_allow & tst_valid)),
            "threshold": thr,
            "cooldown": cd,
            "pnl": pnl,
        }
    except Exception:
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/acplan13_base_wm_s011")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default="documents/20260502_plan3_wm_overlay.json")
    parser.add_argument("--output-md", default="documents/20260502_plan3_wm_overlay.md")
    args = parser.parse_args()

    dev = resolve_device(args.device)
    print(f"[Plan3WM] device={dev}")
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _ = resolve_costs(cfg, None)
    costs_cfg = cfg.get("costs", {})
    bp = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    dc = cfg.get("data", {})
    zw = cfg.get("normalization", {}).get("zscore_window_days", 60)
    ct = f"{dc['symbol']}_{dc['interval']}_{args.start}_{args.end}_z{zw}_v2"
    fdf, rr = load_training_features(
        symbol=dc["symbol"], interval=dc["interval"], start=args.start, end=args.end,
        zscore_window=zw, cache_dir="checkpoints/data_cache", cache_tag=ct,
        extra_series_mode=dc.get("extra_series_mode", "derived"),
        extra_series_include=dc.get("extra_series_include"),
        include_funding=bool(dc.get("include_funding", True)),
        include_oi=bool(dc.get("include_oi", True)),
        include_mark=bool(dc.get("include_mark", True)),
    )
    splits, _ = select_wfo_splits(build_wfo_splits(fdf, dc), args.folds)
    sl = dc.get("seq_len", 64)

    results = []
    for s in splits:
        print(f"[Plan3WM] fold={s.fold_idx}")
        r = _run_fold(s, fdf, rr, cfg, costs_cfg, bp, args.checkpoint_dir, dev, args.seed, sl)
        if r:
            results.append(r)

    if not results:
        print("[Plan3WM] No results")
        return

    alpha = [x["alpha_excess_pt"] for x in results]
    maxdd = [x["maxdd_delta_pt"] for x in results]
    sharpe = [x["sharpe_delta"] for x in results]
    turnover = [x["turnover"] for x in results]

    agg = {
        "folds": len(results),
        "alpha_mean": _mean(alpha), "alpha_worst": _min(alpha), "alpha_median": float(np.median(alpha)),
        "maxdd_mean": _mean(maxdd), "maxdd_worst": _max(maxdd),
        "sharpe_mean": _mean(sharpe), "turnover_max": _max(turnover),
        "pbo": compute_pbo([x["pnl"] for x in results]) if len(results) >= 2 else float("nan"),
    }

    lines = [
        "# Plan 3 WM Overlay Report",
        "", f"Checkpoint: `{args.checkpoint_dir}`", f"Device: `{dev}`",
        f"Folds: `{', '.join(map(str, args.folds))}`", "",
        "## Aggregate", "",
        "| metric | value |", "|---|---:|",
    ]
    for k in ["folds", "alpha_mean", "alpha_worst", "alpha_median", "maxdd_mean", "maxdd_worst", "sharpe_mean", "turnover_max", "pbo"]:
        lines.append(f"| {k} | {_fmt(agg[k])} |")
    lines.extend(["", "## Per-Fold", "",
        "| fold | AlphaEx | MaxDDΔ | SharpeΔ | turnover | flat | active | pullback | danger | thr | cd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in results:
        lines.append(f"| {r['fold']} | {_fmt(r['alpha_excess_pt'])} | {_fmt(r['maxdd_delta_pt'])} | "
                     f"{_fmt(r['sharpe_delta'])} | {_fmt(r['turnover'])} | {_fmt(r['flat_rate'])} | "
                     f"{r['active_events']} | {r['pullback_blocked']} | {r['danger_blocked']} | "
                     f"{_fmt(r['threshold'])} | {r['cooldown']} |")

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"results": results, "aggregate": agg}, f, indent=2, default=str)
    print(f"[Plan3WM] Done → {args.output_md}")


if __name__ == "__main__":
    main()
