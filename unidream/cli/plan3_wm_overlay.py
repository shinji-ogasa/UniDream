"""
Plan 3: WM-based standalone overlay.

Uses trained Transformer World Model predictive heads (return/vol/drawdown)
to compute D risk-sensitive candidate utilities. Applies triple-barrier downside
guard via WM drawdown predictions + pullback false-de-risk guard.

No ridge regression. No configs/trading.yaml modification.
GPU inference via --device cuda.

Usage:
  uv run python -u -m unidream.cli.plan3_wm_overlay \
    --checkpoint-dir checkpoints/acplan13_base_wm_s011 \
    --folds 4,5,6 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from typing import Any

import numpy as np

from unidream.cli.exploration_board_probe import (
    _apply_event_throttle,
    _backtest_positions,
    _candidate_utilities,
    _pullback_no_fire_mask,
    _shift_for_execution,
    _threshold_grid,
    _unit_cost,
)
from unidream.data.dataset import WFODataset
from unidream.device import resolve_device
from unidream.eval.pbo import compute_pbo
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.world_model.train_wm import build_ensemble, WorldModelTrainer


# ── overlay parameters ────────────────────────────────────────────────
CANDIDATES        = (0.75, 1.0, 1.05, 1.10)
HORIZON           = 32
DD_PENALTY        = 1.50
VOL_PENALTY       = 0.15
MIN_THRESHOLD     = 0.001
COOLDOWN_GRID     = (0, 32)
ACTIVE_CAP        = 0.25
BENCHMARK_DEFAULT = 1.0


# ── WM utility ────────────────────────────────────────────────────────

def _h32(arr: np.ndarray | None, n: int) -> np.ndarray:
    """Extract horizon-32 prediction (last column)."""
    if arr is None:
        return np.zeros(n, dtype=np.float64)
    a = np.asarray(arr, dtype=np.float64)
    return a[:, -1] if a.ndim > 1 and a.shape[1] >= 5 else a.squeeze(-1)


def compute_wm_utility(
    wm_trainer: WorldModelTrainer,
    z: np.ndarray,
    h: np.ndarray,
    *,
    candidates: tuple[float, ...],
    benchmark_position: float,
    unit_cost: float,
    dd_penalty: float,
    vol_penalty: float,
) -> np.ndarray:
    """Compute D risk-sensitive utility for each candidate position
    using WM predictive head outputs (return / drawdown / vol at horizon=32)."""
    aux = wm_trainer.predict_auxiliary_from_encoded(z, h)
    n = len(z)
    ret_h32 = _h32(aux.get("return"), n)
    dd_h32  = np.maximum(_h32(aux.get("drawdown"), n), 0.0)
    vol_h32 = np.maximum(_h32(aux.get("vol"), n), 0.0)

    k = len(candidates)
    vals = np.full((n, k), np.nan, dtype=np.float64)
    for ci, pos in enumerate(candidates):
        pos = float(pos)
        pos_dd = np.abs(pos) * dd_h32
        dd_worsen = np.maximum(pos_dd - float(benchmark_position) * dd_h32, 0.0)
        overlay = pos - float(benchmark_position)
        vals[:, ci] = (
            overlay * ret_h32
            - abs(overlay) * unit_cost
            - dd_penalty * dd_worsen
            - vol_penalty * abs(overlay) * vol_h32
        )
    return vals


# ── validation selection ──────────────────────────────────────────────

def select_best_config(
    wm_util: np.ndarray,
    actual_util: np.ndarray,
    valid: np.ndarray,
    *,
    candidates: tuple[float, ...],
    benchmark_position: float,
    cooldown_grid: tuple[int, ...],
    active_cap: float,
    min_threshold: float,
) -> dict[str, Any]:
    """Select threshold + cooldown that maximizes actual utility on validation set."""
    cands = np.asarray(candidates, dtype=np.float64)
    bench_idx = int(np.argmin(np.abs(cands - float(benchmark_position))))
    best_idx = np.argmax(wm_util, axis=1)
    improve = wm_util[np.arange(len(wm_util)), best_idx] - wm_util[:, bench_idx]

    thresholds = _threshold_grid(improve[valid], active_cap=active_cap)
    thresholds = sorted({
        max(t, min_threshold) for t in thresholds if math.isfinite(t)
    } | {float("inf")})

    best_score = -1e12
    best: dict[str, Any] = {"threshold": float("inf"), "cooldown": 0}
    for cd in cooldown_grid:
        for thr in thresholds:
            if thr == float("inf"):
                continue
            sel = np.full(len(wm_util), float(benchmark_position), dtype=np.float64)
            mask = valid & (improve > thr)
            sel[mask] = cands[best_idx[mask]]
            sel = _apply_event_throttle(
                sel, benchmark_position=float(benchmark_position),
                cooldown_bars=int(cd), hold_bars=1,
            )
            sel_idx = np.argmin(np.abs(cands[None, :] - sel[:, None]), axis=1)
            util = actual_util[np.arange(len(actual_util)), sel_idx]
            ok = valid & np.isfinite(util)
            if not np.any(ok):
                continue
            active = float(np.mean(np.abs(sel[valid] - float(benchmark_position)) > 1e-12))
            score = float(np.mean(util[ok]))
            if active <= active_cap and score > best_score:
                best_score = score
                best = {"threshold": thr, "cooldown": int(cd), "val_score": score}
    return best


# ── danger guard ──────────────────────────────────────────────────────

def build_danger_guard(
    wm_trainer: WorldModelTrainer,
    z: np.ndarray,
    h: np.ndarray,
    quantile: float = 0.25,
) -> np.ndarray:
    """Allow de-risk actions only when WM predicts drawdown above quantile threshold."""
    aux = wm_trainer.predict_auxiliary_from_encoded(z, h)
    dd = aux.get("drawdown")
    if dd is None:
        return np.ones(len(z), dtype=bool)
    dd_h32 = _h32(dd, len(z))
    dd_h32 = np.maximum(np.asarray(dd_h32, dtype=np.float64), 0.0)
    finite = dd_h32[np.isfinite(dd_h32)]
    threshold = float(np.quantile(finite, quantile)) if len(finite) else 0.0
    return dd_h32 >= threshold


# ── fold runner ───────────────────────────────────────────────────────

def run_fold(
    split,
    features_df,
    raw_returns,
    cfg: dict,
    costs_cfg: dict,
    *,
    benchmark_position: float,
    checkpoint_dir: str,
    device: str,
    seed: int,
    seq_len: int,
) -> dict | None:
    try:
        dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
        fr = prepare_fold_runtime(
            fold_idx=split.fold_idx, checkpoint_dir=checkpoint_dir,
            ac_cfg=cfg.get("ac", {}),
            resume=False, start_from="test", stop_after="test",
        )
        if not fr["has_wm_ckpt"]:
            print(f"  [SKIP] fold={split.fold_idx} — no WM checkpoint")
            return None

        # load WM
        ensemble = build_ensemble(dataset.obs_dim, cfg)
        wm = WorldModelTrainer(ensemble, cfg, device=device)
        wm.load(fr["wm_path"])

        # encode
        enc_tr  = wm.encode_sequence(dataset.train_features, seq_len=seq_len)
        enc_val = wm.encode_sequence(dataset.val_features, seq_len=seq_len)
        enc_tst = wm.encode_sequence(dataset.test_features, seq_len=seq_len)

        uc = _unit_cost(costs_cfg)

        # WM-predicted utilities
        wm_val = compute_wm_utility(wm, enc_val["z"], enc_val["h"],
                                     candidates=CANDIDATES, benchmark_position=benchmark_position,
                                     unit_cost=uc, dd_penalty=DD_PENALTY, vol_penalty=VOL_PENALTY)
        wm_tst = compute_wm_utility(wm, enc_tst["z"], enc_tst["h"],
                                     candidates=CANDIDATES, benchmark_position=benchmark_position,
                                     unit_cost=uc, dd_penalty=DD_PENALTY, vol_penalty=VOL_PENALTY)

        # actual utilities for validation selection
        act_val, val_valid = _candidate_utilities(
            dataset.val_returns,
            candidates=CANDIDATES, horizon=HORIZON,
            benchmark_position=benchmark_position, unit_cost=uc,
            dd_penalty=DD_PENALTY, vol_penalty=0.0,
        )
        act_tst, tst_valid = _candidate_utilities(
            dataset.test_returns,
            candidates=CANDIDATES, horizon=HORIZON,
            benchmark_position=benchmark_position, unit_cost=uc,
            dd_penalty=DD_PENALTY, vol_penalty=0.0,
        )

        # guards
        danger_allow = build_danger_guard(wm, enc_tst["z"], enc_tst["h"])
        pullback = _pullback_no_fire_mask(dataset.test_returns)

        # validation selection
        best = select_best_config(
            wm_val, act_val, val_valid,
            candidates=CANDIDATES, benchmark_position=benchmark_position,
            cooldown_grid=COOLDOWN_GRID, active_cap=ACTIVE_CAP,
            min_threshold=MIN_THRESHOLD,
        )
        thr = float(best["threshold"]) if math.isfinite(float(best["threshold"])) else float("inf")
        cd  = int(best["cooldown"])

        # apply to test
        cands = np.asarray(CANDIDATES, dtype=np.float64)
        bench_idx = int(np.argmin(np.abs(cands - float(benchmark_position))))
        best_idx = np.argmax(wm_tst, axis=1)
        improve  = wm_tst[np.arange(len(wm_tst)), best_idx] - wm_tst[:, bench_idx]

        sel = np.full(len(wm_tst), float(benchmark_position), dtype=np.float64)
        active = tst_valid & (improve > thr) & danger_allow & (~pullback)
        sel[active] = cands[best_idx[active]]
        sel = _apply_event_throttle(
            sel, benchmark_position=float(benchmark_position),
            cooldown_bars=cd, hold_bars=1,
        )
        positions = _shift_for_execution(sel, benchmark_position)
        metrics, pnl = _backtest_positions(
            dataset.test_returns, positions,
            cfg=cfg, costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )

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


# ── report helpers ────────────────────────────────────────────────────

def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "NA"
    return f"{x:.{digits}f}" if math.isfinite(x) else "NA"


def _agg(vals: list[float], fn) -> float:
    f = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return float(fn(f)) if f else float("nan")


def write_md(path: str, *, checkpoint_dir: str, device: str, fold_ids: list[int],
             results: list[dict], aggregate: dict) -> None:
    lines = [
        "# Plan 3 WM Overlay Report",
        "",
        f"Checkpoint: `{checkpoint_dir}`",
        f"Device: `{device}`",
        f"Folds: `{', '.join(str(f) for f in fold_ids)}`",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for k in ["folds", "alpha_mean", "alpha_worst", "alpha_median",
              "maxdd_mean", "maxdd_worst", "sharpe_mean", "turnover_max", "pbo"]:
        lines.append(f"| {k} | {_fmt(aggregate[k])} |")

    lines.extend([
        "", "## Per-Fold", "",
        "| fold | AlphaEx | MaxDDΔ | SharpeΔ | turnover | flat | active | pullback | danger | thr | cd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in results:
        lines.append(
            f"| {r['fold']} | {_fmt(r['alpha_excess_pt'])} | {_fmt(r['maxdd_delta_pt'])} | "
            f"{_fmt(r['sharpe_delta'])} | {_fmt(r['turnover'])} | {_fmt(r['flat_rate'])} | "
            f"{r['active_events']} | {r['pullback_blocked']} | {r['danger_blocked']} | "
            f"{_fmt(r['threshold'])} | {r['cooldown']} |"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan3_wm_overlay")
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
    bp = float(cfg.get("reward", {}).get("benchmark_position", BENCHMARK_DEFAULT))
    dc = cfg.get("data", {})
    zw = cfg.get("normalization", {}).get("zscore_window_days", 60)

    ct = f"{dc['symbol']}_{dc['interval']}_{args.start}_{args.end}_z{zw}_v2"
    fdf, rr = load_training_features(
        symbol=dc["symbol"], interval=dc["interval"],
        start=args.start, end=args.end, zscore_window=zw,
        cache_dir="checkpoints/data_cache", cache_tag=ct,
        extra_series_mode=dc.get("extra_series_mode", "derived"),
        extra_series_include=dc.get("extra_series_include"),
        include_funding=bool(dc.get("include_funding", True)),
        include_oi=bool(dc.get("include_oi", True)),
        include_mark=bool(dc.get("include_mark", True)),
    )

    splits, _selected = select_wfo_splits(build_wfo_splits(fdf, dc), args.folds)
    sl = dc.get("seq_len", 64)

    results: list[dict] = []
    for s in splits:
        print(f"[Plan3WM] fold={s.fold_idx}")
        r = run_fold(s, fdf, rr, cfg, costs_cfg,
                     benchmark_position=bp, checkpoint_dir=args.checkpoint_dir,
                     device=dev, seed=args.seed, seq_len=sl)
        if r:
            results.append(r)

    if not results:
        print("[Plan3WM] No results")
        return

    alpha    = [x["alpha_excess_pt"] for x in results]
    maxdd    = [x["maxdd_delta_pt"] for x in results]
    sharpe   = [x["sharpe_delta"] for x in results]
    turnover = [x["turnover"] for x in results]

    aggregate = {
        "folds": len(results),
        "alpha_mean": _agg(alpha, np.mean), "alpha_worst": _agg(alpha, np.min),
        "alpha_median": _agg(alpha, np.median),
        "maxdd_mean": _agg(maxdd, np.mean), "maxdd_worst": _agg(maxdd, np.max),
        "sharpe_mean": _agg(sharpe, np.mean), "turnover_max": _agg(turnover, np.max),
        "pbo": compute_pbo([x["pnl"] for x in results]) if len(results) >= 2 else float("nan"),
    }

    folded = [s.fold_idx for s in splits]
    write_md(args.output_md, checkpoint_dir=args.checkpoint_dir, device=dev,
             fold_ids=folded, results=results, aggregate=aggregate)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"results": results, "aggregate": aggregate}, f, indent=2, default=str)
    print(f"[Plan3WM] Done → {args.output_md}")


if __name__ == "__main__":
    main()
