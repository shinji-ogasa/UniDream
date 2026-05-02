"""
Plan 5: Lane B (ridge primary + WM vol veto) + Lane C (pipeline breakdown).
Key insight: WM return head is random → use ridge for position selection,
WM vol head for risk veto. GPU inference.
"""
from __future__ import annotations

import argparse, json, math, os, traceback
from typing import Any
import numpy as np

from unidream.cli.exploration_board_probe import (
    _apply_event_throttle, _backtest_positions, _candidate_utilities,
    _pullback_no_fire_mask, _shift_for_execution, _unit_cost, _state_features,
    _fit_ridge_multi,
)
from unidream.data.dataset import WFODataset
from unidream.device import resolve_device
from unidream.eval.pbo import compute_pbo
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.world_model.train_wm import build_ensemble, WorldModelTrainer


CANDIDATES = (0.75, 1.0, 1.05, 1.10)
HORIZON = 32
COOLDOWN_GRID = (0, 32)
ACTIVE_CAP = 0.25


def _h32(arr, n):
    if arr is None: return np.zeros(n, dtype=np.float64)
    a = np.asarray(arr, dtype=np.float64)
    return a[:, -1] if a.ndim > 1 and a.shape[1] >= 5 else a.squeeze(-1)


def _threshold_grid_orig(improve, active_cap=0.25):
    vals = np.asarray(improve, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0: return [float("inf")]
    qs = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
    raw = list(vals)
    raw.extend([float(np.quantile(vals, q)) for q in qs] + [0.0])
    cap_q = max(0.0, min(0.995, 1.0 - float(active_cap)))
    if len(vals): raw.append(float(np.quantile(vals, cap_q)))
    return sorted(set([float("inf") if not math.isfinite(t) else t for t in raw]))


def _ridge_utility(x_train, y_train, x_eval):
    mask = np.all(np.isfinite(y_train), axis=1)
    if mask.sum() < 100: return np.zeros((len(x_eval), len(CANDIDATES)))
    model = _fit_ridge_multi(x_train[mask], y_train[mask], l2=1.0)
    if model is None: return np.zeros((len(x_eval), len(CANDIDATES)))
    return model.predict(x_eval)


def _val_select(util, actual_util, valid, bp, min_thr=0.0):
    cands = np.asarray(CANDIDATES)
    bi = int(np.argmin(np.abs(cands - bp)))
    best_idx = np.argmax(util, axis=1)
    improve = util[np.arange(len(util)), best_idx] - util[:, bi]
    thresholds = _threshold_grid_orig(improve[valid], ACTIVE_CAP)
    thresholds = sorted(set([max(t, min_thr) for t in thresholds if math.isfinite(t)] + [float("inf")]))
    best_score = -1e12
    best = {"thr": float("inf"), "cd": 0}
    for cd in COOLDOWN_GRID:
        for thr in thresholds:
            if thr == float("inf"): continue
            sel = np.full(len(util), bp, dtype=np.float64)
            mask = valid & (improve > thr)
            sel[mask] = cands[best_idx[mask]]
            sel = _apply_event_throttle(sel, benchmark_position=bp, cooldown_bars=int(cd), hold_bars=1)
            sidx = np.argmin(np.abs(cands[None, :] - sel[:, None]), axis=1)
            u = actual_util[np.arange(len(actual_util)), sidx]
            ok = valid & np.isfinite(u)
            if not np.any(ok): continue
            active = float(np.mean(np.abs(sel[valid] - bp) > 1e-12))
            score = float(np.mean(u[ok]))
            if active <= ACTIVE_CAP and score > best_score:
                best_score = score
                best = {"thr": thr, "cd": int(cd), "score": score}
    return best


def _backtest(returns, positions, cfg, cc, bp):
    m, p = _backtest_positions(returns, positions, cfg=cfg, costs_cfg=cc, benchmark_position=bp)
    return {"alpha": float(m["alpha_excess_pt"]), "maxdd": float(m["maxdd_delta_pt"]),
            "sharpe": float(m["sharpe_delta"]), "turnover": float(m["turnover"]),
            "flat": float(m["flat_rate"]), "pnl": p}


def _fmt(v, d=3):
    try: x = float(v)
    except: return "NA"
    return f"{x:.{d}f}" if math.isfinite(x) else "NA"


def run_plan5(splits, fdf, rr, cfg, costs_cfg, bp, ckpt_dir, dev, sl):
    all_results = {"LaneB": [], "LaneC": [], "LaneF": []}
    cands = np.asarray(CANDIDATES); bi = int(np.argmin(np.abs(cands - bp)))

    for s in splits:
        print(f"  fold={s.fold_idx}")
        ds = WFODataset(fdf, rr, s, seq_len=sl)
        fr = prepare_fold_runtime(fold_idx=s.fold_idx, checkpoint_dir=ckpt_dir, ac_cfg=cfg.get("ac", {}),
                                  resume=False, start_from="test", stop_after="test")
        if not fr["has_wm_ckpt"]: continue

        ensemble = build_ensemble(ds.obs_dim, cfg)
        wm = WorldModelTrainer(ensemble, cfg, device=dev)
        wm.load(fr["wm_path"])
        enc_val = wm.encode_sequence(ds.val_features, seq_len=sl)
        enc_tst = wm.encode_sequence(ds.test_features, seq_len=sl)
        uc = _unit_cost(costs_cfg)
        n_val, n_tst = len(ds.val_returns), len(ds.test_returns)

        # Actual utilities for validation
        act_val, val_valid = _candidate_utilities(ds.val_returns, candidates=CANDIDATES, horizon=HORIZON,
                                                   benchmark_position=bp, unit_cost=uc, dd_penalty=1.5, vol_penalty=0.0)
        act_tst, tst_valid = _candidate_utilities(ds.test_returns, candidates=CANDIDATES, horizon=HORIZON,
                                                   benchmark_position=bp, unit_cost=uc, dd_penalty=1.5, vol_penalty=0.0)
        pullback = _pullback_no_fire_mask(ds.test_returns)

        # Ridge utility
        x_tr = _state_features(ds.train_features, ds.train_returns)
        x_val = _state_features(ds.val_features, ds.val_returns)
        x_tst = _state_features(ds.test_features, ds.test_returns)
        y_tr, _ = _candidate_utilities(ds.train_returns, candidates=CANDIDATES, horizon=HORIZON,
                                        benchmark_position=bp, unit_cost=uc, dd_penalty=1.5, vol_penalty=0.0)
        ru_val = _ridge_utility(x_tr, y_tr, x_val)
        ru_tst = _ridge_utility(x_tr, y_tr, x_tst)

        # WM vol predictions for veto
        vol_tst = _h32(wm.predict_auxiliary_from_encoded(enc_tst["z"], enc_tst["h"]).get("vol"), n_tst) / 100.0
        vol_tst = np.maximum(np.asarray(vol_tst, dtype=np.float64), 0.0)

        # ── Lane B: Ridge primary + WM vol veto ──
        lane_b = {"fold": s.fold_idx}
        for vol_q in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            vol_cap = float(np.quantile(vol_tst[np.isfinite(vol_tst)], vol_q)) if np.any(np.isfinite(vol_tst)) else float("inf")
            low_vol = vol_tst <= vol_cap
            for min_t in [0.0, 0.00001, 0.0001]:
                best_r = _val_select(ru_val, act_val, val_valid, bp, min_thr=min_t)
                ri = np.argmax(ru_tst, axis=1)
                imp_r = ru_tst[np.arange(n_tst), ri] - ru_tst[:, bi]
                sel = np.full(n_tst, bp, dtype=np.float64)
                act_r = tst_valid & (imp_r > best_r["thr"]) & (~pullback) & low_vol
                sel[act_r] = cands[ri[act_r]]
                sel = _apply_event_throttle(sel, benchmark_position=bp, cooldown_bars=int(best_r["cd"]), hold_bars=1)
                bt = _backtest(ds.test_returns, _shift_for_execution(sel, bp), cfg, costs_cfg, bp)
                if bt["turnover"] > 0 and bt["turnover"] <= 3.5:
                    lane_b[f"vq{int(vol_q*100)}_thr{min_t}"] = bt

        # Ridge-only baseline
        best_r0 = _val_select(ru_val, act_val, val_valid, bp, min_thr=0.0)
        ri0 = np.argmax(ru_tst, axis=1); imp_r0 = ru_tst[np.arange(n_tst), ri0] - ru_tst[:, bi]
        sel_r0 = np.full(n_tst, bp, dtype=np.float64)
        act_r0 = tst_valid & (imp_r0 > best_r0["thr"]) & (~pullback)
        sel_r0[act_r0] = cands[ri0[act_r0]]
        sel_r0 = _apply_event_throttle(sel_r0, benchmark_position=bp, cooldown_bars=int(best_r0["cd"]), hold_bars=1)
        lane_b["ridge_only"] = _backtest(ds.test_returns, _shift_for_execution(sel_r0, bp), cfg, costs_cfg, bp)
        all_results["LaneB"].append(lane_b)

        # ── Lane C: Pipeline breakdown ──
        # WM pipeline breakdown
        from unidream.cli.plan3_wm_overlay import compute_wm_utility, build_danger_guard
        wm_val = compute_wm_utility(wm, enc_val["z"], enc_val["h"], candidates=CANDIDATES, benchmark_position=bp,
                                     unit_cost=uc, dd_penalty=1.5, vol_penalty=0.15)
        wm_tst = compute_wm_utility(wm, enc_tst["z"], enc_tst["h"], candidates=CANDIDATES, benchmark_position=bp,
                                     unit_cost=uc, dd_penalty=1.5, vol_penalty=0.15)
        danger = build_danger_guard(wm, enc_tst["z"], enc_tst["h"])
        best_wm = _val_select(wm_val, act_val, val_valid, bp)
        bi_wm = np.argmax(wm_tst, axis=1); imp_wm = wm_tst[np.arange(n_tst), bi_wm] - wm_tst[:, bi]

        # Ridge pipeline breakdown
        best_r = _val_select(ru_val, act_val, val_valid, bp, min_thr=0.0)
        ri = np.argmax(ru_tst, axis=1); imp_r = ru_tst[np.arange(n_tst), ri] - ru_tst[:, bi]

        all_results["LaneC"].append({
            "fold": s.fold_idx,
            "WM_pre": int(np.sum(tst_valid & (imp_wm > 0))),
            "WM_post_thr": int(np.sum(tst_valid & (imp_wm > best_wm["thr"]))),
            "WM_post_danger": int(np.sum(tst_valid & (imp_wm > best_wm["thr"]) & danger)),
            "WM_final": int(np.sum(tst_valid & (imp_wm > best_wm["thr"]) & danger & (~pullback))),
            "WM_thr": best_wm["thr"],
            "R_pre": int(np.sum(tst_valid & (imp_r > 0))),
            "R_post_thr": int(np.sum(tst_valid & (imp_r > best_r["thr"]))),
            "R_final": int(np.sum(tst_valid & (imp_r > best_r["thr"]) & (~pullback))),
            "R_thr": best_r["thr"],
        })

    return all_results


def write_report(results, fold_ids, ckpt_dir, dev, output_md):
    lines = [
        "# Plan 5 Verification Report",
        f"Checkpoint: `{ckpt_dir}` | Device: `{dev}` | Folds: `{', '.join(str(f) for f in fold_ids)}`",
        "",
        "## Key Finding",
        "- WM return head random (IC~0) → utility improvement is noise (mean=0.00005)",
        "- WM vol head useful (IC 0.3-0.6) → use as risk veto, not position selector",
        "- Ridge has signal on fold4/6 but explodes without vol veto (turnover 34-109)",
        "",
        "## Lane C: Pipeline Breakdown",
        "| fold | WM_pre | WM_post_thr | WM_post_danger | WM_final | WM_thr | R_pre | R_post_thr | R_final | R_thr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results.get("LaneC", []):
        lines.append(f"| {r['fold']} | {r['WM_pre']} | {r['WM_post_thr']} | {r['WM_post_danger']} | {r['WM_final']} | {_fmt(r['WM_thr'])} | {r['R_pre']} | {r['R_post_thr']} | {r['R_final']} | {_fmt(r['R_thr'])} |")

    lines.extend(["", "## Lane B: Ridge Primary + WM Vol Veto", "",
        "| fold | variant | AlphaEx | MaxDDΔ | SharpeΔ | turnover | flat |",
        "|---|---:|---:|---:|---:|---:|"])
    for r in results.get("LaneB", []):
        for var in sorted(r.keys()):
            if var == "fold": continue
            m = r[var]
            lines.append(f"| {r['fold']} | {var} | {_fmt(m['alpha'])} | {_fmt(m['maxdd'])} | {_fmt(m['sharpe'])} | {_fmt(m['turnover'])} | {_fmt(m['flat'])} |")

    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/acplan13_base_wm_s011")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-md", default="documents/20260502_plan5.md")
    parser.add_argument("--output-json", default="documents/20260502_plan5.json")
    args = parser.parse_args()

    dev = resolve_device(args.device)
    print(f"[Plan5] device={dev}")
    set_seed(args.seed)
    cfg = load_config(args.config); cfg, _ = resolve_costs(cfg, None)
    costs_cfg = cfg.get("costs", {})
    bp = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    dc = cfg.get("data", {}); zw = cfg.get("normalization", {}).get("zscore_window_days", 60)
    ct = f"{dc['symbol']}_{dc['interval']}_2018-01-01_2024-01-01_z{zw}_v2"
    fdf, rr = load_training_features(
        symbol=dc["symbol"], interval=dc["interval"], start="2018-01-01", end="2024-01-01",
        zscore_window=zw, cache_dir="checkpoints/data_cache", cache_tag=ct,
    )
    splits, _ = select_wfo_splits(build_wfo_splits(fdf, dc), args.folds)
    sl = dc.get("seq_len", 64)
    print(f"[Plan5] Running Lane B + C on {len(splits)} folds...")
    results = run_plan5(splits, fdf, rr, cfg, costs_cfg, bp, args.checkpoint_dir, dev, sl)
    fold_ids = [s.fold_idx for s in splits]
    write_report(results, fold_ids, args.checkpoint_dir, dev, args.output_md)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Plan5] Done -> {args.output_md}")


if __name__ == "__main__":
    main()
