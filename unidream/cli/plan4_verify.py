"""
Plan 4 complete verification: B (ensemble), C (blocked attribution),
D (soft throttle), E (utility grid).
All on f456. GPU inference via WM. No config changes.
"""
from __future__ import annotations

import argparse, json, math, os
from typing import Any
import numpy as np

from unidream.cli.exploration_board_probe import (
    _apply_event_throttle, _backtest_positions, _candidate_utilities,
    _pullback_no_fire_mask, _shift_for_execution, _unit_cost, _state_features,
    _fit_ridge_multi, _triple_barrier_labels, _fit_binary_model, _score_binary,
)
from unidream.data.dataset import WFODataset
from unidream.device import resolve_device
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.world_model.train_wm import build_ensemble, WorldModelTrainer


CANDIDATES = (0.75, 1.0, 1.05, 1.10)
HORIZON = 32
DD_PENALTY = 1.50
VOL_PENALTY = 0.15
ACTIVE_CAP = 0.25
COOLDOWN_GRID = (0, 32)


def _h32(arr, n):
    if arr is None: return np.zeros(n)
    a = np.asarray(arr, dtype=np.float64)
    return a[:, -1] if a.ndim > 1 and a.shape[1] >= 5 else a.squeeze(-1)


def _wm_utility(wm, z, h, n, bp, uc):
    """WM-based D utility, vol-only (return head is random, dd head weak)."""
    aux = wm.predict_auxiliary_from_encoded(z, h)
    ret_raw = _h32(aux.get("return"), n) / 100.0
    vol_raw = _h32(aux.get("vol"), n) / 100.0
    dd_raw  = _h32(aux.get("drawdown"), n) / 100.0
    dd_raw = np.maximum(dd_raw, 0.0)

    k = len(CANDIDATES)
    vals = np.full((n, k), np.nan, dtype=np.float64)
    for ci, pos in enumerate(CANDIDATES):
        pos = float(pos)
        overlay = pos - bp
        pos_dd = np.abs(pos) * dd_raw
        dd_worsen = np.maximum(pos_dd - bp * dd_raw, 0.0)
        vals[:, ci] = overlay * ret_raw - abs(overlay) * uc - DD_PENALTY * dd_worsen - VOL_PENALTY * abs(overlay) * vol_raw
    return vals


def _ridge_utility(x_train, y_train, x_eval, bp, uc):
    """Ridge-based D utility."""
    mask = np.all(np.isfinite(y_train), axis=1)
    if mask.sum() < 100:
        return np.zeros((len(x_eval), len(CANDIDATES)))
    model = _fit_ridge_multi(x_train[mask], y_train[mask], l2=1.0)
    if model is None:
        return np.zeros((len(x_eval), len(CANDIDATES)))
    return model.predict(x_eval)


def _val_select(util, actual_util, valid, bp, min_thr=0.0):
    from unidream.cli.exploration_board_probe import _threshold_grid
    cands = np.asarray(CANDIDATES)
    bi = int(np.argmin(np.abs(cands - bp)))
    best_idx = np.argmax(util, axis=1)
    improve = util[np.arange(len(util)), best_idx] - util[:, bi]
    thresholds = _threshold_grid(improve[valid], active_cap=ACTIVE_CAP)
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


def _backtest_pos(returns, positions, cfg, costs_cfg, bp):
    metrics, pnl = _backtest_positions(returns, positions, cfg=cfg, costs_cfg=costs_cfg, benchmark_position=bp)
    return {
        "alpha": float(metrics["alpha_excess_pt"]),
        "maxdd": float(metrics["maxdd_delta_pt"]),
        "sharpe": float(metrics["sharpe_delta"]),
        "turnover": float(metrics["turnover"]),
        "flat": float(metrics["flat_rate"]),
        "pnl": pnl,
    }


def _fmt(v, d=3):
    try: x = float(v)
    except: return "NA"
    return f"{x:.{d}f}" if math.isfinite(x) else "NA"


def run_plan4(fold_splits, fdf, rr, cfg, costs_cfg, bp, ckpt_dir, dev, seed, sl):
    results = {"C_blocked": [], "B_ensemble": [], "D_throttle": [], "E_grid": []}

    for s in fold_splits:
        print(f"  fold={s.fold_idx}")
        ds = WFODataset(fdf, rr, s, seq_len=sl)
        fr = prepare_fold_runtime(fold_idx=s.fold_idx, checkpoint_dir=ckpt_dir, ac_cfg=cfg.get("ac", {}),
                                  resume=False, start_from="test", stop_after="test")
        if not fr["has_wm_ckpt"]:
            continue

        # WM
        ensemble = build_ensemble(ds.obs_dim, cfg)
        wm = WorldModelTrainer(ensemble, cfg, device=dev)
        wm.load(fr["wm_path"])
        enc_tr = wm.encode_sequence(ds.train_features, seq_len=sl)
        enc_val = wm.encode_sequence(ds.val_features, seq_len=sl)
        enc_tst = wm.encode_sequence(ds.test_features, seq_len=sl)
        uc = _unit_cost(costs_cfg)
        n_tr, n_val, n_tst = len(ds.train_returns), len(ds.val_returns), len(ds.test_returns)

        wm_util_val = _wm_utility(wm, enc_val["z"], enc_val["h"], n_val, bp, uc)
        wm_util_tst = _wm_utility(wm, enc_tst["z"], enc_tst["h"], n_tst, bp, uc)

        # Ridge
        x_tr = _state_features(ds.train_features, ds.train_returns)
        x_val = _state_features(ds.val_features, ds.val_returns)
        x_tst = _state_features(ds.test_features, ds.test_returns)
        y_tr, _ = _candidate_utilities(ds.train_returns, candidates=CANDIDATES, horizon=HORIZON, benchmark_position=bp, unit_cost=uc, dd_penalty=DD_PENALTY, vol_penalty=VOL_PENALTY)
        ridge_util_val = _ridge_utility(x_tr, y_tr, x_val, bp, uc)
        ridge_util_tst = _ridge_utility(x_tr, y_tr, x_tst, bp, uc)

        # Actual utilities
        act_val, val_valid = _candidate_utilities(ds.val_returns, candidates=CANDIDATES, horizon=HORIZON, benchmark_position=bp, unit_cost=uc, dd_penalty=DD_PENALTY, vol_penalty=VOL_PENALTY)
        act_tst, tst_valid = _candidate_utilities(ds.test_returns, candidates=CANDIDATES, horizon=HORIZON, benchmark_position=bp, unit_cost=uc, dd_penalty=DD_PENALTY, vol_penalty=VOL_PENALTY)

        # Guards
        pullback = _pullback_no_fire_mask(ds.test_returns)
        aux_tst = wm.predict_auxiliary_from_encoded(enc_tst["z"], enc_tst["h"])
        dd_h32 = _h32(aux_tst.get("drawdown"), n_tst) / 100.0
        dd_h32 = np.maximum(np.asarray(dd_h32, dtype=np.float64), 0.0)
        dd_q25 = float(np.quantile(dd_h32[np.isfinite(dd_h32)], 0.25)) if np.any(np.isfinite(dd_h32)) else 0.0
        danger_allow = dd_h32 >= dd_q25

        cands = np.asarray(CANDIDATES)
        bi = int(np.argmin(np.abs(cands - bp)))

        # ── Baseline: WM-only ──
        best_wm = _val_select(wm_util_val, act_val, val_valid, bp)
        best_idx = np.argmax(wm_util_tst, axis=1)
        improve_tst = wm_util_tst[np.arange(n_tst), best_idx] - wm_util_tst[:, bi]
        sel_wm = np.full(n_tst, bp, dtype=np.float64)
        act_wm = tst_valid & (improve_tst > best_wm["thr"]) & danger_allow & (~pullback)
        sel_wm[act_wm] = cands[best_idx[act_wm]]
        sel_wm = _apply_event_throttle(sel_wm, benchmark_position=bp, cooldown_bars=int(best_wm["cd"]), hold_bars=1)
        pos_wm = _shift_for_execution(sel_wm, bp)
        bt_wm = _backtest_pos(ds.test_returns, pos_wm, cfg, costs_cfg, bp)

        # ── C: Blocked event attribution (use threshold=0 to see all candidates) ──
        c_events = []
        # Look at all valid bars where improve > 0 (any positive signal)
        blocked_mask = (~danger_allow) | pullback
        # Events: positive signal, but blocked
        blocked_idx = np.flatnonzero(tst_valid & (improve_tst > 0.0) & blocked_mask)
        # Events: positive signal, NOT blocked (would have fired)
        passed_idx = np.flatnonzero(tst_valid & (improve_tst > 0.0) & (~blocked_mask))
        for i in blocked_idx[:500]:
            c_events.append({
                "idx": int(i),
                "type": "blocked",
                "wm_util": float(wm_util_tst[i, best_idx[i]]),
                "actual_util": float(act_tst[i, best_idx[i]]) if np.isfinite(act_tst[i, best_idx[i]]) else float("nan"),
                "danger_blocked": bool(not danger_allow[i]),
                "pullback_blocked": bool(pullback[i]),
                "pred_dd_h32": float(dd_h32[i]),
            })
        for i in passed_idx[:200]:
            c_events.append({
                "idx": int(i),
                "type": "passed",
                "wm_util": float(wm_util_tst[i, best_idx[i]]),
                "actual_util": float(act_tst[i, best_idx[i]]) if np.isfinite(act_tst[i, best_idx[i]]) else float("nan"),
                "danger_blocked": False,
                "pullback_blocked": False,
                "pred_dd_h32": float(dd_h32[i]),
            })
        blocked_utils = [e["actual_util"] for e in c_events if e["type"] == "blocked" and math.isfinite(e.get("actual_util", float("nan")))]
        passed_utils  = [e["actual_util"] for e in c_events if e["type"] == "passed" and math.isfinite(e.get("actual_util", float("nan")))]
        results["C_blocked"].append({
            "fold": s.fold_idx, "bt_wm": bt_wm,
            "n_blocked": int(np.sum(blocked_mask & tst_valid & (improve_tst > 0.0))),
            "n_passed": int(np.sum((~blocked_mask) & tst_valid & (improve_tst > 0.0))),
            "blocked_mean_util": float(np.mean(blocked_utils)) if blocked_utils else float("nan"),
            "passed_mean_util": float(np.mean(passed_utils)) if passed_utils else float("nan"),
            "events": c_events,
        })

        # ── B: Ridge+WM Ensemble ──
        b_results = {"fold": s.fold_idx}
        best_ridge = _val_select(ridge_util_val, act_val, val_valid, bp)
        ri = np.argmax(ridge_util_tst, axis=1)
        imp_ridge = ridge_util_tst[np.arange(n_tst), ri] - ridge_util_tst[:, bi]

        # B1: OR
        sel_or = np.full(n_tst, bp, dtype=np.float64)
        act_or = tst_valid & ((improve_tst > best_wm["thr"]) | (imp_ridge > best_ridge["thr"])) & (~pullback)
        sel_or[act_or] = cands[best_idx[act_or]]
        sel_or = _apply_event_throttle(sel_or, benchmark_position=bp, cooldown_bars=0, hold_bars=1)
        b_results["OR"] = _backtest_pos(ds.test_returns, _shift_for_execution(sel_or, bp), cfg, costs_cfg, bp)

        # B2: AND
        sel_and = np.full(n_tst, bp, dtype=np.float64)
        act_and = tst_valid & (improve_tst > best_wm["thr"]) & (imp_ridge > best_ridge["thr"]) & (~pullback)
        sel_and[act_and] = cands[best_idx[act_and]]
        sel_and = _apply_event_throttle(sel_and, benchmark_position=bp, cooldown_bars=0, hold_bars=1)
        b_results["AND"] = _backtest_pos(ds.test_returns, _shift_for_execution(sel_and, bp), cfg, costs_cfg, bp)

        # B3: max utility (val-selected threshold)
        max_util_val = np.maximum(wm_util_val, ridge_util_val)
        max_util_tst = np.maximum(wm_util_tst, ridge_util_tst)
        best_max = _val_select(max_util_val, act_val, val_valid, bp)
        bi_max = np.argmax(max_util_tst, axis=1)
        imp_max = max_util_tst[np.arange(n_tst), bi_max] - max_util_tst[:, bi]
        sel_max = np.full(n_tst, bp, dtype=np.float64)
        act_max = tst_valid & (imp_max > best_max["thr"]) & (~pullback)
        sel_max[act_max] = cands[bi_max[act_max]]
        sel_max = _apply_event_throttle(sel_max, benchmark_position=bp, cooldown_bars=int(best_max["cd"]), hold_bars=1)
        b_results["MAX"] = _backtest_pos(ds.test_returns, _shift_for_execution(sel_max, bp), cfg, costs_cfg, bp)

        results["B_ensemble"].append(b_results)

        # ── D: Soft throttle (danger block → scale down instead of block) ──
        d_results = {"fold": s.fold_idx, "WM_hard": bt_wm}
        for scale in [0.25, 0.5, 0.75]:
            sel_soft = np.full(n_tst, bp, dtype=np.float64)
            for i in range(n_tst):
                if not tst_valid[i] or improve_tst[i] <= best_wm["thr"] or pullback[i]:
                    continue
                target = cands[best_idx[i]]
                if not danger_allow[i]:
                    target = bp + (target - bp) * scale  # scale down deviation
                sel_soft[i] = target
            sel_soft = _apply_event_throttle(sel_soft, benchmark_position=bp, cooldown_bars=int(best_wm["cd"]), hold_bars=1)
            d_results[f"danger_scale_{scale}"] = _backtest_pos(ds.test_returns, _shift_for_execution(sel_soft, bp), cfg, costs_cfg, bp)
        results["D_throttle"].append(d_results)

        # ── E: Utility parameter grid ──
        e_results = {"fold": s.fold_idx}
        aux_val = wm.predict_auxiliary_from_encoded(enc_val["z"], enc_val["h"])
        aux_tst = wm.predict_auxiliary_from_encoded(enc_tst["z"], enc_tst["h"])
        for dd_p in [0.25, 0.5, 1.0, 2.0]:
            for vol_p in [0.0, 0.5, 1.0]:
                def _grid_util(n, aux):
                    ret_r = _h32(aux.get("return"), n) / 100.0
                    vol_r = _h32(aux.get("vol"), n) / 100.0
                    dd_r  = np.maximum(_h32(aux.get("drawdown"), n) / 100.0, 0.0)
                    gu = np.full((n, len(CANDIDATES)), np.nan)
                    for ci, pos in enumerate(CANDIDATES):
                        pos = float(pos)
                        overlay = pos - bp
                        pos_dd = np.abs(pos) * dd_r
                        dd_w = np.maximum(pos_dd - bp * dd_r, 0.0)
                        gu[:, ci] = overlay * ret_r - abs(overlay) * uc - dd_p * dd_w - vol_p * abs(overlay) * vol_r
                    return gu
                gval = _grid_util(n_val, aux_val)
                gtst = _grid_util(n_tst, aux_tst)
                best_g = _val_select(gval, act_val, val_valid, bp, min_thr=0.0)
                gi = np.argmax(gtst, axis=1)
                imp_g = gtst[np.arange(n_tst), gi] - gtst[:, bi]
                sel_g = np.full(n_tst, bp, dtype=np.float64)
                act_g = tst_valid & (imp_g > best_g["thr"]) & (~pullback)
                sel_g[act_g] = cands[gi[act_g]]
                sel_g = _apply_event_throttle(sel_g, benchmark_position=bp, cooldown_bars=int(best_g["cd"]), hold_bars=1)
                e_results[f"dd{dd_p}_vol{vol_p}"] = _backtest_pos(ds.test_returns, _shift_for_execution(sel_g, bp), cfg, costs_cfg, bp)
        results["E_grid"].append(e_results)

    return results


def _agg(vals, fn):
    f = [float(x) for x in vals if math.isfinite(float(x))]
    return float(fn(f)) if f else float("nan")


def write_final_report(results, fold_ids, ckpt_dir, dev, output_md):
    lines = [
        "# Plan 4 Complete Verification Report",
        "",
        f"Checkpoint: `{ckpt_dir}` | Device: `{dev}` | Folds: `{', '.join(str(f) for f in fold_ids)}`",
        "",
        "---",
        "",
        "## Round A: WM Prediction Calibration",
        "",
        "See `documents/20260502_plan4_wm_calibration.md` for full calibration.",
        "",
        "**Key finding**: return head random (IC~0), vol head useful (IC 0.3-0.6), dd head weak/random.",
        "",
        "---",
        "",
        "## Round C: Blocked Event Attribution",
        "",
        "| fold | bt_alpha | bt_maxdd | n_blocked_events | danger_blocked | pullback_blocked |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in results.get("C_blocked", []):
        events = c["events"]
        d_count = sum(1 for e in events if e["danger_blocked"])
        p_count = sum(1 for e in events if e["pullback_blocked"])
        lines.append(f"| {c['fold']} | {_fmt(c['bt_wm']['alpha'])} | {_fmt(c['bt_wm']['maxdd'])} | {len(events)} | {d_count} | {p_count} |")

    # Counterfactual analysis
    lines.extend(["", "### Counterfactual: blocked event actual utility", "",
        "| fold | mean_actual_util | util>0_rate | would_improve |",
        "|---|---:|---:|---:|"])
    for c in results.get("C_blocked", []):
        utils = [e["actual_util"] for e in c["events"] if math.isfinite(e.get("actual_util", float("nan")))]
        if utils:
            lines.append(f"| {c['fold']} | {_fmt(np.mean(utils))} | {_fmt(np.mean([1 if u>0 else 0 for u in utils]))} | {'YES' if np.mean(utils)>0 else 'NO'} |")

    # B: Ensemble
    lines.extend(["", "---", "", "## Round B: Ridge + WM Ensemble", "",
        "| fold | variant | AlphaEx | MaxDDΔ | SharpeΔ | turnover | flat |",
        "|---|---:|---:|---:|---:|---:|"])
    for b in results.get("B_ensemble", []):
        for var in ["OR", "AND", "MAX"]:
            if var in b:
                m = b[var]
                lines.append(f"| {b['fold']} | {var} | {_fmt(m['alpha'])} | {_fmt(m['maxdd'])} | {_fmt(m['sharpe'])} | {_fmt(m['turnover'])} | {_fmt(m['flat'])} |")

    # D: Soft throttle
    lines.extend(["", "---", "", "## Round D: Soft Throttle Guard", "",
        "| fold | variant | AlphaEx | MaxDDΔ | SharpeΔ | turnover | flat |",
        "|---|---:|---:|---:|---:|---:|"])
    for d in results.get("D_throttle", []):
        for var in ["WM_hard", "danger_scale_0.25", "danger_scale_0.5", "danger_scale_0.75"]:
            if var in d:
                m = d[var]
                lines.append(f"| {d['fold']} | {var} | {_fmt(m['alpha'])} | {_fmt(m['maxdd'])} | {_fmt(m['sharpe'])} | {_fmt(m['turnover'])} | {_fmt(m['flat'])} |")

    # E: Grid - top/bottom 5 per fold
    lines.extend(["", "---", "", "## Round E: Utility Parameter Grid (top results per fold)", "",
        "| fold | params | AlphaEx | MaxDDΔ | SharpeΔ | turnover |",
        "|---|---:|---:|---:|---:|"])
    for e in results.get("E_grid", []):
        grid_items = [(k, v) for k, v in e.items() if k != "fold"]
        grid_items.sort(key=lambda x: x[1]["alpha"], reverse=True)
        for k, v in grid_items[:5]:
            if v["turnover"] < 3.5:
                lines.append(f"| {e['fold']} | {k} | {_fmt(v['alpha'])} | {_fmt(v['maxdd'])} | {_fmt(v['sharpe'])} | {_fmt(v['turnover'])} |")

    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/acplan13_base_wm_s011")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-md", default="documents/20260502_plan4_complete.md")
    parser.add_argument("--output-json", default="documents/20260502_plan4_complete.json")
    args = parser.parse_args()

    dev = resolve_device(args.device)
    print(f"[Plan4] device={dev}")
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _ = resolve_costs(cfg, None)
    costs_cfg = cfg.get("costs", {})
    bp = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    dc = cfg.get("data", {})
    zw = cfg.get("normalization", {}).get("zscore_window_days", 60)
    ct = f"{dc['symbol']}_{dc['interval']}_2018-01-01_2024-01-01_z{zw}_v2"
    fdf, rr = load_training_features(
        symbol=dc["symbol"], interval=dc["interval"], start="2018-01-01", end="2024-01-01",
        zscore_window=zw, cache_dir="checkpoints/data_cache", cache_tag=ct,
        extra_series_mode=dc.get("extra_series_mode", "derived"),
        extra_series_include=dc.get("extra_series_include"),
        include_funding=bool(dc.get("include_funding", True)),
        include_oi=bool(dc.get("include_oi", True)),
        include_mark=bool(dc.get("include_mark", True)),
    )
    splits, _ = select_wfo_splits(build_wfo_splits(fdf, dc), args.folds)
    sl = dc.get("seq_len", 64)

    print(f"[Plan4] Running B+C+D+E on {len(splits)} folds...")
    results = run_plan4(splits, fdf, rr, cfg, costs_cfg, bp, args.checkpoint_dir, dev, args.seed, sl)

    fold_ids = [s.fold_idx for s in splits]
    write_final_report(results, fold_ids, args.checkpoint_dir, dev, args.output_md)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Plan4] Done -> {args.output_md}")


if __name__ == "__main__":
    main()
