"""
Plan 4 Round A: WM prediction calibration (fixed).
Divides WM predictions by target_scale to get actual-scale values.
Checks all horizon columns, correlation vs realized h32.
"""
from __future__ import annotations

import argparse, json, math, os
import numpy as np

from unidream.data.dataset import WFODataset
from unidream.device import resolve_device
from unidream.experiments.fold_runtime import prepare_fold_runtime
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.world_model.train_wm import build_ensemble, WorldModelTrainer


def _realized_h32(returns: np.ndarray, horizon: int = 32):
    r = np.asarray(returns, dtype=np.float64)
    h = max(1, int(horizon))
    n = len(r)
    ret_sum = np.full(n, np.nan)
    vol = np.full(n, np.nan)
    dd  = np.full(n, np.nan)
    for i in range(n - h):
        w = r[i + 1: i + 1 + h]
        ret_sum[i] = np.sum(w)
        vol[i] = np.std(w) * np.sqrt(h)
        dd[i]  = max(0.0, -np.min(np.cumsum(w)))
    return {"return_h32": ret_sum, "vol_h32": vol, "dd_h32": dd}


def _stats(pred: np.ndarray, real: np.ndarray, label: str) -> dict:
    mask = np.isfinite(pred) & np.isfinite(real)
    if mask.sum() < 20:
        return {"label": label, "n": 0, "error": "too few valid samples"}
    p, r = pred[mask], real[mask]
    pstd = float(np.std(p))
    rstd = float(np.std(r))
    if pstd < 1e-15:
        return {"label": label, "n": int(mask.sum()), "error": "pred constant", "pred_mean": float(np.mean(p)), "pred_std": pstd, "real_mean": float(np.mean(r)), "real_std": rstd}
    from scipy.stats import spearmanr
    return {
        "label": label,
        "n": int(mask.sum()),
        "pearson": float(np.corrcoef(p, r)[0, 1]),
        "spearman": float(spearmanr(p, r)[0]),
        "pred_mean": float(np.mean(p)), "real_mean": float(np.mean(r)),
        "pred_std": pstd, "real_std": rstd,
        "scale_ratio": rstd / pstd,
        "sign_acc": float(np.mean(np.sign(p) == np.sign(r))),
        "top10_real": float(np.mean(r[np.argsort(p)[-max(1, len(p)//10):]])),
        "top1_real": float(np.mean(r[np.argsort(p)[-max(1, len(p)//100):]])),
    }


def _fmt(v, d=3):
    try:
        x = float(v)
    except: return "NA"
    return f"{x:.{d}f}" if math.isfinite(x) else "NA"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints/acplan13_base_wm_s011")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-md", default="documents/20260502_plan4_wm_calibration.md")
    parser.add_argument("--output-json", default="documents/20260502_plan4_wm_calibration.json")
    args = parser.parse_args()

    dev = resolve_device(args.device)
    print(f"[Plan4Calib] device={dev}")
    set_seed(args.seed)

    cfg = load_config(args.config)
    cfg, _ = resolve_costs(cfg, None)
    wm_cfg = cfg.get("world_model", {})
    return_target_scale = float(wm_cfg.get("return_target_scale", 1.0))
    risk_target_scale   = float(wm_cfg.get("risk_target_scale", 1.0))
    return_horizons = [int(h) for h in wm_cfg.get("return_horizons", [32])]
    risk_horizons   = [int(h) for h in wm_cfg.get("risk_horizons", [32])]
    print(f"[Plan4Calib] return_target_scale={return_target_scale} risk_target_scale={risk_target_scale}")
    print(f"[Plan4Calib] return_horizons={return_horizons} risk_horizons={risk_horizons}")

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

    all_stats = []
    for s in splits:
        print(f"[Plan4Calib] fold={s.fold_idx}")
        ds = WFODataset(fdf, rr, s, seq_len=sl)
        fr = prepare_fold_runtime(fold_idx=s.fold_idx, checkpoint_dir=args.checkpoint_dir, ac_cfg=cfg.get("ac", {}),
                                  resume=False, start_from="test", stop_after="test")
        if not fr["has_wm_ckpt"]:
            continue
        ensemble = build_ensemble(ds.obs_dim, cfg)
        wm = WorldModelTrainer(ensemble, cfg, device=dev)
        wm.load(fr["wm_path"])
        enc = wm.encode_sequence(ds.test_features, seq_len=sl)
        aux = wm.predict_auxiliary_from_encoded(enc["z"], enc["h"])
        real = _realized_h32(ds.test_returns, 32)
        n = len(ds.test_returns)

        # Return head: each horizon, divide by return_target_scale
        ret_raw = aux.get("return")
        if ret_raw is not None:
            ret_raw = np.asarray(ret_raw, dtype=np.float64) / return_target_scale
            for hi, h in enumerate(return_horizons):
                if hi < ret_raw.shape[1]:
                    all_stats.append(_stats(ret_raw[:, hi], real["return_h32"], f"fold{s.fold_idx}_return_h{h}"))

        # Vol head: each horizon, divide by risk_target_scale
        vol_raw = aux.get("vol")
        if vol_raw is not None:
            vol_raw = np.asarray(vol_raw, dtype=np.float64) / risk_target_scale
            for hi, h in enumerate(risk_horizons):
                if hi < vol_raw.shape[1]:
                    all_stats.append(_stats(vol_raw[:, hi], real["vol_h32"], f"fold{s.fold_idx}_vol_h{h}"))

        # DD head: each horizon, divide by risk_target_scale
        dd_raw = aux.get("drawdown")
        if dd_raw is not None:
            dd_raw = np.asarray(dd_raw, dtype=np.float64) / risk_target_scale
            for hi, h in enumerate(risk_horizons):
                if hi < dd_raw.shape[1]:
                    all_stats.append(_stats(dd_raw[:, hi], real["dd_h32"], f"fold{s.fold_idx}_dd_h{h}"))

    # Write MD
    lines = [
        "# Plan 4 Round A: WM Prediction Calibration",
        "",
        f"Checkpoint: `{args.checkpoint_dir}` | Device: `{dev}`",
        f"return_target_scale={return_target_scale} risk_target_scale={risk_target_scale}",
        f"return_horizons={return_horizons} risk_horizons={risk_horizons}",
        "",
        "## All Signals",
        "",
        "| label | n | pearson | spearman | scale_ratio | sign_acc | top10 | top1 | pred_mean | real_mean | pred_std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in all_stats:
        if s.get("error"):
            lines.append(f"| {s['label']} | {s.get('n','?')} | **{s['error']}** | — | — | — | — | — | {_fmt(s.get('pred_mean'))} | {_fmt(s.get('real_mean'))} | {_fmt(s.get('pred_std'))} |")
        else:
            lines.append(
                f"| {s['label']} | {s['n']} | {_fmt(s['pearson'])} | {_fmt(s['spearman'])} | "
                f"{_fmt(s['scale_ratio'])} | {_fmt(s['sign_acc'])} | {_fmt(s['top10_real'])} | "
                f"{_fmt(s['top1_real'])} | {_fmt(s['pred_mean'])} | {_fmt(s['real_mean'])} | {_fmt(s['pred_std'])} |"
            )

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"[Plan4Calib] Done -> {args.output_md}")


if __name__ == "__main__":
    main()
