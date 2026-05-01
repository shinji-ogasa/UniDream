from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

from unidream.cli.ac_fire_timing_probe import ProbeRun, _load_actor_for_run, _parse_run
from unidream.cli.fire_control_label_probe import (
    _average_precision,
    _format_float,
    _ridge_predict,
    _roc_auc,
    _summary_metrics,
)
from unidream.cli.fire_type_cluster_probe import FireTypeProbeConfig, _build_fire_frame
from unidream.experiments.policy_fire import predict_with_policy_flags
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


@dataclass(frozen=True)
class DangerProbeConfig:
    horizon: int
    fire_eps: float
    train_frac: float
    ridge_l2: float
    min_eval_fire: int
    max_z_dim: int
    rel_vol_window: int
    mdd_rel_threshold: float
    post_dd_quantile: float


def _chronological_fire_split(n: int, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    if n < 20:
        return np.arange(0, 0, dtype=np.int64), np.arange(0, 0, dtype=np.int64)
    cut = max(10, min(n - 10, int(n * float(train_frac))))
    return np.arange(cut, dtype=np.int64), np.arange(cut, n, dtype=np.int64)


def _feature_matrix(frame: dict[str, np.ndarray | dict]) -> tuple[np.ndarray, list[str]]:
    cols: list[tuple[str, np.ndarray]] = []
    for name in (
        "position",
        "no_adapter",
        "delta",
        "current_drawdown_depth",
        "underwater_duration",
        "trailing_return_16",
        "trailing_return_32",
        "trailing_slope_16",
        "trailing_slope_32",
        "trailing_vol_64",
        "equity_slope_32",
        "benchmark_relative_equity_slope_32",
        "fire_pnl",
    ):
        cols.append((name, np.asarray(frame[name], dtype=np.float64)))

    regime = np.asarray(frame["regime"], dtype=object)
    for name in sorted(set(regime.tolist())):
        cols.append((name, (regime == name).astype(np.float64)))

    names = [name for name, _arr in cols]
    x = np.column_stack([arr for _name, arr in cols]).astype(np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), names


def _zscore(train_score: np.ndarray, eval_score: np.ndarray) -> np.ndarray:
    mean = float(np.mean(train_score))
    std = float(np.std(train_score))
    if std < 1e-8:
        std = 1.0
    return (np.asarray(eval_score, dtype=np.float64) - mean) / std


def _binary_metrics(y_eval: np.ndarray, score_eval: np.ndarray) -> dict:
    y = np.asarray(y_eval, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
            "auc": float("nan"),
            "pr_auc": float("nan"),
        }
    order = np.argsort(-score_eval, kind="mergesort")
    top = order[: max(1, int(np.ceil(len(order) * 0.10)))]
    return {
        "positive_rate": float(np.mean(y)),
        "auc": _roc_auc(y, score_eval),
        "pr_auc": _average_precision(y, score_eval),
        "top10_positive_rate": float(np.mean(y[top])),
    }


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) else float("nan")


def _selection_summary(frame: dict[str, np.ndarray | dict], row_idx: np.ndarray) -> dict:
    rows = np.asarray(row_idx, dtype=np.int64)
    if len(rows) == 0:
        return {"count": 0}
    fire_type = np.asarray(frame["fire_type"], dtype=object)[rows]
    type_counts = {str(name): int(np.sum(fire_type == name)) for name in sorted(set(fire_type.tolist()))}
    return {
        "count": int(len(rows)),
        "type_counts": type_counts,
        "fire_advantage": _safe_mean(np.asarray(frame["fire_advantage_h"], dtype=np.float64)[rows]),
        "post_fire_dd_contribution": _safe_mean(np.asarray(frame["post_fire_dd_contribution"], dtype=np.float64)[rows]),
        "future_mdd_overlap_rate": float(np.mean(np.asarray(frame["future_mdd_overlap"], dtype=np.float64)[rows] > 0.5)),
        "global_mdd_overlap_rate": float(np.mean(np.asarray(frame["global_mdd_overlap"], dtype=np.float64)[rows] > 0.5)),
        "fire_pnl": float(np.sum(np.asarray(frame["fire_pnl"], dtype=np.float64)[rows])),
    }


def _take_top(eval_rows: np.ndarray, score: np.ndarray, frac: float) -> np.ndarray:
    if len(eval_rows) == 0:
        return eval_rows
    count = max(1, int(np.ceil(len(eval_rows) * float(frac))))
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    return np.asarray(eval_rows, dtype=np.int64)[order[:count]]


def _run_ids(frame: dict[str, np.ndarray | dict]) -> np.ndarray:
    idx = np.asarray(frame["idx"], dtype=np.int64)
    out = np.zeros(len(idx), dtype=np.int64)
    run = 0
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            run += 1
        out[i] = run
    return out


def _take_top_runs(frame: dict[str, np.ndarray | dict], eval_rows: np.ndarray, score: np.ndarray, frac: float, *, agg: str) -> np.ndarray:
    rows = np.asarray(eval_rows, dtype=np.int64)
    if len(rows) == 0:
        return rows
    run_ids = _run_ids(frame)
    score = np.asarray(score, dtype=np.float64)
    run_scores = []
    for rid in sorted(set(run_ids[rows].tolist())):
        local = np.flatnonzero(run_ids[rows] == rid)
        vals = score[local]
        val = float(np.max(vals)) if agg == "max" else float(np.mean(vals))
        run_scores.append((rid, val))
    if not run_scores:
        return rows[:0]
    count = max(1, int(np.ceil(len(run_scores) * float(frac))))
    selected_runs = {rid for rid, _score in sorted(run_scores, key=lambda x: x[1], reverse=True)[:count]}
    return rows[np.asarray([rid in selected_runs for rid in run_ids[rows]], dtype=bool)]


def _guard_positions(
    positions: np.ndarray,
    no_adapter: np.ndarray,
    frame: dict[str, np.ndarray | dict],
    selected_rows: np.ndarray,
    scale: float,
) -> np.ndarray:
    guarded = np.asarray(positions, dtype=np.float64).copy()
    bar_idx = np.asarray(frame["idx"], dtype=np.int64)[np.asarray(selected_rows, dtype=np.int64)]
    delta = np.asarray(positions, dtype=np.float64) - np.asarray(no_adapter, dtype=np.float64)
    guarded[bar_idx] = np.asarray(no_adapter, dtype=np.float64)[bar_idx] + float(scale) * delta[bar_idx]
    return guarded


def _fit_scores(x: np.ndarray, frame: dict[str, np.ndarray | dict], train_rows: np.ndarray, eval_rows: np.ndarray, cfg: DangerProbeConfig) -> dict:
    targets = {
        "pre_dd_type": (np.asarray(frame["fire_type"], dtype=object) == "pre_dd_danger_fire").astype(np.float64),
        "future_mdd": (np.asarray(frame["future_mdd_overlap"], dtype=np.float64) > 0.5).astype(np.float64),
        "global_mdd": (np.asarray(frame["global_mdd_overlap"], dtype=np.float64) > 0.5).astype(np.float64),
        "post_dd": np.asarray(frame["post_fire_dd_contribution"], dtype=np.float64),
        "adv": np.asarray(frame["fire_advantage_h"], dtype=np.float64),
    }
    train_scores: dict[str, np.ndarray] = {}
    eval_scores: dict[str, np.ndarray] = {}
    for name, target in targets.items():
        train_score, eval_score = _ridge_predict(x[train_rows], target[train_rows], x[eval_rows], l2=cfg.ridge_l2)
        train_scores[name] = train_score
        eval_scores[name] = eval_score

    z_pre = _zscore(train_scores["pre_dd_type"], eval_scores["pre_dd_type"])
    z_future = _zscore(train_scores["future_mdd"], eval_scores["future_mdd"])
    z_global = _zscore(train_scores["global_mdd"], eval_scores["global_mdd"])
    z_post = _zscore(train_scores["post_dd"], eval_scores["post_dd"])
    z_adv = _zscore(train_scores["adv"], eval_scores["adv"])
    combined = {
        "predd_only": z_pre,
        "future_mdd_only": z_future,
        "global_mdd_only": z_global,
        "danger_minus_adv": z_pre + z_future + z_global + z_post - 0.5 * z_adv,
        "danger_strict": 1.5 * z_pre + z_future + z_global + z_post - z_adv,
    }
    metrics = {
        "pre_dd_type": _binary_metrics(targets["pre_dd_type"][eval_rows], eval_scores["pre_dd_type"]),
        "future_mdd": _binary_metrics(targets["future_mdd"][eval_rows], eval_scores["future_mdd"]),
        "global_mdd": _binary_metrics(targets["global_mdd"][eval_rows], eval_scores["global_mdd"]),
    }
    return {"targets": targets, "eval_scores": eval_scores, "combined": combined, "metrics": metrics}


def _evaluate_run_fold(
    *,
    run: ProbeRun,
    split,
    features_df,
    raw_returns,
    cfg: dict,
    probe_cfg: DangerProbeConfig,
    device: str,
) -> dict:
    payload = _load_actor_for_run(
        run=run,
        split=split,
        features_df=features_df,
        raw_returns=raw_returns,
        cfg=cfg,
        device=device,
    )
    actor = payload["actor"]
    enc = payload["enc_test"]
    returns = np.asarray(payload["test_returns"], dtype=np.float64)
    regime = payload["test_regime_probs"]
    advantage = payload["test_advantage_values"]
    costs_cfg = payload["costs_cfg"]
    benchmark_position = float(payload["benchmark_position"])
    positions = actor.predict_positions(
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=device,
    )
    no_adapter = predict_with_policy_flags(
        actor,
        enc["z"],
        enc["h"],
        regime_np=regime,
        advantage_np=advantage,
        device=device,
        use_floor=bool(getattr(actor, "use_benchmark_exposure_floor", False)),
        use_adapter=False,
    )
    t = min(len(returns), len(positions), len(no_adapter))
    returns = np.asarray(returns[:t], dtype=np.float64)
    positions = np.asarray(positions[:t], dtype=np.float64)
    no_adapter = np.asarray(no_adapter[:t], dtype=np.float64)
    frame = _build_fire_frame(
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        regime=regime,
        costs_cfg=costs_cfg,
        cfg=FireTypeProbeConfig(
            horizon=probe_cfg.horizon,
            fire_eps=probe_cfg.fire_eps,
            max_z_dim=probe_cfg.max_z_dim,
            rel_vol_window=probe_cfg.rel_vol_window,
            min_type_count=20,
            mdd_rel_threshold=probe_cfg.mdd_rel_threshold,
            post_dd_quantile=probe_cfg.post_dd_quantile,
        ),
    )
    x, feature_names = _feature_matrix(frame)
    train_rows, eval_rows = _chronological_fire_split(len(x), probe_cfg.train_frac)
    base_summary = _summary_metrics(
        positions=positions,
        no_adapter=no_adapter,
        returns=returns,
        cfg=payload["cfg"],
        costs_cfg=costs_cfg,
        benchmark_position=benchmark_position,
        fire_eps=probe_cfg.fire_eps,
    )
    if len(eval_rows) < probe_cfg.min_eval_fire:
        return {
            "label": run.label,
            "fold": int(split.fold_idx),
            "summary": base_summary,
            "feature_names": feature_names,
            "error": f"not enough eval fires: {len(eval_rows)}",
        }
    scores = _fit_scores(x, frame, train_rows, eval_rows, probe_cfg)

    variants: dict[str, dict] = {}
    for score_name, score in scores["combined"].items():
        for frac in (0.10, 0.20, 0.30):
            selected_rows = _take_top(eval_rows, score, frac)
            for scale in (0.0, 0.5):
                name = f"{score_name}_top{int(frac * 100)}_scale{scale:g}"
                guarded = _guard_positions(positions, no_adapter, frame, selected_rows, scale)
                variants[name] = {
                    "selection": _selection_summary(frame, selected_rows),
                    "summary": _summary_metrics(
                        positions=guarded,
                        no_adapter=no_adapter,
                        returns=returns,
                        cfg=payload["cfg"],
                        costs_cfg=costs_cfg,
                        benchmark_position=benchmark_position,
                        fire_eps=probe_cfg.fire_eps,
                    ),
                }
            for agg in ("mean", "max"):
                selected_run_rows = _take_top_runs(frame, eval_rows, score, frac, agg=agg)
                for scale in (0.0, 0.5):
                    name = f"{score_name}_run{agg}_top{int(frac * 100)}_scale{scale:g}"
                    guarded = _guard_positions(positions, no_adapter, frame, selected_run_rows, scale)
                    variants[name] = {
                        "selection": _selection_summary(frame, selected_run_rows),
                        "summary": _summary_metrics(
                            positions=guarded,
                            no_adapter=no_adapter,
                            returns=returns,
                            cfg=payload["cfg"],
                            costs_cfg=costs_cfg,
                            benchmark_position=benchmark_position,
                            fire_eps=probe_cfg.fire_eps,
                        ),
                    }
    fire_type = np.asarray(frame["fire_type"], dtype=object)
    oracle_specs = {
        "oracle_predd": {"pre_dd_danger_fire"},
        "oracle_predd_noise": {"pre_dd_danger_fire", "noise_fire"},
        "oracle_predd_mdd": {"pre_dd_danger_fire", "mdd_inside_profitable_fire"},
        "oracle_not_lowrisk": {"pre_dd_danger_fire", "noise_fire", "mdd_inside_profitable_fire"},
    }
    for spec_name, types in oracle_specs.items():
        selected_rows = eval_rows[np.asarray([fire_type[i] in types for i in eval_rows], dtype=bool)]
        for scale in (0.0, 0.5):
            name = f"{spec_name}_scale{scale:g}"
            guarded = _guard_positions(positions, no_adapter, frame, selected_rows, scale)
            variants[name] = {
                "selection": _selection_summary(frame, selected_rows),
                "summary": _summary_metrics(
                    positions=guarded,
                    no_adapter=no_adapter,
                    returns=returns,
                    cfg=payload["cfg"],
                    costs_cfg=costs_cfg,
                    benchmark_position=benchmark_position,
                    fire_eps=probe_cfg.fire_eps,
                ),
            }
    return {
        "label": run.label,
        "mode": "ac" if run.use_ac else "bc",
        "checkpoint_dir": run.checkpoint_dir,
        "fold": int(split.fold_idx),
        "summary": base_summary,
        "fire_rows": {
            "total": int(len(x)),
            "train": int(len(train_rows)),
            "eval": int(len(eval_rows)),
        },
        "score_metrics": scores["metrics"],
        "variants": variants,
    }


def _variant_table(records: list[dict]) -> list[dict]:
    names = sorted({name for rec in records for name in rec.get("variants", {})})
    rows = []
    for name in names:
        vals = []
        ok = True
        for rec in records:
            if name not in rec.get("variants", {}):
                ok = False
                break
            vals.append(rec["variants"][name]["summary"])
        if not ok:
            continue
        rows.append(
            {
                "variant": name,
                "alpha_mean": float(np.mean([v["alpha_excess_pt"] for v in vals])),
                "sharpe_mean": float(np.mean([v["sharpe_delta"] for v in vals])),
                "maxdd_mean": float(np.mean([v["maxdd_delta_pt"] for v in vals])),
                "turnover_mean": float(np.mean([v["turnover"] for v in vals])),
                "long_max": float(np.max([v["long"] for v in vals])),
                "short_max": float(np.max([v["short"] for v in vals])),
                "per_fold": vals,
            }
        )
    rows.sort(key=lambda r: (r["maxdd_mean"] <= 0.0, r["sharpe_mean"], r["alpha_mean"]), reverse=True)
    return rows


def _write_markdown(path: str, *, records: list[dict], args: argparse.Namespace, probe_cfg: DangerProbeConfig) -> None:
    rows = _variant_table(records)
    lines = [
        "# Plan15-D Danger Fire Score Probe",
        "",
        "## Setup",
        "",
        f"- config: `{args.config}`",
        f"- folds: `{args.folds}`",
        f"- horizon: `{probe_cfg.horizon}`",
        "- scope: diagnostic score/guard simulation only; no production guard, WM head, AC unlock, or config adoption.",
        "",
        "## Base Policy",
        "",
        "| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | fire | fire_pnl | eval fires |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        s = rec["summary"]
        fire_rows = rec.get("fire_rows", {})
        lines.append(
            f"| {rec['label']} | {rec['fold']} | {_format_float(s['alpha_excess_pt'], 2, True)} | "
            f"{_format_float(s['sharpe_delta'], 3, True)} | {_format_float(s['maxdd_delta_pt'], 2, True)} | "
            f"{_format_float(s['turnover'], 2)} | {s['long']:.1%} | {s['short']:.1%} | "
            f"{s['fire_rate']:.1%}/{s['fire_count']} | {_format_float(s['fire_pnl'], 4, True)} | "
            f"{fire_rows.get('eval', 0)} |"
        )

    lines += [
        "",
        "## Score Predictability",
        "",
        "| run | fold | target | positive | AUC | PR-AUC | top10 positive |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for rec in records:
        for target, m in rec.get("score_metrics", {}).items():
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {target} | {_format_float(m.get('positive_rate', float('nan')))} | "
                f"{_format_float(m.get('auc', float('nan')))} | {_format_float(m.get('pr_auc', float('nan')))} | "
                f"{_format_float(m.get('top10_positive_rate', float('nan')))} |"
            )

    lines += [
        "",
        "## Variant Mean",
        "",
        "| variant | AlphaEx mean | SharpeD mean | MaxDDD mean | turnover mean | long max | short max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:20]:
        lines.append(
            f"| {row['variant']} | {_format_float(row['alpha_mean'], 2, True)} | "
            f"{_format_float(row['sharpe_mean'], 3, True)} | {_format_float(row['maxdd_mean'], 2, True)} | "
            f"{_format_float(row['turnover_mean'], 2)} | {row['long_max']:.1%} | {row['short_max']:.1%} |"
        )

    lines += [
        "",
        "## Best Variant By Fold",
        "",
        "| variant | fold | AlphaEx | SharpeD | MaxDDD | turnover | selected | selected adv | selected fire_pnl | selected preDD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows[:5]:
        name = row["variant"]
        for rec in records:
            variant = rec.get("variants", {}).get(name)
            if not variant:
                continue
            s = variant["summary"]
            sel = variant["selection"]
            predd = sel.get("type_counts", {}).get("pre_dd_danger_fire", 0)
            lines.append(
                f"| {name} | {rec['fold']} | {_format_float(s['alpha_excess_pt'], 2, True)} | "
                f"{_format_float(s['sharpe_delta'], 3, True)} | {_format_float(s['maxdd_delta_pt'], 2, True)} | "
                f"{_format_float(s['turnover'], 2)} | {sel.get('count', 0)} | "
                f"{_format_float(sel.get('fire_advantage', float('nan')), 5, True)} | "
                f"{_format_float(sel.get('fire_pnl', float('nan')), 4, True)} | {predd} |"
            )

    best = rows[0] if rows else None
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if best is None:
        lines.append("- No variants were evaluated.")
    else:
        lines.append(
            f"- Best ranked variant: `{best['variant']}` with mean AlphaEx "
            f"{best['alpha_mean']:+.2f}, SharpeD {best['sharpe_mean']:+.3f}, MaxDDD {best['maxdd_mean']:+.2f}."
        )
        lines.append("- A usable guard needs MaxDDD <= 0, turnover <= 3.5, long <= 3%, short = 0%, and no severe AlphaEx loss across folds.")
        lines.append("- If every variant still has MaxDDD > 0 or kills alpha, do not proceed to Plan16 adoption.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.fire_danger_score_probe")
    parser.add_argument("--config", default="configs/trading_wm_control_headonly.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="checkpoints/data_cache")
    parser.add_argument("--run", action="append", required=True, help="label=checkpoint_dir[@ac_file][:ac|:bc]")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--fire-eps", type=float, default=1e-6)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--ridge-l2", type=float, default=1e-2)
    parser.add_argument("--min-eval-fire", type=int, default=50)
    parser.add_argument("--max-z-dim", type=int, default=128)
    parser.add_argument("--rel-vol-window", type=int, default=64)
    parser.add_argument("--mdd-rel-threshold", type=float, default=0.5)
    parser.add_argument("--post-dd-quantile", type=float, default=0.8)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg)
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
        cache_dir=args.cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, _selected = select_wfo_splits(build_wfo_splits(features_df, data_cfg), str(args.folds))
    if not splits:
        raise RuntimeError(f"No folds matched: {args.folds}")
    probe_cfg = DangerProbeConfig(
        horizon=int(args.horizon),
        fire_eps=float(args.fire_eps),
        train_frac=float(args.train_frac),
        ridge_l2=float(args.ridge_l2),
        min_eval_fire=int(args.min_eval_fire),
        max_z_dim=int(args.max_z_dim),
        rel_vol_window=int(args.rel_vol_window),
        mdd_rel_threshold=float(args.mdd_rel_threshold),
        post_dd_quantile=float(args.post_dd_quantile),
    )
    runs = [_parse_run(spec) for spec in args.run]
    records: list[dict] = []
    for split in splits:
        for run in runs:
            try:
                print(f"[Plan15-D] run={run.label} fold={split.fold_idx} checkpoint={run.checkpoint_dir}")
                records.append(
                    _evaluate_run_fold(
                        run=run,
                        split=split,
                        features_df=features_df,
                        raw_returns=raw_returns,
                        cfg=cfg,
                        probe_cfg=probe_cfg,
                        device=args.device,
                    )
                )
            except FileNotFoundError:
                if not args.skip_missing:
                    raise
                print(f"[Plan15-D] skip missing run={run.label} fold={split.fold_idx}")

    serializable = {
        "config": args.config,
        "folds": args.folds,
        "runs": [run.__dict__ for run in runs],
        "probe": probe_cfg.__dict__,
        "records": records,
        "variant_table": _variant_table(records),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, allow_nan=True)
    _write_markdown(args.output_md, records=records, args=args, probe_cfg=probe_cfg)
    print(f"[Plan15-D] wrote {args.output_md}")


if __name__ == "__main__":
    main()
