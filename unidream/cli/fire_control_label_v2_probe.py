from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from unidream.cli.ac_fire_dd_guard_probe import _maxdd_interval, _pnl_bar
from unidream.cli.ac_fire_timing_probe import ProbeRun, _load_actor_for_run, _parse_run
from unidream.experiments.policy_fire import predict_with_policy_flags
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.cli.fire_control_label_probe import (
    _average_precision,
    _build_feature_matrix,
    _chronological_split,
    _drawdown_magnitude_from_pnl,
    _format_float,
    _future_drawdown_worsening,
    _online_state,
    _ridge_predict,
    _roc_auc,
    _rolling_past_sum,
    _rolling_past_vol,
    _summary_metrics,
    _window_pnl,
)


@dataclass(frozen=True)
class V2ProbeConfig:
    horizons: tuple[int, ...]
    primary_horizon: int
    fire_eps: float
    train_frac: float
    ridge_l2: float
    max_z_dim: int
    rel_vol_window: int
    dd_rel_ks: tuple[float, ...]
    dd_quantiles: tuple[float, ...]


def _linear_slope(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=np.float64)
    x = x - float(np.mean(x))
    denom = float(np.sum(x * x))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * (arr - float(np.mean(arr)))) / denom)


def _future_end_drawdown(
    *,
    returns: np.ndarray,
    positions: np.ndarray,
    start: int,
    horizon: int,
    costs_cfg: dict,
    equity_before: float,
    peak_before: float,
) -> tuple[float, float]:
    equity = float(equity_before)
    peak = float(peak_before)
    min_dd = equity / max(peak, 1e-12) - 1.0
    prev_pos = float(positions[start - 1]) if start > 0 else 0.0
    end = min(len(returns), start + int(horizon))
    for i in range(start, end):
        pnl = _pnl_bar(float(returns[i]), float(positions[i]), prev_pos, costs_cfg)
        equity *= float(math.exp(pnl))
        peak = max(peak, equity)
        min_dd = min(min_dd, equity / max(peak, 1e-12) - 1.0)
        prev_pos = float(positions[i])
    end_dd = equity / max(peak, 1e-12) - 1.0
    return float(end_dd), float(min_dd)


def _mdd_mask(returns: np.ndarray, positions: np.ndarray, costs_cfg: dict) -> np.ndarray:
    t = min(len(returns), len(positions))
    pnl = np.zeros(t, dtype=np.float64)
    prev_pos = 0.0
    for i in range(t):
        pnl[i] = _pnl_bar(float(returns[i]), float(positions[i]), prev_pos, costs_cfg)
        prev_pos = float(positions[i])
    equity = np.exp(np.cumsum(pnl))
    interval = _maxdd_interval(equity)
    mask = np.zeros(t, dtype=bool)
    if interval["trough"] >= interval["peak"]:
        mask[interval["peak"] : interval["trough"] + 1] = True
    return mask


def _build_v2_labels(
    *,
    returns: np.ndarray,
    positions: np.ndarray,
    no_adapter: np.ndarray,
    costs_cfg: dict,
    cfg: V2ProbeConfig,
) -> dict[int, dict[str, np.ndarray]]:
    t = min(len(returns), len(positions), len(no_adapter))
    current_dd, _underwater, equity_before, peak_before = _online_state(returns[:t], positions[:t], costs_cfg)
    roll_vol = _rolling_past_vol(returns[:t], cfg.rel_vol_window)
    past_ret_16 = _rolling_past_sum(returns[:t], 16)
    out: dict[int, dict[str, np.ndarray]] = {}
    for horizon in cfg.horizons:
        valid = np.zeros(t, dtype=bool)
        fire_advantage = np.full(t, np.nan, dtype=np.float64)
        harm_margin = np.full(t, np.nan, dtype=np.float64)
        dd_deepen = np.full(t, np.nan, dtype=np.float64)
        dd_rel = np.full(t, np.nan, dtype=np.float64)
        recovery_slope = np.full(t, np.nan, dtype=np.float64)
        post_trough_momentum = np.full(t, np.nan, dtype=np.float64)
        underwater_recovery = np.full(t, np.nan, dtype=np.float64)
        relative_recovery_slope = np.full(t, np.nan, dtype=np.float64)
        for i in range(0, max(0, t - int(horizon) + 1)):
            pnl_on = _window_pnl(returns, positions, i, horizon, costs_cfg)
            pnl_off = _window_pnl(returns, no_adapter, i, horizon, costs_cfg)
            if len(pnl_on) < int(horizon) or len(pnl_off) < int(horizon):
                continue
            valid[i] = True
            adv_path = np.cumsum(pnl_on - pnl_off)
            cum_on = np.cumsum(pnl_on)
            price_path = np.cumsum(np.asarray(returns[i : i + int(horizon)], dtype=np.float64))
            fire_advantage[i] = float(adv_path[-1])
            harm_margin[i] = _drawdown_magnitude_from_pnl(pnl_on) - _drawdown_magnitude_from_pnl(pnl_off)
            deepen = _future_drawdown_worsening(
                returns=returns,
                positions=positions,
                start=i,
                horizon=horizon,
                costs_cfg=costs_cfg,
                equity_before=float(equity_before[i]),
                peak_before=float(peak_before[i]),
                current_drawdown=float(current_dd[i]),
            )
            dd_deepen[i] = deepen
            denom = max(float(roll_vol[i]) * math.sqrt(float(horizon)), 1e-8)
            dd_rel[i] = deepen / denom
            recovery_slope[i] = _linear_slope(cum_on) / denom
            relative_recovery_slope[i] = _linear_slope(adv_path) / denom
            trough_idx = int(np.argmin(price_path)) if len(price_path) else 0
            rebound = float(price_path[-1] - price_path[trough_idx]) if len(price_path) else 0.0
            post_trough_momentum[i] = rebound if float(past_ret_16[i]) < 0.0 else 0.5 * rebound
            end_dd, _min_dd = _future_end_drawdown(
                returns=returns,
                positions=positions,
                start=i,
                horizon=horizon,
                costs_cfg=costs_cfg,
                equity_before=float(equity_before[i]),
                peak_before=float(peak_before[i]),
            )
            underwater_recovery[i] = float(end_dd - current_dd[i])
        out[int(horizon)] = {
            "valid": valid,
            "fire_advantage": fire_advantage,
            "harm_margin": harm_margin,
            "dd_deepen": dd_deepen,
            "dd_rel": dd_rel,
            "recovery_slope": recovery_slope,
            "relative_recovery_slope": relative_recovery_slope,
            "post_trough_momentum": post_trough_momentum,
            "underwater_recovery": underwater_recovery,
        }
    return out


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def _take_fraction(eval_idx: np.ndarray, score: np.ndarray, frac: float, *, high: bool) -> np.ndarray:
    if len(eval_idx) == 0:
        return eval_idx
    count = max(1, int(math.ceil(len(eval_idx) * float(frac))))
    order = np.argsort(score, kind="mergesort")
    if high:
        order = order[::-1]
    return eval_idx[order[:count]]


def _selection_stats(
    *,
    selected_idx: np.ndarray,
    labels: dict[str, np.ndarray],
    mdd_mask: np.ndarray,
) -> dict:
    idx = np.asarray(selected_idx, dtype=int)
    if len(idx) == 0:
        return {
            "count": 0,
            "fire_advantage": float("nan"),
            "harm_margin": float("nan"),
            "dd_deepen": float("nan"),
            "dd_rel": float("nan"),
            "recovery_slope": float("nan"),
            "relative_recovery_slope": float("nan"),
            "post_trough_momentum": float("nan"),
            "underwater_recovery": float("nan"),
            "harm_nonpositive_rate": float("nan"),
            "mdd_rate": float("nan"),
        }
    return {
        "count": int(len(idx)),
        "fire_advantage": _safe_mean(labels["fire_advantage"][idx]),
        "harm_margin": _safe_mean(labels["harm_margin"][idx]),
        "dd_deepen": _safe_mean(labels["dd_deepen"][idx]),
        "dd_rel": _safe_mean(labels["dd_rel"][idx]),
        "recovery_slope": _safe_mean(labels["recovery_slope"][idx]),
        "relative_recovery_slope": _safe_mean(labels["relative_recovery_slope"][idx]),
        "post_trough_momentum": _safe_mean(labels["post_trough_momentum"][idx]),
        "underwater_recovery": _safe_mean(labels["underwater_recovery"][idx]),
        "harm_nonpositive_rate": float(np.mean(labels["harm_margin"][idx] <= 0.0)),
        "mdd_rate": float(np.mean(mdd_mask[idx])),
    }


def _zscore_from_train(train_score: np.ndarray, eval_score: np.ndarray) -> np.ndarray:
    mean = float(np.mean(train_score))
    std = float(np.std(train_score))
    if std < 1e-8:
        std = 1.0
    return (np.asarray(eval_score, dtype=np.float64) - mean) / std


def _binary_metrics_from_score(y_eval: np.ndarray, score_eval: np.ndarray) -> dict:
    y = np.asarray(y_eval, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
            "auc": float("nan"),
            "pr_auc": float("nan"),
        }
    return {
        "positive_rate": float(np.mean(y)),
        "auc": _roc_auc(y, score_eval),
        "pr_auc": _average_precision(y, score_eval),
    }


def _evaluate_horizon_v2(
    *,
    x: np.ndarray,
    labels: dict[str, np.ndarray],
    sample_mask: np.ndarray,
    mdd_mask: np.ndarray,
    cfg: V2ProbeConfig,
) -> dict:
    valid = sample_mask.copy()
    for key in (
        "fire_advantage",
        "harm_margin",
        "dd_rel",
        "recovery_slope",
        "relative_recovery_slope",
        "post_trough_momentum",
        "underwater_recovery",
    ):
        valid &= np.isfinite(labels[key])
    train_idx, eval_idx = _chronological_split(valid, cfg.train_frac)
    if len(train_idx) == 0 or len(eval_idx) == 0:
        return {"samples": int(np.sum(valid)), "eval_samples": 0}

    targets = {
        "adv": labels["fire_advantage"],
        "harm": labels["harm_margin"],
        "dd": labels["dd_rel"],
        "recovery": labels["recovery_slope"],
        "rel_recovery": labels["relative_recovery_slope"],
        "post_trough": labels["post_trough_momentum"],
        "underwater": labels["underwater_recovery"],
    }
    train_scores: dict[str, np.ndarray] = {}
    eval_scores: dict[str, np.ndarray] = {}
    for name, target in targets.items():
        train_score, eval_score = _ridge_predict(
            x[train_idx],
            target[train_idx],
            x[eval_idx],
            l2=cfg.ridge_l2,
        )
        train_scores[name] = train_score
        eval_scores[name] = eval_score

    adv_top10 = _take_fraction(eval_idx, eval_scores["adv"], 0.10, high=True)
    adv_top20 = _take_fraction(eval_idx, eval_scores["adv"], 0.20, high=True)
    adv_bottom10 = _take_fraction(eval_idx, eval_scores["adv"], 0.10, high=False)

    harm_low10 = _take_fraction(eval_idx, eval_scores["harm"], 0.10, high=False)
    harm_high10 = _take_fraction(eval_idx, eval_scores["harm"], 0.10, high=True)

    dd_metrics: dict[str, dict] = {}
    dd_train = labels["dd_rel"][train_idx]
    dd_eval = labels["dd_rel"][eval_idx]
    for k in cfg.dd_rel_ks:
        dd_metrics[f"k_{k:g}"] = _binary_metrics_from_score(dd_eval > float(k), eval_scores["dd"])
    for q in cfg.dd_quantiles:
        threshold = float(np.quantile(dd_train, float(q)))
        dd_metrics[f"q_{int(round(q * 100))}"] = {
            "threshold": threshold,
            **_binary_metrics_from_score(dd_eval > threshold, eval_scores["dd"]),
        }

    recovery_metrics = {
        "recovery_slope": _binary_metrics_from_score(labels["recovery_slope"][eval_idx] > 0.0, eval_scores["recovery"]),
        "relative_recovery_slope": _binary_metrics_from_score(
            labels["relative_recovery_slope"][eval_idx] > 0.0,
            eval_scores["rel_recovery"],
        ),
        "post_trough_momentum": _binary_metrics_from_score(
            labels["post_trough_momentum"][eval_idx] > 0.0,
            eval_scores["post_trough"],
        ),
        "underwater_recovery": _binary_metrics_from_score(
            labels["underwater_recovery"][eval_idx] > 0.0,
            eval_scores["underwater"],
        ),
    }

    z_adv = _zscore_from_train(train_scores["adv"], eval_scores["adv"])
    z_harm = _zscore_from_train(train_scores["harm"], eval_scores["harm"])
    z_dd = _zscore_from_train(train_scores["dd"], eval_scores["dd"])
    z_rec = _zscore_from_train(train_scores["recovery"], eval_scores["recovery"])
    z_rel_rec = _zscore_from_train(train_scores["rel_recovery"], eval_scores["rel_recovery"])
    z_underwater = _zscore_from_train(train_scores["underwater"], eval_scores["underwater"])
    combined_scores = {
        "A_adv_only": z_adv,
        "B_adv_minus_harm": z_adv - z_harm,
        "C_adv_minus_dd_plus_recovery": z_adv - z_dd + 0.5 * z_rec + 0.5 * z_underwater,
        "D_all": z_adv - z_harm - z_dd + 0.5 * z_rec + 0.5 * z_rel_rec + 0.5 * z_underwater,
    }
    combined = {}
    for name, score in combined_scores.items():
        top10 = _take_fraction(eval_idx, score, 0.10, high=True)
        top20 = _take_fraction(eval_idx, score, 0.20, high=True)
        combined[name] = {
            "top10": _selection_stats(selected_idx=top10, labels=labels, mdd_mask=mdd_mask),
            "top20": _selection_stats(selected_idx=top20, labels=labels, mdd_mask=mdd_mask),
        }

    corr = float("nan")
    adv_eval = labels["fire_advantage"][eval_idx]
    if np.std(eval_scores["adv"]) > 1e-12 and np.std(adv_eval) > 1e-12:
        corr = float(np.corrcoef(eval_scores["adv"], adv_eval)[0, 1])

    return {
        "samples": int(np.sum(valid)),
        "train_samples": int(len(train_idx)),
        "eval_samples": int(len(eval_idx)),
        "advantage": {
            "corr": corr,
            "eval_mean": _safe_mean(labels["fire_advantage"][eval_idx]),
            "top10": _selection_stats(selected_idx=adv_top10, labels=labels, mdd_mask=mdd_mask),
            "top20": _selection_stats(selected_idx=adv_top20, labels=labels, mdd_mask=mdd_mask),
            "bottom10": _selection_stats(selected_idx=adv_bottom10, labels=labels, mdd_mask=mdd_mask),
            "top10_bottom10_spread": (
                _safe_mean(labels["fire_advantage"][adv_top10])
                - _safe_mean(labels["fire_advantage"][adv_bottom10])
            ),
        },
        "recovery": recovery_metrics,
        "drawdown_relative": dd_metrics,
        "harm_ranking": {
            "low_harm_top10": _selection_stats(selected_idx=harm_low10, labels=labels, mdd_mask=mdd_mask),
            "high_harm_top10": _selection_stats(selected_idx=harm_high10, labels=labels, mdd_mask=mdd_mask),
        },
        "combined_scores": combined,
    }


def _evaluate_run_fold_v2(
    *,
    run: ProbeRun,
    split,
    features_df,
    raw_returns,
    cfg: dict,
    probe_cfg: V2ProbeConfig,
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
    returns = returns[:t]
    positions = np.asarray(positions[:t], dtype=np.float64)
    no_adapter = np.asarray(no_adapter[:t], dtype=np.float64)
    fire = np.abs(positions - no_adapter) > float(probe_cfg.fire_eps)
    x = _build_feature_matrix(
        enc=enc,
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        fire=fire,
        regime=regime,
        advantage=advantage,
        costs_cfg=costs_cfg,
        max_z_dim=probe_cfg.max_z_dim,
    )
    labels = _build_v2_labels(
        returns=returns,
        positions=positions,
        no_adapter=no_adapter,
        costs_cfg=costs_cfg,
        cfg=probe_cfg,
    )
    mdd_mask = _mdd_mask(returns, positions, costs_cfg)
    horizons = {}
    for horizon, rec in labels.items():
        horizons[str(horizon)] = _evaluate_horizon_v2(
            x=x,
            labels=rec,
            sample_mask=rec["valid"] & fire,
            mdd_mask=mdd_mask,
            cfg=probe_cfg,
        )
    return {
        "label": run.label,
        "mode": "ac" if run.use_ac else "bc",
        "checkpoint_dir": run.checkpoint_dir,
        "fold": int(split.fold_idx),
        "summary": _summary_metrics(
            positions=positions,
            no_adapter=no_adapter,
            returns=returns,
            cfg=payload["cfg"],
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
            fire_eps=probe_cfg.fire_eps,
        ),
        "horizons": horizons,
    }


def _readiness(records: list[dict], primary_horizon: int) -> dict:
    adv_top10 = []
    adv_top20 = []
    adv_spread = []
    best_combined = []
    for rec in records:
        metrics = rec["horizons"].get(str(primary_horizon))
        if not metrics or metrics.get("eval_samples", 0) == 0:
            continue
        adv = metrics["advantage"]
        adv_top10.append(float(adv["top10"]["fire_advantage"]) > 0.0)
        adv_top20.append(float(adv["top20"]["fire_advantage"]) > 0.0)
        adv_spread.append(float(adv["top10_bottom10_spread"]) > 0.0)
        combo = metrics["combined_scores"]["D_all"]["top10"]
        best_combined.append(float(combo["fire_advantage"]) > 0.0 and float(combo["harm_margin"]) <= 0.0)
    return {
        "primary_adv_top10_all_positive": bool(adv_top10) and all(adv_top10),
        "primary_adv_top20_all_positive": bool(adv_top20) and all(adv_top20),
        "primary_adv_spread_all_positive": bool(adv_spread) and all(adv_spread),
        "combined_D_top10_positive_and_nonharm_all": bool(best_combined) and all(best_combined),
    }


def _write_markdown(path: str, *, records: list[dict], args: argparse.Namespace, cfg: V2ProbeConfig) -> None:
    lines = [
        "# Plan15 Fire-Control Label V2 Probe",
        "",
        "## Setup",
        "",
        f"- config: `{args.config}`",
        f"- folds: `{args.folds}`",
        f"- horizons: `{','.join(str(h) for h in cfg.horizons)}`",
        f"- primary horizon: `{cfg.primary_horizon}`",
        "- scope: label/ranking/combined-score probe only; no fire guard, WM head v2, or AC unlock.",
        "",
        "## Policy Summary",
        "",
        "| run | fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        s = rec["summary"]
        lines.append(
            f"| {rec['label']} | {rec['fold']} | {_format_float(s['alpha_excess_pt'], 2, True)} | "
            f"{_format_float(s['sharpe_delta'], 3, True)} | {_format_float(s['maxdd_delta_pt'], 2, True)} | "
            f"{_format_float(s['turnover'], 2)} | {s['long']:.1%} | {s['short']:.1%} | {s['flat']:.1%} | "
            f"{s['fire_rate']:.1%}/{s['fire_count']} | {_format_float(s['fire_pnl'], 4, True)} |"
        )

    lines += [
        "",
        "## Fire Advantage Ranking",
        "",
        "| run | fold | h | samples | eval | corr | top10 adv | top20 adv | bottom10 adv | spread | top10 harm | top10 dd_rel | top10 mdd |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for horizon, metrics in rec["horizons"].items():
            if metrics.get("eval_samples", 0) == 0:
                continue
            adv = metrics["advantage"]
            top10 = adv["top10"]
            top20 = adv["top20"]
            bottom10 = adv["bottom10"]
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {horizon} | {metrics['samples']} | {metrics['eval_samples']} | "
                f"{_format_float(adv['corr'])} | {_format_float(top10['fire_advantage'], 5, True)} | "
                f"{_format_float(top20['fire_advantage'], 5, True)} | {_format_float(bottom10['fire_advantage'], 5, True)} | "
                f"{_format_float(adv['top10_bottom10_spread'], 5, True)} | {_format_float(top10['harm_margin'], 5, True)} | "
                f"{_format_float(top10['dd_rel'])} | {_format_float(top10['mdd_rate'])} |"
            )

    lines += [
        "",
        "## Recovery / Relative DD Labels",
        "",
        "| run | fold | h | recovery AUC | rel recovery AUC | post-trough AUC | underwater AUC | dd k0.5 AUC | dd q80 AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for horizon, metrics in rec["horizons"].items():
            if metrics.get("eval_samples", 0) == 0:
                continue
            recovery = metrics["recovery"]
            dd = metrics["drawdown_relative"]
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {horizon} | "
                f"{_format_float(recovery['recovery_slope']['auc'])} | "
                f"{_format_float(recovery['relative_recovery_slope']['auc'])} | "
                f"{_format_float(recovery['post_trough_momentum']['auc'])} | "
                f"{_format_float(recovery['underwater_recovery']['auc'])} | "
                f"{_format_float(dd.get('k_0.5', {}).get('auc', float('nan')))} | "
                f"{_format_float(dd.get('q_80', {}).get('auc', float('nan')))} |"
            )

    lines += [
        "",
        "## Low-Harm Ranking",
        "",
        "| run | fold | h | low10 adv | low10 harm | low10 dd_rel | low10 mdd | low10 harm<=0 | high10 harm | high10 dd_rel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for horizon, metrics in rec["horizons"].items():
            if metrics.get("eval_samples", 0) == 0:
                continue
            ranking = metrics["harm_ranking"]
            low = ranking["low_harm_top10"]
            high = ranking["high_harm_top10"]
            lines.append(
                f"| {rec['label']} | {rec['fold']} | {horizon} | {_format_float(low['fire_advantage'], 5, True)} | "
                f"{_format_float(low['harm_margin'], 5, True)} | {_format_float(low['dd_rel'])} | "
                f"{_format_float(low['mdd_rate'])} | {_format_float(low['harm_nonpositive_rate'])} | "
                f"{_format_float(high['harm_margin'], 5, True)} | {_format_float(high['dd_rel'])} |"
            )

    lines += [
        "",
        "## Combined Score Top-Decile",
        "",
        "| run | fold | h | score | top10 adv | top10 harm | top10 dd_rel | top10 recovery | top10 mdd | top20 adv | top20 harm |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in records:
        for horizon, metrics in rec["horizons"].items():
            if metrics.get("eval_samples", 0) == 0:
                continue
            for name, score_metrics in metrics["combined_scores"].items():
                top10 = score_metrics["top10"]
                top20 = score_metrics["top20"]
                lines.append(
                    f"| {rec['label']} | {rec['fold']} | {horizon} | {name} | "
                    f"{_format_float(top10['fire_advantage'], 5, True)} | {_format_float(top10['harm_margin'], 5, True)} | "
                    f"{_format_float(top10['dd_rel'])} | {_format_float(top10['recovery_slope'], 5, True)} | "
                    f"{_format_float(top10['mdd_rate'])} | {_format_float(top20['fire_advantage'], 5, True)} | "
                    f"{_format_float(top20['harm_margin'], 5, True)} |"
                )

    readiness = _readiness(records, cfg.primary_horizon)
    lines += [
        "",
        "## Readiness",
        "",
        "| criterion | pass |",
        "|---|---:|",
    ]
    for key, value in readiness.items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- Plan15 does not change the production training flow.",
        "- `fire_advantage_h32` is the primary signal. It must be positive in top10/top20 and have positive top-bottom spread across folds.",
        "- Low-harm ranking is useful only if low predicted harm has positive fire advantage and non-positive realized harm margin.",
        "- Combined scores are candidates for Plan16 inference-only guard, not adoption by themselves.",
        "",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.fire_control_label_v2_probe")
    parser.add_argument("--config", default="configs/trading_wm_control_headonly.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="4,5,6")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default="checkpoints/data_cache")
    parser.add_argument("--run", action="append", required=True, help="label=checkpoint_dir[@ac_file][:ac|:bc]")
    parser.add_argument("--horizons", default="16,32")
    parser.add_argument("--primary-horizon", type=int, default=32)
    parser.add_argument("--fire-eps", type=float, default=1e-6)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--ridge-l2", type=float, default=1e-2)
    parser.add_argument("--max-z-dim", type=int, default=128)
    parser.add_argument("--rel-vol-window", type=int, default=64)
    parser.add_argument("--dd-rel-ks", default="0.5,1.0,1.5")
    parser.add_argument("--dd-quantiles", default="0.7,0.8,0.9")
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
    probe_cfg = V2ProbeConfig(
        horizons=tuple(int(x) for x in args.horizons.split(",") if x.strip()),
        primary_horizon=int(args.primary_horizon),
        fire_eps=float(args.fire_eps),
        train_frac=float(args.train_frac),
        ridge_l2=float(args.ridge_l2),
        max_z_dim=int(args.max_z_dim),
        rel_vol_window=int(args.rel_vol_window),
        dd_rel_ks=tuple(float(x) for x in args.dd_rel_ks.split(",") if x.strip()),
        dd_quantiles=tuple(float(x) for x in args.dd_quantiles.split(",") if x.strip()),
    )
    runs = [_parse_run(spec) for spec in args.run]
    records: list[dict] = []
    for split in splits:
        for run in runs:
            try:
                print(f"[Plan15] run={run.label} fold={split.fold_idx} checkpoint={run.checkpoint_dir}")
                records.append(
                    _evaluate_run_fold_v2(
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
                print(f"[Plan15] skip missing run={run.label} fold={split.fold_idx}")

    serializable = {
        "config": args.config,
        "folds": args.folds,
        "runs": [run.__dict__ for run in runs],
        "probe": probe_cfg.__dict__,
        "records": records,
        "readiness": _readiness(records, probe_cfg.primary_horizon),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, allow_nan=True)
    _write_markdown(args.output_md, records=records, args=args, cfg=probe_cfg)
    print(f"[Plan15] wrote {args.output_md}")


if __name__ == "__main__":
    main()
