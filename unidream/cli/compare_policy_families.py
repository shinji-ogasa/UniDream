"""Leak-free policy-family comparison on configured WFO folds.

Compares a causal algorithm, tabular ML, a direct WM allocator, and the BC
actor before AC fine-tuning.  Every execution parameter is selected on the
validation slice; test data is report-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from unidream.cli.train import (
    _action_stats,
    _benchmark_position_value,
    _forward_window_stats,
    _fmt_action_stats,
    _ts,
)
from unidream.data.dataset import WFODataset
from unidream.eval.backtest import Backtest
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import build_bc_trainer
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.run_config import (
    checkpoint_semantic_fingerprint,
    configure_determinism,
    data_fingerprint,
    source_fingerprint,
)
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_configured_wfo_splits
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble


METHODS = ("simple_algorithm", "tabular_ml", "wm_only", "bc_only")


@dataclass(frozen=True)
class ExecutionSpec:
    blend: float
    max_step: float
    min_hold: int


def _metrics(returns: np.ndarray, positions: np.ndarray, cfg: dict, benchmark: float) -> dict[str, float]:
    costs = cfg["costs"]
    t = min(len(returns), len(positions))
    result = Backtest(
        returns[:t],
        positions[:t],
        spread_bps=float(costs["spread_bps"]),
        fee_rate=float(costs["fee_rate"]),
        slippage_bps=float(costs["slippage_bps"]),
        interval=str(cfg["data"]["interval"]),
        benchmark_positions=np.full(t, benchmark, dtype=np.float64),
    ).run()
    stats = _action_stats(positions[:t], benchmark_position=benchmark)
    return {
        "alpha_excess_pt": 100.0 * float(result.alpha_excess or 0.0),
        "maxdd_delta_pt": 100.0 * float(result.maxdd_delta or 0.0),
        "sharpe_delta": float(result.sharpe_delta or 0.0),
        "total_return": float(result.total_return),
        "max_drawdown": 100.0 * abs(float(result.max_drawdown)),
        "turnover": float(stats["turnover"]),
    }


def _validation_score(metrics: dict[str, float]) -> float:
    alpha = metrics["alpha_excess_pt"]
    dd = metrics["maxdd_delta_pt"]
    return float(
        2.0 * alpha
        - 6.0 * max(0.0, dd)
        + 2.0 * max(0.0, -dd)
        - 0.5 * metrics["turnover"]
    )


def _execute_targets(
    targets: np.ndarray,
    *,
    benchmark: float,
    min_position: float,
    max_position: float,
    spec: ExecutionSpec,
) -> np.ndarray:
    raw = benchmark + spec.blend * (np.asarray(targets, dtype=np.float64) - benchmark)
    raw = np.clip(raw, min_position, max_position)
    output = np.empty_like(raw)
    current = float(benchmark)
    bars_since_trade = int(spec.min_hold)
    for idx, target in enumerate(raw):
        if bars_since_trade < spec.min_hold:
            target = current
        delta = float(np.clip(target - current, -spec.max_step, spec.max_step))
        next_position = float(np.clip(current + delta, min_position, max_position))
        if abs(next_position - current) > 1e-8:
            bars_since_trade = 0
        else:
            bars_since_trade += 1
        current = next_position
        output[idx] = current
    return output.astype(np.float32)


def _execution_grid() -> list[ExecutionSpec]:
    return [
        ExecutionSpec(blend=blend, max_step=max_step, min_hold=min_hold)
        for blend in (0.25, 0.50, 0.75, 1.0)
        for max_step in (0.01, 0.02, 0.04)
        for min_hold in (16, 32, 64)
    ]


def _select_execution(
    val_returns: np.ndarray,
    val_targets: np.ndarray,
    cfg: dict,
    benchmark: float,
    min_position: float,
    max_position: float,
) -> tuple[ExecutionSpec, dict[str, float]]:
    candidates = []
    for spec in _execution_grid():
        positions = _execute_targets(
            val_targets,
            benchmark=benchmark,
            min_position=min_position,
            max_position=max_position,
            spec=spec,
        )
        metrics = _metrics(val_returns, positions, cfg, benchmark)
        candidates.append((spec, metrics))
    return max(candidates, key=lambda row: _validation_score(row[1]))


def _causal_vol(returns: np.ndarray, lookback: int) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    shifted = np.concatenate([[0.0], values[:-1]])
    squared = shifted * shifted
    csum = np.concatenate([[0.0], np.cumsum(squared)])
    indices = np.arange(len(values))
    starts = np.maximum(indices + 1 - int(lookback), 0)
    sums = csum[indices + 1] - csum[starts]
    counts = indices + 1 - starts
    return np.sqrt(sums / np.maximum(counts, 1))


def _vol_targets(
    returns: np.ndarray,
    train_returns: np.ndarray,
    *,
    benchmark: float,
    lookback: int,
    target_quantile: float,
    min_position: float,
    max_position: float,
) -> np.ndarray:
    train_vol = _causal_vol(train_returns, lookback)
    positive = train_vol[train_vol > 0.0]
    target_vol = float(np.quantile(positive, target_quantile)) if len(positive) else 0.0
    current_vol = _causal_vol(returns, lookback)
    targets = benchmark * target_vol / np.maximum(current_vol, 1e-8)
    return np.clip(targets, min_position, max_position).astype(np.float32)


def _simple_algorithm(
    dataset: WFODataset,
    cfg: dict,
    benchmark: float,
    min_position: float,
    max_position: float,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    candidates = []
    for lookback in (64, 128, 256):
        for quantile in (0.35, 0.50, 0.65):
            val_targets = _vol_targets(
                dataset.val_returns,
                dataset.train_returns,
                benchmark=benchmark,
                lookback=lookback,
                target_quantile=quantile,
                min_position=min_position,
                max_position=max_position,
            )
            spec, val_metrics = _select_execution(
                dataset.val_returns,
                val_targets,
                cfg,
                benchmark,
                min_position,
                max_position,
            )
            candidates.append((lookback, quantile, spec, val_metrics))
    lookback, quantile, spec, val_metrics = max(candidates, key=lambda row: _validation_score(row[3]))
    test_targets = _vol_targets(
        dataset.test_returns,
        dataset.train_returns,
        benchmark=benchmark,
        lookback=lookback,
        target_quantile=quantile,
        min_position=min_position,
        max_position=max_position,
    )
    positions = _execute_targets(
        test_targets,
        benchmark=benchmark,
        min_position=min_position,
        max_position=max_position,
        spec=spec,
    )
    selected = {"lookback": lookback, "target_quantile": quantile, "execution": spec.__dict__}
    return positions, selected, val_metrics


def _tabular_ml(
    dataset: WFODataset,
    train_targets: np.ndarray,
    cfg: dict,
    benchmark: float,
    min_position: float,
    max_position: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    n = min(len(dataset.train_features), len(train_targets))
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=64,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(dataset.train_features[:n], np.asarray(train_targets[:n], dtype=np.float32))
    val_targets = np.clip(model.predict(dataset.val_features), min_position, max_position)
    spec, val_metrics = _select_execution(
        dataset.val_returns,
        val_targets,
        cfg,
        benchmark,
        min_position,
        max_position,
    )
    test_targets = np.clip(model.predict(dataset.test_features), min_position, max_position)
    positions = _execute_targets(
        test_targets,
        benchmark=benchmark,
        min_position=min_position,
        max_position=max_position,
        spec=spec,
    )
    selected = {
        "model": "HistGradientBoostingRegressor",
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "execution": spec.__dict__,
    }
    return positions, selected, val_metrics


def _wm_utility_targets(aux: dict[str, np.ndarray], positions: list[float]) -> np.ndarray:
    utility = aux.get("position_utility")
    if utility is None or utility.ndim != 2 or utility.shape[1] != len(positions):
        raise RuntimeError("WM checkpoint does not contain the configured position_utility head")
    choices = np.asarray(positions, dtype=np.float32)
    return choices[np.argmax(utility, axis=1)]


def _wm_only(
    dataset: WFODataset,
    wm_trainer: WorldModelTrainer,
    encoded_val: dict[str, np.ndarray],
    encoded_test: dict[str, np.ndarray],
    cfg: dict,
    benchmark: float,
    min_position: float,
    max_position: float,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    utility_positions = [float(value) for value in cfg["world_model"]["position_utility_positions"]]
    val_aux = wm_trainer.predict_auxiliary_from_encoded(
        encoded_val["z"], encoded_val["h"], features=dataset.val_features
    )
    test_aux = wm_trainer.predict_auxiliary_from_encoded(
        encoded_test["z"], encoded_test["h"], features=dataset.test_features
    )
    val_targets = _wm_utility_targets(val_aux, utility_positions)
    spec, val_metrics = _select_execution(
        dataset.val_returns,
        val_targets,
        cfg,
        benchmark,
        min_position,
        max_position,
    )
    test_targets = _wm_utility_targets(test_aux, utility_positions)
    positions = _execute_targets(
        test_targets,
        benchmark=benchmark,
        min_position=min_position,
        max_position=max_position,
        spec=spec,
    )
    selected = {"head": "position_utility", "positions": utility_positions, "execution": spec.__dict__}
    return positions, selected, val_metrics


def _bc_only(
    actor,
    encoded_val: dict[str, np.ndarray],
    encoded_test: dict[str, np.ndarray],
    dataset: WFODataset,
    val_regime_probs: np.ndarray | None,
    test_regime_probs: np.ndarray | None,
    val_advantage: np.ndarray | None,
    test_advantage: np.ndarray | None,
    cfg: dict,
    benchmark: float,
    device: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, float]]:
    original_scale = float(getattr(actor, "infer_adjust_rate_scale", 1.0))
    scales = [float(value) for value in cfg["ac"].get("val_adjust_rate_scale_grid", [original_scale])]
    candidates = []
    for scale in scales:
        actor.infer_adjust_rate_scale = scale
        val_positions = actor.predict_positions(
            encoded_val["z"],
            encoded_val["h"],
            regime_np=val_regime_probs,
            advantage_np=val_advantage,
            device=device,
        )
        val_metrics = _metrics(dataset.val_returns, val_positions, cfg, benchmark)
        candidates.append((scale, val_metrics))
    selected_scale, val_metrics = max(candidates, key=lambda row: _validation_score(row[1]))
    actor.infer_adjust_rate_scale = selected_scale
    positions = actor.predict_positions(
        encoded_test["z"],
        encoded_test["h"],
        regime_np=test_regime_probs,
        advantage_np=test_advantage,
        device=device,
    )
    return positions, {"infer_adjust_rate_scale": selected_scale}, val_metrics


def _fold_model_context(
    fold_idx: int,
    dataset: WFODataset,
    cfg: dict,
    checkpoint_dir: Path,
    device: str,
) -> dict[str, Any]:
    ac_cfg = copy.deepcopy(cfg["ac"])
    bc_cfg = cfg["bc"]
    reward_cfg = cfg["reward"]
    benchmark = _benchmark_position_value(cfg)
    fold_inputs = prepare_fold_inputs(
        fold_idx=fold_idx,
        wfo_dataset=dataset,
        cfg=cfg,
        costs_cfg=cfg["costs"],
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        action_stats_fn=_action_stats,
        format_action_stats_fn=_fmt_action_stats,
        benchmark_position=benchmark,
        forward_window_stats_fn=_forward_window_stats,
        log_ts=_ts,
    )
    cfg_model = copy.deepcopy(cfg)
    if fold_inputs["train_regime_probs"] is not None:
        cfg_model.setdefault("world_model", {})["regime_dim"] = int(fold_inputs["train_regime_probs"].shape[1])
    ensemble = build_ensemble(dataset.obs_dim, cfg_model)
    wm_trainer = WorldModelTrainer(ensemble, cfg_model, device=device)
    fold_dir = checkpoint_dir / f"fold_{fold_idx}"
    wm_path = fold_dir / "world_model.pt"
    bc_path = fold_dir / "bc_actor.pt"
    if not wm_path.exists() or not bc_path.exists():
        raise FileNotFoundError(f"missing fold checkpoint under {fold_dir}")
    wm_trainer.load(str(wm_path))
    seq_len = int(cfg["data"]["seq_len"])
    encoded_train = wm_trainer.encode_sequence(dataset.train_features, actions=None, seq_len=seq_len)
    encoded_val = wm_trainer.encode_sequence(dataset.val_features, actions=None, seq_len=seq_len)
    encoded_test = wm_trainer.encode_sequence(dataset.test_features, actions=None, seq_len=seq_len)
    predictive = build_wm_predictive_state_bundle(
        wm_trainer=wm_trainer,
        wfo_dataset=dataset,
        z_train=encoded_train["z"],
        h_train=encoded_train["h"],
        seq_len=seq_len,
        ac_cfg=ac_cfg,
        log_ts=_ts,
    )
    train_advantage = fold_inputs["train_advantage_values"]
    val_advantage = fold_inputs["val_advantage_values"]
    test_advantage = fold_inputs["test_advantage_values"]
    if predictive is not None:
        ac_cfg["advantage_conditioned"] = True
        ac_cfg["advantage_dim"] = int(predictive["train"].shape[1])
        train_advantage = predictive["train"]
        val_advantage = predictive["val"]
        test_advantage = predictive["test"]
    bc_setup = prepare_bc_setup(
        ensemble=ensemble,
        oracle_action_values=fold_inputs["oracle_bundle"]["oracle_action_values"],
        oracle_positions=fold_inputs["oracle_positions"],
        oracle_values=fold_inputs["oracle_bundle"]["oracle_values"],
        train_regime_probs=fold_inputs["train_regime_probs"],
        outcome_edge=fold_inputs["outcome_edge"],
        ac_cfg=ac_cfg,
        bc_cfg=bc_cfg,
        reward_cfg=reward_cfg,
        oracle_teacher_mode=fold_inputs["oracle_bundle"]["oracle_teacher_mode"],
    )
    actor = bc_setup["actor"]
    bc_trainer = build_bc_trainer(
        actor=actor,
        ensemble=ensemble,
        bc_cfg=bc_cfg,
        oracle_cfg=fold_inputs["oracle_cfg"],
        ac_cfg=ac_cfg,
        reward_cfg=reward_cfg,
        device=device,
    )
    bc_trainer.load(str(bc_path))
    actor.eval()
    return {
        "fold_inputs": fold_inputs,
        "wm_trainer": wm_trainer,
        "actor": actor,
        "encoded_val": encoded_val,
        "encoded_test": encoded_test,
        "train_advantage": train_advantage,
        "val_advantage": val_advantage,
        "test_advantage": test_advantage,
        "wm_hash": checkpoint_semantic_fingerprint(wm_path),
        "bc_hash": checkpoint_semantic_fingerprint(bc_path),
    }


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, float | int]:
    metrics = [row["methods"][method]["test"] for row in rows]
    alpha = np.asarray([item["alpha_excess_pt"] for item in metrics], dtype=np.float64)
    dd = np.asarray([item["maxdd_delta_pt"] for item in metrics], dtype=np.float64)
    return {
        "folds": int(len(metrics)),
        "alpha_excess_mean_pt": float(alpha.mean()),
        "alpha_excess_median_pt": float(np.median(alpha)),
        "alpha_excess_worst_pt": float(alpha.min()),
        "maxdd_delta_mean_pt": float(dd.mean()),
        "maxdd_delta_median_pt": float(np.median(dd)),
        "maxdd_delta_worst_pt": float(dd.max()),
        "alpha_positive_folds": int(np.sum(alpha > 0.0)),
        "dd_improved_folds": int(np.sum(dd < 0.0)),
        "turnover_mean": float(np.mean([item["turnover"] for item in metrics])),
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    labels = {
        "simple_algorithm": "単純アルゴリズム (causal vol-target)",
        "tabular_ml": "ML (HistGradientBoosting)",
        "wm_only": "WMのみ (position-utility allocator)",
        "bc_only": "BCのみ (WM+BC, ACなし)",
    }
    lines = [
        "# Policy Family Holdout Comparison",
        "",
        f"- period: `{payload['test_period']['start']}` to `{payload['test_period']['end']}`",
        f"- folds: `{', '.join(map(str, payload['folds']))}`",
        f"- config: `{payload['config']}`",
        f"- seed/device: `{payload['seed']}` / `{payload['device']}`",
        "- selection: train fit + validation selection only; test is report-only",
        "- benchmark: B&H exposure=1.0",
        "",
        "## Summary (mean across 9 quarterly test folds)",
        "",
        "| method | AlphaEx | MaxDDDelta | median AlphaEx | worst AlphaEx | DD improved | mean turnover |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| B&H | +0.00pt | +0.00pt | +0.00pt | +0.00pt | 0/9 | 0.00 |",
    ]
    for method in METHODS:
        row = payload["summary"][method]
        lines.append(
            f"| {labels[method]} | {row['alpha_excess_mean_pt']:+.2f}pt | "
            f"{row['maxdd_delta_mean_pt']:+.2f}pt | {row['alpha_excess_median_pt']:+.2f}pt | "
            f"{row['alpha_excess_worst_pt']:+.2f}pt | {row['dd_improved_folds']}/{row['folds']} | "
            f"{row['turnover_mean']:.2f} |"
        )
    lines.extend([
        "",
        "## Fold Results",
        "",
        "| fold | test period | method | AlphaEx | MaxDDDelta | turnover |",
        "|---:|---|---|---:|---:|---:|",
    ])
    for fold in payload["results"]:
        for method in METHODS:
            metric = fold["methods"][method]["test"]
            lines.append(
                f"| {fold['fold']} | {fold['test_start']} to {fold['test_end']} | {labels[method]} | "
                f"{metric['alpha_excess_pt']:+.2f}pt | {metric['maxdd_delta_pt']:+.2f}pt | "
                f"{metric['turnover']:.2f} |"
            )
    lines.extend([
        "",
        "## Definitions",
        "",
        "- Simple: past returns only. Realized-vol target and execution parameters are selected on validation.",
        "- ML: HistGradientBoosting learns the training-only oracle position from causal tabular features; execution is selected on validation.",
        "- WM only: the Transformer WM position-utility head selects exposure directly; no actor, BC, or AC is used.",
        "- BC only: the Transformer WM encoder/predictive state and BC actor are used; the AC checkpoint is never loaded.",
        "- MaxDDDelta = strategy absolute MaxDD minus B&H absolute MaxDD. Negative is improvement.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", default="configs/plan011_overlay_actor_v31_holdout.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="docs/policy_family_holdout_comparison")
    args = parser.parse_args()

    configure_determinism(args.seed)
    set_seed(args.seed)
    cfg, cost_profile = resolve_costs(load_config(args.config))
    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    symbol = str(data_cfg["symbol"])
    interval = str(data_cfg["interval"])
    zscore_window = int(cfg["normalization"]["zscore_window_days"])
    cache_tag = f"{symbol}_{interval}_{run_cfg['start']}_{run_cfg['end']}_z{zscore_window}_v2"
    features, returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=str(run_cfg["start"]),
        end=str(run_cfg["end"]),
        zscore_window=zscore_window,
        cache_dir=str(cfg["logging"]["cache_dir"]),
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg["include_funding"]),
        include_oi=bool(data_cfg["include_oi"]),
        include_mark=bool(data_cfg["include_mark"]),
    )
    splits, fold_ids = select_configured_wfo_splits(
        build_wfo_splits(features, data_cfg), tuple(int(value) for value in run_cfg["folds"])
    )
    benchmark = _benchmark_position_value(cfg)
    min_position = float(cfg["ac"]["abs_min_position"])
    max_position = float(cfg["ac"]["abs_max_position"])
    checkpoint_dir = Path("checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007")
    results = []
    checkpoint_hashes: dict[str, dict[str, str]] = {}

    for split in splits:
        fold_idx = int(split.fold_idx)
        print(f"\n[Compare] fold={fold_idx} test={split.test_start} -> {split.test_end}")
        dataset = WFODataset(features, returns, split, seq_len=int(data_cfg["seq_len"]))
        context = _fold_model_context(fold_idx, dataset, cfg, checkpoint_dir, args.device)
        fold_inputs = context["fold_inputs"]
        simple_pos, simple_selected, simple_val = _simple_algorithm(
            dataset, cfg, benchmark, min_position, max_position
        )
        ml_pos, ml_selected, ml_val = _tabular_ml(
            dataset,
            fold_inputs["oracle_positions"],
            cfg,
            benchmark,
            min_position,
            max_position,
            args.seed,
        )
        wm_pos, wm_selected, wm_val = _wm_only(
            dataset,
            context["wm_trainer"],
            context["encoded_val"],
            context["encoded_test"],
            cfg,
            benchmark,
            min_position,
            max_position,
        )
        bc_pos, bc_selected, bc_val = _bc_only(
            context["actor"],
            context["encoded_val"],
            context["encoded_test"],
            dataset,
            fold_inputs["val_regime_probs"],
            fold_inputs["test_regime_probs"],
            context["val_advantage"],
            context["test_advantage"],
            cfg,
            benchmark,
            args.device,
        )
        method_data = {}
        for method, positions, selected, val_metrics in (
            ("simple_algorithm", simple_pos, simple_selected, simple_val),
            ("tabular_ml", ml_pos, ml_selected, ml_val),
            ("wm_only", wm_pos, wm_selected, wm_val),
            ("bc_only", bc_pos, bc_selected, bc_val),
        ):
            test_metrics = _metrics(dataset.test_returns, positions, cfg, benchmark)
            method_data[method] = {"selected": selected, "validation": val_metrics, "test": test_metrics}
            print(
                f"  {method:16s} alpha={test_metrics['alpha_excess_pt']:+8.2f}pt "
                f"maxddD={test_metrics['maxdd_delta_pt']:+6.2f}pt to={test_metrics['turnover']:.2f}"
            )
        results.append({
            "fold": fold_idx,
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
            "methods": method_data,
        })
        checkpoint_hashes[str(fold_idx)] = {
            "world_model.pt": context["wm_hash"],
            "bc_actor.pt": context["bc_hash"],
        }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_sha256": source_fingerprint(),
        "data_sha256": data_fingerprint(features, returns),
        "checkpoint_semantic_sha256": checkpoint_hashes,
        "seed": args.seed,
        "device": args.device,
        "cost_profile": cost_profile,
        "costs": cfg["costs"],
        "benchmark_position": benchmark,
        "folds": fold_ids,
        "test_period": {"start": results[0]["test_start"], "end": results[-1]["test_end"]},
        "selection_contract": "train fit and validation selection only; test report-only",
        "results": results,
        "summary": {method: _aggregate(results, method) for method in METHODS},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(payload, output.with_suffix(".md"))
    print(f"\n[Compare] wrote {output.with_suffix('.json')}")
    print(f"[Compare] wrote {output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
