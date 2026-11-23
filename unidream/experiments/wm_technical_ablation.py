"""Run the frozen, chronological 2x2 Transformer-WM feature experiment.

Research-only entrypoint; the trading CLI, selector and inference bundles are
unchanged. All periods, folds, budgets and paths are fixed in the experiment
YAML. Commands: prepare, run, status, summarize; only --config is configurable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .wm_ablation_evaluation import (
    build_future_targets, evaluate_predictions, holm_adjust,
    make_walk_forward_folds, paired_block_bootstrap, split_mask,
)
from .wm_ablation_features import BASE_COLUMNS, TA_COLUMNS, feature_columns, transform_windows
from .wm_ablation_model import FeatureAblationWM


MARKET_COLUMNS = (
    "spot_log_quote_volume", "spot_log_trade_count", "spot_taker_buy_fraction",
    "perp_log_quote_volume", "perp_log_trade_count", "perp_taker_buy_fraction",
    "perp_spot_basis", "mark_spot_basis", "index_spot_basis", "premium_index",
    "funding_rate", "log_open_interest", "log_open_interest_value", "log_long_short_ratio",
)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/wm_technical_ablation_20261109.yaml"


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    temporary.replace(path)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg["evaluation"]["test_selection_allowed"] is not False:
        raise ValueError("test must remain report-only")
    if cfg["groups"] != ["A", "B", "C", "D"] or cfg["evaluation"]["horizons"] != [4, 16, 64]:
        raise ValueError("the preregistered 2x2 comparison/horizons cannot be changed")
    expected_weights = {"normalized_return_mse": 1., "variance_qlike": .2,
        "normalized_downside_mse": .2, "crash_bce": .1, "common_base_reconstruction": .3,
        "dynamics_kl": .2, "representation_kl": .05, "next_market_return_twohot": .3}
    if cfg["loss_weights"] != expected_weights:
        raise ValueError("loss weights must match the fixed experiment model equations")
    cfg["config_path"] = str(Path(path).resolve())
    for key in ("output_dir", "raw_data", "availability"):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def market_features(raw):
    """Observed contemporaneous market fields, without technical lookbacks.

    All transformations are one-to-one/unit conversions within a completed
    row. Sparse top-trader series are excluded based solely on data coverage,
    before observing any predictions. Missing/invalid values stay NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.column_stack([
            np.log1p(raw.spot_quote_volume), np.log1p(raw.spot_n_trades),
            raw.spot_taker_buy_quote / raw.spot_quote_volume,
            np.log1p(raw.perp_quote_volume), np.log1p(raw.perp_n_trades),
            raw.perp_taker_buy_quote / raw.perp_quote_volume,
            np.log(raw.perp_close / raw.spot_close),
            np.log(raw.mark_close / raw.spot_close),
            np.log(raw.index_close / raw.spot_close), raw.premium_close,
            raw.funding_rate,
            np.log(raw.metrics_sum_open_interest.where(raw.metrics_sum_open_interest > 0)),
            np.log(raw.metrics_sum_open_interest_value.where(raw.metrics_sum_open_interest_value > 0)),
            np.log(raw.metrics_count_long_short_ratio.where(raw.metrics_count_long_short_ratio > 0)),
        ])
    values[~np.isfinite(values)] = np.nan
    return values.astype(np.float32)


def channel_mask(group):
    result = np.zeros(5 + len(MARKET_COLUMNS) + len(TA_COLUMNS), dtype=np.float32)
    result[:5] = 1
    if group in ("C", "D"):
        result[5:5 + len(MARKET_COLUMNS)] = 1
    if group in ("B", "D"):
        result[5 + len(MARKET_COLUMNS):] = 1
    if group not in ("A", "B", "C", "D"):
        raise ValueError("invalid ablation group")
    return result


def _context_mask(row_valid, length):
    """Decision t uses exactly raw rows [t-length,t), never row t."""
    counts = np.r_[0, np.cumsum(np.asarray(row_valid, dtype=np.int64))]
    result = np.zeros(len(row_valid), dtype=bool)
    t = np.arange(length, len(row_valid))
    result[t] = counts[t] - counts[t - length] == length
    return result


def _windows(arrays, decisions, context):
    positions = decisions[:, None] - context + np.arange(context)[None, :]
    return positions, arrays[positions]


def _features(raw, market, decisions, context):
    positions, bars = _windows(raw, decisions, context)
    return transform_windows(bars, "D", market_windows=market[positions], market_columns=MARKET_COLUMNS)


def _source_hashes():
    names = ["wm_technical_ablation.py", "wm_ablation_model.py", "wm_ablation_features.py", "wm_ablation_evaluation.py", "wm_ablation_data.py"]
    names += ["wm_ablation_baselines.py"]
    paths = [Path(__file__).parent / name for name in names]
    paths += [ROOT / "unidream/world_model/transformer.py", ROOT / "unidream/world_model/encoder.py"]
    return {str(p.relative_to(ROOT)): _sha(p) for p in paths}


def prepare(cfg):
    out = Path(cfg["output_dir"])
    prepared = out / "prepared"
    if (out / "preregistration.json").exists():
        raise ValueError("an existing preregistration is immutable; do not silently overwrite it")
    prepared.mkdir(parents=True, exist_ok=True)
    raw_frame = pd.read_parquet(cfg["raw_data"])
    raw = raw_frame[["spot_" + key for key in ("open", "high", "low", "close", "volume")]].to_numpy(dtype=np.float64)
    market = market_features(raw_frame)
    row_valid = np.isfinite(raw).all(1) & np.all(raw[:, :4] > 0, axis=1) & (raw[:, 4] >= 0) & np.isfinite(market).all(1)
    context = int(cfg["model"]["context"])
    context_valid = _context_mask(row_valid, context)
    targets = build_future_targets(raw_frame.index, raw[:, 3], crash_threshold=cfg["evaluation"]["crash_fraction"])
    np.savez(prepared / "arrays.npz", raw=raw, market=market, index_ns=raw_frame.index.asi8,
             returns=targets.returns, variance=targets.variance, downside=targets.downside,
             crash=targets.crash, target_mask=targets.mask, baseline_variance=targets.baseline_variance)
    folds = make_walk_forward_folds(cfg["first_test"], cfg["last_test"])
    reports = []
    for fold in folds:
        splits = {}
        for split in ("train", "val", "test"):
            eligible = split_mask(raw_frame.index, targets, fold, split, context_valid=context_valid)
            if split == "train":
                eligible &= raw_frame.index >= fold.train_start + pd.Timedelta(minutes=15 * context)
            selected = np.flatnonzero(eligible)
            if len(selected) < 1000:
                raise ValueError(f"insufficient common support: fold {fold.fold} {split}: {len(selected)}")
            splits[split] = selected
        # One scaler per fold, fitted exclusively on common train windows.
        rng = np.random.default_rng(cfg["training"]["normalization_seed"] + fold.fold)
        chosen = rng.choice(splits["train"], min(len(splits["train"]), cfg["training"]["normalization_windows"]), replace=False)
        total = np.zeros(len(channel_mask("D")), dtype=np.float64)
        square = total.copy()
        count = 0
        for start in range(0, len(chosen), 64):
            values = _features(raw, market, chosen[start:start + 64], context).astype(np.float64)
            total += values.sum((0, 1)); square += np.square(values).sum((0, 1)); count += values.shape[0] * context
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean ** 2, 0))
        std[std < 1e-7] = 1
        train = splits["train"]
        scales = {
            "return": np.maximum(np.sqrt(np.nanmean(targets.returns[train] ** 2, axis=0)), 1e-4).tolist(),
            "variance": np.maximum(np.nanmean(targets.variance[train], axis=0), 1e-8).tolist(),
            "downside": np.maximum(np.sqrt(np.nanmean(targets.downside[train] ** 2, axis=0)), 1e-4).tolist(),
        }
        select_val = splits["val"][np.linspace(0, len(splits["val"]) - 1, min(cfg["training"]["validation_rows_for_selection"], len(splits["val"]))).astype(int)]
        extras = {}
        if fold.fold == len(folds) - 1 and cfg["evaluation"]["extra_month_report_only"]:
            # Apply the same July-frozen final-fold policy to October. Its
            # outcomes never enter a scaler, loss, checkpoint or selector.
            from .wm_ablation_evaluation import WalkForwardFold
            extra = WalkForwardFold(fold.fold, fold.train_start, fold.train_end,
                fold.val_start, fold.val_end, pd.Timestamp("2026-10-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC"))
            extras["october_report_only"] = np.flatnonzero(split_mask(raw_frame.index, targets, extra, "test", context_valid=context_valid))
        np.savez(prepared / f"fold_{fold.fold:02d}.npz", **splits, **extras, selection_val=select_val, mean=mean.astype(np.float32), std=std.astype(np.float32))
        bounds = {key: str(getattr(fold, key)) for key in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")}
        report = {"fold": fold.fold, **bounds, "scales": scales, "crash_train_prevalence": np.mean(targets.crash[train], axis=0).tolist(), "support": {}}
        for split, indices in splits.items():
            start, end = fold.bounds(split)
            report["support"][split] = {"n": len(indices), "first_decision": str(raw_frame.index[indices[0]]),
                "last_decision": str(raw_frame.index[indices[-1]]),
                "last_label_end": str(pd.Timestamp(targets.label_end[indices, -1].max()).tz_localize("UTC")),
                "label_end_strictly_before_split_end": bool(np.all(targets.label_end[indices, -1] < end.tz_localize(None).to_datetime64()))}
        reports.append(report)
        print(f"prepare fold={fold.fold} support=" + str({k: len(v) for k, v in splits.items()}), flush=True)
    _json(prepared / "folds.json", reports)
    prepared_hashes = {str(p.relative_to(out)): _sha(p) for p in sorted(prepared.iterdir()) if p.is_file()}
    tasks = [{"job": f"fold_{f.fold:02d}_{g}_s{s}", "fold": f.fold, "group": g, "seed": s} for f in folds for s in cfg["seeds"] for g in cfg["groups"]]
    manifest = {"experiment_id": cfg["experiment_id"], "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preregistered_before_model_results", "config": cfg, "config_sha256": _sha(cfg["config_path"]),
        "source_sha256": _source_hashes(), "raw_data_sha256": _sha(cfg["raw_data"]),
        "availability_sha256": _sha(cfg["availability"]), "prepared_sha256": prepared_hashes,
        "prepared_arrays_sha256": _sha(prepared / "arrays.npz"), "columns": feature_columns("D", MARKET_COLUMNS),
        "group_masks": {g: channel_mask(g).tolist() for g in cfg["groups"]}, "folds": reports, "tasks": tasks,
        "hypothesis": "finite-budget generalization from deterministic OHLCV transformations, not additional market information",
        "downside_definition": "max(0,1-min(future close[t:t+h])/close[t-1]); not peak-to-trough DD",
        "variance_definition": "sum of squared future 15-minute log returns; not standard deviation",
        "selection": "validation-only normalized return MSE +0.2 variance QLIKE at h16; fixed 100-step checks",
        "limitations": ["historical source publication delays are not verified; a conservative 15-minute source-time lag is assumed", "historical evaluation periods have been reused in prior research; not pristine holdout"],
        "extra_month": "October2026 reserved report-only; not used for training, validation or model selection"}
    manifest["environment"] = {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "python": os.sys.version, "device": cfg["device"], "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "uv_lock_sha256": _sha(ROOT / "uv.lock")}
    _json(out / "preregistration.json", manifest)
    print(f"preregistered {len(tasks)} independent runs, {len(manifest['columns'])} common-width channels", flush=True)


@lru_cache(maxsize=1)
def _arrays(path):
    with np.load(Path(path) / "prepared/arrays.npz") as z:
        return {k: z[k] for k in z.files}


def _input(arrays, decisions, fold_data, group, context):
    values = _features(arrays["raw"], arrays["market"], decisions, context)
    normalized = (values - fold_data["mean"]) / fold_data["std"]
    if not np.isfinite(normalized).all():
        raise ValueError("nonfinite selected feature window")
    # Intentional disabled channels; these are not unavailable observations.
    return normalized * channel_mask(group), normalized[..., :5]


def _predict(model, arrays, decisions, fold_data, group, cfg):
    model.eval()
    result = {key: [] for key in ("return", "variance", "downside", "crash")}
    batch = cfg["training"]["evaluation_batch_size"]
    with torch.no_grad():
        for offset in range(0, len(decisions), batch):
            x, _ = _input(arrays, decisions[offset:offset + batch], fold_data, group, cfg["model"]["context"])
            predictions = model.predict_last(torch.from_numpy(x).to(cfg["device"]))
            for key in result:
                result[key].append(predictions[key].cpu().numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


def _frame(arrays, decisions, predictions, task, split):
    pieces = []
    for j, horizon in enumerate((4, 16, 64)):
        part = pd.DataFrame({"timestamp": pd.to_datetime(arrays["index_ns"][decisions], utc=True),
                             "fold": task["fold"], "group": task["group"], "seed": task["seed"], "horizon": horizon, "split": split})
        for key, actual_key in (("return", "returns"), ("variance", "variance"), ("downside", "downside"), ("crash", "crash")):
            part["actual_" + key] = arrays[actual_key][decisions, j]
            part["pred_" + key] = predictions[key][:, j]
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def _train_task(config_path, task):
    cfg = _load_config(config_path)
    out = Path(cfg["output_dir"])
    job_dir = out / "runs" / task["job"]
    job_dir.mkdir(parents=True, exist_ok=True)
    if (job_dir / "result.json").exists():
        return json.loads((job_dir / "result.json").read_text())
    with (job_dir / "training.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        start_time = time.monotonic()
        torch.set_num_threads(cfg["cpu_threads_per_worker"])
        torch.use_deterministic_algorithms(True)
        random.seed(task["seed"]); np.random.seed(task["seed"]); torch.manual_seed(task["seed"])
        if cfg["device"] == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was required; refusing implicit CPU experiment")
        arrays = _arrays(str(out))
        with np.load(out / "prepared" / f"fold_{task['fold']:02d}.npz") as z:
            fold_data = {k: z[k] for k in z.files}
        folds = json.loads((out / "prepared/folds.json").read_text())
        fold = folds[task["fold"]]
        model = FeatureAblationWM(len(channel_mask("D")), cfg["model"], fold["scales"]).to(cfg["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
        rng = np.random.default_rng(task["seed"])
        context = cfg["model"]["context"]
        best_score = float("inf"); best_step = 0; best_state = None
        gradient_steps = np.zeros(12, dtype=np.int64)
        target_counts = np.zeros(3, dtype=np.int64)
        training_start_ns = pd.Timestamp(fold["train_start"]).value
        training_end_ns = pd.Timestamp(fold["train_end"]).value
        history = []
        for step in range(1, cfg["training"]["max_steps"] + 1):
            model.train()
            decisions = rng.choice(fold_data["train"], cfg["training"]["batch_size"], replace=True)
            x, base = _input(arrays, decisions, fold_data, task["group"], context)
            # For input raw bar i, the associated prediction decision is i+1.
            target_positions = decisions[:, None] - context + np.arange(1, context + 1)[None, :]
            ends = arrays["index_ns"][target_positions, None] + np.array((4, 16, 64))[None, None, :] * 15 * 60 * 10**9
            valid = arrays["target_mask"][target_positions] & (ends < training_end_ns) & (arrays["index_ns"][target_positions, None] >= training_start_ns)
            target_counts += valid.sum((0, 1))
            target_values = {key: torch.from_numpy(arrays[name][target_positions].astype(np.float32)).to(cfg["device"])
                for key, name in (("return", "returns"), ("variance", "variance"), ("downside", "downside"), ("crash", "crash"))}
            current = arrays["raw"][target_positions, 3]
            previous = arrays["raw"][target_positions - 1, 3]
            next_return = np.log(current / previous).astype(np.float32)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(torch.from_numpy(x).to(cfg["device"]), torch.from_numpy(base).to(cfg["device"]),
                                     target_values, torch.from_numpy(valid).to(cfg["device"]), torch.from_numpy(next_return).to(cfg["device"]))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite training loss at step {step}")
            loss.backward()
            gradient = model.head[-1].weight.grad
            gradient_steps += (gradient.detach().abs().sum(1).cpu().numpy() > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            if step % cfg["training"]["validation_every"] == 0 or step == cfg["training"]["max_steps"]:
                predicted = _predict(model, arrays, fold_data["selection_val"], fold_data, task["group"], cfg)
                ids = fold_data["selection_val"]; j = 1
                ratio = np.maximum(arrays["variance"][ids, j], 1e-12) / np.maximum(predicted["variance"][:, j], 1e-12)
                val_mse = np.mean(((predicted["return"][:, j] - arrays["returns"][ids, j]) / fold["scales"]["return"][j]) ** 2)
                val_qlike = np.mean(ratio - np.log(ratio) - 1)
                score = float(val_mse + .2 * val_qlike)
                row = {"step": step, "train_loss": float(loss.detach().cpu()), "val_score": score,
                       "val_return_normalized_mse": float(val_mse), "val_variance_qlike": float(val_qlike)}
                history.append(row); print(json.dumps(row), flush=True)
                if score < best_score:
                    best_score = score; best_step = step
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is None or not np.all(gradient_steps > 0) or not np.all(target_counts > 0):
            raise RuntimeError("no usable validation checkpoint or missing forecast-head target/gradient coverage")
        model.load_state_dict(best_state)
        # Freeze a standalone research checkpoint BEFORE test is evaluated.
        # It is intentionally not accepted by trading checkpoint/bundle loaders.
        checkpoint = {"format": "wm_feature_ablation_research_v1", "task": task, "model": best_state,
                      "best_step": best_step, "best_val_score": best_score, "scales": fold["scales"],
                      "preregistration_sha256": _sha(out / "preregistration.json")}
        torch.save(checkpoint, job_dir / "research_model.pt")
        prediction_frames = []
        metrics = []
        for split in ("val", "test", "october_report_only"):
            if split not in fold_data:
                continue
            ids = fold_data[split]
            predictions = _predict(model, arrays, ids, fold_data, task["group"], cfg)
            frame = _frame(arrays, ids, predictions, task, split)
            frame.to_parquet(job_dir / f"{split}_predictions.parquet", index=False)
            for horizon, group_frame in frame.groupby("horizon", sort=True):
                metrics.append({**task, "split": split, "horizon": int(horizon), **evaluate_predictions(group_frame)})
        result = {**task, "best_step": best_step, "best_val_score": best_score, "steps": cfg["training"]["max_steps"],
                  "parameters": sum(p.numel() for p in model.parameters()), "train_n": len(fold_data["train"]),
                  "val_n": len(fold_data["val"]), "test_n": len(fold_data["test"]),
                  "target_counts": target_counts.tolist(), "nonzero_gradient_steps": gradient_steps.tolist(),
                  "elapsed_seconds": time.monotonic() - start_time, "checkpoint_sha256": _sha(job_dir / "research_model.pt"),
                  "metrics": metrics, "history": history}
        _json(job_dir / "result.json", result)
        del model, optimizer, best_state
        if cfg["device"] == "mps":
            torch.mps.empty_cache()
        return {k: v for k, v in result.items() if k not in ("metrics", "history")}


def _verify_freeze(cfg):
    out = Path(cfg["output_dir"])
    frozen = json.loads((out / "preregistration.json").read_text())
    if frozen["config_sha256"] != _sha(cfg["config_path"]) or frozen["source_sha256"] != _source_hashes():
        raise ValueError("config/source changed after preregistration; refusing mixed-version results")
    if frozen["raw_data_sha256"] != _sha(cfg["raw_data"]) or frozen["availability_sha256"] != _sha(cfg["availability"]):
        raise ValueError("raw data or availability changed after preregistration")
    for relative, expected in frozen["prepared_sha256"].items():
        if _sha(out / relative) != expected:
            raise ValueError(f"prepared split/scaler/target artifact changed after preregistration: {relative}")
    return frozen


def run(cfg):
    frozen = _verify_freeze(cfg)
    out = Path(cfg["output_dir"])
    pending = [task for task in frozen["tasks"] if not (out / "runs" / task["job"] / "result.json").exists()]
    _json(out / "run_status.json", {"status": "running", "tasks": len(frozen["tasks"]), "pending": len(pending), "pid": os.getpid()})
    errors = []
    with ProcessPoolExecutor(max_workers=cfg["parallel_workers"], mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_train_task, cfg["config_path"], task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(json.dumps({"completed": task["job"], "seconds": round(result["elapsed_seconds"], 2), "best_step": result["best_step"]}), flush=True)
            except Exception as exc:
                errors.append({"task": task, "error": repr(exc)})
                print(json.dumps(errors[-1]), flush=True)
            completed = sum((out / "runs" / t["job"] / "result.json").exists() for t in frozen["tasks"])
            _json(out / "run_status.json", {"status": "running", "completed": completed, "total": len(frozen["tasks"]), "errors": errors, "pid": os.getpid()})
    _json(out / "run_status.json", {"status": "failed" if errors else "training_completed", "total": len(frozen["tasks"]), "errors": errors})
    if errors:
        raise RuntimeError(f"{len(errors)} experiment jobs failed; inspect run_status.json")


def summarize(cfg):
    frozen = _verify_freeze(cfg)
    out = Path(cfg["output_dir"])
    missing = [task["job"] for task in frozen["tasks"] if not (out / "runs" / task["job"] / "result.json").exists()]
    if missing:
        raise ValueError(f"cannot summarize an incomplete experiment: {len(missing)} runs missing")
    records = []
    frames = []
    for task in frozen["tasks"]:
        directory = out / "runs" / task["job"]
        result = json.loads((directory / "result.json").read_text())
        records.extend(result["metrics"])
        frames.append(pd.read_parquet(directory / "test_predictions.parquet"))
    pd.DataFrame(records).to_csv(out / "fold_metrics.csv", index=False)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(out / "test_predictions.parquet", index=False)
    aggregate = []
    for (group, horizon), frame in predictions.groupby(["group", "horizon"]):
        aggregate.append({"group": group, "horizon": int(horizon), **evaluate_predictions(frame)})
    aggregate_frame = pd.DataFrame(aggregate)
    aggregate_frame.to_csv(out / "aggregate_metrics.csv", index=False)
    october_frames = [pd.read_parquet(out / "runs" / task["job"] / "october_report_only_predictions.parquet")
        for task in frozen["tasks"]
        if (out / "runs" / task["job"] / "october_report_only_predictions.parquet").exists()]
    october = []
    if october_frames:
        extra_frame = pd.concat(october_frames, ignore_index=True)
        for (group, horizon), frame in extra_frame.groupby(["group", "horizon"]):
            october.append({"group": group, "horizon": int(horizon), **evaluate_predictions(frame)})
        pd.DataFrame(october).to_csv(out / "october_report_only_metrics.csv", index=False)
    bootstrap = []
    for first, second in cfg["evaluation"]["primary_comparisons"]:
        bootstrap += paired_block_bootstrap(predictions, group_a=first, group_b=second,
            horizon=16, block_days=cfg["evaluation"]["bootstrap_block_days"],
            replicates=cfg["evaluation"]["bootstrap_replicates"], random_seed=cfg["evaluation"]["bootstrap_seed"])
    family = {}
    for row in bootstrap:
        if row["block_days"] == cfg["evaluation"]["primary_block_days"]:
            for metric in ("return_mse_improvement", "variance_qlike_improvement"):
                family[row["comparison"] + ":" + metric] = row["metrics"][metric].get("p_value")
    summary = {"experiment_id": cfg["experiment_id"], "completed_runs": len(frozen["tasks"]),
               "aggregate_metrics": aggregate, "paired_block_bootstrap": bootstrap,
               "october_report_only_metrics": october,
               "primary_holm_adjusted_p": holm_adjust(family), "preregistration_sha256": _sha(out / "preregistration.json"),
               "test_selection_allowed": False, "limitations": frozen["limitations"]}
    _json(out / "summary.json", summary)
    print(aggregate_frame[aggregate_frame.horizon == 16].to_string(index=False), flush=True)
    print(json.dumps({"holm": summary["primary_holm_adjusted_p"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "run", "summarize", "status"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = _load_config(args.config)
    if args.command == "prepare": prepare(cfg)
    elif args.command == "run": run(cfg)
    elif args.command == "summarize": summarize(cfg)
    else:
        out = Path(cfg["output_dir"])
        print((out / "run_status.json").read_text() if (out / "run_status.json").exists() else "not started")


if __name__ == "__main__":
    main()
