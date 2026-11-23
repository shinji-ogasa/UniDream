"""Frozen, train-only linear and naive baselines for the WM input ablation.

Ridge sees the normalized five base channels over the same 256 completed
bars as the WM, flattened in time-major order. It additionally sees only the
last token's enabled market14/TA44 channels. This is a fixed linear-access
recipe, not a second Transformer or a search over feature summaries.

All groups fit the same predetermined, evenly spaced <=8192 TRAIN rows.
The prepared TRAIN-only channel means/stds are reused without refitting.
Validation, test, and the optional October report are evaluated only. Ridge
alpha=1 and its intercept are fixed, and each group/fold is fitted once;
these deterministic fits are not repeated as independent random seeds.

Run with --config only, after root prepare has frozen its preregistration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .wm_ablation_evaluation import variance_qlike
from .wm_ablation_features import TA_COLUMNS
from .wm_technical_ablation import (
    DEFAULT_CONFIG, MARKET_COLUMNS, _arrays, _features, _json, _load_config,
    _sha, _verify_freeze,
)


RECIPE = "flattened_base5_context_and_last_enabled_market_and_ta"
GROUPS = ("A", "B", "C", "D")


def even_fit_indices(train: np.ndarray, maximum: int) -> np.ndarray:
    """Choose a fixed sorted endpoint-inclusive subset, never validation rows."""
    train = np.asarray(train, dtype=np.int64)
    if train.ndim != 1 or len(train) == 0 or np.any(np.diff(train) <= 0):
        raise ValueError("train indices must be nonempty and strictly increasing")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    positions = np.linspace(0, len(train) - 1, min(maximum, len(train))).astype(np.int64)
    return train[positions]


def ridge_design(normalized_windows: np.ndarray, group: str) -> np.ndarray:
    """Flatten base5, append enabled last-token columns in registered order."""
    windows = np.asarray(normalized_windows)
    expected = 5 + len(MARKET_COLUMNS) + len(TA_COLUMNS)
    if windows.ndim != 3 or windows.shape[-1] != expected or group not in GROUPS:
        raise ValueError("expected [batch,context,base5+market14+TA44] and group A-D")
    parts = [windows[:, :, :5].reshape(len(windows), -1)]
    if group in ("C", "D"):
        parts.append(windows[:, -1, 5:5 + len(MARKET_COLUMNS)])
    if group in ("B", "D"):
        parts.append(windows[:, -1, 5 + len(MARKET_COLUMNS):])
    result = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("selected baseline design contains nonfinite inputs")
    return result


def _universal_design(arrays, decisions, fold_data, context, batch):
    width = context * 5 + len(MARKET_COLUMNS) + len(TA_COLUMNS)
    result = np.empty((len(decisions), width), dtype=np.float32)
    for offset in range(0, len(decisions), batch):
        selected = decisions[offset:offset + batch]
        values = _features(arrays["raw"], arrays["market"], selected, context)
        normalized = (values - fold_data["mean"]) / fold_data["std"]
        result[offset:offset + len(selected)] = ridge_design(normalized, "D")
    return result


def _enabled_design(universal, group, context):
    base_end = context * 5
    if group == "A":
        return universal[:, :base_end]
    if group == "B":
        return np.concatenate((universal[:, :base_end], universal[:, base_end + len(MARKET_COLUMNS):]), axis=1)
    if group == "C":
        return universal[:, :base_end + len(MARKET_COLUMNS)]
    if group == "D":
        return universal
    raise ValueError("invalid group")


def score_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    """Score return and variance predictions separately; absent heads stay None."""
    actual = frame["actual_return"].to_numpy(dtype=float)
    predicted = frame["pred_return"].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    n = int(valid.sum())
    result: dict[str, Any] = {
        "n": len(frame), "n_return": n, "return_rmse": None, "return_mae": None,
        "return_oos_r2": None, "return_spearman_ic": None,
        "n_variance": 0, "variance_qlike": None, "variance_rmse": None,
    }
    if n:
        a, p = actual[valid], predicted[valid]
        error = p - a
        denominator = float(np.sum(a * a))
        result.update(return_rmse=float(np.sqrt(np.mean(error * error))),
                      return_mae=float(np.mean(np.abs(error))),
                      return_oos_r2=1.0 - float(np.sum(error * error)) / denominator if denominator > 0 else None,
                      return_spearman_ic=float(spearmanr(a, p).statistic) if np.ptp(a) > 0 and np.ptp(p) > 0 else None)
    actual_var = frame["actual_variance"].to_numpy(dtype=float)
    pred_var = frame["pred_variance"].to_numpy(dtype=float)
    valid_var = np.isfinite(actual_var) & np.isfinite(pred_var)
    if valid_var.any():
        av, pv = actual_var[valid_var], pred_var[valid_var]
        result.update(n_variance=int(valid_var.sum()),
                      variance_qlike=float(np.mean(variance_qlike(av, pv))),
                      variance_rmse=float(np.sqrt(np.mean((pv - av) ** 2))))
    return result


def _prediction_frame(arrays, decisions, fold, split, baseline, group, horizon_column, pred_return=None, pred_variance=None):
    count = len(decisions)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(arrays["index_ns"][decisions], utc=True),
        "fold": np.full(count, fold, dtype=np.int64), "group": group,
        "baseline": baseline, "horizon": np.full(count, (4, 16, 64)[horizon_column], dtype=np.int64),
        "split": split,
        "actual_return": arrays["returns"][decisions, horizon_column].astype(np.float64),
        "pred_return": np.full(count, np.nan) if pred_return is None else np.asarray(pred_return, dtype=np.float64),
        "actual_variance": arrays["variance"][decisions, horizon_column].astype(np.float64),
        "pred_variance": np.full(count, np.nan) if pred_variance is None else np.asarray(pred_variance, dtype=np.float64),
    })


def _validate_settings(cfg):
    baseline = cfg["baselines"]
    if baseline["ridge_alpha"] != 1.0 or baseline["max_fit_rows"] != 8192 or baseline["feature_recipe"] != RECIPE:
        raise ValueError("baseline recipe, alpha and fit cap differ from preregistered choices")
    if cfg["model"]["context"] != 256:
        raise ValueError("the fixed baseline recipe uses exactly 256 input bars")
    if baseline["feature_batch_size"] != 64 or baseline["cpu_threads"] != 2:
        raise ValueError("baseline resource settings must be frozen in YAML")
    return baseline


def run(cfg):
    settings = _validate_settings(cfg)
    frozen = _verify_freeze(cfg)
    out = Path(cfg["output_dir"])
    destination = out / "baselines"
    prereg_sha = _sha(out / "preregistration.json")
    own_key = "unidream/experiments/wm_ablation_baselines.py"
    if frozen["source_sha256"].get(own_key) != _sha(Path(__file__)):
        raise ValueError("baseline module must be included in the root source freeze")
    if _sha(out / "prepared/arrays.npz") != frozen["prepared_arrays_sha256"]:
        raise ValueError("prepared arrays differ from the root freeze")
    if (destination / "manifest.json").exists():
        previous = json.loads((destination / "manifest.json").read_text())
        if previous["preregistration_sha256"] != prereg_sha:
            raise ValueError("existing baseline output belongs to a different freeze")
        for name, expected in previous["output_sha256"].items():
            if _sha(destination / name) != expected:
                raise ValueError(f"existing baseline artifact changed: {name}")
        print("baselines already complete and verified", flush=True)
        return previous
    destination.mkdir(parents=True, exist_ok=True)
    folds = json.loads((out / "prepared/folds.json").read_text())
    arrays = _arrays(str(out))
    context = int(cfg["model"]["context"])
    rows = []
    aggregates = defaultdict(list)
    fits = []
    writer = None
    started = time.monotonic()
    temporary_parquet = destination / "predictions.parquet.part"

    def emit(frame):
        nonlocal writer
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(temporary_parquet, table.schema, compression="zstd")
        writer.write_table(table)
        first = frame.iloc[0]
        identity = {key: first[key].item() if isinstance(first[key], np.generic) else first[key] for key in ("baseline", "group", "fold", "split", "horizon")}
        rows.append({**identity, **score_predictions(frame)})
        key = tuple(identity[key] for key in ("baseline", "group", "split", "horizon"))
        aggregates[key].append(frame[["actual_return", "pred_return", "actual_variance", "pred_variance"]].to_numpy())

    try:
        with threadpool_limits(limits=settings["cpu_threads"]):
            for fold in folds:
                fold_id = int(fold["fold"])
                fold_path = out / "prepared" / f"fold_{fold_id:02d}.npz"
                with np.load(fold_path) as z:
                    data = {key: z[key] for key in z.files}
                train = data["train"]
                selected = even_fit_indices(train, settings["max_fit_rows"])
                y = arrays["returns"][selected]
                if not np.isfinite(y).all():
                    raise ValueError("nonfinite TRAIN targets in Ridge fit")
                train_design = _universal_design(arrays, selected, data, context, settings["feature_batch_size"])
                mean_return = np.mean(arrays["returns"][train], axis=0)
                models = {}
                for group in GROUPS:
                    x = _enabled_design(train_design, group, context)
                    model = Ridge(alpha=settings["ridge_alpha"], fit_intercept=True, solver="cholesky", copy_X=True)
                    model.fit(x, y)
                    models[group] = model
                    fit = {"fold": fold_id, "group": group, "train_rows_available": len(train), "fit_rows": len(selected),
                           "input_columns": x.shape[1], "fit_first_decision_ns": int(arrays["index_ns"][selected[0]]),
                           "fit_last_decision_ns": int(arrays["index_ns"][selected[-1]]),
                           "fit_indices_sha256": hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest(),
                           "fold_preparation_sha256": _sha(fold_path)}
                    fits.append(fit)
                    np.savez(destination / f"ridge_fold_{fold_id:02d}_{group}.npz", coefficients=model.coef_, intercept=model.intercept_,
                             fit_indices=selected, alpha=settings["ridge_alpha"], mean=data["mean"], std=data["std"])
                    print(json.dumps({"fit": fit, "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
                del train_design
                for split in ("val", "test", "october_report_only"):
                    if split not in data or len(data[split]) == 0:
                        continue
                    decisions = data[split]
                    design = _universal_design(arrays, decisions, data, context, settings["feature_batch_size"])
                    predictions = {group: model.predict(_enabled_design(design, group, context)) for group, model in models.items()}
                    for j in range(3):
                        emit(_prediction_frame(arrays, decisions, fold_id, split, "zero_return", "common", j, np.zeros(len(decisions))))
                        emit(_prediction_frame(arrays, decisions, fold_id, split, "train_mean_return", "common", j, np.full(len(decisions), mean_return[j])))
                        emit(_prediction_frame(arrays, decisions, fold_id, split, "past_variance", "common", j,
                                               pred_variance=arrays["baseline_variance"][decisions, j]))
                        for group in GROUPS:
                            emit(_prediction_frame(arrays, decisions, fold_id, split, "ridge", group, j, predictions[group][:, j]))
                    del design
                _json(destination / "progress.json", {"status": "running", "folds_completed": fold_id + 1,
                      "total_folds": len(folds), "ridge_fits_completed": len(fits), "elapsed_seconds": time.monotonic() - started})
    finally:
        if writer is not None:
            writer.close()
    temporary_parquet.replace(destination / "predictions.parquet")
    pd.DataFrame(rows).to_csv(destination / "fold_metrics.csv", index=False)
    aggregate_rows = []
    for (baseline, group, split, horizon), chunks in aggregates.items():
        frame = pd.DataFrame(np.concatenate(chunks), columns=["actual_return", "pred_return", "actual_variance", "pred_variance"])
        aggregate_rows.append({"baseline": baseline, "group": group, "split": split, "horizon": int(horizon), **score_predictions(frame)})
    pd.DataFrame(aggregate_rows).to_csv(destination / "aggregate_metrics.csv", index=False)
    outputs = ("predictions.parquet", "fold_metrics.csv", "aggregate_metrics.csv")
    manifest = {
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_sha256": prereg_sha, "source_sha256": _sha(Path(__file__)),
        "config": settings, "ridge_fit_count": len(fits), "fit_once_per_group_fold": True,
        "ridge_input_dimensions": {g: context * 5 + (14 if g in ("C", "D") else 0) + (44 if g in ("B", "D") else 0) for g in GROUPS},
        "normalization": "root prepared channel mean/std fitted on TRAIN windows only, reused unchanged",
        "fit_selection": "endpoint-inclusive evenly spaced sorted TRAIN decisions; maximum8192; no random sampling",
        "variance_baseline": "sum of prior16 completed15m logreturns squared multiplied by horizon/16",
        "common_group_semantics": "same common support as every arm; identical naive forecasts need no duplicate per-arm rows",
        "main_aggregate": "split=test only; october_report_only has separate rows and is never pooled into test",
        "ridge_access_limitation": "base5 full256-token history; only last-token market14 and TA44; a fixed linear comparison, not equal temporal access for market14",
        "folds": fits, "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {name: _sha(destination / name) for name in outputs},
    }
    _json(destination / "manifest.json", manifest)
    _json(destination / "progress.json", {"status": "complete", "folds_completed": len(folds), "ridge_fits_completed": len(fits), "elapsed_seconds": time.monotonic() - started})
    print(json.dumps({"status": "baselines_complete", "ridge_fits": len(fits), "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run(_load_config(args.config))


if __name__ == "__main__":
    main()
