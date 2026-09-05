"""Registered persistence comparisons for existing six-hour HGB risk forecasts.

This is a descriptive diagnostic on reused validation data. It neither fits a
new model nor ranks/selects a policy. Rowwise arrays retain the complete grid so
a later, separately registered dependence-aware comparison remains possible.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .alpha_dd_search import (
    digest, file_digest, fold_spec, load_bars, validate_data_artifact, write_json,
)
from .oracle_frontier import outcome_frame, quarter_regime
from .oracle_frontier_features import make_feature_groups

WINDOWS = (24, 96, 672)
GROUPS = ("base16", "technical", "flow")
LOSSES = ("variance_mse", "qlike", "rms_mse")


def persistence_forecasts(bars: pd.DataFrame, *, horizon: int = 24,
                          windows: tuple[int, ...] = WINDOWS,
                          coverage: float = .995) -> pd.DataFrame:
    """sqrt(horizon * trailing mean(log-return squared)), shifted one bar.

    Windows count scheduled 15-minute rows, including missing rows. A return
    needs both adjacent finite positive closes; no return bridges a missing
    candle. Missing observations are never filled. A measured flat history has
    zero variance. At least window + 1 historical prices must have elapsed.
    """
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.empty:
        raise ValueError("nonempty DatetimeIndex required")
    if bars.index.hasnans or bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError("finite, unique, increasing timestamps required")
    if np.any(bars.index.asi8 % pd.Timedelta(minutes=15).value):
        raise ValueError("15-minute aligned timestamps required")
    if horizon < 1 or not windows or any(w < 1 for w in windows) or not 0 < coverage <= 1:
        raise ValueError("invalid persistence specification")
    grid = pd.date_range(bars.index[0], bars.index[-1], freq="15min", name=bars.index.name)
    close = bars.close.reindex(grid).astype(float)
    close = close.where(np.isfinite(close) & close.gt(0))
    squared_return = np.log(close).diff().pow(2)
    out = pd.DataFrame(index=grid)
    for window in windows:
        variance = squared_return.rolling(window, min_periods=math.ceil(window * coverage)).mean()
        variance = variance.where(np.arange(len(grid)) >= window)
        out[f"persistence_w{window}"] = np.sqrt(horizon * variance).shift(1)
    return out


def loss_arrays(actual_rms: np.ndarray, predicted_rms: np.ndarray, *, epsilon: float = 1e-12) -> dict:
    """MSE on variance, scale-free QLIKE, and original-target RMS MSE.

    QLIKE is y/p - log(y/p) - 1, with observed and predicted variance both
    floored at epsilon. Floors apply to measured zero, never to missing data.
    """
    actual = np.asarray(actual_rms, float)
    predicted = np.asarray(predicted_rms, float)
    if actual.ndim != 1 or predicted.ndim != 2 or len(actual) != len(predicted):
        raise ValueError("aligned vector outcomes and matrix predictions required")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive finite QLIKE epsilon required")
    valid = np.isfinite(actual[:, None]) & np.isfinite(predicted)
    valid &= (actual[:, None] >= 0) & (predicted >= 0)
    observed_variance = actual[:, None] ** 2
    predicted_variance = predicted ** 2
    y = np.maximum(observed_variance, epsilon)
    p = np.maximum(predicted_variance, epsilon)
    ratio = y / p
    # Clamp numerical roundoff around the measured equal-variance minimum.
    qlike = np.maximum(ratio - np.log(ratio) - 1.0, 0.0)
    return {
        "variance_mse": np.where(valid, (observed_variance - predicted_variance) ** 2, np.nan),
        "qlike": np.where(valid, qlike, np.nan),
        "rms_mse": np.where(valid, (actual[:, None] - predicted) ** 2, np.nan),
    }


def common_score_mask(index: pd.DatetimeIndex, *, actual: np.ndarray, predictions: np.ndarray,
                      source_support: np.ndarray, end: pd.Timestamp, horizon: int) -> np.ndarray:
    """Intersect all predictions and labels without allowing a cross-boundary label."""
    actual = np.asarray(actual, float)
    predictions = np.asarray(predictions, float)
    support = np.asarray(source_support, bool)
    if actual.shape != (len(index),) or support.shape != (len(index),):
        raise ValueError("aligned actual and support required")
    if predictions.ndim != 2 or len(predictions) != len(index):
        raise ValueError("aligned prediction matrix required")
    ends_inside = np.asarray(index + pd.Timedelta(minutes=15 * (horizon + 1)) <= end)
    return (support & ends_inside & np.isfinite(actual) & (actual >= 0)
            & np.isfinite(predictions).all(axis=1) & (predictions >= 0).all(axis=1))


def verify_digest(path: Path, expected: str) -> str:
    actual = file_digest(path)
    if actual != expected:
        raise ValueError(f"source digest mismatch: {path}")
    return actual


def _ratio(numerator: float, denominator: float):
    return float(numerator / denominator) if denominator > 0 else None


def summarize(rows: list[dict], *, model_ids: list[str], baseline_ids: list[str]) -> dict:
    """Unweighted-quarter and pooled summaries, without choosing a winner."""
    result = {}
    for regime in ("all", "bull", "bear", "sideways"):
        subset = [r for r in rows if regime == "all" or r["regime"] == regime]
        if not subset:
            result[regime] = {"quarters": 0, "rows": 0, "models": {}, "paired_comparisons": []}
            continue
        total = sum(r["rows"] for r in subset)
        metrics = {}
        for model in model_ids + baseline_ids:
            metrics[model] = {}
            for loss in LOSSES:
                values = [r["losses"][model][loss] for r in subset]
                metrics[model][loss] = {
                    "equal_quarter_mean": float(np.mean(values)),
                    "pooled_mean": float(sum(r["rows"] * r["losses"][model][loss] for r in subset) / total),
                }
        pairs = []
        for model in model_ids:
            for baseline in baseline_ids:
                for loss in LOSSES:
                    m, b = metrics[model][loss], metrics[baseline][loss]
                    ratio = _ratio(m["equal_quarter_mean"], b["equal_quarter_mean"])
                    pairs.append({
                        "model_id": model, "baseline_id": baseline, "loss": loss,
                        "equal_quarter_loss_ratio": ratio,
                        "equal_quarter_relative_loss_reduction": None if ratio is None else 1.0 - ratio,
                        "pooled_loss_ratio": _ratio(m["pooled_mean"], b["pooled_mean"]),
                        "equal_quarter_loss_difference": m["equal_quarter_mean"] - b["equal_quarter_mean"],
                        "quarters_model_better": sum(r["losses"][model][loss] < r["losses"][baseline][loss]
                                                     for r in subset),
                    })
        result[regime] = {"quarters": len(subset), "rows": total,
                          "models": metrics, "paired_comparisons": pairs}
    return result


def _committed_source(paths: list[Path]) -> str:
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    relative = [str(p.resolve().relative_to(repo.resolve())) for p in paths]
    subprocess.run(["git", "ls-files", "--error-unmatch", "--", *relative],
                   check=True, capture_output=True)
    check = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *relative], check=False)
    if check.returncode:
        raise ValueError("diagnostic source, tests, config and registration must be committed before execution")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    if (tuple(config["baseline_windows"]) != WINDOWS or tuple(config["forecast_groups"]) != GROUPS
            or config["horizon_bars"] != 24 or config["minimum_coverage"] != .995
            or config["qlike_epsilon"] != 1e-12):
        raise ValueError("this registration fixes groups, windows, horizon, coverage and QLIKE floor")
    output = Path(config["output_dir"])
    if (output / "results.json").exists() or (output / "registration.json").exists():
        raise ValueError("immutable diagnostic output already exists")
    owned_paths = [Path(__file__), config_path, Path(config["registration_document"]),
                   Path(config["test_path"])]
    revision = _committed_source(owned_paths)
    source_result_path = Path(config["frontier_results"])
    source_registration_path = Path(config["frontier_registration"])
    verify_digest(source_result_path, config["frontier_result_sha256"])
    verify_digest(source_registration_path, config["frontier_registration_sha256"])
    source = json.loads(source_result_path.read_text())
    source_registration = json.loads(source_registration_path.read_text())
    if source["registration_sha256"] != digest(source_registration):
        raise ValueError("source result does not bind source registration")
    verify_digest(Path(config["frontier_config"]), source_registration["config_sha256"])
    fc = source_registration["config"]
    if fc["feature_groups"] != list(GROUPS) or fc["horizons_bars"][0] != 24:
        raise ValueError("unexpected source feature or horizon contract")
    source_bindings = {}
    for filename, expected in source_registration["source_sha256"].items():
        if Path(filename).name != filename:
            raise ValueError("source binding must name a module basename")
        path = Path(__file__).parent / filename
        source_bindings[str(path)] = verify_digest(path, expected)
    data_proof = validate_data_artifact(Path(fc["data_path"]), expected_symbol=fc["symbol"])
    if data_proof != source_registration["data_proof"]:
        raise ValueError("source data proof changed")

    model_ids = [f"{group}_hgb_h24" for group in GROUPS]
    baseline_ids = [f"persistence_w{window}" for window in WINDOWS]
    ledger = {(r["fold"], r["model_id"]): r for r in source["forecast_scores"]}
    if len(ledger) != len(source["forecast_scores"]):
        raise ValueError("duplicate source forecast entries")
    root = source_result_path.parent
    artifact_bindings = {}
    for fold in fc["development_folds"]:
        for model in model_ids:
            info = ledger[fold, model]["provenance"]
            for folder, extension, field in (("forecasts", "npz", "predictions_sha256"),
                                              ("models", "joblib", "model_sha256")):
                path = root / folder / f"fold{fold}_{model}.{extension}"
                artifact_bindings[str(path)] = verify_digest(path, info[field])
    proof = {"config": config, "config_sha256": file_digest(config_path),
             "source_revision": revision, "owned_source_sha256": {str(p): file_digest(p) for p in owned_paths},
             "source_module_bindings": source_bindings, "artifact_bindings": artifact_bindings,
             "frontier_result_sha256": file_digest(source_result_path),
             "frontier_registration_sha256": file_digest(source_registration_path),
             "data_proof": data_proof,
             "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
             "scope": "descriptive reused validation; no ranking, selection, deployment or formal P1"}
    write_json(output / "registration.json", proof)

    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    groups = make_feature_groups(bars)
    baseline = persistence_forecasts(bars, horizon=24)
    if not baseline.index.equals(bars.index):
        raise ValueError("baseline and source data grids differ")
    common = np.logical_and.reduce([np.isfinite(groups[g].to_numpy(float)).all(axis=1) for g in GROUPS])
    cadence = np.asarray((bars.index.hour % fc["decision_cadence_hours"] == 0) & (bars.index.minute == 0))
    actual_full = outcome_frame(bars, 24).to_numpy(float)
    rows = []
    for fold in fc["development_folds"]:
        spec = fold_spec(fold, fc["fold_anchor"])
        start, end = spec["val_start"], spec["val_end"]
        if end > pd.Timestamp(fc["data_cutoff"]):
            raise ValueError("validation exceeds source cutoff")
        ix = np.asarray((bars.index >= start) & (bars.index < end))
        index = bars.index[ix]
        if not index.equals(pd.date_range(start, end, freq="15min", inclusive="left")):
            raise ValueError("validation quarter is incomplete")
        inference = (common & cadence)[ix]
        support = inference & np.asarray(index + pd.Timedelta(minutes=15 * 25) <= end)
        actual = actual_full[ix].copy()
        actual[~support] = np.nan
        predicted = []
        regime = quarter_regime(groups["base16"], ix, fc["regime"]["normalized_momentum_90_threshold"])
        for group, model in zip(GROUPS, model_ids):
            info = ledger[fold, model]
            if info["provenance"]["features"] != list(groups[group].columns) or info["regime"] != regime:
                raise ValueError("source feature schema or regime changed")
            with np.load(root / "forecasts" / f"fold{fold}_{model}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["timestamps"], index.asi8):
                    raise ValueError("forecast timestamps changed")
                if not np.array_equal(saved["inference_mask"], inference):
                    raise ValueError("source inference mask changed")
                if not np.array_equal(saved["score_support"], support):
                    raise ValueError("source boundary support changed")
                if not np.array_equal(saved["actual"], actual, equal_nan=True):
                    raise ValueError("source outcomes differ from recomputed outcomes")
                pred = saved["predictions"].copy()
                if pred.shape != actual.shape or not np.isfinite(pred[inference]).all():
                    raise ValueError("source forecast dimensions or finiteness changed")
                predicted.append(pred[:, 2])
        pred_matrix = np.column_stack([*predicted, baseline.to_numpy(float)[ix]])
        mask = common_score_mask(index, actual=actual[:, 2], predictions=pred_matrix,
                                 source_support=support, end=end, horizon=24)
        n = int(mask.sum())
        if n < 16:
            raise ValueError(f"fold {fold} has insufficient common score support")
        losses = loss_arrays(actual[:, 2], pred_matrix, epsilon=config["qlike_epsilon"])
        for value in losses.values():
            value[~mask] = np.nan
            if not np.isfinite(value[mask]).all():
                raise ValueError("nonfinite scored loss")
        names = model_ids + baseline_ids
        path = output / "rowwise" / f"fold{fold}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, timestamps=index.asi8, common_score_mask=mask,
                            source_score_support=support, model_ids=np.asarray(names),
                            actual_rms=actual[:, 2], predicted_rms=pred_matrix,
                            actual_variance=actual[:, 2] ** 2, predicted_variance=pred_matrix ** 2,
                            **{f"loss_{k}": v for k, v in losses.items()})
        rows.append({"fold": fold, "validation_start": str(start), "validation_end": str(end),
                     "regime": regime["trend"], "rows": n, "grid_rows": len(index),
                     "source_inference_rows": int(inference.sum()),
                     "source_boundary_support_rows": int(support.sum()),
                     "source_valid_outcome_rows": int((support & np.isfinite(actual).all(axis=1)).sum()),
                     "losses": {name: {loss: float(np.mean(value[mask, j])) for loss, value in losses.items()}
                                for j, name in enumerate(names)},
                     "rowwise_path": str(path), "rowwise_sha256": file_digest(path)})
    result = {"registration_sha256": file_digest(output / "registration.json"),
              "model_ids": model_ids, "baseline_ids": baseline_ids, "loss_definitions": {
                  "variance_mse": "mean((actual_RMS^2 - predicted_RMS^2)^2)",
                  "qlike": "mean(y/p - log(y/p) - 1), y=max(actual_RMS^2,1e-12), p=max(predicted_RMS^2,1e-12)",
                  "rms_mse": "mean((actual_RMS - predicted_RMS)^2)"},
              "rows": rows, "summary": summarize(rows, model_ids=model_ids, baseline_ids=baseline_ids),
              "scope": proof["scope"], "ranking_performed": False, "selection_performed": False,
              "formal_p1_result": False, "high_probability_generalization_established": False}
    write_json(output / "results.json", result)
    print(json.dumps({"quarters": len(rows), "common_rows": sum(r["rows"] for r in rows),
                      "models": model_ids, "baselines": baseline_ids,
                      "ranking_performed": False, "output": str(output)}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
