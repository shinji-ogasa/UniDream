"""Leak-safe chronological OOF contracts for the conditional experiment path.

The legacy pipeline intentionally remains available for historical replay.  It
must not, however, be mistaken for a conditional teacher: its future-derived
states are fit on the complete training window and then read in-sample.  This
module provides the small, model-agnostic contract needed by a new path while
the expensive full WM re-training integration is still being staged.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


class ChronologicalOOFError(ValueError):
    """Raised when an OOF request cannot satisfy the causal contract."""


class ConditionalPathBlocked(RuntimeError):
    """Raised when legacy hindsight state is requested by the new path."""


@dataclass(frozen=True)
class OOFOrigin:
    """One chronological prediction origin and its admissible training prefix."""

    prediction_index: int
    train_start: int
    train_end_exclusive: int
    label_cutoff_exclusive: int
    n_train: int


def conditional_path_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return whether a config opts into the new conditional/OOF path.

    Several names are accepted so an experiment manifest can choose a clear
    spelling without weakening the guard.  A truthy ``conditional_oracle``
    mapping may use an explicit ``enabled`` field.
    """
    if not isinstance(config, Mapping):
        return False
    flag_names = (
        "conditional_oracle",
        "conditional_oracle_path",
        "predictable_conditional_path",
        "p0_b_conditional_path",
    )
    sections: list[Mapping[str, Any]] = [config]
    for section_name in ("oracle", "world_model", "ac", "bc"):
        section = config.get(section_name)
        if isinstance(section, Mapping):
            sections.append(section)
    for section in sections:
        for name in flag_names:
            if name not in section:
                continue
            value = section[name]
            if isinstance(value, Mapping):
                value = value.get("enabled", False)
            if bool(value):
                return True
        mode = str(section.get("oracle_mode", section.get("mode", ""))).strip().lower()
        teacher_mode = str(section.get("teacher_mode", "")).strip().lower()
        if mode in {"conditional", "conditional_oof", "predictable_conditional"}:
            return True
        if teacher_mode in {"conditional", "conditional_oof", "predictable_conditional"}:
            return True
    return False


def require_conditional_oof_inputs(
    *,
    config: Mapping[str, Any] | None,
    oof_bundle: Mapping[str, Any] | None,
    caller: str,
) -> None:
    """Fail closed unless a caller supplies an explicitly OOF state bundle."""
    if not conditional_path_enabled(config):
        return
    if not isinstance(oof_bundle, Mapping):
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: complete chronological "
            "OOF WM retraining/state provenance is not supplied; legacy in-sample "
            "future-target state cannot cross this boundary"
        )
    provenance = oof_bundle.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: OOF bundle provenance is missing"
        )
    if str(provenance.get("fit_scheme", "")).strip().lower() not in {
        "chronological_oof",
        "expanding_origin",
        "rolling_origin",
    }:
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: fit_scheme must be chronological OOF"
        )
    if bool(provenance.get("in_sample", False)):
        raise ConditionalPathBlocked(
            f"{caller} is blocked for conditional Oracle: in-sample state is forbidden"
        )


def _as_2d_targets(targets: np.ndarray, n_rows: int) -> np.ndarray:
    arr = np.asarray(targets)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != n_rows:
        raise ChronologicalOOFError(
            f"targets must have shape (n_rows, n_outputs), got {arr.shape} for n_rows={n_rows}"
        )
    return arr


def _as_row_mask(mask: np.ndarray | None, targets: np.ndarray) -> np.ndarray:
    if mask is None:
        result = np.ones(targets.shape[0], dtype=bool)
    else:
        raw = np.asarray(mask, dtype=bool)
        if raw.ndim == 2:
            if raw.shape != targets.shape:
                raise ChronologicalOOFError(
                    f"valid_target_mask shape {raw.shape} does not match targets {targets.shape}"
                )
            result = raw.all(axis=1)
        elif raw.ndim == 1 and raw.shape[0] == targets.shape[0]:
            result = raw.copy()
        else:
            raise ChronologicalOOFError(
                "valid_target_mask must have one value per row or one value per target"
            )
    return result & np.isfinite(targets).all(axis=1)


def _coerce_prediction(value: Any, n_outputs: int) -> tuple[np.ndarray, Mapping[str, Any] | None]:
    metadata: Mapping[str, Any] | None = None
    if isinstance(value, Mapping):
        if "prediction" not in value:
            raise ChronologicalOOFError("fit_predict mapping result must contain 'prediction'")
        metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else None
        value = value["prediction"]
    elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        value, metadata = value
    pred = np.asarray(value, dtype=np.float64)
    if pred.ndim == 0:
        pred = pred.reshape(1, 1)
    elif pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if pred.shape != (1, n_outputs):
        raise ChronologicalOOFError(
            f"fit_predict must return one row with {n_outputs} outputs, got {pred.shape}"
        )
    return pred[0], metadata


def chronological_oof_predict(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], Any],
    horizon: int = 1,
    purge: int | None = None,
    min_train_size: int = 1,
    train_window: int | None = None,
    step: int = 1,
    target_end: np.ndarray | None = None,
    valid_target_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Generate predictions using only label-complete chronological prefixes.

    ``fit_predict`` is called as ``fit_predict(x_train, y_train, x_test)`` for
    one row at a time.  If it returns ``{"prediction": row, "metadata": ...}``
    the metadata is retained per origin, which lets a model attach normalizer
    or calibrator hashes without hiding them in a global, future-fitted state.

    A target at row ``i`` is assumed complete at ``i + horizon`` unless an
    explicit ``target_end`` (exclusive row index) is supplied.  The training
    prefix must end at or before ``prediction_index - purge``.  No early-row
    prediction is imputed: unavailable rows remain NaN and false in
    ``prediction_mask``.
    """
    x = np.asarray(features)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ChronologicalOOFError(f"features must be 2-D, got {x.shape}")
    n_rows = x.shape[0]
    y = _as_2d_targets(np.asarray(targets), n_rows)
    n_outputs = y.shape[1]
    horizon = int(horizon)
    if horizon < 1:
        raise ChronologicalOOFError("horizon must be >= 1")
    if purge is None:
        # A target ending at the prediction origin is already non-overlapping
        # with a future target beginning after that origin.  Extra embargo for
        # serial dependence must be supplied explicitly and is recorded below.
        purge = 0
    purge = int(purge)
    if purge < 0:
        raise ChronologicalOOFError("purge must be >= 0")
    min_train_size = int(min_train_size)
    train_window = None if train_window is None else int(train_window)
    step = int(step)
    if min_train_size < 1:
        raise ChronologicalOOFError("min_train_size must be >= 1")
    if train_window is not None and train_window < min_train_size:
        raise ChronologicalOOFError("train_window must be >= min_train_size")
    if step < 1:
        raise ChronologicalOOFError("step must be >= 1")

    row_valid = _as_row_mask(valid_target_mask, y)
    if target_end is None:
        label_end = np.arange(n_rows, dtype=np.int64) + horizon
    else:
        label_end = np.asarray(target_end)
        if label_end.ndim != 1 or len(label_end) != n_rows:
            raise ChronologicalOOFError("target_end must have one exclusive index per row")
        if not np.isfinite(label_end).all():
            raise ChronologicalOOFError("target_end must be finite")
        label_end = label_end.astype(np.int64)

    predictions = np.full((n_rows, n_outputs), np.nan, dtype=np.float64)
    prediction_mask = np.zeros(n_rows, dtype=bool)
    train_count = np.zeros(n_rows, dtype=np.int64)
    origin_records: list[dict[str, Any]] = []
    metadata_by_row: list[Mapping[str, Any] | None] = [None] * n_rows

    for prediction_index in range(0, n_rows, step):
        label_cutoff_exclusive = prediction_index - purge
        eligible = np.flatnonzero(row_valid & (label_end <= label_cutoff_exclusive))
        if train_window is not None and len(eligible) > train_window:
            eligible = eligible[-train_window:]
        if len(eligible) < min_train_size:
            continue
        # ``eligible`` is already sorted and right-exclusive by construction.
        train_start = int(eligible[0])
        train_end_exclusive = int(eligible[-1]) + 1
        result = fit_predict(
            np.array(x[eligible], copy=True),
            np.array(y[eligible], copy=True),
            np.array(x[prediction_index : prediction_index + 1], copy=True),
        )
        pred_row, metadata = _coerce_prediction(result, n_outputs)
        predictions[prediction_index] = pred_row
        if np.isfinite(pred_row).all():
            prediction_mask[prediction_index] = True
        train_count[prediction_index] = len(eligible)
        metadata_by_row[prediction_index] = metadata
        origin = OOFOrigin(
            prediction_index=prediction_index,
            train_start=train_start,
            train_end_exclusive=train_end_exclusive,
            label_cutoff_exclusive=label_cutoff_exclusive,
            n_train=len(eligible),
        )
        origin_records.append(
            {
                "prediction_index": origin.prediction_index,
                "train_start": origin.train_start,
                "train_end_exclusive": origin.train_end_exclusive,
                "train_indices": eligible.astype(int).tolist(),
                "label_cutoff_exclusive": origin.label_cutoff_exclusive,
                "n_train": origin.n_train,
            }
        )

    result = {
        "predictions": predictions,
        "prediction_mask": prediction_mask,
        "oof_mask": prediction_mask.copy(),
        "train_count": train_count,
        "origins": origin_records,
        "metadata_by_row": metadata_by_row,
        "provenance": {
            "fit_scheme": "chronological_oof",
            "horizon": horizon,
            "purge": purge,
            "min_train_size": min_train_size,
            "train_window": train_window,
            "step": step,
            "n_rows": n_rows,
            "n_predictions": int(prediction_mask.sum()),
            "in_sample": False,
        },
    }
    validate_oof_result(result, target_end=label_end)
    return result


def validate_oof_result(
    result: Mapping[str, Any],
    *,
    target_end: np.ndarray | None = None,
) -> None:
    """Validate NaN/availability and prefix invariants of an OOF result."""
    predictions = np.asarray(result.get("predictions"))
    mask = np.asarray(result.get("prediction_mask", result.get("oof_mask")), dtype=bool)
    if predictions.ndim != 2 or mask.ndim != 1 or predictions.shape[0] != mask.shape[0]:
        raise ChronologicalOOFError("OOF predictions/mask have incompatible shapes")
    if np.any(mask & ~np.isfinite(predictions).all(axis=1)):
        raise ChronologicalOOFError("prediction_mask marks a non-finite OOF row")
    origins = result.get("origins", [])
    provenance = result.get("provenance", {})
    purge = int(provenance.get("purge", 0))
    if target_end is not None:
        ends = np.asarray(target_end, dtype=np.int64)
        for origin in origins:
            t = int(origin["prediction_index"])
            cutoff = t - purge
            indices = origin.get("train_indices")
            if indices is None:
                indices = list(range(int(origin["train_start"]), int(origin["train_end_exclusive"])))
            train_end = ends[np.asarray(indices, dtype=np.int64)]
            if len(train_end) and int(np.max(train_end)) > cutoff:
                raise ChronologicalOOFError(
                    f"OOF origin {t} includes a future/incomplete label: max_end={int(np.max(train_end))} cutoff={cutoff}"
                )
    if bool(provenance.get("in_sample", False)):
        raise ChronologicalOOFError("OOF result is marked in_sample")


def chronological_oof_standardize(
    predictions: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    min_history: int = 1,
    epsilon: float = 1e-6,
) -> dict[str, np.ndarray | dict[str, Any]]:
    """Standardize OOF states with an expanding prefix only.

    The row being standardized is excluded from its own mean/std.  Early rows
    without enough OOF history remain NaN/false; callers must not replace them
    with in-sample values or zeros.
    """
    values = np.asarray(predictions, dtype=np.float64)
    mask = np.asarray(prediction_mask, dtype=bool)
    if values.ndim != 2 or mask.ndim != 1 or len(values) != len(mask):
        raise ChronologicalOOFError("predictions/mask have incompatible shapes")
    if np.any(mask & ~np.isfinite(values).all(axis=1)):
        raise ChronologicalOOFError("usable OOF state contains a non-finite value")
    if np.any(~mask & np.isfinite(values).all(axis=1)):
        raise ChronologicalOOFError("finite state exists outside the OOF mask; refusing an implicit fill")
    min_history = int(min_history)
    if min_history < 1:
        raise ChronologicalOOFError("min_history must be >= 1")
    output = np.full_like(values, np.nan, dtype=np.float64)
    output_mask = np.zeros_like(mask)
    means = np.full_like(values, np.nan, dtype=np.float64)
    scales = np.full_like(values, np.nan, dtype=np.float64)
    for t in range(len(values)):
        if not mask[t]:
            continue
        history = values[:t][mask[:t]]
        history = history[np.isfinite(history).all(axis=1)]
        if len(history) < min_history:
            continue
        mean = history.mean(axis=0)
        std = history.std(axis=0)
        std = np.where(std < float(epsilon), 1.0, std)
        output[t] = (values[t] - mean) / std
        means[t] = mean
        scales[t] = std
        if np.isfinite(output[t]).all():
            output_mask[t] = True
    return {
        "values": output,
        "mask": output_mask,
        "mean_by_row": means,
        "std_by_row": scales,
        "provenance": {
            "fit_scheme": "chronological_oof",
            "normalizer": "expanding_prefix",
            "in_sample": False,
            "min_history": min_history,
        },
    }


# Explicit aliases make the contract discoverable to experiment code without
# introducing a second implementation under a different name.
build_chronological_oof = chronological_oof_predict
build_chronological_oof_predictions = chronological_oof_predict


__all__ = [
    "ChronologicalOOFError",
    "ConditionalPathBlocked",
    "OOFOrigin",
    "build_chronological_oof",
    "build_chronological_oof_predictions",
    "chronological_oof_predict",
    "chronological_oof_standardize",
    "conditional_path_enabled",
    "require_conditional_oof_inputs",
    "validate_oof_result",
]
