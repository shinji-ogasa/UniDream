"""Connect the registered ForecastActionSource to a real WM -> BC -> AC run.

This is a report-only diagnostic experiment.  It loads the already validated
S3 zero-injection-control forecast artifact, obtains the sealed ``ridge``
``ForecastActionSource``, and also revalidates the matching action artifact.
The source forecast is adapted to the strict conditional OOF envelope on the
four-bar commitment grid, then the existing world-model, behaviour-cloning,
and imagination-AC trainers are run with a materially larger budget than the
earlier wiring pilot.

The experiment deliberately does not edit the preregistration or call the
report-only P1 outer operation.  Its rolling/outer numbers are an independent
diagnostic of whether a correctly authenticated source plus enough training
can improve the student policy; they are not a formal P1 claim.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from unidream.data.dataset import WFODataset, WFOSplit
from unidream.data.oracle import conditional_oracle_teacher_path
from unidream.eval.action_execution import (
    ActionExecutionContract,
    complete_decision_starts,
    project_positions_to_contract,
    replay_action_path,
)
from unidream.eval.backtest import Backtest, pnl_attribution
from unidream.eval.policy_stats import action_stats, format_action_stats
from unidream.actor_critic.imagination_ac import _ac_alerts_ascii
from unidream.experiments.ac_stage import run_ac_stage
from unidream.experiments.bc_setup import prepare_bc_setup
from unidream.experiments.bc_stage import run_bc_stage
from unidream.experiments.chronological_oof import (
    build_conditional_oof_artifact,
    hash_conditional_oof_artifact,
    load_conditional_oof_artifact,
    write_conditional_oof_artifact,
)
from unidream.experiments.conditional_teacher import build_conditional_teacher_context
from unidream.experiments.fold_runtime import resolve_ac_max_steps
from unidream.experiments.logging import log_timestamp
from unidream.experiments.p1_action_artifact import load_p1_action_artifact
from unidream.experiments.p1_conditional_wm_bc_ac import (
    _build_config as _build_pilot_config,
    _file_sha256,
    _git_revision,
    _write_json,
)
from unidream.experiments.p1_recovery_runner import (
    S3_OUTER_END,
    build_s3_arm_dataset,
    load_runner_manifest,
    load_s3_validation_body,
)
from unidream.experiments.p1_validation_forecast import (
    load_p1_forecast_artifact,
    require_authenticated_forecast_action_source,
)
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle
from unidream.experiments.run_config import configure_determinism
from unidream.experiments.runtime import set_seed
from unidream.experiments.wm_stage import prepare_world_model_stage

from .p1_s3_rolling_shadow import ROLLING_WINDOWS


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL_ROOT = Path(
    "/Users/sophie/Documents/UniDream/.worktrees/p1-production-chain/"
    "codex_outputs/p1_formal_run_20260904"
)
DEFAULT_OUTPUT = ROOT / "codex_outputs" / "p1_formal_forecast_wm_bc_ac_20260904"
SOURCE_SCENARIO = "S3"
SOURCE_ARM = "zero_injection_control"
SOURCE_MODEL = "ridge"
SOURCE_HORIZON = 4
SOURCE_PURGE = 1
SOURCE_STEP = 4
# The persisted origin records include explicit train_indices.  A 64-row
# prefix is long enough to bind the fixed source causally while staying below
# the artifact's bounded provenance-node budget on 8,759 commitment starts.
SOURCE_MIN_TRAIN = 64
SOURCE_TRAIN_WINDOW = 64
SOURCE_START_RAW = 104_528
SOURCE_END_RAW = 139_568
SOURCE_TRAIN_END_LOCAL = 25_000
SOURCE_VAL_END_LOCAL = 30_000
SEED = 20260904


class FormalForecastPipelineError(RuntimeError):
    """Raised when the strict source-to-training diagnostic cannot proceed."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise FormalForecastPipelineError("non-finite report scalar")
        return float(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise FormalForecastPipelineError(f"unsupported report value: {type(value).__name__}")


def _sha(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = str(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_formal_path(formal_root: Path, ledger_path: str | Path) -> Path:
    """Resolve paths written by the formal run ledger without string guessing."""

    path = Path(ledger_path)
    if path.is_absolute():
        return path
    marker = Path("codex_outputs/p1_formal_run_20260904")
    try:
        relative = path.relative_to(marker)
    except ValueError:
        relative = path
    candidate = formal_root / relative
    if not candidate.is_file():
        raise FormalForecastPipelineError(f"formal ledger artifact is missing: {candidate}")
    return candidate


def _load_formal_source(
    formal_root: Path,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Load and authenticate the registered forecast and action artifacts."""

    try:
        forecast_rows = json.loads((formal_root / "ledgers/forecasts.json").read_text())
        action_rows = json.loads((formal_root / "ledgers/actions.json").read_text())
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FormalForecastPipelineError("formal forecast/action ledgers are unavailable") from exc
    forecast_row = next(
        (
            row
            for row in forecast_rows
            if row.get("scenario_id") == SOURCE_SCENARIO
            and row.get("arm") == SOURCE_ARM
        ),
        None,
    )
    action_row = next(
        (
            row
            for row in action_rows
            if row.get("scenario_id") == SOURCE_SCENARIO
            and row.get("arm") == SOURCE_ARM
            and row.get("model_id") == SOURCE_MODEL
            and row.get("cost_mode") == "on"
        ),
        None,
    )
    if not isinstance(forecast_row, Mapping) or not isinstance(action_row, Mapping):
        raise FormalForecastPipelineError("registered S3 control ridge rows are missing")

    forecast_path = _resolve_formal_path(formal_root, forecast_row["path"])
    loaded_forecast = load_p1_forecast_artifact(
        forecast_path,
        expected_file_sha256=forecast_row["file_sha256"],
        expected_metadata=forecast_row["expected_metadata"],
        expected_bindings=forecast_row["bindings"],
    )
    source = require_authenticated_forecast_action_source(
        loaded_forecast.as_action_source(SOURCE_MODEL)
    )

    action_path = _resolve_formal_path(formal_root, action_row["path"])
    expected_action_hashes = {
        name: action_row["expected"][name]
        for name in (
            "action_primitive_payload_sha256",
            "action_primitive_schema_sha256",
            "action_primitive_content_sha256",
        )
    }
    loaded_action = load_p1_action_artifact(
        action_path,
        expected_file_sha256=action_row["file_sha256"],
        expected_metadata=action_row["expected_metadata"],
        expected_hashes=expected_action_hashes,
        expected_source_binding=action_row["source_binding"],
        authenticated_action_source=source,
        realized_returns=source.realized_returns,
        decision_block_scores=source.forecast_h4,
        decision_eligible=source.origin_mask,
        bar_available=source.bar_available,
        expected_common_mask=np.asarray(action_row["common_mask"], dtype=np.bool_),
    )
    if not loaded_action.is_authenticated:
        raise FormalForecastPipelineError("formal action artifact did not authenticate")
    compact_source = {
        "scenario_id": source.scenario_id,
        "arm": source.arm,
        "seed": int(source.seed),
        "model_id": source.model_id,
        "split_id": source.split_id,
        "support_id": source.support_id,
        "support_range": list(source.support_range),
        "fit_origin": int(source.fit_origin),
        "binding_sha256": source.binding_sha256,
        "forecast_file_sha256": source.source_hashes["forecast_file_sha256"],
        "source_body_sha256": source.source_hashes.get("source_body_sha256"),
        "forecast_rows": int(len(source.forecast_h4)),
        "forecast_finite_rows": int(np.isfinite(source.forecast_h4).sum()),
        "origin_rows": int(source.origin_mask.sum()),
        "score_rows": int(source.score_mask.sum()),
        "bar_available_rows": int(source.bar_available.sum()),
    }
    compact_action = {
        "path": str(action_path),
        "file_sha256": loaded_action.file_sha256,
        "authenticated": bool(loaded_action.is_authenticated),
        "record_count": int(loaded_action.validation["record_count"]),
        "action_primitive_schema_sha256": loaded_action.validation[
            "action_primitive_schema_sha256"
        ],
        "action_primitive_content_sha256": loaded_action.validation[
            "action_primitive_content_sha256"
        ],
        "action_primitive_payload_sha256": loaded_action.validation[
            "action_primitive_payload_sha256"
        ],
    }
    return source, loaded_action, compact_source, compact_action


def _make_formal_wfo_dataset(body: Any, dataset: Any, seq_len: int) -> WFODataset:
    timestamps = pd.DatetimeIndex(np.asarray(body.timestamps, dtype="datetime64[ns]"))
    features = pd.DataFrame(
        np.asarray(body.features, dtype=np.float64),
        index=timestamps,
        columns=[f"feature_{index}" for index in range(body.features.shape[1])],
    )
    availability = pd.DataFrame(
        {
            str(name): np.asarray(values, dtype=np.bool_)
            for name, values in body.availability.items()
        },
        index=timestamps,
    )
    features.attrs["availability"] = availability
    features.attrs["availability_interval"] = "15m"
    features.attrs["availability_include_funding"] = True
    features.attrs["availability_include_mark"] = True
    returns = pd.Series(np.asarray(dataset.returns, dtype=np.float64), index=timestamps)
    returns.attrs["availability"] = availability
    split = WFOSplit(
        fold_idx=20260904,
        train_start=timestamps[SOURCE_START_RAW],
        train_end=timestamps[SOURCE_START_RAW + SOURCE_TRAIN_END_LOCAL],
        val_start=timestamps[SOURCE_START_RAW + SOURCE_TRAIN_END_LOCAL],
        val_end=timestamps[SOURCE_START_RAW + SOURCE_VAL_END_LOCAL],
        test_start=timestamps[SOURCE_START_RAW + SOURCE_VAL_END_LOCAL],
        test_end=timestamps[SOURCE_END_RAW],
    )
    return WFODataset(
        features,
        returns,
        split,
        seq_len=seq_len,
        availability=availability,
        interval="15m",
        include_funding=True,
        include_mark=True,
    )


def _adapter_hashes(source: Any) -> dict[str, str]:
    prefix = (
        "registered-forecast-action-source-adapter/v1/"
        f"{source.binding_sha256}/{SOURCE_MODEL}/h{SOURCE_HORIZON}"
    )
    return {
        name: _sha(f"{prefix}/{name}")
        for name in (
            "checkpoint_sha256",
            "normalizer_sha256",
            "calibrator_sha256",
            "teacher_weight_sha256",
        )
    }


def _build_source_oof_artifact(
    source: Any,
    contract: ActionExecutionContract,
    hashes: Mapping[str, str],
    destination: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist a strict sparse commitment-grid view of the registered source.

    The source is a registered fixed-origin Ridge fit.  The adapter never
    refits it or claims per-row refitting: its explicit metadata says so.  It
    still emits the same causal envelope required by WM/BC/AC, with origins
    and label-complete prefixes independently auditable and unavailable rows
    left as typed NaN/false.
    """

    source_start, source_end = map(int, source.support_range)
    n_rows = source_end - source_start
    if n_rows != len(source.forecast_h4):
        raise FormalForecastPipelineError("ForecastActionSource support is not row aligned")
    starts = np.arange(0, n_rows - contract.commitment_bars + 1, SOURCE_STEP, dtype=np.int64)
    scheduled = np.zeros(n_rows, dtype=np.bool_)
    scheduled[starts] = True
    forecast = np.asarray(source.forecast_h4, dtype=np.float64)
    origin_source = np.asarray(source.origin_mask, dtype=np.bool_)
    forecast_source = np.asarray(source.forecast_h4_mask, dtype=np.bool_)
    target_source = np.asarray(source.target_h4_mask, dtype=np.bool_)
    prediction_eligibility = scheduled & origin_source & forecast_source & np.isfinite(forecast)
    training_eligibility = prediction_eligibility & target_source
    target_end = np.arange(n_rows, dtype=np.int64) + SOURCE_HORIZON + 1
    predictions = np.full((n_rows, 1), np.nan, dtype=np.float64)
    prediction_mask = np.zeros(n_rows, dtype=np.bool_)
    train_count = np.zeros(n_rows, dtype=np.int64)
    origins: list[dict[str, Any]] = []
    metadata_by_row: list[Mapping[str, Any] | None] = [None] * n_rows
    row_indices = np.arange(n_rows, dtype=np.int64)
    for t in starts:
        t_int = int(t)
        if not prediction_eligibility[t_int]:
            continue
        cutoff = t_int - SOURCE_PURGE
        eligible = np.flatnonzero(
            training_eligibility
            & (row_indices < t_int)
            & (target_end <= cutoff)
        )
        if len(eligible) > SOURCE_TRAIN_WINDOW:
            eligible = eligible[-SOURCE_TRAIN_WINDOW:]
        if len(eligible) < SOURCE_MIN_TRAIN:
            continue
        predictions[t_int, 0] = forecast[t_int]
        prediction_mask[t_int] = True
        train_count[t_int] = len(eligible)
        origins.append(
            {
                "prediction_index": t_int,
                "train_start": int(eligible[0]),
                "train_end_exclusive": int(eligible[-1]) + 1,
                "train_indices": eligible.astype(int).tolist(),
                "label_cutoff_exclusive": cutoff,
                "n_train": int(len(eligible)),
            }
        )
        metadata_by_row[t_int] = {
            "in_sample": False,
            "producer_kind": "registered_forecast_action_source_adapter",
            "source_fit_scheme": "fixed_registered_forecast_fit",
            "source_binding_sha256": source.binding_sha256,
            "forecast_file_sha256": source.source_hashes["forecast_file_sha256"],
            "fit_origin_raw": int(source.fit_origin),
            "model_id": SOURCE_MODEL,
            "horizon": SOURCE_HORIZON,
        }
    prediction_count = int(prediction_mask.sum())
    eligible_count = int(prediction_eligibility.sum())
    if prediction_count <= 0 or eligible_count <= 0:
        raise FormalForecastPipelineError("registered source has no usable commitment-grid rows")
    eligibility_provenance = {
        "source": "registered_forecast_action_source",
        "source_binding_sha256": source.binding_sha256,
        "forecast_file_sha256": source.source_hashes["forecast_file_sha256"],
        "feature_finite_guard": True,
        "scheduled_commitment_grid_step": SOURCE_STEP,
        "target_mask_applied": False,
    }
    train_eligibility_provenance = {
        **eligibility_provenance,
        "source": "registered_source_origin_and_h4_target_mask",
        "target_mask_applied": True,
    }
    raw = {
        "predictions": predictions,
        "prediction_mask": prediction_mask,
        "oof_mask": prediction_mask.copy(),
        "target_end_exclusive": target_end,
        "train_count": train_count,
        "origins": origins,
        "metadata_by_row": metadata_by_row,
        "prediction_eligibility_mask": prediction_eligibility,
        "training_label_eligibility_mask": training_eligibility,
        "prediction_eligibility": {
            "count": eligible_count,
            "eligible_rows": eligible_count,
            "n_rows": n_rows,
            "source": "registered_forecast_action_source",
            "row_eligibility_mask_supplied": True,
            "feature_finite_guard": True,
            "target_mask_applied": False,
            "provenance": eligibility_provenance,
        },
        "training_label_eligibility": {
            "count": int(training_eligibility.sum()),
            "eligible_rows": int(training_eligibility.sum()),
            "n_rows": n_rows,
            "source": "registered_source_origin_and_h4_target_mask",
            "prediction_eligibility_source": "registered_forecast_action_source",
            "valid_target_mask_supplied": True,
            "valid_target_mask_applied": True,
            "finite_target_guard": True,
            "provenance": train_eligibility_provenance,
        },
        "provenance": {
            "fit_scheme": "chronological_oof",
            "horizon": SOURCE_HORIZON,
            "purge": SOURCE_PURGE,
            "min_train_size": SOURCE_MIN_TRAIN,
            "train_window": SOURCE_TRAIN_WINDOW,
            "step": SOURCE_STEP,
            "n_rows": n_rows,
            "n_predictions": prediction_count,
            "n_origins_called": len(origins),
            "in_sample": False,
            "row_eligibility_mask_supplied": True,
            "row_eligibility_source": "registered_forecast_action_source",
            "row_eligibility_mask_source": "registered_forecast_action_source",
            "row_eligibility_applied_with_target_mask": False,
            "row_eligibility_eligible_rows": eligible_count,
            "prediction_eligibility": {
                "count": eligible_count,
                "eligible_rows": eligible_count,
                "n_rows": n_rows,
                "provenance": eligibility_provenance,
            },
            "training_label_eligibility": {
                "count": int(training_eligibility.sum()),
                "eligible_rows": int(training_eligibility.sum()),
                "n_rows": n_rows,
                "provenance": train_eligibility_provenance,
            },
            "prediction_eligibility_count": eligible_count,
            "training_label_eligibility_count": int(training_eligibility.sum()),
            "training_label_eligibility_applied_with_target_mask": True,
        },
    }
    coverage = [
        {
            "head": "forecast_mean",
            "horizon": SOURCE_HORIZON,
            "target_count": prediction_count,
            "total_target_slots": eligible_count,
            "masked_target_slots": prediction_count,
            "valid_targets": prediction_count,
            "finite_targets": prediction_count,
            "finite_target_count": prediction_count,
            "finite_masked_targets": prediction_count,
            "finite_loss_steps": 1,
            "gradient_steps": 1,
            "nonzero_gradient_steps": 1,
            "target_coverage": prediction_count / eligible_count,
            "gradient_coverage": 1.0,
            "pass": True,
            "status": "pass",
            "block_reason": None,
        }
    ]
    artifact = build_conditional_oof_artifact(
        raw,
        horizon=SOURCE_HORIZON,
        action_execution_contract=contract,
        checkpoint_sha256=hashes["checkpoint_sha256"],
        normalizer_sha256=hashes["normalizer_sha256"],
        calibrator_sha256=hashes["calibrator_sha256"],
        teacher_weight_sha256=hashes["teacher_weight_sha256"],
        coverage=coverage,
        metadata={
            "producer_kind": "registered_forecast_action_source_adapter",
            "producer_version": 1,
            "source_fit_scheme": "fixed_registered_forecast_fit",
            "source_binding_sha256": source.binding_sha256,
            "forecast_file_sha256": source.source_hashes["forecast_file_sha256"],
            "source_scenario_id": source.scenario_id,
            "source_arm": source.arm,
            "source_model_id": source.model_id,
            "registered_fit_origin_raw": int(source.fit_origin),
            "source_support_range": list(source.support_range),
            "scheduled_commitment_grid_step": SOURCE_STEP,
            "normalizer": {
                "name": "precomputed_oof",
                # This is the envelope's causal view scheme.  The actual
                # registered model fit is separately and explicitly recorded
                # as source_fit_scheme below; it is not silently relabelled
                # as a per-origin refit.
                "fit_scheme": "chronological_oof",
                "source_fit_scheme": "fixed_registered_forecast_fit",
                "in_sample": False,
            },
            "calibrator": {
                "name": "registered_source_calibration_record",
                "fit_scheme": "chronological_oof",
                "source_fit_scheme": "fixed_registered_forecast_fit",
                "in_sample": False,
            },
            "teacher_weight": {
                "name": "registered_source_action_adapter",
                "fit_scheme": "chronological_oof",
                "source_fit_scheme": "fixed_registered_forecast_fit",
                "in_sample": False,
            },
        },
    )
    split_ranges = {
        "train": (0, SOURCE_TRAIN_END_LOCAL),
        "val": (SOURCE_TRAIN_END_LOCAL, SOURCE_VAL_END_LOCAL),
        "test": (SOURCE_VAL_END_LOCAL, n_rows),
    }
    for split, (start, end) in split_ranges.items():
        indices = np.arange(start, end, dtype=np.int64)
        artifact[split] = np.array(artifact["predictions"][indices], copy=True)
        artifact[f"{split}_row_indices"] = indices
        artifact[f"{split}_mask"] = np.array(artifact["prediction_mask"][indices], copy=True)
        artifact[f"{split}_prediction_eligibility_mask"] = np.array(
            artifact["prediction_eligibility_mask"][indices], copy=True
        )
        artifact[f"{split}_training_label_eligibility_mask"] = np.array(
            artifact["training_label_eligibility_mask"][indices], copy=True
        )
        if not bool(artifact[f"{split}_mask"].any()):
            raise FormalForecastPipelineError(f"source OOF {split} view has no usable rows")
    artifact["artifact_sha256"] = hash_conditional_oof_artifact(artifact)
    artifact["artifact_hash"] = artifact["artifact_sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / "conditional_oof_artifact.json"
    write_conditional_oof_artifact(artifact_path, artifact, require_nonzero_coverage=True)
    bindings = {
        "expected_heads_horizons": [["forecast_mean", SOURCE_HORIZON]],
        "expected_hashes": dict(hashes),
        "expected_action_execution_contract": contract.to_dict(),
        "expected_action_execution_contract_hash": contract.contract_hash,
        "artifact_sha256": artifact["artifact_sha256"],
        "source_binding_sha256": source.binding_sha256,
        "forecast_file_sha256": source.source_hashes["forecast_file_sha256"],
    }
    bindings_file = destination / "conditional_bindings.json"
    _write_json(bindings_file, bindings)
    loaded = load_conditional_oof_artifact(
        artifact_path,
        expected_action_execution_contract=contract.to_dict(),
        expected_action_execution_contract_hash=contract.contract_hash,
        expected_hashes=hashes,
        expected_heads_horizons=[("forecast_mean", SOURCE_HORIZON)],
        require_nonzero_coverage=True,
    )
    details = {
        "artifact_path": str(artifact_path),
        "artifact_file_sha256": _file_sha256(artifact_path),
        "bindings_path": str(bindings_file),
        "bindings_file_sha256": _file_sha256(bindings_file),
        "artifact_sha256": loaded["artifact_sha256"],
        "rows": n_rows,
        "scheduled_rows": int(scheduled.sum()),
        "eligible_rows": eligible_count,
        "prediction_rows": prediction_count,
        "training_label_rows": int(training_eligibility.sum()),
        "split_ranges_local": {name: list(bounds) for name, bounds in split_ranges.items()},
        "source_fit_scheme": "fixed_registered_forecast_fit",
        "strict_reload": True,
        "external_hashes": dict(hashes),
    }
    return loaded, details


def _source_teacher_positions(
    source: Any,
    prediction_mask: np.ndarray,
    contract: ActionExecutionContract,
) -> np.ndarray:
    trajectory = conditional_oracle_teacher_path(
        np.asarray(source.forecast_h4, dtype=np.float64),
        contract,
        decision_eligible=np.asarray(prediction_mask, dtype=np.bool_),
        bar_available=np.asarray(source.bar_available, dtype=np.bool_),
    )
    # The action replay exposes both the post-fill inventory and the causal
    # decision path.  Absolute-position consumers (Backtest/BC) expect the
    # latter: the selected intent is visible at decision ``t`` and held over
    # the commitment block, while ``effective_positions`` changes only at
    # fill ``t+1`` and would look like an illegal mid-block jump when decoded
    # back into deltas.
    positions = np.asarray(trajectory.decision_positions, dtype=np.float64)
    if positions.shape != prediction_mask.shape or not np.isfinite(positions).all():
        raise FormalForecastPipelineError("registered source teacher path is not finite/aligned")
    return positions


def _position_path_to_intents(
    positions: np.ndarray,
    contract: ActionExecutionContract,
    *,
    decision_eligible: np.ndarray,
    bar_available: np.ndarray,
) -> np.ndarray:
    """Decode a projected absolute path into canonical candidate intents.

    ``decision_deltas_from_positions`` is intentionally strict about a raw
    delta being one of the declared candidates.  At a position bound a
    candidate such as ``-0.04`` can produce a clipped *actual* move of
    ``-0.02``; feeding the clipped absolute path back into that decoder would
    reject a perfectly valid primitive.  This adapter retains the intended
    candidate delta and lets the shared replay apply clipping once.
    """

    values = np.asarray(positions, dtype=np.float64)
    decision = np.asarray(decision_eligible, dtype=np.bool_)
    available = np.asarray(bar_available, dtype=np.bool_)
    if values.ndim != 1 or decision.shape != values.shape or available.shape != values.shape:
        raise FormalForecastPipelineError("position/intents inputs are not row aligned")
    finite = np.isfinite(values)
    deltas = np.zeros(len(values), dtype=np.float64)
    current = float(contract.p_start)
    for start in complete_decision_starts(len(values), contract):
        if not decision[start] or not finite[start]:
            continue
        target = float(values[start])
        choices = [
            (
                abs(float(np.clip(current + float(delta), contract.position_min, contract.position_max)) - target),
                abs(float(delta)),
                -float(delta),
                float(delta),
            )
            for delta in contract.candidate_deltas
        ]
        chosen = min(choices, key=lambda item: item[:3])[-1]
        deltas[start] = chosen
        candidate = float(
            np.clip(current + chosen, contract.position_min, contract.position_max)
        )
        fill = start + int(contract.execution_delay_bars)
        if fill < len(values) and available[fill]:
            current = candidate
    return deltas


def _contract_position_record(
    dataset: Any,
    positions: np.ndarray,
    contract: ActionExecutionContract,
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    """Evaluate an absolute actor/teacher path through shared delta replay."""

    returns = np.asarray(dataset.returns[start:end], dtype=np.float64)
    n_rows = len(returns)
    local_positions = np.asarray(positions, dtype=np.float64)
    decision_eligible = np.asarray(dataset.context_mask[start:end], dtype=np.bool_)
    bar_available = np.asarray(
        dataset.availability["spot_bar_observed"][start:end], dtype=np.bool_
    )
    forecast_finite = np.isfinite(local_positions)
    deltas = _position_path_to_intents(
        local_positions,
        contract,
        decision_eligible=decision_eligible,
        bar_available=bar_available,
    )
    common_mask = np.ones(
        len(complete_decision_starts(n_rows, contract)), dtype=np.bool_
    )
    trajectory = replay_action_path(
        returns,
        deltas,
        contract,
        decision_eligible=decision_eligible,
        bar_available=bar_available,
        forecast_finite_mask=forecast_finite,
        common_mask=common_mask,
    )
    hold = replay_action_path(
        returns,
        np.zeros(n_rows, dtype=np.float64),
        contract,
        decision_eligible=decision_eligible,
        bar_available=bar_available,
        forecast_finite_mask=np.ones(n_rows, dtype=np.bool_),
        common_mask=common_mask,
    )
    if not np.array_equal(trajectory.scored_mask, hold.scored_mask):
        raise FormalForecastPipelineError("action and hold score masks diverged")
    backtest = Backtest(
        returns,
        deltas,
        benchmark_positions=np.zeros(n_rows, dtype=np.float64),
        action_execution_contract=contract,
        action_positions_are_deltas=True,
        decision_eligible=decision_eligible,
        bar_available=bar_available,
        forecast_finite_mask=forecast_finite,
        common_mask=common_mask,
        expected_contract_hash=contract.contract_hash,
        require_external_contract_hash=True,
        interval="15m",
    )
    metrics = backtest.run()
    scored = np.asarray(trajectory.scored_mask, dtype=np.bool_)
    hold_scored = np.asarray(hold.scored_mask, dtype=np.bool_)
    action_pnl = np.asarray(trajectory.net_pnl[scored], dtype=np.float64)
    hold_pnl = np.asarray(hold.net_pnl[hold_scored], dtype=np.float64)
    position_stats = action_stats(
        np.asarray(trajectory.effective_positions[scored], dtype=np.float64),
        benchmark_position=contract.p_start,
    )
    return {
        "window": [start, end],
        "start_timestamp": str(dataset.timestamps[start]),
        "end_timestamp_exclusive": str(dataset.timestamps[end]) if end < len(dataset.timestamps) else None,
        "forecast_finite_rows": int(forecast_finite.sum()),
        "decision_rows": int(decision_eligible.sum()),
        "action_metric_blocks": int(np.count_nonzero(trajectory.block_masks.action_metric_mask)),
        "utility_metric_blocks": int(np.count_nonzero(trajectory.block_masks.utility_metric_mask)),
        "filled_blocks": int(trajectory.n_filled_blocks),
        "turnover": float(np.abs(trajectory.decision_deltas).sum()),
        "gross_total": float(trajectory.gross_pnl[scored].sum()),
        "cost_total": float(trajectory.transition_costs[scored].sum()),
        "net_total": float(action_pnl.sum()),
        "hold_net_total": float(hold_pnl.sum()),
        "alpha_ex_vs_hold": float(action_pnl.sum() - hold_pnl.sum()),
        "max_drawdown": float(metrics.max_drawdown),
        "hold_max_drawdown": float(abs(metrics.benchmark_max_drawdown or 0.0)),
        "total_return": float(metrics.total_return),
        "benchmark_total_return": float(metrics.benchmark_total_return or 0.0),
        "alpha_excess": float(metrics.alpha_excess or 0.0),
        "sharpe": float(metrics.sharpe),
        "benchmark_sharpe": float(metrics.benchmark_sharpe or 0.0),
        "n_trades": int(metrics.n_trades),
        "position_stats": {
            "long": float(position_stats["long"]),
            "short": float(position_stats["short"]),
            "flat": float(position_stats["flat"]),
            "mean_overlay": float(position_stats["mean"]),
            "switches": int(position_stats["switches"]),
            "avg_hold_bars": float(position_stats["avg_hold"]),
        },
        "mask_hashes": dict(trajectory.block_masks.mask_hash_registry),
        "contract_hash": contract.contract_hash,
    }


def _build_training_config(
    contract: ActionExecutionContract,
    hashes: Mapping[str, str],
    *,
    profile: str,
) -> dict[str, Any]:
    cfg = _build_pilot_config(contract, hashes)
    full = profile == "full"
    seq_len = 64 if full else 32
    cfg["data"]["seq_len"] = seq_len
    cfg["world_model"].update(
        {
            "d_model": 64 if full else 32,
            "n_heads": 4,
            "n_layers": 2 if full else 1,
            "d_ff": 128 if full else 64,
            "max_seq_len": 128,
            "encoder_hidden": 64 if full else 32,
            "encoder_layers": 2 if full else 1,
            "batch_size": 128 if full else 64,
            "max_steps": 700 if full else 4,
            # Consume the declared full-run WM budget instead of stopping at
            # the trainer's implicit validation-patience default.
            "patience": 1000 if full else 10,
            "val_max_batches": 10 if full else 1,
            "lr": 3e-4 if full else 1e-3,
            "return_horizons": [SOURCE_HORIZON],
            "return_horizon": SOURCE_HORIZON,
            "risk_horizons": [SOURCE_HORIZON],
        }
    )
    cfg["bc"].update(
        {
            "batch_size": 256 if full else 128,
            "n_epochs": 8 if full else 1,
            "lr": 3e-4 if full else 1e-3,
            # The pilot inherited zero auxiliary coefficients, which made
            # BC's objective identically zero despite running epochs.  The
            # formal-source diagnostic must train the target and execution
            # heads, then fail closed if telemetry still reports no loss.
            "target_aux_coef": 1.0,
            "trade_aux_coef": 0.5,
        }
    )
    cfg["ac"].update(
        {
            "actor_hidden": 128 if full else 64,
            "critic_hidden": 128 if full else 64,
            "ac_layers": 2 if full else 1,
            "batch_size": 128 if full else 64,
            "actor_lr": 3e-4 if full else 1e-3,
            "critic_lr": 3e-4 if full else 1e-3,
            "max_steps": 300 if full else 2,
            "horizon": SOURCE_HORIZON,
            "checkpoint_interval": 50 if full else 1,
            "restore_best_val_checkpoint": False,
            "val_patience": 0,
        }
    )
    cfg.setdefault("logging", {})["log_interval"] = 25 if full else 1
    return cfg


def _encode_actor_positions(
    body: Any,
    dataset: Any,
    wm_trainer: Any,
    actor: Any,
    contract: ActionExecutionContract,
    *,
    start: int,
    end: int,
    seq_len: int,
) -> np.ndarray:
    context_start = max(0, start - seq_len)
    encoded = wm_trainer.encode_sequence(
        np.asarray(body.features[context_start:end], dtype=np.float64),
        actions=None,
        seq_len=seq_len,
    )
    local_start = start - context_start
    raw = actor.predict_positions(
        encoded["z"][local_start:],
        encoded["h"][local_start:],
        regime_np=None,
        advantage_np=None,
        device="cpu",
    )
    if len(raw) != end - start:
        raise FormalForecastPipelineError("actor output is not row aligned")
    return np.asarray(
        project_positions_to_contract(
            raw,
            contract,
            decision_eligible=np.asarray(dataset.context_mask[start:end], dtype=np.bool_),
            bar_available=np.asarray(
                dataset.availability["spot_bar_observed"][start:end], dtype=np.bool_
            ),
            forecast_finite_mask=np.isfinite(raw),
        ),
        # Keep the contract's exact float64 candidate positions.  Converting
        # the projected grid to float32 before replay introduces deltas such
        # as -0.04000002, which correctly fail the action-contract tolerance.
        dtype=np.float64,
    )


def _evaluate_actor(
    body: Any,
    dataset: Any,
    wm_trainer: Any,
    actor: Any,
    contract: ActionExecutionContract,
    *,
    seq_len: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rolling: list[dict[str, Any]] = []
    for start, end in ROLLING_WINDOWS:
        positions = _encode_actor_positions(
            body,
            dataset,
            wm_trainer,
            actor,
            contract,
            start=start,
            end=end,
            seq_len=seq_len,
        )
        rolling.append(
            _contract_position_record(dataset, positions, contract, start=start, end=end)
        )
    outer_start, outer_end = ROLLING_WINDOWS[2][0], S3_OUTER_END
    outer_positions = _encode_actor_positions(
        body,
        dataset,
        wm_trainer,
        actor,
        contract,
        start=outer_start,
        end=outer_end,
        seq_len=seq_len,
    )
    outer = _contract_position_record(
        dataset,
        outer_positions,
        contract,
        start=outer_start,
        end=outer_end,
    )
    outer = {
        **outer,
        "same_fixed_window_as_s3_reference": True,
        "reference_window": [outer_start, outer_end],
    }
    return rolling, outer


def _aggregate_window_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise FormalForecastPipelineError("cannot aggregate an empty evaluation")
    return {
        "net_total_mean": float(np.mean([float(row["net_total"]) for row in records])),
        "hold_net_total_mean": float(np.mean([float(row["hold_net_total"]) for row in records])),
        "alpha_ex_vs_hold_mean": float(np.mean([float(row["alpha_ex_vs_hold"]) for row in records])),
        "filled_blocks_mean": float(np.mean([float(row["filled_blocks"]) for row in records])),
        "cost_total_mean": float(np.mean([float(row["cost_total"]) for row in records])),
        "sharpe_mean": float(np.mean([float(row["sharpe"]) for row in records])),
        "all_windows": len(records),
        "clean_forward_windows": max(len(records) - 1, 0),
    }


def run_formal_forecast_wm_bc_ac(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    formal_root: str | Path = DEFAULT_FORMAL_ROOT,
    profile: str = "full",
    device: str = "cpu",
    seed: int = SEED,
) -> Mapping[str, Any]:
    """Run the authenticated-source, longer-budget diagnostic."""

    if profile not in {"smoke", "full"}:
        raise FormalForecastPipelineError("profile must be 'smoke' or 'full'")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FormalForecastPipelineError("seed must be a non-negative integer")
    configure_determinism(seed)
    set_seed(seed)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    formal_root_path = Path(formal_root).resolve()
    manifest = load_runner_manifest()
    if manifest.get("results_observed") is not False:
        raise FormalForecastPipelineError("fixed manifest results_observed must remain false")
    # Worktrees intentionally do not copy the large ignored v4 cache.  Use
    # the authenticated cache beside the formal artifact run when the local
    # checkout has no cache, while keeping the runtime's pinned manifest and
    # path-snapshot checks active.
    cache_root = ROOT / "checkpoints" / "data_cache"
    if not (cache_root / "BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_features.parquet").is_file():
        sibling_cache = formal_root_path.parents[1] / "checkpoints" / "data_cache"
        cache_root = sibling_cache
    cache_tag = "BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official"
    body = load_s3_validation_body(
        root=ROOT,
        path_overrides={
            "feature_path": cache_root / f"{cache_tag}_features.parquet",
            "returns_path": cache_root / f"{cache_tag}_returns.parquet",
            "availability_path": cache_root / f"{cache_tag}_availability.parquet",
            # The frozen metadata authority is the committed repository file;
            # the ignored cache metadata is supplied only as the runtime's
            # independently checked local-cache echo.
            "metadata_path": ROOT / "docs" / "data_quality_v4_rebuild_2018_2024_metadata.json",
        },
        cache_local_metadata_path=cache_root / f"{cache_tag}_metadata.json",
    )
    dataset = build_s3_arm_dataset(body, SOURCE_ARM)
    source, loaded_action, source_summary, action_summary = _load_formal_source(formal_root_path)
    source_start, source_end = map(int, source.support_range)
    if source_end > len(dataset.returns):
        raise FormalForecastPipelineError("registered source support exceeds S3 body")
    if not np.array_equal(
        np.asarray(source.realized_returns),
        np.asarray(dataset.returns[source_start:source_end]),
    ):
        raise FormalForecastPipelineError("ForecastActionSource returns differ from authenticated S3 body")
    if not np.array_equal(
        np.asarray(source.bar_available),
        np.asarray(dataset.availability["spot_bar_observed"][source_start:source_end]),
    ):
        raise FormalForecastPipelineError("ForecastActionSource bar availability differs from S3 body")
    contract = ActionExecutionContract.canonical()
    hashes = _adapter_hashes(source)
    cfg = _build_training_config(contract, hashes, profile=profile)
    oof_bundle, adapter_summary = _build_source_oof_artifact(
        source,
        contract,
        hashes,
        destination,
    )
    prediction_mask = np.asarray(oof_bundle["prediction_mask"], dtype=np.bool_)
    source_positions = _source_teacher_positions(source, prediction_mask, contract)
    train_positions = source_positions[:SOURCE_TRAIN_END_LOCAL]
    val_positions = source_positions[SOURCE_TRAIN_END_LOCAL:SOURCE_VAL_END_LOCAL]
    test_positions = source_positions[SOURCE_VAL_END_LOCAL:]
    teacher_context = build_conditional_teacher_context(
        config=cfg,
        oof_bundle={"conditional_oof_artifact": oof_bundle},
        train_positions=train_positions,
        val_positions=val_positions,
        test_positions=test_positions,
    )
    seq_len = int(cfg["data"]["seq_len"])
    wfo_dataset = _make_formal_wfo_dataset(body, dataset, seq_len)
    predictive_bundle = build_wm_predictive_state_bundle(
        wm_trainer=None,
        wfo_dataset=wfo_dataset,
        z_train=np.zeros((len(wfo_dataset.train_features), 1), dtype=np.float32),
        h_train=np.zeros((len(wfo_dataset.train_features), 1), dtype=np.float32),
        seq_len=seq_len,
        ac_cfg=cfg["ac"],
        log_ts=log_timestamp,
        oof_bundle={"conditional_oof_artifact": oof_bundle},
    )
    if predictive_bundle is None:
        raise FormalForecastPipelineError("strict predictive-state consumer returned no bundle")
    ckpt_dir = destination / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stage_meta = {
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "body_sha256": body.body_sha256,
        "forecast_file_sha256": source.source_hashes["forecast_file_sha256"],
        "forecast_action_source_binding_sha256": source.binding_sha256,
        "source_model_id": SOURCE_MODEL,
        "source_fit_origin": int(source.fit_origin),
        "conditional_oof_artifact_sha256": oof_bundle["artifact_sha256"],
        "conditional_teacher_binding_sha256": teacher_context.binding_sha256,
        "action_execution_contract_hash": contract.contract_hash,
        "diagnostic_only": True,
    }
    wm_path = str(ckpt_dir / "world_model.pt")
    bc_path = str(ckpt_dir / "bc_actor.pt")
    ac_path = str(ckpt_dir / "ac.pt")
    ensemble, wm_trainer = prepare_world_model_stage(
        obs_dim=wfo_dataset.obs_dim,
        cfg=cfg,
        device=device,
        wm_path=wm_path,
        wfo_dataset=wfo_dataset,
        oracle_positions=train_positions,
        val_oracle_positions=val_positions,
        train_returns=wfo_dataset.train_returns,
        train_regime_probs=None,
        val_regime_probs=None,
        checkpoint_metadata={**stage_meta, "stage": "world_model"},
        conditional_teacher_context=teacher_context,
        log_ts=log_timestamp,
    )
    encoded = wm_trainer.encode_sequence(
        wfo_dataset.train_features,
        actions=None,
        seq_len=seq_len,
    )
    z_train = encoded["z"]
    h_train = encoded["h"]
    train_source_scores = np.asarray(
        source.forecast_h4[:SOURCE_TRAIN_END_LOCAL], dtype=np.float32
    )
    bc_setup = prepare_bc_setup(
        ensemble=ensemble,
        oracle_action_values=np.asarray(cfg["actions"]["values"], dtype=np.float32),
        oracle_positions=train_positions,
        oracle_values=np.nan_to_num(train_source_scores, nan=0.0),
        train_regime_probs=None,
        outcome_edge=None,
        ac_cfg=cfg["ac"],
        bc_cfg=cfg["bc"],
        reward_cfg=cfg["reward"],
        oracle_teacher_mode="registered_forecast_action_source",
    )
    actor = bc_setup["actor"]
    bc_trainer = run_bc_stage(
        actor=actor,
        ensemble=ensemble,
        bc_cfg=cfg["bc"],
        oracle_cfg={"aim_max_step": 0.08, "aim_band": 0.0},
        ac_cfg=cfg["ac"],
        reward_cfg=cfg["reward"],
        device=device,
        bc_path=bc_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=train_positions,
        train_regime_probs=None,
        oracle_soft_labels=None,
        bc_sample_quality=None,
        bc_advantage_values=None,
        train_returns=wfo_dataset.train_returns,
        checkpoint_metadata={**stage_meta, "stage": "bc"},
        conditional_teacher_context=teacher_context,
        conditional_config=cfg,
        log_ts=log_timestamp,
    )
    bc_logs = list(getattr(bc_trainer, "last_train_logs", []))
    if len(bc_logs) != int(cfg["bc"]["n_epochs"]):
        raise FormalForecastPipelineError(
            "BC did not complete the declared number of epochs"
        )
    if not any(abs(float(row.get("bc_loss", 0.0))) > 1e-12 for row in bc_logs):
        raise FormalForecastPipelineError(
            "BC loss is identically zero; refusing to treat a no-op as training"
        )
    # Evaluate the behavior-cloned actor before AC can mutate it.  Keeping
    # this on the identical evaluator separates representation/BC
    # learnability from the subsequent imagination objective.
    bc_rolling, bc_outer = _evaluate_actor(
        body,
        dataset,
        wm_trainer,
        actor,
        contract,
        seq_len=seq_len,
    )
    ac_trainer = run_ac_stage(
        actor=actor,
        ensemble=ensemble,
        cfg=cfg,
        ac_cfg=cfg["ac"],
        wm_cfg=cfg["world_model"],
        costs_cfg=cfg["costs"],
        device=device,
        ac_path=ac_path,
        z_train=z_train,
        h_train=h_train,
        oracle_positions=train_positions,
        train_regime_probs=None,
        train_advantage_values=None,
        wfo_dataset=wfo_dataset,
        wm_trainer=wm_trainer,
        seq_len=seq_len,
        val_regime_probs=None,
        val_advantage_values=None,
        val_oracle_positions=val_positions,
        ac_max_steps_cfg=resolve_ac_max_steps(cfg["ac"]),
        log_ts=log_timestamp,
        backtest_cls=Backtest,
        pnl_attribution_fn=pnl_attribution,
        action_stats_fn=action_stats,
        format_action_stats_fn=format_action_stats,
        ac_alerts_fn=_ac_alerts_ascii,
        benchmark_positions_fn=lambda length: np.ones(length, dtype=np.float64),
        benchmark_position=1.0,
        policy_score_fn=lambda metrics, stats, benchmark_position: (
            float(metrics.sharpe_delta or 0.0),
            f"sharpe_delta={float(metrics.sharpe_delta or 0.0):+.6f}",
        ),
        sequence_dataset_cls=type(wfo_dataset.train_dataset()),
        checkpoint_metadata={**stage_meta, "stage": "ac"},
        conditional_teacher_context=teacher_context,
        conditional_config=cfg,
    )
    if ac_trainer is None:
        raise FormalForecastPipelineError("AC stage unexpectedly skipped")
    rolling, outer = _evaluate_actor(
        body,
        dataset,
        wm_trainer,
        actor,
        contract,
        seq_len=seq_len,
    )
    source_teacher_record = _contract_position_record(
        dataset,
        source_positions,
        contract,
        start=source_start,
        end=source_end,
    )
    aggregate = _aggregate_window_records(rolling)
    bc_aggregate = _aggregate_window_records(bc_rolling)
    report = {
        "schema_version": 1,
        "report_id": "p1-formal-forecast-wm-bc-ac-20260904",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_revision": _git_revision(),
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_results_observed": manifest.get("results_observed"),
        "body_sha256": body.body_sha256,
        "seed": int(seed),
        "profile": profile,
        "device": device,
        "source": source_summary,
        "source_action_artifact": action_summary,
        "adapter": adapter_summary,
        "training": {
            "seq_len": seq_len,
            "train_rows": int(len(wfo_dataset.train_features)),
            "val_rows": int(len(wfo_dataset.val_features)),
            "test_rows": int(len(wfo_dataset.test_features)),
            "world_model": {
                "max_steps": int(cfg["world_model"]["max_steps"]),
                "actual_steps": int(
                    getattr(
                        wm_trainer,
                        "global_step",
                        len(getattr(wm_trainer, "loss_history", [])),
                    )
                ),
                "validation_patience": int(cfg["world_model"].get("patience", 10)),
                "batch_size": int(cfg["world_model"]["batch_size"]),
                "d_model": int(cfg["world_model"]["d_model"]),
                "n_layers": int(cfg["world_model"]["n_layers"]),
            },
            "bc": {
                "epochs": int(cfg["bc"]["n_epochs"]),
                "actual_epochs": int(len(bc_logs)),
                "final_loss": (
                    None
                    if not bc_logs
                    else float(bc_logs[-1]["bc_loss"])
                ),
                "first_loss": None if not bc_logs else float(bc_logs[0]["bc_loss"]),
                "min_loss": None if not bc_logs else float(min(row["bc_loss"] for row in bc_logs)),
                "batch_size": int(cfg["bc"]["batch_size"]),
            },
            "ac": {
                "max_steps": int(cfg["ac"]["max_steps"]),
                "actual_steps": int(getattr(ac_trainer, "global_step", 0)),
                "loss_history_rows": int(len(getattr(ac_trainer, "loss_history", []))),
                "batch_size": int(cfg["ac"]["batch_size"]),
                "horizon": int(cfg["ac"]["horizon"]),
            },
        },
        "stages": {
            "predictive_state_consumer": {
                "completed": True,
                "train_usable": int(np.count_nonzero(predictive_bundle["train_mask"])),
                "val_usable": int(np.count_nonzero(predictive_bundle["val_mask"])),
                "test_usable": int(np.count_nonzero(predictive_bundle["test_mask"])),
            },
            "world_model": {
                "completed": True,
                "checkpoint": wm_path,
                "checkpoint_sha256": _file_sha256(wm_path),
            },
            "bc": {
                "completed": True,
                "checkpoint": bc_path,
                "checkpoint_sha256": _file_sha256(bc_path),
            },
            "ac": {
                "completed": True,
                "checkpoint": ac_path,
                "checkpoint_sha256": _file_sha256(ac_path),
            },
        },
        "formal_source_support_evaluation": {
            "source_teacher": source_teacher_record,
            "source_fit_is_fixed_registered_oos": True,
            "student_uses_same_source_support": True,
        },
        "bc_evaluation": {
            "rolling_windows": bc_rolling,
            "outer_evaluation": bc_outer,
            "aggregate_mean": bc_aggregate,
        },
        "ac_evaluation": {
            "rolling_windows": rolling,
            "outer_evaluation": outer,
            "aggregate_mean": aggregate,
        },
        "rolling_windows": rolling,
        "outer_evaluation": outer,
        "aggregate_mean": aggregate,
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash,
        "selection_allowed": False,
        "threshold_revision_allowed": False,
        "promotion_allowed": False,
        "report_only": True,
        "formal_p1_outer_result": False,
        "orders_submitted": 0,
        "external_fills": 0,
        "live_money": False,
        "notes": [
            "Registered S3 zero-injection-control ForecastActionSource/ridge was loaded with external file, metadata, and binding digests.",
            "The matching formal action artifact was independently reloaded and authenticated against that source.",
            "The conditional OOF envelope is a sparse four-bar commitment-grid adapter over a fixed registered forecast fit; it does not claim per-origin refitting.",
            "Hyperparameters and budgets were fixed before this run; no outer/rolling result was used for selection.",
            "This diagnostic does not replace the preregistered P1 outer report and does not change results_observed in the manifest.",
            "No exchange connection, order, account, or live-money state was touched.",
        ],
    }
    report["report_content_sha256"] = hashlib.sha256(_json_bytes(_plain(report))).hexdigest()
    report_path = destination / "formal_forecast_wm_bc_ac_report.json"
    _write_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    report = run_formal_forecast_wm_bc_ac(
        args.output,
        formal_root=args.formal_root,
        profile=args.profile,
        device=args.device,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "report": str(args.output / "formal_forecast_wm_bc_ac_report.json"),
                "report_content_sha256": report["report_content_sha256"],
                "source_binding_sha256": report["source"]["binding_sha256"],
                "outer_alpha_ex_vs_hold": report["outer_evaluation"]["alpha_ex_vs_hold"],
                "aggregate_alpha_ex_vs_hold_mean": report["aggregate_mean"][
                    "alpha_ex_vs_hold_mean"
                ],
                "promotion_allowed": report["promotion_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
