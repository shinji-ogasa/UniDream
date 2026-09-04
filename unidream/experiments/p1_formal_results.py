"""Execute the amended, validation-only P1 result calculation.

This module is intentionally a narrow orchestration layer around the already
authenticated runner, forecast artifact, action artifact, and exact MBB
boundaries.  It never calls the report-only outer operation and never changes
the preregistration after the run starts.  Every persisted input/result is
bound by an independent file digest and the run ledger records the immutable
manifest/registry digests used for the calculation.

The module is not a model-selection tool.  It materializes exactly the 52
registered validation arms, the action artifacts required by the 16 frozen
comparisons, the fixed L={8,16,32} MBB draw artifacts, and then computes the
registered comparison family with the production reducers.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from unidream.eval.action_execution import ActionExecutionContract, complete_decision_starts

from .action_primitives import (
    expected_authenticated_action_metadata,
    produce_authenticated_action_primitive_grid,
)
from .p1_action_artifact import load_p1_action_artifact, save_p1_action_artifact
from .p1_mbb import (
    P1MBBIndexArtifact,
    P1MBBResultArtifact,
    P1_MBB_BLOCK_LENGTHS,
    bootstrap_p1_action_metric,
    bootstrap_p1_action_metric_seed_sensitivity,
    bootstrap_p1_metric,
    bootstrap_p1_metric_seed_sensitivity,
    build_p1_mbb_index_artifact,
    load_p1_mbb_index_artifact,
    load_p1_mbb_result,
    p1_mask_sha256 as mbb_mask_sha256,
    save_p1_mbb_index_artifact,
    save_p1_mbb_result_artifact,
)
from .p1_recovery_runner import (
    build_s3_arm_dataset,
    build_synthetic_dataset,
    build_runner_plan,
    fit_model_at_origin,
    load_s3_validation_body,
    run_s3_validation_fits,
)
from .p1_result_registry import P1ResultRegistry
from .p1_statistical_gates import (
    evaluate_s0_safety_bounds,
    holm_bonferroni_fixed_family,
    wilson_score_interval,
)
from .p1_validation_forecast import (
    P1ForecastContract,
    P1_SCENARIO_ARMS,
    P1_SYNTHETIC_SCENARIO_ARMS,
    authenticate_p1_forecast_contract,
    build_p1_forecast_artifact,
    expected_metadata_for_arm,
    load_p1_forecast_artifact,
    save_p1_forecast_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "codex_outputs" / "p1_formal_run_20260904"
SEEDS = tuple(range(20_260_830, 20_260_840))
ACTION_COST_MODE = "on"
ACTION_MODELS = {
    ("S0", "zero_signal"): ("ridge", "persistence_last_observed"),
    ("S1", "known_high_snr_dgp"): ("ridge",),
    ("S2-high", "high"): ("ridge",),
    ("S2-medium", "medium"): ("ridge",),
    ("S2-low", "low"): ("ridge",),
    ("S3", "injected"): ("ridge",),
    ("S3", "zero_injection_control"): ("ridge",),
}
ACTION_HASH_FIELDS = (
    "action_primitive_payload_sha256",
    "action_primitive_schema_sha256",
    "action_primitive_content_sha256",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plain(value: Any) -> Any:
    """Convert immutable/NumPy values into finite JSON values."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite value cannot enter the run ledger")
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, P1MBBIndexArtifact):
        return value.artifact_sha256
    raise TypeError(f"unsupported ledger value: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_sha(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded + b"\n")
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _fit_grid(dataset: Any, spec: Any, contract: P1ForecastContract) -> dict[tuple[int, str, str], Any]:
    fits: dict[tuple[int, str, str], Any] = {}
    for horizon in contract.horizons:
        for model_id, task in contract.model_task_keys:
            fits[(horizon, model_id, task)] = fit_model_at_origin(
                dataset,
                model_id,
                spec.fit_origin,
                horizon,
                task=task,
                prediction_range=spec.support_range,
                train_start=spec.train_start,
            )
    return fits


def _fit_record(validation: Mapping[str, Any], model_id: str, task: str) -> Mapping[str, Any]:
    # JSON decoding preserves the producer's typed fit keys as tuples in the
    # loaded validation view, while the provenance binding uses its canonical
    # string spelling.  Accept only those two equivalent representations.
    key = f"h4::{model_id}::{task}"
    fits = validation.get("fits")
    if not isinstance(fits, Mapping):
        raise RuntimeError(f"loaded forecast is missing fit mapping for {key}")
    record = fits.get((4, model_id, task), fits.get(key))
    if record is None:
        raise RuntimeError(f"loaded forecast is missing {key}")
    if not isinstance(record, Mapping):
        raise RuntimeError(f"loaded forecast fit {key} is malformed")
    return record


def _forecast_arrays(loaded: Any, model_id: str, task: str) -> dict[str, np.ndarray]:
    validation = loaded.validation
    target = np.asarray(validation["targets"], dtype="<f8")[:, 1]
    target_mask = np.asarray(validation["target_mask"], dtype=np.bool_)[:, 1]
    labels = np.asarray(validation["binary_labels"], dtype=np.int8)[:, 1]
    origin = np.asarray(validation["origin_mask"], dtype=np.bool_)
    score = np.asarray(validation["score_eligible_mask"], dtype=np.bool_)
    record = _fit_record(validation, model_id, task)
    predictions = np.asarray(record["predictions"], dtype="<f8")
    prediction_mask = np.asarray(record["prediction_mask"], dtype=np.bool_)
    if predictions.shape != target.shape or prediction_mask.shape != target.shape:
        raise RuntimeError(f"forecast fit {model_id}/{task} is not support aligned")
    return {
        "target": target,
        "target_mask": target_mask,
        "labels": labels,
        "origin": origin,
        "score": score,
        "predictions": predictions,
        "prediction_mask": prediction_mask,
    }


def _se(predictions: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(len(target), np.nan, dtype="<f8")
    result[mask] = (predictions[mask] - target[mask]) ** 2
    return result


def _logloss(predictions: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(len(labels), np.nan, dtype="<f8")
    p = np.clip(predictions[mask], 1e-6, 1.0 - 1e-6)
    y = labels[mask].astype(np.float64)
    result[mask] = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return result


def _composite_forecast_digest(refs: list[Mapping[str, Any]], fit_keys: list[str]) -> str:
    rows: list[dict[str, Any]] = []
    for ref, key in zip(refs, fit_keys, strict=True):
        rows.append(
            {
                "forecast_file_sha256": ref["file_sha256"],
                "fit_key": key,
                "fit_record_sha256": ref["bindings"]["fit_record_sha256"][key],
            }
        )
    return _json_sha(rows)


@dataclass
class FormalState:
    contract: P1ForecastContract
    plan: Any
    output: Path
    forecast_refs: dict[tuple[str, str, int], dict[str, Any]]
    action_refs: dict[tuple[str, str, int, str, str], dict[str, Any]]
    index_refs: dict[tuple[str, int, int], dict[str, Any]]
    forecast_data: dict[tuple[str, str, int], dict[str, Any]]
    action_loaded: dict[tuple[str, str, int, str, str], Any]
    comparison_results: dict[str, dict[str, Any]]
    # Keep the complete in-memory reducer output until the gates have run.
    # The typed result artifact deliberately omits the bootstrap vector from
    # its metadata, while S0's preregistered safety gate must re-use each
    # L-specific vector.  This also avoids attempting to recover inferential
    # samples from a summary-only ledger row.
    result_objects: dict[str, dict[str, Any]]


def _materialize_forecast(
    state: FormalState,
    scenario_id: str,
    arm: str,
    seed: int,
    dataset: Any,
    *,
    fits: Mapping[tuple[int, str, str], Any] | None = None,
) -> None:
    spec = state.contract.spec(scenario_id, arm)
    fit_grid = _fit_grid(dataset, spec, state.contract) if fits is None else dict(fits)
    artifact = build_p1_forecast_artifact(state.contract, spec, dataset, fit_grid)
    path = state.output / "forecasts" / f"{scenario_id}__{arm}__{seed}.json"
    expected_metadata = dict(expected_metadata_for_arm(state.contract, scenario_id, arm, seed))
    file_sha = save_p1_forecast_artifact(path, artifact, expected_metadata=expected_metadata)
    bindings = _plain(dict(artifact._bindings))
    loaded = load_p1_forecast_artifact(
        path,
        expected_file_sha256=file_sha,
        expected_metadata=expected_metadata,
        expected_bindings=bindings,
    )
    if not loaded.promotion_allowed:
        raise RuntimeError(f"forecast arm is not promotion-eligible: {scenario_id}/{arm}/{seed}")
    validation = loaded.validation
    coverage = _plain(validation.get("coverage", {}))
    ref = {
        "scenario_id": scenario_id,
        "arm": arm,
        "seed": seed,
        "path": _relative(path),
        "file_sha256": file_sha,
        "expected_metadata": expected_metadata,
        "bindings": bindings,
        "coverage": coverage,
        "validation_status": validation.get("status"),
        "support_count": int(validation.get("support_count", 0)),
    }
    key = (scenario_id, arm, seed)
    state.forecast_refs[key] = ref
    # Keep compact arrays only; the full loaded forecast is reloaded at the
    # action boundary when needed, avoiding a large resident object graph.
    state.forecast_data[key] = {
        "h4_target": np.asarray(validation["targets"], dtype="<f8")[:, 1],
        "h4_target_mask": np.asarray(validation["target_mask"], dtype=np.bool_)[:, 1],
        "h4_labels": np.asarray(validation["binary_labels"], dtype=np.int8)[:, 1],
        "origin_mask": np.asarray(validation["origin_mask"], dtype=np.bool_),
        "score_mask": np.asarray(validation["score_eligible_mask"], dtype=np.bool_),
        "forecast_file_sha256": file_sha,
        "fit_record_sha256": dict(bindings["fit_record_sha256"]),
    }
    for model_id, task in (
        ("zero_return", "continuous"),
        ("zero_return", "binary"),
        ("ridge", "continuous"),
        ("logistic", "binary"),
    ):
        record = _fit_record(validation, model_id, task)
        state.forecast_data[key][f"{model_id}::{task}::predictions"] = np.asarray(
            record["predictions"], dtype="<f8"
        )
        state.forecast_data[key][f"{model_id}::{task}::mask"] = np.asarray(
            record["prediction_mask"], dtype=np.bool_
        )
    _write_json(state.output / "ledgers" / "forecasts.json", list(state.forecast_refs.values()))


def _load_forecast(state: FormalState, key: tuple[str, str, int]) -> Any:
    ref = state.forecast_refs[key]
    return load_p1_forecast_artifact(
        ROOT / ref["path"],
        expected_file_sha256=ref["file_sha256"],
        expected_metadata=ref["expected_metadata"],
        expected_bindings=ref["bindings"],
    )


def _action_expected(artifact: Mapping[str, Any], source_file_sha: str, action_file_sha: str) -> dict[str, str]:
    expected = {field: str(artifact["header"][field]) for field in ACTION_HASH_FIELDS}
    expected["source_result_sha256"] = source_file_sha
    expected["source_action_file_sha256"] = action_file_sha
    return expected


def _materialize_actions_for_arm(state: FormalState, scenario_id: str, arm: str, seed: int) -> None:
    models = ACTION_MODELS[(scenario_id, arm)]
    forecast = _load_forecast(state, (scenario_id, arm, seed))
    for model_id in models:
        source = forecast.as_action_source(model_id)
        contract = ActionExecutionContract.canonical()
        starts = complete_decision_starts(len(source.timestamps), contract)
        block_common = np.asarray(source.common_mask, dtype=np.bool_)[np.asarray(starts, dtype=np.int64)]
        metadata = dict(
            expected_authenticated_action_metadata(
                source,
                cost_mode=ACTION_COST_MODE,
                paired_common_mask=block_common,
            )
        )
        action = produce_authenticated_action_primitive_grid(
            action_source=source,
            cost_mode=ACTION_COST_MODE,
            paired_common_mask=block_common,
            expected_metadata=metadata,
        )
        path = state.output / "actions" / f"{scenario_id}__{arm}__{seed}__{model_id}__on.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        action_file_sha = save_p1_action_artifact(
            path,
            action,
            expected_metadata=metadata,
            expected_source_binding=action["header"]["source_binding"],
            authenticated_action_source=source,
            realized_returns=source.realized_returns,
            decision_block_scores=source.forecast_h4,
            decision_eligible=source.origin_mask,
            bar_available=source.bar_available,
            expected_common_mask=block_common,
        )
        expected = _action_expected(action, state.forecast_refs[(scenario_id, arm, seed)]["file_sha256"], action_file_sha)
        loaded = load_p1_action_artifact(
            path,
            expected_file_sha256=action_file_sha,
            expected_metadata=metadata,
            expected_hashes={field: expected[field] for field in ACTION_HASH_FIELDS},
            expected_source_binding=action["header"]["source_binding"],
            authenticated_action_source=source,
            realized_returns=source.realized_returns,
            decision_block_scores=source.forecast_h4,
            decision_eligible=source.origin_mask,
            bar_available=source.bar_available,
            expected_common_mask=block_common,
        )
        if not loaded.is_authenticated:
            raise RuntimeError(f"action artifact did not authenticate: {path}")
        key = (scenario_id, arm, seed, model_id, ACTION_COST_MODE)
        state.action_loaded[key] = loaded
        state.action_refs[key] = {
            "scenario_id": scenario_id,
            "arm": arm,
            "seed": seed,
            "model_id": model_id,
            "cost_mode": ACTION_COST_MODE,
            "path": _relative(path),
            "file_sha256": action_file_sha,
            "expected_metadata": _plain(metadata),
            "expected": expected,
            "source_binding": _plain(action["header"]["source_binding"]),
            "common_mask": block_common.astype(bool).tolist(),
            "common_mask_sha256": mbb_mask_sha256(block_common),
            "mask_hashes": _plain(dict(loaded.as_mbb_input().mask_hashes)),
            "record_count": len(action["records"]),
        }
    _write_json(state.output / "ledgers" / "actions.json", list(state.action_refs.values()))


def _materialize_index(state: FormalState, unit: str, ordinal: int, block_length: int, n: int) -> dict[str, Any]:
    key = (unit, ordinal, block_length)
    if key in state.index_refs:
        return state.index_refs[key]
    raw = build_p1_mbb_index_artifact(
        n,
        unit=unit,
        support_id="synthetic_validation" if unit.startswith("synthetic") else "s3_validation",
        seed_ordinal=ordinal,
        block_length=block_length,
    )
    path = state.output / "indices" / f"{unit}__seed{ordinal}__L{block_length}.npz"
    file_sha = save_p1_mbb_index_artifact(path, raw)
    loaded = load_p1_mbb_index_artifact(
        path,
        expected_artifact_sha256=raw.artifact_sha256,
        expected_file_sha256=file_sha,
    )
    ref = {
        "unit": unit,
        "support_id": raw.support_id,
        "seed_ordinal": ordinal,
        "block_length": block_length,
        "n": n,
        "path": _relative(path),
        "artifact_sha256": loaded.artifact_sha256,
        "file_sha256": loaded.file_sha256,
        "starts_sha256": loaded.starts_sha256,
        "artifact": loaded,
    }
    state.index_refs[key] = ref
    return ref


def _index_maps(state: FormalState, unit: str, n: int, ordinals: tuple[int, ...]) -> tuple[dict[int, dict[int, P1MBBIndexArtifact]], dict[int, dict[int, str]], dict[int, dict[int, str]], dict[int, dict[int, str]]]:
    artifacts: dict[int, dict[int, P1MBBIndexArtifact]] = {}
    expected: dict[int, dict[int, str]] = {}
    expected_file: dict[int, dict[int, str]] = {}
    paths: dict[int, dict[int, str]] = {}
    for length in P1_MBB_BLOCK_LENGTHS:
        artifacts[length] = {}
        expected[length] = {}
        expected_file[length] = {}
        paths[length] = {}
        for ordinal in ordinals:
            ref = _materialize_index(state, unit, ordinal, length, n)
            artifacts[length][ordinal] = ref["artifact"]
            expected[length][ordinal] = ref["artifact_sha256"]
            expected_file[length][ordinal] = ref["file_sha256"]
            paths[length][ordinal] = ref["path"]
    return artifacts, expected, expected_file, paths


def _seed_ordinal(seed: int) -> int:
    try:
        return SEEDS.index(seed)
    except ValueError as exc:
        raise RuntimeError(f"unregistered synthetic seed: {seed}") from exc


def _forecast_payload_for_pair(
    state: FormalState,
    candidate_key: tuple[str, str, int],
    baseline_key: tuple[str, str, int],
    metric: str,
    level_metric: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cand = state.forecast_data[candidate_key]
    base = state.forecast_data[baseline_key]
    target_mask = cand["h4_target_mask"] & base["h4_target_mask"]
    origin = cand["origin_mask"] & base["origin_mask"]
    if metric == "mse_delta":
        cp = cand["ridge::continuous::predictions"]
        bp = base["zero_return::continuous::predictions"]
        cm = cand["ridge::continuous::mask"]
        bm = base["zero_return::continuous::mask"]
        mask = target_mask & origin & cm & bm
        payload = {
            "mask": mask,
            "candidate_mask": mask,
            "baseline_mask": mask,
            "candidate_se": _se(cp, cand["h4_target"], mask),
            "baseline_se": _se(bp, base["h4_target"], mask),
        }
    elif metric == "skill":
        a_model = cand["ridge::continuous::predictions"]
        a_zero = cand["zero_return::continuous::predictions"]
        b_model = base["ridge::continuous::predictions"]
        b_zero = base["zero_return::continuous::predictions"]
        am = cand["ridge::continuous::mask"] & cand["zero_return::continuous::mask"]
        bm = base["ridge::continuous::mask"] & base["zero_return::continuous::mask"]
        mask = target_mask & origin & am & bm
        payload = {
            "mask": mask,
            "candidate_mask": mask,
            "baseline_mask": mask,
            "level_a_model_se": _se(a_model, cand["h4_target"], mask),
            "level_a_zero_se": _se(a_zero, cand["h4_target"], mask),
            "level_b_model_se": _se(b_model, base["h4_target"], mask),
            "level_b_zero_se": _se(b_zero, base["h4_target"], mask),
        }
    elif metric == "logloss":
        a_model = cand["logistic::binary::predictions"]
        b_model = base["logistic::binary::predictions"]
        am = cand["logistic::binary::mask"]
        bm = base["logistic::binary::mask"]
        mask = target_mask & origin & am & bm & (cand["h4_labels"] >= 0) & (base["h4_labels"] >= 0)
        payload = {
            "mask": mask,
            "candidate_mask": mask,
            "baseline_mask": mask,
            "level_a_values": _logloss(a_model, cand["h4_labels"], mask),
            "level_b_values": _logloss(b_model, base["h4_labels"], mask),
        }
    else:
        raise RuntimeError(f"unknown forecast pair metric {metric}/{level_metric}")
    fit_keys = [
        "h4::ridge::continuous",
        "h4::zero_return::continuous",
    ]
    if metric == "logloss":
        fit_keys = ["h4::logistic::binary", "h4::logistic::binary"]
    provenance = {
        "kind": "forecast",
        "common_mask_sha256": mbb_mask_sha256(payload["mask"]),
        "common_mask_field": "common_mask",
        "forecast_artifact_sha256": _composite_forecast_digest(
            [state.forecast_refs[candidate_key], state.forecast_refs[baseline_key]], fit_keys
        ),
        "forecast_result_sha256": _composite_forecast_digest(
            [state.forecast_refs[candidate_key], state.forecast_refs[baseline_key]], fit_keys
        ),
    }
    expected = {
        "provenance": provenance,
        "expected_common_mask_sha256": provenance["common_mask_sha256"],
        "expected_common_mask_field": "common_mask",
        "expected_forecast_artifact_sha256": provenance["forecast_artifact_sha256"],
        "expected_forecast_result_sha256": provenance["forecast_result_sha256"],
    }
    return payload, expected


def _forecast_comparison(state: FormalState, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cid = str(row["comparison_id"])
    metric_label = str(row["metric"])
    direction = str(row["direction"])
    if cid.startswith("S1__ridge__mse"):
        pairs = [("S1", "known_high_snr_dgp", seed, "S1", "known_high_snr_dgp", seed) for seed in SEEDS]
        internal, level_metric = "mse_delta", None
    elif "mse_skill" in cid and cid.startswith("S2"):
        if "high_vs_medium" in cid:
            left_scenario, left_arm, right_scenario, right_arm = (
                "S2-high", "high", "S2-medium", "medium"
            )
        else:
            left_scenario, left_arm, right_scenario, right_arm = (
                "S2-medium", "medium", "S2-low", "low"
            )
        pairs = [
            (left_scenario, left_arm, seed, right_scenario, right_arm, seed)
            for seed in SEEDS
        ]
        internal, level_metric = "skill", "skill"
    elif "logistic__log_loss" in cid:
        if "high_vs_medium" in cid:
            left_scenario, left_arm, right_scenario, right_arm = (
                "S2-high", "high", "S2-medium", "medium"
            )
        else:
            left_scenario, left_arm, right_scenario, right_arm = (
                "S2-medium", "medium", "S2-low", "low"
            )
        pairs = [
            (left_scenario, left_arm, seed, right_scenario, right_arm, seed)
            for seed in SEEDS
        ]
        internal, level_metric = "logloss", "logloss"
    else:
        raise RuntimeError(f"unsupported forecast comparison: {cid} / {metric_label}")
    seed_inputs: dict[int, dict[str, Any]] = {}
    provenance_by_seed: dict[int, dict[str, Any]] = {}
    for ordinal, (cs, ca, seed, bs, ba, bseed) in enumerate(pairs):
        payload, provenance = _forecast_payload_for_pair(
            state,
            (cs, ca, seed),
            (bs, ba, bseed),
            internal,
            level_metric,
        )
        seed_inputs[ordinal] = payload
        provenance_by_seed[ordinal] = provenance
    unit = "synthetic_forecast"
    n = len(seed_inputs[0]["mask"])
    artifacts, expected_index, expected_file, paths = _index_maps(state, unit, n, tuple(range(10)))
    result = bootstrap_p1_metric_seed_sensitivity(
        "s2_contrast" if internal in {"skill", "logloss"} else internal,
        unit=unit,
        support_id="synthetic_validation",
        seed_inputs=seed_inputs,
        level_direction=direction if internal in {"skill", "logloss"} else None,
        level_metric=level_metric,
        provenance_by_seed=provenance_by_seed,
        index_artifacts_by_block_length=artifacts,
        expected_index_artifact_sha256_by_block_length=expected_index,
        expected_index_artifact_file_sha256_by_block_length=expected_file,
        index_artifact_paths_by_block_length=paths,
    )
    return _add_sensitivity_summary(result), {
        "internal_metric": "s2_contrast" if internal in {"skill", "logloss"} else internal,
        "level_metric": level_metric,
    }


def _action_key_for_trial(state: FormalState, trial_id: str, seed: int) -> tuple[str, str, int, str, str]:
    parts = trial_id.rsplit("__", 2)
    if len(parts) != 3:
        raise RuntimeError(f"malformed action trial id: {trial_id}")
    prefix, model_id, cost_mode = parts
    if cost_mode != ACTION_COST_MODE:
        raise RuntimeError(f"formal action reducer requires cost-on trial: {trial_id}")
    if prefix.startswith("S3-injected"):
        scenario_id, arm = "S3", "injected"
    elif prefix.startswith("S3-control"):
        scenario_id, arm = "S3", "zero_injection_control"
    else:
        scenario_id = prefix
        arm = {
            "S0": "zero_signal",
            "S1": "known_high_snr_dgp",
            "S2-high": "high",
            "S2-medium": "medium",
            "S2-low": "low",
        }[scenario_id]
    key = (scenario_id, arm, seed, model_id, cost_mode)
    if key not in state.action_loaded:
        raise RuntimeError(f"action artifact was not materialized for {trial_id}/{seed}")
    return key


def _action_expectations_by_seed(state: FormalState, keys: Mapping[int, tuple[str, str, int, str, str]]) -> dict[int, dict[str, str]]:
    return {ordinal: dict(state.action_refs[key]["expected"]) for ordinal, key in keys.items()}


def _action_masks_by_seed(state: FormalState, keys: Mapping[int, tuple[str, str, int, str, str]]) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    masks = {ordinal: np.asarray(state.action_refs[key]["common_mask"], dtype=np.bool_) for ordinal, key in keys.items()}
    return masks, {ordinal: mbb_mask_sha256(mask) for ordinal, mask in masks.items()}


def _action_comparison(state: FormalState, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cid = str(row["comparison_id"])
    metric_label = str(row["metric"])
    direction = str(row["direction"])
    if cid.startswith("S0__"):
        model_id = "ridge" if "ridge" in cid else "persistence_last_observed"
        candidate_keys = {i: _action_key_for_trial(state, f"S0__{model_id}__on", seed) for i, seed in enumerate(SEEDS)}
        baseline_keys = None
        internal, level_metric = "policy_utility_delta", None
    elif cid.startswith("S1__ridge__utility"):
        candidate_keys = {i: _action_key_for_trial(state, "S1__ridge__on", seed) for i, seed in enumerate(SEEDS)}
        baseline_keys = None
        internal, level_metric = "policy_utility_delta", None
    elif cid.startswith("S2__"):
        left, right = ("S2-high", "S2-medium") if "high_vs_medium" in cid else ("S2-medium", "S2-low")
        left_arm = "high" if left.endswith("high") else "medium"
        right_arm = "medium" if right.endswith("medium") else "low"
        left_trial = f"{left}__ridge__on"
        right_trial = f"{right}__ridge__on"
        candidate_keys = {i: _action_key_for_trial(state, left_trial, seed) for i, seed in enumerate(SEEDS)}
        baseline_keys = {i: _action_key_for_trial(state, right_trial, seed) for i, seed in enumerate(SEEDS)}
        internal = "s2_contrast"
        if "normalized_regret" in cid:
            level_metric = "normalized_regret"
        elif "agreement" in cid:
            level_metric = "agreement"
        else:
            level_metric = "policy_utility_delta"
    else:
        raise RuntimeError(f"unsupported synthetic action comparison: {cid} / {metric_label}")
    candidate_artifacts = {ordinal: state.action_loaded[key] for ordinal, key in candidate_keys.items()}
    baseline_artifacts = None if baseline_keys is None else {ordinal: state.action_loaded[key] for ordinal, key in baseline_keys.items()}
    candidate_expected = _action_expectations_by_seed(state, candidate_keys)
    baseline_expected = None if baseline_keys is None else _action_expectations_by_seed(state, baseline_keys)
    candidate_masks, candidate_digests = _action_masks_by_seed(state, candidate_keys)
    if baseline_keys is None:
        common_masks, common_digests = candidate_masks, candidate_digests
    else:
        baseline_masks, _ = _action_masks_by_seed(state, baseline_keys)
        common_masks = {i: candidate_masks[i] & baseline_masks[i] for i in range(10)}
        common_digests = {i: p1_mask_sha256(common_masks[i]) for i in range(10)}
    artifacts, expected_index, expected_file, paths = _index_maps(
        state, "synthetic_action", len(common_masks[0]), tuple(range(10))
    )
    result = bootstrap_p1_action_metric_seed_sensitivity(
        internal,
        unit="synthetic_action",
        support_id="synthetic_validation",
        candidate_action_artifacts=candidate_artifacts,
        candidate_expected_by_seed=candidate_expected,
        common_masks_by_seed=common_masks,
        expected_common_mask_sha256_by_seed=common_digests,
        baseline_action_artifacts=baseline_artifacts,
        baseline_expected_by_seed=baseline_expected,
        level_direction=direction if internal == "s2_contrast" else None,
        level_metric=level_metric,
        index_artifacts_by_block_length=artifacts,
        expected_index_artifact_sha256_by_block_length=expected_index,
        expected_index_artifact_file_sha256_by_block_length=expected_file,
        index_artifact_paths_by_block_length=paths,
    )
    return _add_sensitivity_summary(result), {"internal_metric": internal, "level_metric": level_metric}


def _s3_forecast_comparison(state: FormalState, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    injected_key = ("S3", "injected", 20_260_830)
    control_key = ("S3", "zero_injection_control", 20_260_830)
    a = state.forecast_data[injected_key]
    b = state.forecast_data[control_key]
    ridge_a = a["ridge::continuous::predictions"]
    zero_a = a["zero_return::continuous::predictions"]
    ridge_b = b["ridge::continuous::predictions"]
    zero_b = b["zero_return::continuous::predictions"]
    mask = (
        a["h4_target_mask"]
        & b["h4_target_mask"]
        & a["origin_mask"]
        & b["origin_mask"]
        & a["ridge::continuous::mask"]
        & b["ridge::continuous::mask"]
        & a["zero_return::continuous::mask"]
        & b["zero_return::continuous::mask"]
    )
    arrays = {
        "injected_model_se": _se(ridge_a, a["h4_target"], mask),
        "injected_zero_se": _se(zero_a, a["h4_target"], mask),
        "control_model_se": _se(ridge_b, b["h4_target"], mask),
        "control_zero_se": _se(zero_b, b["h4_target"], mask),
    }
    refs = [state.forecast_refs[injected_key], state.forecast_refs[control_key]]
    digest = _composite_forecast_digest(refs, ["h4::ridge::continuous", "h4::ridge::continuous"])
    provenance = {
        "kind": "forecast",
        "common_mask_sha256": mbb_mask_sha256(mask),
        "common_mask_field": "common_mask",
        "forecast_artifact_sha256": digest,
        "forecast_result_sha256": digest,
    }
    children: dict[int, dict[str, Any]] = {}
    indexes: dict[int, dict[str, Any]] = {}
    for length in P1_MBB_BLOCK_LENGTHS:
        ref = _materialize_index(state, "s3_forecast", 0, length, len(mask))
        child = bootstrap_p1_metric(
            "s3_skill_did",
            artifact=ref["artifact"],
            mask=mask,
            candidate_mask=mask,
            baseline_mask=mask,
            provenance=provenance,
            expected_common_mask_sha256=provenance["common_mask_sha256"],
            expected_common_mask_field="common_mask",
            expected_forecast_artifact_sha256=digest,
            expected_forecast_result_sha256=digest,
            **arrays,
        )
        children[length] = child
        indexes[length] = ref
    return _sensitivity_envelope(children, indexes), {"internal_metric": "s3_skill_did", "level_metric": None}


def _s3_action_comparison(state: FormalState, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    injected = _action_key_for_trial(state, "S3-injected__ridge__on", 20_260_830)
    control = _action_key_for_trial(state, "S3-control__ridge__on", 20_260_830)
    candidate = state.action_loaded[injected]
    baseline = state.action_loaded[control]
    candidate_expected = state.action_refs[injected]["expected"]
    baseline_expected = state.action_refs[control]["expected"]
    common = np.asarray(state.action_refs[injected]["common_mask"], dtype=np.bool_) & np.asarray(
        state.action_refs[control]["common_mask"], dtype=np.bool_
    )
    children: dict[int, dict[str, Any]] = {}
    indexes: dict[int, dict[str, Any]] = {}
    for length in P1_MBB_BLOCK_LENGTHS:
        ref = _materialize_index(state, "s3_action", 0, length, len(common))
        children[length] = bootstrap_p1_action_metric(
            "s3_utility_did",
            artifact=ref["artifact"],
            candidate_action_artifact=candidate,
            candidate_expected=candidate_expected,
            baseline_action_artifact=baseline,
            baseline_expected=baseline_expected,
            common_mask=common,
            expected_common_mask_sha256=mbb_mask_sha256(common),
            direction="positive",
        )
        indexes[length] = ref
    return _sensitivity_envelope(children, indexes), {"internal_metric": "s3_utility_did", "level_metric": None}


def _sensitivity_envelope(children: Mapping[int, Mapping[str, Any]], indexes: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    primary = dict(children[16])
    envelope = dict(primary)
    envelope.update(
        {
            "block_lengths": list(P1_MBB_BLOCK_LENGTHS),
            "per_block_length": {length: dict(children[length]) for length in P1_MBB_BLOCK_LENGTHS},
            "raw_p": max(float(children[length]["p_value"]) for length in P1_MBB_BLOCK_LENGTHS),
            "raw_p_rule": "max(p_block_length_8,p_block_length_16,p_block_length_32)",
            "index_artifacts": {length: indexes[length]["artifact"] for length in P1_MBB_BLOCK_LENGTHS},
            "index_artifact_expected_sha256_by_block_length": {
                length: indexes[length]["artifact_sha256"] for length in P1_MBB_BLOCK_LENGTHS
            },
            "index_artifact_file_sha256_by_block_length": {
                length: indexes[length]["file_sha256"] for length in P1_MBB_BLOCK_LENGTHS
            },
            "index_artifact_expected_file_sha256_by_block_length": {
                length: indexes[length]["file_sha256"] for length in P1_MBB_BLOCK_LENGTHS
            },
            "index_artifact_bindings": {
                str(length): {
                    "artifact_sha256": indexes[length]["artifact_sha256"],
                    "starts_sha256": indexes[length]["starts_sha256"],
                    "expected_artifact_sha256": indexes[length]["artifact_sha256"],
                    "file_sha256": indexes[length]["file_sha256"],
                    "expected_file_sha256": indexes[length]["file_sha256"],
                    "source_path": indexes[length]["path"],
                }
                for length in P1_MBB_BLOCK_LENGTHS
            },
        }
    )
    return envelope


def _add_sensitivity_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the fixed L=16 child as the typed envelope's primary statistic.

    The seed-sensitivity reducers intentionally return only their L-specific
    children and raw-p rule.  Result persistence, however, has one typed
    bootstrap vector at the top level.  Copying the preregistered L=16 child
    statistics (without copying its nested metadata) makes the envelope both
    directly reportable and loadable while retaining every L child below.
    """
    if not isinstance(result, Mapping):
        raise RuntimeError("sensitivity reducer did not return a mapping")
    nested = result.get("per_block_length")
    if not isinstance(nested, Mapping) or 16 not in nested:
        raise RuntimeError("sensitivity reducer is missing its fixed L=16 child")
    primary = nested[16]
    if not isinstance(primary, Mapping) or "bootstrap_values" not in primary:
        raise RuntimeError("sensitivity L=16 child lacks bootstrap values")
    envelope = dict(result)
    for field in (
        "unit",
        "support_id",
        "replicates",
        "point_estimate",
        "favorable_point_estimate",
        "ci",
        "p_value",
        "p_value_formula",
        "bootstrap_values",
    ):
        if field in primary:
            envelope[field] = primary[field]
    if result.get("metric") == "s2_contrast":
        for field in ("level_direction", "level_metric"):
            if field in primary:
                envelope[field] = primary[field]
    return envelope


def _persist_result(state: FormalState, comparison_id: str, result: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    internal_metric = str(extra["internal_metric"])
    level_metric = extra.get("level_metric")
    action_result = internal_metric in {"policy_utility_delta", "s3_utility_did"} or (
        internal_metric == "s2_contrast" and level_metric in {"agreement", "policy_utility_delta", "normalized_regret"}
    )
    kwargs: dict[str, Any] = {}
    if action_result:
        # The result contains provenance per seed for synthetic sensitivity and
        # scalar provenance for the S3 envelope.  Bind the exact upstream
        # action output/mask hashes from the action ledger.
        if "provenance_by_seed" in result:
            candidate_by_seed: dict[int, dict[str, str]] = {}
            paired_by_seed: dict[int, dict[str, str]] = {}
            candidate_registry: dict[int, Mapping[str, str]] = {}
            paired_registry: dict[int, Mapping[str, str]] = {}
            # Read the provenance's source file/hash and resolve the matching
            # ledger row; the result reducer already validated these values.
            for ordinal, prov in result["provenance_by_seed"].items():
                source_file = prov["source_action_file_sha256"]
                matches = [ref for ref in state.action_refs.values() if ref["file_sha256"] == source_file]
                if len(matches) != 1:
                    raise RuntimeError(f"cannot resolve action result source file {source_file}")
                ref = matches[0]
                candidate_by_seed[int(ordinal)] = dict(ref["expected"])
                candidate_registry[int(ordinal)] = dict(ref["mask_hashes"])
                paired = prov.get("paired_source_action_binding")
                if isinstance(paired, Mapping):
                    paired_file = paired["source_action_file_sha256"]
                    paired_matches = [item for item in state.action_refs.values() if item["file_sha256"] == paired_file]
                    if len(paired_matches) != 1:
                        raise RuntimeError(f"cannot resolve paired action result source file {paired_file}")
                    paired_ref = paired_matches[0]
                    paired_by_seed[int(ordinal)] = dict(paired_ref["expected"])
                    paired_registry[int(ordinal)] = dict(paired_ref["mask_hashes"])
            kwargs["expected_action_mask_hash_registry_by_seed"] = candidate_registry
            kwargs["expected_action_output_hashes_by_seed"] = {
                ordinal: {field: values[field] for field in ACTION_HASH_FIELDS}
                for ordinal, values in candidate_by_seed.items()
            }
            if paired_by_seed:
                kwargs["expected_paired_action_mask_hash_registry_by_seed"] = paired_registry
                kwargs["expected_paired_action_output_hashes_by_seed"] = {
                    ordinal: {field: values[field] for field in ACTION_HASH_FIELDS}
                    for ordinal, values in paired_by_seed.items()
                }
        else:
            prov = result.get("provenance")
            if not isinstance(prov, Mapping):
                raise RuntimeError(f"action result lacks scalar provenance: {comparison_id}")
            source_file = prov["source_action_file_sha256"]
            matches = [ref for ref in state.action_refs.values() if ref["file_sha256"] == source_file]
            if len(matches) != 1:
                raise RuntimeError(f"cannot resolve scalar action source file {source_file}")
            ref = matches[0]
            kwargs["expected_action_mask_hash_registry"] = ref["mask_hashes"]
            kwargs["expected_action_output_hashes"] = {
                field: ref["expected"][field] for field in ACTION_HASH_FIELDS
            }
            paired = prov.get("paired_source_action_binding")
            if isinstance(paired, Mapping):
                paired_file = paired["source_action_file_sha256"]
                paired_matches = [item for item in state.action_refs.values() if item["file_sha256"] == paired_file]
                if len(paired_matches) != 1:
                    raise RuntimeError(f"cannot resolve paired scalar action source file {paired_file}")
                paired_ref = paired_matches[0]
                kwargs["expected_paired_action_mask_hash_registry"] = paired_ref["mask_hashes"]
                kwargs["expected_paired_action_output_hashes"] = {
                    field: paired_ref["expected"][field] for field in ACTION_HASH_FIELDS
                }
    typed = P1MBBResultArtifact.from_result_production(result, **kwargs)
    path = state.output / "results" / f"{comparison_id}.npz"
    file_sha = save_p1_mbb_result_artifact(path, typed)
    loaded = load_p1_mbb_result(
        path,
        expected_result_sha256=typed.result_sha256,
        expected_file_sha256=file_sha,
        **kwargs,
    )
    if loaded.result_sha256 != typed.result_sha256:
        raise RuntimeError(f"result reload digest changed: {comparison_id}")
    summary = {
        "comparison_id": comparison_id,
        "internal_metric": internal_metric,
        "level_metric": level_metric,
        "status": result["status"],
        "point_estimate": float(result["point_estimate"]),
        "favorable_point_estimate": float(result["favorable_point_estimate"]),
        "direction": result["direction"],
        "raw_p": float(result.get("raw_p", result["p_value"])),
        "p_value": float(result["p_value"]) if "p_value" in result else None,
        "ci": _plain(result.get("ci")),
        "block_lengths": list(result.get("block_lengths", [result.get("block_length")])),
        "per_block_length": {
            str(length): {
                "point_estimate": float(child["point_estimate"]),
                "p_value": float(child["p_value"]),
                "ci": _plain(child["ci"]),
            }
            for length, child in result.get("per_block_length", {}).items()
        },
        "result_path": _relative(path),
        "result_sha256": typed.result_sha256,
        "result_file_sha256": file_sha,
        "prereg_results_observed": bool(result["prereg_results_observed"]),
        "validation_results_observed": bool(result["validation_results_observed"]),
        "outer_results_observed": bool(result["outer_results_observed"]),
    }
    state.comparison_results[comparison_id] = summary
    state.result_objects[comparison_id] = dict(result)
    _write_json(state.output / "ledgers" / "results.json", list(state.comparison_results.values()))
    return summary


def _coverage_report(state: FormalState) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ref in state.forecast_refs.values():
        for key, value in ref["coverage"].items():
            rows.append(
                {
                    "scenario_id": ref["scenario_id"],
                    "arm": ref["arm"],
                    "seed": ref["seed"],
                    "fit_key": key,
                    "status": value.get("status"),
                    "eligible_fraction": value.get("eligible_fraction"),
                    "label_complete_fraction": value.get("label_complete_fraction"),
                    "finite_prediction_fraction": value.get("finite_prediction_fraction"),
                }
            )
    finite = [float(row["finite_prediction_fraction"]) for row in rows if row["finite_prediction_fraction"] is not None]
    label = [float(row["label_complete_fraction"]) for row in rows if row["label_complete_fraction"] is not None]
    eligible = [float(row["eligible_fraction"]) for row in rows if row["eligible_fraction"] is not None]
    action_rows: list[dict[str, Any]] = []
    for key, loaded in state.action_loaded.items():
        selected = loaded.as_mbb_input().select_metric("policy_utility_delta")
        common = np.asarray(state.action_refs[key]["common_mask"], dtype=np.bool_)
        effective = common & np.asarray(selected.effective_mask, dtype=np.bool_)
        action_rows.append({
            "scenario_id": key[0],
            "arm": key[1],
            "seed": key[2],
            "model_id": key[3],
            "cost_mode": key[4],
            "scored_action_fraction": float(np.mean(effective)),
            "scored_action_count": int(effective.sum()),
            "record_count": int(len(effective)),
        })
    return {
        "forecast_fit_rows": len(rows),
        "forecast_min_eligible_fraction": min(eligible) if eligible else None,
        "forecast_min_label_complete_fraction": min(label) if label else None,
        "forecast_min_finite_prediction_fraction": min(finite) if finite else None,
        "forecast_rows": rows,
        "action_rows": action_rows,
        "action_min_scored_fraction": min((row["scored_action_fraction"] for row in action_rows), default=None),
    }


def _gate_report(state: FormalState, registry: P1ResultRegistry, coverage: Mapping[str, Any]) -> dict[str, Any]:
    raw_p = {cid: float(summary["raw_p"]) for cid, summary in state.comparison_results.items()}
    if set(raw_p) != {str(row["comparison_id"]) for row in registry.comparisons}:
        raise RuntimeError("formal comparison calculation did not cover exactly the 16 registered rows")
    holm = holm_bonferroni_fixed_family(raw_p, registry=registry)
    rows: dict[str, Any] = {}
    s0_bounds: dict[str, Any] = {}
    for row in registry.comparisons:
        cid = str(row["comparison_id"])
        summary = state.comparison_results[cid]
        holm_row = holm.by_id[cid]
        direction = str(row["direction"])
        if cid in {"S0__ridge__utility_vs_hold__cost_on", "S0__persistence__utility_vs_hold__cost_on"}:
            # The serialized result metadata intentionally contains only
            # summaries.  Use the exact reducer vectors retained in memory to
            # evaluate the preregistered adjusted lower bound for each L.
            result_object = state.result_objects[cid]
            try:
                child_values = {
                    int(length): np.asarray(
                        result_object["per_block_length"][length]["bootstrap_values"],
                        dtype="<f8",
                    )
                    for length in P1_MBB_BLOCK_LENGTHS
                }
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"S0 result is missing an L-specific bootstrap vector: {cid}") from exc
            safety = evaluate_s0_safety_bounds(cid, child_values, holm=holm)
            s0_bounds[cid] = _plain(dict(safety))
            passed = bool(safety["passed"])
        elif cid == "S1__ridge__utility_vs_hold__cost_on":
            # This row has a preregistered per-seed guard in addition to the
            # familywise bootstrap test: every seed must have a positive,
            # non-N/A utility delta and clairvoyant same-state value must be
            # strictly above Ridge on that seed's identical scored rows.
            result_object = state.result_objects[cid]
            per_seed = result_object.get("per_seed")
            if not isinstance(per_seed, Mapping) or set(per_seed) != set(range(10)):
                raise RuntimeError("S1 utility result does not contain exactly ten seed reducers")
            seed_checks: list[dict[str, Any]] = []
            for ordinal, seed in enumerate(SEEDS):
                key = ("S1", "known_high_snr_dgp", seed, "ridge", ACTION_COST_MODE)
                loaded = state.action_loaded[key]
                mbb_input = loaded.as_mbb_input()
                selected = mbb_input.select_metric("policy_utility_delta")
                common = np.asarray(state.action_refs[key]["common_mask"], dtype=np.bool_)
                mask = common & np.asarray(selected.effective_mask, dtype=np.bool_)
                values = selected.metric_values
                candidate = np.asarray(values["candidate_utility"], dtype="<f8")
                clairvoyant = np.asarray(mbb_input.metric_values["clairvoyant_utility"], dtype="<f8")
                valid = mask & np.isfinite(candidate) & np.isfinite(clairvoyant)
                delta = float(per_seed[ordinal]["point_estimate"])
                candidate_mean = float(np.mean(candidate[valid])) if np.any(valid) else float("nan")
                clairvoyant_mean = float(np.mean(clairvoyant[valid])) if np.any(valid) else float("nan")
                agreement = np.asarray(mbb_input.metric_values["agreement"], dtype="<f8")
                agreement_valid = mask & np.isfinite(agreement)
                agreement_successes = int(np.count_nonzero(agreement[agreement_valid] == 1.0))
                agreement_total = int(np.count_nonzero(agreement_valid))
                agreement_interval = (
                    wilson_score_interval(agreement_successes, agreement_total)
                    if agreement_total > 0
                    else None
                )
                seed_checks.append(
                    {
                        "ordinal": ordinal,
                        "seed": seed,
                        "delta": delta,
                        "valid_rows": int(np.count_nonzero(valid)),
                        "candidate_mean": candidate_mean,
                        "clairvoyant_mean": clairvoyant_mean,
                        "clairvoyant_strictly_greater": bool(
                            np.isfinite(candidate_mean)
                            and np.isfinite(clairvoyant_mean)
                            and clairvoyant_mean > candidate_mean
                        ),
                        "agreement": (
                            {
                                "successes": agreement_interval.successes,
                                "total": agreement_interval.total,
                                "point": agreement_interval.point,
                                "lower": agreement_interval.lower,
                                "upper": agreement_interval.upper,
                                "confidence_level": agreement_interval.confidence_level,
                            }
                            if agreement_interval is not None
                            else None
                        ),
                    }
                )
            s1_checks = {
                "all_seed_deltas_positive": all(
                    np.isfinite(item["delta"]) and item["delta"] > 0.0 for item in seed_checks
                ),
                "all_clairvoyant_strictly_greater": all(
                    item["clairvoyant_strictly_greater"] for item in seed_checks
                ),
                "seeds": seed_checks,
            }
            direction_passed = bool(float(summary["point_estimate"]) > 0.0)
            passed = bool(
                direction_passed
                and holm_row.adjusted_p <= 0.05
                and s1_checks["all_seed_deltas_positive"]
                and s1_checks["all_clairvoyant_strictly_greater"]
            )
        else:
            favorable = (
                float(summary["point_estimate"]) > 0.0
                if direction in {"positive", "high_ge_medium", "medium_ge_low"}
                else float(summary["point_estimate"]) <= 1e-12
            )
            passed = bool(favorable and holm_row.adjusted_p <= 0.05)
        rows[cid] = {
            "raw_p": float(holm_row.raw_p),
            "holm_rank": int(holm_row.rank),
            "holm_alpha_rank": float(holm_row.alpha_rank),
            "holm_adjusted_p": float(holm_row.adjusted_p),
            "holm_rejected": bool(holm_row.rejected),
            "point_estimate": float(summary["point_estimate"]),
            "direction": direction,
            "passed": passed,
        }
        if cid == "S1__ridge__utility_vs_hold__cost_on":
            rows[cid]["s1_utility_checks"] = s1_checks
    thresholds = state.contract.coverage_thresholds
    coverage_passed = bool(
        coverage["forecast_min_eligible_fraction"] is not None
        and coverage["forecast_min_eligible_fraction"] >= min(thresholds["synthetic_eligible_origin_fraction_min"], thresholds["s3_eligible_origin_fraction_min"])
        and coverage["forecast_min_label_complete_fraction"] is not None
        and coverage["forecast_min_label_complete_fraction"] >= thresholds["label_complete_fraction_min"]
        and coverage["forecast_min_finite_prediction_fraction"] is not None
        and coverage["forecast_min_finite_prediction_fraction"] >= thresholds["finite_oof_prediction_fraction_min"]
        and coverage["action_min_scored_fraction"] is not None
        and coverage["action_min_scored_fraction"] >= thresholds["scored_action_fraction_min"]
    )
    return {
        "family_size": len(registry.comparisons),
        "alpha": 0.05,
        "comparisons": rows,
        "s0_safety": s0_bounds,
        "coverage_passed": coverage_passed,
        "all_comparison_direction_gates_passed": all(bool(row["passed"]) for row in rows.values()),
        "promotion_gate_passed": bool(coverage_passed and all(bool(row["passed"]) for row in rows.values())),
    }


def _render_report(state: FormalState, gate: Mapping[str, Any], coverage: Mapping[str, Any], *, error: str | None = None) -> str:
    lines = [
        "# P1 formal validation calculation (amended manifest)",
        "",
        f"- Run output: `{_relative(state.output)}`",
        f"- Manifest SHA-256: `{state.contract.manifest_sha256}`",
        f"- Trial registry SHA-256: `{state.contract.trial_registry_sha256}`",
        f"- Comparison registry SHA-256: `{state.contract.comparison_registry_sha256}`",
        "- Validation results observed: `true`; preregistration results observed: `false`; outer results observed: `false`.",
        "- The report-only outer operation was not executed.",
        "",
    ]
    if error is not None:
        lines += ["## Status", "", f"- **BLOCKED** during execution: `{error}`", ""]
        return "\n".join(lines)
    lines += [
        "## Coverage",
        "",
        f"- Forecast rows: {coverage['forecast_fit_rows']}; minimum eligible fraction: {coverage['forecast_min_eligible_fraction']:.6f}",
        f"- Minimum label-complete fraction: {coverage['forecast_min_label_complete_fraction']:.6f}",
        f"- Minimum finite-prediction fraction: {coverage['forecast_min_finite_prediction_fraction']:.6f}",
        f"- Minimum scored-action fraction: {coverage['action_min_scored_fraction']:.6f}",
        f"- Coverage gate: **{'PASS' if gate['coverage_passed'] else 'FAIL'}**",
        "",
        "## Registered primary comparisons",
        "",
        "| comparison | point | conservative raw p | Holm adjusted p | direction gate | result artifact |",
        "|---|---:|---:|---:|:---:|---|",
    ]
    for cid, row in gate["comparisons"].items():
        summary = state.comparison_results[cid]
        lines.append(
            f"| `{cid}` | {row['point_estimate']:.9g} | {row['raw_p']:.6g} | {row['holm_adjusted_p']:.6g} | {'PASS' if row['passed'] else 'FAIL'} | [`{summary['result_path']}`]({_relative(state.output / summary['result_path'])}) |"
        )
    lines += [
        "",
        f"## Overall promotion gate: **{'PASS' if gate['promotion_gate_passed'] else 'FAIL'}**",
        "",
        "A failed gate is an observed preregistered outcome, not a reason to tune thresholds or execute the outer report.",
    ]
    return "\n".join(lines) + "\n"


def run_formal(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = _now()
    contract = authenticate_p1_forecast_contract()
    plan = build_runner_plan()
    state = FormalState(contract, plan, output, {}, {}, {}, {}, {}, {}, {})
    run_manifest: dict[str, Any] = {
        "run_id": f"p1-formal-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "started_at": started,
        "code_revision": "94e44a6",
        "manifest_id": contract.manifest["manifest_id"],
        "manifest_sha256": contract.manifest_sha256,
        "amends_manifest_sha256": contract.manifest["amends_manifest_sha256"],
        "trial_registry_sha256": contract.trial_registry_sha256,
        "comparison_registry_sha256": contract.comparison_registry_sha256,
        "results_observed": False,
        "validation_results_observed": True,
        "outer_results_observed": False,
        "outer_report_executed": False,
        "action_replay_parity_fix_revision": "2171a34",
        "phase": "authenticated",
    }
    _write_json(output / "run_manifest.json", run_manifest)
    try:
        # Synthetic shared-base arms: each beta is constructed from the fixed
        # seed stream; no arm-specific random redraw is permitted.
        for scenario_id, arm in P1_SYNTHETIC_SCENARIO_ARMS:
            spec = contract.spec(scenario_id, arm)
            for seed in spec.seeds:
                print(f"[forecast] {scenario_id}/{arm}/{seed}", flush=True)
                dataset = build_synthetic_dataset(seed, beta=spec.beta)
                _materialize_forecast(state, scenario_id, arm, seed, dataset)
        # S3 body authentication is explicitly before fitting/scoring.
        print("[s3] loading authenticated v4 body", flush=True)
        body = load_s3_validation_body(root=ROOT)
        for arm in ("injected", "zero_injection_control"):
            spec = contract.spec("S3", arm)
            print(f"[forecast] S3/{arm}/20260830", flush=True)
            dataset = build_s3_arm_dataset(body, arm)
            fits = run_s3_validation_fits(dataset, outer_report_only=True).fits
            _materialize_forecast(state, "S3", arm, 20_260_830, dataset, fits=fits)
        run_manifest["phase"] = "forecast_complete"
        run_manifest["forecast_artifact_count"] = len(state.forecast_refs)
        _write_json(output / "run_manifest.json", run_manifest)

        for scenario_id, arm in P1_SCENARIO_ARMS:
            spec = contract.spec(scenario_id, arm)
            for seed in spec.seeds:
                if (scenario_id, arm) in ACTION_MODELS:
                    print(f"[action] {scenario_id}/{arm}/{seed}", flush=True)
                    _materialize_actions_for_arm(state, scenario_id, arm, seed)
        run_manifest["phase"] = "action_complete"
        run_manifest["action_artifact_count"] = len(state.action_refs)
        _write_json(output / "run_manifest.json", run_manifest)

        registry = contract.registry
        for row in registry.comparisons:
            cid = str(row["comparison_id"])
            print(f"[mbb] {cid}", flush=True)
            if cid.startswith("S3__"):
                result, extra = _s3_forecast_comparison(state, row) if "mse_skill" in cid else _s3_action_comparison(state, row)
            elif "utility" in cid or "regret" in cid or "agreement" in cid:
                result, extra = _action_comparison(state, row)
            else:
                result, extra = _forecast_comparison(state, row)
            _persist_result(state, cid, result, extra)
        coverage = _coverage_report(state)
        gate = _gate_report(state, registry, coverage)
        run_manifest.update(
            {
                "phase": "complete",
                "ended_at": _now(),
                "forecast_artifact_count": len(state.forecast_refs),
                "action_artifact_count": len(state.action_refs),
                "comparison_result_count": len(state.comparison_results),
                "promotion_gate_passed": gate["promotion_gate_passed"],
                "results_observed": False,
            }
        )
        _write_json(output / "coverage.json", coverage)
        _write_json(output / "gates.json", gate)
        report = _render_report(state, gate, coverage)
        (output / "report.md").write_text(report, encoding="utf-8")
        _write_json(output / "run_manifest.json", run_manifest)
        print(f"[complete] promotion_gate_passed={gate['promotion_gate_passed']}", flush=True)
        return {"run_manifest": run_manifest, "gate": gate, "coverage": coverage}
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        run_manifest.update({"phase": "blocked", "ended_at": _now(), "error": message, "traceback": traceback.format_exc()})
        _write_json(output / "run_manifest.json", run_manifest)
        (output / "report.md").write_text(_render_report(state, {}, {}, error=message), encoding="utf-8")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        run_formal(args.output.resolve())
    except Exception:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
