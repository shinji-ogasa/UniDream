"""Fail-closed preregistration boundary for the P2-D0 feature ablation.

This module is deliberately a protocol validator, not an experiment runner.
It fixes the canonical v4 full17 versus first-13-column OHLCV13 comparison,
the timestamp folds, candidate family, common-row masks, and inferential
registry before any fit, score, or outer operation is allowed.  The existing
2018--2024 body is historical; the 2023 interval is report-only and is never
described as an untouched holdout.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .p1_recovery_prereg import (
    DEFAULT_MANIFEST_PATH as P1_DEFAULT_MANIFEST_PATH,
    P1_V4_RUNTIME_VALIDATION_ENTRYPOINT,
    REGISTERED_MANIFEST_SHA256 as P1_REGISTERED_MANIFEST_SHA256,
)


class P2D0PreregistrationError(ValueError):
    """Raised when the immutable P2-D0 protocol is missing or altered."""


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiments"
    / "p2_d0_prereg_manifest.json"
)

# Replaced after the manifest is finalized.  This independent pin means an
# edited P2-D0 manifest cannot make its own self-reported digest authoritative.
REGISTERED_MANIFEST_SHA256 = "a0ac7357abadb4b459f0687b12fb5926089fe9e1bd0987990ede82750b952cd2"
REGISTERED_BASE_REVISION = "01f79db3b624187a857eb0a4105d466281259490"
P2_D0_RUNTIME_VALIDATION_ENTRYPOINT = (
    "unidream.experiments.p2_d0_prereg.load_authenticated_v4_runtime"
)
V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT = (
    "unidream.experiments.runtime.validate_v4_runtime_inputs"
)

FULL17_COLUMNS = (
    "open_ret",
    "high_ret",
    "low_ret",
    "close_ret",
    "vol_ret",
    "RSI_14",
    "macd",
    "macd_signal",
    "atr_norm_ret",
    "atr",
    "rv_4",
    "rv_16",
    "rv_96",
    "funding_rate",
    "basis",
    "basis_mom",
    "basis_abs",
)
OHLCV13_COLUMNS = FULL17_COLUMNS[:13]
FORECAST_HORIZONS = (1, 4, 8, 16)
SEEDS = (
    20260830,
    20260831,
    20260832,
    20260833,
    20260834,
    20260835,
    20260836,
    20260837,
    20260838,
    20260839,
)
REQUIRED_TOP_LEVEL_FIELDS = (
    "manifest_id",
    "schema_version",
    "status",
    "registered_date",
    "base_revision",
    "amends_manifest_sha256",
    "amendment_reason",
    "amendment_history",
    "results_observed",
    "manifest_sha256",
    "critical_field_paths",
    "common",
    "provenance",
)


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash canonical manifest content after removing its self-digest field."""

    if not isinstance(manifest, Mapping):
        raise P2D0PreregistrationError("manifest must be an object")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise P2D0PreregistrationError("manifest is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def exact_file_sha256(path: str | Path) -> str:
    """Hash exact bytes for a pinned registry file."""

    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise P2D0PreregistrationError(f"could not read pinned file: {path}") from exc


def _artifact_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts:
        raise P2D0PreregistrationError("pinned artifact path must stay inside the repository")
    return root / path


def _path_value(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise P2D0PreregistrationError(f"critical field is missing: {path}")
        current = current[component]
    return current


def _require_mapping(payload: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    value = _path_value(payload, path)
    if not isinstance(value, Mapping):
        raise P2D0PreregistrationError(f"{path} must be an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise P2D0PreregistrationError(f"{label} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise P2D0PreregistrationError(f"{label} must be hexadecimal SHA-256") from exc
    return value


def _freeze(value: Any) -> Any:
    """Deep-freeze JSON values returned by the production loader."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_registry(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate registry hashes, schemas, row counts, and fixed ordering."""

    common = _require_mapping(manifest, "common")
    trial_ref = _require_mapping(common, "trial_registry")
    comparison_ref = _require_mapping(common, "primary_comparison_registry")
    registries: list[tuple[Mapping[str, Any], str]] = [
        (trial_ref, "trial registry"),
        (comparison_ref, "primary comparison registry"),
    ]
    parsed: list[list[dict[str, Any]]] = []
    for reference, label in registries:
        path = _artifact_path(root, str(reference.get("path", "")))
        digest = _require_sha256(reference.get("sha256"), f"{label} hash")
        if exact_file_sha256(path) != digest:
            raise P2D0PreregistrationError(f"{label} hash mismatch")
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise P2D0PreregistrationError(f"could not parse {label}") from exc
        if any(not isinstance(row, dict) for row in rows):
            raise P2D0PreregistrationError(f"{label} rows must be JSON objects")
        if len(rows) != reference.get("record_count"):
            raise P2D0PreregistrationError(f"{label} record_count mismatch")
        required_fields = reference.get("required_fields")
        if (
            not isinstance(required_fields, Sequence)
            or isinstance(required_fields, (str, bytes, bytearray))
            or any(not isinstance(field, str) for field in required_fields)
        ):
            raise P2D0PreregistrationError(f"{label} required_fields are invalid")
        if any(not set(required_fields).issubset(row) for row in rows):
            raise P2D0PreregistrationError(f"{label} has a row missing required fields")
        parsed.append(rows)
    trials, comparisons = parsed
    expected_trial_ids = [
        f"{arm}__{model}"
        for arm in ("full17", "ohlcv13")
        for model in (
            "zero_return",
            "persistence_last_observed",
            "ridge",
            "logistic",
            "hist_gradient_boosting",
        )
    ]
    expected_split_ids = [
        "train_2018_2021",
        "inner_calibration_2021",
        "outer_validation_2022",
        "historical_report_2023",
    ]
    if [row.get("trial_id") for row in trials] != expected_trial_ids:
        raise P2D0PreregistrationError("trial registry order or IDs are altered")
    expected_tasks = {
        "zero_return": ["continuous", "binary"],
        "persistence_last_observed": ["continuous", "binary"],
        "ridge": ["continuous"],
        "logistic": ["binary"],
        "hist_gradient_boosting": ["continuous", "binary"],
    }
    for row in trials:
        if row.get("feature_arm") not in {"full17", "ohlcv13"}:
            raise P2D0PreregistrationError("trial registry feature arm is unknown")
        if row.get("model_id") not in {
            "zero_return",
            "persistence_last_observed",
            "ridge",
            "logistic",
            "hist_gradient_boosting",
        }:
            raise P2D0PreregistrationError("trial registry model is unknown")
        if row.get("tasks") != expected_tasks[row["model_id"]]:
            raise P2D0PreregistrationError("trial registry tasks are not fixed")
        if row.get("horizons") != list(FORECAST_HORIZONS) or row.get("seed_values") != list(SEEDS):
            raise P2D0PreregistrationError("trial registry schedule is altered")
        if row.get("split_ids") != expected_split_ids:
            raise P2D0PreregistrationError("trial registry split schedule is altered")
        if row.get("results_observed") is not False or row.get("status") != "preregistered":
            raise P2D0PreregistrationError("trial registry must remain result-free")

    expected_comparison_ids = [
        *[
            f"full17_vs_ohlcv13__ridge__mse__h{horizon}"
            for horizon in FORECAST_HORIZONS
        ],
        *[
            f"full17_vs_ohlcv13__hist_gradient_boosting__mse__h{horizon}"
            for horizon in FORECAST_HORIZONS
        ],
        *[
            f"full17_vs_ohlcv13__logistic__log_loss__h{horizon}"
            for horizon in FORECAST_HORIZONS
        ],
        "full17_vs_ohlcv13__ridge__utility__h4",
        "full17_vs_ohlcv13__hist_gradient_boosting__utility__h4",
    ]
    if [row.get("comparison_id") for row in comparisons] != expected_comparison_ids:
        raise P2D0PreregistrationError("comparison registry order or IDs are altered")
    if len(comparisons) != 14 or comparison_ref.get("family_size") != 14:
        raise P2D0PreregistrationError("primary comparison family size must remain 14")
    comparison_trial_prefixes = {
        "full17_vs_ohlcv13__ridge__mse__h1": ("ridge", 1, "continuous", "off"),
        "full17_vs_ohlcv13__ridge__mse__h4": ("ridge", 4, "continuous", "off"),
        "full17_vs_ohlcv13__ridge__mse__h8": ("ridge", 8, "continuous", "off"),
        "full17_vs_ohlcv13__ridge__mse__h16": ("ridge", 16, "continuous", "off"),
        "full17_vs_ohlcv13__hist_gradient_boosting__mse__h1": ("hist_gradient_boosting", 1, "continuous", "off"),
        "full17_vs_ohlcv13__hist_gradient_boosting__mse__h4": ("hist_gradient_boosting", 4, "continuous", "off"),
        "full17_vs_ohlcv13__hist_gradient_boosting__mse__h8": ("hist_gradient_boosting", 8, "continuous", "off"),
        "full17_vs_ohlcv13__hist_gradient_boosting__mse__h16": ("hist_gradient_boosting", 16, "continuous", "off"),
        "full17_vs_ohlcv13__logistic__log_loss__h1": ("logistic", 1, "binary", "off"),
        "full17_vs_ohlcv13__logistic__log_loss__h4": ("logistic", 4, "binary", "off"),
        "full17_vs_ohlcv13__logistic__log_loss__h8": ("logistic", 8, "binary", "off"),
        "full17_vs_ohlcv13__logistic__log_loss__h16": ("logistic", 16, "binary", "off"),
        "full17_vs_ohlcv13__ridge__utility__h4": ("ridge", 4, "continuous", "on"),
        "full17_vs_ohlcv13__hist_gradient_boosting__utility__h4": ("hist_gradient_boosting", 4, "continuous", "on"),
    }
    for row in comparisons:
        model_id, expected_horizon, expected_task, expected_cost_mode = comparison_trial_prefixes[row.get("comparison_id", "")]
        expected_candidate = f"full17__{model_id}__h{expected_horizon}"
        expected_baseline = f"ohlcv13__{model_id}__h{expected_horizon}"
        if expected_cost_mode == "on":
            expected_candidate += "__cost_on"
            expected_baseline += "__cost_on"
        if (
            row.get("candidate_id") != expected_candidate
            or row.get("baseline_id") != expected_baseline
            or row.get("horizon") != expected_horizon
            or row.get("task") != expected_task
            or row.get("cost_mode") != expected_cost_mode
        ):
            raise P2D0PreregistrationError("comparison trial binding is altered")
        if row.get("primary") is not True:
            raise P2D0PreregistrationError("every D0 comparison row must be primary")
        if row.get("support_id") != "outer_validation_2022":
            raise P2D0PreregistrationError("D0 comparison support is not the fixed outer validation")
        if row.get("support_start") != "2022-01-01T00:00:00Z" or row.get("support_end") != "2023-01-01T00:00:00Z":
            raise P2D0PreregistrationError("D0 comparison timestamp support is altered")
        if row.get("horizon") not in FORECAST_HORIZONS:
            raise P2D0PreregistrationError("D0 comparison horizon is not fixed")
        if row.get("cost_mode") == "on" and row.get("horizon") != 4:
            raise P2D0PreregistrationError("only h4 may carry action utility")
        if row.get("metric") in {"mse", "log_loss"}:
            if row.get("cost_mode") != "off" or row.get("direction") != "negative":
                raise P2D0PreregistrationError("forecast comparison semantics are altered")
        elif row.get("metric") == "paired_net_utility_delta":
            if row.get("horizon") != 4 or row.get("task") != "continuous" or row.get("cost_mode") != "on" or row.get("direction") != "positive":
                raise P2D0PreregistrationError("action utility comparison semantics are altered")
        else:
            raise P2D0PreregistrationError("unknown D0 comparison metric")
        if row.get("support_role") != "primary_inferential_gate":
            raise P2D0PreregistrationError("D0 comparison support role is altered")
    return trials, comparisons


def validate_fixed_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_digest: str | None = None,
    root: str | Path | None = None,
) -> None:
    """Validate every fixed D0 field without reading data or running models."""

    if not isinstance(manifest, Mapping):
        raise P2D0PreregistrationError("manifest must be an object")
    missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in manifest]
    if missing:
        raise P2D0PreregistrationError("manifest is missing: " + ", ".join(missing))
    if manifest.get("manifest_id") != "p2-d0-full17-vs-ohlcv13-preregister-20260831-v1":
        raise P2D0PreregistrationError("manifest_id is not fixed")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "preregistered":
        raise P2D0PreregistrationError("manifest schema/status is altered")
    if manifest.get("registered_date") != "2026-08-31":
        raise P2D0PreregistrationError("registered_date is altered")
    if manifest.get("base_revision") != REGISTERED_BASE_REVISION:
        raise P2D0PreregistrationError("base_revision is not the requested main revision")
    if manifest.get("amends_manifest_sha256") != P1_REGISTERED_MANIFEST_SHA256:
        raise P2D0PreregistrationError("amended P1 manifest digest is not pinned")
    if manifest.get("amendment_reason") != "P2-D0 corrected full17 versus OHLCV13 availability-aware same-common-row preregistration boundary":
        raise P2D0PreregistrationError("amendment reason is altered")
    if manifest.get("amendment_history") != [
        {
            "manifest_sha256": P1_REGISTERED_MANIFEST_SHA256,
            "reason": "P1 recovery preregistration predecessor",
            "results_observed": False,
        }
    ]:
        raise P2D0PreregistrationError("amendment history is altered")
    if manifest.get("results_observed") is not False:
        raise P2D0PreregistrationError("results_observed must remain false")
    reported_digest = _require_sha256(manifest.get("manifest_sha256"), "manifest_sha256")
    digest = canonical_manifest_sha256(manifest)
    pinned = REGISTERED_MANIFEST_SHA256 if expected_digest is None else expected_digest
    _require_sha256(pinned, "expected manifest digest")
    if reported_digest != digest or digest != pinned:
        raise P2D0PreregistrationError("manifest canonical digest mismatch")

    critical = manifest.get("critical_field_paths")
    if not isinstance(critical, list) or not critical or any(not isinstance(path, str) or not path for path in critical):
        raise P2D0PreregistrationError("critical_field_paths must be non-empty strings")
    for path in critical:
        _path_value(manifest, path)

    common = _require_mapping(manifest, "common")
    if common.get("symbol") != "BTCUSDT" or common.get("data_frequency") != "15m" or common.get("timezone") != "UTC":
        raise P2D0PreregistrationError("symbol/frequency/timezone are altered")
    if common.get("forecast_horizons") != list(FORECAST_HORIZONS) or common.get("forbidden_horizons") != [64] or common.get("utility_head") is not False:
        raise P2D0PreregistrationError("D0 horizon/head boundary is altered")
    if common.get("action_horizon") != 4 or common.get("forecast_only_horizons") != [1, 8, 16]:
        raise P2D0PreregistrationError("D0 action/forecast-only horizon boundary is altered")
    arms = _require_mapping(common, "feature_arms")
    full17 = _require_mapping(arms, "full17")
    ohlcv13 = _require_mapping(arms, "ohlcv13")
    if full17.get("arm_id") != "full17" or full17.get("input_shape") != [17] or full17.get("source") != "authenticated v4 feature body" or tuple(full17.get("columns", ())) != FULL17_COLUMNS:
        raise P2D0PreregistrationError("full17 columns are altered")
    if ohlcv13.get("arm_id") != "ohlcv13" or ohlcv13.get("input_shape") != [13] or ohlcv13.get("source") != "canonical v4 feature body column projection only" or tuple(ohlcv13.get("columns", ())) != OHLCV13_COLUMNS:
        raise P2D0PreregistrationError("OHLCV13 must be exactly the first 13 full17 columns")
    shared_rule = str(arms.get("shared_common_row_rule", ""))
    if (
        "exactly the intersection" not in shared_rule
        or "identical timestamp rows" not in shared_rule
        or "neither arm may recover" not in shared_rule
        or "compact the grid" not in shared_rule
    ):
        raise P2D0PreregistrationError("full17/OHLCV13 common-row intersection is not fixed")
    if "common_mask[t,h]" not in str(arms.get("row_intersection_formula", "")):
        raise P2D0PreregistrationError("common-row mask formula is missing")

    target = _require_mapping(common, "target_contract")
    if target.get("target_formula") != "y[t,h] = sum(return[t+1:t+h+1])" or target.get("target_end_formula") != "target_end[t,h] = t + h + 1 (exclusive)" or target.get("target_rows") != "t+1 through t+h" or target.get("following_edge") != "t+h->t+h+1 is not required":
        raise P2D0PreregistrationError("target contract is altered")
    context = _require_mapping(common, "context_contract")
    if context.get("context_bars") != 64 or context.get("window") != "current-inclusive [t-63,t]":
        raise P2D0PreregistrationError("context window contract is altered")
    split = _require_mapping(common, "split_contract")
    if split.get("purge_bars") != 16 or split.get("body_end_exclusive") != "2024-01-01T00:00:00Z":
        raise P2D0PreregistrationError("split/purge contract is altered")
    folds = _require_mapping(split, "folds")
    expected_folds = {
        "train": ("train_2018_2021", "2018-01-19T17:00:00Z", "2021-01-01T00:00:00Z", "fit_prefix"),
        "inner_calibration": ("inner_calibration_2021", "2021-01-01T00:00:00Z", "2022-01-01T00:00:00Z", "nested_calibration_only"),
        "outer_validation": ("outer_validation_2022", "2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z", "primary_inferential_gate"),
        "historical_report_only": ("historical_report_2023", "2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "report_only"),
    }
    for key, (split_id, start, end, role) in expected_folds.items():
        fold = _require_mapping(folds, key)
        if (fold.get("split_id"), fold.get("start"), fold.get("end"), fold.get("role")) != (split_id, start, end, role):
            raise P2D0PreregistrationError(f"fold {key} is altered")
    if (
        "right-exclusive" not in str(split.get("range_semantics", ""))
        or "inner calibration" not in str(split.get("nested_rule", ""))
        or "before outer validation" not in str(split.get("nested_rule", ""))
    ):
        raise P2D0PreregistrationError("right-exclusive/nested split policy is missing")

    if common.get("minimum_history_rows") != 16384 or "16384" not in str(common.get("minimum_history_rule", "")):
        raise P2D0PreregistrationError("minimum history rule is altered")
    learned = _require_mapping(common, "learned_fit_contract")
    for required in ("train_mask", "feature_scaler", "target_scaling", "baseline_scaling", "one_class_rule"):
        if required not in learned:
            raise P2D0PreregistrationError("P1 causal fit contract is incomplete")
    models = _require_mapping(common, "models")
    expected_model_ids = {"zero_return", "persistence_last_observed", "ridge", "logistic", "hist_gradient_boosting"}
    if set(models) != expected_model_ids:
        raise P2D0PreregistrationError("D0 model family is altered")
    expected_models = {
        "zero_return": {
            "kind": "fixed_baseline",
            "tasks": ["continuous", "binary"],
            "continuous_prediction": "0.0 cumulative return",
            "binary_prediction": "class-1 probability 0.5",
            "action_role": "h4 fixed comparator only",
        },
        "persistence_last_observed": {
            "kind": "fixed_baseline",
            "tasks": ["continuous", "binary"],
            "continuous_prediction": "h * return[t]",
            "binary_prediction": "1-eps when return[t] > 0, otherwise eps, eps=1e-6",
            "action_role": "h4 fixed comparator only",
        },
        "ridge": {
            "kind": "sklearn.linear_model.Ridge",
            "tasks": ["continuous"],
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "lsqr",
            "tol": 1e-12,
            "max_iter": 10000,
            "random_state": None,
            "action_role": "h4 continuous action candidate; h1/h8/h16 forecast-only",
        },
        "logistic": {
            "kind": "sklearn.linear_model.LogisticRegression",
            "tasks": ["binary"],
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "tol": 1e-10,
            "max_iter": 1000,
            "class_weight": None,
            "random_state": 0,
            "action_role": "forecast-only proper-score diagnostic; no utility/action output",
        },
        "hist_gradient_boosting": {
            "kind": "sklearn.ensemble.HistGradientBoostingRegressor_and_Classifier",
            "tasks": ["continuous", "binary"],
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "max_depth": 4,
            "min_samples_leaf": 64,
            "l2_regularization": 0.0,
            "early_stopping": False,
            "categorical_features": None,
            "random_state": "seed",
            "deep_model": False,
            "action_role": "h4 continuous action candidate; h1/h8/h16 forecast-only",
        },
    }
    for model_id in expected_model_ids:
        model = _require_mapping(models, model_id)
        if model != expected_models[model_id]:
            raise P2D0PreregistrationError(f"fixed model contract altered: {model_id}")
        if model.get("deep_model") is True:
            raise P2D0PreregistrationError("deep model is forbidden")
    hgb = models["hist_gradient_boosting"]
    if hgb.get("kind") != "sklearn.ensemble.HistGradientBoostingRegressor_and_Classifier" or hgb.get("max_iter") != 200 or hgb.get("max_leaf_nodes") != 15 or hgb.get("max_depth") != 4 or hgb.get("early_stopping") is not False:
        raise P2D0PreregistrationError("HistGradientBoosting fixed budget is altered")
    candidate_families = _require_mapping(common, "candidate_families")
    expected_candidate_families = {
        "fixed_baselines": ["zero_return", "persistence_last_observed"],
        "linear": ["ridge"],
        "tree": ["hist_gradient_boosting"],
        "binary": ["logistic", "hist_gradient_boosting"],
        "action_capable": ["zero_return", "persistence_last_observed", "ridge", "hist_gradient_boosting"],
        "forbidden": ["MLP", "deep_neural_network", "transformer", "deep_boosting", "unregistered_model"],
        "family_selection": "candidate family, feature arm, task, horizon, hyperparameters, and seed list are fixed before any output; no post-output family expansion or deletion",
    }
    if candidate_families != expected_candidate_families:
        raise P2D0PreregistrationError("forbidden candidate family is altered")
    if common.get("seeds") != list(SEEDS):
        raise P2D0PreregistrationError("seed schedule is altered")

    calibration = _require_mapping(common, "calibration_contract")
    if calibration.get("inner_calibration_split") != "inner_calibration_2021" or calibration.get("outer_validation_never_used_for_calibration") is not True or calibration.get("historical_report_never_used_for_calibration") is not True:
        raise P2D0PreregistrationError("nested calibration boundary is altered")
    action = _require_mapping(common, "action_contract")
    if action.get("action_horizon") != 4 or action.get("forecast_only_horizons") != [1, 8, 16] or action.get("decision_to_fill") != "decision t -> fill t+1" or action.get("generic_mbb_forbidden") is not True:
        raise P2D0PreregistrationError("action horizon/fill/MBB boundary is altered")
    metrics = _require_mapping(common, "metrics")
    if metrics.get("paired_utility") != ["mean_net_log_utility", "paired_net_utility_delta_vs_hold"] or "proper_score_rule" not in metrics or "identical common" not in str(metrics.get("proper_score_rule")):
        raise P2D0PreregistrationError("proper-score/paired utility contract is altered")
    coverage = _require_mapping(common, "coverage_contract")
    thresholds = _require_mapping(coverage, "thresholds")
    if thresholds != {
        "common_row_fraction_min": 0.9,
        "context_complete_fraction_min": 0.9,
        "label_complete_fraction_min": 0.9,
        "finite_prediction_fraction_min": 0.95,
        "scored_action_fraction_min": 0.8,
    }:
        raise P2D0PreregistrationError("coverage thresholds are altered")
    if coverage.get("any_na_rule") != "a required N/A cell blocks that comparison; never impute, drop, compact, or convert N/A to zero" or coverage.get("all_na_rule") != "if every primary comparison cell is N/A or undefined, status is blocked_no_inferential_result and no claim or promotion is permitted":
        raise P2D0PreregistrationError("N/A coverage policy is altered")
    statistical = _require_mapping(common, "statistical_contract")
    if statistical.get("family_id") != "p2_d0_full17_vs_ohlcv13_primary" or statistical.get("family_size") != 14 or statistical.get("multiplicity_method") != "Holm-Bonferroni":
        raise P2D0PreregistrationError("Holm family is altered")
    bootstrap = _require_mapping(statistical, "bootstrap")
    if bootstrap.get("replicates") != 2000 or bootstrap.get("block_lengths") != [8, 16, 32] or bootstrap.get("pairing", "").find("identical sampled indices") == -1:
        raise P2D0PreregistrationError("paired bootstrap contract is altered")

    historical = _require_mapping(common, "historical_boundary")
    if historical.get("historical_report_only") is not True or historical.get("untouched_claim_forbidden") is not True or historical.get("future_holdout_required") is not True:
        raise P2D0PreregistrationError("historical report-only boundary is missing")
    if "stop" not in str(historical.get("future_stop_condition", "")).lower() or "newly acquired" not in str(historical.get("future_stop_condition", "")):
        raise P2D0PreregistrationError("future holdout stop condition is missing")

    runtime = _require_mapping(common, "runtime_contract")
    expected_runtime = {
        "loader": "unidream.data.cache_v4.load_cache_v4",
        "p2_runtime_entrypoint": P2_D0_RUNTIME_VALIDATION_ENTRYPOINT,
        "runtime_validation_entrypoint": P1_V4_RUNTIME_VALIDATION_ENTRYPOINT,
        "runtime_body_validator_entrypoint": V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT,
        "require_explicit_paths": True,
        "cache_dir_cache_tag_fallback": "forbidden",
        "metadata_authority": "repo_frozen_metadata",
        "cache_tag": "BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official",
        "feature_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_features.parquet",
        "returns_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_returns.parquet",
        "availability_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_availability.parquet",
        "metadata_path": "docs/data_quality_v4_rebuild_2018_2024_metadata.json",
        "cache_local_metadata_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_metadata.json",
        "frozen_metadata_sha256": "2c9db28deebe7e6b08f4ffedf65c3cdb51a78cfd7ee7d6580f76a62cc424bdcb",
        "frozen_source_provenance_digest": "aa320222dca0a46b2a0730f17bb1665f31a70074aa3bafcc6bff58ca21618fad",
        "frozen_schema_digest": "1c1c41a9aca3e8af22b357a8483ea6419745ee4b24c10c09c47289df3744c616",
        "frozen_content_digests": {
            "features": "8a7aad5809c7a21e614da7d836629309cda9c2de74553bf1fbc6934f7b07f5e2",
            "returns": "c33a00cac4cf169f01e3ba5823a3f6d9bae17da5add5f8d5a3538d4142a0fabb",
            "availability": "630de125ae9bc04cd0376404c7cff07f8e7d06c3bec2eece1b546e05959e292f",
        },
        "frozen_rows": {"features": 173111, "availability": 210336},
        "runtime_status_required": "passed",
        "results_observed_required": False,
    }
    for field, expected in expected_runtime.items():
        if runtime.get(field) != expected:
            raise P2D0PreregistrationError(f"runtime contract field altered: {field}")
    if "all required file/content/schema hashes" not in str(runtime.get("manifest_authentication", "")) and "canonical registered digest" not in str(runtime.get("manifest_authentication", "")):
        raise P2D0PreregistrationError("runtime authentication/hash binding is incomplete")
    if "complete explicit four-file" not in str(runtime.get("path_override_policy", "")):
        raise P2D0PreregistrationError("runtime explicit path policy is incomplete")

    runner = _require_mapping(common, "runner_contract")
    if runner.get("d1_excluded") is not True or runner.get("results_observed") is not False or runner.get("outer_validation_selection_allowed") is not False or runner.get("historical_report_selection_allowed") is not False or runner.get("post_output_tuning_allowed") is not False:
        raise P2D0PreregistrationError("runner fail-closed/report-only policy is altered")
    if runner.get("outer_operation_policy") != {
        "mode": "report_only",
        "max_runs": 1,
        "rerun_policy": "one report-only execution after all prerequisites are independently validated; any failed, incomplete, or blocked run remains N/A and is not rerun or replaced",
    }:
        raise P2D0PreregistrationError("outer report-only execution policy is altered")
    if runner.get("action_primitive_execution_status") != "blocked_not_implemented":
        raise P2D0PreregistrationError("action primitive blocker must remain explicit")

    provenance = _require_mapping(manifest, "provenance.v4_parent")
    runtime_parent = _require_mapping(common, "runtime_contract")
    runtime_fields = {
        "cache_tag": "cache_tag",
        "schema_digest": "frozen_schema_digest",
        "source_provenance_digest": "frozen_source_provenance_digest",
        "content_digests": "frozen_content_digests",
    }
    for field, runtime_field in runtime_fields.items():
        if provenance.get(field) != runtime_parent.get(runtime_field):
            raise P2D0PreregistrationError(f"v4 provenance/runtime {field} mismatch")
    if provenance.get("metadata_sha256") != runtime_parent.get("frozen_metadata_sha256") or provenance.get("feature_rows") != 173111 or provenance.get("sidecar_rows") != 210336:
        raise P2D0PreregistrationError("v4 parent metadata/hash/row binding is altered")
    if provenance.get("required_availability_columns") != ["spot_bar_observed", "funding_rate_available", "mark_close_available"]:
        raise P2D0PreregistrationError("v4 required availability columns are altered")
    _validate_registry(manifest, root=Path(root) if root is not None else DEFAULT_MANIFEST_PATH.parents[2])


def validate_pinned_artifacts(
    manifest: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> None:
    """Validate only registry paths/hashes for a previously checked manifest."""

    root_path = Path(root) if root is not None else DEFAULT_MANIFEST_PATH.parents[2]
    _validate_registry(manifest, root=root_path)


def load_fixed_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> Mapping[str, Any]:
    """Load, validate, and deeply freeze the fixed P2-D0 manifest."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P2D0PreregistrationError("could not read P2-D0 manifest") from exc
    validate_fixed_manifest(payload, root=Path(path).resolve().parents[2])
    return _freeze(payload)


def load_authenticated_v4_runtime(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Authenticate P2-D0 first, then invoke only the P1 v4 wrapper.

    This helper is intentionally a runtime-boundary fixture.  It performs no
    fit, score, accuracy calculation, action replay, or outer operation.
    """

    p2_manifest = load_fixed_manifest(manifest_path)
    from .runtime import validate_p1_v4_runtime_inputs

    result = validate_p1_v4_runtime_inputs(**kwargs)
    if not isinstance(result, Mapping):
        raise P2D0PreregistrationError("authenticated v4 wrapper returned no mapping")
    if result.get("p1_runtime_validation_entrypoint") != P1_V4_RUNTIME_VALIDATION_ENTRYPOINT or result.get("p1_manifest_sha256") != P1_REGISTERED_MANIFEST_SHA256 or result.get("p1_results_observed") is not False:
        raise P2D0PreregistrationError("authenticated v4 wrapper identity is missing")
    return MappingProxyType(
        {
            "p2_manifest_id": p2_manifest["manifest_id"],
            "p2_manifest_sha256": p2_manifest["manifest_sha256"],
            "p2_base_revision": p2_manifest["base_revision"],
            "p2_results_observed": p2_manifest["results_observed"],
            "v4_runtime_validation_status": result.get("v4_runtime_validation_status"),
            "v4_runtime_provenance_disposition": _freeze(result.get("v4_runtime_provenance_disposition")),
            "v4_runtime_frozen_metadata_sha256": result.get("v4_runtime_frozen_metadata_sha256"),
            "v4_frozen_metadata_sha256": result.get("v4_frozen_metadata_sha256"),
            "v4_frozen_source_provenance_digest": result.get("v4_frozen_source_provenance_digest"),
            "v4_cache_local_metadata_sha256": result.get("v4_cache_local_metadata_sha256"),
            "v4_cache_local_source_provenance_digest": result.get("v4_cache_local_source_provenance_digest"),
            "v4_cache_local_schema_digest": result.get("v4_cache_local_schema_digest"),
            "v4_cache_local_content_digests": _freeze(result.get("v4_cache_local_content_digests")),
            "v4_cache_local_row_counts": _freeze(result.get("v4_cache_local_row_counts")),
            "v4_feature_path": result.get("v4_feature_path"),
            "v4_returns_path": result.get("v4_returns_path"),
            "v4_availability_path": result.get("v4_availability_path"),
            "v4_frozen_metadata_path": result.get("v4_frozen_metadata_path"),
            "v4_cache_local_metadata_path": result.get("v4_cache_local_metadata_path"),
        }
    )


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "FORECAST_HORIZONS",
    "FULL17_COLUMNS",
    "OHLCV13_COLUMNS",
    "P2D0PreregistrationError",
    "P2_D0_RUNTIME_VALIDATION_ENTRYPOINT",
    "REGISTERED_BASE_REVISION",
    "REGISTERED_MANIFEST_SHA256",
    "SEEDS",
    "canonical_manifest_sha256",
    "exact_file_sha256",
    "load_authenticated_v4_runtime",
    "load_fixed_manifest",
    "validate_fixed_manifest",
    "validate_pinned_artifacts",
]
