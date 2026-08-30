"""Fail-closed loader for the pre-registered P1 recovery protocol.

This module does not run an experiment and does not read result artifacts.  It
only validates the committed, machine-readable protocol before a future
runner is allowed to construct data or model jobs.  The registered digest is
pinned in code so a runner cannot silently replace a critical field together
with the manifest's self-reported digest.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any


class P1PreregistrationError(ValueError):
    """Raised when a P1 preregistration is missing or has been altered."""


DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiments"
    / "p1_recovery_prereg_manifest.json"
)

# Filled from the committed manifest after its canonical JSON digest is
# calculated.  Keeping this independently pinned is what makes an edited
# ``manifest_sha256`` field fail closed as well.
REGISTERED_MANIFEST_SHA256 = "9ba18e3e1226cbcbe57e6dfc40050036b1e70b92e58a75e73f8e6ad6c3bc747d"
REGISTERED_BASE_REVISION = "881e5e08e9b413b51b0a2faf5c49592ce13329d1"

REQUIRED_TOP_LEVEL_FIELDS = (
    "manifest_id",
    "schema_version",
    "status",
    "registered_date",
    "base_revision",
    "manifest_sha256",
    "critical_field_paths",
    "common",
    "synthetic_contract",
    "scenarios",
    "provenance",
)


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash the manifest after removing its self-referential digest field."""
    if not isinstance(manifest, Mapping):
        raise P1PreregistrationError("manifest must be an object")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON payload using the contract's canonical UTF-8 encoding."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exact_file_sha256(path: str | Path) -> str:
    """Hash exact artifact bytes, used for immutable JSONL registries."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _artifact_path(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise P1PreregistrationError("artifact path must stay inside the repository")
    return root / path


def validate_pinned_artifacts(
    manifest: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> None:
    """Verify every pinned contract/registry digest and derived family size."""
    if root is None:
        root_path = DEFAULT_MANIFEST_PATH.parents[2]
    else:
        root_path = Path(root)
    common = _require_mapping(manifest, "common")

    def check_json(ref: Mapping[str, Any], label: str) -> dict[str, Any]:
        path = _artifact_path(root_path, str(ref.get("path", "")))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise P1PreregistrationError(f"could not read pinned {label}: {path}") from exc
        expected = ref.get("sha256")
        actual = canonical_json_sha256(payload)
        if expected != actual:
            raise P1PreregistrationError(f"pinned {label} hash mismatch")
        return payload

    action_ref = _require_mapping(common, "action_execution_contract_reference")
    check_json(action_ref, "action contract")
    mode_refs = _require_mapping(common, "cost_mode_contracts")
    mode_payloads: dict[str, dict[str, Any]] = {}
    for mode in ("off", "on"):
        mode_ref = _require_mapping(mode_refs, mode)
        mode_payloads[mode] = check_json(mode_ref, f"cost-mode {mode} contract")
        if mode_ref.get("sha256") != _path_value(common, f"cost_modes.{mode}.contract_hash"):
            raise P1PreregistrationError(f"cost-mode {mode} hash echo mismatch")
    if mode_refs["on"].get("sha256") != action_ref.get("sha256"):
        raise P1PreregistrationError("cost-on action contract hash echo mismatch")
    cost_fields = {"spread_bps", "slippage_bps", "fee_rate", "transition_cost_rate"}
    for key in mode_payloads["on"]:
        if key not in cost_fields and mode_payloads["off"].get(key) != mode_payloads["on"].get(key):
            raise P1PreregistrationError("cost-off contract changed non-cost action semantics")
    if any(mode_payloads["off"].get(key) != 0.0 for key in cost_fields):
        raise P1PreregistrationError("cost-off contract must zero every cost field")

    v4_parent = _require_mapping(manifest, "provenance.v4_parent")
    v4_load = _require_mapping(common, "v4_load_contract")
    expected_v4_paths = {
        "feature_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_features.parquet",
        "returns_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_returns.parquet",
        "availability_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_availability.parquet",
        "metadata_path": "docs/data_quality_v4_rebuild_2018_2024_metadata.json",
        "cache_local_metadata_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_metadata.json",
    }
    if v4_load.get("loader") != "unidream.data.cache_v4.load_cache_v4":
        raise P1PreregistrationError("v4 loader is immutable")
    if v4_load.get("metadata_authority") != "repo_frozen_metadata":
        raise P1PreregistrationError("repo-frozen metadata must remain authoritative")
    if v4_load.get("require_explicit_paths") is not True or v4_load.get("cache_dir_cache_tag_fallback") != "forbidden":
        raise P1PreregistrationError("v4 loader must require explicit paths")
    if v4_load.get("cache_tag") != v4_parent.get("cache_tag"):
        raise P1PreregistrationError("v4 cache tag is not pinned to the parent metadata")
    for field, expected_path in expected_v4_paths.items():
        if v4_load.get(field) != expected_path:
            raise P1PreregistrationError(f"v4 {field} is immutable")
    if v4_load.get("metadata_path") != v4_parent.get("metadata_path"):
        raise P1PreregistrationError("v4 loader must use the frozen metadata path")
    if v4_load.get("frozen_metadata_sha256") != v4_parent.get("metadata_sha256"):
        raise P1PreregistrationError("v4 frozen metadata hash echo is immutable")
    if v4_load.get("frozen_source_provenance_digest") != v4_parent.get("source_provenance_digest"):
        raise P1PreregistrationError("v4 frozen source digest echo is immutable")
    if v4_load.get("frozen_schema_digest") != v4_parent.get("schema_digest"):
        raise P1PreregistrationError("v4 frozen schema digest echo is immutable")
    if v4_load.get("frozen_content_digests") != v4_parent.get("content_digests"):
        raise P1PreregistrationError("v4 frozen content digest echo is immutable")
    local_snapshot = _require_mapping(v4_load, "known_cache_local_snapshot")
    if local_snapshot.get("metadata_sha256") != "bade1775884cd22c8675af225b429976aa6b2c60b859b4a591c76f8a87d17450":
        raise P1PreregistrationError("known cache-local metadata hash is immutable")
    if local_snapshot.get("source_provenance_digest") != "1e78ccf3162567e799b05a1c25dbe12a1c4c37e8e5a2abf2f9b95a70c380e2db":
        raise P1PreregistrationError("known cache-local source digest is immutable")
    if local_snapshot.get("schema_digest") != v4_parent.get("schema_digest") or local_snapshot.get("content_digests") != v4_parent.get("content_digests"):
        raise P1PreregistrationError("known cache-local content/schema digest baseline is immutable")
    if local_snapshot.get("rows") != v4_parent.get("feature_rows") or local_snapshot.get("sidecar_rows") != v4_parent.get("sidecar_rows"):
        raise P1PreregistrationError("known cache-local row-count baseline is immutable")
    if "metadata_path=metadata_path" not in str(v4_load.get("load_call", "")):
        raise P1PreregistrationError("v4 load call must pass the frozen metadata path")
    if "never pass cache-local metadata as metadata_path" not in str(v4_load.get("cache_local_metadata_policy", "")):
        raise P1PreregistrationError("cache-local metadata must remain audit-only")
    if "do not hide" not in str(v4_load.get("cache_local_frozen_difference_policy", "")):
        raise P1PreregistrationError("cache-local/frozen differences must be visible")
    required_echoes = {
        "v4_feature_path",
        "v4_returns_path",
        "v4_availability_path",
        "v4_frozen_metadata_path",
        "v4_frozen_metadata_sha256",
        "v4_frozen_source_provenance_digest",
        "v4_cache_local_metadata_path",
        "v4_cache_local_metadata_sha256",
        "v4_cache_local_source_provenance_digest",
        "v4_cache_local_schema_digest",
        "v4_cache_local_content_digests",
        "v4_cache_local_row_counts",
    }
    if not required_echoes.issubset(set(v4_load.get("artifact_echo_fields", []))):
        raise P1PreregistrationError("v4 provenance echo fields are incomplete")
    frozen_metadata_path = _artifact_path(root_path, str(v4_load.get("metadata_path", "")))
    if exact_file_sha256(frozen_metadata_path) != v4_parent.get("metadata_sha256"):
        raise P1PreregistrationError("frozen v4 metadata file hash mismatch")
    try:
        frozen_metadata = json.loads(frozen_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P1PreregistrationError("could not parse frozen v4 metadata") from exc
    if not isinstance(frozen_metadata, Mapping):
        raise P1PreregistrationError("frozen v4 metadata must be an object")
    for field in ("cache_tag", "schema_version", "schema_digest", "source_provenance_digest", "content_digests"):
        if frozen_metadata.get(field) != v4_parent.get(field if field != "source_provenance_digest" else "source_provenance_digest"):
            raise P1PreregistrationError(f"frozen v4 metadata {field} mismatch")
    if frozen_metadata.get("rows") != v4_parent.get("feature_rows") or frozen_metadata.get("sidecar_rows") != v4_parent.get("sidecar_rows"):
        raise P1PreregistrationError("frozen v4 metadata row counts mismatch")

    ledger = _require_mapping(common, "trial_registry")
    ledger_path = _artifact_path(root_path, str(ledger.get("path", "")))
    if exact_file_sha256(ledger_path) != ledger.get("sha256"):
        raise P1PreregistrationError("reporting-arm ledger hash mismatch")
    comparisons = _require_mapping(common, "primary_comparison_registry")
    comparisons_path = _artifact_path(root_path, str(comparisons.get("path", "")))
    if exact_file_sha256(comparisons_path) != comparisons.get("sha256"):
        raise P1PreregistrationError("primary comparison registry hash mismatch")
    try:
        comparison_rows = [
            json.loads(line)
            for line in comparisons_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise P1PreregistrationError("could not parse primary comparison registry") from exc
    primary_rows = [row for row in comparison_rows if row.get("primary") is True]
    if len(primary_rows) != comparisons.get("family_size"):
        raise P1PreregistrationError("primary comparison family size does not derive from registry")


def _path_value(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise P1PreregistrationError(
                f"critical preregistration field is missing: {path}"
            )
        current = current[component]
    return current


def _require_mapping(payload: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    value = _path_value(payload, path)
    if not isinstance(value, Mapping):
        raise P1PreregistrationError(f"{path} must be an object")
    return value


def validate_fixed_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_digest: str | None = None,
    allow_pending_artifact_hashes: bool = False,
) -> None:
    """Validate the fixed schema and reject any changed registered value.

    ``allow_pending_artifact_hashes`` exists only for preregistration review
    while a dependent P0-C contract is still being finalized.  A production
    runner must leave it false so an unpinned artifact fails closed.
    """
    if not isinstance(manifest, Mapping):
        raise P1PreregistrationError("manifest must be an object")
    missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in manifest]
    if missing:
        raise P1PreregistrationError(
            "manifest is missing required top-level fields: " + ", ".join(missing)
        )
    if manifest["manifest_id"] != "p1-recovery-preregister-20260830-v1":
        raise P1PreregistrationError("unexpected manifest_id")
    if manifest["schema_version"] != 1:
        raise P1PreregistrationError("unsupported preregistration schema_version")
    if manifest["status"] != "preregistered":
        raise P1PreregistrationError("manifest status must remain preregistered")
    if manifest["base_revision"] != REGISTERED_BASE_REVISION:
        raise P1PreregistrationError("base_revision differs from registered origin/main")
    if not isinstance(manifest["critical_field_paths"], list):
        raise P1PreregistrationError("critical_field_paths must be a list")
    if any(not isinstance(path, str) or not path for path in manifest["critical_field_paths"]):
        raise P1PreregistrationError("critical_field_paths must contain non-empty strings")
    for path in manifest["critical_field_paths"]:
        _path_value(manifest, path)

    # These are the protocol invariants that a runner must never override via
    # a convenience default.  The full manifest digest covers every other
    # field, while these checks provide readable fail-closed errors.
    common = _require_mapping(manifest, "common")
    v4_parent = _require_mapping(manifest, "provenance.v4_parent")
    v4_load = _require_mapping(common, "v4_load_contract")
    expected_v4_paths = {
        "feature_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_features.parquet",
        "returns_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_returns.parquet",
        "availability_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_availability.parquet",
        "metadata_path": "docs/data_quality_v4_rebuild_2018_2024_metadata.json",
        "cache_local_metadata_path": "checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_metadata.json",
    }
    if v4_load.get("loader") != "unidream.data.cache_v4.load_cache_v4":
        raise P1PreregistrationError("v4 loader is immutable")
    if v4_load.get("metadata_authority") != "repo_frozen_metadata":
        raise P1PreregistrationError("repo-frozen metadata must remain authoritative")
    if v4_load.get("require_explicit_paths") is not True or v4_load.get("cache_dir_cache_tag_fallback") != "forbidden":
        raise P1PreregistrationError("v4 loader must require explicit paths")
    if v4_load.get("cache_tag") != v4_parent.get("cache_tag"):
        raise P1PreregistrationError("v4 cache tag is not pinned to the parent metadata")
    for field, expected_path in expected_v4_paths.items():
        if v4_load.get(field) != expected_path:
            raise P1PreregistrationError(f"v4 {field} is immutable")
    if v4_load.get("metadata_path") != v4_parent.get("metadata_path"):
        raise P1PreregistrationError("v4 loader must use the frozen metadata path")
    if v4_load.get("frozen_metadata_sha256") != v4_parent.get("metadata_sha256"):
        raise P1PreregistrationError("v4 frozen metadata hash echo is immutable")
    if v4_load.get("frozen_source_provenance_digest") != v4_parent.get("source_provenance_digest"):
        raise P1PreregistrationError("v4 frozen source digest echo is immutable")
    if v4_load.get("frozen_schema_digest") != v4_parent.get("schema_digest"):
        raise P1PreregistrationError("v4 frozen schema digest echo is immutable")
    if v4_load.get("frozen_content_digests") != v4_parent.get("content_digests"):
        raise P1PreregistrationError("v4 frozen content digest echo is immutable")
    local_snapshot = _require_mapping(v4_load, "known_cache_local_snapshot")
    if local_snapshot.get("metadata_sha256") != "bade1775884cd22c8675af225b429976aa6b2c60b859b4a591c76f8a87d17450":
        raise P1PreregistrationError("known cache-local metadata hash is immutable")
    if local_snapshot.get("source_provenance_digest") != "1e78ccf3162567e799b05a1c25dbe12a1c4c37e8e5a2abf2f9b95a70c380e2db":
        raise P1PreregistrationError("known cache-local source digest is immutable")
    if local_snapshot.get("schema_digest") != v4_parent.get("schema_digest") or local_snapshot.get("content_digests") != v4_parent.get("content_digests"):
        raise P1PreregistrationError("known cache-local content/schema digest baseline is immutable")
    if local_snapshot.get("rows") != v4_parent.get("feature_rows") or local_snapshot.get("sidecar_rows") != v4_parent.get("sidecar_rows"):
        raise P1PreregistrationError("known cache-local row-count baseline is immutable")
    if "metadata_path=metadata_path" not in str(v4_load.get("load_call", "")):
        raise P1PreregistrationError("v4 load call must pass the frozen metadata path")
    if "never pass cache-local metadata as metadata_path" not in str(v4_load.get("cache_local_metadata_policy", "")):
        raise P1PreregistrationError("cache-local metadata must remain audit-only")
    if "do not hide" not in str(v4_load.get("cache_local_frozen_difference_policy", "")):
        raise P1PreregistrationError("cache-local/frozen differences must be visible")
    if not {
        "v4_feature_path",
        "v4_returns_path",
        "v4_availability_path",
        "v4_frozen_metadata_path",
        "v4_frozen_metadata_sha256",
        "v4_frozen_source_provenance_digest",
        "v4_cache_local_metadata_path",
        "v4_cache_local_metadata_sha256",
        "v4_cache_local_source_provenance_digest",
        "v4_cache_local_schema_digest",
        "v4_cache_local_content_digests",
        "v4_cache_local_row_counts",
    }.issubset(set(v4_load.get("artifact_echo_fields", []))):
        raise P1PreregistrationError("v4 provenance echo fields are incomplete")
    if common.get("data_frequency") != "15m":
        raise P1PreregistrationError("common.data_frequency is immutable")
    if common.get("return_unit") != "additive_log_return":
        raise P1PreregistrationError("common.return_unit is immutable")
    if common.get("forecast_horizons") != [1, 4, 8, 16]:
        raise P1PreregistrationError("common.forecast_horizons are immutable")
    if common.get("target_end_formula") != "target_end[t,h] = t + h + 1 (exclusive)":
        raise P1PreregistrationError("common.target_end_formula is immutable")
    action_contract = _require_mapping(common, "action_contract")
    expected_action_fields = {
        "commitment_bars": 4,
        "delay_bars": 1,
        "fill_mode": "all_or_none",
        "incomplete_tail_policy": "exclude_incomplete",
        "execution_skip_policy": "hold_commitment",
        "feature_unavailable_policy": "hold_and_score_commitment",
        "outcome_unavailable_policy": "exclude_block",
        "eligibility_masks_required": True,
        "boundary_cost_policy": "fill_only",
        "spread_convention": "full_quoted",
        "funding_included": False,
    }
    if any(action_contract.get(field) != value for field, value in expected_action_fields.items()):
        raise P1PreregistrationError("action commitment horizon must remain four bars")
    inventory_fields = (
        "clairvoyant_state_policy",
        "inventory_transition_rule",
        "u0_global_dp_role",
    )
    if any(not isinstance(action_contract.get(field), str) for field in inventory_fields):
        raise P1PreregistrationError("per-policy inventory isolation is required")
    if "same current inventory p_{t-1}" not in action_contract["clairvoyant_state_policy"]:
        raise P1PreregistrationError("clairvoyant state must use current policy inventory")
    if "report-only" not in action_contract["u0_global_dp_role"]:
        raise P1PreregistrationError("U0 global DP must remain report-only")
    if common.get("q_backtest_horizon") != 4 or common.get("q_action_horizons") != [4]:
        raise P1PreregistrationError("Q/backtest horizon must remain h4 only")
    if common.get("sequence_context_bars") != 64:
        raise P1PreregistrationError("sequence/context length must remain 64 bars")
    oof = _require_mapping(common, "oof")
    schedule = _require_mapping(oof, "origin_schedule")
    if schedule != {
        "first_origin": 20000,
        "step": 10000,
        "count": 8,
        "formula": "origin[k] = 20000 + 10000*k for k=0..7",
        "origins": [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],
        "batch_span": 10000,
    }:
        raise P1PreregistrationError("OOF origin schedule is immutable")
    if oof.get("min_history_rows") != 16384 or oof.get("purge_bars") != 16:
        raise P1PreregistrationError("OOF history/purge contract is immutable")
    if oof.get("split_order") != ["fit", "oof_development", "validation", "outer_test"]:
        raise P1PreregistrationError("OOF split order is immutable")
    if oof.get("train_window_rows") is not None or oof.get("train_window_rule") != "expanding eligible prefix with no cap":
        raise P1PreregistrationError("OOF train-window contract is immutable")
    if oof.get("target_mask_rule") != "all target bars t+1..t+h must have spot_bar_observed=true, a finite return, and contiguous 15m timestamps; future funding/mark masks do not invalidate a return label":
        raise P1PreregistrationError("target label mask must remain Spot-only")
    availability = _require_mapping(common, "availability")
    if availability.get("required_columns") != [
        "spot_bar_observed",
        "funding_rate_available",
        "mark_close_available",
    ]:
        raise P1PreregistrationError("availability sidecar columns are immutable")
    if availability.get("origin_context_row_rule") != "decision origin/context row requires spot_bar_observed, funding_rate_available, mark_close_available, and all 17 model features finite":
        raise P1PreregistrationError("forecast context must require all three availability masks")
    if availability.get("outcome_label_row_rule") != "each target bar t+1..t+h requires spot_bar_observed, a finite return, and contiguous 15m adjacency; funding_rate_available and mark_close_available are not required for a return label":
        raise P1PreregistrationError("outcome label availability must remain Spot-only")
    if availability.get("mask_dtype") != "strict bool only" or availability.get("missing_policy") != "fail closed":
        raise P1PreregistrationError("availability masks must be strict and fail closed")
    runner = _require_mapping(common, "runner_contract")
    if runner.get("post_output_tuning_allowed") is not False:
        raise P1PreregistrationError("post-output tuning must remain forbidden")
    if runner.get("outer_test_selection_allowed") is not False:
        raise P1PreregistrationError("outer-test selection must remain forbidden")
    if common.get("seeds") != [
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
    ]:
        raise P1PreregistrationError("synthetic seed schedule is immutable")
    gap_semantics = _require_mapping(common, "gap_scoring_semantics")
    expected_masks = {
        "forecast_origin_mask": "origin_eligible AND finite_forecast",
        "action_agreement_mask": "forecast_origin_mask AND fill_complete AND four_bar_outcome_complete",
        "pnl_scored_mask": "valid_fill_or_active_hold_commitment AND four_bar_outcome_complete; an active feature gap is scored as the committed hold",
    }
    if any(gap_semantics.get(field) != value for field, value in expected_masks.items()):
        raise P1PreregistrationError("forecast/agreement/PnL masks must remain distinct")

    contract_ref = _require_mapping(common, "action_execution_contract_reference")
    if contract_ref.get("path") != "docs/experiments/action_execution_contract.json":
        raise P1PreregistrationError("action contract path is immutable")
    contract_hash = contract_ref.get("sha256")
    if not isinstance(contract_hash, str) or (
        contract_hash == "TO_BE_COMPUTED" and not allow_pending_artifact_hashes
    ) or (contract_hash != "TO_BE_COMPUTED" and len(contract_hash) != 64):
        raise P1PreregistrationError("action contract hash must be pinned")
    mode_contracts = _require_mapping(common, "cost_mode_contracts")
    if mode_contracts.get("mode_hash_echo_required") is not True:
        raise P1PreregistrationError("cost-mode contract hashes must be echoed")
    expected_mode_paths = {
        "off": "docs/experiments/action_execution_contract_cost_off.json",
        "on": "docs/experiments/action_execution_contract.json",
    }
    for mode, expected_path in expected_mode_paths.items():
        mode_ref = _require_mapping(mode_contracts, mode)
        if mode_ref.get("path") != expected_path:
            raise P1PreregistrationError(f"cost-mode {mode} contract path is immutable")
        mode_hash = mode_ref.get("sha256")
        if not isinstance(mode_hash, str) or (
            mode_hash == "TO_BE_COMPUTED" and not allow_pending_artifact_hashes
        ) or (mode_hash != "TO_BE_COMPUTED" and len(mode_hash) != 64):
            raise P1PreregistrationError(f"cost-mode {mode} contract hash must be pinned")
    cost_modes = _require_mapping(common, "cost_modes")
    expected_cost_fields = {
        "off": {
            "spread_bps_full": 0.0,
            "half_spread_bps": 0.0,
            "slippage_bps_one_way": 0.0,
            "fee_rate_one_way": 0.0,
            "transition_cost_rate": 0.0,
        },
        "on": {
            "spread_bps_full": 3.0,
            "half_spread_bps": 1.5,
            "slippage_bps_one_way": 1.0,
            "fee_rate_one_way": 0.0003,
            "transition_cost_rate": 0.00055,
        },
    }
    for mode, expected in expected_cost_fields.items():
        configured = _require_mapping(cost_modes, mode)
        if any(configured.get(field) != value for field, value in expected.items()):
            raise P1PreregistrationError(f"cost-mode {mode} values are immutable")
        if configured.get("contract_path") != expected_mode_paths[mode]:
            raise P1PreregistrationError(f"cost-mode {mode} path echo is immutable")
    ledger = _require_mapping(common, "trial_registry")
    if ledger.get("path") != "docs/experiments/p1_recovery_trial_registry.jsonl" or ledger.get("record_count") != 56:
        raise P1PreregistrationError("reporting-arm ledger is immutable")
    ledger_hash = ledger.get("sha256")
    if not isinstance(ledger_hash, str) or (
        ledger_hash == "TO_BE_COMPUTED" and not allow_pending_artifact_hashes
    ) or (ledger_hash != "TO_BE_COMPUTED" and len(ledger_hash) != 64):
        raise P1PreregistrationError("reporting-arm ledger hash must be pinned")
    comparisons = _require_mapping(common, "primary_comparison_registry")
    if comparisons.get("path") != "docs/experiments/p1_recovery_primary_comparisons.jsonl":
        raise P1PreregistrationError("primary comparison registry path is immutable")
    if comparisons.get("family_size") != 16:
        raise P1PreregistrationError("primary comparison family size is immutable")
    comparison_hash = comparisons.get("sha256")
    if not isinstance(comparison_hash, str) or (
        comparison_hash == "TO_BE_COMPUTED" and not allow_pending_artifact_hashes
    ) or (comparison_hash != "TO_BE_COMPUTED" and len(comparison_hash) != 64):
        raise P1PreregistrationError("primary comparison registry hash must be pinned")
    bootstrap = _require_mapping(_require_mapping(common, "gates"), "block_bootstrap")
    if bootstrap.get("sensitivity_block_lengths") != [8, 16, 32]:
        raise P1PreregistrationError("bootstrap block-length sensitivity set is immutable")
    if "raw_p = max(p_block_length_8, p_block_length_16, p_block_length_32)" not in bootstrap.get("sensitivity_conservative_rule", ""):
        raise P1PreregistrationError("bootstrap conservative p aggregation is required")
    if "equal 1/10 weight" not in bootstrap.get("seed_aggregation", ""):
        raise P1PreregistrationError("seed aggregation must be equal-weighted")

    synthetic = _require_mapping(manifest, "synthetic_contract")
    if synthetic.get("n_rows") != 120000 or synthetic.get("burn_in_rows") != 512:
        raise P1PreregistrationError("synthetic row/burn-in contract is immutable")
    if synthetic.get("feature_dimension") != 17:
        raise P1PreregistrationError("synthetic feature dimension is immutable")
    if synthetic.get("base_seed_formula") != "np.random.default_rng(seed + 100)":
        raise P1PreregistrationError("synthetic base RNG is immutable")
    if "scenario_seed_formula" in synthetic:
        raise P1PreregistrationError("scenario-specific RNG would break paired S2 support")
    availability = _require_mapping(synthetic, "availability")
    if availability.get("gap_block_count") != 40 or availability.get("gap_block_length_bars") != 2:
        raise P1PreregistrationError("synthetic gap schedule is immutable")
    if availability.get("start_rng_formula") != "np.random.default_rng(seed + 50000 + source_offset)" or availability.get("shared_across_s2_levels") is not True:
        raise P1PreregistrationError("synthetic availability pairing is immutable")
    sanity = _require_mapping(synthetic, "mask_only_sanity")
    if sanity.get("minimum_eligible_fraction_across_fixed_seeds") != 0.9245 or sanity.get("gate") != ">= 0.90":
        raise P1PreregistrationError("mask-only coverage derivation is immutable")
    if sanity.get("context_required_sources") != ["spot_bar_observed", "funding_rate_available", "mark_close_available"] or sanity.get("target_required_sources") != ["spot_bar_observed"]:
        raise P1PreregistrationError("context/target source masks must remain distinct")
    expected_synthetic_splits = {
        "fit": [0, 20000],
        "oof_development": [20000, 90000],
        "validation": [90000, 100000],
        "outer_test": [100000, 120000],
    }
    if synthetic.get("splits") != expected_synthetic_splits:
        raise P1PreregistrationError("synthetic split ranges are immutable")

    scenarios = _require_mapping(manifest, "scenarios")
    for scenario_id in ("S0", "S1", "S2", "S3"):
        if scenario_id not in scenarios or not isinstance(scenarios[scenario_id], Mapping):
            raise P1PreregistrationError(f"scenario {scenario_id} is required")
        scenario = scenarios[scenario_id]
        if scenario.get("outer_test_is_report_only") is not True:
            raise P1PreregistrationError(
                f"scenarios.{scenario_id}.outer_test_is_report_only must be true"
            )

    for scenario_id in ("S0", "S1", "S2"):
        if scenarios[scenario_id].get("splits") != expected_synthetic_splits:
            raise P1PreregistrationError(
                f"scenarios.{scenario_id}.split ranges are immutable"
            )
    if scenarios["S2"].get("randomness_role") != "one shared base stream per seed for all three levels; beta is the only mutation":
        raise P1PreregistrationError("S2 shared-randomness policy is immutable")

    s3 = scenarios["S3"]
    signal = s3.get("signal")
    if not isinstance(signal, Mapping):
        raise P1PreregistrationError("scenarios.S3.signal is required")
    if signal.get("source_feature") != "close_ret":
        raise P1PreregistrationError("S3 must use the named existing close_ret feature")
    if signal.get("generated_latent") is not False:
        raise P1PreregistrationError("S3 hidden/generated latent signal is forbidden")
    if signal.get("prefix_scaling") is not True:
        raise P1PreregistrationError("S3 prefix-only scaling is required")
    if signal.get("future_target_never_used_for_scaling") is not True:
        raise P1PreregistrationError("S3 future targets cannot enter signal scaling")
    if s3.get("model_input_columns") != common.get("feature_columns"):
        raise P1PreregistrationError("S3 model input must remain canonical 17 columns")
    if s3.get("seeds") != [20260830]:
        raise P1PreregistrationError("S3 must use one deterministic preregistered seed")
    expected_raw_indices = {
        "2020-01-01T00:00:00Z": 52491,
        "first_common_row_2020-01-01T00:15:00Z": 52492,
        "2022-01-01T00:00:00Z": 104528,
        "2023-01-01T00:00:00Z": 139568,
        "2024-01-01T00:00:00Z": 173111,
    }
    if s3.get("raw_body_indices") != expected_raw_indices:
        raise P1PreregistrationError("S3 raw timestamp/index boundaries are immutable")
    if s3.get("dev_raw_range") != [52492, 139568] or s3.get("outer_test_raw_range") != [139568, 173111]:
        raise P1PreregistrationError("S3 raw split ranges are immutable")
    if s3.get("dev_origin_raw_indices") != [72492, 82492, 92492, 102492, 112492, 122492, 132492]:
        raise P1PreregistrationError("S3 development origin schedule is immutable")
    if s3.get("excluded_common_schedule_origin_raw_index") != 142492:
        raise P1PreregistrationError("S3 outer-boundary schedule exclusion is immutable")
    if s3.get("split_resolution", "").find("raw body thirds") == -1:
        raise P1PreregistrationError("S3 timestamp-aligned split rule is required")

    v4 = _require_mapping(manifest, "provenance.v4_parent")
    expected_rows = {
        "feature_rows": 173111,
        "sidecar_rows": 210336,
        "all_three_available_rows": 119849,
        "observed_spot_rows": 209805,
    }
    for field, expected in expected_rows.items():
        if v4.get(field) != expected:
            raise P1PreregistrationError(f"provenance.v4_parent.{field} is immutable")

    digest = canonical_manifest_sha256(manifest)
    self_reported = manifest.get("manifest_sha256")
    if self_reported != digest:
        raise P1PreregistrationError(
            "manifest_sha256 does not match canonical manifest content"
        )
    pinned = REGISTERED_MANIFEST_SHA256 if expected_digest is None else expected_digest
    if pinned == "TO_BE_COMPUTED":
        raise P1PreregistrationError("registered manifest digest has not been pinned")
    if digest != pinned:
        raise P1PreregistrationError(
            "manifest differs from the registered preregistration digest"
        )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def load_fixed_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> Mapping[str, Any]:
    """Load and deeply freeze the registered manifest for a future runner."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P1PreregistrationError(
            f"could not load fixed preregistration manifest: {manifest_path}"
        ) from exc
    validate_fixed_manifest(payload)
    validate_pinned_artifacts(payload, root=manifest_path.parents[2])
    return _freeze(payload)


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "P1PreregistrationError",
    "REGISTERED_BASE_REVISION",
    "REGISTERED_MANIFEST_SHA256",
    "canonical_json_sha256",
    "canonical_manifest_sha256",
    "exact_file_sha256",
    "load_fixed_manifest",
    "validate_pinned_artifacts",
    "validate_fixed_manifest",
]
