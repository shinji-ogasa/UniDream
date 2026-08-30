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
REGISTERED_MANIFEST_SHA256 = "de422979bf263677d10c689beb77b2c6ec44c26aec458779cce01083d3ceb481"
REGISTERED_BASE_REVISION = "881e5e08e9b413b51b0a2faf5c49592ce13329d1"

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
    expected_v4_runtime_policy = {
        "body_validation_policy": "the runner must call load_cache_v4 with all explicit feature, returns, availability, and frozen metadata paths, then verify content digests, schema digest, cache tag, and row counts before any S3 run",
        "source_provenance_difference_policy": "a known source-provenance-only difference is recorded separately and permits the run only when body content/schema/cache-tag/row-count checks match; promotion requires an explicit disposition field",
        "missing_unknown_mismatch_policy": "absent or unknown provenance, missing body, or any body content/schema/cache-tag/row-count mismatch blocks S3 before fitting or scoring",
        "promotion_disposition_required": True,
        "runtime_validation_entrypoint": "unidream.experiments.runtime.validate_v4_runtime_inputs",
        "runtime_validation_required_before_fit_or_score": True,
        "runtime_path_override_policy": "path_overrides may replace the four explicit body paths only as a complete set; cache_dir/cache_tag-only lookup is forbidden",
        "runtime_disposition_fields": ["status", "reason", "body_match", "source_provenance_match"],
        "runtime_disposition_statuses": ["absent", "identical", "source_provenance_only_difference"],
    }
    if any(v4_load.get(field) != value for field, value in expected_v4_runtime_policy.items()):
        raise P1PreregistrationError("v4 body/provenance runtime policy is immutable")
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
        "v4_runtime_validation_status",
        "v4_runtime_provenance_disposition",
        "v4_runtime_body_match",
        "v4_runtime_source_provenance_match",
        "v4_runtime_frozen_metadata_sha256",
        "v4_runtime_cache_local_metadata_sha256",
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
    expected_required_fields = {
        "comparison_id",
        "candidate_id",
        "baseline_id",
        "metric",
        "horizon",
        "cost_mode",
        "direction",
        "gate",
        "support_id",
        "support_range",
        "support_range_semantics",
        "support_role",
    }
    if set(comparisons.get("required_fields", [])) != expected_required_fields:
        raise P1PreregistrationError(
            "primary comparison required fields must include fixed support metadata"
        )
    expected_action_required_fields = ["action_bootstrap_replay_policy"]
    if comparisons.get("action_required_fields") != expected_action_required_fields:
        raise P1PreregistrationError(
            "action comparison required fields are not pinned"
        )
    expected_support = {
        "S0": ("synthetic_validation", [90000, 100000]),
        "S1": ("synthetic_validation", [90000, 100000]),
        "S2": ("synthetic_validation", [90000, 100000]),
        "S3": ("s3_validation", [104528, 139568]),
    }
    for row in primary_rows:
        scenario_id = row.get("scenario_id")
        try:
            expected_id, expected_range = expected_support[scenario_id]
        except KeyError as exc:
            raise P1PreregistrationError(
                "primary comparison scenario is not in the fixed support registry"
            ) from exc
        if (
            row.get("support_id") != expected_id
            or row.get("support_range") != expected_range
            or row.get("support_role") != "primary_inferential_gate"
        ):
            raise P1PreregistrationError(
                "primary comparison support must be the fixed validation operation"
            )
    expected_ids = [
        "S0__ridge__utility_vs_hold__cost_on",
        "S0__persistence__utility_vs_hold__cost_on",
        "S1__ridge__mse_vs_zero__cost_off",
        "S1__ridge__utility_vs_hold__cost_on",
        "S2__high_vs_medium__ridge__mse_skill__cost_off",
        "S2__high_vs_medium__ridge__normalized_regret__cost_on",
        "S2__high_vs_medium__ridge__utility__cost_on",
        "S2__high_vs_medium__ridge__agreement__cost_on",
        "S2__high_vs_medium__logistic__log_loss__cost_off",
        "S2__medium_vs_low__ridge__mse_skill__cost_off",
        "S2__medium_vs_low__ridge__normalized_regret__cost_on",
        "S2__medium_vs_low__ridge__utility__cost_on",
        "S2__medium_vs_low__ridge__agreement__cost_on",
        "S2__medium_vs_low__logistic__log_loss__cost_off",
        "S3__injected_vs_control__ridge__mse_skill_did__cost_off",
        "S3__injected_vs_control__ridge__utility__cost_on",
    ]
    if [row.get("comparison_id") for row in primary_rows] != expected_ids:
        raise P1PreregistrationError("primary comparison IDs/order are immutable")
    if any(row.get("horizon") != 4 or row.get("primary") is not True for row in primary_rows):
        raise P1PreregistrationError("primary comparisons must be fixed h4 records")
    fixed_range_semantics = "zero-based [start,end) right-exclusive; end excluded"
    s0_gate = (
        "Holm-rank-adjusted direction-aware lower percentile <= 0 for every fixed block length; "
        "positive-edge Holm rejection is false; never promote"
    )
    s1_mse_gate = "Holm-adjusted one-sided paired bootstrap p <= 0.05 and direction-aware point delta < 0"
    s1_utility_gate = (
        "all ten seed-level validation utility deltas > 0 and non-N/A; every seed on the identical "
        "scored mask has mean realized same-state clairvoyant net utility/value strictly greater than "
        "Ridge mean realized net utility/value; aggregate Holm-adjusted one-sided paired bootstrap "
        "p <= 0.05 and favorable point delta > 0"
    )
    s2_ge_gate = (
        "Holm-adjusted monotonic contrast p <= 0.05 and median paired contrast {pair} >= -1e-12"
    )
    s2_le_gate = (
        "Holm-adjusted monotonic contrast p <= 0.05 and median paired contrast {pair} <= 1e-12"
    )
    s3_gate = "Holm-adjusted one-sided paired bootstrap p <= 0.05 and favorable point delta > 0"
    expected_semantic_tuples = {
        "S0__ridge__utility_vs_hold__cost_on": (
            "S0__ridge__on", "S0__benchmark_hold__off", "paired_net_utility_delta_vs_hold", "on", "non_positive", s0_gate,
            "synthetic_validation", [90000, 100000],
        ),
        "S0__persistence__utility_vs_hold__cost_on": (
            "S0__persistence_last_observed__on", "S0__benchmark_hold__off", "paired_net_utility_delta_vs_hold", "on", "non_positive", s0_gate,
            "synthetic_validation", [90000, 100000],
        ),
        "S1__ridge__mse_vs_zero__cost_off": (
            "S1__ridge__off", "S1__zero_return__off", "mse_delta_vs_baseline", "off", "negative", s1_mse_gate,
            "synthetic_validation", [90000, 100000],
        ),
        "S1__ridge__utility_vs_hold__cost_on": (
            "S1__ridge__on", "S1__benchmark_hold__off", "paired_net_utility_delta_vs_hold", "on", "positive", s1_utility_gate,
            "synthetic_validation", [90000, 100000],
        ),
        "S2__high_vs_medium__ridge__mse_skill__cost_off": (
            "S2-high__ridge__off", "S2-medium__ridge__off", "forecast_mse_skill_vs_zero", "off", "high_ge_medium", s2_ge_gate.format(pair="high-medium"),
            "synthetic_validation", [90000, 100000],
        ),
        "S2__high_vs_medium__ridge__normalized_regret__cost_on": (
            "S2-high__ridge__on", "S2-medium__ridge__on", "normalized_action_regret", "on", "high_le_medium", s2_le_gate.format(pair="high-medium"),
            "synthetic_validation", [90000, 100000],
        ),
        "S2__high_vs_medium__ridge__utility__cost_on": (
            "S2-high__ridge__on", "S2-medium__ridge__on", "s2_timing_net_utility_delta", "on", "high_ge_medium", s2_ge_gate.format(pair="high-medium"),
            "synthetic_validation", [90000, 100000],
        ),
        "S2__high_vs_medium__ridge__agreement__cost_on": (
            "S2-high__ridge__on", "S2-medium__ridge__on", "feasible_action_agreement", "on", "high_ge_medium", s2_ge_gate.format(pair="high-medium"),
            "synthetic_validation", [90000, 100000],
        ),
        "S2__high_vs_medium__logistic__log_loss__cost_off": (
            "S2-high__logistic__off", "S2-medium__logistic__off", "log_loss", "off", "high_le_medium", s2_le_gate.format(pair="high-medium"),
            "synthetic_validation", [90000, 100000],
        ),
        "S2__medium_vs_low__ridge__mse_skill__cost_off": (
            "S2-medium__ridge__off", "S2-low__ridge__off", "forecast_mse_skill_vs_zero", "off", "medium_ge_low", s2_ge_gate.format(pair="medium-low"),
            "synthetic_validation", [90000, 100000],
        ),
        "S2__medium_vs_low__ridge__normalized_regret__cost_on": (
            "S2-medium__ridge__on", "S2-low__ridge__on", "normalized_action_regret", "on", "medium_le_low", s2_le_gate.format(pair="medium-low"),
            "synthetic_validation", [90000, 100000],
        ),
        "S2__medium_vs_low__ridge__utility__cost_on": (
            "S2-medium__ridge__on", "S2-low__ridge__on", "s2_timing_net_utility_delta", "on", "medium_ge_low", s2_ge_gate.format(pair="medium-low"),
            "synthetic_validation", [90000, 100000],
        ),
        "S2__medium_vs_low__ridge__agreement__cost_on": (
            "S2-medium__ridge__on", "S2-low__ridge__on", "feasible_action_agreement", "on", "medium_ge_low", s2_ge_gate.format(pair="medium-low"),
            "synthetic_validation", [90000, 100000],
        ),
        "S2__medium_vs_low__logistic__log_loss__cost_off": (
            "S2-medium__logistic__off", "S2-low__logistic__off", "log_loss", "off", "medium_le_low", s2_le_gate.format(pair="medium-low"),
            "synthetic_validation", [90000, 100000],
        ),
        "S3__injected_vs_control__ridge__mse_skill_did__cost_off": (
            "S3-injected__ridge__off", "S3-control__ridge__off", "s3_mse_skill_difference_in_differences", "off", "positive", s3_gate,
            "s3_validation", [104528, 139568],
        ),
        "S3__injected_vs_control__ridge__utility__cost_on": (
            "S3-injected__ridge__on", "S3-control__ridge__on", "s3_timing_net_utility_difference_in_differences", "on", "positive", s3_gate,
            "s3_validation", [104528, 139568],
        ),
    }
    expected_action_ids = {
        "S0__ridge__utility_vs_hold__cost_on",
        "S0__persistence__utility_vs_hold__cost_on",
        "S1__ridge__utility_vs_hold__cost_on",
        "S2__high_vs_medium__ridge__normalized_regret__cost_on",
        "S2__high_vs_medium__ridge__utility__cost_on",
        "S2__high_vs_medium__ridge__agreement__cost_on",
        "S2__medium_vs_low__ridge__normalized_regret__cost_on",
        "S2__medium_vs_low__ridge__utility__cost_on",
        "S2__medium_vs_low__ridge__agreement__cost_on",
        "S3__injected_vs_control__ridge__utility__cost_on",
    }
    expected_action_replay = (
        "resample stored canonical action block record indices and recompute declared means/sums/ratios/DiD; "
        "never replay policy state over a resampled or nonchronological sequence"
    )
    for row in primary_rows:
        comparison_id = row.get("comparison_id")
        expected = expected_semantic_tuples.get(comparison_id)
        if expected is None:
            raise P1PreregistrationError("primary comparison semantic tuple is not registered")
        actual = (
            row.get("candidate_id"), row.get("baseline_id"), row.get("metric"),
            row.get("cost_mode"), row.get("direction"), row.get("gate"),
            row.get("support_id"), row.get("support_range"),
        )
        if actual != expected:
            raise P1PreregistrationError(
                f"primary comparison semantic tuple altered: {comparison_id}"
            )
        if row.get("support_range_semantics") != fixed_range_semantics:
            raise P1PreregistrationError(
                f"primary comparison range semantics altered: {comparison_id}"
            )
        if comparison_id in expected_action_ids:
            if row.get("action_bootstrap_replay_policy") != expected_action_replay:
                raise P1PreregistrationError(
                    f"action bootstrap replay semantics altered: {comparison_id}"
                )
        elif "action_bootstrap_replay_policy" in row:
            raise P1PreregistrationError(
                f"forecast comparison cannot carry action replay semantics: {comparison_id}"
            )
    try:
        trial_ids = {
            json.loads(line)["trial_id"]
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
        }
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise P1PreregistrationError("could not parse reporting-arm ledger") from exc
    if any(
        row.get("candidate_id") not in trial_ids
        or (
            row.get("baseline_id") not in trial_ids
            and row.get("baseline_id") != f"{row.get('scenario_id')}__benchmark_hold__off"
        )
        for row in primary_rows
    ):
        raise P1PreregistrationError("primary comparisons reference unknown execution arms")
    s1_utility = next(
        row for row in primary_rows
        if row.get("comparison_id") == "S1__ridge__utility_vs_hold__cost_on"
    )
    if s1_utility.get("candidate_id") != "S1__ridge__on" or s1_utility.get("baseline_id") != "S1__benchmark_hold__off":
        raise P1PreregistrationError("S1 utility comparison arms are immutable")
    if s1_utility.get("metric") != "paired_net_utility_delta_vs_hold" or s1_utility.get("cost_mode") != "on" or s1_utility.get("direction") != "positive":
        raise P1PreregistrationError("S1 utility comparison metric is immutable")
    if s1_utility.get("gate") != (
        "all ten seed-level validation utility deltas > 0 and non-N/A; every seed on the identical scored mask has mean realized same-state clairvoyant net utility/value strictly greater than Ridge mean realized net utility/value; aggregate Holm-adjusted one-sided paired bootstrap p <= 0.05 and favorable point delta > 0"
    ):
        raise P1PreregistrationError("S1 per-seed utility/clairvoyant gate is missing")


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
    if manifest["amends_manifest_sha256"] != (
        "1ea702af170408f023f7c7b6e83eef2056df9523259b0fd9812ee99946a1c485"
    ):
        raise P1PreregistrationError("amended manifest digest is not pinned")
    if manifest["amendment_reason"] != "third pre-execution independent audit":
        raise P1PreregistrationError("amendment reason is not pinned")
    if manifest["amendment_history"] != [
        {
            "manifest_sha256": "9ba18e3e1226cbcbe57e6dfc40050036b1e70b92e58a75e73f8e6ad6c3bc747d",
            "reason": "pre-execution independent audit",
            "results_observed": False,
        },
        {
            "manifest_sha256": "5f8dbd798cf6dc44e15c94b45bc49081c1f7eefea2b89369b682e8e1c7f5d0cc",
            "reason": "first pre-execution independent audit predecessor",
            "results_observed": False,
        },
        {
            "manifest_sha256": "1ea702af170408f023f7c7b6e83eef2056df9523259b0fd9812ee99946a1c485",
            "reason": "second pre-execution independent audit predecessor",
            "results_observed": False,
        },
    ]:
        raise P1PreregistrationError("amendment history is incomplete or altered")
    if manifest["results_observed"] is not False:
        raise P1PreregistrationError("preregistration must be validated before results")
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
    expected_features = [
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
    ]
    if common.get("feature_columns") != expected_features:
        raise P1PreregistrationError("canonical feature columns are immutable")
    if common.get("model_input_rule") != (
        "X[t] is exactly the canonical current-row 17-feature vector; the 64-bar "
        "context is used only for context/availability eligibility and is never "
        "flattened or augmented with lagged or rolling features"
    ):
        raise P1PreregistrationError("model input must remain canonical current-row 17")
    if common.get("binary_label_rule") != (
        "label[t,h] = 1 iff y[t,h] > 0; exact y[t,h] == 0 maps to class 0"
    ) or common.get("binary_probability_clip_eps") != 1e-6:
        raise P1PreregistrationError("binary target/probability contract is immutable")
    if common.get("split_end_rule") != (
        "for every evaluation split, potential origins, target labels, fills, and "
        "four-bar outcomes must remain within that split's right-exclusive end; "
        "incomplete cross-boundary tails are excluded from the split score"
    ):
        raise P1PreregistrationError("split-end exclusion rule is immutable")
    if common.get("evaluation_split_state_policy") != (
        "reset each independent diagnostic, primary-validation, and outer-report "
        "split to p_start=1.0, commitment countdown=0, and position=1.0; carry "
        "policy inventory only across non-overlapping batches within that split; "
        "reset separately for model, seed, cost mode, and injected/control arm"
    ):
        raise P1PreregistrationError("split inventory reset/carry policy is immutable")
    if common.get("index_range_contract") != (
        "all numeric split_range, support_range, fit_prefix_range, prediction_range, "
        "fit_raw_range, prediction_raw_range, and body index ranges are zero-based "
        "[start,end) right-exclusive; end is excluded and the origin row is never "
        "admitted to its fit prefix"
    ):
        raise P1PreregistrationError("index range/exclusive-end contract is immutable")
    learned_fit = _require_mapping(common, "learned_fit_contract")
    expected_learned_fit = {
        "train_mask": "context_eligible AND target_complete[h] AND target_end <= origin - purge_bars AND row < origin",
        "feature_scaler": "sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True) fit separately for each origin and horizon using only train_mask rows; sklearn ddof=0; zero-variance scale is 1; transform-only on evaluation rows",
        "target_scaling": "none",
        "baseline_scaling": "none",
        "one_class_rule": "a LogisticRegression prefix with one observed class is N/A for that origin and horizon, is not repaired or oversampled, and cannot promote",
    }
    if dict(learned_fit) != expected_learned_fit:
        raise P1PreregistrationError("learned fit/scaler contract is immutable")
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
    expected_v4_runtime_policy = {
        "body_validation_policy": "the runner must call load_cache_v4 with all explicit feature, returns, availability, and frozen metadata paths, then verify content digests, schema digest, cache tag, and row counts before any S3 run",
        "source_provenance_difference_policy": "a known source-provenance-only difference is recorded separately and permits the run only when body content/schema/cache-tag/row-count checks match; promotion requires an explicit disposition field",
        "missing_unknown_mismatch_policy": "absent or unknown provenance, missing body, or any body content/schema/cache-tag/row-count mismatch blocks S3 before fitting or scoring",
        "promotion_disposition_required": True,
        "runtime_validation_entrypoint": "unidream.experiments.runtime.validate_v4_runtime_inputs",
        "runtime_validation_required_before_fit_or_score": True,
        "runtime_path_override_policy": "path_overrides may replace the four explicit body paths only as a complete set; cache_dir/cache_tag-only lookup is forbidden",
        "runtime_disposition_fields": ["status", "reason", "body_match", "source_provenance_match"],
        "runtime_disposition_statuses": ["absent", "identical", "source_provenance_only_difference"],
    }
    if any(v4_load.get(field) != value for field, value in expected_v4_runtime_policy.items()):
        raise P1PreregistrationError("v4 body/provenance runtime policy is immutable")
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
        "v4_runtime_validation_status",
        "v4_runtime_provenance_disposition",
        "v4_runtime_body_match",
        "v4_runtime_source_provenance_match",
        "v4_runtime_frozen_metadata_sha256",
        "v4_runtime_cache_local_metadata_sha256",
    }.issubset(set(v4_load.get("artifact_echo_fields", []))):
        raise P1PreregistrationError("v4 provenance echo fields are incomplete")
    if common.get("data_frequency") != "15m":
        raise P1PreregistrationError("common.data_frequency is immutable")
    if common.get("return_unit") != "additive_log_return" or common.get("return_definition") != "log(close[t] / close[t-1])":
        raise P1PreregistrationError("common.return_unit is immutable")
    if common.get("forecast_horizons") != [1, 4, 8, 16]:
        raise P1PreregistrationError("common.forecast_horizons are immutable")
    if common.get("target_end_formula") != "target_end[t,h] = t + h + 1 (exclusive)" or common.get("target_definition") != "y[t,h] = sum(return[t+1 : t+h+1])" or common.get("target_end_is_exclusive") is not True:
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
        "spread_side": "half_transition",
        "funding_included": False,
    }
    if any(action_contract.get(field) != value for field, value in expected_action_fields.items()):
        raise P1PreregistrationError("action commitment horizon must remain four bars")
    expected_action_semantics = {
        "position_range": [0.5, 1.0],
        "delta_grid": [-0.08, -0.04, 0.0, 0.04, 0.08],
        "candidate_rule": "clip(previous_position + delta, 0.5, 1.0), deduplicate after clipping",
        "candidate_dedup_rule": "canonical candidate_positions clips current_position + each candidate delta to [0.5,1.0], rounds to 12 decimal places, and applies np.unique; action selection still evaluates the canonical delta list",
        "argmax_selector": "unidream.eval.action_execution.select_block_decisions",
        "argmax_tie_rule": "max(value, -abs(delta), -delta): maximize value, then choose the smallest absolute delta, then the more-negative delta",
        "clairvoyant_state_policy": "at every scored origin, use the same current inventory p_{t-1} carried by the forecast policy; never source inventory from hindsight, U0 global DP, or a teacher trajectory",
        "action_agreement_definition": "compare forecast-optimal next position and realized four-bar one-block optimal next position from the same p_{t-1} and feasible action set",
        "regret_definition": "realized best four-bar utility minus chosen utility from the same p_{t-1}; opportunity denominator is realized clairvoyant utility minus same_state_local_hold utility from that same p_{t-1}",
        "benchmark_hold_path": "independent benchmark replay reset to p_start=1.0, commitment countdown=0, and position=1.0; delta=0/position=1.0 throughout; use the same score mask and cost-free transition semantics",
        "same_state_local_hold_path": "for regret/opportunity only, evaluate delta=0 from each candidate policy's own carried p_{t-1}; do not substitute the independent benchmark hold path",
        "s3_did_hold_path": "S3 timing DID subtracts each injected/control candidate's independent benchmark_hold_path; local same-state hold is reserved for that candidate's regret/opportunity",
        "inventory_transition_rule": "only the chosen policy action advances that policy's p_{t-1} to the next block; clairvoyant/U0 paths never feed back into policy inventory",
        "u0_global_dp_role": "report-only upper bound; it cannot define per-row clairvoyant state, action agreement, regret, or policy inventory",
    }
    if any(action_contract.get(field) != value for field, value in expected_action_semantics.items()):
        raise P1PreregistrationError("action state/selector semantics are immutable")
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
    if common.get("initial_position") != 1.0 or common.get("initial_commitment_bars_remaining") != 0:
        raise P1PreregistrationError("initial action state is immutable")
    models = _require_mapping(common, "models")
    expected_models = {
        "zero_return": {
            "kind": "fixed_baseline",
            "continuous_prediction": "zero cumulative return for every horizon h",
            "binary_prediction": "class-1 probability 0.5 for every horizon",
            "action_role": "fixed non-learned comparator; h4 action uses the same Q mapper",
        },
        "persistence_last_observed": {
            "kind": "fixed_baseline",
            "continuous_prediction": "h * return[t] where return[t] is the last observed one-bar log return at decision t",
            "binary_prediction": "class-1 probability 1-eps when return[t] > 0, otherwise eps, with eps=1e-6; exact zero is class 0",
            "action_role": "fixed non-learned comparator; h4 action uses the same Q mapper",
        },
        "ridge": {
            "kind": "sklearn.linear_model.Ridge",
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "lsqr",
            "tol": 1e-12,
            "max_iter": 10000,
            "random_state": None,
            "action_role": "sole learned h4 action mapper; no probability-to-return conversion",
        },
        "logistic": {
            "kind": "sklearn.linear_model.LogisticRegression",
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "tol": 1e-10,
            "max_iter": 1000,
            "class_weight": None,
            "random_state": 0,
            "action_role": "binary proper-score diagnostic only; action utility is N/A",
        },
    }
    if {key: dict(_require_mapping(models, key)) for key in expected_models} != expected_models:
        raise P1PreregistrationError("model IDs and fixed solver contracts are immutable")
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
    if oof.get("batch_fit_policy") != "seven chronological OOF-development batches at origins 20000..80000 plus one validation batch at origin 90000; each batch predicts only its next fixed 10000-row interval using one model fit at its origin and its admissible prefix; no later origin label can enter an earlier batch":
        raise P1PreregistrationError("OOF development/validation batch roles are immutable")
    if oof.get("oof_development_origins") != [20000, 30000, 40000, 50000, 60000, 70000, 80000] or oof.get("validation_origin") != 90000:
        raise P1PreregistrationError("OOF development/validation origins are immutable")
    if oof.get("min_history_rule") != (
        "count rows satisfying the exact train_mask after context, target, purge, and row<origin filters; this is eligible train-mask row count, not raw prefix length; if count < 16384, that model/horizon/origin is N/A and cannot promote"
    ):
        raise P1PreregistrationError("minimum history must count eligible train rows")
    if oof.get("range_semantics") != (
        "all numeric ranges in this OOF contract are zero-based [start,end) right-exclusive; fit prefixes end before the origin and never include the origin row"
    ):
        raise P1PreregistrationError("OOF range semantics are immutable")
    if oof.get("primary_inferential_support") != {
        "support_id": "synthetic_validation",
        "split": "validation",
        "origin": 90000,
        "fit_prefix_range": [0, 90000],
        "prediction_range": [90000, 100000],
        "range_semantics": "fit_prefix_range and prediction_range are zero-based [start,end) right-exclusive; origin 90000 is excluded from the fit prefix",
        "fit_rule": "one fit at origin 90000 using admissible prefix [0,90000) filtered by the train mask; score only the next validation interval [90000,100000)",
        "oof_development_role": "diagnostic_only",
        "outer_test_role": "report_only",
    }:
        raise P1PreregistrationError("synthetic primary inferential support is immutable")
    if oof.get("outer_report_operation") != {
        "origin": 100000,
        "fit_prefix_range": [0, 100000],
        "range_semantics": "fit_prefix_range and prediction_range are zero-based [start,end) right-exclusive; origin 100000 is excluded from the fit prefix",
        "fit_rule": "after every threshold and manifest field is fixed, fit exactly once at origin 100000 on the admissible prefix [0,100000) with target_end <= origin - purge_bars and label row < origin",
        "prediction_range": [100000, 120000],
        "refit_origins": [],
        "role": "report_only",
        "selection_allowed": False,
        "threshold_revision_allowed": False,
    }:
        raise P1PreregistrationError("synthetic outer report operation is immutable")
    if oof.get("min_history_rows") != 16384 or oof.get("purge_bars") != 16:
        raise P1PreregistrationError("OOF history/purge contract is immutable")
    if oof.get("split_order") != ["fit", "oof_development", "validation", "outer_test"]:
        raise P1PreregistrationError("OOF split order is immutable")
    if oof.get("train_window_rows") is not None or oof.get("train_window_rule") != "expanding eligible prefix with no cap":
        raise P1PreregistrationError("OOF train-window contract is immutable")
    if oof.get("target_mask_rule") != "all target bars t+1..t+h must have spot_bar_observed=true, a finite return, and contiguous 15m timestamps; future funding/mark masks do not invalidate a return label":
        raise P1PreregistrationError("target label mask must remain Spot-only")
    availability = _require_mapping(common, "availability")
    expected_availability = {
        "required_columns": [
            "spot_bar_observed",
            "funding_rate_available",
            "mark_close_available",
        ],
        "origin_context_row_rule": "decision origin/context row requires spot_bar_observed, funding_rate_available, mark_close_available, and all 17 model features finite",
        "outcome_label_row_rule": "each target bar t+1..t+h requires spot_bar_observed, a finite return, and contiguous 15m adjacency; funding_rate_available and mark_close_available are not required for a return label",
        "context_window_rule": "for output-coordinate decision t, require t >= 63 and every current-inclusive index in [t-63,t] (64 rows) to have consecutive 15m timestamps, finite canonical 17 features, and spot_bar_observed=true, funding_rate_available=true, and mark_close_available=true; apply after the raw burn-in slice and pass only X[t] to the model",
        "target_window_rule": "for output-coordinate decision t and horizon h, require decision row t plus target rows [t+1,...,t+h] and all h consecutive 15m edges t->t+1 through t+h-1->t+h; returns and spot masks are required only on t+1..t+h, edge t+h->t+h+1 is not required, and target_end=t+h+1 is exclusive",
        "window_rule": "one false/missing/non-contiguous required row invalidates the corresponding context or target window; required masks are not interchangeable",
        "mask_dtype": "strict bool only",
        "missing_policy": "fail closed",
        "gap_policy": "retain original timestamps and false masks; never sort, compress, interpolate, or convert missing to observed zero",
    }
    if dict(availability) != expected_availability:
        raise P1PreregistrationError("availability sidecar/window contract is immutable")
    runner = _require_mapping(common, "runner_contract")
    expected_runner_fields = {
        "manifest_is_input": True,
        "manifest_hash_must_match_registered_value": True,
        "critical_fields_required": True,
        "critical_fields_mutable_at_runtime": False,
        "scenario_order_fixed": ["S0", "S1", "S2", "S3"],
        "model_order_fixed": ["zero_return", "persistence_last_observed", "ridge", "logistic"],
        "cost_order_fixed": ["off", "on"],
        "outer_test_selection_allowed": False,
        "outer_test_rows": "report-only; never tune, select, or revise thresholds",
        "post_output_tuning_allowed": False,
        "missing_artifact_policy": "fail closed with N/A/blocked status",
        "v4_runtime_validation_entrypoint": "unidream.experiments.runtime.validate_v4_runtime_inputs",
        "v4_runtime_validation_required_before_fit_or_score": True,
        "cost_contract_consistency": "optimizer, teacher, student replay, U0, Q, and Backtest must all verify the mode-specific contract hash before scoring; a missing or mismatched hash fails closed",
        "inventory_consistency": "agreement and regret use each forecast policy's own carried p_{t-1}; U0/global hindsight inventory cannot enter row scoring or update policy state",
        "unknown_or_overridden_field_policy": "reject before fitting",
    }
    if any(runner.get(field) != value for field, value in expected_runner_fields.items()):
        raise P1PreregistrationError("runner fail-closed contract is immutable")
    required_result_echoes = {
        "manifest_id",
        "manifest_sha256",
        "base_revision",
        "scenario_id",
        "seed",
        "split_id",
        "support_id",
        "support_range",
        "fit_origin",
        "cost_mode",
        "cost_contract_hash",
        "model_id",
        "comparison_registry_sha256",
        "coverage_by_horizon_model_seed",
        "v4_runtime_validation_status",
        "v4_runtime_provenance_disposition",
        "v4_runtime_body_match",
        "v4_runtime_source_provenance_match",
        "v4_runtime_frozen_metadata_sha256",
        "v4_runtime_cache_local_metadata_sha256",
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
        "action_primitive_payload_sha256",
        "action_primitive_schema_sha256",
        "action_primitive_content_sha256",
    }
    if not required_result_echoes.issubset(set(runner.get("result_must_echo", []))):
        raise P1PreregistrationError("runner provenance/result echoes are incomplete")
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
    expected_gap_semantics = {
        "feature_or_context_gap": "decision origin is ineligible for forecast and action-agreement scoring; retain timestamp and false mask; do not synthesize a new action",
        "active_commitment_feature_gap": "after a valid fill, hold the committed position and include the four-bar PnL if its outcome window is complete; do not reoptimize or impute a feature",
        "execution_gap": "fill/observation skip follows execution_skip_policy=hold_commitment, is recorded, and is not treated as a new action",
        "outcome_gap": "any missing/non-finite/non-contiguous spot return in t+1..t+4 excludes the complete block from PnL, utility, regret, agreement, and all benchmarks; funding/mark gaps alone do not",
        "forecast_origin_mask": "origin_eligible AND finite_forecast",
        "action_agreement_mask": "forecast_origin_mask AND fill_complete AND four_bar_outcome_complete",
        "pnl_scored_mask": "valid_fill_or_active_hold_commitment AND four_bar_outcome_complete; an active feature gap is scored as the committed hold",
        "no_partial_scoring": True,
    }
    if dict(gap_semantics) != expected_gap_semantics:
        raise P1PreregistrationError("forecast/agreement/PnL masks must remain distinct")
    metrics = _require_mapping(common, "metrics")
    if metrics.get("continuous_primary") != ["mse", "mae"] or metrics.get("binary_primary") != ["log_loss", "brier_score"] or metrics.get("action_primary") != [
        "mean_net_log_utility",
        "paired_net_utility_delta_vs_hold",
        "action_regret_vs_clairvoyant",
        "normalized_action_regret",
        "feasible_action_agreement",
        "active_rate",
        "turnover",
    ] or metrics.get("coverage") != [
        "eligible_origin_fraction",
        "context_complete_fraction",
        "label_complete_fraction",
        "finite_oof_prediction_fraction",
        "scored_action_fraction",
    ]:
        raise P1PreregistrationError("metric reporting fields are immutable")
    expected_coverage = {
        "potential_origin_rule": "within each right-exclusive prediction support, an origin with 64-bar history and a target tail t+1..t+h fully inside that split end; no cross-split tail is potential",
        "context_fraction": "context_complete / potential_origins",
        "label_fraction": "target_complete[h] / potential_origins for each h",
        "eligible_fraction": "context_complete AND target_complete[h] / potential_origins for each h",
        "finite_prediction_fraction": "finite model prediction / eligible context-and-target origins for each h",
        "scored_action_fraction": "scheduled complete canonical four-bar blocks with eligible origin, finite h4 forecast, and complete realized four-bar outcome / all scheduled complete canonical four-bar blocks inside the split",
        "split_end_rule": "exclude any target, fill, or outcome crossing the split's right-exclusive end; preserve full row grid and masks",
        "reporting_scope": "echo every required horizon, model, seed, cost mode, and injected/control arm before applying thresholds; undefined or N/A coverage blocks promotion",
    }
    if dict(_require_mapping(metrics, "coverage_definitions")) != expected_coverage:
        raise P1PreregistrationError("coverage definitions are immutable")
    if metrics.get("primary_support_policy") != {
        "synthetic": {
            "support_id": "synthetic_validation",
            "split": "validation",
            "fit_prefix_range": [0, 90000],
            "prediction_range": [90000, 100000],
            "range_semantics": "fit_prefix_range and prediction_range are zero-based [start,end) right-exclusive; origin 90000 is excluded from the fit prefix",
            "origin": 90000,
            "role": "primary_inferential_gate",
        },
        "s3": {
            "support_id": "s3_validation",
            "split": "validation",
            "fit_raw_range": [52492, 104528],
            "prediction_range_raw": [104528, 139568],
            "range_semantics": "fit_raw_range and prediction_range_raw are original zero-based [start,end) raw-body indices; validation origin 104528 is excluded from the fit prefix",
            "origin_raw": 104528,
            "role": "primary_inferential_gate",
        },
        "oof_development_role": "diagnostic_only",
        "outer_test_role": "report_only; never a gate, selection, or tuning support",
    }:
        raise P1PreregistrationError("primary support policy is immutable")
    expected_metric_formulas = {
        "forecast_mse_skill": "1 - MSE(model, y_h4) / MSE(zero_return, y_h4), evaluated on the same complete-target rows",
        "normalized_action_regret": "per seed and bootstrap replicate, sum(action_regret_vs_clairvoyant) / sum(clairvoyant_net_utility - same_state_local_hold_net_utility) using the same current inventory p_{t-1} for every action; require a strictly positive aggregate opportunity denominator",
        "s2_timing_net_utility_delta": "mean over scored rows of [Ridge policy net utility - independent benchmark_hold_path net utility] at each SNR level, with the policy's own carried p_{t-1}; benchmark hold is reset p=1 and cost-free",
        "s3_mse_skill_difference_in_differences": "skill(injected Ridge vs injected zero) - skill(control Ridge vs control zero), where skill(A vs B)=1-MSE(A)/MSE(B), on identical timestamps",
        "s3_timing_net_utility_difference_in_differences": "[Ridge-minus-independent-benchmark_hold_path net utility]_injected - [Ridge-minus-independent-benchmark_hold_path net utility]_control using only a common timestamp score mask; injected/control candidate inventories reset and carry independently, and local same-state hold is only for each candidate's regret/opportunity",
        "benchmark_hold_utility_delta": "candidate net utility minus an independent p_start=1, position=1, delta=0, cost-free benchmark hold on the same score mask",
        "s1_clairvoyant_comparison": "for each S1 seed, compare mean realized net utility/value of the same-state clairvoyant and Ridge on the identical scored validation mask; require clairvoyant strictly greater, with no cumulative-path or mask mismatch",
        "s2_shared_randomness": "S2 high, medium, and low use identical base features, return-noise draws, sidecar gap masks, seeds, and row support; only beta changes",
    }
    if dict(_require_mapping(metrics, "primary_metric_formulas")) != expected_metric_formulas:
        raise P1PreregistrationError("primary metric formulas are immutable")

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
    cost_on = _require_mapping(cost_modes, "on")
    if cost_on.get("transition_cost_formula") != "0.00055 * abs(a - previous_position)" or cost_on.get("round_trip_rule") != "charge the same one-way transition rule on each position change" or cost_on.get("return_accounting") != "net_log = allocation * bar_log_return - transition_cost":
        raise P1PreregistrationError("cost-on timing/sign/accounting semantics are immutable")
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
    expected_bootstrap = {
        "method": "moving_block",
        "replicates": 2000,
        "primary_block_length": 16,
        "sensitivity_block_lengths": [8, 16, 32],
        "seed": 20260830,
        "forecast_primitive_grid": "the complete validation-split time-series row grid in original order; N/A/missing rows remain in place and are never compressed",
        "action_primitive_grid": "all structurally complete scheduled non-overlapping four-bar blocks inside the split, one record per block in original chronological order; split-local scheduled starts are 0,4,... from canonical complete_decision_starts; outcome/forecast gaps remain false-mask N/A records and are never compressed",
        "action_primitive_record_fields": "each action record stores primitive_index, decision_index, fill_index, end_index, previous_position, selected_delta, selected_position, candidate_utility, benchmark_hold_utility, same_state_local_hold_utility, clairvoyant_utility, regret, opportunity, agreement, turnover, active_indicator, origin_eligible_mask, forecast_finite_mask, fill_complete_mask, outcome_complete_mask, scored_action_mask, scenario_id, seed, split_id, support_id, model_id, cost_mode, and cost_contract_hash; selected_delta is the canonical chosen delta, selected_position is the clipped/deduplicated chosen position, previous_position is the policy state before the block, turnover=abs(selected_position-previous_position), and active_indicator=1 iff turnover>0",
        "action_primitive_schema": {
            "schedule_rule": "split-local scheduled starts are 0,4,... from canonical complete_decision_starts; global decision index = support_start + local decision index; one record per scheduled non-overlapping four-bar block in original order",
            "index_fields": ["primitive_index", "decision_index", "fill_index", "end_index"],
            "index_dtype": "int64",
            "value_fields": ["previous_position", "selected_delta", "selected_position", "candidate_utility", "benchmark_hold_utility", "same_state_local_hold_utility", "clairvoyant_utility", "regret", "opportunity", "agreement", "turnover", "active_indicator"],
            "value_dtype": "float64",
            "mask_fields": ["origin_eligible_mask", "forecast_finite_mask", "fill_complete_mask", "outcome_complete_mask", "scored_action_mask", "common_mask"],
            "mask_dtype": "bool",
            "arm_id_fields": ["scenario_id", "seed", "split_id", "support_id", "model_id", "cost_mode", "cost_contract_hash"],
            "hash_fields": ["action_primitive_payload_sha256", "action_primitive_schema_sha256", "action_primitive_content_sha256"],
            "gap_policy": "retain every scheduled primitive record in original order; mask forecast/outcome gaps false and exclude them from metric denominators without dropping or compressing records",
        },
        "action_bootstrap_replay_policy": "resample stored canonical action block record indices and recompute declared means/sums/ratios/DiD; never replay policy state over a resampled or nonchronological sequence",
        "action_inventory_boundary_policy": "inventory and state are fixed inside each stored block record; bootstrap block boundaries and duplicate sampled records never carry or replay inventory state",
        "primitive_unit_for_L": "each moving-block length L is measured in the applicable primitive records, not compacted valid rows",
        "resampling_unit": "contiguous non-circular blocks within each seed/split primitive grid; same sampled indices for candidate and paired baseline",
        "non_circular_mbb": "require n >= L; for each replicate draw starts=rng.integers(low=0, high=n-L+1, size=ceil(n/L), endpoint=False, dtype=np.int64), materialize indices=starts[:,None]+np.arange(L,dtype=np.int64) in C-order, flatten and truncate to the first n primitive records; no circular wrap and no gap compression",
        "mbb_draw_api": "starts=rng.integers(low=0,high=n-L+1,size=ceil(n/L),endpoint=False,dtype=np.int64)",
        "mbb_index_materialization": "indices = starts[:,None] + np.arange(L,dtype=np.int64); flatten in C order and take the first n indices",
        "rng_seed_formula": "20260830 + 100000*unit_code + 1000*L + seed_ordinal",
        "rng_lifecycle": "for each unit/support/seed/L create np.random.default_rng(derived_seed) exactly once, then draw all replicate starts in replicate order b=0..1999; do not reinitialize per replicate, arm, or comparison",
        "replicate_order": "b=0,1,...,1999 in ascending order",
        "quantile_method": "np.quantile(values, q, method='linear')",
        "rng_seed_ordinal": "synthetic uses the fixed seed-list ordinal 0..9; S3 uses ordinal 0",
        "index_reuse": "within a fixed unit/support/seed/L, reuse identical sampled primitive indices for every arm and comparison; seed derivation is independent of loop order",
        "paired_common_mask": "for every paired arm/comparison, use the fixed intersection eligible mask on the full primitive grid; non-common or N/A records stay present with false mask and are excluded only by metric masking",
        "paired_delta_formula": "d_i = candidate_i - baseline_i on the same primitive record; each replicate resamples the full primitive arrays and recomputes the declared metric before forming its paired contrast",
        "invalid_replicate_policy": "keep the fixed full-grid common eligible mask; sampled N/A records remain mask-out and are not dropped or compacted; N/A the entire comparison only when n<L, valid primitive count is zero, an arm's required metric is unavailable, or a denominator is zero/nonpositive",
        "denominator_policy": "if any required comparison denominator is zero or nonpositive in a replicate or required arm, mark the entire comparison N/A/blocked; never repair, omit, or resample away the denominator failure",
        "seed_aggregation": "for synthetic comparisons, independently resample blocks within each seed and compute that seed's mean; aggregate the ten seed means with equal 1/10 weight, never row-count weighting; S3 is one timestamp stratum",
        "interval_formula": "two-sided percentile interval [quantile_0.025, quantile_0.975] for each fixed block length over exactly 2000 fixed-seed replicates; intervals are diagnostic only",
        "sensitivity_conservative_rule": "for each comparison, raw_p = max(p_block_length_8, p_block_length_16, p_block_length_32); use this intersection-union conservative p as the sole input to Holm; each block-length interval remains diagnostic",
        "utility_gate": "non-S0 utility gates use the Holm-adjusted one-sided p from raw_p; no gate is formed by requiring an unadjusted lower bound at all block lengths",
    }
    if any(bootstrap.get(field) != value for field, value in expected_bootstrap.items()):
        raise P1PreregistrationError("moving-block bootstrap contract is immutable")
    expected_units = {
        "synthetic_forecast": 1,
        "synthetic_action": 2,
        "s3_forecast": 3,
        "s3_action": 4,
    }
    if bootstrap.get("rng_unit_codes") != expected_units:
        raise P1PreregistrationError("bootstrap RNG unit codes are immutable")
    expected_recompute = {
        "mse_delta": "mean(SE_candidate)-mean(SE_baseline) on the retained full grid mask",
        "skill": "1-sum(SE_model)/sum(SE_zero) on the retained full grid mask",
        "logloss": "mean per-record log loss and paired contrast on the retained full grid mask",
        "agreement": "mean agreement indicator and paired contrast on the retained full grid mask",
        "policy_utility_delta": "mean(candidate_net_utility-independent_benchmark_hold_net_utility) using the same primitive action records",
        "s2_contrast": "recompute each level's metric, then form the registry-directed adjacent level contrast",
        "normalized_regret": "recompute sum(regret)/sum(opportunity) separately at each level, require positive aggregate opportunity, then form the registry-directed contrast",
        "s3_skill_did": "recompute skill(injected Ridge vs injected zero)-skill(control Ridge vs control zero)",
        "s3_utility_did": "recompute mean(candidate-independent_benchmark_hold) in injected minus the same mean in control",
    }
    if dict(_require_mapping(bootstrap, "replicate_metric_recomputation")) != expected_recompute:
        raise P1PreregistrationError("bootstrap metric recomputation contract is immutable")
    gates = _require_mapping(common, "gates")
    if gates.get("confidence_level") != 0.95 or gates.get("familywise_alpha") != 0.05:
        raise P1PreregistrationError("gate confidence levels are immutable")
    if gates.get("multiplicity_method") != "Holm-Bonferroni over the fixed primary comparison family" or gates.get("primary_family_size") != 16:
        raise P1PreregistrationError("multiplicity family contract is immutable")
    wilson = _require_mapping(gates, "wilson")
    expected_wilson = {
        "confidence_level": 0.95,
        "z": 1.959963984540054,
        "formula": "phat=x/n; center=(phat+z^2/(2n))/(1+z^2/n); half=z*sqrt(phat*(1-phat)/n+z^2/(4n^2))/(1+z^2/n); interval=[center-half,center+half]; no normal approximation",
        "action_agreement_gate": "S1 Ridge cost-on only: point agreement >= 0.90 on every seed and pooled Wilson lower bound >= 0.90 on synthetic_validation; S2-high is evaluated only by its S2 monotonic registry comparisons",
        "coverage_gate": "all required coverage fractions must meet their fixed scenario threshold",
    }
    if dict(wilson) != expected_wilson:
        raise P1PreregistrationError("Wilson agreement contract is immutable")
    thresholds = _require_mapping(gates, "coverage_thresholds")
    expected_thresholds = {
        "synthetic_eligible_origin_fraction_min": 0.9,
        "s3_eligible_origin_fraction_min": 0.5,
        "label_complete_fraction_min": 0.9,
        "finite_oof_prediction_fraction_min": 0.95,
        "scored_action_fraction_min": 0.8,
        "target_gradient_min_if_neural_head_enabled": 1.0,
        "zero_valid_target_or_gradient": "contract failure/N/A, never model accuracy",
    }
    if dict(thresholds) != expected_thresholds:
        raise P1PreregistrationError("coverage thresholds are immutable")
    zero_signal = _require_mapping(gates, "zero_signal")
    expected_zero_signal = {
        "scope": "S0 action-capable Ridge and persistence only, cost-on; zero_return is the hold baseline, Logistic action is N/A, and cost-off is diagnostic-only",
        "promotion_rule": "no candidate may pass the positive utility or high-agreement gate; any apparent pass is a preregistration/implementation failure",
        "utility_rule": "for each L in {8,16,32}, the Holm-rank-adjusted positive-direction lower percentile for candidate-minus-independent-benchmark_hold_path net utility must be <= 0 and the positive-edge Holm rejection must be false; this is a safety gate, not evidence that the true edge is negative",
        "agreement_rule": "pooled Wilson lower bound must remain below 0.90",
    }
    if dict(zero_signal) != expected_zero_signal:
        raise P1PreregistrationError("S0 safety gate is immutable")
    high_snr = _require_mapping(gates, "high_snr_recovery")
    expected_high_snr = {
        "scope": "S1 Ridge cost-on on synthetic_validation, ten seeds; S2-high is handled by the fixed S2 monotonic comparisons",
        "action_agreement_point_min": 0.9,
        "action_agreement_pooled_wilson_lower_min": 0.9,
        "agreement_per_seed_rule": "all ten S1 Ridge cost-on validation seed-level feasible-action agreement point estimates must be >= 0.90 and non-N/A",
        "utility_rule": "all ten S1 Ridge cost-on validation seed-level candidate-minus-independent-benchmark-hold utility deltas must be strictly > 0, and the aggregate must also pass Holm-adjusted one-sided p from conservative raw_p <= 0.05 with favorable point delta; cost-off is a paired diagnostic",
        "utility_per_seed_rule": "for every seed in common.seeds, validation utility delta > 0 on the fixed synthetic_validation support; any N/A or nonpositive seed fails promotion",
        "clairvoyant_rule": "for every seed on the identical synthetic_validation scored mask, mean realized same-state clairvoyant net utility/value must be strictly greater than the S1 Ridge validation mean realized net utility/value; any N/A, mask mismatch, or non-strict comparison fails; this is an upper-bound sanity check, not a selection target",
    }
    if dict(high_snr) != expected_high_snr:
        raise P1PreregistrationError("S1 per-seed recovery gate is immutable")
    monotonicity = _require_mapping(gates, "monotonicity")
    expected_monotonicity = {
        "scope": "S2 high, medium, low SNR, evaluated on the same seed and synthetic_validation row support [90000,100000)",
        "required_order": {
            "forecast_mse_skill": "high >= medium >= low",
            "forecast_log_loss": "high <= medium <= low",
            "normalized_action_regret": "high <= medium <= low",
            "timing_net_utility_delta": "high >= medium >= low",
            "action_agreement": "high >= medium >= low",
        },
        "point_estimate": "display the median of the ten per-seed metric values; adjacent point contrasts use the median contrast",
        "point_gate": "both adjacent registry-directed median contrasts and their 1e-12 tie-tolerance direction conditions must pass; point direction and Holm-adjusted p are jointly required",
        "aggregation": "within each seed compute the metric on its common support, then the bootstrap statistic equal-weights the ten seed metric values at 1/10; point reporting/gating uses their median, never row-count weighting; S3 has one timestamp stratum",
        "tie_tolerance": 1e-12,
        "decision_formula": "for an ordered triple (high,medium,low), pass iff each adjacent inequality holds after adding tie_tolerance to the right-hand side for <= or subtracting it for >=",
        "violation_policy": "record violation and fail the monotonicity gate; no post-output tuning or scenario deletion",
    }
    if dict(monotonicity) != expected_monotonicity:
        raise P1PreregistrationError("S2 monotonicity gate is immutable")
    s3_gate = _require_mapping(gates, "s3_injected_signal")
    expected_s3_gate = {
        "scope": "injected BTC versus same-row zero-injection parent control on s3_validation [104528,139568), with one deterministic validation fit and independent benchmark hold paths",
        "utility_rule": "Holm-adjusted one-sided p from conservative raw_p <= 0.05 and favorable point delta > 0 for the injected-control timing net-utility difference-in-differences using independent benchmark_hold_path in both arms",
        "forecast_rule": "Holm-adjusted one-sided p from conservative raw_p <= 0.05 and favorable point delta > 0 for h4 MSE-skill difference-in-differences",
        "prefix_invariance_rule": "future perturbation after origin cannot alter any earlier OOF prediction, mask, or fitted-prefix digest",
    }
    if dict(s3_gate) != expected_s3_gate:
        raise P1PreregistrationError("S3 gate is immutable")

    synthetic = _require_mapping(manifest, "synthetic_contract")
    if synthetic.get("n_rows") != 120000 or synthetic.get("burn_in_rows") != 512:
        raise P1PreregistrationError("synthetic row/burn-in contract is immutable")
    if synthetic.get("feature_dimension") != 17:
        raise P1PreregistrationError("synthetic feature dimension is immutable")
    if synthetic.get("raw_n_rows") != 120512 or synthetic.get("output_slice") != (
        "raw rows [512,120512) become output rows [0,120000); discard the first burn_in_rows from features and returns"
    ):
        raise P1PreregistrationError("synthetic raw/output slicing is immutable")
    if synthetic.get("base_seed_formula") != "np.random.default_rng(seed + 100)":
        raise P1PreregistrationError("synthetic base RNG is immutable")
    if synthetic.get("draw_order") != [
        "z0 scalar",
        "xi shape (120511,)",
        "noise_features shape (120512,16) in C order",
        "epsilon shape (120512,)",
    ]:
        raise P1PreregistrationError("synthetic RNG draw order is immutable")
    if synthetic.get("random_generator") != "np.random.default_rng(seed + 100).standard_normal" or synthetic.get("random_distribution") != "z0, every xi entry, every noise_features entry, and every epsilon entry are mutually independent iid standard normal N(0,1) draws" or synthetic.get("random_independence") != "z0, xi, noise_features, and epsilon are mutually independent; entries within each vector or matrix are iid" or synthetic.get("random_dtype") != "float64":
        raise P1PreregistrationError("synthetic RNG distribution/dtype contract is immutable")
    if synthetic.get("raw_array_shapes") != {
        "z_raw": [120512],
        "xi": [120511],
        "noise_features": [120512, 16],
        "epsilon": [120512],
    }:
        raise P1PreregistrationError("synthetic raw array shapes are immutable")
    if synthetic.get("state_formula") != (
        "draw z0 as one scalar; draw xi with shape (120511,), then z_raw[0]=z0 and z_raw[k]=ar_rho*z_raw[k-1]+sqrt(1-ar_rho^2)*xi[k-1] for k=1..120511"
    ):
        raise P1PreregistrationError("synthetic state recurrence is immutable")
    if synthetic.get("observed_features_formula") != (
        "x_raw[k,0] = z_raw[k]; x_raw[k,j] = noise_features[k,j-1] for j=1..16, with noise_features drawn before epsilon and all features generated before target; output x[t] = x_raw[t+512]"
    ):
        raise P1PreregistrationError("synthetic feature-generation order is immutable")
    if synthetic.get("return_formula") != (
        "r_raw[0] = return_noise_std*epsilon[0] (beta-independent sentinel); r_raw[k+1] = beta*z_raw[k] + return_noise_std*epsilon[k+1] for k=0..120510"
    ):
        raise P1PreregistrationError("synthetic return recurrence is immutable")
    if synthetic.get("base_array_reuse") != (
        "z_raw, noise_features, epsilon, and availability starts are identical for every beta; only the beta term in r_raw[k+1] changes"
    ):
        raise P1PreregistrationError("synthetic paired-array reuse is immutable")
    if "scenario_seed_formula" in synthetic:
        raise P1PreregistrationError("scenario-specific RNG would break paired S2 support")
    availability = _require_mapping(synthetic, "availability")
    if availability.get("gap_block_count") != 40 or availability.get("gap_block_length_bars") != 2:
        raise P1PreregistrationError("synthetic gap schedule is immutable")
    if availability.get("start_rng_formula") != "np.random.default_rng(seed + 50000 + source_offset)" or availability.get("shared_across_s2_levels") is not True:
        raise P1PreregistrationError("synthetic availability pairing is immutable")
    if availability.get("start_sampling_api") != (
        "rng=np.random.default_rng(seed+50000+source_offset); relative=rng.choice(119998-512,size=40,replace=False,shuffle=True); starts=np.asarray(relative,dtype=np.int64)+512"
    ):
        raise P1PreregistrationError("synthetic gap choice API is immutable")
    if availability.get("start_coordinate_system") != (
        "starts are output-coordinate indices after the raw burn-in slice; valid start values are [512,119998) and are not raw pre-slice indices"
    ):
        raise P1PreregistrationError("synthetic gap coordinate system is immutable")
    if availability.get("start_range") != (
        "without-replacement starts in output-index range [512,119998) after raw burn-in slicing; the integer population passed to choice is 119998-512"
    ) or availability.get("false_range") != (
        "output indices [start, start + gap_block_length_bars); preserve those output rows and set only the sidecar flag false"
    ):
        raise P1PreregistrationError("synthetic gap mask range/preservation is immutable")
    if availability.get("start_order_policy") != "retain the returned choice order in the artifact exactly; never sort the starts" or availability.get("interval_union_policy") != "for each source, false_mask is the union of half-open output intervals [start,start+2); starts are unique but adjacent starts may overlap bars and the union is applied":
        raise P1PreregistrationError("synthetic gap ordering/union semantics are immutable")
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
    if synthetic.get("outer_report_operation") != {
        "origin": 100000,
        "fit_prefix_range": [0, 100000],
        "range_semantics": "fit_prefix_range and prediction_range are zero-based [start,end) right-exclusive; origin 100000 is excluded from the fit prefix",
        "fit_rule": "after every threshold and manifest field is fixed, fit exactly once at origin 100000 on the admissible prefix [0,100000) with target_end <= origin - purge_bars and label row < origin",
        "prediction_range": [100000, 120000],
        "refit_origins": [],
        "role": "report_only",
        "selection_allowed": False,
        "threshold_revision_allowed": False,
    }:
        raise P1PreregistrationError("synthetic outer report operation is immutable")

    scenarios = _require_mapping(manifest, "scenarios")
    for scenario_id in ("S0", "S1", "S2", "S3"):
        if scenario_id not in scenarios or not isinstance(scenarios[scenario_id], Mapping):
            raise P1PreregistrationError(f"scenario {scenario_id} is required")
        scenario = scenarios[scenario_id]
        if scenario.get("outer_test_is_report_only") is not True:
            raise P1PreregistrationError(
                f"scenarios.{scenario_id}.outer_test_is_report_only must be true"
            )

    if scenarios["S1"].get("beta") != 0.004 or scenarios["S1"].get("snr") != 4.0:
        raise P1PreregistrationError("S1 DGP signal strength is immutable")
    if scenarios["S1"].get("seeds") != common.get("seeds"):
        raise P1PreregistrationError("S1 seed schedule must match common seeds")

    for scenario_id in ("S0", "S1", "S2"):
        if scenarios[scenario_id].get("splits") != expected_synthetic_splits:
            raise P1PreregistrationError(
                f"scenarios.{scenario_id}.split ranges are immutable"
            )
    if scenarios["S2"].get("randomness_role") != "one shared base stream per seed for all three levels; beta is the only mutation":
        raise P1PreregistrationError("S2 shared-randomness policy is immutable")
    if scenarios["S2"].get("seeds") != common.get("seeds") or scenarios["S2"].get("levels") != {
        "high": {"beta": 0.004, "snr": 4.0},
        "medium": {"beta": 0.001, "snr": 1.0},
        "low": {"beta": 0.00025, "snr": 0.25},
    }:
        raise P1PreregistrationError("S2 beta/SNR levels are immutable")

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
    expected_signal = {
        "source_feature": "close_ret",
        "generated_latent": False,
        "observable_at": "decision timestamp t after the close of bar t",
        "prefix_scaling": True,
        "prefix_scaling_formula": "z[t] = (close_ret[t] - mean(close_ret[u] for context_eligible[u] and u<t)) / max(std(close_ret[u] for context_eligible[u] and u<t, ddof=0), 1e-12)",
        "prefix_eligibility": "context_eligible[u] only; target_complete and any future target/return mask are never inspected for prefix scaling",
        "prefix_rows_min": 256,
        "future_target_never_used_for_scaling": True,
        "injection_formula": "only when context_eligible[t], t+1 is inside the feature body, t+1 is contiguous with t, and spot_bar_observed[t+1] is true: returns_injected[t+1] = returns_v4[t+1] + 0.0005*z[t]",
        "zero_injection_control_formula": "returns_control[t+1] = returns_v4[t+1]",
        "injection_beta": 0.0005,
        "control_beta": 0.0,
        "apply_only_to": "context-eligible decision origins with an in-body, contiguous, spot-observed t+1 bar; invalid/gapped rows remain false and are not repaired",
        "features_recomputed": False,
        "hidden_state_policy": "no generated z or latent column may be passed to the model; z is recomputable from the original named v4 close_ret feature and prefix fit record",
    }
    if dict(signal) != expected_signal:
        raise P1PreregistrationError("S3 observable prefix injection contract is immutable")
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
    if s3.get("dev_raw_range") != [52492, 104528] or s3.get("outer_test_raw_range") != [139568, 173111]:
        raise P1PreregistrationError("S3 raw split ranges are immutable")
    if s3.get("oof_development_raw_range") != [72492, 104528]:
        raise P1PreregistrationError("S3 OOF-development range is immutable")
    if s3.get("dev_origin_raw_indices") != [72492, 82492, 92492, 102492]:
        raise P1PreregistrationError("S3 development origin schedule is immutable")
    if s3.get("dev_batch_spans_raw") != [[72492, 82492], [82492, 92492], [92492, 102492], [102492, 104528]]:
        raise P1PreregistrationError("S3 development batch spans are immutable")
    if s3.get("primary_inferential_operation") != {
        "support_id": "s3_validation",
        "origin_raw_index": 104528,
        "fit_raw_range": [52492, 104528],
        "prediction_raw_range": [104528, 139568],
        "range_semantics": "fit_raw_range and prediction_raw_range are original zero-based [start,end) raw-body indices; validation origin 104528 is excluded from the fit prefix",
        "fit_rule": "one fixed fit at the validation boundary using only admissible pre-validation rows with target_end <= origin - purge_bars; no validation or outer target enters the fit",
        "refit_origins": [],
        "role": "primary_inferential_gate",
        "oof_development_role": "diagnostic_only",
        "outer_test_role": "report_only",
        "selection_allowed": False,
        "threshold_revision_allowed": False,
    }:
        raise P1PreregistrationError("S3 primary inferential operation is immutable")
    if s3.get("excluded_common_schedule_origin_raw_index") != 142492:
        raise P1PreregistrationError("S3 outer-boundary schedule exclusion is immutable")
    if s3.get("outer_report_origin_raw_index") != 139568 or s3.get("outer_report_fit_raw_range") != [52492, 139568] or s3.get("outer_report_prediction_raw_range") != [139568, 173111] or s3.get("outer_report_refit_origins") != [] or s3.get("outer_report_range_semantics") != "outer_report_fit_raw_range and outer_report_prediction_raw_range are original zero-based [start,end) raw-body indices; outer origin 139568 is excluded from the fit prefix":
        raise P1PreregistrationError("S3 outer report operation is immutable")
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
