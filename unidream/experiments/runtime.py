from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import yaml

from unidream.data.availability_contract import (
    AvailabilityContractError,
    validate_availability,
)
from unidream.data.cache_v4 import CacheV4Error, cache_v4_paths, load_cache_v4
from unidream.data.download import (
    fetch_binance_ohlcv,
    fetch_funding_rate,
    fetch_mark_price_klines,
    fetch_open_interest_hist,
)
from unidream.data.features import align_extra_series, compute_features, get_raw_returns
from unidream.experiments.checkpointing import atomic_text_write


CACHE_CONTRACT_VERSION = 1
_BASE_FEATURE_COLUMNS = {
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
}


def _attach_availability_metadata(
    features_df: pd.DataFrame,
    raw_returns: pd.Series,
    availability: pd.DataFrame,
    *,
    include_funding: bool,
    include_mark: bool,
    interval: str,
) -> pd.DataFrame:
    """Validate a v4 sidecar and attach it without changing model columns.

    Existing callers intentionally continue to unpack ``(features, returns)``.
    The sidecar travels through pandas attrs and is consumed by ``WFODataset``
    and ``SequenceDataset``; an optional runtime return flag below is available
    for new callers that want the explicit third value.
    """
    try:
        selected = validate_availability(
            availability,
            features_df.index,
            include_funding=include_funding,
            include_mark=include_mark,
        )
    except AvailabilityContractError as exc:
        raise ValueError(f"availability sidecar failed training eligibility validation: {exc}") from exc
    attrs = dict(features_df.attrs)
    attrs.update(
        {
            "availability": selected.sidecar,
            "availability_interval": str(interval),
            "availability_include_funding": bool(include_funding),
            "availability_include_mark": bool(include_mark),
            "availability_required_columns": list(selected.required_columns),
            "availability_row_eligible": selected.row_eligible.copy(),
            "availability_status": "v4_verified",
        }
    )
    features_df.attrs = attrs
    returns_attrs = dict(raw_returns.attrs)
    returns_attrs.update(
        {
            "availability": selected.sidecar,
            "availability_interval": str(interval),
            "availability_include_funding": bool(include_funding),
            "availability_include_mark": bool(include_mark),
            "availability_required_columns": list(selected.required_columns),
            "availability_row_eligible": selected.row_eligible.copy(),
            "availability_status": "v4_verified",
        }
    )
    raw_returns.attrs = returns_attrs
    return selected.sidecar


def _return_training_frames(
    features_df: pd.DataFrame,
    raw_returns: pd.Series,
    availability: pd.DataFrame | None,
    *,
    return_availability: bool,
):
    if return_availability:
        return features_df, raw_returns, availability
    return features_df, raw_returns


def resolve_cache_pair(cache_dir: str, cache_tag: str) -> tuple[str, str]:
    features_cache = os.path.join(cache_dir, f"{cache_tag}_features.parquet")
    returns_cache = os.path.join(cache_dir, f"{cache_tag}_returns.parquet")
    return features_cache, returns_cache


def read_optional_parquet(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=False)
            df = df.set_index("time")
        elif "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
            df = df.drop(columns=["timestamp"]).set_index(ts.rename("time"))
    return df.sort_index()


def read_extra_series_caches(cache_dir: str, cache_tag: str) -> dict[str, pd.Series]:
    series_map: dict[str, pd.Series] = {}
    prefix = f"{cache_tag}_series_"
    if not os.path.isdir(cache_dir):
        return series_map
    for filename in sorted(os.listdir(cache_dir)):
        if not filename.startswith(prefix) or not filename.endswith(".parquet"):
            continue
        path = os.path.join(cache_dir, filename)
        df = read_optional_parquet(path)
        if df is None or df.empty or df.shape[1] == 0:
            continue
        name = filename[len(prefix) : -len(".parquet")]
        series_map[name] = df.iloc[:, 0].rename(name)
    return series_map


def _cache_metadata_path(cache_dir: str, cache_tag: str) -> str:
    return os.path.join(cache_dir, f"{cache_tag}_metadata.json")


def cache_quality_status(cache_dir: str, cache_tag: str) -> str:
    """Return an explicit status for a legacy or schema v4 cache hit.

    Historical v3 files intentionally remain readable for compatibility, but
    they are never reported as quality-passed because they have no
    availability sidecar.  A partial or invalid v4 set is also surfaced as a
    failure instead of triggering a raw-data rebuild that could hide the
    broken artifact.
    """
    paths = cache_v4_paths(cache_dir, cache_tag)
    metadata_path = paths["metadata"]
    availability_path = paths["availability"]
    metadata: dict | None = None
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                metadata = payload
        except (OSError, ValueError, json.JSONDecodeError):
            return "v4_invalid" if availability_path.exists() else "legacy_v3_unverified"
    if not availability_path.exists() and (metadata is None or metadata.get("schema_version") != 4):
        return "legacy_v3_unverified"
    if not all(path.exists() for path in paths.values()):
        return "v4_incomplete"
    try:
        load_cache_v4(cache_dir, cache_tag)
    except CacheV4Error:
        return "v4_invalid"
    return "v4_verified"


class V4RuntimeInputError(ValueError):
    """Raised when a preregistered v4 runtime input cannot be verified."""


_V4_RUNTIME_BODY_FIELDS = ("feature_path", "returns_path", "availability_path", "metadata_path")
_V4_RUNTIME_DISPOSITION_FIELDS = (
    "status",
    "reason",
    "body_match",
    "source_provenance_match",
)
_V4_RUNTIME_DISPOSITION_STATUSES = (
    "absent",
    "identical",
    "source_provenance_only_difference",
)
_V4_RUNTIME_BODY_METADATA_FIELDS = (
    "cache_tag",
    "schema_version",
    "schema_digest",
    "content_digests",
    "rows",
    "sidecar_rows",
    "feature_columns",
    "availability_columns",
    "returns_columns",
)

# ``validate_v4_runtime_inputs`` deliberately remains a small body validator.
# It is useful in cache fixtures and in tests which construct a temporary v4
# body, but accepting an arbitrary mapping there must never be mistaken for an
# authenticated P1 run boundary.  The production entrypoint below pins the
# manifest first and then delegates to this body validator.
V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT = (
    "unidream.experiments.runtime.validate_v4_runtime_inputs"
)
P1_V4_RUNTIME_VALIDATION_ENTRYPOINT = (
    "unidream.experiments.runtime.validate_p1_v4_runtime_inputs"
)


def _v4_runtime_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _v4_runtime_resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _v4_runtime_require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V4RuntimeInputError(f"{label} must be an object")
    return value


def _v4_runtime_plain(value: Any) -> Any:
    """Normalize frozen tuples/mappings for authenticated identity checks."""
    if isinstance(value, Mapping):
        return {key: _v4_runtime_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_v4_runtime_plain(item) for item in value]
    return value


def _v4_runtime_body_metadata_matches(
    candidate: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for field in _V4_RUNTIME_BODY_METADATA_FIELDS:
        expected_value = expected.get(field)
        candidate_value = candidate.get(field)
        if candidate_value != expected_value:
            raise V4RuntimeInputError(
                f"{label} {field} mismatch: {candidate_value!r} != {expected_value!r}"
            )


def _v4_runtime_validate_disposition(
    disposition: Mapping[str, Any],
    derived: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [field for field in _V4_RUNTIME_DISPOSITION_FIELDS if field not in disposition]
    if missing:
        raise V4RuntimeInputError(
            "v4 provenance disposition is missing fields: " + ", ".join(missing)
        )
    status = disposition.get("status")
    if status not in _V4_RUNTIME_DISPOSITION_STATUSES:
        raise V4RuntimeInputError(f"unknown v4 provenance disposition status: {status!r}")
    if not isinstance(disposition.get("reason"), str) or not disposition["reason"].strip():
        raise V4RuntimeInputError("v4 provenance disposition reason must be non-empty")
    for field in ("body_match", "source_provenance_match"):
        value = disposition.get(field)
        if value is not None and not isinstance(value, bool):
            raise V4RuntimeInputError(f"v4 provenance disposition {field} must be bool or null")
    for field in ("status", "body_match", "source_provenance_match"):
        if disposition.get(field) != derived.get(field):
            raise V4RuntimeInputError(
                f"v4 provenance disposition {field} does not match observed inputs"
            )
    return dict(disposition)


def validate_v4_runtime_inputs(
    manifest: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    path_overrides: Mapping[str, str | Path] | None = None,
    paths: Mapping[str, str | Path] | None = None,
    feature_path: str | Path | None = None,
    returns_path: str | Path | None = None,
    availability_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    cache_local_metadata_path: str | Path | None = None,
    provenance_disposition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate explicit v4 bodies before a preregistered run can fit or score.

    The frozen repository metadata is always passed to ``load_cache_v4`` as the
    metadata path.  A cache-local metadata file is an optional audit snapshot;
    when present, its body fields must match the frozen metadata and a source
    provenance-only difference is surfaced in the returned disposition.  No
    cache-directory fallback or data repair is performed here.
    """
    if not isinstance(manifest, Mapping):
        raise V4RuntimeInputError("manifest must be an object")
    common = _v4_runtime_require_mapping(manifest.get("common"), "common")
    contract = _v4_runtime_require_mapping(common.get("v4_load_contract"), "common.v4_load_contract")
    parent = _v4_runtime_require_mapping(
        _v4_runtime_require_mapping(manifest.get("provenance"), "provenance").get("v4_parent"),
        "provenance.v4_parent",
    )
    if contract.get("loader") != "unidream.data.cache_v4.load_cache_v4":
        raise V4RuntimeInputError("manifest does not pin the v4 cache loader")
    if contract.get("require_explicit_paths") is not True or contract.get("cache_dir_cache_tag_fallback") != "forbidden":
        raise V4RuntimeInputError("v4 runtime requires explicit paths and forbids cache fallback")
    runtime_entrypoint = contract.get("runtime_validation_entrypoint")
    body_validator_entrypoint = contract.get("runtime_body_validator_entrypoint")
    if body_validator_entrypoint is not None:
        if body_validator_entrypoint != V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT:
            raise V4RuntimeInputError("manifest does not pin the v4 body validator")
        if runtime_entrypoint != P1_V4_RUNTIME_VALIDATION_ENTRYPOINT:
            raise V4RuntimeInputError(
                "manifest does not pin the authenticated P1 v4 runtime validator"
            )
    elif runtime_entrypoint != V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT:
        # Legacy fixture-shaped contracts predate the authenticated wrapper.
        # Keep the generic function usable for those fixtures, while the fixed
        # P1 manifest validator requires the two explicit entrypoint fields.
        raise V4RuntimeInputError("manifest does not pin the v4 body validator")
    if contract.get("runtime_validation_required_before_fit_or_score") is not True:
        raise V4RuntimeInputError("v4 runtime validation is not required before fit/score")
    if contract.get("runtime_disposition_fields") != list(_V4_RUNTIME_DISPOSITION_FIELDS):
        raise V4RuntimeInputError("v4 runtime disposition fields are not pinned")
    if contract.get("runtime_disposition_statuses") != list(_V4_RUNTIME_DISPOSITION_STATUSES):
        raise V4RuntimeInputError("v4 runtime disposition statuses are not pinned")

    merged_overrides: dict[str, str | Path] = {}
    for source_name, source in (("path_overrides", path_overrides), ("paths", paths)):
        if source is None:
            continue
        if not isinstance(source, Mapping):
            raise V4RuntimeInputError(f"{source_name} must be an object")
        for key, value in source.items():
            if key in merged_overrides and merged_overrides[key] != value:
                raise V4RuntimeInputError(f"conflicting v4 path override for {key!r}")
            merged_overrides[str(key)] = value
    keyword_overrides = {
        "feature_path": feature_path,
        "returns_path": returns_path,
        "availability_path": availability_path,
        "metadata_path": metadata_path,
    }
    for key, value in keyword_overrides.items():
        if value is not None:
            if key in merged_overrides and merged_overrides[key] != value:
                raise V4RuntimeInputError(f"conflicting v4 path override for {key!r}")
            merged_overrides[key] = value
    aliases = {"features": "feature_path", "returns": "returns_path", "availability": "availability_path", "metadata": "metadata_path"}
    normalised_overrides: dict[str, str | Path] = {}
    for key, value in merged_overrides.items():
        canonical_key = aliases.get(key, key)
        if canonical_key not in _V4_RUNTIME_BODY_FIELDS:
            raise V4RuntimeInputError(f"unknown v4 path override: {key!r}")
        if canonical_key in normalised_overrides and normalised_overrides[canonical_key] != value:
            raise V4RuntimeInputError(f"conflicting v4 path override for {canonical_key!r}")
        normalised_overrides[canonical_key] = value
    if normalised_overrides and set(normalised_overrides) != set(_V4_RUNTIME_BODY_FIELDS):
        missing = sorted(set(_V4_RUNTIME_BODY_FIELDS) - set(normalised_overrides))
        raise V4RuntimeInputError(
            "v4 path overrides must provide all explicit body paths: " + ", ".join(missing)
        )

    root_path = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    configured_paths = {
        field: normalised_overrides.get(field, contract.get(field))
        for field in _V4_RUNTIME_BODY_FIELDS
    }
    if any(value is None or not str(value) for value in configured_paths.values()):
        raise V4RuntimeInputError("v4 manifest is missing an explicit body path")
    resolved_paths = {
        field: _v4_runtime_resolve_path(value, root_path)
        for field, value in configured_paths.items()
    }
    missing_paths = [str(path) for path in resolved_paths.values() if not path.is_file()]
    if missing_paths:
        raise V4RuntimeInputError("v4 runtime body is incomplete; missing files: " + ", ".join(missing_paths))

    expected_cache_tag = parent.get("cache_tag")
    if contract.get("cache_tag") != expected_cache_tag:
        raise V4RuntimeInputError("v4 manifest/cache metadata cache-tag mismatch")
    try:
        features, returns, availability, frozen_metadata = load_cache_v4(
            cache_tag=str(expected_cache_tag),
            feature_path=resolved_paths["feature_path"],
            returns_path=resolved_paths["returns_path"],
            availability_path=resolved_paths["availability_path"],
            metadata_path=resolved_paths["metadata_path"],
        )
    except (CacheV4Error, OSError, TypeError, ValueError) as exc:
        raise V4RuntimeInputError(f"explicit v4 body validation failed: {exc}") from exc

    expected_frozen_metadata = {
        "cache_tag": parent.get("cache_tag"),
        "schema_version": parent.get("schema_version"),
        "schema_digest": parent.get("schema_digest"),
        "content_digests": parent.get("content_digests"),
        "rows": parent.get("feature_rows"),
        "sidecar_rows": parent.get("sidecar_rows"),
        "feature_columns": list(common.get("feature_columns", [])),
        "availability_columns": list(parent.get("required_availability_columns", [])),
        "returns_columns": ["returns"],
    }
    _v4_runtime_body_metadata_matches(
        frozen_metadata,
        expected_frozen_metadata,
        label="frozen v4 metadata",
    )
    if frozen_metadata.get("source_provenance_digest") != parent.get("source_provenance_digest"):
        raise V4RuntimeInputError("frozen v4 source provenance digest mismatch")
    frozen_metadata_sha256 = _v4_runtime_sha256(resolved_paths["metadata_path"])
    if frozen_metadata_sha256 != parent.get("metadata_sha256"):
        raise V4RuntimeInputError("frozen v4 metadata file SHA-256 mismatch")

    local_path_value = cache_local_metadata_path
    if local_path_value is None:
        local_path_value = contract.get("cache_local_metadata_path")
    local_path = (
        _v4_runtime_resolve_path(local_path_value, root_path)
        if local_path_value is not None and str(local_path_value)
        else None
    )
    local_metadata: Mapping[str, Any] | None = None
    local_sha256: str | None = None
    local_source_digest: str | None = None
    local_body_match: bool | None = None
    local_source_match: bool | None = None
    if local_path is not None and local_path.exists() and not local_path.is_file():
        raise V4RuntimeInputError(f"cache-local v4 metadata path is not a file: {local_path}")
    if local_path is not None and local_path.is_file():
        try:
            local_payload = json.loads(local_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise V4RuntimeInputError(f"could not parse cache-local v4 metadata: {local_path}") from exc
        local_metadata = _v4_runtime_require_mapping(local_payload, "cache-local v4 metadata")
        local_sha256 = _v4_runtime_sha256(local_path)
        local_source_digest = local_metadata.get("source_provenance_digest")
        if not isinstance(local_source_digest, str) or not local_source_digest:
            raise V4RuntimeInputError("cache-local v4 source provenance is absent or unknown")
        source_provenance = local_metadata.get("source_provenance")
        if not isinstance(source_provenance, Mapping):
            raise V4RuntimeInputError("cache-local v4 source provenance is absent or unknown")
        source_payload = json.dumps(
            source_provenance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        if hashlib.sha256(source_payload).hexdigest() != local_source_digest:
            raise V4RuntimeInputError("cache-local v4 source provenance digest mismatch")
        try:
            _v4_runtime_body_metadata_matches(
                local_metadata,
                expected_frozen_metadata,
                label="cache-local v4 metadata",
            )
        except V4RuntimeInputError as exc:
            raise V4RuntimeInputError("cache-local v4 body metadata mismatch") from exc
        local_body_match = True
        local_source_match = local_source_digest == parent.get("source_provenance_digest")
        if not local_source_match:
            known_snapshot = contract.get("known_cache_local_snapshot")
            known_snapshot = (
                _v4_runtime_require_mapping(known_snapshot, "known_cache_local_snapshot")
                if known_snapshot is not None
                else None
            )
            if (
                known_snapshot is None
                or local_source_digest != known_snapshot.get("source_provenance_digest")
                or (
                    known_snapshot.get("metadata_sha256") is not None
                    and local_sha256 != known_snapshot.get("metadata_sha256")
                )
            ):
                raise V4RuntimeInputError("cache-local source provenance differs with an unknown digest")

    if local_path is None or local_metadata is None:
        derived_disposition = {
            "status": "absent",
            "reason": "cache-local metadata is absent; frozen repository metadata remains authoritative",
            "body_match": None,
            "source_provenance_match": None,
        }
    elif local_source_match:
        derived_disposition = {
            "status": "identical",
            "reason": "cache-local metadata body and source provenance match frozen metadata",
            "body_match": True,
            "source_provenance_match": True,
        }
    else:
        derived_disposition = {
            "status": "source_provenance_only_difference",
            "reason": "cache-local body matches but its known source provenance digest differs from frozen metadata",
            "body_match": True,
            "source_provenance_match": False,
        }
    disposition = (
        _v4_runtime_validate_disposition(provenance_disposition, derived_disposition)
        if provenance_disposition is not None
        else dict(derived_disposition)
    )
    local_content_digests = (
        dict(local_metadata.get("content_digests", {})) if local_metadata is not None else None
    )
    local_row_counts = (
        {"rows": local_metadata.get("rows"), "sidecar_rows": local_metadata.get("sidecar_rows")}
        if local_metadata is not None
        else None
    )
    result = {
        "status": "v4_runtime_validated",
        "features": features,
        "returns": returns,
        "availability": availability,
        "metadata": frozen_metadata,
        "paths": {field: str(path) for field, path in resolved_paths.items()},
        "v4_runtime_validation_status": "passed",
        "v4_runtime_provenance_disposition": disposition,
        "v4_runtime_body_match": local_body_match,
        "v4_runtime_loaded_body_match": True,
        "v4_runtime_source_provenance_match": local_source_match,
        "v4_runtime_frozen_metadata_sha256": frozen_metadata_sha256,
        "v4_runtime_cache_local_metadata_sha256": local_sha256,
        "v4_runtime_cache_local_source_provenance_digest": local_source_digest,
        "v4_runtime_cache_local_schema_digest": (
            local_metadata.get("schema_digest") if local_metadata is not None else None
        ),
        "v4_runtime_cache_local_content_digests": local_content_digests,
        "v4_runtime_cache_local_row_counts": local_row_counts,
        "v4_feature_path": str(resolved_paths["feature_path"]),
        "v4_returns_path": str(resolved_paths["returns_path"]),
        "v4_availability_path": str(resolved_paths["availability_path"]),
        "v4_frozen_metadata_path": str(resolved_paths["metadata_path"]),
        "v4_frozen_metadata_sha256": frozen_metadata_sha256,
        "v4_frozen_source_provenance_digest": frozen_metadata.get("source_provenance_digest"),
        "v4_cache_local_metadata_path": str(local_path) if local_path is not None else None,
        "v4_cache_local_metadata_sha256": local_sha256,
        "v4_cache_local_source_provenance_digest": local_source_digest,
        "v4_cache_local_schema_digest": (
            local_metadata.get("schema_digest") if local_metadata is not None else None
        ),
        "v4_cache_local_content_digests": local_content_digests,
        "v4_cache_local_row_counts": local_row_counts,
    }
    return result


def validate_p1_v4_runtime_inputs(
    manifest: Mapping[str, Any] | str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    root: str | Path | None = None,
    path_overrides: Mapping[str, str | Path] | None = None,
    paths: Mapping[str, str | Path] | None = None,
    feature_path: str | Path | None = None,
    returns_path: str | Path | None = None,
    availability_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    cache_local_metadata_path: str | Path | None = None,
    provenance_disposition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the P1 manifest before validating a v4 body.

    ``validate_v4_runtime_inputs`` is intentionally retained as the generic
    body validator used by fixtures.  A production P1 caller must enter here:
    this function always invokes :func:`load_fixed_manifest`, checks any
    caller-supplied manifest against the loaded frozen mapping, and only then
    delegates to the generic body validator.  Consequently a forged
    ``manifest_sha256``, ``results_observed`` flag, or frozen v4 digest cannot
    be paired with an otherwise valid body through this API.

    The first positional argument accepts either an optional manifest mapping
    (for explicit identity checking) or a manifest path.  A mapping is never
    used as the authority; the result is always based on ``load_fixed_manifest``.
    """
    # Import lazily to keep ordinary cache helpers independent from the P1
    # preregistration module while still making the authenticated call
    # observable/patchable in contract tests.
    from . import p1_recovery_prereg

    candidate_manifest: Mapping[str, Any] | None = None
    if manifest is not None and isinstance(manifest, Mapping):
        candidate_manifest = manifest
    elif manifest is not None:
        if manifest_path is not None:
            raise V4RuntimeInputError(
                "manifest path was supplied both positionally and by keyword"
            )
        manifest_path = manifest

    try:
        selected_manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else p1_recovery_prereg.DEFAULT_MANIFEST_PATH
        )
    except (TypeError, ValueError) as exc:
        raise V4RuntimeInputError("P1 manifest path must be path-like") from exc
    try:
        fixed_manifest = p1_recovery_prereg.load_fixed_manifest(selected_manifest_path)
    except (p1_recovery_prereg.P1PreregistrationError, OSError, TypeError, ValueError) as exc:
        raise V4RuntimeInputError(
            "authenticated P1 manifest validation failed before v4 body validation"
        ) from exc

    if candidate_manifest is not None:
        # Compare the entire mapping, not only a self-reported digest.  This
        # catches forged results_observed/frozen-digest fields even when a
        # caller recomputes the candidate's canonical digest.
        if not isinstance(candidate_manifest, Mapping):  # defensive for proxies
            raise V4RuntimeInputError("P1 manifest must be an object")
        if candidate_manifest.get("results_observed") is not False:
            raise V4RuntimeInputError(
                "authenticated P1 manifest must keep results_observed=false"
            )
        if candidate_manifest.get("manifest_sha256") != fixed_manifest.get(
            "manifest_sha256"
        ):
            raise V4RuntimeInputError("P1 manifest_sha256 differs from the pinned manifest")
        if _v4_runtime_plain(candidate_manifest) != _v4_runtime_plain(fixed_manifest):
            raise V4RuntimeInputError(
                "supplied P1 manifest differs from the authenticated fixed manifest"
            )

    # The call is deliberately explicit rather than forwarding an arbitrary
    # mapping.  Body paths remain caller-supplied only as the complete explicit
    # set enforced by the generic validator; frozen metadata expectations come
    # from the authenticated manifest.
    result = validate_v4_runtime_inputs(
        fixed_manifest,
        root=root,
        path_overrides=path_overrides,
        paths=paths,
        feature_path=feature_path,
        returns_path=returns_path,
        availability_path=availability_path,
        metadata_path=metadata_path,
        cache_local_metadata_path=cache_local_metadata_path,
        provenance_disposition=provenance_disposition,
    )
    result.update(
        {
            "manifest_id": fixed_manifest.get("manifest_id"),
            "manifest_sha256": fixed_manifest.get("manifest_sha256"),
            "base_revision": fixed_manifest.get("base_revision"),
            "results_observed": fixed_manifest.get("results_observed"),
            "p1_manifest_id": fixed_manifest.get("manifest_id"),
            "p1_manifest_sha256": fixed_manifest.get("manifest_sha256"),
            "p1_base_revision": fixed_manifest.get("base_revision"),
            "p1_results_observed": fixed_manifest.get("results_observed"),
            "p1_runtime_validation_entrypoint": P1_V4_RUNTIME_VALIDATION_ENTRYPOINT,
            "p1_runtime_body_validator_entrypoint": V4_RUNTIME_BODY_VALIDATOR_ENTRYPOINT,
        }
    )
    return result


# Descriptive aliases keep callers from bypassing the authenticated boundary
# merely because they use the shorter P1 naming used by the preregistration
# protocol.  The manifest pins the long, unambiguous name above.
validate_p1_runtime_inputs = validate_p1_v4_runtime_inputs
validate_p1_recovery_runtime_inputs = validate_p1_v4_runtime_inputs


def _cache_parameters(
    *,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    zscore_window: int,
    extra_series_mode: str,
    extra_series_include: list[str] | None,
    include_funding: bool,
    include_oi: bool,
    include_mark: bool,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "interval": interval,
        "start": start,
        "end": end,
        "zscore_window_days": int(zscore_window),
        "extra_series_mode": str(extra_series_mode),
        "extra_series_include": sorted(str(name) for name in (extra_series_include or [])),
        "include_funding": bool(include_funding),
        "include_oi": bool(include_oi),
        "include_mark": bool(include_mark),
    }


def _validate_training_cache(
    features_df: pd.DataFrame,
    raw_returns: pd.Series,
    *,
    include_funding: bool,
    include_oi: bool,
    include_mark: bool,
    cache_tag: str,
) -> None:
    if features_df.empty or raw_returns.empty:
        raise ValueError(f"cache {cache_tag} is empty")
    if not isinstance(features_df.index, pd.DatetimeIndex):
        raise ValueError(f"cache {cache_tag} features index is not DatetimeIndex")
    if not features_df.index.is_monotonic_increasing or not features_df.index.is_unique:
        raise ValueError(f"cache {cache_tag} features index is not sorted and unique")
    if not isinstance(raw_returns.index, pd.DatetimeIndex):
        raise ValueError(f"cache {cache_tag} returns index is not DatetimeIndex")
    if not features_df.index.equals(raw_returns.index):
        raise ValueError(f"cache {cache_tag} features/returns indices differ")
    columns = set(str(column) for column in features_df.columns)
    required = set(_BASE_FEATURE_COLUMNS)
    if include_funding:
        required.add("funding_rate")
    if include_oi:
        required.add("oi_change")
    if include_mark:
        required.update({"basis", "basis_mom", "basis_abs"})
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"cache {cache_tag} is missing required feature columns: {missing}")
    try:
        feature_values = features_df.to_numpy(dtype=np.float64)
        return_values = raw_returns.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cache {cache_tag} contains non-numeric values") from exc
    if not np.isfinite(feature_values).all() or not np.isfinite(return_values).all():
        raise ValueError(f"cache {cache_tag} contains NaN or infinite values")


def _write_cache_metadata(
    *,
    cache_dir: str,
    cache_tag: str,
    parameters: dict[str, object],
    features_df: pd.DataFrame,
    provenance: str,
) -> None:
    metadata = {
        "schema_version": CACHE_CONTRACT_VERSION,
        "cache_tag": cache_tag,
        "parameters": parameters,
        "feature_columns": [str(column) for column in features_df.columns],
        "rows": int(len(features_df)),
        "first_timestamp": str(features_df.index[0]),
        "last_timestamp": str(features_df.index[-1]),
        "provenance": provenance,
    }
    atomic_text_write(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        _cache_metadata_path(cache_dir, cache_tag),
    )


def _atomic_parquet_write(frame: pd.DataFrame, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        frame.to_parquet(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_costs(cfg: dict, cost_profile: str | None = None) -> tuple[dict, str]:
    resolved_cfg = dict(cfg)
    profile_name = cost_profile or cfg.get("cost_profile") or "default"
    profiles = cfg.get("cost_profiles")

    if profiles:
        if profile_name == "default":
            profile_name = "base" if "base" in profiles else next(iter(profiles))
        if profile_name not in profiles:
            available = ", ".join(profiles.keys())
            raise KeyError(f"Unknown cost profile '{profile_name}'. Available: {available}")
        resolved_cfg["costs"] = dict(profiles[profile_name])
        resolved_cfg["cost_profile"] = profile_name
    else:
        resolved_cfg["costs"] = dict(cfg.get("costs", {}))
        resolved_cfg["cost_profile"] = profile_name

    return resolved_cfg, resolved_cfg["cost_profile"]


def load_training_features(
    *,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    zscore_window: int,
    cache_dir: str,
    cache_tag: str,
    extra_series_mode: str = "derived",
    extra_series_include: list[str] | None = None,
    include_funding: bool = True,
    include_oi: bool = True,
    include_mark: bool = True,
    require_v4_cache: bool = False,
    return_availability: bool = False,
):
    """Load training frames and, for v4, propagate eligibility metadata.

    The default two-value return preserves the public v3 API.  New callers can
    request ``return_availability=True`` for an explicit third sidecar value;
    both forms carry the same validated sidecar through ``DataFrame.attrs`` so
    existing training stage boundaries remain compatible.
    """
    features_cache, returns_cache = resolve_cache_pair(cache_dir, cache_tag)
    v4_paths = cache_v4_paths(cache_dir, cache_tag)
    metadata_path = _cache_metadata_path(cache_dir, cache_tag)
    v4_sidecar_exists = v4_paths["availability"].exists()
    metadata_version: int | None = None
    if os.path.exists(metadata_path):
        try:
            metadata_payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            if isinstance(metadata_payload, dict):
                metadata_version = metadata_payload.get("schema_version")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if require_v4_cache or v4_sidecar_exists:
                raise ValueError(f"cache metadata cannot be read for v4 validation: {exc}") from exc
    v4_declared = metadata_version == 4
    if (require_v4_cache or v4_declared or v4_sidecar_exists) and not all(
        path.exists() for path in v4_paths.values()
    ):
        missing = [str(path) for path in v4_paths.values() if not path.exists()]
        reason = "required" if require_v4_cache else "declared or partially present"
        raise ValueError(f"v4 cache is {reason} but incomplete; missing files: {missing}")
    parameters = _cache_parameters(
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
        zscore_window=zscore_window,
        extra_series_mode=extra_series_mode,
        extra_series_include=extra_series_include,
        include_funding=include_funding,
        include_oi=include_oi,
        include_mark=include_mark,
    )
    ohlcv_cache = os.path.join(cache_dir, f"{cache_tag}_ohlcv.parquet")
    funding_cache = os.path.join(cache_dir, f"{cache_tag}_funding.parquet")
    oi_cache = os.path.join(cache_dir, f"{cache_tag}_oi.parquet")
    mark_cache = os.path.join(cache_dir, f"{cache_tag}_mark.parquet")

    if os.path.exists(features_cache) and os.path.exists(returns_cache):
        print("\n[Data] Loading cached features...")
        if v4_sidecar_exists or v4_declared or require_v4_cache:
            try:
                features_df, raw_returns, availability, v4_metadata = load_cache_v4(
                    cache_dir,
                    cache_tag,
                )
                _validate_training_cache(
                    features_df,
                    raw_returns,
                    include_funding=include_funding,
                    include_oi=include_oi,
                    include_mark=include_mark,
                    cache_tag=cache_tag,
                )
                cached_parameters = v4_metadata.get("parameters")
                if cached_parameters is not None and cached_parameters != parameters:
                    raise ValueError("cache v4 parameters do not match the requested config")
                _attach_availability_metadata(
                    features_df,
                    raw_returns,
                    availability,
                    include_funding=include_funding,
                    include_mark=include_mark,
                    interval=interval,
                )
            except (CacheV4Error, ValueError) as exc:
                raise ValueError(f"cache {cache_tag} failed v4 validation: {exc}") from exc
            print(
                f"  Cached: {features_df.shape} | obs_dim={features_df.shape[1]} "
                "| quality_status=v4_verified"
            )
            return _return_training_frames(
                features_df,
                raw_returns,
                availability,
                return_availability=return_availability,
            )
        try:
            features_df = read_optional_parquet(features_cache)
            returns_frame = read_optional_parquet(returns_cache)
            if features_df is None or returns_frame is None:
                raise ValueError("cache pair disappeared while loading")
            raw_returns = returns_frame.squeeze("columns")
            if isinstance(raw_returns, pd.DataFrame):
                raise ValueError("returns must contain exactly one column")
            _validate_training_cache(
                features_df,
                raw_returns,
                include_funding=include_funding,
                include_oi=include_oi,
                include_mark=include_mark,
                cache_tag=cache_tag,
            )
            if os.path.exists(metadata_path):
                metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("cache metadata must be a mapping")
                if metadata.get("schema_version") != CACHE_CONTRACT_VERSION:
                    raise ValueError(f"unsupported metadata schema: {metadata.get('schema_version')!r}")
                if metadata.get("cache_tag") != cache_tag:
                    raise ValueError("cache tag does not match metadata")
                if metadata.get("parameters") != parameters:
                    raise ValueError("cache parameters do not match the requested config")
                if metadata.get("feature_columns") != [str(column) for column in features_df.columns]:
                    raise ValueError("cache feature columns do not match metadata")
                if metadata.get("rows") != len(features_df):
                    raise ValueError("cache row count does not match metadata")
                if metadata.get("first_timestamp") != str(features_df.index[0]):
                    raise ValueError("cache first timestamp does not match metadata")
                if metadata.get("last_timestamp") != str(features_df.index[-1]):
                    raise ValueError("cache last timestamp does not match metadata")
            else:
                _write_cache_metadata(
                    cache_dir=cache_dir,
                    cache_tag=cache_tag,
                    parameters=parameters,
                    features_df=features_df,
                    provenance="legacy_unverified",
                )
            print(
                f"  Cached: {features_df.shape} | obs_dim={features_df.shape[1]} "
                "| quality_status=legacy_v3_unverified"
            )
            return _return_training_frames(
                features_df,
                raw_returns,
                None,
                return_availability=return_availability,
            )
        except Exception as exc:
            print(f"  Cache invalid; rebuilding from raw data: {exc}")

    # A v4 request may never silently fall back to the legacy raw downloader:
    # that path has no point-in-time sidecar and would make zero/missing values
    # indistinguishable.  The official v4 rebuild CLI must create all four
    # artifacts before this runtime can consume them.
    if require_v4_cache or v4_declared or v4_sidecar_exists:
        raise ValueError(
            "v4 cache is required for training but no complete validated cache exists; "
            "run the official v4 rebuild to create the feature, returns, availability, "
            "and metadata artifacts"
        )

    df = read_optional_parquet(ohlcv_cache)
    if df is not None:
        print(f"\n[Data] Spot OHLCV cache loaded: {len(df)} bars")
    else:
        print("\n[Data] Fetching OHLCV...")
        df = fetch_binance_ohlcv(symbol, interval, start, end)
        print(f"  Raw data: {len(df)} bars ({df.index[0]} -> {df.index[-1]})")

    funding_df = read_optional_parquet(funding_cache)
    oi_df = read_optional_parquet(oi_cache)
    mark_price_df = read_optional_parquet(mark_cache)
    extra_series = read_extra_series_caches(cache_dir, cache_tag)
    if not include_funding:
        funding_df = None
    elif funding_df is not None:
        print(f"[Data] Funding cache loaded: {len(funding_df)} records")
    else:
        try:
            print("[Data] Fetching funding rate...")
            funding_df = fetch_funding_rate(symbol, start, end)
            print(f"  Funding rate: {len(funding_df)} records")
        except Exception as exc:
            raise RuntimeError(
                "funding rate is required by this training config but could not be fetched"
            ) from exc
    if not include_oi:
        oi_df = None
    elif oi_df is not None:
        print(f"[Data] OI cache loaded: {len(oi_df)} records")
    else:
        try:
            print("[Data] Fetching open interest...")
            oi_df = fetch_open_interest_hist(symbol, interval, start, end)
            print(f"  Open interest: {len(oi_df)} records")
        except Exception as exc:
            raise RuntimeError(
                "open interest is required by this training config but could not be fetched"
            ) from exc
    if not include_mark:
        mark_price_df = None
    elif mark_price_df is not None:
        print(f"[Data] Mark cache loaded: {len(mark_price_df)} records")
    else:
        try:
            print("[Data] Fetching futures mark price...")
            mark_price_df = fetch_mark_price_klines(symbol, interval, start, end)
            print(f"  Mark price: {len(mark_price_df)} records")
        except Exception as exc:
            raise RuntimeError(
                "mark price is required by this training config but could not be fetched"
            ) from exc
    if extra_series_include:
        include_set = set(extra_series_include)
        extra_series = {k: v for k, v in extra_series.items() if k in include_set}

    print("[Data] Computing features...")
    if extra_series_mode == "raw_only":
        features_df = compute_features(
            df,
            zscore_window_days=zscore_window,
            interval=interval,
            funding_df=funding_df,
            oi_df=oi_df,
            mark_price_df=mark_price_df,
            extra_series=None,
        )
        extra_parts = align_extra_series(extra_series, df.index)
        if extra_parts:
            features_df = pd.concat([features_df, *extra_parts], axis=1).dropna()
    else:
        features_df = compute_features(
            df,
            zscore_window_days=zscore_window,
            interval=interval,
            funding_df=funding_df,
            oi_df=oi_df,
            mark_price_df=mark_price_df,
            extra_series=extra_series,
        )
    raw_returns = get_raw_returns(df)
    common_idx = features_df.index.intersection(raw_returns.index)
    features_df = features_df.loc[common_idx]
    raw_returns = raw_returns.loc[common_idx]
    _validate_training_cache(
        features_df,
        raw_returns,
        include_funding=include_funding,
        include_oi=include_oi,
        include_mark=include_mark,
        cache_tag=cache_tag,
    )
    os.makedirs(cache_dir, exist_ok=True)
    _atomic_parquet_write(features_df, features_cache)
    _atomic_parquet_write(raw_returns.to_frame(name="returns"), returns_cache)
    _write_cache_metadata(
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        parameters=parameters,
        features_df=features_df,
        provenance="generated",
    )
    print(f"  Features: {features_df.shape} | obs_dim={features_df.shape[1]}")
    print(f"  Saved cache: {features_cache}")
    return _return_training_frames(
        features_df,
        raw_returns,
        None,
        return_availability=return_availability,
    )
