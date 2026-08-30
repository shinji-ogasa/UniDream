"""Runtime cache-hit tests for schema v4 and explicit legacy status."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.data.cache_v4 import MODEL_FEATURE_COLUMNS, cache_v4_paths, write_cache_v4
from unidream.data.dataset import WFODataset, WFOSplit
from unidream.experiments.runtime import (
    V4RuntimeInputError,
    cache_quality_status,
    load_training_features,
    validate_p1_v4_runtime_inputs,
    validate_v4_runtime_inputs,
)
from unidream.experiments.train_app import resolve_training_cache_selection


def _v4_fixture(root: Path, tag: str = "runtime-v4") -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=5, freq="15min")
    features = pd.DataFrame(
        np.arange(len(index) * len(MODEL_FEATURE_COLUMNS), dtype=np.float64).reshape(
            len(index), len(MODEL_FEATURE_COLUMNS)
        ),
        index=index,
        columns=MODEL_FEATURE_COLUMNS,
    )
    returns = pd.Series(np.linspace(0.0, 0.01, len(index)), index=index, name="returns")
    availability = pd.DataFrame(
        {
            "spot_bar_observed": True,
            "funding_rate_available": [True, False, True, True, True],
            "mark_close_available": True,
        },
        index=index,
    )
    parameters = {
        "symbol": "BTCUSDT",
        "interval": "15m",
        "start": "2024-01-01",
        "end": "2024-01-02",
        "zscore_window_days": 60,
        "extra_series_mode": "derived",
        "extra_series_include": [],
        "include_funding": True,
        "include_oi": False,
        "include_mark": True,
    }
    metadata = write_cache_v4(
        features,
        returns,
        availability,
        cache_dir=root,
        cache_tag=tag,
        source_provenance={"source": "synthetic"},
        parameters=parameters,
        start="2024-01-01",
        end="2024-01-02",
    )
    return metadata, features, availability


def _runtime_manifest(root: Path, metadata: dict, tag: str = "runtime-v4") -> dict:
    """Build a minimal manifest-shaped runtime contract from the test fixture."""
    paths = cache_v4_paths(root, tag)
    frozen_sha = hashlib.sha256(paths["metadata"].read_bytes()).hexdigest()
    return {
        "common": {
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "v4_load_contract": {
                "loader": "unidream.data.cache_v4.load_cache_v4",
                "metadata_authority": "repo_frozen_metadata",
                "require_explicit_paths": True,
                "cache_dir_cache_tag_fallback": "forbidden",
                "cache_tag": tag,
                "feature_path": paths["features"].name,
                "returns_path": paths["returns"].name,
                "availability_path": paths["availability"].name,
                "metadata_path": paths["metadata"].name,
                "cache_local_metadata_path": "local_metadata.json",
                "runtime_validation_entrypoint": "unidream.experiments.runtime.validate_p1_v4_runtime_inputs",
                "runtime_body_validator_entrypoint": "unidream.experiments.runtime.validate_v4_runtime_inputs",
                "runtime_authentication_policy": "production P1 entrypoint must call load_fixed_manifest first, then delegate the authenticated frozen manifest to the body validator; caller mappings and forged manifest_sha256/results_observed/frozen digests are rejected before body validation",
                "runtime_validation_required_before_fit_or_score": True,
                "runtime_disposition_fields": ["status", "reason", "body_match", "source_provenance_match"],
                "runtime_disposition_statuses": ["absent", "identical", "source_provenance_only_difference"],
                "known_cache_local_snapshot": {
                    "source_provenance_digest": "",
                },
            },
        },
        "provenance": {
            "v4_parent": {
                "metadata_path": paths["metadata"].name,
                "metadata_sha256": frozen_sha,
                "cache_tag": tag,
                "schema_version": 4,
                "schema_digest": metadata["schema_digest"],
                "source_provenance_digest": metadata["source_provenance_digest"],
                "content_digests": dict(metadata["content_digests"]),
                "feature_rows": metadata["rows"],
                "sidecar_rows": metadata["sidecar_rows"],
                "required_availability_columns": list(metadata["availability_columns"]),
            }
        },
    }


class RuntimeV4Test(unittest.TestCase):
    def test_training_entrypoint_selects_v4_explicitly_and_requires_it(self) -> None:
        tag, require_v4 = resolve_training_cache_selection(
            symbol="BTCUSDT",
            interval="15m",
            start="2018-01-01",
            end="2024-01-01",
            zscore_window=60,
            data_cfg={"cache_schema": "v4"},
        )
        self.assertEqual(tag, "BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official")
        self.assertTrue(require_v4)
        with self.assertRaisesRegex(ValueError, "unsupported data.cache_schema"):
            resolve_training_cache_selection(
                symbol="BTCUSDT",
                interval="15m",
                start="2018-01-01",
                end="2024-01-01",
                zscore_window=60,
                data_cfg={"cache_schema": "v2"},
            )

    def test_v4_cache_hit_is_validated_and_reports_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _v4_fixture(root)
            self.assertEqual(cache_quality_status(temp_dir, "runtime-v4"), "v4_verified")
            features, returns, availability = load_training_features(
                symbol="BTCUSDT",
                interval="15m",
                start="2024-01-01",
                end="2024-01-02",
                zscore_window=60,
                cache_dir=temp_dir,
                cache_tag="runtime-v4",
                include_funding=True,
                include_oi=False,
                include_mark=True,
                return_availability=True,
            )
            self.assertEqual(features.shape[1], 17)
            pd.testing.assert_frame_equal(features.attrs["availability"], availability)
            split = WFOSplit(
                fold_idx=0,
                train_start=features.index[0],
                train_end=features.index[4],
                val_start=features.index[4],
                val_end=features.index[5 - 1] + pd.Timedelta(minutes=15),
                test_start=features.index[5 - 1] + pd.Timedelta(minutes=15),
                test_end=features.index[5 - 1] + pd.Timedelta(minutes=30),
            )
            dataset = WFODataset(features, returns, split, seq_len=2)
            self.assertEqual(dataset.obs_dim, 17)
            # Funding is unavailable at row 1.  The body keeps every row and
            # only the original-offset window [2, 3] remains eligible.
            np.testing.assert_array_equal(dataset.train_row_eligible, [True, False, True, True])
            np.testing.assert_array_equal(dataset.train_dataset().valid_starts, [2])

    def test_invalid_v4_hit_does_not_fall_back_to_raw_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _v4_fixture(root)
            paths = cache_v4_paths(root, "runtime-v4")
            availability = pd.read_parquet(paths["availability"])
            availability["funding_rate_available"] = 1
            availability.to_parquet(paths["availability"])
            with self.assertRaisesRegex(ValueError, "boolean dtype"):
                load_training_features(
                    symbol="BTCUSDT",
                    interval="15m",
                    start="2024-01-01",
                    end="2024-01-02",
                    zscore_window=60,
                    cache_dir=temp_dir,
                    cache_tag="runtime-v4",
                    include_funding=True,
                    include_oi=False,
                    include_mark=True,
                )

    def test_production_v4_runtime_validator_requires_explicit_complete_bodies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata, _, _ = _v4_fixture(root)
            manifest = _runtime_manifest(root, metadata)
            paths = cache_v4_paths(root, "runtime-v4")
            result = validate_v4_runtime_inputs(
                manifest,
                root=root,
                path_overrides={
                    "feature_path": paths["features"],
                    "returns_path": paths["returns"],
                    "availability_path": paths["availability"],
                    "metadata_path": paths["metadata"],
                },
            )
            self.assertEqual(result["v4_runtime_validation_status"], "passed")
            self.assertEqual(result["v4_runtime_provenance_disposition"]["status"], "absent")
            self.assertIsNone(result["v4_runtime_body_match"])
            self.assertTrue(result["v4_runtime_loaded_body_match"])
            self.assertIsNone(result["v4_runtime_source_provenance_match"])
            self.assertEqual(result["features"].shape[1], 17)

            missing = dict(paths)
            missing["features"] = root / "missing_features.parquet"
            with self.assertRaisesRegex(V4RuntimeInputError, "missing files"):
                validate_v4_runtime_inputs(
                    manifest,
                    root=root,
                    path_overrides={
                        "feature_path": missing["features"],
                        "returns_path": paths["returns"],
                        "availability_path": paths["availability"],
                        "metadata_path": paths["metadata"],
                    },
                )

    def test_production_v4_runtime_validator_surfaces_known_source_only_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata, _, _ = _v4_fixture(root)
            manifest = _runtime_manifest(root, metadata)
            paths = cache_v4_paths(root, "runtime-v4")
            local_payload = copy.deepcopy(metadata)
            local_payload["source_provenance"] = {"source": "known-local-revision"}
            local_payload["source_provenance_digest"] = hashlib.sha256(
                json.dumps(
                    local_payload["source_provenance"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            local_path = root / "local_metadata.json"
            local_path.write_text(
                json.dumps(local_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest["common"]["v4_load_contract"]["known_cache_local_snapshot"] = {
                "source_provenance_digest": local_payload["source_provenance_digest"],
            }
            result = validate_v4_runtime_inputs(
                manifest,
                root=root,
                cache_local_metadata_path=local_path,
                provenance_disposition={
                    "status": "source_provenance_only_difference",
                    "reason": "known local source revision",
                    "body_match": True,
                    "source_provenance_match": False,
                },
            )
            self.assertEqual(
                result["v4_runtime_provenance_disposition"]["status"],
                "source_provenance_only_difference",
            )
            self.assertFalse(result["v4_runtime_source_provenance_match"])
            self.assertEqual(
                result["v4_runtime_cache_local_source_provenance_digest"],
                local_payload["source_provenance_digest"],
            )

            unknown = copy.deepcopy(manifest)
            unknown["common"]["v4_load_contract"]["known_cache_local_snapshot"] = {
                "source_provenance_digest": "unknown"
            }
            with self.assertRaisesRegex(V4RuntimeInputError, "unknown digest"):
                validate_v4_runtime_inputs(
                    unknown,
                    root=root,
                    cache_local_metadata_path=local_path,
                )

    def test_authenticated_p1_wrapper_loads_fixed_manifest_before_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata, _, _ = _v4_fixture(root)
            fixed = _runtime_manifest(root, metadata)
            fixed.update(
                {
                    "manifest_id": "fixture-p1",
                    "manifest_sha256": "f" * 64,
                    "base_revision": "b" * 40,
                    "results_observed": False,
                }
            )
            paths = cache_v4_paths(root, "runtime-v4")
            explicit = {
                "feature_path": paths["features"],
                "returns_path": paths["returns"],
                "availability_path": paths["availability"],
                "metadata_path": paths["metadata"],
            }
            with mock.patch(
                "unidream.experiments.p1_recovery_prereg.load_fixed_manifest",
                return_value=fixed,
            ) as loader:
                result = validate_p1_v4_runtime_inputs(
                    fixed,
                    root=root,
                    path_overrides=explicit,
                )
                loader.assert_called_once()
            self.assertEqual(result["p1_manifest_sha256"], "f" * 64)
            self.assertFalse(result["p1_results_observed"])
            self.assertEqual(
                result["p1_runtime_body_validator_entrypoint"],
                "unidream.experiments.runtime.validate_v4_runtime_inputs",
            )

            forged = copy.deepcopy(fixed)
            forged["results_observed"] = True
            with mock.patch(
                "unidream.experiments.p1_recovery_prereg.load_fixed_manifest",
                return_value=fixed,
            ):
                with self.assertRaisesRegex(V4RuntimeInputError, "results_observed"):
                    validate_p1_v4_runtime_inputs(
                        forged,
                        root=root,
                        path_overrides=explicit,
                    )

            forged = copy.deepcopy(fixed)
            forged["common"]["v4_load_contract"]["frozen_schema_digest"] = "0" * 64
            with mock.patch(
                "unidream.experiments.p1_recovery_prereg.load_fixed_manifest",
                return_value=fixed,
            ):
                with self.assertRaisesRegex(V4RuntimeInputError, "differs"):
                    validate_p1_v4_runtime_inputs(
                        forged,
                        root=root,
                        path_overrides=explicit,
                    )

    def test_legacy_v3_cache_is_explicitly_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index = pd.date_range("2024-01-01", periods=2, freq="15min")
            features = pd.DataFrame(
                np.ones((2, len(MODEL_FEATURE_COLUMNS))),
                index=index,
                columns=MODEL_FEATURE_COLUMNS,
            )
            features.to_parquet(root / "legacy_features.parquet")
            pd.Series([0.0, 0.1], index=index, name="returns").to_frame().to_parquet(
                root / "legacy_returns.parquet"
            )
            self.assertEqual(cache_quality_status(temp_dir, "legacy"), "legacy_v3_unverified")
            with self.assertRaisesRegex(ValueError, "v4 cache is required"):
                load_training_features(
                    symbol="BTCUSDT",
                    interval="15m",
                    start="2024-01-01",
                    end="2024-01-02",
                    zscore_window=60,
                    cache_dir=temp_dir,
                    cache_tag="legacy",
                    include_funding=True,
                    include_oi=False,
                    include_mark=True,
                    require_v4_cache=True,
                )


if __name__ == "__main__":
    unittest.main()
