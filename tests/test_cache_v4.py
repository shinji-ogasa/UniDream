"""Regression tests for the fail-closed research cache schema v4."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.data.cache_v4 import (
    MODEL_FEATURE_COLUMNS,
    CacheV4Error,
    build_v4_metadata,
    load_cache_v4,
    validate_cache_v4,
    write_cache_v4,
)


def _frames() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    sidecar_index = pd.date_range("2018-01-01", periods=8, freq="15min")
    body_index = sidecar_index.delete(3)
    values = np.arange(len(body_index) * len(MODEL_FEATURE_COLUMNS), dtype=np.float64).reshape(
        len(body_index), len(MODEL_FEATURE_COLUMNS)
    )
    features = pd.DataFrame(values + 1.0, index=body_index, columns=MODEL_FEATURE_COLUMNS)
    # A real zero is retained in the model body; it is not used as a missing flag.
    features.loc[body_index[1], "funding_rate"] = 0.0
    returns = pd.Series(np.linspace(-0.01, 0.01, len(body_index)), index=body_index, name="returns")
    availability = pd.DataFrame(
        {
            "spot_bar_observed": [True, True, True, False, True, True, True, True],
            "funding_rate_available": [True, False, True, False, True, True, True, True],
            "mark_close_available": [True, True, True, False, False, True, True, True],
        },
        index=sidecar_index,
    )
    return features, returns, availability


class CacheV4Test(unittest.TestCase):
    def test_write_load_preserves_body_and_distinguishes_zero_from_missing(self) -> None:
        features, returns, availability = _frames()
        original = features.copy(deep=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata = write_cache_v4(
                features,
                returns,
                availability,
                cache_dir=temp_dir,
                cache_tag="synthetic-v4",
                source_provenance={"source": "synthetic", "digest": "abc"},
                symbol="BTCUSDT",
                start="2018-01-01",
                end="2018-01-01 02:00:00",
            )
            loaded_features, loaded_returns, loaded_availability, loaded_metadata = load_cache_v4(
                temp_dir, "synthetic-v4"
            )
        pd.testing.assert_frame_equal(loaded_features, original)
        pd.testing.assert_series_equal(loaded_returns, returns.rename("returns"))
        pd.testing.assert_frame_equal(loaded_availability, availability, check_freq=False)
        self.assertEqual(metadata, loaded_metadata)
        self.assertEqual(loaded_metadata["schema_version"], 4)
        self.assertEqual(len(loaded_metadata["gap_list"]), 1)
        self.assertEqual(loaded_metadata["gap_list"][0]["expected_missing_count"], 1)
        self.assertEqual(loaded_availability.loc[loaded_availability.index[3], "spot_bar_observed"], False)
        self.assertEqual(loaded_features.loc[loaded_features.index[1], "funding_rate"], 0.0)

    def test_missing_availability_column_is_named_and_rejected(self) -> None:
        features, returns, availability = _frames()
        metadata = build_v4_metadata(
            features,
            returns,
            availability,
            source_provenance={"source": "synthetic"},
        )
        broken = availability.drop(columns=["funding_rate_available"])
        with self.assertRaisesRegex(CacheV4Error, "funding_rate_available"):
            validate_cache_v4(features, returns, broken, metadata)

    def test_nonfinite_body_and_nonboolean_sidecar_fail_closed(self) -> None:
        features, returns, availability = _frames()
        metadata = build_v4_metadata(
            features,
            returns,
            availability,
            source_provenance={"source": "synthetic"},
        )
        nonfinite = features.copy()
        nonfinite.iloc[0, 0] = np.nan
        with self.assertRaisesRegex(CacheV4Error, "NaN or infinite"):
            validate_cache_v4(nonfinite, returns, availability, metadata)

        nonboolean = availability.copy()
        nonboolean["funding_rate_available"] = nonboolean["funding_rate_available"].astype(np.int8)
        with self.assertRaisesRegex(CacheV4Error, "funding_rate_available.*boolean"):
            validate_cache_v4(features, returns, nonboolean, metadata)

    def test_sidecar_duplicate_or_noncontiguous_index_is_not_repaired(self) -> None:
        features, returns, availability = _frames()
        metadata = build_v4_metadata(
            features,
            returns,
            availability,
            source_provenance={"source": "synthetic"},
        )
        duplicate = pd.concat([availability.iloc[:2], availability.iloc[1:]])
        with self.assertRaisesRegex(CacheV4Error, "duplicate timestamps"):
            validate_cache_v4(features, returns, duplicate, metadata)

        noncontiguous = availability.drop(index=availability.index[4])
        with self.assertRaisesRegex(CacheV4Error, "non-contiguous interval"):
            validate_cache_v4(features, returns, noncontiguous, metadata)

    def test_metadata_order_gap_and_provenance_digests_are_contracts(self) -> None:
        features, returns, availability = _frames()
        metadata = build_v4_metadata(
            features,
            returns,
            availability,
            source_provenance={"source": "synthetic"},
        )
        reordered = features[list(reversed(features.columns))]
        with self.assertRaisesRegex(CacheV4Error, "column order/schema mismatch"):
            validate_cache_v4(reordered, returns, availability, metadata)

        broken_gap = dict(metadata)
        broken_gap["gap_list"] = []
        with self.assertRaisesRegex(CacheV4Error, "gap_list.*missing"):
            validate_cache_v4(features, returns, availability, broken_gap)

        broken_provenance = dict(metadata)
        broken_provenance["source_provenance_digest"] = "0" * 64
        with self.assertRaisesRegex(CacheV4Error, "provenance digest mismatch"):
            validate_cache_v4(features, returns, availability, broken_provenance)

    def test_same_shape_body_or_sidecar_tampering_is_detected_by_content_digest(self) -> None:
        features, returns, availability = _frames()
        metadata = build_v4_metadata(
            features,
            returns,
            availability,
            source_provenance={"source": "synthetic"},
        )
        changed_features = features.copy()
        changed_features.iloc[0, 0] += 1.0
        with self.assertRaisesRegex(CacheV4Error, "features content digest mismatch"):
            validate_cache_v4(changed_features, returns, availability, metadata)

        changed_availability = availability.copy()
        changed_availability.iloc[1, 1] = not changed_availability.iloc[1, 1]
        with self.assertRaisesRegex(CacheV4Error, "availability content digest mismatch"):
            validate_cache_v4(features, returns, changed_availability, metadata)

    def test_v3_or_incomplete_cache_never_passes_as_v4(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "legacy_features.parquet").write_text("not parquet", encoding="utf-8")
            with self.assertRaisesRegex(CacheV4Error, "missing files.*availability"):
                load_cache_v4(temp_dir, "legacy")


if __name__ == "__main__":
    unittest.main()
