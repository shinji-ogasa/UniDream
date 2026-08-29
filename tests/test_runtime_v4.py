"""Runtime cache-hit tests for schema v4 and explicit legacy status."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.data.cache_v4 import MODEL_FEATURE_COLUMNS, cache_v4_paths, write_cache_v4
from unidream.experiments.runtime import cache_quality_status, load_training_features


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


class RuntimeV4Test(unittest.TestCase):
    def test_v4_cache_hit_is_validated_and_reports_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _v4_fixture(root)
            self.assertEqual(cache_quality_status(temp_dir, "runtime-v4"), "v4_verified")
            loaded_features, loaded_returns = load_training_features(
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
            self.assertEqual(loaded_features.shape, (5, 17))
            self.assertEqual(len(loaded_returns), 5)

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
