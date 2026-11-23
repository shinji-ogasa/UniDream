"""Chronological support and label-free inference for auxiliary forecasts."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments import wm_classification_forecasters as f


class TestWMClassificationForecasters(TestCase):
    def test_early_bounds_cap_train_and_leave_three_month_validation(self):
        bounds = f.forecaster_bounds("2021-07-01")
        self.assertEqual(bounds["train_start"], pd.Timestamp("2020-09-01", tz="UTC"))
        self.assertEqual(bounds["train_end"], pd.Timestamp("2021-04-01", tz="UTC"))
        self.assertEqual(bounds["val_end"], pd.Timestamp("2021-07-01", tz="UTC"))
        later = f.forecaster_bounds("2026-07-01")
        self.assertEqual(later["train_start"], pd.Timestamp("2024-04-01", tz="UTC"))
        with self.assertRaises(ValueError):
            f.forecaster_bounds("2021-08-01")

    def test_training_history_and_all_labels_stay_in_declared_period(self):
        index = pd.date_range("2020-09-01", "2021-07-03", freq="15min", tz="UTC")
        raw, market = np.ones((len(index), 5)), np.ones((len(index), 14))
        targets = np.ones((len(index), 3), dtype=bool)
        bounds = f.forecaster_bounds("2021-07-01")
        data = f.causal_forecaster_cohorts(index.asi8, raw, market, targets, bounds)
        for split in ("train", "val"):
            ids = data[split]
            self.assertTrue(len(ids) > 1000)
            self.assertTrue(np.all(index.asi8[ids] >= bounds[split + "_start"].value))
            self.assertTrue(np.all(index.asi8[ids] + 64 * f.BAR_NS < bounds[split + "_end"].value))
        self.assertTrue(np.all(index.asi8[data["train"] - 256] >= bounds["train_start"].value))
        # Altering future inputs/target masks cannot change Train or Val.
        future = index >= bounds["quarter_start"]
        raw[future], targets[future] = np.nan, False
        changed = f.causal_forecaster_cohorts(index.asi8, raw, market, targets, bounds)
        for split in ("train", "val"):
            np.testing.assert_array_equal(data[split], changed[split])
        gap = int(index.get_indexer([pd.Timestamp("2021-02-01", tz="UTC")])[0])
        raw[gap] = np.nan
        changed = f.causal_forecaster_cohorts(index.asi8, raw, market, targets, bounds)
        self.assertFalse(np.isin(np.arange(gap + 1, gap + 257), changed["train"]).any())

    def test_first_forecast_can_be_at_quarter_start_after_purged_selection(self):
        start = pd.Timestamp("2023-01-01", tz="UTC").value
        f.assert_oof_cutoff([start, start + f.BAR_NS], start - f.BAR_NS)
        with self.assertRaises(ValueError):
            f.assert_oof_cutoff([start], start)

    def test_inference_loader_never_opens_future_label_members(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = Path(directory) / "prepared"
            prepared.mkdir()
            # Accessing this object label with allow_pickle=False would fail.
            np.savez(prepared / "arrays.npz", raw=np.ones((3, 5)), market=np.ones((3, 14)),
                     index_ns=np.arange(3), returns=np.array([object()], dtype=object))
            f.inference_arrays.cache_clear()
            result = f.inference_arrays(directory)
            self.assertEqual(set(result), {"raw", "market", "index_ns"})
            f.inference_arrays.cache_clear()

    def test_partial_cache_uses_current_train_scaler_and_computes_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            ids = np.array([10, 12, 16])
            def features(raw, market, selected, context):
                base = np.arange(context * 63, dtype=np.float32).reshape(context, 63) / 100
                return np.stack([base + decision for decision in selected]).astype(np.float32)
            lags = [0, 192, 240, 252]
            cached_ids = np.array([12, 16])
            np.save(cache / "decisions.npy", cached_ids)
            np.save(cache / "tabular_raw.npy", f.tabular_design(features(None, None, cached_ids, 256), lags))
            parent = {"model": {"context": 256}}
            comparison = {"output_dir": directory, "tabular": {"market_lag_positions": lags, "feature_batch": 2}}
            fold = {"mean": np.linspace(1, 2, 63, dtype=np.float32), "std": np.linspace(2, 3, 63, dtype=np.float32)}
            with patch.object(f, "_features", side_effect=features) as compute:
                got = f._tabular_rows(parent, comparison, {"raw": None, "market": None}, ids, fold)
                self.assertEqual(compute.call_count, 1)
                np.testing.assert_array_equal(compute.call_args.args[2], [10])
            mean, std = f.tabular_scaler(fold, 256, lags)
            expected = (f.tabular_design(features(None, None, ids, 256), lags) - mean) / std
            np.testing.assert_array_equal(got, expected)
            self.assertEqual(got.shape, (3, 1394))
