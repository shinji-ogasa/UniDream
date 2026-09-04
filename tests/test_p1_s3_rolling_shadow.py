"""Unit checks for fixed-window rolling diagnostics and offline shadow rules."""
from __future__ import annotations

import unittest

import numpy as np

from unidream.experiments.p1_s3_rolling_shadow import (
    ROLLING_WINDOWS,
    RollingShadowError,
    _forecast_metrics,
    _window_bar_available,
)


class _Dataset:
    def __init__(self, n: int) -> None:
        self.returns = np.zeros(n, dtype=np.float64)
        self.availability = {
            "spot_bar_observed": np.ones(n, dtype=np.bool_),
        }


class P1S3RollingShadowTest(unittest.TestCase):
    def test_declared_windows_are_ordered_and_grid_aligned(self) -> None:
        previous = None
        for start, end in ROLLING_WINDOWS:
            self.assertEqual(start % 4, 0)
            if end != 173_111:
                self.assertEqual(end % 4, 0)
            if previous is not None:
                self.assertGreaterEqual(start, previous)
            previous = end

    def test_window_availability_is_isolated_from_other_periods(self) -> None:
        dataset = _Dataset(40)
        dataset.availability["spot_bar_observed"][13] = False
        mask = _window_bar_available(dataset, 8, 24)
        self.assertFalse(mask[:8].any())
        self.assertFalse(mask[24:].any())
        self.assertFalse(mask[13])
        self.assertTrue(mask[8])
        self.assertTrue(mask[23])

    def test_invalid_window_fails_closed(self) -> None:
        dataset = _Dataset(40)
        with self.assertRaises(RollingShadowError):
            _window_bar_available(dataset, 24, 8)

    def test_forecast_metrics_drop_nonfinite_rows(self) -> None:
        predictions = np.array([1.0, np.nan, 3.0, 100.0], dtype=np.float64)
        target = np.array([2.0, 4.0, 1.0, np.nan], dtype=np.float64)
        metrics = _forecast_metrics(predictions, target, np.ones(4, dtype=np.bool_))
        self.assertEqual(metrics["score_rows"], 2)
        self.assertAlmostEqual(metrics["mse"], 2.5)
        self.assertAlmostEqual(metrics["mae"], 1.5)


if __name__ == "__main__":
    unittest.main()
