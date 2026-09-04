"""Unit checks for the terminal, report-only S3 outer evaluator."""
from __future__ import annotations

import unittest

import numpy as np

from unidream.experiments.p1_s3_outer_report import (
    _forecast_metrics,
    _manifest_outer_contract,
    _max_drawdown,
)
from unidream.experiments.p1_recovery_runner import (
    S3_OUTER_END,
    S3_TRAIN_START,
    S3_VALIDATION_END,
)


class P1S3OuterReportTest(unittest.TestCase):
    def test_manifest_binds_fixed_s3_outer_ranges(self) -> None:
        manifest, scenario = _manifest_outer_contract()
        self.assertFalse(manifest["results_observed"])
        self.assertEqual(scenario["outer_report_origin_raw_index"], S3_VALIDATION_END)
        self.assertEqual(
            tuple(scenario["outer_report_fit_raw_range"]),
            (S3_TRAIN_START, S3_VALIDATION_END),
        )
        self.assertEqual(
            tuple(scenario["outer_report_prediction_raw_range"]),
            (S3_VALIDATION_END, S3_OUTER_END),
        )
        self.assertEqual(tuple(scenario["outer_report_refit_origins"]), ())
        self.assertTrue(scenario["outer_test_is_report_only"])

    def test_forecast_metrics_use_only_finite_score_rows(self) -> None:
        predictions = np.array([1.0, np.nan, 3.0, 100.0], dtype=np.float64)
        target = np.array([2.0, 4.0, 1.0, np.nan], dtype=np.float64)
        mask = np.array([True, True, True, True], dtype=np.bool_)
        metrics = _forecast_metrics(predictions, target, mask)
        self.assertEqual(metrics["score_rows"], 2)
        self.assertAlmostEqual(metrics["mse"], 2.5)
        self.assertAlmostEqual(metrics["mae"], 1.5)

    def test_max_drawdown_is_zero_init(self) -> None:
        self.assertAlmostEqual(_max_drawdown(np.array([1.0, -2.0, 0.5])), 2.0)
        self.assertEqual(_max_drawdown(np.array([], dtype=np.float64)), 0.0)


if __name__ == "__main__":
    unittest.main()
