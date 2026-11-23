"""Exact sufficient-statistic bootstrap, supported blocks and no IID fallback."""
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_joint_grid_statistics import paired_native_day_inference


def panel(task="return", days=40):
    records = []
    for day in range(days):
        for formulation, multiplier in (("regression", 4.), ("joint", 1.)):
            records.append({"split": "test", "group": "A", "model": "wm", "task": task,
                "formulation": formulation, "fold": 0, "seed": 7,
                "utc_day": str((pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)).date()),
                "segment_id": 0, "n_rows": 2,
                "sum_squared_error": multiplier * (1 + day % 3),
                "sum_absolute_error": multiplier, "sum_qlike": multiplier,
                "sum_actual_squared": 20., "sum_brier": multiplier / 10})
    return pd.DataFrame(records)


class JointGridStatisticsTests(unittest.TestCase):
    def test_r2_uses_pooled_zero_prediction_denominator(self):
        frame = panel()
        rows = paired_native_day_inference(frame, block_days=[7], replicates=100)
        row = rows[0]
        base = frame.loc[frame.formulation == "regression"]
        joint = frame.loc[frame.formulation == "joint"]
        expected = (base.sum_squared_error.sum() - joint.sum_squared_error.sum()) / base.sum_actual_squared.sum()
        self.assertAlmostEqual(row["estimate"], expected)
        self.assertTrue(row["inference_available"])
        self.assertGreater(row["estimate"], 0)

    def test_rmse_is_recomputed_after_bootstrap_not_mean_of_daily_rmse(self):
        frame = panel("advantage")
        row = paired_native_day_inference(frame, block_days=[7], replicates=100)[0]
        base = frame.loc[frame.formulation == "regression"]
        joint = frame.loc[frame.formulation == "joint"]
        expected = np.sqrt(joint.sum_squared_error.sum() / joint.n_rows.sum()) - np.sqrt(base.sum_squared_error.sum() / base.n_rows.sum())
        self.assertAlmostEqual(row["estimate"], expected)
        self.assertEqual(row["inference_metric"], "rmse")
        self.assertEqual(row["report_primary_metric"], "spearman_ic")
        self.assertGreater(row["improvement"], 0)

    def test_unsupported_segment_remains_in_estimate_without_shorter_blocks(self):
        frame = panel(days=3)
        row = paired_native_day_inference(frame, block_days=[7], replicates=100)[0]
        self.assertFalse(row["inference_available"])
        self.assertIsNone(row["ci_low"])
        self.assertEqual(row["n_nonresampleable_rows"], 6)
        self.assertEqual(row["resampleable_row_fraction"], 0)
        self.assertIsNotNone(row["estimate"])

    def test_day_gap_must_start_new_contiguous_segment(self):
        frame = panel(days=3)
        frame = frame.loc[frame.utc_day != "2024-01-02"]
        with self.assertRaisesRegex(ValueError, "bridges missing UTC days"):
            paired_native_day_inference(frame, block_days=[1], replicates=10)

    def test_missing_pair_and_different_outcome_denominator_are_rejected(self):
        frame = panel(days=3)
        with self.assertRaisesRegex(ValueError, "coverage differ"):
            paired_native_day_inference(frame.iloc[:-1], block_days=[1], replicates=10)
        frame.loc[frame.formulation == "joint", "sum_actual_squared"] += .1
        with self.assertRaisesRegex(ValueError, "outcomes/support differ"):
            paired_native_day_inference(frame, block_days=[1], replicates=10)

    def test_resampling_is_reproducible_and_fold_segments_do_not_bridge(self):
        frame = panel(days=40)
        frame.loc[frame.utc_day >= "2024-01-21", "fold"] = 1
        first = paired_native_day_inference(frame, block_days=[7], replicates=100, random_seed=3)
        second = paired_native_day_inference(frame, block_days=[7], replicates=100, random_seed=3)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["n_segments"], 2)


if __name__ == "__main__":
    unittest.main()
