import json
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_ablation_evaluation import make_walk_forward_folds
from unidream.experiments.wm_seven_task_evaluation import (
    TASKS,
    build_seven_targets,
    fit_target_scales,
    score_seven,
    seven_split_mask,
)


class SevenTaskTargetTest(unittest.TestCase):
    def test_known_future_path_and_synthetic_formulas(self):
        index = pd.date_range("2023-01-01", periods=6, freq="15min", tz="UTC")
        close = np.array([100.0, 110.0, 98.0, 102.0, 105.0, 99.0])
        result = build_seven_targets(index, close, horizons=(2,))
        ret = np.log(98 / 100)
        variance = np.log(110 / 100) ** 2 + np.log(98 / 110) ** 2
        self.assertAlmostEqual(result["return"][1, 0], ret)
        self.assertAlmostEqual(result["variance"][1, 0], variance)
        # Reference-to-minimum downside is 2%; an intra-window peak-to-trough
        # drawdown would be 1-98/110 and is deliberately a different label.
        self.assertAlmostEqual(result["downside"][1, 0], 0.02)
        self.assertEqual(result["crash"][1, 0], 1.0)
        self.assertAlmostEqual(result["dd_excess"][1, 0], 0.008)
        self.assertAlmostEqual(result["advantage"][1, 0], 0.06 * (ret - 0.35 * 0.02 - 0.00055))
        self.assertAlmostEqual(result["recovery"][1, 0], (ret - 0.5 * 0.02) / (np.sqrt(variance) + 1e-6) / 5)
        self.assertEqual(result["label_end_ns"][1, 0], pd.Timestamp("2023-01-01T00:45:00Z").value)
        changed = close.copy()
        changed[3:] = 10000.0
        other = build_seven_targets(index, changed, horizons=(2,))
        for task in TASKS:
            self.assertEqual(result[task][1, 0], other[task][1, 0])

    def test_all_horizons_and_strict_end_purge(self):
        index = pd.date_range("2022-12-30", "2023-01-02", freq="15min", tz="UTC")
        targets = build_seven_targets(index, np.ones(len(index)), horizons=(4, 16, 48, 96))
        fold = make_walk_forward_folds()[0]
        selected = seven_split_mask(index, targets, fold, "val")
        exact_boundary = index.get_loc("2022-12-31 00:00:00+00:00")
        self.assertFalse(selected[exact_boundary])
        self.assertTrue(selected[exact_boundary - 1])
        shorter = seven_split_mask(index, targets, fold, "val", horizon=4)
        self.assertTrue(shorter[exact_boundary])
        with self.assertRaisesRegex(ValueError, "grids must match"):
            seven_split_mask(index + pd.Timedelta(minutes=15), targets, fold, "val")
        context = np.ones(len(index), bool)
        context[exact_boundary - 1] = False
        self.assertFalse(seven_split_mask(index, targets, fold, "val", context_valid=context)[exact_boundary - 1])

    def test_zero_prices_missing_paths_and_causal_baseline(self):
        index = pd.date_range("2023-01-01", periods=60, freq="15min", tz="UTC")
        close = np.exp(np.arange(len(index)) * 0.01)
        targets = build_seven_targets(index, close, horizons=(4, 16))
        np.testing.assert_allclose(targets["baseline_variance"][17], np.array([4, 16]) * 0.01 ** 2)
        self.assertTrue(np.isnan(targets["baseline_variance"][16]).all())
        modified = close.copy()
        modified[17:] *= 10
        other = build_seven_targets(index, modified, horizons=(4, 16))
        np.testing.assert_array_equal(targets["baseline_variance"][17], other["baseline_variance"][17])
        modified = close.copy()
        modified[5] = 0
        invalid = build_seven_targets(index, modified, horizons=(4, 16))
        self.assertFalse(invalid["mask"][2, 0])
        for task in TASKS:
            self.assertTrue(np.isnan(invalid[task][2, 0]))
        observed = np.ones(len(index), bool)
        observed[5] = False
        missing = build_seven_targets(index, close, observed, horizons=(4, 16))
        np.testing.assert_array_equal(invalid["mask"], missing["mask"])

    def test_constant_market_has_zero_risk_and_fixed_cost_only(self):
        index = pd.date_range("2023-01-01", periods=30, freq="15min", tz="UTC")
        targets = build_seven_targets(index, np.ones(len(index)), horizons=(4, 16))
        valid = targets["mask"]
        for task in TASKS:
            expected = -0.06 * 0.00055 if task == "advantage" else 0.0
            np.testing.assert_allclose(targets[task][valid], expected)
        scales = fit_target_scales(targets, [1, 2, 3])
        self.assertEqual(scales["variance"], [1e-8, 1e-8])
        self.assertEqual(scales["return"], [1e-4, 1e-4])
        self.assertEqual(scales["crash_prior"], [0.0, 0.0])
        self.assertEqual(scales["crash"], [1.0, 1.0])

    def test_target_scales_ignore_nontrain_outcomes(self):
        index = pd.date_range("2023-01-01", periods=80, freq="15min", tz="UTC")
        targets = build_seven_targets(index, np.exp(np.sin(np.arange(len(index))) * 0.01), horizons=(4, 16))
        train_ids = np.array([1, 2, 3, 4])
        before = fit_target_scales(targets, train_ids)
        for task in TASKS:
            targets[task][20:] = 99999.0
        self.assertEqual(before, fit_target_scales(targets, train_ids))
        for invalid in ([], [0], [1, 1], [-1], [1.5]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                fit_target_scales(targets, invalid)
        self.assertEqual(before["return"], np.sqrt(np.mean(targets["return"][train_ids] ** 2, axis=0)).clip(1e-4).tolist())


class SevenTaskMetricsTest(unittest.TestCase):
    @staticmethod
    def data():
        actual = {
            "return": np.array([-0.02, -0.01, 0.01, 0.02]),
            "variance": np.array([1e-4, 2e-4, 1e-4, 2e-4]),
            "downside": np.array([0.02, 0.01, 0.0, 0.0]),
            "crash": np.array([1.0, 0.0, 0.0, 0.0]),
            "dd_excess": np.array([0.008, 0.0, 0.0, 0.0]),
            "advantage": np.array([-0.001, -0.0005, 0.0005, 0.001]),
            "recovery": np.array([-0.8, -0.5, 0.4, 0.9]),
        }
        predicted = {task: value.copy() for task, value in actual.items()}
        predicted["return"] = np.array([-0.01, 0.0, 0.0, 0.01])
        predicted["crash"] = np.array([0.8, 0.2, 0.1, 0.1])
        return actual, predicted

    def test_known_scoring_and_variance_qlike(self):
        actual, predicted = self.data()
        metrics = score_seven(actual, predicted, baseline_variance=2 * actual["variance"])
        self.assertEqual(metrics["n"], 4)
        self.assertAlmostEqual(metrics["return"]["rmse"], 0.01)
        self.assertAlmostEqual(metrics["return"]["zero_oos_r2"], 0.6)
        self.assertEqual(metrics["variance"]["qlike"], 0.0)
        self.assertEqual(metrics["variance"]["calibration_mean_ratio"], 1.0)
        self.assertAlmostEqual(metrics["variance"]["baseline_qlike"], 0.5 - np.log(0.5) - 1)
        self.assertAlmostEqual(metrics["variance"]["qlike_improvement_vs_baseline"], metrics["variance"]["baseline_qlike"])
        self.assertAlmostEqual(metrics["crash"]["brier"], 0.025)
        self.assertEqual(metrics["crash"]["pr_auc"], 1.0)
        self.assertEqual(sum(row["n"] for row in metrics["crash"]["calibration"]["bins"]), 4)
        self.assertEqual(metrics["dd_excess"]["mae"], 0.0)
        json.dumps(metrics, allow_nan=False)

    def test_zero_return_baseline_and_undefined_metrics(self):
        actual, predicted = self.data()
        predicted["return"][:] = 0
        metrics = score_seven(actual, predicted)
        self.assertEqual(metrics["return"]["zero_oos_r2"], 0.0)
        self.assertIsNone(metrics["return"]["pearson_ic"])
        self.assertIsNone(metrics["return"]["spearman_ic"])
        for task in TASKS:
            actual[task][:] = 0
            predicted[task][:] = 0
        metrics = score_seven(actual, predicted)
        self.assertIsNone(metrics["return"]["zero_oos_r2"])
        self.assertIsNone(metrics["crash"]["pr_auc"])
        self.assertIsNone(metrics["variance"]["calibration_mean_ratio"])
        self.assertEqual(metrics["variance"]["qlike"], 0)
        self.assertTrue(np.isfinite(metrics["crash"]["log_loss"]))
        json.dumps(metrics, allow_nan=False)

    def test_common_finite_set_and_optional_baseline_cohort(self):
        actual, predicted = self.data()
        predicted["recovery"][0] = np.nan
        baseline = np.array([1e-4, 1e-4, np.nan, 1e-4])
        metrics = score_seven(actual, predicted, baseline)
        self.assertEqual(metrics["n"], 3)
        self.assertEqual(metrics["n_nonfinite_excluded"], 1)
        self.assertEqual(metrics["variance"]["baseline_n"], 2)
        self.assertIsNone(metrics["crash"]["pr_auc"])
        self.assertEqual(metrics["crash"]["prevalence"], 0.0)

    def test_pinball_checks_conditional_quantile_not_event_probability(self):
        actual, predicted = self.data()
        point = np.full(4, 0.004)
        metrics = score_seven(actual, predicted, quantile_predictions={0.5: point, 0.9: point})
        error = actual["dd_excess"] - point
        self.assertAlmostEqual(metrics["dd_excess"]["pinball"]["0.5"]["loss"], np.abs(error).mean() / 2)
        self.assertAlmostEqual(metrics["dd_excess"]["pinball"]["0.9"]["loss"], np.maximum(0.9 * error, -0.1 * error).mean())
        with self.assertRaises(ValueError):
            score_seven(actual, predicted, quantile_predictions={1.0: point})

    def test_invalid_shapes_and_domains_rejected(self):
        actual, predicted = self.data()
        predicted["crash"][0] = 1.01
        with self.assertRaisesRegex(ValueError, "predictions probabilities"):
            score_seven(actual, predicted)
        predicted["crash"][0] = 0.8
        actual["variance"][0] = -1
        with self.assertRaisesRegex(ValueError, "must be nonnegative"):
            score_seven(actual, predicted)
        actual["variance"][0] = 1e-4
        predicted["recovery"] = np.ones((4, 1))
        with self.assertRaisesRegex(ValueError, "same 1-D or 2-D shape"):
            score_seven(actual, predicted)


if __name__ == "__main__":
    unittest.main()
