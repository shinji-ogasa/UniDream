import json
import unittest

import numpy as np

from unidream.experiments.wm_full_grid_evaluation import fit_label_contract, make_task_labels, score_grid


class FullGridEvaluationTest(unittest.TestCase):
    @staticmethod
    def targets(n=900):
        values = {
            "return": np.tile([-.02, 0., .02], n // 3),
            "variance": np.linspace(.00001, .001, n),
            "downside": np.linspace(0., .03, n),
            "crash": np.tile([0., 0., 1.], n // 3),
            "dd_excess": np.r_[np.zeros(n // 3), np.linspace(.001, .02, 2 * n // 3)],
            "advantage": np.tile([-.001, 0., .001], n // 3),
            "recovery": np.tile([-.6, 0., .6], n // 3),
        }
        return {task: value[:, None] for task, value in values.items()}

    def test_train_only_fitting_and_json_label_contract(self):
        targets = self.targets()
        targets = {task: np.r_[value, np.ones((30, 1))] for task, value in targets.items()}
        ids = np.arange(900)
        before = fit_label_contract(targets, ids)
        for task in targets:
            targets[task][900:] = np.nan
        after = fit_label_contract(targets, ids)
        self.assertEqual(before, after)
        json.dumps(before, allow_nan=False)
        self.assertEqual(before["crash"]["n_classes"], 2)
        self.assertEqual(before["dd_excess"]["support"], [300, 300, 300])
        self.assertEqual(before["return"]["support"], [300, 300, 300])
        self.assertEqual(before["variance"]["quantile_method"], "linear")
        np.testing.assert_allclose(before["variance"]["bounds"], np.quantile(targets["variance"][ids, 0], [1 / 3, 2 / 3], method="linear"))

    def test_boundary_equality_and_zero_inflated_excess_keep_shapes(self):
        contract = fit_label_contract(self.targets(), np.arange(900))
        low, high = contract["return"]["bounds"]
        y = np.array([[low - .01, low, 0.], [high, high + .01, 0.]])
        np.testing.assert_array_equal(make_task_labels(y, "return", contract), [[0, 1, 1], [1, 2, 1]])
        low, high = contract["variance"]["bounds"]
        np.testing.assert_array_equal(make_task_labels(np.array([low, high, high + .001]), "variance", contract), [0, 1, 2])
        boundary = contract["dd_excess"]["bounds"][1]
        np.testing.assert_array_equal(make_task_labels(np.array([-.001, 0, boundary / 2, boundary, boundary * 2]), "dd_excess", contract), [0, 0, 1, 1, 2])

    def test_tied_values_and_inadequate_train_class_counts_fail_explicitly(self):
        targets = self.targets()
        targets["dd_excess"][300:] = .01
        with self.assertRaisesRegex(ValueError, "class support"):
            fit_label_contract(targets, np.arange(900))
        with self.assertRaisesRegex(ValueError, "class support"):
            fit_label_contract(self.targets(), np.arange(300))
        targets = self.targets()
        targets["variance"][:] = .001
        with self.assertRaisesRegex(ValueError, "distinct and increasing"):
            fit_label_contract(targets, np.arange(900))

    def test_native_return_metrics_and_no_probability_claim_for_scalar(self):
        contract = fit_label_contract(self.targets(), np.arange(900))
        actual = np.array([-.02, 0., .02])
        result = score_grid("return", actual, actual / 2, "regression", contract, {"return": [.02]})
        self.assertAlmostEqual(result["native"]["zero_oos_r2"], .75)
        self.assertEqual(result["primary_metric"], "zero_oos_r2")
        self.assertEqual(result["classification"]["balanced_accuracy"], 1.)
        self.assertFalse(result["classification"]["probability_metrics_available"])
        self.assertNotIn("log_loss", result["classification"])
        self.assertEqual(result["target_scale"], .02)
        zero = score_grid("return", np.zeros(3), np.zeros(3), "regression", contract, None)
        self.assertIsNone(zero["primary_value"])
        self.assertIsNone(zero["classification"]["balanced_accuracy"])
        json.dumps(zero, allow_nan=False)

    def test_binary_regression_and_classification_share_event_scores(self):
        contract = fit_label_contract(self.targets(), np.arange(900))
        actual, p = np.array([0., 1., 1., 0.]), np.array([.2, .4, .8, .4])
        regression = score_grid("crash", actual, p, "regression", contract, None)
        classification = score_grid("crash", actual, np.column_stack((1 - p, p)), "classification", contract, None)
        self.assertAlmostEqual(regression["native"]["brier"], .15)
        self.assertEqual(regression["native"], classification["native"])
        self.assertAlmostEqual(classification["classification"]["multiclass_brier"], .3)
        self.assertEqual(regression["classification"]["per_class_recall"], [.5, 1.])
        self.assertEqual(regression["classification"]["raw"]["per_class_recall"], [1., .5])
        self.assertEqual(classification["primary_metric"], "balanced_accuracy")
        with self.assertRaisesRegex(ValueError, "event probabilities"):
            score_grid("crash", actual, p + 1, "regression", contract, None)

    def test_classification_prior_adjustment_and_missing_class_definition(self):
        contract = fit_label_contract(self.targets(), np.arange(900))
        contract["return"]["priors"] = [.2, .6, .2]
        p = np.tile([.15, .45, .4], (3, 1))
        result = score_grid("return", np.array([-.02, 0., .02]), p, "classification", contract, None)
        self.assertIsNone(result["native"])
        self.assertEqual(result["classification"]["predicted_class_frequency"], [0., 0., 1.])
        self.assertEqual(result["classification"]["raw"]["predicted_class_frequency"], [0., 1., 0.])
        missing = score_grid("return", np.zeros(3), p, "classification", contract, None)
        self.assertIsNone(missing["primary_value"])
        self.assertEqual(len(missing["classification"]["per_class_recall"]), 3)
        json.dumps(missing, allow_nan=False)

    def test_nonfinite_inputs_are_not_dropped_and_variance_uses_integrated_squares(self):
        contract = fit_label_contract(self.targets(), np.arange(900))
        with self.assertRaisesRegex(ValueError, "finite"):
            score_grid("return", np.zeros(3), np.array([0., np.nan, 0.]), "regression", contract, None)
        actual = np.array([4e-4, 4e-4, 4e-4])
        result = score_grid("variance", actual, np.full(3, 2e-4), "regression", contract, None)
        self.assertAlmostEqual(result["native"]["qlike"], 1 - np.log(2))
        self.assertAlmostEqual(result["native"]["calibration_mean_ratio"], .5)


if __name__ == "__main__":
    unittest.main()
