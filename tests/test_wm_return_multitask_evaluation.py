import unittest

import numpy as np

from unidream.experiments.wm_return_multitask_evaluation import (
    classification_metrics, coarse_return_projection, regression_metrics,
    scalar_classification_metrics,
)


class ReturnMultitaskEvaluationTests(unittest.TestCase):
    def test_return_r2_uses_zero_forecast_not_demeaned_reference(self):
        actual = np.array([1., 2., 3.])
        predicted = np.array([0., 2., 2.])
        result = regression_metrics(actual, predicted)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2 / 3))
        self.assertAlmostEqual(result["zero_oos_r2"], 1 - 2 / 14)
        self.assertAlmostEqual(result["pearson_ic"], np.corrcoef(actual, predicted)[0, 1])
        zero = regression_metrics(actual, np.zeros(3))
        self.assertEqual(zero["zero_oos_r2"], 0.)
        self.assertIsNone(zero["pearson_ic"])
        self.assertIsNone(regression_metrics(np.zeros(3), np.ones(3))["zero_oos_r2"])

    def test_prior_rule_and_raw_argmax_are_distinct_and_losses_remain_proper(self):
        actual = np.array([0, 1, 2])
        probabilities = np.array([[.4, .5, .1], [.1, .8, .1], [.1, .5, .4]])
        result = classification_metrics(actual, probabilities, np.array([.2, .6, .2]))
        self.assertEqual(result["balanced_accuracy"], 1.)
        self.assertAlmostEqual(result["raw_balanced_accuracy"], 1 / 3)
        self.assertAlmostEqual(result["log_loss"], -np.log([.4, .8, .4]).mean())
        expected = np.square(probabilities - np.eye(3)).sum(1).mean()
        self.assertAlmostEqual(result["multiclass_brier"], expected)

    def test_auxiliary_projections_do_not_claim_native_probability_losses(self):
        actual = np.array([0, 1, 2])
        result = scalar_classification_metrics(actual, np.array([-.02, .01, .02]), .01)
        self.assertEqual(result["balanced_accuracy"], 1.)
        self.assertNotIn("log_loss", result)
        self.assertNotIn("multiclass_brier", result)
        probs = np.array([[.1, .7, .2], [.8, .1, .1]])
        means = np.array([-.03, .001, .02])
        np.testing.assert_allclose(coarse_return_projection(probs, means), probs @ means)
        with self.assertRaises(ValueError):
            coarse_return_projection(probs * 2, means)


if __name__ == "__main__":
    unittest.main()
