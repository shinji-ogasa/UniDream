import json
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from unidream.experiments.wm_return_formulation_evaluation import (
    calibrated_probs,
    compare_matched_formulations,
    fit_probability_calibrator,
    fit_return_calibrator,
    make_return_labels,
    return_class_priors,
    score_probability_formulation,
    score_return_formulation,
)


class ReturnFormulationEvaluationTest(unittest.TestCase):
    @staticmethod
    def calibration_data():
        rng = np.random.default_rng(12)
        predicted = rng.normal(0, .01, 300)
        actual = .6 * predicted + rng.normal(0, .006, 300)
        return actual, predicted

    def test_fixed_threshold_and_class_priors(self):
        labels = make_return_labels(np.array([-.02, -.01, 0., .01, .02]), .01)
        np.testing.assert_array_equal(labels, [0, 1, 1, 1, 2])
        np.testing.assert_allclose(return_class_priors(labels), [.2, .6, .2])
        with self.assertRaisesRegex(ValueError, "all three classes"):
            return_class_priors([0, 1, 1])
        with self.assertRaises(ValueError):
            make_return_labels(np.zeros((3, 1)), .01)
        with self.assertRaises(ValueError):
            return_class_priors([0, 1.5, 2])

    def test_json_logistic_replay_matches_sklearn_and_scale_invariance(self):
        actual, scalar = self.calibration_data()
        fit = fit_return_calibrator(actual, scalar, .005, .01)
        frozen = json.loads(json.dumps(fit, allow_nan=False))
        labels = make_return_labels(actual, .005)
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000, tol=1e-8, random_state=20261110)
        model.fit((scalar / .01)[:, None], labels)
        query = np.array([-.03, -.005, 0., .002, .06])
        np.testing.assert_allclose(calibrated_probs(query, frozen), model.predict_proba((query / .01)[:, None]), atol=1e-11)
        larger = fit_return_calibrator(actual * 10, scalar * 10, .05, .1)
        np.testing.assert_allclose(calibrated_probs(query * 10, larger), calibrated_probs(query, frozen), atol=1e-10)
        self.assertEqual(frozen, fit)

    def test_missing_calibration_classes_preserve_three_column_probability_space(self):
        for actual in (np.array([-.02, -.01, .01, .02]), np.zeros(4)):
            scalar = np.array([-.01, -.005, .005, .01])
            fit = fit_return_calibrator(actual, scalar, .005, .01)
            probabilities = calibrated_probs(scalar, json.loads(json.dumps(fit, allow_nan=False)))
            self.assertEqual(probabilities.shape, (4, 3))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1)
            self.assertTrue((probabilities > 0).all())
            present = set(make_return_labels(actual, .005))
            for absent in set((0, 1, 2)) - present:
                self.assertTrue((probabilities[:, absent] <= 1e-12).all())

    def test_temperature_improves_validation_ce_and_keeps_raw_order(self):
        labels = np.tile([0, 1, 2], 30)
        probabilities = np.tile([.98, .01, .01], (len(labels), 1))
        fit = fit_probability_calibrator(labels, probabilities)
        self.assertLessEqual(fit["fit_log_loss_after"], fit["fit_log_loss_before"])
        self.assertGreater(fit["temperature"], 1)
        self.assertLessEqual(fit["temperature"], 10)
        corrected = calibrated_probs(probabilities, json.loads(json.dumps(fit, allow_nan=False)))
        np.testing.assert_array_equal(corrected.argmax(axis=1), probabilities.argmax(axis=1))
        np.testing.assert_allclose(corrected.sum(axis=1), 1)
        # Test application does not mutate frozen calibration parameters.
        before = json.dumps(fit, sort_keys=True)
        calibrated_probs(np.eye(3), fit)
        self.assertEqual(before, json.dumps(fit, sort_keys=True))

    def test_scalar_projection_has_no_probability_scores_and_matched_calibration(self):
        actual, scalar = self.calibration_data()
        calibrator = fit_return_calibrator(actual, scalar, .005, .01)
        returns = np.array([-.02, -.01, .0, .01, .02])
        prediction = np.array([-.01, 0., 0., 0., .01])
        priors = np.array([.2, .6, .2])
        scalar_scores = score_return_formulation(returns, prediction, .005, priors, calibrator)
        self.assertNotIn("log_loss", scalar_scores["scalar_projection"])
        self.assertNotIn("multiclass_brier", scalar_scores["scalar_projection"])
        self.assertIn("log_loss", scalar_scores["calibrated"])
        probabilities = np.array([[.8, .1, .1], [.4, .5, .1], [.2, .6, .2], [.1, .5, .4], [.1, .1, .8]])
        temperature = fit_probability_calibrator(make_return_labels(returns, .005), probabilities)
        result = compare_matched_formulations(returns, prediction, probabilities, .005, priors, calibrator, temperature)
        self.assertEqual(result["n_matched"], 5)
        self.assertFalse(result["seed_stability_claim"])
        self.assertIn("raw", result["classification"]["calibrated_probabilities"])
        json.dumps(result, allow_nan=False)
        with self.assertRaisesRegex(ValueError, "identical row coverage"):
            compare_matched_formulations(returns, prediction, probabilities[:-1], .005, priors, calibrator)

    def test_primary_decision_uses_train_prior_and_raw_is_auxiliary(self):
        probabilities = np.tile([.15, .45, .4], (3, 1))
        result = score_probability_formulation(np.array([0, 1, 2]), probabilities, np.array([.2, .6, .2]))
        score = result["raw_probabilities"]
        self.assertEqual(score["predicted_class_frequency"], [0., 0., 1.])
        self.assertEqual(score["raw"]["predicted_class_frequency"], [0., 1., 0.])


if __name__ == "__main__":
    unittest.main()
