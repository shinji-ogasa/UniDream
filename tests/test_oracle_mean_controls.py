import unittest

import numpy as np

from unidream.experiments.oracle_mean_controls import constant_means, return_scores


def outcomes(returns):
    return np.column_stack([returns, np.full(len(returns), np.nan), np.full(len(returns), np.nan)])


class OracleMeanControlsTests(unittest.TestCase):
    def test_hand_constants_with_distinct_calibration_and_inference_lengths(self):
        calibration = outcomes([-.25, .75, 100., np.nan])
        infer = np.array([True, False, True, True, False])
        result = constant_means(inference_mask=infer, fit_mean=-.125,
                               calibration_actual=calibration,
                               scale_mask=np.array([True, True, False, False]))
        self.assertEqual(set(result), {"zero", "fit_mean", "scale_mean"})
        for name, value in (("zero", 0.), ("fit_mean", -.125), ("scale_mean", .25)):
            np.testing.assert_array_equal(result[name][infer], np.full(3, value))
            self.assertTrue(np.isnan(result[name][~infer]).all())
        np.testing.assert_array_equal(calibration[:, 0], [-.25, .75, 100., np.nan])

    def test_interval_future_and_other_outcome_values_never_change_scale_mean(self):
        calibration = outcomes([-.25, .75, 100., np.nan])
        arguments = {"inference_mask": np.array([True, True, False]), "fit_mean": .125,
                     "scale_mask": np.array([True, True, False, False])}
        original = constant_means(calibration_actual=calibration, **arguments)
        mutated = calibration.copy()
        mutated[2:, 0] = [np.inf, -1e200]
        mutated[:, 1:] = np.inf
        changed = constant_means(calibration_actual=mutated, **arguments)
        for name in original:
            np.testing.assert_array_equal(original[name], changed[name])

    def test_scoring_support_cannot_cancel_an_existing_causal_forecast(self):
        infer = np.array([True, True, True, False])
        constant = constant_means(inference_mask=infer, fit_mean=.125,
            calibration_actual=outcomes([-.25, .75]), scale_mask=np.array([True, True]))
        prediction = constant["scale_mean"].copy()
        actual = outcomes([.5, np.nan, -.5, np.nan])
        first = return_scores(actual, prediction, np.array([True, False, True, False]), .125)
        self.assertEqual(first["rows"], 2)
        self.assertEqual(prediction[1], .25)  # Outcome unavailable, causal prediction remains.
        actual[1, 0] = 3.
        second = return_scores(actual, prediction, np.array([True, True, True, False]), .125)
        self.assertEqual(second["rows"], 3)
        self.assertNotEqual(first["return_mse"], second["return_mse"])
        np.testing.assert_array_equal(prediction, constant["scale_mean"])

    def test_invalid_constant_masks_shapes_and_selected_values_fail_closed(self):
        valid = {"inference_mask": np.array([True, False]), "fit_mean": .125,
                 "calibration_actual": outcomes([-.25, .75]), "scale_mask": np.array([True, True])}
        cases = [("inference_mask", np.array([1, 0])), ("inference_mask", np.array([[True, False]])),
                 ("scale_mask", np.array([1., 0.])), ("scale_mask", np.array([True])),
                 ("scale_mask", np.array([[True, True]])), ("scale_mask", np.array([False, False])),
                 ("fit_mean", np.nan), ("fit_mean", np.inf), ("fit_mean", np.array([.125])),
                 ("calibration_actual", np.array([-.25, .75])),
                 ("calibration_actual", np.ones((2, 2))),
                 ("calibration_actual", outcomes([np.nan, .75])),
                 ("calibration_actual", outcomes([-.25, np.inf]))]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    constant_means(**{**valid, key: value})

    def test_all_false_inference_keeps_unavailable_forecasts_as_nan(self):
        result = constant_means(inference_mask=np.zeros(3, bool), fit_mean=.125,
            calibration_actual=outcomes([-.25, .75]), scale_mask=np.ones(2, bool))
        for values in result.values():
            self.assertTrue(np.isnan(values).all())

    def test_return_scores_hand_mse_mae_sign_references_and_rank(self):
        actual = outcomes([-2., 0., 2., 1000.])
        mu = np.array([-1., 1., 2., np.nan])
        result = return_scores(actual, mu, np.array([True, True, True, False]), .5)
        self.assertEqual(set(result), {"rows", "return_mse", "return_mae", "return_sign_accuracy",
                                      "zero_return_mse", "fit_mean_return_mse", "return_rank_ic"})
        self.assertEqual(result["rows"], 3)
        self.assertAlmostEqual(result["return_mse"], 2 / 3)
        self.assertAlmostEqual(result["return_mae"], 2 / 3)
        self.assertAlmostEqual(result["return_sign_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["zero_return_mse"], 8 / 3)
        self.assertAlmostEqual(result["fit_mean_return_mse"], 35 / 12)
        self.assertAlmostEqual(result["return_rank_ic"], 1.)
        tied = return_scores(outcomes([1., 1., 3.]), np.array([3., 1., 1.]), np.ones(3, bool), 0.)
        self.assertAlmostEqual(tied["return_rank_ic"], -.5)

    def test_constant_or_degenerate_rank_is_none_and_zero_loss_is_valid(self):
        for actual, mu in ((outcomes([1., 2., 3.]), np.zeros(3)),
                           (outcomes([2., 2., 2.]), np.array([1., 2., 3.])),
                           (outcomes([1.]), np.array([.5]))):
            with self.subTest(rows=len(mu)):
                self.assertIsNone(return_scores(actual, mu, np.ones(len(mu), bool), 0.)["return_rank_ic"])
        zero = return_scores(outcomes([0., 0.]), np.zeros(2), np.ones(2, bool), 0.)
        self.assertEqual(zero["return_mse"], 0.)
        self.assertEqual(zero["return_mae"], 0.)
        self.assertEqual(zero["zero_return_mse"], 0.)
        self.assertEqual(zero["fit_mean_return_mse"], 0.)
        self.assertEqual(zero["return_sign_accuracy"], 1.)

    def test_unscored_nonfinite_values_and_unused_outcomes_are_ignored(self):
        actual = outcomes([-.25, .5, np.nan])
        mu = np.array([0., .25, np.nan])
        scoring = np.array([True, True, False])
        expected = return_scores(actual, mu, scoring, 0.)
        changed = actual.copy()
        changed[2, 0] = np.inf
        changed[:, 1:] = np.inf
        mu[2] = -np.inf
        self.assertEqual(return_scores(changed, mu, scoring, 0.), expected)

    def test_invalid_scoring_shapes_support_and_nonfinite_values_reject(self):
        valid = {"actual": outcomes([-.25, .5]), "mu": np.array([0., .25]),
                 "score_mask": np.array([True, True]), "fit_mean": 0.}
        cases = [("actual", np.array([-.25, .5])), ("actual", np.ones((2, 2))),
                 ("actual", outcomes([np.nan, .5])), ("mu", np.array([np.inf, .25])),
                 ("mu", np.array([0.])), ("mu", np.zeros((2, 1))),
                 ("score_mask", np.array([1, 1])), ("score_mask", np.array([True])),
                 ("score_mask", np.ones((2, 1), bool)), ("score_mask", np.zeros(2, bool)),
                 ("fit_mean", np.inf), ("fit_mean", np.nan)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    return_scores(**{**valid, key: value})
        with self.assertRaisesRegex(ValueError, "nonfinite loss"):
            return_scores(outcomes([1e308]), np.array([-1e308]), np.ones(1, bool), 0.)


if __name__ == "__main__":
    unittest.main()
