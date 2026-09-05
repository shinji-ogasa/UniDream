import inspect
import unittest

import numpy as np

from unidream.experiments.oracle_mean_shrinkage import half_mean


class OracleMeanShrinkageTests(unittest.TestCase):
    def test_hand_half_distance_to_supplied_constant_anchor(self):
        mu = np.array([-.75, .25, 1.25, np.nan])
        anchor = np.array([.25, .25, .25, np.inf])
        inference = np.array([True, True, True, False])
        result = half_mean(mu, anchor, inference_mask=inference)
        np.testing.assert_array_equal(result, [-.25, .25, .75, np.nan])
        np.testing.assert_array_equal(result[inference] - anchor[inference],
                                      .5 * (mu[inference] - anchor[inference]))

    def test_forecast_equal_to_anchor_is_valid_and_unchanged(self):
        for value in (0., -.125, .375, np.finfo(float).max):
            with self.subTest(value=value):
                values = np.full(3, value)
                np.testing.assert_array_equal(half_mean(values, values, inference_mask=np.ones(3, bool)), values)

    def test_extreme_finite_values_do_not_overflow_sum_or_difference(self):
        limit = np.finfo(float).max
        for sign in (-1., 1.):
            anchor = np.full(4, sign * limit)
            mu = sign * np.array([limit, -limit, 0., limit / 2])
            with self.subTest(sign=sign), np.errstate(over="raise", invalid="raise"):
                result = half_mean(mu, anchor, inference_mask=np.ones(4, bool))
            np.testing.assert_array_equal(result, sign * np.array([limit, 0., limit / 2, .75 * limit]))
            self.assertTrue(np.isfinite(result).all())

    def test_unavailable_values_are_ignored_and_inputs_preserved(self):
        mu = np.array([-.5, np.nan, .5, np.inf])
        anchor = np.array([.25, np.nan, .25, -np.inf])
        inference = np.array([True, False, True, False])
        originals = [value.copy() for value in (mu, anchor, inference)]
        for value in (mu, anchor, inference):
            value.setflags(write=False)
        expected = half_mean(mu, anchor, inference_mask=inference)
        changed_mu, changed_anchor = mu.copy(), anchor.copy()
        changed_mu[~inference], changed_anchor[~inference] = [-np.inf, 1e300], [1e300, np.inf]
        np.testing.assert_array_equal(half_mean(changed_mu, changed_anchor, inference_mask=inference), expected)
        for value, original in zip((mu, anchor, inference), originals):
            np.testing.assert_array_equal(value, original)
        expected[0] = 99.
        self.assertEqual(mu[0], -.5)
        self.assertEqual(anchor[0], .25)

    def test_future_values_and_support_do_not_change_existing_prefix(self):
        mu = np.array([-.5, .5, np.nan, .75, -.75])
        anchor = np.full(5, .25)
        inference = np.array([True, True, False, True, True])
        original = half_mean(mu, anchor, inference_mask=inference)
        mu[3:] = [100., -100.]
        available_future = half_mean(mu, anchor, inference_mask=inference)
        np.testing.assert_array_equal(original[:3], available_future[:3])
        mu[3:], anchor[3:], inference[3:] = [np.inf, np.nan], [-100., 100.], False
        after = half_mean(mu, anchor, inference_mask=inference)
        np.testing.assert_array_equal(original[:3], after[:3])
        prefix = half_mean(mu[:3], anchor[:3], inference_mask=inference[:3])
        np.testing.assert_array_equal(original[:3], prefix)

    def test_fixed_positive_half_weight_preserves_order_and_ties_in_hand_example(self):
        mu = np.array([2., -3., -1., 4., -1.])
        result = half_mean(mu, np.ones(5), inference_mask=np.ones(5, bool))
        np.testing.assert_array_equal(result, [1.5, -1., 0., 2.5, 0.])
        np.testing.assert_array_equal(np.sign(result[:, None] - result), np.sign(mu[:, None] - mu))

    def test_nonconstant_selected_anchor_is_rejected_without_tolerance(self):
        for anchor in ([.25, .5], [.25, np.nextafter(.25, 1.)]):
            with self.subTest(anchor=anchor), self.assertRaisesRegex(ValueError, "constant"):
                half_mean([-.5, .5], anchor, inference_mask=np.ones(2, bool))

    def test_invalid_shapes_masks_and_empty_support_are_rejected(self):
        valid = {"mu": np.zeros(2), "anchor": np.ones(2), "inference_mask": np.ones(2, bool)}
        cases = [("mu", np.zeros(1)), ("mu", np.zeros((2, 1))), ("mu", 1.),
                 ("anchor", np.ones(3)), ("anchor", np.ones((1, 2))),
                 ("inference_mask", np.ones(2)), ("inference_mask", np.ones(2, int)),
                 ("inference_mask", np.ones((2, 1), bool)), ("inference_mask", np.ones(1, bool)),
                 ("inference_mask", np.zeros(2, bool)), ("inference_mask", True)]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                half_mean(**{**valid, key: value})
        with self.assertRaisesRegex(ValueError, "nonempty"):
            half_mean([], [], inference_mask=np.array([], bool))

    def test_claimed_valid_nonfinite_values_are_rejected(self):
        for key in ("mu", "anchor"):
            for value in (np.nan, np.inf, -np.inf):
                arguments = {"mu": np.zeros(2), "anchor": np.zeros(2), "inference_mask": np.ones(2, bool)}
                arguments[key][0] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "finite"):
                    half_mean(**arguments)

    def test_api_has_no_weight_outcome_or_scoring_inputs(self):
        self.assertEqual(tuple(inspect.signature(half_mean).parameters), ("mu", "anchor", "inference_mask"))
        for key in ("weight", "weights", "actual", "score_mask"):
            with self.subTest(key=key), self.assertRaises(TypeError):
                half_mean([0.], [1.], inference_mask=np.ones(1, bool), **{key: .5})


if __name__ == "__main__":
    unittest.main()
