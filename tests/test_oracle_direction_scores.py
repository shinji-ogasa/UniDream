import copy
import json
import math
import sys
import unittest

import numpy as np

from unidream.experiments.oracle_direction_scores import direction_scores


class Poison:
    def __float__(self):
        raise AssertionError("unselected value inspected")


class DirectionScoresTests(unittest.TestCase):
    def score(self, y, logits):
        return direction_scores([[v, Poison(), Poison()] for v in y], logits,
                                np.ones(len(y), dtype=bool))

    def test_known_probability_and_weighted_formulas(self):
        # Probabilities 1/4, 3/4, 1/2; labels 0, 1, 1; weights 1, 3, 2.
        result = self.score([-1., 3., 2.], [-math.log(3), math.log(3), 0.])
        self.assertAlmostEqual(result["log_loss"], (2 * math.log(4 / 3) + math.log(2)) / 3)
        self.assertAlmostEqual(result["brier"], (1 / 16 + 1 / 16 + 1 / 4) / 3)
        self.assertAlmostEqual(result["binary_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["signed_return_mean"], 4 / 3)
        self.assertAlmostEqual(result["weighted_log_loss"],
                               (4 * math.log(4 / 3) + 2 * math.log(2)) / 6)
        self.assertAlmostEqual(result["weighted_brier"], (4 / 16 + 2 / 4) / 6)
        self.assertAlmostEqual(result["weighted_binary_accuracy"], 4 / 6)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["zero_actual_rows"], 0)
        self.assertEqual(result["zero_logit_rows"], 1)
        self.assertEqual(result["absolute_return_sum"], 6.)
        self.assertEqual(result["absolute_return_mean"], 2.)
        json.dumps(result, allow_nan=False)

    def test_extreme_logits_stable_without_probability_clipping(self):
        result = self.score([1., -1., 1., -1.], [1000., -1000., -1000., 1000.])
        self.assertEqual(result["log_loss"], 500.)
        self.assertEqual(result["weighted_log_loss"], 500.)
        self.assertEqual(result["brier"], .5)
        self.assertEqual(result["binary_accuracy"], .5)
        perfect = self.score([1., -1.], [1000., -1000.])
        self.assertEqual(perfect["brier"], 0.)
        self.assertEqual(perfect["log_loss"], 0.)
        largest = self.score([-1., 1.], [sys.float_info.max, -sys.float_info.max])
        self.assertEqual(largest["log_loss"], sys.float_info.max)
        self.assertEqual(largest["weighted_log_loss"], sys.float_info.max)

    def test_zero_returns_have_zero_weight_but_are_nonpositive_labels(self):
        result = self.score([0., -0., 2.], [3., -0., 0.])
        self.assertEqual(result["zero_actual_rows"], 2)
        self.assertEqual(result["zero_logit_rows"], 2)
        self.assertEqual(result["binary_accuracy"], 1 / 3)
        self.assertEqual(result["signed_return_mean"], 0.)
        self.assertEqual(result["weighted_log_loss"], math.log(2))
        self.assertEqual(result["weighted_brier"], .25)
        self.assertEqual(result["weighted_binary_accuracy"], 0.)

    def test_all_zero_weight_group_has_null_weighted_metrics(self):
        result = self.score([0., -0.], [1., -1.])
        self.assertEqual(result["absolute_return_sum"], 0.)
        self.assertEqual(result["absolute_return_mean"], 0.)
        self.assertEqual(result["binary_accuracy"], .5)
        for key in ("weighted_log_loss", "weighted_brier", "weighted_binary_accuracy"):
            self.assertIsNone(result[key])
        json.dumps(result, allow_nan=False)

    def test_scalar_precision_and_small_probability_are_retained(self):
        result = self.score([1.], [-40.])
        p = math.exp(-40.) / (1 + math.exp(-40.))
        self.assertEqual(result["brier"], (1 - p) ** 2)
        self.assertEqual(result["log_loss"], 40.)
        result = self.score([-1.], [-20.])
        p = math.exp(-20.) / (1 + math.exp(-20.))
        self.assertEqual(result["brier"], p ** 2)
        self.assertGreater(result["brier"], 0.)
        self.assertLess(result["brier"], 1e-16)

    def test_unselected_and_other_outcome_poison_is_never_converted(self):
        actual = np.array([[1., 0., 0.], [-1., 0., 0.], [3., 0., 0.]], dtype=object)
        logits = np.array([.3, -.2, 1.1], dtype=object)
        mask = np.array([True, False, True])
        expected = direction_scores(actual, logits, mask)
        actual[:, 1:] = Poison()
        actual[1, 0] = Poison()
        logits[1] = Poison()
        self.assertEqual(direction_scores(actual, logits, mask), expected)
        self.assertEqual(self.score([1., 3.], [.3, 1.1]), expected)

    def test_nonmutation_and_return_types(self):
        actual = np.array([[1., 2., 3.], [-4., 5., 6.]])
        logits = np.array([.2, -.3])
        mask = np.ones(2, bool)
        before = copy.deepcopy((actual, logits, mask))
        result = direction_scores(actual, logits, mask)
        for current, old in zip((actual, logits, mask), before):
            np.testing.assert_array_equal(current, old)
        for key, value in result.items():
            self.assertIs(type(value), int if key in
                          ("rows", "zero_actual_rows", "zero_logit_rows") else float)

    def test_weight_scaling_preserves_weighted_probability_scores(self):
        small, large = self.score([.1, -.3, 0.], [.2, .3, .5]), self.score([1., -3., 0.], [.2, .3, .5])
        for key in ("weighted_log_loss", "weighted_brier", "weighted_binary_accuracy"):
            self.assertAlmostEqual(small[key], large[key])
        # A subnormal nonzero weight must not be confused with a zero denominator.
        subnormal = self.score([np.nextafter(0., 1.)], [0.])
        self.assertEqual(subnormal["weighted_log_loss"], math.log(2))
        self.assertEqual(subnormal["weighted_brier"], .25)
        repeated = self.score([np.nextafter(0., 1.)] * 2, [0., 0.])
        self.assertEqual(repeated["absolute_return_mean"], np.nextafter(0., 1.))

    def test_malformed_shapes_and_nonboolean_or_empty_masks_fail(self):
        actual, logits = np.zeros((2, 3)), np.zeros(2)
        for mask in ([1, 1], [.1, .2], [True], [False, False], [], [[True], [True]],
                     np.array([True, True], dtype=object)):
            with self.subTest(mask=repr(mask)), self.assertRaises(ValueError):
                direction_scores(actual, logits, mask)
        for bad_actual in (np.zeros((2, 2)), np.zeros((3, 3)), np.zeros(2)):
            with self.assertRaises(ValueError):
                direction_scores(bad_actual, logits, [True, True])
        for bad_logits in (np.zeros((2, 1)), np.zeros(3), 0.):
            with self.assertRaises(ValueError):
                direction_scores(actual, bad_logits, [True, True])

    def test_selected_type_and_finiteness_validation_precedes_conversion(self):
        for bad in (True, np.bool_(False), 1 + 0j, "1", None, Poison(), np.nan,
                    np.inf, -np.inf, 10 ** 1000):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(ValueError):
                self.score([bad], [0.])
            with self.subTest(bad=type(bad).__name__), self.assertRaises(ValueError):
                self.score([1.], [bad])
        with self.assertRaises(ValueError):
            direction_scores(np.ones((2, 3), complex), [0., 0.], [True, True])

    def test_unrepresentable_absolute_sum_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "absolute_return_sum"):
            self.score([sys.float_info.max, sys.float_info.max], [0., 0.])


if __name__ == "__main__":
    unittest.main()
