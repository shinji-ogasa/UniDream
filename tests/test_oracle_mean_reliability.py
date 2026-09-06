"""Synthetic scale-only fitting and drift-aware reliability identities."""
import copy
import json
import math
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from unidream.experiments.oracle_mean_reliability import (
    apply_reliability, fit_reliability, score_decomposition,
)
from unidream.experiments.oracle_mean_shrinkage import half_mean


def fitting_case(d=(-2., 2.), r=(-1., 1.), n=80):
    actual = np.full((n, 3), np.nan)
    full = np.full(n, np.nan)
    mask = np.arange(n) < 64
    actual[mask, 0] = np.resize(r, 64)
    full[mask] = np.resize(d, 64)
    return full, actual, mask


class MeanReliabilityTests(unittest.TestCase):
    def test_known_continuous_slope_and_boundary_cases(self):
        for d, r, expected, case in [((-2., 2.), (-1., 1.), .5, "interior"),
                ((-4., 4.), (-1., 1.), .25, "interior"),
                ((-1., 1.), (-3., 3.), 1., "upper_endpoint"),
                ((-1., 1.), (1., -1.), 0., "nonpositive_crossmoment"),
                ((1., 1.), (-1., 1.), 0., "nonpositive_crossmoment")]:
            full, actual, mask = fitting_case(d, r)
            result = fit_reliability(full, actual, scale_mask=mask, anchor=0.)
            self.assertEqual(result["weight"], expected)
            self.assertEqual(result["weight_case"], case)
            self.assertTrue(result["identifiable"])
            self.assertEqual(result["n"], 64)
        full, actual, mask = fitting_case()
        result = fit_reliability(full, actual, scale_mask=mask, anchor=0.)
        self.assertEqual(result["innovation_secondmoment"], 4.)
        self.assertEqual(result["crossmoment"], 2.)

    def test_zero_second_moment_is_explicitly_unidentified(self):
        full, actual, mask = fitting_case((0., 0.), (-1., 1.))
        result = fit_reliability(full, actual, scale_mask=mask, anchor=0.)
        self.assertEqual(result["weight"], 0.)
        self.assertEqual(result["innovation_secondmoment"], 0.)
        self.assertEqual(result["crossmoment"], 0.)
        self.assertFalse(result["identifiable"])
        self.assertEqual(result["weight_case"], "zero_dispersion")

    def test_anchor_uses_exact_inherited_fsum_not_an_approximate_mean(self):
        full, actual, mask = fitting_case()
        actual[mask, 0] = np.resize([1e16, 1., -1e16, 1.], 64)
        anchor = math.fsum(float(v) / 64 for v in actual[mask, 0])
        full[mask] = anchor
        self.assertEqual(anchor, .5)
        result = fit_reliability(full, actual, scale_mask=mask, anchor=anchor)
        self.assertEqual(result["anchor"], anchor)
        with self.assertRaises(ValueError):
            fit_reliability(full, actual, scale_mask=mask, anchor=np.nextafter(anchor, 1.))

    def test_future_poison_and_unused_actual_columns_cannot_change_fitted_weight(self):
        full, actual, mask = fitting_case()
        expected = fit_reliability(full, actual, scale_mask=mask, anchor=0.)
        actual = actual.astype(object); full = full.astype(object)
        actual[~mask] = "unobserved outcome"
        actual[:, 1] = True
        actual[:, 2] = 1 + 5j
        full[~mask] = "unavailable forecast"
        self.assertEqual(fit_reliability(full, actual, scale_mask=mask, anchor=0.), expected)

    def test_endpoint_and_half_parity_including_signed_zero_and_extremes(self):
        maximum = np.finfo(float).max
        full = np.array([-0., maximum, -maximum, np.nan])
        anchor = np.array([-0., -0., -0., np.nan])
        mask = np.array([True, True, True, False])
        for weight, expected in ((0., anchor), (1., full)):
            result = apply_reliability(full, anchor, inference_mask=mask, weight=weight)
            assert_array_equal(result, expected)
            assert_array_equal(np.signbit(result[mask]), np.signbit(expected[mask]))
        result = apply_reliability(full, anchor, inference_mask=mask, weight=.5)
        assert_array_equal(result, half_mean(full, anchor, inference_mask=mask))
        for weight in (.25, .5, .75):
            result = apply_reliability([maximum, -maximum], [maximum, maximum],
                                       inference_mask=[True, True], weight=weight)
            self.assertTrue(np.isfinite(result).all())

    def test_application_ignores_unavailable_values_and_preserves_future_prefix(self):
        mask = np.array([True, False, True, True])
        original = apply_reliability([2., None, 4., 8.], [1., "poison", 1., 1.],
                                     inference_mask=mask, weight=.25)
        assert_array_equal(original, [1.25, np.nan, 1.75, 2.75])
        changed = apply_reliability([2., 1 + 2j, 4., -100.], [1., True, 1., 1.],
                                    inference_mask=mask, weight=.25)
        assert_array_equal(original[:3], changed[:3])
        short = apply_reliability([2., None, 4.], [1., None, 1.], inference_mask=mask[:3], weight=.25)
        assert_array_equal(short, original[:3])

    def test_decomposition_with_nonzero_drift_has_hand_computable_terms(self):
        actual = np.full((20, 3), np.nan); mask = np.arange(20) < 16
        actual[mask, 0] = np.tile([1., 2., 3., 4.], 4)
        mu = np.full(20, np.nan); mu[mask] = np.tile([2., 3., 4., 5.], 4)
        anchor = np.full(20, 2.)
        got = score_decomposition(actual, mu, anchor, mask)
        expected = {"n": 16, "candidate_mse": 1., "anchor_mse": 1.5, "lossdiff": -.5,
            "mean_d": 1.5, "mean_r": .5, "innovation_secondmoment": 3.5,
            "crossmoment": 2., "centered_variance_d": 1.25, "centered_covariance": 1.25,
            "centered_component": -1.25, "drift_component": .75, "identityresidual": 0.}
        self.assertEqual(got, expected)
        self.assertEqual(got["lossdiff"], got["innovation_secondmoment"] - 2 * got["crossmoment"])
        poison = actual.astype(object); poison[~mask] = "future"
        poison[:, 1:] = 1 + 1j
        self.assertEqual(score_decomposition(poison, mu, anchor, mask), got)
        json.dumps(got, allow_nan=False)

    def test_all_inputs_remain_unchanged(self):
        full, actual, scale = fitting_case()
        anchor = np.zeros(len(full)); snapshot = copy.deepcopy((full, actual, scale, anchor))
        for array in (full, actual, scale, anchor): array.setflags(write=False)
        fitted = fit_reliability(full, actual, scale_mask=scale, anchor=0.)
        mu = apply_reliability(full, anchor, inference_mask=scale, weight=fitted["weight"])
        score_decomposition(actual, mu, anchor, scale)
        for original, before in zip((full, actual, scale, anchor), snapshot): assert_array_equal(original, before)

    def test_selected_bool_complex_nonnumeric_and_nonfinite_values_rejected(self):
        full, actual, mask = fitting_case()
        for bad in (True, 1 + 0j, "1.0", np.nan, np.inf):
            with self.subTest(bad=bad):
                changed_mu = full.astype(object); changed_mu[0] = bad
                changed_y = actual.astype(object); changed_y[0, 0] = bad
                with self.assertRaises(ValueError): fit_reliability(changed_mu, actual, scale_mask=mask, anchor=0.)
                with self.assertRaises(ValueError): fit_reliability(full, changed_y, scale_mask=mask, anchor=0.)
                with self.assertRaises(ValueError): apply_reliability(changed_mu, np.zeros(len(full)), inference_mask=mask, weight=.5)
                with self.assertRaises(ValueError): score_decomposition(changed_y, full, np.zeros(len(full)), mask)
        mixed = [1.] * 80; mixed[0] = True
        with self.assertRaises(ValueError): fit_reliability(mixed, actual, scale_mask=mask, anchor=0.)

    def test_mask_shape_minima_weight_anchor_and_overflow_guards(self):
        full, actual, mask = fitting_case()
        for badmask in (mask.astype(int), mask[:-1], mask[:, None], np.arange(80) < 63):
            with self.assertRaises(ValueError): fit_reliability(full, actual, scale_mask=badmask, anchor=0.)
        with self.assertRaises(ValueError): score_decomposition(actual, full, np.zeros(80), np.arange(80) < 15)
        for badweight in (True, 1 + 0j, np.nan, -1., 1.01, [0.5]):
            with self.assertRaises(ValueError): apply_reliability(full, np.zeros(80), inference_mask=mask, weight=badweight)
        anchor = np.zeros(80); anchor[1] = .1
        with self.assertRaises(ValueError): apply_reliability(full, anchor, inference_mask=mask, weight=.5)
        with self.assertRaises(ValueError): score_decomposition(actual, full, anchor, mask)
        with self.assertRaises(ValueError): fit_reliability(full, actual[:, :2], scale_mask=mask, anchor=0.)
        with self.assertRaises(ValueError): apply_reliability(full, np.zeros(79), inference_mask=mask, weight=.5)
        huge = np.full(80, 1e308)
        with self.assertRaises(ValueError): fit_reliability(huge, actual, scale_mask=mask, anchor=0.)
        with self.assertRaises(ValueError): score_decomposition(actual, huge, np.zeros(80), mask)


if __name__ == "__main__":
    unittest.main()
