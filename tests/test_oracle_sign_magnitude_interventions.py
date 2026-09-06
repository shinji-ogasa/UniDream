import copy
import json
import unittest

import numpy as np

from unidream.experiments.oracle_sign_magnitude_interventions import substitute_return_component


class Poison:
    def __float__(self):
        raise AssertionError("unselected outcome inspected")


class SignMagnitudeInterventionTests(unittest.TestCase):
    def setUp(self):
        self.args = dict(mu=np.array([.02, -.03, 0., .04, -.0, np.nan, -.05]),
            variance=np.array([.001, .002, 0., .003, -.0, np.nan, .004]),
            inference_mask=np.array([True, True, True, True, True, False, True]),
            score_support=np.array([True, True, True, True, True, False, False]),
            actual=np.array([[-.2, 1., 1.], [.1, 1., 1.], [-.4, 1., 1.],
                             [0., 1., 1.], [0., 1., 1.], [np.nan]*3, [np.inf]*3]))

    def apply(self, component, **changes):
        return substitute_return_component(**{**self.args, **changes}, component=component)

    def test_known_factorial_math_wrong_sign_and_zero_components(self):
        sign, magnitude = self.apply("sign"), self.apply("magnitude")
        np.testing.assert_array_equal(sign["mu"], [-.02, .03, -.0, 0., 0., np.nan, -.05])
        np.testing.assert_array_equal(magnitude["mu"], [.2, -.1, 0., 0., 0., np.nan, -.05])
        # A sign oracle fixes direction but retains learned magnitude. A
        # magnitude oracle deliberately retains a wrong learned direction.
        self.assertEqual(abs(sign["mu"][0]), abs(self.args["mu"][0]))
        self.assertGreater(magnitude["mu"][0], 0.)
        self.assertLess(self.args["actual"][0, 0], 0.)
        # Zero learned magnitude cannot be rescued by sign information;
        # zero learned sign cannot be rescued by magnitude information.
        self.assertEqual(sign["mu"][2], 0.)
        self.assertEqual(magnitude["mu"][2], 0.)
        self.assertTrue(np.all(sign["mu"][[3, 4]] == 0.))
        self.assertTrue(np.all(magnitude["mu"][[3, 4]] == 0.))

    def test_replacing_both_components_recovers_y_for_nonzero_intermediate(self):
        args = dict(mu=np.array([.02, -.03]), variance=np.array([.001, .002]),
            inference_mask=np.ones(2, bool), score_support=np.ones(2, bool),
            actual=np.array([[-.2, 0., 0.], [.1, 0., 0.]]))
        signed = substitute_return_component(**args, component="sign")
        both = substitute_return_component(**{**args, "mu": signed["mu"]}, component="magnitude")
        np.testing.assert_array_equal(both["mu"], args["actual"][:, 0])
        # This is a hand-check of the decomposition, not an added runtime mode.
        with self.assertRaises(ValueError):
            substitute_return_component(**args, component="both")

    def test_unscored_and_other_outcome_poison_is_ignored(self):
        poison = self.args["actual"].astype(object)
        poison[~self.args["score_support"], :] = Poison()
        poison[:, 1:] = Poison()
        for component in ("sign", "magnitude"):
            expected, actual = self.apply(component), self.apply(component, actual=poison)
            for key in ("mu", "variance", "inference_mask", "score_support"):
                np.testing.assert_array_equal(actual[key], expected[key])
            self.assertEqual(actual["metadata"], expected["metadata"])
            self.assertEqual(actual["mu"][-1], self.args["mu"][-1])

    def test_future_adversary_affects_only_permitted_selected_component(self):
        changed = self.args["actual"].copy()
        changed[0, 0] *= 10.  # same sign, different future magnitude
        sign_before, sign_after = self.apply("sign"), self.apply("sign", actual=changed)
        np.testing.assert_array_equal(sign_before["mu"], sign_after["mu"])
        magnitude = self.apply("magnitude", actual=changed)
        self.assertEqual(magnitude["mu"][0], 2.)
        changed[0, 0] *= -1.  # future sign flips; magnitude is now unchanged
        np.testing.assert_array_equal(self.apply("magnitude", actual=changed)["mu"], magnitude["mu"])
        sign_flipped = self.apply("sign", actual=changed)
        self.assertEqual(sign_flipped["mu"][0], .02)
        np.testing.assert_array_equal(sign_flipped["mu"][1:], sign_before["mu"][1:])

    def test_exact_risk_support_and_signed_zero_unscored_values_are_preserved(self):
        score = self.args["score_support"].copy(); score[4] = False
        for component in ("sign", "magnitude"):
            result = self.apply(component, score_support=score)
            np.testing.assert_array_equal(result["variance"], self.args["variance"])
            np.testing.assert_array_equal(result["inference_mask"], self.args["inference_mask"])
            np.testing.assert_array_equal(result["score_support"], score)
            np.testing.assert_array_equal(result["mu"][~score], self.args["mu"][~score])
            self.assertTrue(np.signbit(result["mu"][4]))
            self.assertTrue(np.signbit(result["variance"][4]))
            self.assertEqual(result["metadata"]["replacement_rows"], 4)
            self.assertEqual(result["metadata"]["learned_remainder_rows"], 2)

    def test_hindsight_metadata_and_input_nonmutation(self):
        before = copy.deepcopy(self.args)
        for component in ("sign", "magnitude"):
            result = self.apply(component)
            meta = result["metadata"]
            for key in ("hindsight_only", "future_information_used_for_decisions", "variance_unchanged",
                        "inference_and_missing_action_support_unchanged"):
                self.assertTrue(meta[key])
            for key in ("deployable", "teacher_use_allowed", "global_optimum_claimed", "other_outcome_columns_used"):
                self.assertFalse(meta[key])
            json.dumps(meta, allow_nan=False)
            for key in ("mu", "variance", "inference_mask", "score_support"):
                self.assertFalse(np.shares_memory(result[key], self.args[key]))
                np.testing.assert_array_equal(self.args[key], before[key])
            np.testing.assert_array_equal(self.args["actual"], before["actual"])
            result["inference_mask"][:] = False
            np.testing.assert_array_equal(self.args["inference_mask"], before["inference_mask"])

    def test_invalid_selected_actual_types_and_nonfinite_fail(self):
        for bad in (True, "0.1", 1 + 0j, np.nan, np.inf, -np.inf, Poison()):
            actual = self.args["actual"].astype(object); actual[0, 0] = bad
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                self.apply("sign", actual=actual)
        # Even a complex array with zero imaginary parts is not real typed data.
        with self.assertRaises(ValueError):
            self.apply("magnitude", actual=self.args["actual"].astype(complex))

    def test_invalid_schema_forecasts_masks_and_component_fail(self):
        changes = [dict(inference_mask=self.args["inference_mask"].astype(int)),
            dict(score_support=self.args["score_support"].astype(int)),
            dict(score_support=np.zeros(7, bool)), dict(score_support=np.ones(7, bool)),
            dict(inference_mask=np.ones((7, 1), bool)), dict(score_support=np.ones(6, bool)),
            dict(actual=np.zeros((7, 2))), dict(mu=np.zeros((7, 1))), dict(variance=np.zeros(6))]
        for change in changes:
            with self.subTest(change=list(change)), self.assertRaises(ValueError): self.apply("sign", **change)
        for name in ("mu", "variance"):
            for row, bad in ((0, np.nan), (0, np.inf), (0, True), (0, "1"), (0, 1+0j),
                             (5, 0.), (5, np.inf), (5, Poison())):
                values = self.args[name].astype(object); values[row] = bad
                with self.subTest(name=name, row=row, bad=repr(bad)), self.assertRaises(ValueError):
                    self.apply("magnitude", **{name: values})
        risk = self.args["variance"].copy(); risk[0] = -.001
        with self.assertRaises(ValueError): self.apply("sign", variance=risk)
        for component in ("return", "both", "SIGN", None, True, ["sign"]):
            with self.subTest(component=component), self.assertRaises(ValueError): self.apply(component)
        with self.assertRaises(ValueError):
            substitute_return_component([], [], inference_mask=np.zeros(0,bool),
                score_support=np.zeros(0,bool), actual=np.empty((0,3)), component="sign")

    def test_extreme_finite_magnitudes_do_not_overflow(self):
        args = dict(mu=np.array([1.7e308, -1.7e308]), variance=np.zeros(2),
            inference_mask=np.ones(2,bool), score_support=np.ones(2,bool),
            actual=np.array([[-1.6e308, 0., 0.], [1.6e308, 0., 0.]]))
        for component, expected in (("sign", [-1.7e308, 1.7e308]), ("magnitude", [1.6e308, -1.6e308])):
            result = substitute_return_component(**args, component=component)
            np.testing.assert_array_equal(result["mu"], expected)
            self.assertTrue(np.isfinite(result["mu"]).all())


if __name__ == "__main__":
    unittest.main()
