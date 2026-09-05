"""Synthetic contracts for the fixed chronological forecast procedure."""
import json
import math
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import HistGradientBoostingRegressor

from unidream.experiments.oracle_frozen_forecasts import fit_frozen_forecasts


def fixture():
    rng = np.random.default_rng(106)
    n = 800
    technical = rng.normal(size=(n, 3))
    perp = np.column_stack((technical, rng.normal(size=(n, 2))))
    returns = .002 * technical[:, 0] + .001 * perp[:, 3] + .0005 * rng.normal(size=n)
    outcomes = np.column_stack((returns, np.abs(returns) + .001,
                                np.sqrt(.0001 + .00005 * (technical[:, 1] > 0))))
    masks = {}
    for name, start, stop in (("fit", 0, 520), ("scale", 528, 600),
                              ("interval", 608, 680), ("predict", 528, 780),
                              ("inference", 688, 760)):
        masks[name + "_mask"] = (np.arange(n) >= start) & (np.arange(n) < stop)
    selected = masks["fit_mask"] | masks["predict_mask"]
    technical[~selected], perp[~selected] = np.nan, np.nan
    outcomes[~(masks["fit_mask"] | masks["scale_mask"] | masks["interval_mask"])] = np.nan
    return {"technical": technical, "perp_delay0": perp}, outcomes, masks


class FrozenForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.outcomes, cls.masks = fixture()
        cls.result = fit_frozen_forecasts(cls.groups, cls.outcomes, **cls.masks)

    def assert_same_predictions(self, candidate):
        for name, array in self.result["means"].items():
            assert_array_equal(candidate["means"][name], array)
        assert_array_equal(candidate["variance"], self.result["variance"])
        for name, predictions in self.result["raw_predictions"].items():
            for key, array in predictions.items():
                assert_array_equal(candidate["raw_predictions"][name][key], array)

    def test_output_support_and_three_model_inventory(self):
        result = self.result
        inference, predict = self.masks["inference_mask"], self.masks["predict_mask"]
        self.assertEqual(set(result["means"]), {"scale_mean", "technical_scaled", "perp_delay0_scaled",
                                                "technical_half", "perp_delay0_half"})
        self.assertEqual(set(result["models"]), {"technical_mean", "perp_delay0_mean", "technical_variance"})
        self.assertEqual(sum(isinstance(m, HistGradientBoostingRegressor) for m in result["models"].values()), 1)
        for array in [*result["means"].values(), result["variance"]]:
            self.assertEqual(array.shape, (800,))
            self.assertTrue(np.isfinite(array[inference]).all())
            self.assertTrue(np.isnan(array[~inference]).all())
        self.assertTrue((result["variance"][inference] > 0).all())
        for group in result["raw_predictions"].values():
            for array in group.values():
                self.assertTrue(np.isfinite(array[predict]).all())
                self.assertTrue(np.isnan(array[~predict]).all())
        selected = self.masks["scale_mask"] | self.masks["interval_mask"]
        actual = result["calibration_arrays"]["actual"]
        assert_array_equal(actual[selected], self.outcomes[selected])
        self.assertTrue(np.isnan(actual[~selected]).all())
        for name, value in self.masks.items():
            assert_array_equal(value, result["masks"][name.removesuffix("_mask")])
            self.assertFalse(np.shares_memory(value, result["masks"][name.removesuffix("_mask")]))
        json.dumps(result["calibration"], allow_nan=False)
        json.dumps(result["provenance"], allow_nan=False)

    def test_frozen_calibration_arithmetic_and_half_anchor(self):
        result, y = self.result, self.outcomes
        fit, scale, inference = (self.masks[k + "_mask"] for k in ("fit", "scale", "inference"))
        calibration = result["calibration"]
        expected_anchor = math.fsum(float(v) / int(scale.sum()) for v in y[scale, 0])
        self.assertEqual(calibration["fit_mean"], float(np.mean(y[fit, 0])))
        self.assertEqual(calibration["scale_mean"], expected_anchor)
        self.assertEqual(calibration["counts"], {"fit": 520, "scale": 72, "interval": 72})
        expected_multiplier = float(np.mean(y[scale, 2] ** 2 / result["raw_predictions"]["technical"]["variance"][scale]))
        self.assertEqual(calibration["variance_multiplier"], expected_multiplier)
        for name in ("technical", "perp_delay0"):
            mu = result["raw_predictions"][name]["mu"]
            bias = float(np.mean(y[scale, 0] - mu[scale]))
            self.assertEqual(calibration["return_bias"][name], bias)
            assert_array_equal(result["means"][name + "_scaled"][inference], (mu + bias)[inference])
            assert_array_equal(result["means"][name + "_half"][inference],
                               .5 * expected_anchor + .5 * result["means"][name + "_scaled"][inference])

    def test_ridge_predictions_match_independent_scalar_evaluation(self):
        # The platform BLAS emits warnings even for small finite inputs; verify
        # every synthetic prediction without matrix multiplication or einsum.
        predict = self.masks["predict_mask"]
        for name in ("technical", "perp_delay0"):
            scaler, ridge = self.result["models"][name + "_mean"].steps[0][1], self.result["models"][name + "_mean"].steps[1][1]
            scalar = [float(ridge.intercept_) + math.fsum(
                ((float(v) - float(center)) / float(scale)) * float(weight)
                for v, center, scale, weight in zip(row, scaler.mean_, scaler.scale_, ridge.coef_))
                for row in self.groups[name][predict]]
            assert_allclose(scalar, self.result["raw_predictions"][name]["mu"][predict], rtol=1e-13, atol=1e-17)

    def test_evaluation_labels_are_never_converted_or_filter_inference(self):
        changed = self.outcomes.astype(object)
        selected = self.masks["fit_mask"] | self.masks["scale_mask"] | self.masks["interval_mask"]
        changed[~selected] = "unobserved future outcome"
        result = fit_frozen_forecasts(self.groups, changed, **self.masks)
        self.assert_same_predictions(result)
        self.assertEqual(result["calibration"], self.result["calibration"])
        assert_array_equal(result["masks"]["inference"], self.masks["inference_mask"])
        self.assertFalse(result["provenance"]["evaluation_labels_used"])

    def test_interval_outcomes_change_quantiles_but_not_forecasts_or_scale(self):
        changed = self.outcomes.copy()
        interval = self.masks["interval_mask"]
        changed[interval, 0] += .15
        changed[interval, 2] *= 4
        result = fit_frozen_forecasts(self.groups, changed, **self.masks)
        self.assert_same_predictions(result)
        for key in ("fit_mean", "scale_mean", "return_bias", "variance_multiplier", "counts"):
            self.assertEqual(result["calibration"][key], self.result["calibration"][key])
        self.assertGreater(result["calibration"]["technical_quantiles"]["scaled"]["return_quantile"],
                           self.result["calibration"]["technical_quantiles"]["scaled"]["return_quantile"])
        self.assertGreater(result["calibration"]["technical_quantiles"]["scaled"]["volatility_quantile"],
                           self.result["calibration"]["technical_quantiles"]["scaled"]["volatility_quantile"])

    def test_unused_features_ignored_and_inputs_remain_unchanged(self):
        groups, outcomes, masks = fixture()
        unused = ~(masks["fit_mask"] | masks["predict_mask"])
        for array in groups.values():
            array[unused] = np.inf
        snapshots = {key: array.copy() for key, array in groups.items()}
        original_y = outcomes.copy()
        original_masks = {name: mask.copy() for name, mask in masks.items()}
        for array in [*groups.values(), outcomes, *masks.values()]:
            array.setflags(write=False)
        result = fit_frozen_forecasts(groups, outcomes, **masks)
        self.assert_same_predictions(result)
        for name in groups:
            assert_array_equal(groups[name], snapshots[name])
        assert_array_equal(outcomes, original_y)
        for name in masks:
            assert_array_equal(masks[name], original_masks[name])

    def test_perp_features_cannot_change_shared_technical_variance(self):
        groups = {key: array.copy() for key, array in self.groups.items()}
        groups["perp_delay0"][:, 3] = np.where(self.masks["predict_mask"], 7., groups["perp_delay0"][:, 3])
        result = fit_frozen_forecasts(groups, self.outcomes, **self.masks)
        assert_array_equal(result["variance"], self.result["variance"])
        assert_array_equal(result["means"]["technical_scaled"], self.result["means"]["technical_scaled"])
        self.assertFalse(np.array_equal(result["raw_predictions"]["perp_delay0"]["mu"][self.masks["predict_mask"]],
                                        self.result["raw_predictions"]["perp_delay0"]["mu"][self.masks["predict_mask"]]))

    def test_masks_reject_wrong_dtype_shape_order_and_insufficient_support(self):
        variants = []
        for name in self.masks:
            variants.extend(((name, self.masks[name].astype(int)), (name, self.masks[name][:-1])))
        for name, minimum in (("fit_mask", 512), ("scale_mask", 64), ("interval_mask", 64)):
            mask = self.masks[name].copy()
            mask[np.flatnonzero(mask)[minimum - 1:]] = False
            variants.append((name, mask))
        overlap = self.masks["interval_mask"].copy(); overlap[599] = True
        variants.append(("interval_mask", overlap))
        early_inference = self.masks["inference_mask"].copy(); early_inference[679] = True
        variants.append(("inference_mask", early_inference))
        missing_predict = self.masks["predict_mask"].copy(); missing_predict[700] = False
        variants.append(("predict_mask", missing_predict))
        early_predict = self.masks["predict_mask"].copy(); early_predict[519] = True
        variants.append(("predict_mask", early_predict))
        for name, value in variants:
            with self.subTest(name=name, shape=value.shape, count=value.sum()):
                with self.assertRaises(ValueError):
                    fit_frozen_forecasts(self.groups, self.outcomes, **{**self.masks, name: value})

    def test_selected_invalid_values_raise_instead_of_reducing_support(self):
        for index in (10, 550, 650, 700):
            with self.subTest(feature_row=index):
                groups = {key: array.copy() for key, array in self.groups.items()}
                groups["technical"][index, 0] = np.nan
                with self.assertRaises(ValueError):
                    fit_frozen_forecasts(groups, self.outcomes, **self.masks)
        for index, column, value in ((10, 0, np.nan), (550, 2, -1.), (650, 1, np.inf), (10, 2, 1e308)):
            with self.subTest(label_row=index, column=column):
                y = self.outcomes.copy(); y[index, column] = value
                with self.assertRaises(ValueError):
                    fit_frozen_forecasts(self.groups, y, **self.masks)
        with self.assertRaises(ValueError):
            fit_frozen_forecasts(self.groups, self.outcomes[:, :2], **self.masks)
        with self.assertRaises(ValueError):
            fit_frozen_forecasts({"technical": self.groups["technical"]}, self.outcomes, **self.masks)

    def test_complex_inputs_rejected_without_discarding_imaginary_parts(self):
        for dtype in (complex, object):
            groups = {name: values.astype(dtype) for name, values in self.groups.items()}
            groups["technical"][10, 0] = 1 + 2j
            with self.assertRaises(ValueError):
                fit_frozen_forecasts(groups, self.outcomes, **self.masks)
            y = self.outcomes.astype(dtype)
            y[550, 0] = 1 + 2j
            with self.assertRaises(ValueError):
                fit_frozen_forecasts(self.groups, y, **self.masks)

    def test_dataframe_calendar_alignment_and_named_provenance(self):
        index = pd.date_range("2000-01-01", periods=800, freq="15min", tz="UTC")
        groups = {name: pd.DataFrame(array, index=index) for name, array in self.groups.items()}
        y = pd.DataFrame(self.outcomes, index=index)
        result = fit_frozen_forecasts(groups, y, **self.masks)
        self.assert_same_predictions(result)
        json.dumps(result["provenance"], allow_nan=False)
        groups["perp_delay0"].index = index + pd.Timedelta(minutes=15)
        with self.assertRaises(ValueError):
            fit_frozen_forecasts(groups, y, **self.masks)
        groups["perp_delay0"].index = index[::-1]
        with self.assertRaises(ValueError):
            fit_frozen_forecasts(groups, y, **self.masks)

    def test_empty_inference_still_calibrates_without_inventing_support(self):
        masks = {**self.masks, "inference_mask": np.zeros(800, bool)}
        result = fit_frozen_forecasts(self.groups, self.outcomes, **masks)
        self.assertEqual(result["calibration"], self.result["calibration"])
        for array in [*result["means"].values(), result["variance"]]:
            self.assertTrue(np.isnan(array).all())
        self.assertIsNone(result["provenance"]["mask_ranges"]["inference"])


if __name__ == "__main__":
    unittest.main()
