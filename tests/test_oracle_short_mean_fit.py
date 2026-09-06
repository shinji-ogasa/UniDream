"""Synthetic-only tests of fixed mean fitting and selected-value boundaries."""
import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from unidream.experiments.oracle_short_mean_fit import GROUPS, fit_raw_mean_family


def fixture():
    rng = np.random.default_rng(815)
    n = 608
    index = pd.date_range("2000-01-01", periods=n, freq="15min", tz="UTC")
    base = rng.normal(size=(n, 3))
    price = rng.normal(size=(n, 2))
    flow = rng.normal(size=(n, 1))
    groups = {
        "technical": pd.DataFrame(base, index=index, columns=["z", "a", "constant"]),
        "technical_short_price": pd.DataFrame(np.column_stack((base, price)), index=index,
                                               columns=["z", "a", "constant", "r4", "body"]),
        "technical_short_flow": pd.DataFrame(np.column_stack((base, flow)), index=index,
                                              columns=["z", "a", "constant", "flow4"]),
        "technical_short_both": pd.DataFrame(np.column_stack((base, price, flow)), index=index,
                                              columns=["z", "a", "constant", "r4", "body", "flow4"]),
    }
    for frame in groups.values():
        frame["constant"] = 1.
    returns = .003 * base[:, 0] - .002 * base[:, 1] + .001 * price[:, 0] + .0002
    outcomes = pd.DataFrame(np.column_stack((returns, np.abs(returns), np.full(n, .01))),
                            index=index, columns=["return", "adverse", "volatility"])
    fit = np.arange(n) < 512
    predict = (np.arange(n) >= 528) & (np.arange(n) < 592)
    return groups, outcomes, {"fit_mask": fit, "predict_mask": predict}


class ShortMeanFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.outcomes, cls.masks = fixture()
        cls.result = fit_raw_mean_family(cls.groups, cls.outcomes, **cls.masks)

    def assert_same(self, result):
        for name in GROUPS:
            assert_array_equal(result["raw"][name], self.result["raw"][name])
            for step in ("standardscaler", "ridge"):
                attrs = ("mean_", "var_", "scale_") if step == "standardscaler" else ("coef_", "intercept_")
                for attr in attrs:
                    assert_array_equal(getattr(result["models"][name][step], attr),
                                       getattr(self.result["models"][name][step], attr))
        self.assertEqual(result["fit_return_mean"], self.result["fit_return_mean"])
        self.assertEqual(result["provenance"], self.result["provenance"])

    def test_exact_inventory_support_parameters_and_provenance(self):
        result = self.result
        self.assertEqual(tuple(result["models"]), GROUPS)
        self.assertEqual(tuple(result["raw"]), GROUPS)
        self.assertEqual(result["fit_return_mean"],
                         float(self.outcomes.to_numpy()[self.masks["fit_mask"], 0].mean()))
        self.assertEqual(result["provenance"]["mask_counts"], {"fit": 512, "predict": 64})
        self.assertFalse(result["provenance"]["evaluation_labels_used"])
        self.assertFalse(result["provenance"]["risk_or_calibration_fitted"])
        self.assertFalse(result["provenance"]["timestamp_feature_causality_and_label_completion_verified"])
        for name in GROUPS:
            pipeline = result["models"][name]
            self.assertEqual(len(pipeline.steps), 2)
            self.assertEqual(pipeline["ridge"].get_params(), Ridge(alpha=100.).get_params())
            self.assertEqual(pipeline["standardscaler"].get_params(), StandardScaler().get_params())
            mu = result["raw"][name]
            self.assertEqual(mu.shape, (608,))
            self.assertEqual(mu.dtype, np.dtype("float64"))
            self.assertTrue(np.isfinite(mu[self.masks["predict_mask"]]).all())
            self.assertTrue(np.isnan(mu[~self.masks["predict_mask"]]).all())
            self.assertEqual(result["provenance"]["feature_columns"][name], list(self.groups[name].columns))
        for name in ("fit", "predict"):
            mask = self.masks[name + "_mask"]
            assert_array_equal(mask, result["masks"][name])
            self.assertFalse(np.shares_memory(mask, result["masks"][name]))
            digest = hashlib.sha256(np.asarray([608], "<i8").tobytes() + mask.astype("u1").tobytes()).hexdigest()
            self.assertEqual(result["provenance"]["mask_position_sha256"][name], digest)
        json.dumps(result["provenance"], allow_nan=False)

    def test_exact_old_direct_pipeline_and_independent_scalar_parity(self):
        fit, predict = self.masks["fit_mask"], self.masks["predict_mask"]
        y = self.outcomes.to_numpy()
        with threadpool_limits(limits=2):
            for name in GROUPS:
                x = np.asarray(self.groups[name].to_numpy(), float)
                direct = make_pipeline(StandardScaler(), Ridge(alpha=100.))
                direct.fit(x[fit], y[fit, 0])
                assert_array_equal(self.result["raw"][name][predict], direct.predict(x[predict]))
                scaler, ridge = direct["standardscaler"], direct["ridge"]
                scalar = [float(ridge.intercept_) + math.fsum(
                    (float(v) - float(center)) / float(scale) * float(weight)
                    for v, center, scale, weight in zip(row, scaler.mean_, scaler.scale_, ridge.coef_))
                    for row in x[predict]]
                assert_allclose(self.result["raw"][name][predict], scalar, rtol=1e-13, atol=1e-17)

    def test_normal_equation_without_using_fitted_coefficients(self):
        fit, predict = self.masks["fit_mask"], self.masks["predict_mask"]
        x = self.groups["technical"].to_numpy()
        y = self.outcomes.to_numpy()[fit, 0]
        center, scale = x[fit].mean(axis=0), x[fit].std(axis=0)
        scale[scale == 0] = 1.
        z = (x[fit] - center) / scale
        zcenter = z.mean(axis=0)
        z -= zcenter
        with threadpool_limits(limits=2):
            coefficient = np.linalg.solve(z.T @ z + 100. * np.eye(x.shape[1]), z.T @ (y - y.mean()))
            expected = ((x[predict] - center) / scale - zcenter) @ coefficient + y.mean()
        assert_allclose(self.result["raw"]["technical"][predict], expected, rtol=1e-13, atol=1e-17)

    def test_unselected_labels_and_feature_poison_are_invariant(self):
        groups = {name: frame.astype(object) for name, frame in self.groups.items()}
        outcomes = self.outcomes.astype(object)
        fit, predict = self.masks["fit_mask"], self.masks["predict_mask"]
        outcomes.iloc[~fit, 0] = "unobserved future"
        outcomes.iloc[:, 1] = complex(1, 2)
        outcomes.iloc[:, 2] = True
        for frame in groups.values():
            frame.iloc[~(fit | predict), :] = "unavailable future feature"
            frame.iloc[-1, 0] = complex(1, 3)
        self.assert_same(fit_raw_mean_family(groups, outcomes, **self.masks))
        self.assert_same(fit_raw_mean_family(groups, outcomes.to_numpy(), **self.masks))

    def test_inputs_unchanged_and_return_mask_copies(self):
        groups, outcomes, masks = fixture()
        old_groups = {name: frame.copy(deep=True) for name, frame in groups.items()}
        old_outcomes = outcomes.copy(deep=True)
        old_masks = {name: mask.copy() for name, mask in masks.items()}
        for mask in masks.values():
            mask.setflags(write=False)
        result = fit_raw_mean_family(groups, outcomes, **masks)
        for name in GROUPS:
            pd.testing.assert_frame_equal(groups[name], old_groups[name])
        pd.testing.assert_frame_equal(outcomes, old_outcomes)
        result["masks"]["fit"][:] = False
        for name in masks:
            assert_array_equal(masks[name], old_masks[name])

    def test_minimum_counts_and_strict_ordered_masks(self):
        cases = []
        short = self.masks["fit_mask"].copy()
        short[511] = False
        cases.append({**self.masks, "fit_mask": short})
        cases.append({**self.masks, "predict_mask": np.zeros(608, bool)})
        overlap = self.masks["predict_mask"].copy()
        overlap[511] = True
        cases.append({**self.masks, "predict_mask": overlap})
        interleaved_fit = self.masks["fit_mask"].copy()
        interleaved_fit[550] = True
        cases.append({**self.masks, "fit_mask": interleaved_fit})
        for name in ("fit_mask", "predict_mask"):
            for bad in (self.masks[name].astype(int), self.masks[name].astype(object),
                        self.masks[name][:-1], self.masks[name][:, None]):
                cases.append({**self.masks, name: bad})
        for masks in cases:
            with self.subTest(masks={k: (v.shape, v.dtype, int(v.sum())) for k, v in masks.items()}):
                with self.assertRaises(ValueError):
                    fit_raw_mean_family(self.groups, self.outcomes, **masks)

    def test_bad_indices_rejected_without_fitting(self):
        bad_indices = [self.groups["technical"].index[::-1],
                       pd.Index([0] * 608), pd.Index([*range(607), np.nan]),
                       pd.MultiIndex.from_arrays([np.arange(608), np.zeros(608)])]
        shifted = self.groups["technical"].index + pd.Timedelta(minutes=15)
        bad_indices.append(shifted)
        for index in bad_indices:
            groups = {name: frame.copy() for name, frame in self.groups.items()}
            groups["technical_short_both"].index = index
            with self.subTest(index=type(index).__name__):
                with self.assertRaises(ValueError), patch(
                        "unidream.experiments.oracle_short_mean_fit.make_pipeline") as make:
                    fit_raw_mean_family(groups, self.outcomes, **self.masks)
                make.assert_not_called()
        outcomes = self.outcomes.copy()
        outcomes.index = shifted
        with self.assertRaises(ValueError):
            fit_raw_mean_family(self.groups, outcomes, **self.masks)

    def test_group_inventory_frame_and_column_errors(self):
        cases = [{key: value for key, value in self.groups.items() if key != "technical"},
                 {**self.groups, "extra": self.groups["technical"]}]
        frame = self.groups["technical"]
        for bad in (frame.to_numpy(), frame.iloc[:-1], frame.iloc[:, :0],
                    frame.set_axis(["same"] * 3, axis=1),
                    frame.set_axis(["z", "", "c"], axis=1),
                    frame.set_axis(["z", 4, "c"], axis=1)):
            cases.append({**self.groups, "technical": bad})
        for groups in cases:
            with self.assertRaises(ValueError):
                fit_raw_mean_family(groups, self.outcomes, **self.masks)

    def test_selected_bool_complex_string_and_nonfinite_rejected(self):
        bad_values = (True, np.bool_(False), complex(1., 0.), "0.1", None, pd.NA, np.nan, np.inf, -np.inf)
        for bad in bad_values:
            for position in (0, 530):
                groups = {name: frame.astype(object) for name, frame in self.groups.items()}
                groups["technical_short_both"].iloc[position, 0] = bad
                with self.subTest(feature=repr(bad), row=position), self.assertRaises(ValueError):
                    fit_raw_mean_family(groups, self.outcomes, **self.masks)
            outcomes = self.outcomes.astype(object)
            outcomes.iloc[0, 0] = bad
            with self.subTest(return_value=repr(bad)), self.assertRaises(ValueError):
                fit_raw_mean_family(self.groups, outcomes, **self.masks)

    def test_malformed_outcome_shapes_and_types(self):
        for y in (np.zeros(608), np.zeros((608, 2)), np.zeros((0, 3)),
                  np.zeros((608, 3, 1)), self.outcomes.to_numpy().tolist()):
            with self.assertRaises(ValueError):
                fit_raw_mean_family(self.groups, y, **self.masks)

    def test_selected_hashes_match_independent_encoding_and_changes(self):
        fit, predict = self.masks["fit_mask"], self.masks["predict_mask"]

        def digest(a):
            a = np.asarray(a, dtype="<f8", order="C")
            return hashlib.sha256(np.asarray([a.ndim, *a.shape], "<i8").tobytes() +
                                  a.tobytes(order="C")).hexdigest()

        p = self.result["provenance"]
        y = self.outcomes.to_numpy()[fit, 0]
        self.assertEqual(p["fit_return_sha256"], digest(y))
        for name in GROUPS:
            x = self.groups[name].to_numpy()
            self.assertEqual(p["fit_features_sha256"][name], digest(x[fit]))
            self.assertEqual(p["predict_features_sha256"][name], digest(x[predict]))
            self.assertEqual(p["fit_features_and_return_sha256"][name],
                             digest(np.column_stack((x[fit], y))))
        groups = {name: frame.copy() for name, frame in self.groups.items()}
        groups["technical_short_both"].iloc[530, 0] += 1.
        changed = fit_raw_mean_family(groups, self.outcomes, **self.masks)
        self.assertNotEqual(changed["provenance"]["predict_features_sha256"]["technical_short_both"],
                            p["predict_features_sha256"]["technical_short_both"])
        self.assertEqual(changed["provenance"]["fit_features_sha256"], p["fit_features_sha256"])
        for name in GROUPS[:-1]:
            assert_array_equal(changed["raw"][name], self.result["raw"][name])

    def test_thread_limit_and_nonfinite_model_prediction_guard(self):
        with patch("unidream.experiments.oracle_short_mean_fit.threadpool_limits",
                   wraps=threadpool_limits) as limits:
            self.assert_same(fit_raw_mean_family(self.groups, self.outcomes, **self.masks))
        limits.assert_called_once_with(limits=2)
        with patch("sklearn.pipeline.Pipeline.predict", return_value=np.full(64, np.nan)):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                fit_raw_mean_family(self.groups, self.outcomes, **self.masks)


if __name__ == "__main__":
    unittest.main()
