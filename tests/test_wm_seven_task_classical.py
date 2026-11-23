import inspect
import json
import pickle
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from unidream.experiments import wm_seven_task_classical as classical


def examples(n=220):
    rng = np.random.default_rng(2041)
    x = rng.normal(size=(n, 7)).astype(np.float32)
    x[:, -1] = 3.  # A constant input must not create NaNs in the scaler.
    movement = .01 * (.7 * x[:, 0] - .3 * x[:, 1] + rng.normal(scale=.3, size=n))
    variance = .0001 * np.exp(.5 * x[:, 1] + rng.normal(scale=.1, size=n))
    downside = np.maximum(-movement, 0.) + .001
    targets = {
        "return": movement, "variance": variance, "downside": downside,
        "crash": (movement < -.006).astype(float),
        "dd_excess": np.maximum(downside - .004, 0.),
        "advantage": .5 * movement - .1 * downside,
        "recovery": np.tanh(x[:, 2] - .2 * x[:, 0]),
    }
    return x, targets


SMALL = {
    "cpu_threads": 1, "linear_max_train_rows": 100,
    "ridge_alphas": [1., 50., 1000.], "logistic_Cs": [.01, .1],
    "lightgbm_leaves": [3, 7], "lightgbm_rounds": 14,
    "lightgbm_patience": 4, "lightgbm_min_data_leaf": 5,
    "lightgbm_max_bin": 15, "lightgbm_learning_rate": .1,
}


class SevenTaskClassicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.targets = examples()
        cls.train = {task: y[:140] for task, y in cls.targets.items()}
        cls.val = {task: y[140:200] for task, y in cls.targets.items()}
        cls.original = cls.x.copy()
        cls.bundle = classical.fit_classical_bundle(cls.x[:140], cls.train,
                                                     cls.x[140:200], cls.val, SMALL)

    def test_fit_boundary_has_no_test_argument_and_scaler_uses_all_train_only(self):
        self.assertEqual(tuple(inspect.signature(classical.fit_classical_bundle).parameters),
                         ("x_train", "targets_train", "x_val", "targets_val", "config"))
        np.testing.assert_array_equal(self.x, self.original)
        np.testing.assert_allclose(self.bundle["scaler"]["mean"], self.x[:140].mean(0, dtype=np.float64))
        expected_std = self.x[:140].std(0, dtype=np.float64)
        expected_std[expected_std < 1e-12] = 1.
        np.testing.assert_allclose(self.bundle["scaler"]["scale"], expected_std)
        metadata = self.bundle["metadata"]
        self.assertFalse(metadata["test_predictions_produced"])
        self.assertEqual(metadata["selection_split"], "validation")
        self.assertEqual(len(metadata["linear_train_rows"]), 100)
        self.assertEqual(len(metadata["lightgbm_train_rows"]), 140)

    def test_regression_candidates_match_independently_fitted_ridge_on_train_subset(self):
        scaler = self.bundle["scaler"]
        rows = self.bundle["metadata"]["linear_train_rows"]
        xs = (self.x[:140][rows].astype(float) - scaler["mean"]) / scaler["scale"]
        xv = (self.x[140:200].astype(float) - scaler["mean"]) / scaler["scale"]
        scale = self.bundle["metadata"]["train_scales"]["return"]
        candidates = self.bundle["metadata"]["models"]["linear"]["return"]["candidates"]
        independently_scored = []
        with threadpool_limits(limits=1):
            for row in candidates:
                model = Ridge(alpha=row["alpha"], solver="cholesky")
                model.fit(xs, self.train["return"][rows] / scale)
                prediction = model.predict(xv) * scale
                expected = float(np.mean(np.square((self.val["return"] - prediction) / scale)))
                self.assertAlmostEqual(row["score"], expected, places=11)
                independently_scored.append((expected, row["alpha"]))
        self.assertEqual(self.bundle["metadata"]["models"]["linear"]["return"]["selected"]["alpha"],
                         min(independently_scored)[1])

    def test_all_predictions_respect_probability_variance_and_recovery_domains(self):
        predictions = classical.predict_classical_bundle(self.bundle, self.x[200:])
        self.assertEqual(set(predictions), {"unconditional", "linear", "lightgbm"})
        for heads in predictions.values():
            self.assertEqual(set(heads), set(classical.TASKS))
            for task, vector in heads.items():
                self.assertEqual(vector.shape, (20,))
                self.assertTrue(np.isfinite(vector).all())
                if task == "variance":
                    self.assertTrue((vector >= 1e-12).all())
                if task in ("downside", "dd_excess", "crash"):
                    self.assertTrue((vector >= 0).all())
                if task == "crash":
                    self.assertTrue((vector <= 1).all())
                if task == "recovery":
                    self.assertTrue((np.abs(vector) <= 1).all())
        np.testing.assert_array_equal(predictions["unconditional"]["return"], 0.)
        np.testing.assert_array_equal(predictions["unconditional"]["advantage"], 0.)
        np.testing.assert_allclose(predictions["unconditional"]["crash"], self.train["crash"].mean())

    def test_selected_tree_scores_reproduce_with_explicit_saved_iterations(self):
        predictions = classical.predict_classical_bundle(self.bundle, self.x[140:200])["lightgbm"]
        for task in classical.TASKS:
            metadata = self.bundle["metadata"]["models"]["lightgbm"][task]
            selected = metadata["selected"]
            self.assertAlmostEqual(selected["score"], min(row["score"] for row in metadata["candidates"]))
            self.assertEqual(selected["selected_iteration"], int(np.argmin(selected["history"])) + 1)
            score = classical.task_selection_score(task, self.val[task], predictions[task],
                                                    self.bundle["metadata"]["train_scales"][task])
            self.assertAlmostEqual(score, selected["score"], places=11)

    def test_bundle_pickle_preserves_all_selected_predictions(self):
        before = classical.predict_classical_bundle(self.bundle, self.x[200:])
        restored = pickle.loads(pickle.dumps(self.bundle))
        after = classical.predict_classical_bundle(restored, self.x[200:])
        for model in before:
            for task in classical.TASKS:
                np.testing.assert_array_equal(before[model][task], after[model][task])

    def test_constant_crash_training_is_explicit_and_evaluable_when_val_has_events(self):
        x, targets = examples(90)
        train = {task: values[:60].copy() for task, values in targets.items()}
        val = {task: values[60:].copy() for task, values in targets.items()}
        train["crash"][:] = 0.
        val["crash"][:5] = 1.
        bundle = classical.fit_classical_bundle(x[:60], train, x[60:], val,
                                                {**SMALL, "lightgbm_leaves": [3], "ridge_alphas": [1.]})
        predicted = classical.predict_classical_bundle(bundle, x[60:])
        for name in ("linear", "lightgbm"):
            np.testing.assert_array_equal(predicted[name]["crash"], 0.)
            info = bundle["metadata"]["models"][name]["crash"]
            self.assertTrue(info["constant_train_class"])
            self.assertAlmostEqual(info["selected"]["score"], val["crash"].mean())

    def test_invalid_targets_features_and_prediction_width_fail_before_fitting(self):
        for task, bad in (("variance", -1.), ("downside", -1.), ("crash", .5), ("recovery", 1.1)):
            targets = {name: vector.copy() for name, vector in self.train.items()}
            targets[task][0] = bad
            with self.assertRaises(ValueError):
                classical.fit_classical_bundle(self.x[:140], targets, self.x[140:200], self.val, SMALL)
        bad_features = self.x[:140].copy()
        bad_features[0, 0] = np.nan
        with self.assertRaises(ValueError):
            classical.fit_classical_bundle(bad_features, self.train, self.x[140:200], self.val, SMALL)
        with self.assertRaises(ValueError):
            classical.predict_classical_bundle(self.bundle, self.x[200:, :-1])

    def test_validation_score_uses_variance_qlike_in_variance_units(self):
        actual = np.array([.0001, .0004])
        predicted = np.array([.0002, .0001])
        ratio = actual / predicted
        expected = np.mean(ratio - np.log(ratio) - 1.)
        self.assertAlmostEqual(classical.task_selection_score("variance", actual, predicted, .0003), expected)
        zero = classical.task_selection_score("variance", np.array([0.]), np.array([0.]), 1e-12)
        self.assertEqual(zero, 0.)


def variance_examples():
    rng = np.random.default_rng(81)
    close = 100. * np.exp(np.r_[0., np.cumsum(rng.normal(0., .001, 499))])
    timestamps = np.arange(len(close), dtype=np.int64) * (15 * 60 * 1_000_000_000)
    ids = np.array([320, 340, 360])
    actual = np.array([1.6e-5, 2.e-5, 1.2e-5])
    garch = classical.GARCHFit(
        omega=1e-8, alpha=.08, beta=.9, initial_variance=1e-6,
        return_scale=100., success=True, optimizer_success=True, status=0,
        message="deterministic fixture", nobs=289, n_segments=1, iterations=1,
        function_evaluations=1, optimizer_objective_scaled=0.,
        mean_negative_log_likelihood=0., gradient_norm=0., constraints_valid=True)
    return close, timestamps, ids, actual, garch


class SevenTaskVarianceFiltersTests(unittest.TestCase):
    def test_train_return_slice_excludes_start_boundary_and_ewma_selects_supplied_val_only(self):
        close, timestamps, ids, actual, garch = variance_examples()
        captured = []

        def fit(values, **kwargs):
            captured.append((values.copy(), kwargs.copy()))
            return garch

        # A price jump before the Train interval must not enter its first return.
        close[:10] *= 100.
        with patch.object(classical, "fit_gap_reset_garch", side_effect=fit):
            fitted = classical.fit_variance_filters(close, timestamps, timestamps[10], timestamps[300],
                                                      ids, actual, context=64)
        expected_train = np.r_[np.nan, np.diff(np.log(close[10:300]))]
        np.testing.assert_allclose(captured[0][0], expected_train, rtol=0., atol=1e-15, equal_nan=True)
        expected_initial = float(np.nanmean(np.square(expected_train)))
        self.assertAlmostEqual(captured[0][1]["initial_variance"], expected_initial, places=18)
        self.assertEqual(captured[0][1]["maxiter"], 500)
        independent = []
        for decay in (.94, .97, .99):
            forecast = classical.ewma_variance_forecast(close, ids, initial_variance=expected_initial,
                                                        decay=decay, context=64, horizon=16)
            ratio = actual / forecast
            independent.append((float(np.mean(ratio - np.log(ratio) - 1.)), decay))
        self.assertEqual(fitted["ewma_decay"], min(independent)[1])
        self.assertAlmostEqual(fitted["ewma_selection_score"], min(independent)[0])
        self.assertFalse(fitted["metadata"]["test_labels_used"])
        self.assertEqual(fitted["metadata"]["train_return_n"], 289)
        json.dumps(fitted, allow_nan=False)

    def test_all_filters_ignore_future_and_history_before_common_context(self):
        close, timestamps, ids, actual, garch = variance_examples()
        with patch.object(classical, "fit_gap_reset_garch", return_value=garch):
            fit = classical.fit_variance_filters(close, timestamps, timestamps[10], timestamps[300],
                                                  ids, actual, context=64, horizon=4)
        before = classical.predict_variance_filters(close, [400], fit)
        changed = close.copy()
        changed[:336] *= 50.
        changed[400:] *= 50.
        after = classical.predict_variance_filters(changed, [400], pickle.loads(pickle.dumps(fit)))
        self.assertEqual(set(before), {"past_variance", "ewma", "garch"})
        for model in before:
            np.testing.assert_array_equal(before[model], after[model])
        trailing_returns = np.diff(np.log(close[383:400]))
        expected = np.sum(np.square(trailing_returns)) * 4. / 16.
        np.testing.assert_allclose(before["past_variance"], [expected], rtol=0., atol=1e-18)

    def test_failed_garch_keeps_failure_metadata_and_omits_prediction_without_fallback(self):
        close, timestamps, ids, actual, garch = variance_examples()
        failed = replace(garch, success=False, optimizer_success=False, status=1,
                         message="fixture iteration limit", iterations=500)
        with patch.object(classical, "fit_gap_reset_garch", return_value=failed):
            fit = classical.fit_variance_filters(close, timestamps, timestamps[10], timestamps[300],
                                                  ids, actual, context=64)
        self.assertFalse(fit["garch_success"])
        self.assertEqual(fit["garch"]["message"], "fixture iteration limit")
        self.assertEqual(fit["garch"]["iterations"], 500)
        self.assertIsNone(fit["metadata"]["failed_garch_fallback"])
        predicted = classical.predict_variance_filters(close, [400], fit)
        self.assertEqual(set(predicted), {"past_variance", "ewma"})

    def test_missing_price_keeps_common_context_unavailable_for_every_filter(self):
        close, timestamps, ids, actual, garch = variance_examples()
        with patch.object(classical, "fit_gap_reset_garch", return_value=garch):
            fit = classical.fit_variance_filters(close, timestamps, timestamps[10], timestamps[300],
                                                  ids, actual, context=64)
        close[350] = np.nan  # Within context, outside the past-16 return slice.
        predicted = classical.predict_variance_filters(close, [400], fit)
        self.assertTrue(all(np.isnan(vector).all() for vector in predicted.values()))
        with self.assertRaisesRegex(ValueError, "complete finite price context"):
            classical.fit_variance_filters(close, timestamps, timestamps[10], timestamps[300],
                                            ids, actual, context=64)

    def test_train_overlap_in_selection_and_compressed_time_grid_are_rejected(self):
        close, timestamps, ids, actual, _ = variance_examples()
        with self.assertRaisesRegex(ValueError, "inside Train"):
            classical.fit_variance_filters(close, timestamps, timestamps[10], timestamps[350],
                                            ids, actual, context=64)
        broken_timestamps = timestamps.copy()
        broken_timestamps[200:] += 15 * 60 * 1_000_000_000
        with self.assertRaisesRegex(ValueError, "regular 15-minute grid"):
            classical.fit_variance_filters(close, broken_timestamps, timestamps[10], timestamps[300],
                                            ids, actual, context=64)


if __name__ == "__main__":
    unittest.main()
