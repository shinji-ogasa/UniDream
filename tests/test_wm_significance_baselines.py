"""Causality and Train-only contracts for the extra significance comparators."""
import copy
import unittest

import numpy as np

from unidream.experiments.wm_comparison_classical import ewma_variance_forecast
from unidream.experiments.wm_full_grid_evaluation import make_task_labels
from unidream.experiments.wm_seven_task_evaluation import TASKS
from unidream.experiments.wm_significance_baselines import (
    causal_features, fit_baselines, predict_baselines,
)


class SignificanceBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20261112)
        n = 240
        close = 100 * np.exp(np.cumsum(rng.normal(0, .002, n)))
        cls.arrays = {"raw": np.column_stack([close] * 5)}
        cls.train = np.arange(40, 160)
        cls.test = np.arange(175, 185)
        values = {
            "return": rng.normal(0, .012, n),
            "variance": np.exp(rng.normal(-9, 1, n)),
            "downside": rng.uniform(0, .02, n),
            "crash": (np.arange(n) % 5 == 0).astype(float),
            "dd_excess": np.resize([0, 0, 0, .003, .012], n),
            "advantage": rng.normal(0, .002, n),
            "recovery": rng.uniform(-1, 1, n),
        }
        cls.arrays.update({task: values[task][:, None] for task in TASKS})
        cls.contract = {}
        for task in TASKS:
            if task in ("return", "advantage", "recovery"):
                bound = .5 * np.sqrt(np.mean(values[task][cls.train] ** 2))
                entry = {"kind": "signed_rms", "bounds": [-bound, bound], "n_classes": 3}
            elif task in ("variance", "downside"):
                entry = {"kind": "train_tertiles", "bounds": np.quantile(values[task][cls.train], [1/3, 2/3]).tolist(), "n_classes": 3}
            elif task == "dd_excess":
                entry = {"kind": "zero_positive_median", "bounds": [0., .006], "n_classes": 3}
            else:
                entry = {"kind": "binary_event", "bounds": [.5], "n_classes": 2}
            label = make_task_labels(values[task][cls.train], task, entry)
            entry["priors"] = (np.bincount(label, minlength=entry["n_classes"]) / len(cls.train)).tolist()
            cls.contract[task] = entry
        cls.filter = {"format": "seven_task_variance_filters_v1", "context": 32, "horizon": 16,
            "one_bar_variance": 1e-5, "ewma_decay": .97, "metadata": {"test_labels_used": False}}
        cls.bundle = cls.fit(cls.arrays)

    @classmethod
    def fit(cls, arrays, **kwargs):
        return fit_baselines(arrays, cls.train, cls.contract, cls.filter, context=32, fold=0, **kwargs)

    def test_future_prices_do_not_change_decision_features_or_predictions(self):
        ids = self.test[:1]
        changed = copy.deepcopy(self.arrays)
        changed["raw"][ids[0]:, 3] *= 1000
        original = predict_baselines(self.bundle, self.arrays, ids, fold=0)
        after = predict_baselines(self.bundle, changed, ids, fold=0)
        for name in original:
            for task in TASKS:
                for key in ("scalar", "predicted_class", "probabilities"):
                    if original[name][task][key] is not None:
                        np.testing.assert_array_equal(original[name][task][key], after[name][task][key])

    def test_unavailable_decision_close_is_not_observed(self):
        ids = self.test[:1]
        changed = copy.deepcopy(self.arrays)
        changed["raw"][ids[0], 3] = np.nan
        p = self.bundle["filter_parameters"]
        np.testing.assert_array_equal(causal_features(self.arrays, ids, **p), causal_features(changed, ids, **p))

    def test_nontrain_targets_do_not_change_fit(self):
        changed = copy.deepcopy(self.arrays)
        for task in TASKS:
            changed[task][self.train[-1] + 1:] = np.nan
        fit = self.fit(changed)
        np.testing.assert_array_equal(fit["scaler"].mean_, self.bundle["scaler"].mean_)
        np.testing.assert_array_equal(fit["ridge_model"].coef_, self.bundle["ridge_model"].coef_)
        for task in TASKS:
            np.testing.assert_array_equal(fit["logistic_models"][task].coef_, self.bundle["logistic_models"][task].coef_)

    def test_prediction_does_not_require_any_future_target_arrays(self):
        output = predict_baselines(self.bundle, {"raw": self.arrays["raw"]}, self.test)
        self.assertEqual(set(output), {"causal_simple", "train_constant", "lowdim_ridge"})
        self.assertEqual(set(output["causal_simple"]), set(TASKS))

    def test_scaler_uses_all_and_only_train_features(self):
        features = causal_features(self.arrays, self.train, **self.bundle["filter_parameters"])
        np.testing.assert_allclose(self.bundle["scaler"].mean_, features.mean(0), atol=1e-14)
        self.assertEqual(self.bundle["metadata"]["n_train"], len(self.train))
        self.assertFalse(self.bundle["metadata"]["test_labels_used"])
        self.assertEqual(self.bundle["metadata"]["logistic_C"], 1)
        self.assertEqual(self.bundle["metadata"]["ridge_alpha"], 1)

    def test_primary_point_comparators_match_fixed_mapping(self):
        output = predict_baselines(self.bundle, self.arrays, self.test)["causal_simple"]
        np.testing.assert_array_equal(output["return"]["scalar"], 0)
        for task in ("downside", "dd_excess"):
            np.testing.assert_array_equal(output[task]["scalar"], np.median(self.arrays[task][self.train, 0]))
        expected = ewma_variance_forecast(self.arrays["raw"][:, 3], self.test,
            initial_variance=1e-5, decay=.97, context=32, horizon=16)
        np.testing.assert_allclose(output["variance"]["scalar"], expected, rtol=1e-14)
        np.testing.assert_array_equal(output["crash"]["scalar"], output["crash"]["probabilities"][:, 1])
        self.assertTrue(np.all(np.abs(output["recovery"]["scalar"]) <= 1))

    def test_class_rule_uses_train_prior_adjusted_probabilities(self):
        output = predict_baselines(self.bundle, self.arrays, self.test)
        for task in TASKS:
            item = output["causal_simple"][task]
            np.testing.assert_array_equal(item["predicted_class"], np.argmax(item["probabilities"] / self.contract[task]["priors"], axis=1))
            np.testing.assert_allclose(item["probabilities"].sum(1), 1)
            self.assertEqual(item["probabilities"].shape[1], 2 if task == "crash" else 3)
            np.testing.assert_array_equal(output["train_constant"][task]["predicted_class"], 0)

    def test_small_feature_einsum_predictions_match_fitted_estimators(self):
        output = predict_baselines(self.bundle, self.arrays, self.test)
        features = causal_features(self.arrays, self.test, **self.bundle["filter_parameters"])
        x = self.bundle["scaler"].transform(features)
        ridge = self.bundle["ridge_model"].predict(x)
        for column, task in enumerate(TASKS):
            np.testing.assert_allclose(output["causal_simple"][task]["probabilities"],
                self.bundle["logistic_models"][task].predict_proba(x), rtol=1e-13, atol=1e-14)
            expected = ridge[:, column]
            if task in ("variance", "downside", "dd_excess"):
                expected = np.maximum(expected, 1e-12 if task == "variance" else 0)
            if task == "crash":
                expected = np.clip(expected, 0, 1)
            if task == "recovery":
                expected = np.clip(expected, -1, 1)
            np.testing.assert_allclose(output["lowdim_ridge"][task]["scalar"], expected, rtol=1e-13, atol=1e-14)

    def test_future_nan_targets_do_not_remove_rows(self):
        changed = copy.deepcopy(self.arrays)
        for task in TASKS:
            changed[task][self.test] = np.nan
        output = predict_baselines(self.bundle, changed, self.test)
        for task in TASKS:
            self.assertEqual(len(output["causal_simple"][task]["scalar"]), len(self.test))

    def test_missing_observed_history_is_explicit_failure(self):
        changed = copy.deepcopy(self.arrays)
        changed["raw"][self.test[0] - 20, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "incomplete or nonfinite"):
            predict_baselines(self.bundle, changed, self.test)

    def test_wrong_fold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "different fold"):
            predict_baselines(self.bundle, self.arrays, self.test, fold=1)

    def test_wrong_contract_train_prior_is_rejected(self):
        contract = copy.deepcopy(self.contract)
        contract["return"]["priors"] = [1/3] * 3
        with self.assertRaisesRegex(ValueError, "Train class support"):
            fit_baselines(self.arrays, self.train, contract, self.filter, context=32)

    def test_train_outcome_missingness_is_not_imputed(self):
        changed = copy.deepcopy(self.arrays)
        changed["return"][self.train[0]] = np.nan
        with self.assertRaisesRegex(ValueError, "Train return"):
            self.fit(changed)

    def test_duplicate_train_rows_are_not_allowed(self):
        with self.assertRaisesRegex(ValueError, "unique increasing"):
            fit_baselines(self.arrays, np.r_[self.train, self.train[-1]], self.contract, self.filter, context=32)

    def test_filter_context_cannot_differ_from_original(self):
        with self.assertRaisesRegex(ValueError, "context/horizon differs"):
            fit_baselines(self.arrays, self.train, self.contract, self.filter, context=33)

    def test_invalid_target_domains_are_rejected(self):
        for task, value in (("crash", .2), ("downside", -.1), ("recovery", 2)):
            changed = copy.deepcopy(self.arrays)
            changed[task][self.train[0], 0] = value
            with self.subTest(task=task), self.assertRaises(ValueError):
                self.fit(changed)


if __name__ == "__main__":
    unittest.main()
