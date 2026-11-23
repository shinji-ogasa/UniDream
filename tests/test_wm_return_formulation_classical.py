import inspect
import pickle
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from unidream.experiments import wm_return_formulation_classical as classical


def examples():
    rng = np.random.default_rng(617)
    x = rng.normal(size=(220, 7)).astype(np.float32)
    x[:, -1] = 4.
    movement = x[:, 0] + .5 * x[:, 1] + rng.normal(scale=.25, size=len(x))
    labels = np.where(movement < -.5, 0, np.where(movement > .5, 2, 1))
    return x, labels


SMALL = {
    "cpu_threads": 1, "linear_max_train_rows": 90,
    "logistic_Cs": [.01, .1, 1.], "logistic_max_iter": 200,
    "lightgbm_leaves": [3, 7], "lightgbm_rounds": 16,
    "lightgbm_patience": 4, "lightgbm_min_data_leaf": 5,
    "lightgbm_max_bin": 15, "lightgbm_learning_rate": .1,
}


class ReturnFormulationClassicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.labels = examples()
        cls.original = cls.x.copy()
        cls.bundle = classical.fit_return_classifiers(cls.x[:140], cls.labels[:140],
                                                       cls.x[140:200], cls.labels[140:200], SMALL)

    def test_no_test_boundary_scaler_and_priors_use_full_train_only(self):
        self.assertEqual(tuple(inspect.signature(classical.fit_return_classifiers).parameters),
                         ("x_train", "labels_train", "x_early", "labels_early", "config"))
        np.testing.assert_array_equal(self.x, self.original)
        np.testing.assert_allclose(self.bundle["scaler"]["mean"], self.x[:140].mean(0, dtype=np.float64))
        expected = self.x[:140].std(0, dtype=np.float64)
        expected[expected <= 1e-12] = 1.
        np.testing.assert_allclose(self.bundle["scaler"]["scale"], expected)
        metadata = self.bundle["metadata"]
        np.testing.assert_allclose(metadata["train_priors"], np.bincount(self.labels[:140], minlength=3) / 140.)
        self.assertEqual(metadata["lightgbm_train_n"], 140)
        self.assertEqual(len(metadata["linear_train_rows"]), 90)
        self.assertFalse(metadata["test_labels_used"])
        self.assertIsNone(metadata["class_weight"])
        self.assertFalse(metadata["probabilities_prior_adjusted"])

    def test_selected_hyperparameters_and_iteration_minimize_unweighted_val_log_loss(self):
        metadata = self.bundle["metadata"]
        for name in ("logistic", "lightgbm"):
            info = metadata[name]
            self.assertEqual(info["selected"], min(info["candidates"], key=lambda row: row["log_loss"]))
        for row in metadata["lightgbm"]["candidates"]:
            self.assertEqual(row["selected_iteration"], int(np.argmin(row["history"])) + 1)
        probabilities = classical.predict_return_classifiers(self.bundle, self.x[140:200])
        for name, values in probabilities.items():
            expected = -np.log(np.maximum(values[np.arange(60), self.labels[140:200]], 1e-12)).mean()
            self.assertAlmostEqual(metadata[name]["selected"]["log_loss"], expected, places=12)
        self.assertEqual(metadata["lightgbm"]["parameters"]["metric"], "None")

    def test_outputs_are_raw_normalized_probabilities_with_stable_pickle_and_column_order(self):
        predicted = classical.predict_return_classifiers(self.bundle, self.x[200:])
        restored = classical.predict_return_classifiers(pickle.loads(pickle.dumps(self.bundle)), self.x[200:])
        self.assertEqual(set(predicted), {"logistic", "lightgbm"})
        for name, p in predicted.items():
            self.assertEqual(p.shape, (20, 3))
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue((p >= 0).all())
            np.testing.assert_allclose(p.sum(1), 1., atol=1e-12)
            np.testing.assert_array_equal(p, restored[name])
        model = self.bundle["fits"]["logistic"]
        self.assertIsNone(model.class_weight)
        np.testing.assert_array_equal(model.classes_, [0, 1, 2])
        standardized = (self.x[200:].astype(float) - self.bundle["scaler"]["mean"]) / self.bundle["scaler"]["scale"]
        np.testing.assert_array_equal(predicted["logistic"], model.predict_proba(standardized))

    def test_rare_train_class_retained_under_even_row_cap_without_reestimating_priors(self):
        labels = np.ones(101, dtype=int)
        labels[50] = 0
        labels[3] = 2
        rows = classical._linear_rows(labels, 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(np.unique(rows)), 3)
        np.testing.assert_array_equal(np.unique(labels[rows]), [0, 1, 2])
        self.assertTrue((np.diff(rows) > 0).all())

    def test_missing_class_in_val_is_allowed_but_missing_train_class_is_rejected(self):
        tiny = {**SMALL, "logistic_Cs": [.01], "lightgbm_leaves": [3], "lightgbm_rounds": 3}
        bundle = classical.fit_return_classifiers(self.x[:140], self.labels[:140], self.x[140:160],
                                                   np.ones(20, dtype=int), tiny)
        self.assertTrue(np.isfinite(bundle["metadata"]["lightgbm"]["selected"]["log_loss"]))
        with self.assertRaisesRegex(ValueError, "all three"):
            classical.fit_return_classifiers(self.x[:140], np.ones(140), self.x[140:160], self.labels[140:160], tiny)

    def test_nonconverged_logistic_is_retried_and_exhaustion_raises(self):
        seen = []

        class FakeLogistic:
            def __init__(self, **kwargs):
                self.max_iter = kwargs["max_iter"]
                self.n_iter_ = np.array([self.max_iter])
                self.classes_ = np.array([0, 1, 2])
                seen.append(self.max_iter)

            def fit(self, _x, _y):
                if self.max_iter == 10:
                    warnings.warn("fixture convergence failure", ConvergenceWarning)
                return self

            def predict_proba(self, values):
                return np.full((len(values), 3), 1. / 3.)

        with patch.object(classical, "LogisticRegression", FakeLogistic):
            _, metadata = classical._fit_logistic(self.x[:140], self.labels[:140], self.x[140:160], self.labels[140:160],
                                                   {"logistic_Cs": [.1], "logistic_max_iter": 10})
        self.assertEqual(seen, [10, 40])
        self.assertEqual(len(metadata["selected"]["attempts"]), 2)

        class NeverConverges(FakeLogistic):
            def fit(self, _x, _y):
                warnings.warn("fixture persistent failure", ConvergenceWarning)
                return self

        with patch.object(classical, "LogisticRegression", NeverConverges):
            with self.assertRaisesRegex(RuntimeError, "failed convergence"):
                classical._fit_logistic(self.x[:140], self.labels[:140], self.x[140:160], self.labels[140:160],
                                         {"logistic_Cs": [.1], "logistic_max_iter": 10})

    def test_input_domains_and_probability_floor_are_explicit(self):
        with self.assertRaises(ValueError):
            classical.predict_return_classifiers(self.bundle, self.x[200:, :-1])
        bad = self.x[200:].copy()
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            classical.predict_return_classifiers(self.bundle, bad)
        p = np.array([[0., 1., 0.], [0., 1., 0.], [1., 0., 0.]])
        expected = -2. / 3. * np.log(1e-12)
        self.assertAlmostEqual(classical.return_classification_log_loss([0, 1, 2], p), expected)
        with self.assertRaises(ValueError):
            classical.return_classification_log_loss([0, 1, 2], np.ones((3, 3)))


if __name__ == "__main__":
    unittest.main()
