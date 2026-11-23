import inspect
import pickle
import unittest

import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from unidream.experiments import wm_full_grid_classical as grid


def examples():
    rng = np.random.default_rng(9021)
    x = rng.normal(size=(210, 8)).astype(np.float32)
    x[:, -1] = 2.
    scores = [x[:, 0] + .2 * x[:, 1], x[:, 2], -x[:, 0] + .3 * x[:, 3],
              x[:, 4], x[:, 0] - .1 * x[:, 2], x[:, 5]]
    labels = {task: np.where(score < -.4, 0, np.where(score > .4, 2, 1))
              for task, score in zip(grid.CLASSIFICATION_TASKS, scores)}
    crash = (x[:, 0] + .5 * x[:, 1] < -.8).astype(float)
    return x, labels, {"crash": crash}


SMALL = {
    "cpu_threads": 1, "linear_max_train_rows": 90,
    "ridge_alphas": [1., 100., 10000.], "logistic_Cs": [.01, .1],
    "logistic_max_iter": 300, "lightgbm_leaves": [3, 7],
    "lightgbm_rounds": 12, "lightgbm_patience": 3,
    "lightgbm_min_data_leaf": 5, "lightgbm_max_bin": 15,
    "lightgbm_learning_rate": .1,
}


class FullGridClassicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.labels, cls.targets = examples()
        cls.original_x = cls.x.copy()
        cls.train_labels = {key: value[:140] for key, value in cls.labels.items()}
        cls.early_labels = {key: value[140:190] for key, value in cls.labels.items()}
        cls.bundle = grid.fit_grid_classical(cls.x[:140], {"crash": cls.targets["crash"][:140]},
                                             cls.train_labels, cls.x[140:190],
                                             {"crash": cls.targets["crash"][140:190]}, cls.early_labels, SMALL)

    def test_fit_accepts_only_train_and_earlystop_and_uses_one_full_train_scaler(self):
        self.assertEqual(tuple(inspect.signature(grid.fit_grid_classical).parameters),
                         ("x_train", "targets_train", "labels_train", "x_early", "targets_early", "labels_early", "config"))
        np.testing.assert_array_equal(self.x, self.original_x)
        np.testing.assert_allclose(self.bundle["scaler"]["mean"], self.x[:140].mean(0, dtype=np.float64))
        expected = self.x[:140].std(0, dtype=np.float64)
        expected[expected <= 1e-12] = 1.
        np.testing.assert_allclose(self.bundle["scaler"]["scale"], expected)
        metadata = self.bundle["metadata"]
        self.assertFalse(metadata["test_labels_used"])
        self.assertFalse(metadata["existing_objectives_refitted"])
        self.assertEqual(metadata["lightgbm_train_n"], 140)
        self.assertEqual(len(metadata["linear_train_rows"]), 90)
        for task in grid.CLASSIFICATION_TASKS:
            np.testing.assert_allclose(metadata["train_priors"][task], np.bincount(self.train_labels[task], minlength=3) / 140.)

    def test_only_missing_objectives_are_returned_with_valid_domains(self):
        predictions = grid.predict_grid_classical(self.bundle, self.x[190:])
        self.assertEqual(set(predictions), {"classification", "regression"})
        for model in ("linear", "lightgbm"):
            self.assertEqual(set(predictions["classification"][model]), set(grid.CLASSIFICATION_TASKS))
            self.assertEqual(set(predictions["regression"][model]), {"crash"})
            for task, probability in predictions["classification"][model].items():
                self.assertEqual(probability.shape, (20, 3))
                self.assertTrue(np.isfinite(probability).all())
                self.assertTrue((probability >= 0).all())
                np.testing.assert_allclose(probability.sum(1), 1., atol=1e-12)
            crash = predictions["regression"][model]["crash"]
            self.assertEqual(crash.shape, (20,))
            self.assertTrue(np.isfinite(crash).all())
            self.assertTrue(((crash >= 0) & (crash <= 1)).all())

    def test_selected_classification_scores_use_unweighted_earlystop_log_loss(self):
        predictions = grid.predict_grid_classical(self.bundle, self.x[140:190])
        selection = self.bundle["metadata"]["selection"]["classification"]
        for model in ("linear", "lightgbm"):
            for task in grid.CLASSIFICATION_TASKS:
                probability = predictions["classification"][model][task]
                independent = float(-np.log(np.maximum(probability[np.arange(50), self.early_labels[task]], 1e-12)).mean())
                info = selection[model][task]
                self.assertAlmostEqual(independent, info["selected"]["log_loss"], places=12)
                self.assertEqual(info["selected"], min(info["candidates"], key=lambda row: row["log_loss"]))
                if model == "lightgbm":
                    self.assertEqual(info["selected"]["selected_iteration"], int(np.argmin(info["selected"]["history"])) + 1)
                    self.assertEqual(info["selected"]["objective"], "multiclass")

    def test_crash_ridge_matches_independent_sklearn_and_selects_clipped_brier(self):
        metadata = self.bundle["metadata"]
        rows = metadata["linear_train_rows"]
        mean, scale = self.bundle["scaler"]["mean"], self.bundle["scaler"]["scale"]
        x = (self.x[:140][rows].astype(float) - mean) / scale
        xv = (self.x[140:190].astype(float) - mean) / scale
        actual = self.targets["crash"][140:190]
        independently_scored = []
        with threadpool_limits(limits=1):
            for row in metadata["selection"]["regression"]["linear"]["crash"]["candidates"]:
                model = Ridge(alpha=row["alpha"], solver="cholesky").fit(x, self.targets["crash"][:140][rows])
                predicted = np.clip(model.predict(xv), 0., 1.)
                score = float(np.mean((actual - predicted) ** 2))
                self.assertAlmostEqual(score, row["brier"], places=12)
                independently_scored.append((score, row["alpha"]))
        selected = metadata["selection"]["regression"]["linear"]["crash"]["selected"]
        self.assertEqual(selected["alpha"], min(independently_scored)[1])
        tree = metadata["selection"]["regression"]["lightgbm"]["crash"]
        self.assertEqual(tree["selected"]["objective"], "regression")
        predicted = grid.predict_grid_classical(self.bundle, self.x[140:190])["regression"]["lightgbm"]["crash"]
        self.assertAlmostEqual(tree["selected"]["brier"], np.mean((predicted - actual) ** 2), places=12)

    def test_pickle_replays_all_fourteen_new_heads_exactly(self):
        before = grid.predict_grid_classical(self.bundle, self.x[190:])
        after = grid.predict_grid_classical(pickle.loads(pickle.dumps(self.bundle)), self.x[190:])
        for mode in before:
            for model in before[mode]:
                for task in before[mode][model]:
                    np.testing.assert_array_equal(before[mode][model][task], after[mode][model][task])

    def test_wide_design_uses_dual_ridge_and_matches_independent_solver(self):
        rng = np.random.default_rng(82)
        x, xv = rng.normal(size=(18, 40)), rng.normal(size=(9, 40))
        actual, val = (x[:, 0] < 0).astype(float), (xv[:, 0] < 0).astype(float)
        with threadpool_limits(limits=1):
            head, info = grid._fit_crash_ridge(x, actual, xv, val, {"ridge_alphas": [1., 100.]})
            scores = []
            for row in info["candidates"]:
                independent = Ridge(alpha=row["alpha"], solver="cholesky").fit(x, actual)
                score = np.mean((np.clip(independent.predict(xv), 0., 1.) - val) ** 2)
                self.assertAlmostEqual(row["brier"], score, places=12)
                scores.append((score, row["alpha"]))
        self.assertEqual(info["gram_form"], "dual")
        self.assertEqual(info["selected"]["alpha"], min(scores)[1])
        self.assertTrue(np.isfinite(head["coefficients"]).all())

    def test_rare_class_adjustment_preserves_all_six_label_coverages(self):
        labels = {task: np.arange(100) % 3 for task in grid.CLASSIFICATION_TASKS}
        labels["dd_excess"] = np.ones(100, dtype=int)
        labels["dd_excess"][[5, 7]] = [0, 2]
        rows, adjusted = grid._linear_rows(labels, 20)
        self.assertTrue(adjusted)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len(np.unique(rows)), 20)
        for task in grid.CLASSIFICATION_TASKS:
            np.testing.assert_array_equal(np.unique(labels[task][rows]), [0, 1, 2])

    def test_missing_val_class_is_allowed_and_invalid_train_classes_are_rejected(self):
        bad_train = {**self.train_labels, "dd_excess": np.ones(140, dtype=int)}
        with self.assertRaisesRegex(ValueError, "all three"):
            grid.fit_grid_classical(self.x[:140], {"crash": self.targets["crash"][:140]}, bad_train,
                                    self.x[140:190], {"crash": self.targets["crash"][140:190]}, self.early_labels, SMALL)
        accepted = grid._classification_labels({task: np.ones(50, dtype=int) for task in grid.CLASSIFICATION_TASKS},
                                               50, "earlystop labels")
        self.assertEqual(set(accepted), set(grid.CLASSIFICATION_TASKS))
        with self.assertRaises(ValueError):
            grid.predict_grid_classical(self.bundle, self.x[190:, :-1])
        with self.assertRaises(ValueError):
            grid._crash_target({"crash": [.5]}, 1, "invalid")


if __name__ == "__main__":
    unittest.main()
