import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from unidream.experiments import wm_classification_classical as classical
from unidream.experiments.wm_classification_evaluation import score_classification


def examples():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(180, 6)).astype(np.float32)
    labels = np.argmax(np.column_stack((x[:, 0], x[:, 1], -x[:, 0] - x[:, 1])), axis=1)
    train, val = np.arange(120), np.arange(120, 180)
    priors = np.bincount(labels[train], minlength=3) / len(train)
    return x, labels, train, val, priors


SMALL_TABULAR = {"logistic_Cs": [.01, 1.], "logistic_max_iter": 100,
                 "lightgbm_leaves": [3, 7], "lightgbm_rounds": 12,
                 "lightgbm_patience": 3, "lightgbm_min_data_leaf": 4,
                 "lightgbm_max_bin": 15, "lightgbm_learning_rate": .1}


class ClassificationClassicalTests(unittest.TestCase):
    def test_prior_adjustment_changes_decision_without_changing_probabilities(self):
        probabilities = np.array([[.2, .6, .2], [.1, .7, .2]])
        before = probabilities.copy()
        predicted = classical.prior_adjusted_decision(probabilities, [.1, .8, .1])
        np.testing.assert_array_equal(predicted, [0, 2])
        np.testing.assert_array_equal(probabilities, before)

    def test_append_side_uses_only_selected_theta_and_train_rms(self):
        design = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        theta = np.array([np.nan, .01, np.nan, .04, np.nan])
        result = classical.append_threshold_side(design, theta, [1, 3], .02)
        np.testing.assert_array_equal(result[:, :2], design)
        np.testing.assert_allclose(result[:, 2], np.log([.5, 2.]))
        self.assertEqual(result.dtype, np.float32)

    def test_validation_logloss_uses_the_shared_one_e_minus_twelve_floor(self):
        labels = np.array([0, 1, 2])
        probabilities = np.array([[0., 1., 0.], [0., 1e-30, 1.], [1., 0., 0.]])
        priors = np.full(3, 1. / 3.)
        result = classical.validation_scores(labels, probabilities, priors)
        shared = score_classification(labels, probabilities, priors)
        self.assertEqual(result["log_loss"], shared["log_loss"])
        self.assertAlmostEqual(result["log_loss"], -np.log(1e-12))

    def test_logistic_selects_true_validation_ba_then_logloss(self):
        x, labels, train, val, priors = examples()
        with threadpool_limits(limits=1):
            model, history = classical.fit_logistic_candidates(x[train], labels[train], x[val], labels[val], priors, SMALL_TABULAR)
        self.assertIsNone(model.class_weight)
        self.assertEqual(model.solver, "lbfgs")
        np.testing.assert_array_equal(model.classes_, [0, 1, 2])
        selected = min(history["candidates"], key=lambda row: (-row["balanced_accuracy"], row["log_loss"]))
        self.assertEqual(history["selected"], selected)
        self.assertEqual(model.C, selected["C"])

    def test_logistic_convergence_warning_triggers_higher_iteration_retry(self):
        x, labels, train, val, priors = examples()
        instances = []

        class FakeLogistic:
            def __init__(self, **kwargs):
                self.max_iter = kwargs["max_iter"]
                self.classes_ = np.array([0, 1, 2])
                self.n_iter_ = np.array([self.max_iter])
                instances.append(self)

            def fit(self, _x, _labels):
                if len(instances) == 1:
                    warnings.warn("not converged", ConvergenceWarning)
                return self

            def predict_proba(self, values):
                return np.tile(priors, (len(values), 1))

        with patch.object(classical, "LogisticRegression", FakeLogistic):
            _, history = classical.fit_logistic_candidates(x[train], labels[train], x[val], labels[val], priors,
                                                          {"logistic_Cs": [.1], "logistic_max_iter": 10})
        self.assertEqual([instance.max_iter for instance in instances], [10, 40])
        self.assertEqual(len(history["selected"]["fit_attempts"]), 2)

    def test_lightgbm_monitors_only_ba_and_selects_true_observed_tiebreak(self):
        import lightgbm as lgb
        x, labels, train, val, priors = examples()
        actual_train = lgb.train
        observed_parameters = []

        def observe(parameters, *args, **kwargs):
            observed_parameters.append(parameters.copy())
            self.assertTrue(kwargs["keep_training_booster"])
            return actual_train(parameters, *args, **kwargs)

        with patch.object(lgb, "train", side_effect=observe):
            model, history = classical.fit_lightgbm_candidates(x[train], labels[train], x[val], labels[val], priors,
                                                              SMALL_TABULAR, cpu_threads=1)
        self.assertTrue(all(parameters["metric"] == "None" for parameters in observed_parameters))
        self.assertTrue(all(parameters["objective"] == "multiclass" for parameters in observed_parameters))
        selected = history["selected"]
        best = min(history["candidates"], key=lambda row: (-row["balanced_accuracy"], row["log_loss"]))
        self.assertEqual(selected, best)
        for candidate in history["candidates"]:
            true_best = min(candidate["history"], key=lambda row: (-row["balanced_accuracy"], row["log_loss"]))
            self.assertEqual(candidate["selected_iteration"], true_best["iteration"])
        predicted = model.predict(x[val], num_iteration=selected["selected_iteration"], num_threads=1)
        score = classical.validation_scores(labels[val], predicted, priors)
        self.assertAlmostEqual(score["balanced_accuracy"], selected["balanced_accuracy"])

    def test_task_fits_only_train_and_selection_panel_then_persists_artifacts(self):
        x, labels, train, val, priors = examples()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "prepared").mkdir()
            # Unavailable labels and thresholds outside the allowed fitting
            # cohorts, including Test, must not be read by selection code.
            valid_ids = np.r_[train, val[:30]]
            theta = np.full(200, np.nan)
            theta[valid_ids] = .01
            full_labels = np.full(200, -1, dtype=np.int8)
            full_labels[valid_ids] = labels[valid_ids]
            test = np.arange(190, 200)
            np.savez(out / "prepared/fold_00.npz", train=train, val=val,
                     selection_val=val[:30], test=test, theta_fixed=theta, label_fixed=full_labels)
            (out / "prepared/folds.json").write_text(json.dumps([{"return_rms": .02, "priors": {"fixed": priors.tolist()}}]))
            (out / "preregistration.json").write_text('{"synthetic_verification":true}')
            config = {"output_dir": str(out), "tabular": SMALL_TABULAR, "cpu_threads": 1}
            design_calls = []

            def design(_cfg, ids, _data):
                design_calls.append(ids.copy())
                if not np.isin(ids, valid_ids).all():
                    raise AssertionError("selection attempted to inspect non-panel/Test rows")
                return x[ids]

            def sha(path):
                return hashlib.sha256(Path(path).read_bytes()).hexdigest()

            def write_json(path, value):
                Path(path).write_text(json.dumps(value, allow_nan=False))

            root = SimpleNamespace(load_config=lambda _: config, get_design=design, _sha=sha, _json=write_json)
            task = {"condition": "fixed", "fold": 0, "job": "fold_00_fixed"}
            with patch.object(classical.importlib, "import_module", return_value=root):
                result = classical.fit_classical_task("fixture.yaml", task)
            self.assertEqual(result["job"], task["job"])
            self.assertEqual(len(design_calls), 2)
            np.testing.assert_array_equal(design_calls[0], train)
            np.testing.assert_array_equal(design_calls[1], val[:30])
            directory = out / "classical" / task["job"]
            metadata = json.loads((directory / "fit.json").read_text())
            self.assertFalse(metadata["test_predictions_produced"])
            self.assertEqual(metadata["design_width"], 7)
            for name, digest in metadata["artifact_sha256"].items():
                self.assertEqual(sha(directory / name), digest)
            self.assertFalse(list(out.rglob("*predictions*")))


if __name__ == "__main__":
    unittest.main()
