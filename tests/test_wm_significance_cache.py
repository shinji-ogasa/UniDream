"""Synthetic checks for the immutable common-cohort significance cache."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments.wm_full_grid_evaluation import make_task_labels
from unidream.experiments.wm_seven_task_evaluation import TASKS
from unidream.experiments.wm_significance_cache import (
    _assert_outcomes, _ordered, _prior_cache, _sha, _verified_prediction,
    atomic_npz, build_cache, panel_arrays, require_registration, verify_balanced_accuracies,
)


class SignificanceCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = np.arange(100, 106)
        self.times = pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC").asi8
        self.frame = pd.DataFrame({"decision_id": self.ids, "timestamp": pd.to_datetime(self.times, utc=True),
            "split": "test", "fold": 0, "group": "A", "model": "wm", "seed": 7, "formulation": "regression"})
        self.contract = {}
        for task in TASKS:
            if task == "crash":
                entry = {"kind": "binary_event", "bounds": [.5], "n_classes": 2, "priors": [.7, .3]}
                actual = np.resize([0., 1.], 6)
                scalar = np.resize([.2, .7], 6)
                p = np.column_stack((1 - scalar, scalar))
            elif task in ("variance", "downside"):
                entry = {"kind": "train_tertiles", "bounds": [.01, .02], "n_classes": 3, "priors": [1/3] * 3}
                actual = np.resize([.005, .015, .03], 6)
                scalar = actual.copy()
                p = np.tile([.6, .3, .1], (6, 1))
            elif task == "dd_excess":
                entry = {"kind": "zero_positive_median", "bounds": [0., .01], "n_classes": 3, "priors": [.6, .2, .2]}
                actual = np.resize([0., .005, .02], 6)
                scalar = actual.copy()
                p = np.tile([.6, .3, .1], (6, 1))
            else:
                entry = {"kind": "signed_rms", "bounds": [-.01, .01], "n_classes": 3, "priors": [.25, .5, .25]}
                actual = np.resize([-.02, 0., .02], 6)
                scalar = actual.copy()
                p = np.tile([.6, .3, .1], (6, 1))
            self.contract[task] = entry
            self.frame["actual_" + task] = actual
            self.frame["label_" + task] = make_task_labels(actual, task, entry)
            self.frame["scalar_" + task] = scalar
            for c in range(3):
                self.frame[f"p_{task}_{c}"] = p[:, c] if c < entry["n_classes"] else np.nan
        self.actual = self.frame[["actual_" + t for t in TASKS]].to_numpy()
        self.labels = self.frame[["label_" + t for t in TASKS]].to_numpy()

    def test_registration_required_before_source_access(self):
        with patch("unidream.experiments.wm_significance_cache.joint_grid.load_config") as load:
            with self.assertRaisesRegex(ValueError, "before any historical Test"):
                build_cache({"output_dir": self.root}, {})
            load.assert_not_called()

    def test_disk_registration_must_equal_supplied_registration(self):
        (self.root / "preregistration.json").write_text('{"analysis": 1}')
        with self.assertRaisesRegex(ValueError, "differs"):
            require_registration(self.root, {"analysis": 2})

    def test_atomic_npz_has_no_pickle_arrays_or_leftover_tempfile(self):
        path = self.root / "cache.npz"
        atomic_npz(path, scalar=np.ones((5, 7)), group=np.asarray("A"))
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data["scalar"].shape, (5, 7))
            self.assertEqual(data["group"].item(), "A")
        self.assertFalse(path.with_name("cache.npz.tmp").exists())
        with self.assertRaisesRegex(ValueError, "pickle"):
            atomic_npz(path, bad=np.asarray([{}], dtype=object))

    def test_regression_uses_scalar_thresholds_and_crash_train_prior(self):
        result = panel_arrays(self.frame, self.contract, "regression")
        np.testing.assert_array_equal(result["scalar"], self.frame[["scalar_" + t for t in TASKS]].to_numpy())
        for c, task in enumerate(TASKS):
            expected = make_task_labels(result["scalar"][:, c], task, self.contract)
            if task == "crash":
                probability = result["scalar"][:, c]
                expected = np.argmax(np.column_stack((1 - probability, probability)) / [.7, .3], axis=1)
            np.testing.assert_array_equal(result["classes"][:, c], expected)
        np.testing.assert_array_equal(result["crash_p"], self.frame.scalar_crash)

    def test_classification_ignores_scalar_head_and_normalizes_probabilities(self):
        frame = self.frame.copy()
        frame[["scalar_" + t for t in TASKS]] = np.nan
        result = panel_arrays(frame, self.contract, "classification")
        self.assertTrue(np.isnan(result["scalar"]).all())
        for c, task in enumerate(TASKS):
            p = frame[[f"p_{task}_{i}" for i in range(self.contract[task]["n_classes"])]].to_numpy()
            expected = np.argmax((p / p.sum(1, keepdims=True)) / self.contract[task]["priors"], axis=1)
            np.testing.assert_array_equal(result["classes"][:, c], expected)

    def test_joint_keeps_both_heads(self):
        result = panel_arrays(self.frame, self.contract, "joint")
        self.assertTrue(np.isfinite(result["scalar"]).all())
        np.testing.assert_array_equal(result["classes"], panel_arrays(self.frame, self.contract, "classification")["classes"])

    def test_invalid_probabilities_and_missing_native_values_fail(self):
        frame = self.frame.copy()
        frame.loc[0, "p_return_0"] = -1
        with self.assertRaisesRegex(ValueError, "probabilities"):
            panel_arrays(frame, self.contract, "classification")
        frame = self.frame.copy()
        frame.loc[0, "scalar_return"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            panel_arrays(frame, self.contract, "joint")

    def test_reordered_rows_are_restored_but_missing_or_duplicate_rows_fail(self):
        shuffled = self.frame.iloc[::-1]
        result = _ordered(shuffled, {"split": "test"}, self.ids, self.times)
        np.testing.assert_array_equal(result.decision_id, self.ids)
        for frame in (self.frame.iloc[1:], pd.concat([self.frame, self.frame.iloc[:1]])):
            with self.assertRaises(ValueError):
                _ordered(frame, {"split": "test"}, self.ids, self.times)

    def test_timestamp_and_outcome_mismatch_are_fatal(self):
        changed = self.frame.copy()
        changed.loc[0, "timestamp"] += pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            _ordered(changed, {}, self.ids, self.times)
        changed = self.frame.copy()
        changed.loc[0, "actual_return"] = 12
        with self.assertRaisesRegex(ValueError, "outcome"):
            _assert_outcomes(changed, self.actual, self.labels)
        changed = self.frame.copy()
        changed.loc[0, "label_return"] = 1
        with self.assertRaisesRegex(ValueError, "class labels"):
            _assert_outcomes(changed, self.actual, self.labels)

    def test_every_cached_ba_must_reproduce_frozen_point_table(self):
        panel = panel_arrays(self.frame, self.contract, "regression")
        rows = []
        for c, task in enumerate(TASKS):
            k = 2 if task == "crash" else 3
            ba = np.mean([np.mean(panel["classes"][self.labels[:, c] == label, c] == label) for label in range(k)])
            rows.append({"group": "A", "model": "wm", "formulation": "regression", "task": task, "balanced_accuracy": ba})
        table = pd.DataFrame(rows)
        verify_balanced_accuracies(panel, self.labels, table, "A", "wm", "regression")
        table.loc[0, "balanced_accuracy"] -= .001
        with self.assertRaisesRegex(ValueError, "frozen336"):
            verify_balanced_accuracies(panel, self.labels, table, "A", "wm", "regression")

    def source_fixture(self):
        root = self.root / "old"
        directory = root / "classical/fold_00_D"
        directory.mkdir(parents=True)
        (root / "preregistration.json").write_text("{}")
        (directory / "models.pkl").write_bytes(b"synthetic trusted model bytes")
        (directory / "test_predictions.parquet").write_bytes(b"synthetic hashed predictions")
        fit = {"registration_sha256": _sha(root / "preregistration.json"), "model_sha256": _sha(directory / "models.pkl")}
        (directory / "fit.json").write_text(json.dumps(fit))
        barrier = {"registration_sha256": fit["registration_sha256"], "frozen_fit_sha256": {"classical/fold_00_D/fit.json": _sha(directory / "fit.json")}}
        (root / "test_barrier.json").write_text(json.dumps(barrier))
        evaluation = {"barrier_sha256": _sha(root / "test_barrier.json"), "prediction_sha256": _sha(directory / "test_predictions.parquet"), "selected_model_sha256": fit["model_sha256"]}
        (directory / "evaluation.json").write_text(json.dumps(evaluation))
        return root, directory

    def test_source_prediction_model_and_barrier_hashes_are_verified(self):
        root, directory = self.source_fixture()
        provenance = {}
        _verified_prediction(root, "classical", "fold_00_D", provenance, load_model=True)
        self.assertIn(str((directory / "models.pkl").resolve()), provenance)
        (directory / "models.pkl").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model bundle changed"):
            _verified_prediction(root, "classical", "fold_00_D", {}, load_model=True)

    def test_source_prediction_replacement_is_rejected(self):
        root, directory = self.source_fixture()
        (directory / "test_predictions.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "predictions"):
            _verified_prediction(root, "classical", "fold_00_D", {})

    def test_resume_checks_every_cache_and_source_checksum(self):
        base, panel, source = [self.root / n for n in ("base.npz", "panel.npz", "source")]
        for path in (base, panel, source):
            path.write_bytes(b"frozen")
        path = self.root / "cache_manifest.json"
        manifest = {"complete": True, "registration_sha256": "registered", "baseline_path": str(base), "baseline_sha256": _sha(base),
            "panels": {"A_wm_joint": {"path": str(panel), "sha256": _sha(panel)}}, "source_artifact_sha256": {str(source): _sha(source)}}
        path.write_text(json.dumps(manifest))
        self.assertEqual(_prior_cache(path, "registered")["cache_manifest_sha"], _sha(path))
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            _prior_cache(path, "registered")


if __name__ == "__main__":
    unittest.main()
