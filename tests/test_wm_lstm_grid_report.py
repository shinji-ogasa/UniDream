"""Synthetic LSTM report checks; no existing Test prediction is read."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments.wm_full_grid_evaluation import NATIVE_PRIMARY, make_task_labels, score_grid
from unidream.experiments.wm_full_grid_report import _sha
from unidream.experiments.wm_lstm_grid_report import (
    FORMULATIONS, _pooled, _row, _validate, write_lstm_grid_report,
)
from unidream.experiments.wm_seven_task_evaluation import TASKS


class LSTMGridReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contract = {}
        for task in TASKS:
            if task == "crash":
                entry = {"kind": "binary_event", "bounds": [.5], "n_classes": 2, "priors": [.7, .3]}
            elif task in ("variance", "downside"):
                entry = {"kind": "train_tertiles", "bounds": [.01, .02], "n_classes": 3, "priors": [1/3] * 3}
            elif task == "dd_excess":
                entry = {"kind": "zero_positive_median", "bounds": [0., .01], "n_classes": 3, "priors": [.6, .2, .2]}
            else:
                entry = {"kind": "signed_rms", "bounds": [-.01, .01], "n_classes": 3, "priors": [.25, .5, .25]}
            self.contract[task] = entry

    def frame(self, expected):
        fold, mode = expected["fold"], expected["formulation"]
        times = pd.date_range(pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(days=fold * 8), periods=6, freq="15min")
        frame = pd.DataFrame({"decision_id": fold * 100 + np.arange(6), "timestamp": times,
            **{key: value for key, value in expected.items() if key != "fold"}, "fold": fold, "seed": 7})
        for task in TASKS:
            actual = (np.resize([0., 1.], 6) if task == "crash" else np.resize([.005, .015, .03], 6)
                if task in ("variance", "downside") else np.resize([0., .005, .02], 6)
                if task == "dd_excess" else np.resize([-.02, 0., .02], 6))
            scalar = np.where(actual == 0, .1, .8) if task == "crash" else actual * .9
            p = np.tile([.7, .3], (6, 1)) if task == "crash" else np.tile([.2, .5, .3], (6, 1))
            frame["actual_" + task] = actual
            frame["label_" + task] = make_task_labels(actual, task, self.contract)
            frame["scalar_" + task] = scalar if mode != "classification" else np.nan
            for c in range(3):
                frame[f"p_{task}_{c}"] = p[:, c] if mode != "regression" and c < p.shape[1] else np.nan
        return frame

    def fake_reader(self, path, expected, provenance, include_test):
        provenance[str(path)] = "synthetic_hash"
        return self.frame(expected)

    def fixture(self):
        point = self.root / "old336.csv"
        rows = [{"split": "test", "group": group, "model": model, "task": task, "formulation": mode,
            "primary_metric": "balanced_accuracy" if mode != "regression" else NATIVE_PRIMARY[task],
            "primary_value": .5, "balanced_accuracy": .5}
            for group in "ABCD" for model in ("linear", "lightgbm", "direct", "wm") for task in TASKS for mode in FORMULATIONS]
        pd.DataFrame(rows).to_csv(point, index=False)
        tasks = [{"fold": fold, "group": group, "model": "lstm", "seed": 7, "formulation": mode,
            "job": f"fold_{fold:02d}_{group}_lstm_{mode}_s7"}
            for fold in range(15) for group in "ABCD" for mode in FORMULATIONS]
        registration = {"neural_tasks": tasks, "classical_tasks": [],
            "folds": [{"fold": fold, "label_contract": self.contract} for fold in range(15)],
            "parent_results_336_sha256": _sha(point)}
        (self.root / "preregistration.json").write_text(json.dumps(registration))
        (self.root / "test_barrier.json").write_text("{}")
        barrier = _sha(self.root / "test_barrier.json")
        for task in tasks:
            directory = self.root / "neural" / task["job"]
            directory.mkdir(parents=True)
            (directory / "fit.json").write_text(json.dumps({"parameters": 12345, "steps": 1000, "best_step": 800, "stop_reason": "max_steps"}))
            (directory / "evaluation.json").write_text(json.dumps({"barrier_sha256": barrier}))
        cfg = {"point_table_path": str(point), "model": {"context": 256, "hidden_size": 192}, "training": {"max_steps": 3000}}
        return cfg, registration

    def test_joint_row_keeps_two_native_scores_and_actual_shared_learning(self):
        frame = self.frame({"fold": 0, "group": "A", "model": "lstm", "formulation": "joint", "split": "test"})
        task = "variance"
        score = score_grid(task, frame.actual_variance, frame[[f"p_variance_{i}" for i in range(3)]].to_numpy(), "classification", self.contract, None)
        score["native"] = score_grid(task, frame.actual_variance, frame.scalar_variance, "regression", self.contract, None)["native"]
        row = _row({"split": "test", "group": "A", "model": "lstm", "task": task, "formulation": "joint"}, score, 1, [score["primary_value"]])
        self.assertEqual(row["regression_primary_metric"], "qlike")
        self.assertEqual(row["classification_primary_metric"], "balanced_accuracy")
        self.assertTrue(row["shared_parameters"])
        self.assertTrue(row["fresh_training"])
        self.assertFalse(row["control_only"])
        self.assertEqual(row["primary_unit"], "fraction")

    def test_pooled_scores_recompute_native_metrics_on_all_rows(self):
        frames = [self.frame({"fold": fold, "group": "A", "model": "lstm", "formulation": "regression", "split": "test"}) for fold in (0, 1)]
        task = "return"
        scores = [score_grid(task, f.actual_return, f.scalar_return, "regression", self.contract, None) for f in frames]
        pooled = _pooled(task, frames, scores, "regression")
        actual = np.concatenate([f.actual_return for f in frames])
        predicted = np.concatenate([f.scalar_return for f in frames])
        self.assertAlmostEqual(pooled["native"]["zero_oos_r2"], 1 - np.square(actual - predicted).sum() / np.square(actual).sum())
        self.assertEqual(pooled["n"], 12)

    def test_validation_report_84_does_not_read_old_test_table(self):
        cfg, _ = self.fixture()
        cfg["point_table_path"] = str(self.root / "NO_TEST_READ.csv")
        with patch("unidream.experiments.wm_lstm_grid_report._gate", return_value=("reg", {})), \
             patch("unidream.experiments.wm_lstm_grid_report._read_panel", side_effect=self.fake_reader):
            complete = write_lstm_grid_report(self.root, cfg, include_test=False)
        self.assertEqual(complete["cells"], 84)
        self.assertIsNone(complete["combined_cells"])
        table = pd.read_csv(self.root / "selection/results_84.csv")
        self.assertEqual(len(table), 84)
        self.assertTrue(table.split.eq("selection").all())
        self.assertFalse((self.root / "selection/results_420.csv").exists())
        self.assertEqual(len(json.loads((self.root / "selection/capacity_audit.json").read_text())["fits"]), 180)

    def test_final_report_writes_84_and_420_without_modifying_controls(self):
        cfg, _ = self.fixture()
        before = _sha(cfg["point_table_path"])
        with patch("unidream.experiments.wm_lstm_grid_report._gate", return_value=("reg", {})), \
             patch("unidream.experiments.wm_lstm_grid_report._read_panel", side_effect=self.fake_reader):
            complete = write_lstm_grid_report(self.root, cfg, include_test=True)
        self.assertEqual(complete["combined_cells"], 420)
        self.assertEqual(len(pd.read_csv(self.root / "final/results_84.csv")), 84)
        self.assertEqual(len(pd.read_csv(self.root / "final/results_420.csv")), 420)
        self.assertEqual(_sha(cfg["point_table_path"]), before)
        contract = json.loads((self.root / "final/report_contract.json").read_text())
        self.assertFalse(contract["new_significance_tests_included"])
        self.assertTrue((self.root / "final/index420.html").exists())

    def test_incomplete_lstm_grid_is_rejected_before_prediction_access(self):
        cfg, registration = self.fixture()
        registration["neural_tasks"].pop()
        (self.root / "preregistration.json").write_text(json.dumps(registration))
        with patch("unidream.experiments.wm_lstm_grid_report._gate", return_value=("reg", {})), \
             patch("unidream.experiments.wm_lstm_grid_report._read_panel") as read:
            with self.assertRaisesRegex(ValueError, "exactly180"):
                write_lstm_grid_report(self.root, cfg)
            read.assert_not_called()

    def test_duplicate_or_missing_result_cells_are_rejected(self):
        table = pd.DataFrame([{"group": g, "model": "lstm", "task": t, "formulation": f, "primary_value": .1}
            for g in "ABCD" for t in TASKS for f in FORMULATIONS])
        _validate(table, ("lstm",), 84)
        with self.assertRaisesRegex(ValueError, "exactly84"):
            _validate(pd.concat([table.iloc[:-1], table.iloc[:1]]), ("lstm",), 84)


if __name__ == "__main__":
    unittest.main()
