"""Joint-grid scope, exact controls, chronology and artifact barriers."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments import wm_joint_grid_benchmark as benchmark
from unidream.experiments.wm_full_grid_evaluation import fit_label_contract
from unidream.experiments.wm_seven_task_evaluation import TASKS


def fixture_contract():
    n = 900
    actual = {t: np.linspace(-.02, .02, n)[:, None] for t in TASKS}
    for t in ("variance", "downside"): actual[t] = np.linspace(.00001, .03, n)[:, None]
    actual["crash"] = np.tile([0., 0., 1.], 300)[:, None]
    actual["dd_excess"] = np.r_[np.zeros(300), np.linspace(.001, .02, 600)][:, None]
    return actual, fit_label_contract(actual, np.arange(n))


class JointGridBenchmarkTests(unittest.TestCase):
    def test_scope_is_all_four_inputs_fifteen_folds_joint_and_report_only(self):
        cfg = benchmark.load_config()
        self.assertEqual(cfg["groups"], list("ABCD"))
        self.assertEqual(cfg["folds"], list(range(15)))
        self.assertEqual(cfg["formulations"], ["joint"])
        self.assertEqual(cfg["complete_formulations"], ["regression", "classification", "joint"])
        self.assertFalse(cfg["test_selection_allowed"])
        self.assertEqual(cfg["classical_joint"], "reused_independent_heads")
        for key, value in (("groups", ["D"]), ("test_selection_allowed", True),
                           ("joint_loss_weights", {"regression": .1, "classification": .9}),
                           ("classical_joint", "shared_lightgbm")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.yaml"; path.write_text(json.dumps({**cfg, key: value}))
                with self.assertRaises(ValueError): benchmark.load_config(path)

    def test_runtime_poll_does_not_block_for_over_a_minute(self):
        cfg = benchmark.load_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"; path.write_text(json.dumps({**cfg, "parent_poll_seconds": 3600}))
            with self.assertRaises(ValueError): benchmark.load_config(path)

    def test_training_mask_excludes_missing_and_boundary_crossing_labels(self):
        index = pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC")
        arrays = {"index_ns": index.asi8, "label_end_ns": index.asi8[:, None] + 16 * 15 * 60 * 10**9,
                  "mask": np.ones((6, 1), bool)}
        arrays["mask"][2, 0] = False
        mask = benchmark.training_mask(arrays, np.arange(6)[None, :],
            {"train_start": "2024-01-01T00:15:00Z", "train_end": "2024-01-01T04:45:00Z"})
        np.testing.assert_array_equal(mask, np.array([False, True, False, False, False, False])[None, :, None])

    def test_numpy_joint_selector_is_exact_half_of_each_frozen_objective(self):
        actual, contract = fixture_contract()
        actual = {t: values[:20] for t, values in actual.items()}
        regression = {t: values.copy() for t, values in actual.items()}
        classification = {t: np.ones((20, contract[t]["n_classes"])) / contract[t]["n_classes"] for t in TASKS}
        scales = {t: [.02] for t in TASKS if t != "crash"}
        result = benchmark.supervised_objective(actual, {"regression": regression, "classification": classification}, contract, scales)
        expected = sum(benchmark.full_grid.supervised_objective(actual, values, mode, contract, scales)["score"]
            for mode, values in (("regression", regression), ("classification", classification))) / 2
        self.assertAlmostEqual(result["score"], expected)

    def _classical_frame(self):
        actual, contract = fixture_contract(); arrays = {**actual, "index_ns": pd.date_range("2024-01-01", periods=900, freq="15min", tz="UTC").asi8}
        frames = []
        for model in ("linear", "lightgbm"):
            for mode in ("regression", "classification"):
                values = {t: actual[t][:5].copy() if mode == "regression" else
                    np.ones((5, contract[t]["n_classes"])) / contract[t]["n_classes"] for t in TASKS}
                frames.append(benchmark.full_grid.prediction_frame(arrays, np.arange(5), values,
                    {"fold": 0, "group": "D", "seed": 7, "model": model, "formulation": mode}, "selection", contract))
        return pd.concat(frames, ignore_index=True)

    def test_classical_joint_controls_exactly_reuse_both_native_heads(self):
        frame = self._classical_frame(); joint = benchmark.combine_classical_heads(frame)
        self.assertEqual(len(joint), 10)
        self.assertEqual(set(joint.formulation), {"joint"})
        self.assertFalse(joint.shared_parameters.any())
        self.assertEqual(set(joint.implementation), {"reused_independent_heads"})
        for model in ("linear", "lightgbm"):
            result = joint.loc[joint.model == model].reset_index(drop=True)
            scalar = frame.loc[(frame.model == model) & (frame.formulation == "regression")].reset_index(drop=True)
            cls = frame.loc[(frame.model == model) & (frame.formulation == "classification")].reset_index(drop=True)
            for task in TASKS:
                np.testing.assert_array_equal(result["scalar_" + task], scalar["scalar_" + task])
                for c in range(3): np.testing.assert_array_equal(result[f"p_{task}_{c}"], cls[f"p_{task}_{c}"])

    def test_classical_control_rejects_misalignment_outcomes_and_invalid_probabilities(self):
        for column, value in (("decision_id", 999), ("actual_return", 20.), ("p_return_0", -1.)):
            with self.subTest(column=column):
                frame = self._classical_frame()
                row = frame.index[(frame.model == "linear") & (frame.formulation == "classification")][0]
                frame.loc[row, column] = value
                with self.assertRaises(ValueError): benchmark.combine_classical_heads(frame)

    def test_kernel_task_claim_rejects_duplicate_live_work_and_allows_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            with benchmark.task_claim(dest):
                with self.assertRaisesRegex(RuntimeError, "already claimed"):
                    with benchmark.task_claim(dest): pass
            with benchmark.task_claim(dest): pass

    def test_missing_fit_is_not_retrained_once_test_has_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "test_barrier.json").write_text("{}")
            cfg = {"output_dir": str(root)}; task = {"job": "missing"}
            with patch.object(benchmark, "load_config", return_value=cfg), \
                    patch.object(benchmark, "verify", return_value=({}, {}, {})), \
                    patch.object(benchmark.parent, "arrays_for") as arrays:
                with self.assertRaisesRegex(ValueError, "cannot be retrained"):
                    benchmark.train_neural("unused", task)
            arrays.assert_not_called()

    def test_test_worker_cannot_load_market_data_before_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"output_dir": tmp}
            with patch.object(benchmark, "load_config", return_value=cfg), \
                    patch.object(benchmark, "verify", return_value=({"neural_tasks": [], "classical_tasks": []}, {}, {})), \
                    patch.object(benchmark.parent, "arrays_for") as arrays:
                with self.assertRaisesRegex(ValueError, "barrier closed"):
                    benchmark.evaluate_task("unused", "neural", {"job": "missing"})
            arrays.assert_not_called()

    def test_all_registered_fits_required_for_global_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing fit"):
                benchmark.frozen_fit_manifest({"output_dir": tmp}, {"neural_tasks": [{"job": "one"}], "classical_tasks": []})

    def test_parent_wait_polls_without_reading_test_predictions(self):
        cfg = {"parent_poll_seconds": 1}
        with patch.object(benchmark, "parent_ready", side_effect=[False, False, True]), \
                patch.object(benchmark, "status") as status, patch.object(benchmark.time, "sleep") as sleep, \
                patch.object(benchmark.pd, "read_parquet") as read:
            benchmark.wait_for_parent(cfg, require_complete=True)
        self.assertEqual(sleep.call_count, 2); self.assertEqual(status.call_count, 2)
        read.assert_not_called()

    def test_parent_completion_is_required_after_all_own_fits_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out / "preregistration.json").write_text("{}")
            (out / "selection").mkdir(); (out / "selection/joint_results_112.csv").write_text("fixed val metrics")
            (out / "validation_choices.json").write_text(json.dumps({"test_used": False,
                "registration_sha256": benchmark._sha(out / "preregistration.json"), "frozen_fit_sha256": {},
                "selection_report_sha256": benchmark._sha(out / "selection/joint_results_112.csv")}))
            cfg = {"output_dir": str(out)}
            with patch.object(benchmark, "verify", return_value=({}, {}, {})), \
                    patch.object(benchmark, "frozen_fit_manifest", return_value={}), \
                    patch.object(benchmark, "parent_ready", return_value=False):
                with self.assertRaisesRegex(ValueError, "parent224 incomplete"): benchmark.open_test_barrier(cfg)
            self.assertFalse((out / "test_barrier.json").exists())

    def test_changed_sources_are_rejected_before_parent_or_market_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "config.yaml"; source.write_text("fixed")
            (root / "preregistration.json").write_text(json.dumps({"config_sha256": "wrong", "source_sha256": {}}))
            with patch.object(benchmark.full_grid, "load_config") as load:
                with self.assertRaisesRegex(ValueError, "source/config changed"):
                    benchmark.verify({"output_dir": str(root), "config_path": str(source)})
            load.assert_not_called()


if __name__ == "__main__": unittest.main()
