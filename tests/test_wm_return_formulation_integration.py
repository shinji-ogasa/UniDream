"""Independent shape/objective and immutable Test-gate integration checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from unidream.experiments import wm_return_formulation as runner
from unidream.experiments.wm_return_formulation_models import (
    OTHER_TASKS, DirectReturnClassifier, WMReturnClassifier,
)
from unidream.experiments.wm_seven_task_models import TASKS


class ReturnFormulationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_real_model_prediction_shapes_and_numpy_selection_match_masked_loss(self):
        cfg = {"device": "cpu", "training": {"evaluation_batch_size": 2},
               "model": {"context": 8, "d_model": 16, "n_heads": 2,
                         "n_layers": 1, "d_ff": 32, "encoder_hidden": 16,
                         "n_categoricals": 3, "n_classes": 4, "dropout": .1}}
        scales = {key: [.001 if key == "variance" else .02] for key in TASKS if key != "crash"}
        torch.manual_seed(123)
        x = torch.randn(4, 8, 63)
        arrays = {key: np.array([.01, -.005, .02, -.01])[:, None] for key in TASKS}
        for key in ("variance", "downside", "dd_excess"):
            arrays[key] = np.abs(arrays[key]) + .0001
        arrays["crash"] = np.array([0., 0., 1., 1.])[:, None]
        theta = .005
        labels = runner.make_return_labels(arrays["return"][:, 0], theta)
        token_labels = torch.from_numpy(np.repeat(labels[:, None, None], 8, axis=1))
        targets = {key: torch.from_numpy(np.repeat(value[:, None], 8, axis=1)).float()
                   for key, value in arrays.items()}
        valid = torch.zeros(4, 8, 1, dtype=torch.bool)
        valid[:, -1] = True

        def fake_input(_arrays, ids, _data, _group, _context):
            value = x.numpy()[ids]
            return value, value[..., :5]

        for cls in (DirectReturnClassifier, WMReturnClassifier):
            model = cls(63, cfg["model"], scales).eval()
            with patch.object(runner, "_input", side_effect=fake_input):
                result = runner.predict(model, arrays, np.arange(4), {}, cfg)
            self.assertEqual(set(result), {*OTHER_TASKS, "class_probabilities"})
            self.assertEqual(result["class_probabilities"].shape, (4, 1, 3))
            for key in OTHER_TASKS:
                self.assertEqual(result[key].shape, (4, 1))
            measured = runner.objective(arrays, np.arange(4), result, scales, theta)
            _, parts = model.loss(x, x[..., :5], targets, valid, torch.zeros(4, 8), token_labels)
            self.assertAlmostEqual(measured["score"], float(parts["supervised"]), places=5)
            for key in (*OTHER_TASKS, "classification"):
                self.assertAlmostEqual(measured["parts"][key], float(parts[key]), places=5)
            self.assertNotIn("return", measured["parts"])

    def _fixture(self, root):
        parent = root / "parent"
        child = root / "child"
        parent.mkdir(); child.mkdir()
        preg = {"neural_tasks": [{"job": "p_neural"}], "classical_tasks": [{"job": "p_classical"}]}
        (parent / "preregistration.json").write_text(json.dumps(preg))
        (parent / "validation_choices.json").write_text('{"frozen": true}')
        manifest = {}
        for kind, job in (("neural", "p_neural"), ("classical", "p_classical")):
            path = parent / kind / job / "fit.json"
            path.parent.mkdir(parents=True)
            path.write_text('{"parent fit": true}')
            manifest[str(path.relative_to(parent))] = runner._sha(path)
        parent_barrier = {"registration_sha256": runner._sha(parent / "preregistration.json"),
                          "validation_choices_sha256": runner._sha(parent / "validation_choices.json"),
                          "frozen_fit_sha256": manifest}
        (parent / "test_barrier.json").write_text(json.dumps(parent_barrier))
        task = {"job": "fold_00_D_direct_s7", "fold": 0, "model": "direct", "seed": 7, "group": "D"}
        reg = {"neural_tasks": [task], "classical_tasks": [], "folds": [{"fold": 0}]}
        (child / "preregistration.json").write_text(json.dumps(reg))
        directory = child / "neural" / task["job"]
        directory.mkdir(parents=True)
        (directory / "research_model.pt").write_bytes(b"frozen research checkpoint")
        (directory / "validation_predictions.parquet").write_bytes(b"frozen validation forecasts")
        fit = {"registration_sha256": runner._sha(child / "preregistration.json"),
               "checkpoint_sha256": runner._sha(directory / "research_model.pt"),
               "validation_predictions_sha256": runner._sha(directory / "validation_predictions.parquet")}
        (directory / "fit.json").write_text(json.dumps(fit))
        frozen = {"registration_sha256": runner._sha(child / "preregistration.json"),
                  "frozen_fit_sha256": {str((directory / "fit.json").relative_to(child)): runner._sha(directory / "fit.json")},
                  "reference_validation_sha256": {}, "calibrators": {}}
        (child / "calibrators.json").write_text(json.dumps(frozen))
        cfg = {"output_dir": str(child), "parent_config": "fixture-parent.yaml"}
        pcfg = {"output_dir": str(parent)}
        return cfg, pcfg, reg, task

    def test_parent_gate_requires_complete_manifest_and_detects_fit_mutation(self):
        for change in ("incomplete_manifest", "changed_fit"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                cfg, pcfg, _, _ = self._fixture(Path(tmp))
                if change == "incomplete_manifest":
                    path = Path(pcfg["output_dir"]) / "test_barrier.json"
                    barrier = json.loads(path.read_text())
                    barrier["frozen_fit_sha256"].pop("classical/p_classical/fit.json")
                    path.write_text(json.dumps(barrier))
                else:
                    (Path(pcfg["output_dir"]) / "neural/p_neural/fit.json").write_text('{"changed": true}')
                with patch.object(runner.parent, "load_config", return_value=pcfg):
                    with self.assertRaisesRegex(ValueError, "complete fit manifest|fit missing or changed"):
                        runner.enforce_parent_test_barrier(cfg)

    def test_child_gate_cannot_open_before_parent_test_is_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, pcfg, reg, _ = self._fixture(Path(tmp))
            (Path(pcfg["output_dir"]) / "test_barrier.json").unlink()
            with patch.object(runner, "verify", return_value=(reg, pcfg)), patch.object(runner.parent, "load_config", return_value=pcfg):
                with self.assertRaisesRegex(ValueError, "parent global Test barrier is closed"):
                    runner.open_test_barrier(cfg)
            self.assertFalse((Path(cfg["output_dir"]) / "test_barrier.json").exists())

    def test_test_worker_rejects_changed_fit_before_priority_or_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, pcfg, reg, task = self._fixture(Path(tmp))
            with patch.object(runner, "verify", return_value=(reg, pcfg)), patch.object(runner.parent, "load_config", return_value=pcfg):
                runner.open_test_barrier(cfg)
                path = runner.task_dir(cfg, "neural", task) / "fit.json"
                fit = json.loads(path.read_text())
                fit["changed_after_calibration"] = True
                path.write_text(json.dumps(fit))
                with patch.object(runner, "load_config", return_value=cfg), patch.object(runner, "low_priority_threads") as priority, patch.object(runner.parent, "arrays_for") as load_arrays:
                    with self.assertRaisesRegex(ValueError, "fit changed after classification freeze"):
                        runner.evaluate_task("fixture", "neural", task)
                    priority.assert_not_called()
                    load_arrays.assert_not_called()

    def test_test_worker_rejects_changed_parent_barrier_or_calibration(self):
        for target in ("parent_barrier", "calibration"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                cfg, pcfg, reg, task = self._fixture(Path(tmp))
                with patch.object(runner, "verify", return_value=(reg, pcfg)), patch.object(runner.parent, "load_config", return_value=pcfg):
                    runner.open_test_barrier(cfg)
                    path = Path(pcfg["output_dir"]) / "test_barrier.json" if target == "parent_barrier" else Path(cfg["output_dir"]) / "calibrators.json"
                    value = json.loads(path.read_text())
                    value["changed_after_child_barrier"] = True
                    path.write_text(json.dumps(value))
                    with patch.object(runner, "load_config", return_value=cfg), patch.object(runner, "low_priority_threads") as priority, patch.object(runner.parent, "arrays_for") as load_arrays:
                        with self.assertRaisesRegex(ValueError, "changed after Test barrier"):
                            runner.evaluate_task("fixture", "neural", task)
                        priority.assert_not_called()
                        load_arrays.assert_not_called()

    def test_combined_test_report_must_pass_gate_before_reading_predictions(self):
        with patch.object(runner, "verify", return_value=({}, {})), patch.object(runner, "open_test_barrier", side_effect=ValueError("closed fixture gate")) as gate, patch.object(runner.pd, "read_parquet") as read:
            with self.assertRaisesRegex(ValueError, "closed fixture gate"):
                runner.combined_predictions({"output_dir": "unused"}, include_test=True)
            gate.assert_called_once()
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
