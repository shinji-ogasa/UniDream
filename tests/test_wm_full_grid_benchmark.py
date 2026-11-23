"""Cross-module supervision, chronology and artifact barriers without fitting."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_full_grid_benchmark as benchmark
from unidream.experiments.wm_full_grid_evaluation import fit_label_contract, make_task_labels
from unidream.experiments.wm_full_grid_models import DirectFullGrid, WMFullGrid
from unidream.experiments.wm_seven_task_evaluation import TASKS


MODEL = {"context": 8, "d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32,
         "encoder_hidden": 16, "n_categoricals": 3, "n_classes": 4, "dropout": 0.}


def label_contract():
    n = 900
    targets = {
        "return": np.tile([-.02, 0., .02], n // 3), "variance": np.linspace(.00001, .001, n),
        "downside": np.linspace(0., .03, n), "crash": np.tile([0., 0., 1.], n // 3),
        "dd_excess": np.r_[np.zeros(n // 3), np.linspace(.001, .02, 2 * n // 3)],
        "advantage": np.tile([-.001, 0., .001], n // 3), "recovery": np.tile([-.6, 0., .6], n // 3),
    }
    return fit_label_contract({key: value[:, None] for key, value in targets.items()}, np.arange(n))


class FullGridBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_training_tokens_respect_start_end_missingness_and_strict_label_boundary(self):
        index = pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC")
        arrays = {"index_ns": index.asi8, "label_end_ns": index.asi8[:, None] + 16 * 15 * 60 * 10**9,
                  "mask": np.ones((6, 1), bool)}
        arrays["mask"][2, 0] = False
        positions = np.arange(6)[None, :]
        fold = {"train_start": "2024-01-01T00:15:00Z", "train_end": "2024-01-01T04:45:00Z"}
        mask = benchmark.training_mask(arrays, positions, fold)
        np.testing.assert_array_equal(mask, np.array([False, True, False, False, False, False])[None, :, None])
        self.assertFalse(mask[0, 3, 0])  # A label ending exactly at the boundary is excluded.

    def test_fixed_config_has_every_input_formulation_and_fold_and_one_seed(self):
        cfg = benchmark.load_config(benchmark.DEFAULT_CONFIG)
        self.assertEqual(cfg["groups"], ["A", "B", "C", "D"])
        self.assertEqual(set(cfg["formulations"]), {"regression", "classification"})
        self.assertEqual(sorted(cfg["folds"]), list(range(15)))
        self.assertEqual(cfg.get("seeds", [cfg.get("seed")]), [7])
        self.assertFalse(cfg["test_selection_allowed"])
        for key, value in (("groups", ["D"]), ("formulations", ["classification"]),
                           ("folds", [0, 7, 14]), ("test_selection_allowed", True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.yaml"
                path.write_text(json.dumps({**cfg, key: value}))
                with self.assertRaises(ValueError):
                    benchmark.load_config(path)

    def test_numpy_selector_matches_torch_supervision_in_both_formulations(self):
        torch.manual_seed(113)
        x = torch.randn(2, 8, 10)
        scales = {task: [.001 if task == "variance" else .02] for task in TASKS if task != "crash"}
        actual = {task: torch.randn(2, 8, 1) * .01 for task in TASKS}
        for task in ("variance", "downside", "dd_excess"):
            actual[task] = actual[task].abs() + .0001
        actual["crash"] = (actual["crash"] > 0).float()
        contract = label_contract()
        for formulation in ("regression", "classification"):
            targets = actual if formulation == "regression" else {
                task: torch.from_numpy(make_task_labels(values.numpy(), task, contract)) for task, values in actual.items()}
            for cls in (DirectFullGrid, WMFullGrid):
                model = cls(10, MODEL, scales, formulation).eval()
                prediction = model(x)
                _, parts = model.loss(x, x[..., :5], targets, torch.ones(2, 8, 1, dtype=torch.bool), torch.zeros(2, 8))
                numpy_actual = {task: values.numpy().reshape(-1, 1) for task, values in actual.items()}
                numpy_prediction = {task: values.detach().numpy().reshape(-1, 1, values.shape[-1])
                                    if formulation == "classification" else values.detach().numpy().reshape(-1, 1)
                                    for task, values in prediction.items()}
                result = benchmark.supervised_objective(numpy_actual, numpy_prediction, formulation, contract, scales)
                self.assertAlmostEqual(result["score"], float(parts["supervised"]), places=5)
                for task in TASKS:
                    self.assertAlmostEqual(result["parts"][task], float(parts[task]), places=5)

    def _barrier_fixture(self, root):
        root.mkdir(exist_ok=True)
        (root / "preregistration.json").write_text('{"study":"unit fixture"}')
        cfg = {"output_dir": str(root)}
        registration = {"neural_tasks": [{"job": "neural_fixture"}],
                        "classical_tasks": [{"job": "classical_fixture"}]}
        registration_sha = benchmark._sha(root / "preregistration.json")
        for kind, key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
            directory = root / kind / registration[key][0]["job"]
            directory.mkdir(parents=True)
            model = directory / ("research_model.pt" if kind == "neural" else "models.pkl")
            model.write_bytes(b"selected frozen model")
            validation = directory / "validation_predictions.parquet"
            validation.write_bytes(b"selected frozen Val predictions")
            fit = {"registration_sha256": registration_sha,
                   "checkpoint_sha256" if kind == "neural" else "model_sha256": benchmark._sha(model),
                   "validation_predictions_sha256": benchmark._sha(validation)}
            (directory / "fit.json").write_text(json.dumps(fit))
        (root / "validation_choices.json").write_text(json.dumps({"test_used": False, "registration_sha256": registration_sha}))
        return cfg, registration

    def test_global_barrier_requires_all_fits_and_validated_val_choices(self):
        for missing in ("classical/classical_fixture/fit.json", "validation_choices.json"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, reg = self._barrier_fixture(root)
                (root / missing).unlink()
                with patch.object(benchmark, "verify", return_value=(reg, {})):
                    with self.assertRaises(ValueError):
                        benchmark.open_test_barrier(cfg)
                self.assertFalse((root / "test_barrier.json").exists())

    def test_test_worker_cannot_load_data_or_predictions_before_global_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, reg = self._barrier_fixture(root)
            with patch.object(benchmark, "load_config", return_value=cfg), \
                    patch.object(benchmark, "verify", return_value=(reg, {})), \
                    patch.object(benchmark.parent, "arrays_for") as arrays, \
                    patch.object(benchmark.pd, "read_parquet") as predictions:
                with self.assertRaisesRegex(ValueError, "global Test barrier closed"):
                    benchmark.evaluate_task("unused", "neural", reg["neural_tasks"][0])
            arrays.assert_not_called()
            predictions.assert_not_called()

    def test_global_manifest_is_complete_and_remains_frozen_after_opening(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, reg = self._barrier_fixture(root)
            with patch.object(benchmark, "verify", return_value=(reg, {})):
                first = benchmark.open_test_barrier(cfg)
                self.assertEqual(len(first["frozen_fit_sha256"]), 2)
                self.assertFalse(first["test_selection_allowed"])
                benchmark.test_access(cfg, "neural", reg["neural_tasks"][0])
                fit = root / "classical/classical_fixture/fit.json"
                value = json.loads(fit.read_text())
                value["changed_after_global_barrier"] = True
                fit.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    benchmark.test_access(cfg, "neural", reg["neural_tasks"][0])

    def test_changed_model_is_rejected_before_loading_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, reg = self._barrier_fixture(root)
            with patch.object(benchmark, "verify", return_value=(reg, {})):
                benchmark.open_test_barrier(cfg)
                (root / "neural/neural_fixture/research_model.pt").write_bytes(b"changed checkpoint")
                with patch.object(benchmark, "load_config", return_value=cfg), \
                        patch.object(benchmark.parent, "arrays_for") as arrays:
                    with self.assertRaisesRegex(ValueError, "selected model/Val"):
                        benchmark.evaluate_task("unused", "neural", reg["neural_tasks"][0])
                arrays.assert_not_called()

    def test_parent_classical_prediction_hash_is_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "classical/fold_00_D"
            directory.mkdir(parents=True)
            source = directory / "validation_predictions.parquet"
            pd.DataFrame({"split": ["earlystop", "selection"], "decision_id": [1, 2]}).to_parquet(source, index=False)
            (directory / "fit.json").write_text(json.dumps({"validation_predictions_sha256": benchmark._sha(source)}))
            selected = benchmark.read_parent_classical({"output_dir": str(root)}, {"job": "fold_00_D"}, "selection")
            self.assertEqual(selected.decision_id.tolist(), [2])
            source.write_bytes(source.read_bytes() + b"mutated frozen predictions")
            with patch.object(benchmark.pd, "read_parquet") as read:
                with self.assertRaisesRegex(ValueError, "parent predictions changed"):
                    benchmark.read_parent_classical({"output_dir": str(root)}, {"job": "fold_00_D"}, "selection")
                read.assert_not_called()

    def test_parent_test_prediction_requires_matching_parent_barrier_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "classical/fold_00_D"
            directory.mkdir(parents=True)
            (root / "test_barrier.json").write_text('{"parent_test":"frozen"}')
            source = directory / "test_predictions.parquet"
            pd.DataFrame({"split": ["test"], "decision_id": [3]}).to_parquet(source, index=False)
            (directory / "fit.json").write_text(json.dumps({"validation_predictions_sha256": "not needed for Test"}))
            evaluation = {"barrier_sha256": benchmark._sha(root / "test_barrier.json"), "prediction_sha256": benchmark._sha(source)}
            (directory / "evaluation.json").write_text(json.dumps(evaluation))
            with patch.object(benchmark.pd, "read_parquet") as read:
                with self.assertRaisesRegex(ValueError, "new-study Test barrier context"):
                    benchmark.read_parent_classical({"output_dir": str(root)}, {"job": "fold_00_D"}, "test")
                read.assert_not_called()
            study_cfg = {"output_dir": "separate complete full-grid study"}
            with patch.object(benchmark, "test_access", return_value={}) as gate:
                selected = benchmark.read_parent_classical({"output_dir": str(root)}, {"job": "fold_00_D"}, "test", study_cfg=study_cfg)
                self.assertEqual(selected.decision_id.tolist(), [3])
                gate.assert_called_once_with(study_cfg, "classical", {"job": "fold_00_D"})
            (root / "test_barrier.json").write_text('{"parent_test":"changed"}')
            with patch.object(benchmark.pd, "read_parquet") as read, patch.object(benchmark, "test_access", return_value={}):
                with self.assertRaisesRegex(ValueError, "parent Test artifact provenance"):
                    benchmark.read_parent_classical({"output_dir": str(root)}, {"job": "fold_00_D"}, "test", study_cfg=study_cfg)
                read.assert_not_called()

    def test_classical_reuse_keeps_six_regressions_and_crash_classifier_exact(self):
        contract = label_contract()
        n = 4
        arrays = {task: np.zeros((n, 1)) for task in TASKS}
        arrays["variance"][:] = .0001
        arrays["index_ns"] = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC").asi8
        ids = np.arange(n)
        reference_frames = []
        for family in ("linear", "lightgbm"):
            row = pd.DataFrame({"model": family, "decision_id": ids})
            for task in TASKS:
                row["actual_" + task] = arrays[task][:, 0]
                row["pred_" + task] = np.linspace(.002, .005, n) if task != "crash" else [.2, .4, .6, .8]
            reference_frames.append(row)
        old = pd.concat(reference_frames, ignore_index=True)
        fresh = {"classification": {family: {task: np.tile([.2, .3, .5], (n, 1)) for task in TASKS if task != "crash"}
                                    for family in ("linear", "lightgbm")},
                 "regression": {family: {"crash": np.array([.1, .2, .7, .8])} for family in ("linear", "lightgbm")}}
        with patch.object(benchmark, "predict_grid_classical", return_value=fresh), \
                patch.object(benchmark.parent, "tabular_x", return_value=np.zeros((n, 2))), \
                patch.object(benchmark, "read_parent_classical", return_value=old):
            frames = benchmark.classical_frames({}, arrays, ids, {}, {"fold": 0, "group": "D", "job": "fold_00_D"},
                "selection", {"formulations": ["regression", "classification"]}, {}, {"label_contract": contract})
        self.assertEqual(len(frames), 4)
        for frame in frames:
            family = frame.model.iloc[0]
            old_family = old.loc[old.model == family]
            if frame.formulation.iloc[0] == "regression":
                for task in TASKS:
                    expected = fresh["regression"][family]["crash"] if task == "crash" else old_family["pred_" + task].to_numpy()
                    np.testing.assert_array_equal(frame["scalar_" + task], expected)
            else:
                np.testing.assert_array_equal(frame.p_crash_1, old_family.pred_crash)
                np.testing.assert_array_equal(frame.p_crash_0, 1 - old_family.pred_crash)
                self.assertTrue(frame.p_crash_2.isna().all())
                for task in TASKS:
                    if task != "crash":
                        np.testing.assert_array_equal(frame[[f"p_{task}_{i}" for i in range(3)]], fresh["classification"][family][task])


if __name__ == "__main__":
    unittest.main()
