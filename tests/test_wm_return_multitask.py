"""Chronology, return-only labels, active heads and global artifact gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_return_multitask as benchmark


class ReturnMultitaskBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_fixed_scope_and_shared_budget_config(self):
        cfg = benchmark.load_config()
        self.assertEqual(cfg["arms"], ["reg_only", "class_only", "joint"])
        self.assertEqual(cfg["groups"], ["D"])
        self.assertEqual(cfg["folds"], list(range(15)))
        self.assertFalse(cfg["other_supervised_tasks"])
        self.assertFalse(cfg["test_selection_allowed"])
        for key, value in (("other_supervised_tasks", True), ("test_selection_allowed", True),
                           ("arms", ["joint"]), ("folds", [0, 7, 14]),
                           ("joint_loss_weights", {"normalized_return_mse": 1., "return_cross_entropy": 1.})):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.yaml"; path.write_text(json.dumps({**cfg, key: value}))
                with self.assertRaises(ValueError): benchmark.load_config(path)

    def test_labels_and_train_projection_contract_ignore_held_out_returns(self):
        train = np.tile([-.02, 0., .02], 300)
        whole = np.r_[train, -100., 100.][:, None]
        ids = np.arange(len(train))
        first = benchmark.train_label_contract(whole, ids)
        whole[-2:] = [[-1e6], [1e6]]
        self.assertEqual(first, benchmark.train_label_contract(whole, ids))
        theta = first["theta"]
        np.testing.assert_array_equal(benchmark.return_labels(np.array([-2*theta, -theta, 0, theta, 2*theta]), theta), [0, 1, 1, 1, 2])
        np.testing.assert_allclose(first["priors"], [1/3, 1/3, 1/3])
        np.testing.assert_allclose(first["train_class_return_means"], [-.02, 0., .02])
        with self.assertRaisesRegex(ValueError, "support"):
            benchmark.train_label_contract(np.full((1000, 1), .01), np.arange(1000))

    def test_training_tokens_require_train_start_validity_and_strict_future_label_end(self):
        index = pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC")
        arrays = {"index_ns": index.asi8, "label_end_ns": index.asi8[:, None] + 16*15*60*10**9,
                  "mask": np.ones((6, 1), bool)}
        arrays["mask"][2] = False
        fold = {"train_start": "2024-01-01T00:15:00Z", "train_end": "2024-01-01T04:45:00Z"}
        valid = benchmark.training_mask(arrays, np.arange(6)[None, :], fold)
        np.testing.assert_array_equal(valid, np.array([False, True, False, False, False, False])[None, :, None])

    def test_arm_objectives_use_only_the_registered_active_heads_and_joint_weights(self):
        actual = np.array([-.03, 0., .03])[:, None]
        result = {"return": np.array([-.02, .01, .01])[:, None],
                  "class_probabilities": np.array([[.7, .2, .1], [.2, .6, .2], [.1, .3, .6]])[:, None, :]}
        contract = {"rms": .02, "theta": .01}
        mse = np.mean(((result["return"] - actual) / .02) ** 2)
        ce = -np.log([.7, .6, .6]).mean()
        self.assertAlmostEqual(benchmark.supervised_objective(actual, {"return": result["return"]}, "reg_only", contract)["score"], mse)
        self.assertAlmostEqual(benchmark.supervised_objective(actual, {"class_probabilities": result["class_probabilities"]}, "class_only", contract)["score"], ce)
        joint = benchmark.supervised_objective(actual, result, "joint", contract)
        self.assertEqual(set(joint["parts"]), {"return_mse", "return_ce"})
        self.assertAlmostEqual(joint["score"], .5*mse + .5*ce)
        with self.assertRaisesRegex(ValueError, "probabilities"):
            benchmark.supervised_objective(actual, {**result, "class_probabilities": np.ones((3, 3))}, "joint", contract)

    def test_numpy_checkpoint_objective_matches_each_models_active_supervised_loss(self):
        from unidream.experiments.wm_return_multitask_models import DirectReturnMultitask, WMReturnMultitask
        model_cfg = {"context": 8, "d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32,
                     "encoder_hidden": 16, "n_categoricals": 3, "n_classes": 4, "dropout": 0.}
        torch.manual_seed(37)
        x, actual = torch.randn(2, 8, 10), torch.randn(2, 8, 1) * .02
        contract = {"rms": .02, "theta": .01}
        labels = torch.from_numpy(benchmark.return_labels(actual.numpy(), contract["theta"]))
        for cls in (DirectReturnMultitask, WMReturnMultitask):
            for arm in benchmark.ARMS:
                with self.subTest(model=cls.__name__, arm=arm):
                    model = cls(10, model_cfg, {"return": [.02]}, arm).eval()
                    prediction = model(x)
                    _, parts = model.loss(x, x[..., :5], {"return": actual}, torch.ones_like(actual, dtype=torch.bool),
                                          torch.zeros(2, 8), labels)
                    objective = benchmark.supervised_objective(actual.numpy(),
                        {key: value.detach().numpy() for key, value in prediction.items()}, arm, contract)
                    self.assertAlmostEqual(objective["score"], float(parts["supervised"]), places=5)
                    for key, value in objective["parts"].items():
                        self.assertAlmostEqual(value, float(parts[key]), places=5)

    def test_frames_do_not_present_inactive_heads_as_trained_forecasts(self):
        arrays = {"return": np.array([-.02, 0., .02])[:, None],
                  "index_ns": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC").asi8}
        contract = {"rms": .02, "theta": .01, "priors": [.2, .6, .2], "train_class_return_means": [-.03, 0., .03]}
        predictions = {"return": arrays["return"] / 2,
                       "class_probabilities": np.array([[.7, .2, .1], [.2, .6, .2], [.1, .3, .6]])[:, None, :]}
        for arm in benchmark.ARMS:
            task = {"fold": 0, "seed": 7, "model": "wm", "arm": arm}
            frame = benchmark.prediction_frame(arrays, np.arange(3), predictions, task, "selection", contract)
            self.assertEqual(frame.actual_class.tolist(), [0, 1, 2])
            self.assertTrue((frame.train_class_return_mean_0 == -.03).all())
            self.assertTrue((frame.train_prior_1 == .6).all())
            self.assertEqual(frame.pred_return.isna().all(), arm == "class_only")
            self.assertEqual(frame[["p0", "p1", "p2"]].isna().all().all(), arm == "reg_only")

    def fixture(self, root):
        (root / "preregistration.json").write_text('{"study":"return-only fixture"}')
        cfg = {"output_dir": str(root)}
        tasks = [{"fold": 0, "group": "D", "seed": 7, "model": "direct", "arm": arm, "job": f"fold_00_D_direct_{arm}_s7"}
                 for arm in benchmark.ARMS]
        registration = {"neural_tasks": tasks}
        for task in tasks:
            dest = benchmark.directory(cfg, task); dest.mkdir(parents=True)
            (dest / "research_model.pt").write_bytes(b"selected frozen checkpoint")
            (dest / "validation_predictions.parquet").write_bytes(b"frozen Val forecasts")
            fit = {**task, "registration_sha256": benchmark._sha(root / "preregistration.json"),
                   "checkpoint_sha256": benchmark._sha(dest / "research_model.pt"),
                   "validation_predictions_sha256": benchmark._sha(dest / "validation_predictions.parquet"),
                   "initialization_sha256": "same-init-all-arms"}
            (dest / "fit.json").write_text(json.dumps(fit))
        (root / "selection").mkdir()
        (root / "selection/metrics.csv").write_text("frozen,Val\n1,2\n")
        (root / "selection/day_statistics.csv").write_text("frozen,daily\n1,2\n")
        choices = {"registration_sha256": benchmark._sha(root / "preregistration.json"), "test_used": False,
                   "calibration": "identity",
                   "frozen_fit_sha256": benchmark.frozen_fit_manifest(cfg, registration),
                   "selection_metrics_sha256": benchmark._sha(root / "selection/metrics.csv"),
                   "selection_day_statistics_sha256": benchmark._sha(root / "selection/day_statistics.csv")}
        (root / "validation_choices.json").write_text(json.dumps(choices))
        return cfg, registration

    def test_global_barrier_requires_every_fit_and_frozen_val_choices(self):
        for missing in ("neural/fold_00_D_direct_class_only_s7/fit.json", "validation_choices.json", "selection/metrics.csv"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); cfg, reg = self.fixture(root); (root / missing).unlink()
                with patch.object(benchmark, "verify", return_value=(reg, {})):
                    with self.assertRaises((ValueError, FileNotFoundError)): benchmark.open_test_barrier(cfg)
                self.assertFalse((root / "test_barrier.json").exists())

    def test_test_worker_cannot_load_data_checkpoint_or_predictions_before_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, reg = self.fixture(Path(tmp))
            with patch.object(benchmark, "load_config", return_value=cfg), patch.object(benchmark, "verify", return_value=(reg, {})), \
                    patch.object(benchmark.parent, "arrays_for") as arrays, patch.object(benchmark.torch, "load") as checkpoint, \
                    patch.object(benchmark.pd, "read_parquet") as predictions:
                with self.assertRaisesRegex(ValueError, "global Test barrier closed"):
                    benchmark.evaluate_task("unused", reg["neural_tasks"][0])
            arrays.assert_not_called(); checkpoint.assert_not_called(); predictions.assert_not_called()

    def test_changed_other_fit_is_rejected_before_loading_test_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, reg = self.fixture(root)
            with patch.object(benchmark, "verify", return_value=(reg, {})):
                benchmark.open_test_barrier(cfg)
                other = benchmark.directory(cfg, reg["neural_tasks"][-1]) / "fit.json"
                other.write_text(other.read_text() + " ")
                with patch.object(benchmark, "load_config", return_value=cfg), patch.object(benchmark.parent, "arrays_for") as arrays:
                    with self.assertRaisesRegex(ValueError, "global fit"):
                        benchmark.evaluate_task("unused", reg["neural_tasks"][0])
                arrays.assert_not_called()

    def test_changed_checkpoint_is_rejected_before_loading_test_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, reg = self.fixture(root)
            with patch.object(benchmark, "verify", return_value=(reg, {})):
                benchmark.open_test_barrier(cfg)
                (benchmark.directory(cfg, reg["neural_tasks"][0]) / "research_model.pt").write_bytes(b"mutated")
                with patch.object(benchmark, "load_config", return_value=cfg), patch.object(benchmark.parent, "arrays_for") as arrays:
                    with self.assertRaisesRegex(ValueError, "model/Val"):
                        benchmark.evaluate_task("unused", reg["neural_tasks"][0])
                arrays.assert_not_called()

    def test_arm_initialization_mismatch_and_task_metadata_mismatch_fail_closed(self):
        for field, value in (("initialization_sha256", "different-init"), ("arm", "class_only")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                cfg, reg = self.fixture(Path(tmp)); task = reg["neural_tasks"][0]
                path = benchmark.directory(cfg, task) / "fit.json"
                fit = json.loads(path.read_text()); fit[field] = value; path.write_text(json.dumps(fit))
                with self.assertRaises(ValueError): benchmark.frozen_fit_manifest(cfg, reg)

    def test_completed_fit_cache_is_verified_without_loading_data_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, reg = self.fixture(Path(tmp)); task = reg["neural_tasks"][0]
            with patch.object(benchmark, "load_config", return_value=cfg), patch.object(benchmark, "verify", return_value=(reg, {})), \
                    patch.object(benchmark.parent, "arrays_for") as arrays, patch.object(benchmark, "worker_setup") as setup:
                self.assertEqual(benchmark.train_neural("unused", task), task)
            arrays.assert_not_called(); setup.assert_not_called()

    def test_frozen_val_choices_are_verified_and_preserved_for_test_inference_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, reg = self.fixture(root)
            path = root / "validation_choices.json"; original = path.read_bytes()
            with patch.object(benchmark, "verify", return_value=(reg, {})), \
                    patch.object(benchmark, "prediction_files") as read_predictions:
                benchmark.open_test_barrier(cfg)
                benchmark.freeze_validation_choices(cfg)
                benchmark.open_test_barrier(cfg)
            self.assertEqual(path.read_bytes(), original)
            read_predictions.assert_not_called()

    def test_missing_fit_is_never_retrained_after_test_barrier_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, reg = self.fixture(root); task = reg["neural_tasks"][0]
            with patch.object(benchmark, "verify", return_value=(reg, {})):
                benchmark.open_test_barrier(cfg)
                (benchmark.directory(cfg, task) / "fit.json").unlink()
                with patch.object(benchmark, "load_config", return_value=cfg), \
                        patch.object(benchmark.parent, "arrays_for") as arrays, \
                        patch.object(benchmark, "run_tasks") as jobs:
                    with self.assertRaisesRegex(ValueError, "cannot be retrained"):
                        benchmark.train_neural("unused", task)
                    with self.assertRaisesRegex(ValueError, "cannot be retrained"):
                        benchmark.run_fits(cfg)
                arrays.assert_not_called(); jobs.assert_not_called()


if __name__ == "__main__": unittest.main()
