from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_policy_ablation_world as world
from unidream.experiments.wm_seven_task_models import TASKS


def fixture(directory):
    rng = np.random.default_rng(73)
    n = 100
    times = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC").astype("int64").to_numpy()
    decisions = np.arange(12, 92)
    targets = rng.normal(size=(n, 7)) * .01
    targets[:, 1] = np.exp(rng.normal(size=n)) * 1e-4
    targets[:, 2] = np.abs(targets[:, 2])
    targets[:, 3] = rng.binomial(1, .2, n)
    targets[:, 4] = np.maximum(targets[:, 2] - .012, 0)
    arrays = {"decision_id": decisions, "index_ns": times[decisions],
        "context_valid": np.ones(len(decisions), bool), "x_grid": rng.normal(size=(n, 5)).astype(np.float32),
        "raw_close": np.exp(np.cumsum(rng.normal(size=n) * .001)), "raw_index_ns": times,
        "wm_mask_grid": np.ones(n, bool), "wm_targets_grid": targets,
        "wm_label_end_ns": times + 2 * 15 * 60 * 10**9}
    fold = {"train": np.flatnonzero(decisions < 51), "val": np.flatnonzero((decisions >= 55) & (decisions < 75)),
        "test": np.flatnonzero(decisions >= 75), "wm_train": np.flatnonzero(decisions < 51),
        "wm_earlystop": np.flatnonzero((decisions >= 55) & (decisions < 62)),
        "x_mean": np.zeros(5), "x_std": np.ones(5)}
    for task in TASKS:
        fold["scale_" + task] = np.array([1e-4 if task == "variance" else .1])
    fold["scale_crash_prior"] = np.array([.2])
    meta = {"fold": 0, "train_start": pd.Timestamp(times[10], tz="UTC").isoformat(),
        "train_end": pd.Timestamp(times[55], tz="UTC").isoformat(),
        "val_start": pd.Timestamp(times[55], tz="UTC").isoformat(),
        "val_end": pd.Timestamp(times[75], tz="UTC").isoformat()}
    cfg = {"output_dir": directory, "device": "cpu", "runtime": {"cpu_threads": 1}, "policy": {"normalizer_clip": 10.},
        "wm_model": {"context": 4, "d_model": 16, "n_heads": 4, "n_layers": 1, "d_ff": 32,
            "encoder_hidden": 16, "n_categoricals": 3, "n_classes": 4, "dropout": 0.},
        "wm_training": {"max_steps": 2, "min_steps": 1, "patience_steps": 1, "validation_every": 1,
            "relative_improvement": .0005, "batch_size": 2, "evaluation_batch_size": 4,
            "selection_rows": 3, "learning_rate": .0001, "weight_decay": .0001, "gradient_clip": 10.}}
    root = Path(directory)
    (root / "prepared").mkdir()
    registration = root / "preregistration.json"
    registration.write_text('{"synthetic_test": true}\n')
    registration_sha = hashlib.sha256(registration.read_bytes()).hexdigest()
    (root / "prepared/cache_manifest.json").write_text(json.dumps({"registration_sha256": registration_sha}))
    np.savez(root / "prepared/fold_00.npz", **fold)
    return cfg, arrays, fold, meta, registration_sha


class PolicyWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_every_context_token_masks_pretrain_and_boundary_crossing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            _, arrays, _, meta, _ = fixture(directory)
            positions = np.arange(5, 60).reshape(5, 11)
            valid, reward = world.training_supervision(arrays, positions, meta)
            expected = (positions >= 10) & (positions + 2 < 55)
            np.testing.assert_array_equal(valid[..., 0], expected)
            self.assertTrue(np.isnan(reward[positions < 10]).all())
            self.assertTrue(np.isnan(reward[positions >= 55]).all())
            target = np.log(arrays["raw_close"][positions] / arrays["raw_close"][positions - 1])
            np.testing.assert_allclose(reward[(positions >= 10) & (positions < 55)],
                target[(positions >= 10) & (positions < 55)], rtol=1e-6, atol=1e-10)

    def test_registration_is_required_before_any_data_access(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _, _, _ = fixture(directory)
            with patch.object(world.data, "load_arrays") as reader:
                with self.assertRaisesRegex(ValueError, "registration"):
                    world.train_wm(cfg, 0, 7, "wrong-registration")
                reader.assert_not_called()

    def test_miniature_fresh_fit_selects_val_and_encodes_train_val_only(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, arrays, fold, meta, registration_sha = fixture(directory)
            with patch.object(world.data, "load_arrays", return_value=arrays), \
                    patch.object(world.data, "fold_arrays", return_value=fold), \
                    patch.object(world.data, "policy_folds", return_value=[meta]):
                result = world.train_wm(cfg, 0, 7, registration_sha)
            self.assertTrue(result["complete"])
            self.assertFalse(result["test_used"])
            self.assertEqual(result["latent_dim"], 28)
            self.assertTrue(all(result["supervised_gradient_steps"].values()))
            self.assertGreater(result["supervised_target_count"], 0)
            cache = np.load(result["latent_file"])
            eligible = np.unique(np.concatenate((fold["train"], fold["val"])))
            self.assertTrue(np.isfinite(cache[eligible]).all())
            self.assertTrue(np.isnan(cache[np.setdiff1d(np.arange(len(cache)), eligible)]).all())
            self.assertTrue(np.isnan(cache[fold["test"]]).all())
            with np.load(result["latent_norm_file"]) as normalization:
                np.testing.assert_allclose(normalization["mean"], cache[fold["train"]].mean(0), atol=1e-6)
            selected = json.loads((Path(cfg["output_dir"]) / "world_models/fold_00_s7/progress.json").read_text())
            self.assertGreaterEqual(selected["step"], 1)
            # Resume verifies immutable artifacts rather than fitting again.
            with patch.object(world.data, "load_arrays") as reader:
                resumed = world.train_wm(cfg, 0, 7, registration_sha)
                reader.assert_not_called()
            self.assertEqual(result, resumed)

    def test_latent_normalization_ignores_validation_and_test_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latents.npy"
            values = np.array([[1., 2.], [3., 4.], [1e9, -1e9], [np.nan, np.nan]], np.float32)
            np.save(path, values)
            mean, std = world._latent_normalizer(path, np.array([0, 1]))
            np.testing.assert_array_equal(mean, [2., 3.])
            np.testing.assert_array_equal(std, [1., 1.])

    def test_global_test_barrier_is_verified_before_any_test_data_read(self):
        from unidream.experiments import wm_policy_ablation_benchmark as benchmark
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _, _, registration_sha = fixture(directory)
            with patch.object(benchmark, "verify_test_barrier", side_effect=ValueError("global Test closed")), \
                    patch.object(world.data, "load_arrays") as reader:
                with self.assertRaisesRegex(ValueError, "global Test closed"):
                    world.encode_test_latents(cfg, 0, 7, registration_sha)
                reader.assert_not_called()

    def test_test_cache_is_separate_checks_barrier_and_does_not_mutate_fit(self):
        from unidream.experiments import wm_policy_ablation_benchmark as benchmark
        with tempfile.TemporaryDirectory() as directory:
            cfg, arrays, fold, meta, registration_sha = fixture(directory)
            with patch.object(world.data, "load_arrays", return_value=arrays), \
                    patch.object(world.data, "fold_arrays", return_value=fold), \
                    patch.object(world.data, "policy_folds", return_value=[meta]):
                world.train_wm(cfg, 0, 7, registration_sha)
                root = Path(directory)
                fit_path = root / "world_models/fold_00_s7/fit.json"
                original_fit = fit_path.read_bytes()
                barrier = {"registration_sha256": registration_sha, "test_selection_allowed": False,
                    "frozen_fit_sha256": {str(fit_path.relative_to(root)): world._sha(fit_path)}}
                (root / "test_barrier.json").write_text(json.dumps(barrier))
                with patch.object(benchmark, "verify_test_barrier", return_value=barrier) as guard:
                    encoded = world.encode_test_latents(cfg, 0, 7, registration_sha)
                    guard.assert_called_once_with(cfg)
                    self.assertTrue(np.isfinite(encoded[fold["test"]]).all())
                    other = np.setdiff1d(np.arange(len(encoded)), fold["test"])
                    self.assertTrue(np.isnan(encoded[other]).all())
                    self.assertEqual(original_fit, fit_path.read_bytes())
                    again = world.encode_test_latents(cfg, 0, 7, registration_sha)
                    np.testing.assert_allclose(encoded, again, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
