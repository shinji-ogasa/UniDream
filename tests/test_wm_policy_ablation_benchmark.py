"""Whole-study freeze, paired initial states and Val-only control selection."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments import wm_policy_ablation_benchmark as runner


def fixture(root, fits=True):
    cfg = runner.load_config(runner.DEFAULT_CONFIG)
    cfg["output_dir"] = str(root)
    registration = {"config": cfg, "source_sha256": {"new_science.py": "frozen"}, "jobs": runner.jobs(cfg)}
    runner.write_json(root / "preregistration.json", registration)
    toy_array = root / "prepared/toy.npy"
    toy_array.parent.mkdir(parents=True, exist_ok=True)
    np.save(toy_array, np.array([1., 2., 3.]))
    runner.write_json(root / "prepared/cache_manifest.json", {"complete": True,
        "registration_sha256": runner.sha(root / "preregistration.json"), "source_artifact_sha256": {},
        "arrays": {"toy": {"path": str(toy_array), "sha256": runner.sha(toy_array)}}, "fold_arrays": {}})
    fold_arrays = {fold: {"train": np.arange(fold * 100, fold * 100 + 50, dtype=np.int64)} for fold in cfg["folds"]}
    if fits:
        for task in registration["jobs"]:
            path = root / task["fit_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            artifact = path.parent / "checkpoint.bin"
            artifact.write_bytes(b"selected fresh Train/Val fit")
            fit = {"complete": True, "test_used": False, "registration_sha256": runner.sha(root / "preregistration.json"),
                "artifact_sha256": {"checkpoint.bin": runner.sha(artifact)}}
            if task["kind"] == "policy":
                fit.update(normalizer_source_id=runner.train_source_id(fold_arrays[task["fold"]]),
                    metadata={"total_parameters": 12345}, initial_actor_state_sha256=f"init_fold{task['fold']}_seed{task['seed']}")
            runner.write_json(path, fit)
    return cfg, registration, fold_arrays


def frozen(cfg, registration, fold_arrays):
    with patch.object(runner.data, "fold_arrays", side_effect=lambda _cfg, fold: fold_arrays[fold]), \
         patch.object(runner, "_control_choices", return_value={"fixed": "validation_only"}):
        return runner.freeze_selection(cfg, registration)


class PolicyCoordinatorTests(unittest.TestCase):
    def test_job_DAG_has18_shared_world_models_and72_paired_policy_fits(self):
        cfg = runner.load_config(runner.DEFAULT_CONFIG)
        tasks = runner.jobs(cfg)
        self.assertEqual(len(tasks), 90)
        self.assertEqual(sum(task["kind"] == "wm" for task in tasks), 18)
        self.assertEqual(sum(task["kind"] == "policy" for task in tasks), 72)
        self.assertEqual(len({task["job"] for task in tasks}), 90)
        for task in tasks:
            if task.get("arm") in ("C", "D"):
                self.assertEqual(task["dependencies"], [f"wm_fold_{task['fold']:02d}_s{task['seed']}"])
            else:
                self.assertEqual(task["dependencies"], [])

    def test_registration_precedes_individual_data_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = runner.load_config(runner.DEFAULT_CONFIG)
            cfg["output_dir"] = str(root)
            def build(_cfg, registration):
                self.assertTrue((root / "preregistration.json").exists())
                self.assertEqual(json.loads((root / "preregistration.json").read_text()), registration)
                return {"cache_manifest_sha": "cached", "n_heads": 1, "forecast_dim": 3, "n_context_valid": 512}
            with patch.object(runner, "source_hashes", return_value={"new_science.py": "frozen"}), \
                 patch.object(runner.data, "inspect_sources", return_value={"folds": [{"fold": fold} for fold in cfg["folds"]]}), \
                 patch.object(runner.data, "build_cache", side_effect=build):
                result = runner.prepare(cfg)
            self.assertFalse(result["independently_unused_holdout"])
            self.assertEqual(len(result["jobs"]), 90)

    def test_configuration_source_and_task_grid_freeze_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, registration, _ = fixture(Path(temp), fits=False)
            with patch.object(runner, "source_hashes", return_value={"new_science.py": "frozen"}):
                self.assertEqual(runner.verify(cfg), registration)
                changed = copy.deepcopy(cfg)
                changed["accounting"]["cost_rate"] = 0.
                with self.assertRaisesRegex(ValueError, "configuration or source changed"):
                    runner.verify(changed)
                changed_reg = copy.deepcopy(registration)
                changed_reg["jobs"].pop()
                runner.write_json(Path(cfg["output_dir"]) / "preregistration.json", changed_reg)
                with self.assertRaisesRegex(ValueError, "task grid changed"):
                    runner.verify(cfg)
            with patch.object(runner, "source_hashes", return_value={"new_science.py": "modified"}):
                with self.assertRaisesRegex(ValueError, "configuration or source changed"):
                    runner.verify(cfg)

    def test_all90_fit_hashes_and72_Train_normalizers_freeze_before_Test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, registration, folds = fixture(root)
            barrier = frozen(cfg, registration, folds)
            self.assertEqual(len(barrier["frozen_fit_sha256"]), 90)
            provenance = json.loads((root / "normalizer_provenance.json").read_text())
            self.assertEqual(len(provenance["normalizers"]), 72)
            choices = json.loads((root / "validation_choices.json").read_text())
            self.assertFalse(choices["test_used"])
            with patch.object(runner, "source_hashes", return_value={"new_science.py": "frozen"}):
                self.assertEqual(runner.verify_test_barrier(cfg), barrier)

    def test_partial_barrier_or_late_fit_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, registration, folds = fixture(root)
            barrier = frozen(cfg, registration, folds)
            with patch.object(runner, "source_hashes", return_value={"new_science.py": "frozen"}):
                partial = copy.deepcopy(barrier)
                partial["frozen_fit_sha256"].pop(next(iter(partial["frozen_fit_sha256"])))
                runner.write_json(root / "test_barrier.json", partial)
                with self.assertRaisesRegex(ValueError, "every registered"):
                    runner.verify_test_barrier(cfg)
                runner.write_json(root / "test_barrier.json", barrier)
                path = root / registration["jobs"][0]["fit_path"]
                path.write_text('{"changed_after_freeze":true}')
                with self.assertRaisesRegex(ValueError, "fit changed"):
                    runner.verify_test_barrier(cfg)

    def test_missing_fits_prevent_global_Test_opening(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, registration, folds = fixture(root)
            (root / registration["jobs"][-1]["fit_path"]).unlink()
            with self.assertRaises((FileNotFoundError, ValueError)):
                frozen(cfg, registration, folds)
            self.assertFalse((root / "test_barrier.json").exists())

    def test_prepared_array_mutation_is_caught_before_Test_barrier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, registration, folds = fixture(root)
            np.save(root / "prepared/toy.npy", np.array([99.]))
            with self.assertRaisesRegex(ValueError, "policy cache or immutable forecast source changed"):
                frozen(cfg, registration, folds)
            self.assertFalse((root / "test_barrier.json").exists())

    def test_arm_capacity_and_paired_initial_actor_weights_must_match(self):
        for field in ("initial_actor_state_sha256", "total_parameters"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cfg, registration, folds = fixture(root)
                task = next(task for task in registration["jobs"] if task.get("arm") == "D")
                path = root / task["fit_path"]
                fit = json.loads(path.read_text())
                if field == "total_parameters":
                    fit["metadata"][field] += 1
                else:
                    fit[field] = "different_initial_weights"
                runner.write_json(path, fit)
                with self.assertRaisesRegex(ValueError, "capacity and paired initial weights"):
                    frozen(cfg, registration, folds)
                self.assertFalse((root / "test_barrier.json").exists())

    def test_Val_selected_control_choices_ignore_all_Test_values(self):
        cfg = runner.load_config(runner.DEFAULT_CONFIG)
        cfg["folds"] = [0]
        n = 100
        arrays = {"index_ns": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC").astype("int64").to_numpy(),
            "bar_log_return": np.random.default_rng(17).normal(.0001, .001, n), "ewma_variance": np.linspace(.0001, .0002, n)}
        folds = {"train": np.arange(60), "selection": np.arange(60, 80), "test": np.arange(80, 100)}
        with patch.object(runner.data, "load_arrays", return_value=arrays), \
             patch.object(runner.data, "fold_arrays", return_value=folds):
            first = runner._control_choices(cfg)
            arrays["bar_log_return"][folds["test"]] = -5.
            arrays["ewma_variance"][folds["test"]] = 100000.
            second = runner._control_choices(cfg)
        self.assertEqual(first, second)
        self.assertFalse(first["0"]["constant"]["test_used"])
        self.assertFalse(first["0"]["ewma"]["test_used"])

    def test_upstream_WM_must_complete_before_C_or_D_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, registration, _ = fixture(root, fits=False)
            task = next(task for task in registration["jobs"] if task.get("arm") == "C")
            with patch.object(runner, "verify", return_value=registration):
                with self.assertRaisesRegex(ValueError, "upstream WM is not complete"):
                    runner._worker(cfg, task, evaluation_phase=False)


if __name__ == "__main__":
    unittest.main()
