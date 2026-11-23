"""Seed-expansion provenance guards plus actual tiny-data PPO continuation."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_seed_robustness_training as training
from unidream.experiments.wm_policy_ablation_models import PolicyActorCritic


class SeedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.kernel = {"output_dir": str(self.root), "device": "cpu",
            "splits": {"start": "2023-01-01", "end": "2026-10-01"},
            "policy": {"context": 4, "normalizer_clip": 10., "purge_bars": 64, "horizon": 16,
                "train_years": 2, "val_months": 3, "test_months": 3, "wm_update_during_policy_training": False},
            "actor": {"context": 4, "lstm_hidden_size": 4, "lstm_layers": 1, "projection_dim": 4,
                "hidden_size": 8, "dropout": 0., "min_log_std": -4., "max_log_std": 0.},
            "accounting": {"position_min": .5, "position_max": 1.12, "initial_position": 1., "max_position_step": .08,
                "cost_rate": .00055, "borrow_apr": .1, "annual_bars": 35040, "drawdown_penalty": .25},
            "rl_training": {"rollout_batch_size": 2, "rollout_horizon": 4, "minibatch_size": 4, "ppo_epochs": 1,
                "learning_rate": .001, "critic_learning_rate": .001, "gamma": .99, "gae_lambda": .95,
                "clip_ratio": .2, "value_coefficient": .5, "entropy_coefficient": .001,
                "bc_prior_coefficient": .01, "reward_scale": 100., "gradient_clip": 10.}}
        stage = lambda end, floor: {"through_update": end, "position_min": floor, "bc_prior_coefficient": .01}
        self.cfg = {"output_dir": str(self.root), "kernel_config": self.kernel, "new_seeds": [3],
            "runtime": {"cpu_threads": 1}, "conditions": ["C_floor_gradual", "A_floor_gradual", "C_control"],
            "condition_definitions": {"C_floor_gradual": {"arm": "C", "condition": "floor_gradual"},
                "A_floor_gradual": {"arm": "A", "condition": "floor_gradual"}, "C_control": {"arm": "C", "condition": "control"}},
            "continuation": {"max_updates": 200, "selection_updates": [125, 150, 175, 200], "selection_rows": 8},
            "schedules": {"control": [stage(200, .5)], "floor_gradual": [stage(50, .5), stage(100, .25), stage(200, 0.)]}}
        self.tasks = [{"kind": "continuation", "job": f"fold_00_C_s3_{condition}", "fold": 0, "seed": 3,
            "arm": "C", "condition": condition, "fit_path": f"neural/fold_00_C_s3_{condition}/fit.json"}
            for condition in ("control", "floor_gradual")]
        self.wm_task = {"kind": "wm", "job": "wm_fold_00_s3", "fold": 0, "seed": 3}
        self.policy_task = {"kind": "policy", "job": "policy_fold_00_C_s3", "fold": 0, "seed": 3, "arm": "C"}
        self.registration = {"config": self.cfg, "jobs": self.tasks + [self.wm_task, self.policy_task]}
        training.write_json(self.root / "preregistration.json", self.registration)
        self.registration_sha = training.sha(self.root / "preregistration.json")
        self.source = self.root / "neural/fold_00_C_s3"
        self.source.mkdir(parents=True)
        torch.manual_seed(6)
        model = PolicyActorCritic(2, 3, 5, self.kernel, "C")
        training.original._save_checkpoint(self.source / "bc.pt", model, self.registration_sha, 1, "bc")
        with torch.no_grad():
            model.policy_head[-1].bias.sub_(.3)
        training.original._save_checkpoint(self.source / "rl.pt", model, self.registration_sha, 2, "rl")
        rng, n = np.random.default_rng(93), 140
        times = np.r_[pd.date_range("2023-01-01", periods=80, freq="15min", tz="UTC").asi8,
            pd.date_range("2025-01-01", periods=40, freq="15min", tz="UTC").asi8,
            pd.date_range("2025-04-01", periods=20, freq="15min", tz="UTC").asi8]
        arrays = {"decision_id": np.arange(n) + 3, "index_ns": times, "x_grid": rng.normal(size=(n+4, 2)).astype(np.float32),
            "f": rng.normal(size=(n, 3)).astype(np.float32), "context_valid": np.ones(n, bool), "bar_log_return": rng.normal(0, .005, n)}
        fold = {"train": np.arange(80), "earlystop": np.arange(80, 100), "val": np.arange(80, 120), "test": np.arange(120, 140),
            "x_mean": np.zeros(2), "x_std": np.ones(2), "f_mean": np.zeros(3), "f_std": np.ones(3)}
        self.bundle = {"cfg": self.kernel, "arrays": arrays, "fold": fold, "latent": rng.normal(size=(n, 5)).astype(np.float32),
            "latent_norm": {"mean": np.zeros(5), "std": np.ones(5)}, "latent_dim": 5, "source_directory": self.source,
            "source_artifact_sha256": {str((self.source / name).resolve()): training.sha(self.source / name) for name in ("bc.pt", "rl.pt")}}

    def test_schedules_preserve_budget_costs_and_configuration(self):
        frozen = copy.deepcopy(self.cfg)
        for update, floor in ((1, .5), (50, .5), (51, .25), (100, .25), (101, 0.), (200, 0.)):
            settings = training.condition_config(self.cfg, "floor_gradual", update)
            self.assertEqual(settings["accounting"]["position_min"], floor)
            self.assertEqual(settings["accounting"]["cost_rate"], .00055)
            self.assertEqual(settings["rl_training"]["bc_prior_coefficient"], .01)
            self.assertEqual(Path(settings["output_dir"]), self.root)
        self.assertEqual(self.cfg, frozen)
        changed = copy.deepcopy(self.cfg)
        changed["kernel_config"]["output_dir"] = str(self.root / "old-study")
        with self.assertRaisesRegex(ValueError, "new study"):
            training.kernel_config(changed)
        for value in (0, 201, True, 10.5):
            with self.assertRaises(ValueError):
                training.condition_config(self.cfg, "floor_gradual", value)

    def test_worker_dispatches_new_independent_wm_and_policy_without_old_test_entry(self):
        with patch.object(training.world, "train_wm", return_value={"wm": True}) as wm, \
                patch.object(training.original, "train_policy", return_value={"policy": True}) as policy:
            self.assertEqual(training.worker(self.cfg, self.wm_task, self.registration_sha), {"wm": True})
            self.assertEqual(training.worker(self.cfg, self.policy_task, self.registration_sha), {"policy": True})
            self.assertEqual(wm.call_args.args[1:], (0, 3, self.registration_sha))
            self.assertEqual(policy.call_args.args[1:], (0, 3, "C", self.registration_sha))
            self.assertEqual(wm.call_args.args[0]["output_dir"], str(self.root))

    def test_unregistered_worker_cannot_read_any_market_or_checkpoint(self):
        with patch.object(training.data, "load_arrays") as arrays, patch.object(training, "_bundle") as bundle:
            with self.assertRaisesRegex(ValueError, "registration required"):
                training.worker(self.cfg, self.wm_task, "wrong")
            arrays.assert_not_called()
            bundle.assert_not_called()
        with self.assertRaisesRegex(ValueError, "differs from registration"):
            training.worker(self.cfg, {**self.wm_task, "seed": 7}, self.registration_sha)

    def test_real_ppo_common_initialization_and_starts_with200_updates(self):
        fits = []
        with patch.object(training, "_bundle", return_value=self.bundle), redirect_stdout(io.StringIO()):
            for task in self.tasks:
                fit = training.train_continuation(self.cfg, task, self.registration_sha)
                fits.append(fit)
                self.assertEqual(fit["updates"], 200)
                self.assertEqual(fit["historical_transitions"], 1600)
                self.assertTrue(fit["bc_prior_immutable"])
                self.assertFalse(fit["test_used"])
                self.assertFalse(fit["optimizer_moments_inherited"])
                self.assertGreater(fit["updates_with_policy_parameter_changes"], 0)
                history = json.loads((self.root / "neural" / task["job"] / "training_history.json").read_text())
                self.assertEqual([row["update"] for row in history if "validation_score" in row], [125, 150, 175, 200])
                self.assertTrue(all(row["fresh_on_policy_rollout"] for row in history))
                with patch.object(training, "_bundle", side_effect=AssertionError("must reuse completed fit")):
                    self.assertEqual(training.train_continuation(self.cfg, task, self.registration_sha), fit)
        for key in ("initial_state_sha256", "immutable_bc_prior_state_sha256", "market_start_sequence_sha256"):
            self.assertEqual(len({fit[key] for fit in fits}), 1)
        self.assertNotEqual(fits[0]["initial_state_sha256"], fits[0]["final_selected_state_sha256"])

    def test_test_guard_precedes_any_latent_or_policy_forward(self):
        module = ModuleType("unidream.experiments.wm_seed_robustness_benchmark")
        module.verify_test_barrier = lambda cfg: (_ for _ in ()).throw(ValueError("global barrier closed"))
        with patch.dict("sys.modules", {module.__name__: module}), patch.object(training, "_bundle") as bundle:
            with self.assertRaisesRegex(ValueError, "global barrier closed"):
                training.evaluate_policy(self.cfg, self.tasks[0], self.registration_sha)
            bundle.assert_not_called()

    def test_report_only_ledger_and_immutable_test_cache(self):
        task = self.tasks[1]
        with patch.object(training, "_bundle", return_value=self.bundle), redirect_stdout(io.StringIO()):
            training.train_continuation(self.cfg, task, self.registration_sha)
        barrier = {"registration_sha256": self.registration_sha,
            "frozen_fit_sha256": {task["fit_path"]: training.sha(self.root / task["fit_path"])}}
        training.write_json(self.root / "test_barrier.json", barrier)
        module = ModuleType("unidream.experiments.wm_seed_robustness_benchmark")
        module.verify_test_barrier = lambda cfg: barrier
        with patch.dict("sys.modules", {module.__name__: module}), patch.object(training, "_bundle", return_value=self.bundle):
            result = training.evaluate_policy(self.cfg, task, self.registration_sha)
            self.assertTrue(result["test_report_only"])
            self.assertEqual(result["n_rows"], 20)
            path = self.root / "evaluations" / task["job"] / "executed_ledger.parquet"
            ledger = pd.read_parquet(path)
            self.assertTrue(ledger.split.eq("test").all())
            self.assertEqual(float(ledger.executed_position.iloc[0]), 1.)
            self.assertEqual(result["prediction_sha256"], training.sha(path))
            with patch.object(training, "_bundle", side_effect=AssertionError("cached Test must not forward")):
                self.assertEqual(training.evaluate_policy(self.cfg, task, self.registration_sha), result)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Test result changed"):
                training.evaluate_policy(self.cfg, task, self.registration_sha)


if __name__ == "__main__":
    unittest.main()
