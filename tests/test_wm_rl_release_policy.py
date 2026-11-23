"""Registered tiny-data continuation smoke; no frozen market data or refits."""
import copy
from contextlib import ExitStack, redirect_stdout
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

from unidream.experiments import wm_rl_release_policy as policy
from unidream.experiments.wm_policy_ablation_models import PolicyActorCritic


class ReleasePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        source_config_path = self.root / "source.yaml"
        source_config_path.write_text("toy source config")
        self.scfg = {"config_path": str(source_config_path), "output_dir": str(self.source), "device": "cpu",
            "splits": {"start": "2023-01-01", "end": "2026-10-01"},
            "policy": {"context": 4, "normalizer_clip": 10., "purge_bars": 64, "horizon": 16,
                "train_years": 2, "val_months": 3, "test_months": 3},
            "actor": {"context": 4, "lstm_hidden_size": 4, "lstm_layers": 1, "projection_dim": 4,
                "hidden_size": 8, "dropout": 0., "min_log_std": -4., "max_log_std": 0.},
            "wm_model": {"n_categoricals": 2, "n_classes": 2, "d_model": 1},
            "accounting": {"position_min": .5, "position_max": 1.12, "initial_position": 1., "max_position_step": .08,
                "cost_rate": .00055, "borrow_apr": .1, "annual_bars": 35040, "drawdown_penalty": .25},
            "rl_training": {"rollout_batch_size": 2, "rollout_horizon": 4, "minibatch_size": 4, "ppo_epochs": 1,
                "learning_rate": .001, "critic_learning_rate": .001, "gamma": .99, "gae_lambda": .95,
                "clip_ratio": .2, "value_coefficient": .5, "entropy_coefficient": .001,
                "bc_prior_coefficient": .01, "reward_scale": 100., "gradient_clip": 10.}}
        stage = lambda end, floor, prior: {"through_update": end, "position_min": floor, "bc_prior_coefficient": prior}
        self.cfg = {"output_dir": str(self.root / "output"), "source_config": str(source_config_path), "device": "cpu",
            "runtime": {"cpu_threads": 1}, "conditions": ["control", "floor_abrupt", "floor_gradual", "prior_abrupt", "prior_gradual"],
            "continuation": {"max_updates": 200, "selection_updates": [125, 150, 175, 200], "selection_rows": 8},
            "schedules": {"control": [stage(200, .5, .01)], "floor_abrupt": [stage(200, 0., .01)],
                "floor_gradual": [stage(50, .5, .01), stage(100, .25, .01), stage(200, 0., .01)],
                "prior_abrupt": [stage(200, .5, 0.)],
                "prior_gradual": [stage(50, .5, .01), stage(100, .5, .003), stage(200, .5, 0.)]}}
        self.tasks = [{"job": f"fold_00_C_s7_{condition}", "fold": 0, "seed": 7, "arm": "C", "condition": condition,
            "fit_path": f"neural/fold_00_C_s7_{condition}/fit.json"} for condition in self.cfg["conditions"]]
        rng = np.random.default_rng(93)
        n = 140
        times = np.r_[pd.date_range("2023-01-01", periods=80, freq="15min", tz="UTC").asi8,
            pd.date_range("2025-01-01", periods=40, freq="15min", tz="UTC").asi8,
            pd.date_range("2025-04-01", periods=20, freq="15min", tz="UTC").asi8]
        arrays = {"decision_id": np.arange(n)+3, "index_ns": times,
            "x_grid": rng.normal(size=(n+4, 2)).astype(np.float32), "f": rng.normal(size=(n, 3)).astype(np.float32),
            "context_valid": np.ones(n, bool), "bar_log_return": rng.normal(0, .005, n)}
        fold = {"train": np.arange(80), "earlystop": np.arange(80, 100), "val": np.arange(80, 120),
            "test": np.arange(120, 140), "x_mean": np.zeros(2), "x_std": np.ones(2), "f_mean": np.zeros(3), "f_std": np.ones(3)}
        self.directory = self.source / "neural/fold_00_C_s7"
        self.directory.mkdir(parents=True)
        (self.source / "preregistration.json").write_text('{}')
        source_sha = policy.sha(self.source / "preregistration.json")
        torch.manual_seed(6)
        model = PolicyActorCritic(2, 3, 5, self.scfg, "C")
        policy.original._save_checkpoint(self.directory/"bc.pt", model, source_sha, 1, "bc")
        with torch.no_grad():
            model.policy_head[-1].bias.sub_(.3)
        policy.original._save_checkpoint(self.directory/"rl.pt", model, source_sha, 2, "rl")
        (self.directory/"fit.json").write_text('{"complete": true}')
        names = [self.directory/name for name in ("fit.json", "rl.pt", "bc.pt")]
        provenance = {str(path.resolve()): policy.sha(path) for path in names}
        self.bundle = {"cfg": self.scfg, "arrays": arrays, "fold": fold,
            "latent": rng.normal(size=(n, 5)).astype(np.float32), "latent_norm": {"mean": np.zeros(5), "std": np.ones(5)},
            "latent_dim": 5, "source_directory": self.directory, "source_registration_sha256": source_sha,
            "source_fit": {}, "source_artifact_sha256": provenance}
        sources = {"source_root": str(self.source), "source_config": self.scfg,
            "artifact_sha256": {str(Path(name).relative_to(self.source)): digest for name, digest in provenance.items()}}
        output = Path(self.cfg["output_dir"])
        output.mkdir()
        self.registration = {"config": self.cfg, "jobs": self.tasks, "sources": sources}
        (output/"preregistration.json").write_text(json.dumps(self.registration))
        self.registration_sha = policy.sha(output/"preregistration.json")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(policy, "source_config", side_effect=lambda cfg: copy.deepcopy(self.scfg)))

    def task(self, condition):
        return next(t for t in self.tasks if t["condition"] == condition)

    def test_floor_and_prior_schedules_are_independent_exact_and_do_not_mutate_source(self):
        original = copy.deepcopy(self.scfg)
        for update, floor in ((1, .5), (50, .5), (51, .25), (100, .25), (101, 0.), (200, 0.)):
            cfg = policy.condition_cfg(self.cfg, "floor_gradual", update)
            self.assertEqual(cfg["accounting"]["position_min"], floor)
            self.assertEqual(cfg["rl_training"]["bc_prior_coefficient"], .01)
            self.assertEqual(cfg["output_dir"], self.scfg["output_dir"])
        for update, prior in ((1, .01), (50, .01), (51, .003), (100, .003), (101, 0.), (200, 0.)):
            cfg = policy.condition_cfg(self.cfg, "prior_gradual", update)
            self.assertEqual(cfg["rl_training"]["bc_prior_coefficient"], prior)
            self.assertEqual(cfg["accounting"]["position_min"], .5)
        self.assertEqual(self.scfg, original)
        self.assertEqual(policy.condition_cfg(self.cfg, "floor_abrupt")["accounting"]["position_min"], 0.)
        self.assertEqual(policy.condition_cfg(self.cfg, "prior_abrupt")["rl_training"]["bc_prior_coefficient"], 0.)

    def test_registration_guard_precedes_any_source_access(self):
        (Path(self.cfg["output_dir"])/"preregistration.json").unlink()
        with patch.object(policy, "_source_inputs") as load:
            with self.assertRaisesRegex(ValueError, "registration required"):
                policy.train_policy(self.cfg, self.tasks[0], "wrong")
            load.assert_not_called()

    def test_source_checkpoint_binding_rejects_changed_ancestor(self):
        altered = copy.deepcopy(self.registration)
        altered["sources"]["artifact_sha256"]["neural/fold_00_C_s7/rl.pt"] = "changed"
        with self.assertRaisesRegex(ValueError, "differs from"):
            policy._bind_source(altered, self.bundle)

    def test_all_branches_spend200_updates_with_common_starts_and_final_only_val(self):
        fits = []
        with patch.object(policy, "_source_inputs", return_value=self.bundle), redirect_stdout(io.StringIO()):
            for task in self.tasks:
                fit = policy.train_policy(self.cfg, task, self.registration_sha)
                fits.append(fit)
                path = Path(self.cfg["output_dir"])/"neural"/task["job"]
                history = json.loads((path/"training_history.json").read_text())
                self.assertEqual([row["update"] for row in history], list(range(1, 201)))
                self.assertEqual([row["update"] for row in history if "validation_score" in row], [125, 150, 175, 200])
                self.assertTrue(all(row["fresh_on_policy_rollout"] for row in history))
                self.assertEqual(fit["updates"], 200)
                self.assertEqual(fit["historical_transitions"], 1600)
                self.assertGreater(fit["updates_with_policy_gradient"], 0)
                self.assertGreater(fit["updates_with_policy_parameter_changes"], 0)
                self.assertFalse(fit["optimizer_moments_inherited"])
                self.assertFalse(fit["optimizer_reset_at_stage_boundary"])
                self.assertTrue(fit["bc_prior_immutable"])
                self.assertTrue(fit["wm_frozen"])
                self.assertFalse(fit["test_used"])
                self.assertIn("mean_exposure", fit["validation_metrics"])
                self.assertIn("mean_exposure", fit["full_validation_metrics"])
                frame = pd.read_parquet(path/"validation_predictions.parquet")
                self.assertTrue(frame.split.eq("val").all())
                self.assertTrue(frame.decision_id.isin(self.bundle["arrays"]["decision_id"][self.bundle["fold"]["val"]]).all())
                self.assertEqual(sum("gradient_coverage" in row for row in history), 2)
                with patch.object(policy, "_source_inputs", side_effect=AssertionError("cachedfit must not retrain")):
                    self.assertEqual(policy.train_policy(self.cfg, task, self.registration_sha), fit)
        self.assertEqual(len({f["initial_state_sha256"] for f in fits}), 1)
        self.assertEqual(len({f["market_start_sequence_sha256"] for f in fits}), 1)
        self.assertEqual(len({f["immutable_bc_prior_state_sha256"] for f in fits}), 1)
        self.assertEqual(policy.sha(self.directory/"rl.pt"), fits[0]["source_rl_checkpoint_sha256"])

    def test_budget_rejects_early_validation_or_incomplete_schedule(self):
        changed = copy.deepcopy(self.cfg)
        changed["continuation"]["selection_updates"] = [100, 125, 150, 200]
        with self.assertRaisesRegex(ValueError, "common125"):
            policy._budget(changed)
        changed = copy.deepcopy(self.cfg)
        changed["schedules"]["floor_gradual"][-1]["through_update"] = 150
        with self.assertRaisesRegex(ValueError, "exhaust"):
            policy._budget(changed)

    def test_test_forward_blocked_before_new_complete_global_barrier(self):
        module = ModuleType("unidream.experiments.wm_rl_release_benchmark")
        module.verify_test_barrier = lambda cfg: (_ for _ in ()).throw(ValueError("all180 barrier closed"))
        with patch.dict("sys.modules", {module.__name__: module}), patch.object(policy, "_source_inputs") as load:
            with self.assertRaisesRegex(ValueError, "barrier closed"):
                policy.evaluate_policy(self.cfg, self.tasks[0], self.registration_sha)
            load.assert_not_called()

    def test_synthetic_evaluation_writes_executed_contract_and_is_immutable(self):
        task = self.task("floor_gradual")
        with patch.object(policy, "_source_inputs", return_value=self.bundle), redirect_stdout(io.StringIO()):
            fit = policy.train_policy(self.cfg, task, self.registration_sha)
        output = Path(self.cfg["output_dir"])
        barrier = {"registration_sha256": self.registration_sha,
            "frozen_fit_sha256": {task["fit_path"]: policy.sha(output/task["fit_path"])}}
        (output/"selection_barrier.json").write_text(json.dumps(barrier))
        module = ModuleType("unidream.experiments.wm_rl_release_benchmark")
        module.verify_test_barrier = lambda cfg: barrier
        with patch.dict("sys.modules", {module.__name__: module}), \
                patch.object(policy, "_source_inputs", return_value=self.bundle):
            result = policy.evaluate_policy(self.cfg, task, self.registration_sha)
            self.assertTrue(result["complete"])
            self.assertTrue(result["test_report_only"])
            self.assertEqual(result["n_rows"], 20)
            self.assertEqual(result["final_accounting"]["position_min"], 0.)
            self.assertIn("mean_exposure", result["metrics"])
            path = output/"evaluations"/task["job"]/"executed_ledger.parquet"
            self.assertEqual(result["artifact_sha256"][path.name], policy.sha(path))
            frame = pd.read_parquet(path)
            self.assertTrue(frame.split.eq("test").all())
            self.assertTrue(frame.condition.eq("floor_gradual").all())
            self.assertEqual(float(frame.executed_position.iloc[0]), 1.)
            self.assertEqual(result["fit_sha256"], policy.sha(output/task["fit_path"]))
            with patch.object(policy, "_source_inputs", side_effect=AssertionError("cached Test must not forward")):
                self.assertEqual(policy.evaluate_policy(self.cfg, task, self.registration_sha), result)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Test result changed"):
                policy.evaluate_policy(self.cfg, task, self.registration_sha)


if __name__ == "__main__":
    unittest.main()
