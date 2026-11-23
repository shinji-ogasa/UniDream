"""Tiny synthetic end-to-end BC/PPO smoke and causal policy evaluation checks."""
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

from unidream.experiments import wm_policy_ablation_data as data
from unidream.experiments import wm_policy_ablation_policy as policy
from unidream.experiments.wm_policy_ablation_models import PolicyActorCritic


class PolicyTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {"output_dir": str(self.root), "device": "cpu", "arms": list("ABCD"), "folds": [0], "seeds": [7],
            "policy": {"context": 4, "normalizer_clip": 10.},
            "actor": {"context": 4, "lstm_hidden_size": 4, "lstm_layers": 1, "projection_dim": 4, "hidden_size": 8,
                "dropout": 0., "min_log_std": -4., "max_log_std": 0.},
            "wm_model": {"n_categoricals": 2, "n_classes": 2, "d_model": 1},
            "accounting": {"position_min": .5, "position_max": 1.12, "initial_position": 1., "max_position_step": .08,
                "cost_rate": .00055, "borrow_apr": .1, "annual_bars": 35040, "drawdown_penalty": .25},
            "teacher": {"horizon": 16, "position_grid": [.5, .7, .85, .94, 1., 1.06, 1.12]},
            "bc_training": {"max_steps": 2, "min_steps": 1, "patience_steps": 5, "validation_every": 1,
                "relative_improvement": .0005, "batch_size": 4, "learning_rate": .003, "weight_decay": 0.,
                "gradient_clip": 10., "selection_rows": 8},
            "rl_training": {"max_updates": 2, "min_updates": 1, "patience_updates": 5, "validation_every": 1,
                "relative_improvement": .0005, "selection_rows": 8, "rollout_batch_size": 2, "rollout_horizon": 4,
                "minibatch_size": 4, "ppo_epochs": 1, "learning_rate": .001, "critic_learning_rate": .001,
                "gamma": .99, "gae_lambda": .95, "clip_ratio": .2, "value_coefficient": .5,
                "entropy_coefficient": .001, "bc_prior_coefficient": .01, "reward_scale": 100., "gradient_clip": 10.}}
        rng = np.random.default_rng(173)
        n = 120
        self.arrays = {"decision_id": np.arange(n) + 3,
            "index_ns": pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC").asi8,
            "x_grid": rng.normal(size=(n+4, 2)).astype(np.float32), "f": rng.normal(size=(n, 3)).astype(np.float32),
            "context_valid": np.ones(n, bool), "bar_log_return": rng.normal(0, .005, n),
            "future_path_log_returns": np.cumsum(rng.normal(0, .005, (n, 16)), axis=1),
            "teacher_valid": np.ones(n, bool)}
        self.fold = {"train": np.arange(60), "earlystop": np.arange(60, 80), "selection": np.arange(80, 100),
            "val": np.arange(60, 100), "test": np.arange(100, 120), "x_mean": np.zeros(2), "x_std": np.ones(2),
            "f_mean": np.zeros(3), "f_std": np.ones(3)}
        self.latents = rng.normal(size=(n, 5)).astype(np.float32)
        self.norm = {"mean": np.zeros(5), "std": np.ones(5)}
        prepared = self.root / "prepared"
        prepared.mkdir()
        np.savez(prepared / "fold_00.npz", **self.fold)
        wm = self.root / "world_models/fold_00_s7"
        wm.mkdir(parents=True)
        (wm / "fit.json").write_text('{"complete": true}')

    def test_contiguous_starts_require_final_bootstrap_and_never_cross_gap(self):
        times = np.arange(12) * data.BAR_NS
        times[6:] += 3 * data.BAR_NS
        starts = policy.contiguous_starts(np.arange(12), times, 4)
        np.testing.assert_array_equal(starts, [0, 1, 6, 7])
        np.testing.assert_array_equal(policy.contiguous_starts(np.arange(4), times, 4), [])

    def test_all_arms_train_actual_bc_and_ppo_on_synthetic_data(self):
        with patch.object(policy.data, "load_arrays", return_value=self.arrays), \
                patch.object(policy.data, "fold_arrays", return_value=self.fold), \
                patch.object(policy, "_latent_data", return_value=(self.latents, self.norm, 5)):
            for arm in "ABCD":
                with self.subTest(arm=arm):
                    with redirect_stdout(io.StringIO()):
                        fit = policy.train_policy(self.cfg, 0, 7, arm, "synthetic-registration")
                    self.assertTrue(fit["complete"])
                    self.assertFalse(fit["test_used"])
                    self.assertGreater(fit["rl"]["real_policy_gradient_steps"], 0)
                    self.assertGreater(fit["rl"]["changed_policy_steps"], 0)
                    self.assertEqual(fit["rl"]["historical_transitions"], 16)
                    root = self.root / "neural" / f"fold_00_{arm}_s7"
                    frame = pd.read_parquet(root / "validation_predictions.parquet")
                    self.assertTrue(frame.split.eq("val").all())
                    self.assertTrue(frame.decision_id.isin(self.arrays["decision_id"][self.fold["val"]]).all())
                    self.assertEqual(set(frame.stage), {"bc", "rl"})
                    # Cached resume verifies immutable artifacts and does not refit.
                    with patch.object(policy.data, "load_arrays", side_effect=AssertionError("must not reload/refit")):
                        self.assertEqual(policy.train_policy(self.cfg, 0, 7, arm, "synthetic-registration"), fit)

    def test_current_return_cannot_change_current_action(self):
        torch.manual_seed(29)
        model = PolicyActorCritic(2, 3, 5, self.cfg, "A")
        ids = np.arange(12)
        first = policy.predict_requests(model, self.arrays, ids, self.fold, self.cfg)
        changed = dict(self.arrays)
        changed["bar_log_return"] = self.arrays["bar_log_return"].copy()
        changed["bar_log_return"][5] += .2
        second = policy.predict_requests(model, changed, ids, self.fold, self.cfg)
        np.testing.assert_array_equal(first[:6], second[:6])
        self.assertTrue(np.all((first >= .5 - 1e-6) & (first <= 1.12 + 1e-6)))
        self.assertTrue(np.all(np.abs(np.diff(np.r_[1., first])) <= .080001))

    def test_future_market_and_outcome_arrays_cannot_change_past_requests(self):
        torch.manual_seed(47)
        model = PolicyActorCritic(2, 3, 5, self.cfg, "B")
        ids = np.arange(12)
        first = policy.predict_requests(model, self.arrays, ids, self.fold, self.cfg)
        changed = copy.deepcopy(self.arrays)
        changed["x_grid"][self.arrays["decision_id"][11]+1:] = 1e20
        changed["f"][12:] = 1e20
        changed["future_path_log_returns"][:] = np.nan
        changed["teacher_valid"][:] = False
        second = policy.predict_requests(model, changed, ids, self.fold, self.cfg)
        np.testing.assert_array_equal(first, second)

    def test_gap_resets_own_inventory_before_first_post_gap_action(self):
        torch.manual_seed(61)
        model = PolicyActorCritic(2, 3, 5, self.cfg, "A")
        ids = np.array([0, 1, 2, 20, 21, 22])
        together = policy.predict_requests(model, self.arrays, ids, self.fold, self.cfg)
        separate = policy.predict_requests(model, self.arrays, ids[3:], self.fold, self.cfg)
        np.testing.assert_array_equal(together[3:], separate)

    def test_test_policy_inference_fails_without_global_barrier(self):
        world = ModuleType("unidream.experiments.wm_policy_ablation_world")
        world.encode_test_latents = lambda *a: self.fail("Test latent producer must remain closed")
        with patch.object(policy.data, "load_arrays") as load, \
                patch.dict("sys.modules", {world.__name__: world}):
            with self.assertRaises((FileNotFoundError, ValueError)):
                policy.evaluate_policy(self.cfg, 0, 7, "A", "synthetic-registration")
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
