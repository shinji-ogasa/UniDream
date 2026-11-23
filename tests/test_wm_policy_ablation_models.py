from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from unidream.experiments.wm_policy_ablation_models import (
    ARMS, PolicyAblationWM, PolicyActorCritic, _rl, advance_portfolio,
    collect_history_rollout, compute_gae, freeze_bc_prior, initial_portfolio,
    inventory_features, ppo_update,
)
from unidream.experiments.wm_policy_ablation_evaluation import portfolio_step
from unidream.experiments.wm_seven_task_models import TASKS


CFG = {"actor": {"context": 8, "lstm_hidden_size": 8, "lstm_layers": 1,
    "projection_dim": 4, "hidden_size": 16, "dropout": 0.},
    "accounting": {"position_min": .5, "position_max": 1.12, "max_position_step": .08,
        "initial_position": 1., "cost_rate": .00055, "borrow_apr": .10,
        "annual_bars": 35040, "drawdown_penalty": .25},
    "rl_training": {"ppo_epochs": 2, "minibatch_size": 8, "market_batch_size": 8,
        "gamma": .99, "gae_lambda": .95, "clip_ratio": .2, "value_coefficient": .5,
        "bc_prior_coefficient": .01, "entropy_coefficient": .001, "reward_scale": 100.}}


def actor(arm="D", cfg=None):
    return PolicyActorCritic(5, 3, 4, cfg or CFG, arm)


def inputs(batch=4):
    torch.manual_seed(23)
    x, f, z = torch.randn(batch, 8, 5), torch.randn(batch, 3), torch.randn(batch, 4)
    inventory = inventory_features(initial_portfolio(batch, "cpu", CFG))
    return x, f, z, inventory


def rollout_inputs(batch=2, length=5):
    torch.manual_seed(31)
    return (torch.randn(batch, length + 1, 8, 5), torch.randn(batch, length + 1, 3),
            torch.randn(batch, length + 1, 4), torch.randn(batch, length) * .003)


class PolicyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_all_four_arms_have_identical_capacity_and_seeded_weights(self):
        models = []
        for arm in ARMS:
            torch.manual_seed(7)
            models.append(actor(arm))
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models}), 1)
        for model in models[1:]:
            for name, value in models[0].state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)

    def test_inactive_forecasts_and_latents_cannot_leak_including_nan_placeholders(self):
        x, f, z, inventory = inputs()
        for arm, (use_f, use_z) in ARMS.items():
            model = actor(arm).eval()
            expected = model(x, f, z, inventory)["mean"]
            ff = f if use_f else torch.full_like(f, float("nan"))
            zz = z if use_z else torch.full_like(z, float("nan"))
            torch.testing.assert_close(expected, model(x, ff, zz, inventory)["mean"], rtol=0, atol=0)

    def test_bc_updates_only_expected_state_paths_not_critic(self):
        x, f, z, inventory = inputs()
        target = inventory[:, 0] - .07
        for arm, (use_f, use_z) in ARMS.items():
            model = actor(arm)
            loss, _ = model.bc_loss(x, f, z, inventory, target)
            loss.backward()
            coverage = model.state_gradient_coverage()
            self.assertTrue(coverage["history"] and coverage["policy"])
            self.assertEqual(coverage["forecast"], use_f)
            self.assertEqual(coverage["latent"], use_z)
            self.assertFalse(coverage["critic"])

    def test_requested_positions_always_obey_bounds_and_rate_cap(self):
        model = actor()
        previous = torch.tensor([.5, .51, 1., 1.11, 1.12])
        for action in (-100., 0., 100.):
            requested = model.request_from_action(torch.full_like(previous, action), previous)
            self.assertTrue(((requested >= .5) & (requested <= 1.12)).all())
            self.assertTrue(((requested - previous).abs() <= .080001).all())

    def test_actor_history_encoder_has_no_future_or_persistent_state(self):
        x, f, z, _ = inputs()
        model = actor().eval()
        changed = x.clone()
        changed[:, 4:] += 100
        with torch.no_grad():
            h1 = model.history_encoder(model.x_encoder(x))[0]
            h2 = model.history_encoder(model.x_encoder(changed))[0]
            torch.testing.assert_close(h1[:, :4], h2[:, :4], atol=0, rtol=0)
            before = model.encode_context(x, f, z)
            model.encode_context(changed, f * 30, z * 40)
            torch.testing.assert_close(before, model.encode_context(x, f, z), atol=0, rtol=0)

    def test_one_bar_lag_costs_and_borrow_match_independent_numpy_accounting(self):
        state = initial_portfolio(1, "cpu", CFG, torch.float64)
        returns = [.004, -.006, .002, -.001]
        requests = [1.08, 1.12, 1.04, .96]
        previous_executed = 1.
        for index, (request, realized) in enumerate(zip(requests, returns)):
            execution = 1. if index == 0 else requests[index - 1]
            expected = portfolio_step(realized, execution, previous_executed, CFG)
            state, metrics = advance_portfolio(state, torch.tensor([request], dtype=torch.float64),
                torch.tensor([realized], dtype=torch.float64), CFG)
            self.assertAlmostEqual(float(metrics["executed"]), execution, places=14)
            self.assertAlmostEqual(float(metrics["net_log_return"]), float(expected["net_log_return"]), places=13)
            self.assertAlmostEqual(float(metrics["cost"]), float(expected["cost_fraction"]), places=14)
            self.assertAlmostEqual(float(metrics["borrow"]), float(expected["borrow_fraction"]), places=14)
            previous_executed = execution

    def test_reward_max_drawdown_penalty_telescopes_to_registered_utility(self):
        state = initial_portfolio(1, "cpu", CFG, torch.float64)
        total_reward = total_excess = 0.
        for requested, realized in zip([.96, .9, .82, .9, .98], [.01, -.02, .006, -.03, .015]):
            state, metrics = advance_portfolio(state, torch.tensor([requested], dtype=torch.float64),
                torch.tensor([realized], dtype=torch.float64), CFG)
            total_reward += float(metrics["reward"]) / 100
            total_excess += float(metrics["excess_log_return"])
        self.assertAlmostEqual(total_reward, total_excess - .25 * float(state["max_drawdown"]), places=13)

    def test_gae_uses_bootstrap_and_terminal_boundaries(self):
        rewards = torch.tensor([[1.], [2.], [3.]])
        values = torch.tensor([[.2], [.3], [.4]])
        done = torch.tensor([[False], [True], [False]])
        advantage, returns = compute_gae(rewards, values, done, torch.tensor([.5]), gamma=.9, gae_lambda=1.)
        torch.testing.assert_close(returns, torch.tensor([[2.8], [2.], [3.45]]))
        torch.testing.assert_close(advantage, returns - values)

    def test_rollout_does_not_read_future_market_features_or_returns_for_current_action(self):
        x, f, z, returns = rollout_inputs()
        model = actor().eval()
        torch.manual_seed(83)
        first = collect_history_rollout(model, x, f, z, returns, CFG)
        ff, zz, rr = f.clone(), z.clone(), returns.clone()
        ff[:, 3:] += 100
        zz[:, 3:] += 100
        rr[:, 3:] -= .1
        torch.manual_seed(83)
        second = collect_history_rollout(model, x, ff, zz, rr, CFG)
        torch.testing.assert_close(first["ledger"]["request"][:, :3], second["ledger"]["request"][:, :3], rtol=0, atol=0)
        self.assertFalse(any("teacher" in key or "future" in key for key in first["batch"]))

    def test_episode_terminal_resets_own_inventory_without_teacher_state(self):
        x, f, z, returns = rollout_inputs()
        terminals = torch.zeros_like(returns, dtype=torch.bool)
        terminals[:, 1] = True
        result = collect_history_rollout(actor(), x, f, z, returns, CFG, terminals=terminals)
        expected = inventory_features(initial_portfolio(2, "cpu", CFG))
        torch.testing.assert_close(result["ledger"]["inventory"][:, 2], expected, rtol=0, atol=0)

    def test_true_ppo_updates_policy_critic_and_preserves_immutable_bc_prior(self):
        model = actor()
        prior = freeze_bc_prior(model)
        saved = {key: value.clone() for key, value in prior.state_dict().items()}
        rollout = collect_history_rollout(model, *rollout_inputs(), CFG)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = ppo_update(model, prior, optimizer, rollout, CFG)
        self.assertGreater(result["real_policy_gradient_steps"], 0)
        self.assertGreater(result["policy_gradient_norm"], 0)
        self.assertGreater(result["changed_policy_steps"], 0)
        self.assertGreater(result["critic_loss"], 0)
        self.assertTrue(model.state_gradient_coverage()["critic"])
        for key, value in prior.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in prior.parameters()))

    def test_canonical_nondefault_config_controls_actual_ppo_updates_and_reward_scale(self):
        cfg = copy.deepcopy(CFG)
        cfg["rl_training"].update(ppo_epochs=3, minibatch_size=100, clip_ratio=.13,
            value_coefficient=.73, entropy_coefficient=.004, bc_prior_coefficient=.023, reward_scale=7.)
        model = actor(cfg=cfg)
        rollout = collect_history_rollout(model, *rollout_inputs(), cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        result = ppo_update(model, freeze_bc_prior(model), optimizer, rollout, cfg)
        self.assertEqual(result["optimizer_steps"], 3)
        self.assertEqual(result["ppo_epochs"], 3)
        self.assertEqual(result["clip_ratio"], .13)
        self.assertEqual(result["value_coefficient"], .73)
        self.assertEqual(result["entropy_coefficient"], .004)
        self.assertEqual(result["bc_prior_coefficient"], .023)
        self.assertEqual(model.hidden_size, 8)
        self.assertEqual(model.head_hidden, 16)
        state = initial_portfolio(1, "cpu", cfg)
        _, outcome = advance_portfolio(state, torch.ones(1), torch.tensor([.01]), cfg)
        torch.testing.assert_close(outcome["reward"], 7 * outcome["unscaled_reward"])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            _rl({"rl_training": {"ppo_epochs": 3, "epochs": 2}})

    def test_policy_checkpoint_roundtrip_and_bc_snapshot_are_distinct_from_rl(self):
        model = actor()
        x, f, z, inventory = inputs()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, _ = model.bc_loss(x, f, z, inventory, inventory[:, 0] - .06)
        loss.backward()
        optimizer.step()
        model.eval()
        before = model(x, f, z, inventory)["mean_request"].detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bc.pt"
            torch.save(model.state_dict(), path)
            restored = actor().eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(before, restored(x, f, z, inventory)["mean_request"], rtol=0, atol=0)

    def test_frozen_wm_has_real_categorical_dynamics_heads_and_deterministic_latents(self):
        cfg = {"context": 8, "d_model": 16, "n_heads": 4, "n_layers": 1,
            "d_ff": 32, "encoder_hidden": 16, "n_categoricals": 3, "n_classes": 4, "dropout": 0.}
        scales = {t: [1e-4 if t == "variance" else .1] for t in TASKS if t != "crash"}
        wm = PolicyAblationWM(5, cfg, scales)
        x = torch.randn(2, 8, 5)
        targets = {t: torch.randn(2, 8, 1) * .1 for t in TASKS}
        targets["variance"] = targets["variance"].abs() * .001 + 1e-6
        targets["crash"] = torch.randint(0, 2, (2, 8, 1)).float()
        mask = torch.ones(2, 8, 1, dtype=torch.bool)
        loss, parts = wm.loss(x, torch.randn(2, 8, 5), targets, mask, torch.randn(2, 8) * .001)
        loss.backward()
        self.assertTrue(all(wm.supervised_gradient_coverage().values()))
        for key in ("reconstruction", "kl_dyn", "kl_rep", "next_return"):
            self.assertIn(key, parts)
            self.assertTrue(torch.isfinite(parts[key]))
        with self.assertRaisesRegex(RuntimeError, "freeze"):
            wm.encode_policy_state(x)
        wm.freeze_for_policy()
        before = wm.encode_policy_state(x)
        after = wm.encode_policy_state(x)
        self.assertEqual(before.shape, (2, 28))
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(before[:, :12].reshape(2, 3, 4).sum(-1), torch.ones(2, 3))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in wm.parameters()))
        with self.assertRaisesRegex(RuntimeError, "frozen"):
            wm.train()


if __name__ == "__main__":
    unittest.main()
