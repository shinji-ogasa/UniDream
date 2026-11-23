import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_tail_transfer_policy as policy
from unidream.experiments import wm_tail_transfer_accounting as account


def config():
    return {"policy": {"context": 8, "hidden_size": 8, "projection_dim": 4, "head_hidden": 16,
            "prediction_horizon_bars": 4, "bc_steps": 2, "bc_batch_size": 4, "teacher_max_rows": 16,
            "ppo_updates": 2, "selection_updates": [1, 2], "rollout_batch_size": 4, "rollout_horizon": 8,
            "minibatch_size": 16, "ppo_epochs": 1, "context_batch_size": 32},
            "accounting": {"mode": "cash_borrow", "position_min": 0., "position_max": 2., "max_position_step": .08,
             "initial_position": 1., "cost_rate": .00055, "borrow_apr": .1,
             "reset_positions_at_segment_boundary": False}, "runtime": {"cpu_threads": 1}}


def arrays(n=180):
    rng = np.random.default_rng(4)
    previous = 100*np.exp(np.arange(n)*.0004)
    bars = np.column_stack((previous, previous*1.001, previous*.999, previous*1.0003, np.ones(n)))
    return {"index_ns": pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC").asi8,
            "ohlcspot": bars, "ohlcperp": bars.copy(), "ohlcmark": bars.copy(),
            "funding_settlement_rate": np.zeros(n), "funding_event_count": np.zeros(n, np.int64),
            "market": {"raw": rng.normal(size=(n, 3)).astype(np.float32), "old_z": rng.normal(size=(n, 5)).astype(np.float32),
                       "new_z": rng.normal(size=(n, 5)).astype(np.float32), "forecasts": rng.normal(size=(n, 4)).astype(np.float32),
                       "uncertainty": rng.normal(size=(n, 2)).astype(np.float32), "oracle": rng.normal(size=(n, 4)).astype(np.float32)},
            "available": np.ones(n, bool), "train_ids": np.arange(8, 105), "val_ids": np.arange(105, 137),
            "test_ids": np.arange(137, n), "risk_sigma": np.full(n, .01), "up_probability": np.full(n, .3),
            "down_probability": np.full(n, .2), "uncertainty_score": np.full(n, .1)}


class TailTransferPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_all_input_arms_same_geometry_and_inactive_nan_hard_mask(self):
        cfg, source = config(), arrays()
        dims = policy._dims(source, cfg)
        models = [policy.CachedTailPolicy(dims, cfg, f"R{i:02d}") for i in range(8)]
        self.assertEqual(len({sum(p.numel() for p in model.parameters()) for model in models}), 1)
        raw = torch.ones(2, 8, 3)
        z, forecasts, uncertainty = torch.full((2, 5), float("nan")), torch.full((2, 4), float("nan")), torch.full((2, 2), float("nan"))
        context = models[0].encode_context(raw, z, forecasts, uncertainty, torch.zeros(2, 4))
        self.assertTrue(torch.isfinite(context).all())
        inv = torch.zeros(2, 14); inv[:, 6:] = float("nan")
        distribution, value = models[0].heads(context, inv)
        self.assertTrue(torch.isfinite(distribution.mean).all())
        self.assertTrue(torch.isfinite(value).all())

    def test_lazy_window_is_causal_and_train_normalizer_ignores_future(self):
        cfg, source = config(), arrays()
        model = policy.CachedTailPolicy(policy._dims(source, cfg), cfg, "R04")
        policy.fit_normalizers(model, source, source["train_ids"], cfg)
        mean = model.raw_mean.clone()
        original = policy._inputs(source, [20], model, cfg)[0].clone()
        source["market"]["raw"][21:] = 9999
        self.assertTrue(torch.equal(original, policy._inputs(source, [20], model, cfg)[0]))
        source["market"]["raw"][137:] = -1e9
        self.assertTrue(torch.equal(mean, model.raw_mean))
        self.assertTrue(np.array_equal(original.numpy()[0], source["market"]["raw"][13:21]))

    def test_expanded_own_account_information_only_r06(self):
        cfg, source = config(), arrays()
        model = policy.CachedTailPolicy(policy._dims(source, cfg), cfg, "R04")
        context = torch.zeros(2, model.context_dim)
        inv = torch.zeros(2, 14); inv[1, 6:] = 10.
        self.assertTrue(torch.equal(model.heads(context, inv)[0].mean[0], model.heads(context, inv)[0].mean[1]))
        expanded = policy.CachedTailPolicy(policy._dims(source, cfg), cfg, "R06")
        self.assertFalse(torch.equal(expanded.heads(context, inv)[0].mean[0], expanded.heads(context, inv)[0].mean[1]))

    def test_hierarchical_and_gate_transform_preserve_raw_likelihood_and_limits(self):
        cfg, source = config(), arrays()
        cfg["policy"].update(controller="hierarchical", risk_gate=True, uncertainty_gate_threshold=.9)
        model = policy.CachedTailPolicy(policy._dims(source, cfg), cfg, "R06")
        context = torch.zeros(2, model.context_dim); context[:, -4] = .01; context[1, -1] = 1.
        inv = torch.zeros(2, 14); inv[:, 0] = 1.
        action = model.act_from_context(context, inv)
        self.assertEqual(action["raw_action"].shape, (2, 3))
        self.assertEqual(action["log_prob"].shape, (2,))
        self.assertTrue(torch.isfinite(action["log_prob"]).all())
        self.assertLessEqual(float((action["request"]-1).abs().max()), .080001)
        self.assertAlmostEqual(action["request"][1].item(), .92, places=6)

    def test_empirical_cvar_is_episode_lower_tail_with_fractional_boundary(self):
        self.assertEqual(policy.lower_tail_cvar(torch.tensor([-4., -2., 2., 10.]), .25).item(), -4.)
        self.assertAlmostEqual(policy.lower_tail_cvar(torch.tensor([-4., -2., 2., 10.]), .375).item(), -10/3, places=6)
        self.assertEqual(policy.lower_tail_cvar(torch.tensor([-4., -2., 2., 10.]), 1.).item(), 1.5)
        cfg = config(); cfg["policy"]["objective"] = "episodic_cvar_5pct"
        ledger = pd.DataFrame({"training_log_return": [-.01]*8+[.02]*8, "max_dd_increment": [0.]*16,
                               "equity": [1.]*16, "equity_previous": [1.]*16, "max_drawdown": [0.]*16,
                               "liquidation_event": [False]*16, "bankrupt": [False]*16, "decision_available": [True]*16})
        result = policy.path_score(ledger, cfg)
        self.assertAlmostEqual(result["selection_score"], -.08)
        self.assertEqual(result["cvar_validation_episodes"], 2)

    def test_replay_carries_holdings_over_gap_and_fold_without_reset(self):
        cfg, source = config(), arrays()
        model = policy.CachedTailPolicy(policy._dims(source, cfg), cfg, "R00")
        with torch.no_grad():
            for p in model.policy_head.parameters(): p.zero_()
            model.policy_head[-1].bias.fill_(1.)
        cache = policy.encode_contexts(model, source, np.r_[np.arange(105, 115), np.arange(120, 137)], cfg)
        whole, _ = policy.replay_cached(model, cache, source, np.arange(105, 137), cfg)
        first, carried = policy.replay_cached(model, cache, source, np.arange(105, 121), cfg)
        second, _ = policy.replay_cached(model, cache, source, np.arange(121, 137), cfg, initial_state=carried)
        self.assertTrue(np.allclose(whole.equity, pd.concat([first, second]).equity))
        self.assertEqual(int(whole.context_available.sum()), 27)
        gap = whole.loc[~whole.context_available]
        self.assertGreater(gap.equity.std(), 0)
        self.assertEqual(int(gap.order_filled.sum()), 1)
        with self.assertRaisesRegex(ValueError, "every consecutive"):
            policy.replay_cached(model, cache, source, np.array([105, 107]), cfg)

    def test_gate_only_condition_does_not_add_uncertainty_to_actor(self):
        cfg, source = config(), arrays()
        cfg["policy"]["risk_gate"] = True
        model = policy.CachedTailPolicy(policy._dims(source, cfg), cfg, "R04")
        source["uncertainty_score"][20] = 1.
        context = model.encode_context(*policy._inputs(source, [19, 20], model, cfg))
        self.assertEqual(context[1, -1].item(), 1.)
        common = context[:1].repeat(2, 1); common[1, -1] = 1.
        inv = torch.zeros(2, 14); inv[:, 0] = 1.
        self.assertTrue(torch.equal(model.heads(common, inv)[0].mean[0], model.heads(common, inv)[0].mean[1]))
        request = model.act_from_context(common, inv, deterministic=True)["request"]
        self.assertAlmostEqual(request[1].item(), .92, places=6)

    def test_fit_is_fresh_bc_then_real_ppo_encoder_stays_fixed_and_test_is_unread(self):
        cfg, source = config(), arrays()
        # Test futures are poisonous. A training run must not produce or score
        # Test contexts, even though the raw calendar is globally addressable.
        source["market"]["raw"][137:] = np.nan
        source["market"]["new_z"][137:] = np.nan
        task = {"fold": 0, "seed": 7, "input_variant": "D03", "policy_variant": "R04", "control": {}}
        with tempfile.TemporaryDirectory() as tmp:
            fit = policy.train_cached(cfg, task, source, tmp, "registered-test")
            self.assertTrue(fit["complete"])
            self.assertEqual(fit["actual_bc_steps"], 2)
            self.assertEqual(fit["actual_ppo_updates"], 2)
            self.assertEqual(fit["actual_ppo_transitions"], 64)
            self.assertEqual(fit["actual_ppo_optimizer_steps"], 4)
            self.assertFalse(fit["test_read"])
            self.assertTrue(fit["market_encoder_frozen_during_ppo"])
            self.assertLess(np.load(Path(tmp)/"context_raw_ids.npy").max(), 137)
            bc, _ = policy.load_policy(Path(tmp)/"bc.pt", registration_sha="registered-test")
            selected, _ = policy.load_policy(Path(tmp)/"selected.pt", registration_sha="registered-test")
            for key, value in bc.state_dict().items():
                if not key.startswith(("policy_head.", "critic_head.")) and key != "log_std":
                    self.assertTrue(torch.equal(value, selected.state_dict()[key]), key)
            history = __import__("json").loads((Path(tmp)/"history.json").read_text())
            self.assertGreater(history["ppo"][0]["policy_parameter_delta"], 0)
            self.assertEqual(policy.train_cached(cfg, task, source, tmp, "registered-test"), fit)
            with self.assertRaisesRegex(ValueError, "another registered"):
                policy.load_policy(Path(tmp)/"selected.pt", registration_sha="foreign")

    def test_loading_base_config_restores_registered_control_bounds(self):
        cfg, source = config(), arrays()
        task = {"fold": 0, "seed": 7, "policy_variant": "R04", "control": {"cap": 1.5, "speed": .24}}
        effective = policy.effective_config(cfg, task)
        model = policy.CachedTailPolicy(policy._dims(source, cfg), effective, "R04")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            policy._save(path, model, effective, task, "registered", "BC", 2)
            loaded, obj = policy.load_policy(path, cfg=cfg, registration_sha="registered")
            self.assertEqual(loaded.position_max, 1.5)
            self.assertEqual(loaded.max_position_step, .24)

    def test_cvar_uses_larger_tail_batch_with_equal_total_transition_budget(self):
        cfg = config()
        cfg["policy"].update(rollout_batch_size=16, rollout_horizon=64, ppo_updates=200,
                             cvar_rollout_batch_size=64, cvar_ppo_updates=50, cvar_selection_updates=[25, 32, 38, 44, 50])
        selected = policy.effective_config(cfg, {"control": {"objective": "episodic_cvar_5pct"}})
        p = selected["policy"]
        self.assertEqual(p["rollout_batch_size"]*p["rollout_horizon"]*p["ppo_updates"], 204800)
        self.assertEqual(p["rollout_batch_size"]*.05, 3.2)
        self.assertEqual(p["selection_updates"][-1], p["ppo_updates"])

    def test_producer_future_selection_and_invalid_active_inputs_rejected(self):
        cfg, source = config(), arrays()
        source["producer_label_end_ns"] = source["index_ns"][20]
        task = {"fold": 0, "seed": 7, "policy_variant": "R04", "control": {}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "finish before policy Train"):
                policy.train_cached(cfg, task, source, tmp, "test")
        del source["producer_label_end_ns"]
        del source["market"]["new_z"]
        with self.assertRaisesRegex(ValueError, "requires producer feature"):
            policy._eligible(source, source["train_ids"], cfg, "R04")


if __name__ == "__main__":
    unittest.main()
