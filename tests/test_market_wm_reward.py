from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from unidream.actor_critic.critic import Critic
from unidream.actor_critic.imagination_ac import ImagACTrainer
from unidream.actor_critic.market_reward import (
    MarketExecution, compound_drawdown, market_log_return_target, market_portfolio_step,
)
from unidream.data.dataset import SequenceDataset
from unidream.world_model.train_wm import WorldModelTrainer


def config():
    return {
        "data": {"interval": "15m", "seq_len": 2},
        "reward": {"benchmark_position": 1.0},
        "costs": {"fee_rate": .0003, "spread_bps": 3.0, "slippage_bps": 1.0,
                  "borrow_annual": .10},
        "world_model": {"reward_mode": "market_log_return", "action_context": "actionless",
                        "reward_scale": 1.0, "done_scale": 0.0, "idm_scale": 0.0,
                        "return_scale": 0.0, "batch_size": 2, "max_steps": 1},
        "ac": {"reward_objective": "benchmark_absolute_constraint", "horizon": 3,
               "abs_min_position": .5, "abs_max_position": 1.12,
               "max_position_step": .08, "market_deadband": .01},
        "logging": {"log_interval": 1},
    }


class _Metadata(nn.Module):
    obs_dim = 2


class _WmSpy(nn.Module):
    """No predictive auxiliary head; loss/encoding spies only on synthetic data."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.5))
        self.models = nn.ModuleList([_Metadata()])
        self.calls = []
        self.encode_actions = []

    def get_z_dim(self):
        return 2

    def get_d_model(self):
        return 2

    def compute_losses(self, obs, actions, rewards, dones, **kwargs):
        self.calls.append((obs.detach().clone(), actions.detach().clone(), rewards.detach().clone()))
        loss = self.weight.square()
        return {"loss": loss, "base_loss": loss, "disagreement": loss * 0}

    def encode(self, obs):
        return obs, None

    def forward(self, z, actions):
        self.encode_actions.append(actions.detach().clone())
        return {"h": z}


class _ActorSpy(nn.Module):
    def __init__(self, targets=(1., 1., 1.)):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.01))
        self.targets = targets
        self.inventories = []

    def get_action(self, z, h, inventory=None, **kwargs):
        ix = len(self.inventories) % len(self.targets)
        self.inventories.append(inventory.detach().clone())
        target = torch.full((len(z), 1), self.targets[ix], dtype=z.dtype, device=z.device)
        return target, self.weight.expand(len(z)), self.weight.expand(len(z)) * 0


class _MarketSpy(nn.Module):
    def __init__(self, returns=(.02, -.03, .01), *, done=float("nan"), disagreement=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.5))
        self.returns = returns
        self.done = done
        self.disagreement = disagreement
        self.disagree_scale = .1
        self.calls = []

    def get_bins(self):
        return torch.linspace(-2, 2, 5)

    def imagine_step(self, z, h, action, past_zs=None, past_as=None):
        ix = len(self.calls) % len(self.returns)
        self.calls.append((action.detach().clone(), None if past_as is None else past_as.clone()))
        pzs = z.unsqueeze(1) if past_zs is None else torch.cat([past_zs, z.unsqueeze(1)], 1)
        pas = action.unsqueeze(1) if past_as is None else torch.cat([past_as, action.unsqueeze(1)], 1)
        return {"reward": z.new_full((len(z),), self.returns[ix] - .1 * self.disagreement),
                "disagreement": z.new_full((len(z),), self.disagreement),
                "done": z.new_full((len(z),), self.done), "next_z": z + .001,
                "next_h": h + .001, "past_zs": pzs, "past_as": pas,
                # Unused future/auxiliary payload must never become the return.
                "actual_future_return": z.new_full((len(z),), float("nan")),
                "return_head": z.new_full((len(z),), 999.)}


def ac_trainer(cfg=None, targets=(1., 1., 1.), returns=(.02, -.03, .01), **kwargs):
    return ImagACTrainer(_ActorSpy(targets), Critic(2, 2, hidden_dim=4, n_layers=1, n_bins=5),
                         _MarketSpy(returns, **kwargs), cfg or config(), device="cpu")


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


class MarketWmTargetTests(unittest.TestCase):
    def test_raw_market_target_is_nonzero_and_independent_of_actions(self):
        trainer = WorldModelTrainer(_WmSpy(), config(), device="cpu")
        raw = tensor([[.01, -.02, .03], [-.05, .04, 0.]])
        for actions in (torch.ones(2, 3, 1), torch.zeros(2, 3, 1),
                        torch.full((2, 3, 1), float("nan"))):
            self.assertIs(trainer._compute_net_returns(actions, raw), raw)
        self.assertGreater(torch.count_nonzero(raw).item(), 0)

    def test_rejects_nonfinite_or_invalid_raw_targets(self):
        for raw in (tensor([[float("nan")]]), tensor([[float("inf")]]),
                    torch.ones(1, 2, dtype=torch.bool), torch.ones(2), torch.empty(0, 2)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                market_log_return_target(raw)

    def test_train_eval_encode_use_fixed_benchmark_and_observed_market_returns(self):
        ensemble = _WmSpy()
        trainer = WorldModelTrainer(ensemble, config(), device="cpu")
        # Each observation's first column encodes its actual raw return for
        # selected-window assertions, independently of shuffled batch order.
        raw = np.array([.01, -.02, .03, -.04], dtype=np.float32)
        features = np.column_stack([raw, raw * 2])
        dataset = SequenceDataset(features, seq_len=2, returns=raw,
                                  actions=np.full(4, float("nan"), dtype=np.float32))
        trainer.train_on_dataset(dataset, checkpoint_path=None)
        trainer._eval_loss(dataset)
        self.assertGreaterEqual(len(ensemble.calls), 3)
        for obs, actions, rewards in ensemble.calls:
            torch.testing.assert_close(actions, torch.ones_like(actions), rtol=0, atol=0)
            torch.testing.assert_close(rewards, obs[:, :, 0], rtol=0, atol=0)
        trainer.encode_sequence(features, actions=np.full(4, float("nan"), dtype=np.float32))
        self.assertTrue(ensemble.encode_actions)
        for actions in ensemble.encode_actions:
            torch.testing.assert_close(actions, torch.ones_like(actions), rtol=0, atol=0)

    def test_eval_does_not_zero_fill_missing_or_nonfinite_market_returns(self):
        features = np.ones((4, 2), dtype=np.float32)
        for returns in (None, np.full(4, float("nan"), dtype=np.float32)):
            trainer = WorldModelTrainer(_WmSpy(), config(), device="cpu")
            with self.subTest(returns=returns), self.assertRaises(ValueError):
                trainer._eval_loss(SequenceDataset(features, seq_len=2, returns=returns))

    def test_mode_coupling_and_cost_guards(self):
        variants = [
            ("world_model", "action_context", "oracle"),
            ("reward", "benchmark_position", 0.),
            ("reward", "benchmark_position", True),
            ("ac", "reward_objective", "relative_constraint"),
            ("data", "interval", "1h"),
            ("costs", "borrow_annual", float("nan")),
            ("world_model", "done_scale", float("nan")),
        ]
        for section, key, value in variants:
            cfg = config()
            cfg[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                ac_trainer(cfg)
        for section, key, value in variants[:3] + [("world_model", "reward_scale", 0.)]:
            cfg = config()
            cfg[section][key] = value
            with self.subTest(wm=key), self.assertRaises(ValueError):
                WorldModelTrainer(_WmSpy(), cfg, device="cpu")

    def test_legacy_target_and_context_arithmetic_unchanged(self):
        raw = tensor([[.01, -.02, .03]])
        actions = tensor([[[1.], [.5], [1.12]]])
        for mode in ("absolute", "excess_bh"):
            cfg = config()
            cfg["world_model"]["reward_mode"] = mode
            trainer = WorldModelTrainer(_WmSpy(), cfg, device="cpu")
            expected = actions.squeeze(-1) * raw - trainer.cost_rate * tensor([[1., .5, .62]])
            if mode == "excess_bh":
                expected -= raw
            torch.testing.assert_close(trainer._compute_net_returns(actions, raw), expected, rtol=0, atol=1e-18)
            self.assertEqual(trainer.default_context_action, 1. if mode == "excess_bh" else 0.)

    def test_market_wm_checkpoint_refuses_legacy_or_mislabeled_reward(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "wm.pt")
            market = WorldModelTrainer(_WmSpy(), config(), device="cpu")
            market.save(path)
            restored = WorldModelTrainer(_WmSpy(), config(), device="cpu")
            restored.load(path)
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["market_reward_contract"]["context_action"] = 0.
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                restored.load(path)
            cfg = config()
            cfg["world_model"]["reward_mode"] = "excess_bh"
            legacy = WorldModelTrainer(_WmSpy(), cfg, device="cpu")
            legacy.save(path)
            self.assertNotIn("market_reward_contract", torch.load(path, weights_only=False))
            legacy.load(path)
            with self.assertRaises(ValueError):
                restored.load(path)


class MarketAccountTests(unittest.TestCase):
    def test_bh_same_path_without_trading_cost_or_borrow(self):
        cash, asset = tensor([0.]), tensor([1.])
        nav = 1.
        for log_return in (.03, -.1, .04):
            step = market_portfolio_step(cash, asset, tensor([1.]), tensor([log_return]), MarketExecution())
            nav *= math.exp(log_return)
            self.assertAlmostEqual(step["nav"].item(), nav, places=14)
            self.assertAlmostEqual(step["simple_return"].item(), math.expm1(log_return), places=14)
            for name in ("fee", "borrow", "trade_value"):
                self.assertEqual(step[name].item(), 0.)
            cash, asset = step["cash"], step["asset_value"]

    def test_buy_sell_exact_postfee_solve_and_exponential_borrow(self):
        for target in (1.08, .92):
            step = market_portfolio_step(tensor([0.]), tensor([1.]), tensor([target]), tensor([0.]), MarketExecution())
            delta = target - 1
            trade = delta / (1 + .00055 * target * np.sign(delta))
            fee = abs(trade) * .00055
            after_cash = -trade - fee
            borrow = max(-after_cash, 0) * math.expm1(.1 / 35040)
            self.assertAlmostEqual(step["executed_position"].item(), target, places=14)
            self.assertAlmostEqual(step["fee"].item(), fee, places=14)
            self.assertAlmostEqual(step["borrow"].item(), borrow, places=14)
            self.assertAlmostEqual(step["nav"].item(), 1 - fee - borrow, places=14)

    def test_hold_preserves_passive_out_of_bounds_exposure(self):
        # Under a crash, borrowed long exposure drifts above the intent bound.
        first = market_portfolio_step(tensor([-.12]), tensor([1.12]), tensor([1.12]),
                                      tensor([-.6]), MarketExecution())
        self.assertGreater(first["exposure"].item(), 1.12)
        second = market_portfolio_step(first["cash"], first["asset_value"], tensor([1.12]),
                                       tensor([0.]), MarketExecution())
        self.assertAlmostEqual(second["executed_position"].item(), first["exposure"].item() - .08, places=14)
        self.assertGreater(second["executed_position"].item(), 1.12)
        # A below-deadband intent does not magically rebalance cash/asset.
        hold = market_portfolio_step(tensor([.1]), tensor([.9]), tensor([.905]), tensor([.1]), MarketExecution())
        self.assertEqual(hold["trade_value"].item(), 0.)
        self.assertEqual(hold["cash"].item(), .1)
        self.assertAlmostEqual(hold["asset_value"].item(), .9 * math.exp(.1), places=14)

    def test_accounting_repeated_path_against_independent_scalar_bisection(self):
        cfg = MarketExecution()
        cash, asset = .0, 1.
        tcash, tasset = tensor([cash]), tensor([asset])
        for target, log_return in ((.5, .12), (1.12, -.25), (1.05, .04), (.5, -.07), (1.12, .2)):
            nav = cash + asset
            current = asset / nav
            desired = current + np.clip(np.clip(target, .5, 1.12) - current, -.08, .08)
            trade = 0.
            if abs(desired - current) >= .01:
                lower, upper = -asset, nav * .2
                for _ in range(100):
                    mid = (lower + upper) / 2
                    ratio = (asset + mid) / (nav - .00055 * abs(mid))
                    if ratio < desired:
                        lower = mid
                    else:
                        upper = mid
                trade = (lower + upper) / 2
            cash -= trade + .00055 * abs(trade)
            asset += trade
            if cash < 0:
                cash *= math.exp(.1 / 35040)
            asset *= math.exp(log_return)
            step = market_portfolio_step(tcash, tasset, tensor([target]), tensor([log_return]), cfg)
            self.assertAlmostEqual(step["cash"].item(), cash, places=13)
            self.assertAlmostEqual(step["asset_value"].item(), asset, places=13)
            tcash, tasset = step["cash"], step["asset_value"]

    def test_initial_inclusive_compound_drawdown_and_recovery(self):
        actual = compound_drawdown(tensor([[-.1, .1, .2, -.25]]))
        torch.testing.assert_close(actual, tensor([[.1, .01, 0., .25]]), rtol=0, atol=1e-15)
        with self.assertRaises(ValueError):
            compound_drawdown(tensor([[-1.]]))

    def test_invalid_inputs_and_insolvency_fail_closed(self):
        for changes in ({"deadband": .2}, {"one_way_cost": float("nan")},
                        {"borrow_annual": -1}, {"position_min": -.5}, {"bars_per_year": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MarketExecution(**changes)
        for cash, asset, target, ret in ((0., 1., float("nan"), 0.),
                                         (-2., 1., 1., 0.), (-.12, 1.12, 1.12, -10.),
                                         (0., 1., 1., 10000.)):
            with self.subTest(ret=ret), self.assertRaises(ValueError):
                market_portfolio_step(tensor([cash]), tensor([asset]), tensor([target]), tensor([ret]), MarketExecution())


class MarketAcTests(unittest.TestCase):
    def test_fixed_wm_context_shared_benchmark_and_unused_future_payload(self):
        trainer = ac_trainer(targets=(.92, 1.08, 1.), disagreement=.04)
        z = torch.zeros(2, 2)
        result = trainer._imagination_rollout(z, z, torch.zeros(2, 2, 2),
                                              torch.full((2, 2, 1), float("nan")))
        self.assertEqual(len(trainer.ensemble.calls), 3)
        for action, context in trainer.ensemble.calls:
            torch.testing.assert_close(action, torch.ones_like(action), rtol=0, atol=0)
            torch.testing.assert_close(context, torch.ones_like(context), rtol=0, atol=0)
        expected = torch.tensor([[.02, -.03, .01]]).repeat(2, 1)
        torch.testing.assert_close(result["market_log_returns"], expected)
        torch.testing.assert_close(result["benchmark_rewards"], torch.expm1(expected))
        self.assertTrue(torch.isfinite(result["rewards"]).all())
        self.assertEqual(torch.count_nonzero(result["dones"]).item(), 0)
        next_actual = result["asset_values"][:, 0] / result["nav"][:, 0] - 1
        torch.testing.assert_close(trainer.actor.inventories[1][:, 0], next_actual)

    def test_bh_rollout_reward_parity(self):
        trainer = ac_trainer()
        result = trainer._imagination_rollout(torch.zeros(2, 2), torch.zeros(2, 2))
        torch.testing.assert_close(result["rewards"], result["benchmark_rewards"], rtol=1e-5, atol=1e-7)
        self.assertEqual(torch.count_nonzero(result["fees"]).item(), 0)

    def test_done_positive_scale_is_respected_and_bad_market_output_rejected(self):
        cfg = config()
        cfg["world_model"]["done_scale"] = 1.
        trainer = ac_trainer(cfg, done=.25)
        z = torch.zeros(2, 2)
        result = trainer._imagination_rollout(z, z)
        torch.testing.assert_close(result["dones"], torch.full((2, 3), .25))
        for kwargs in ({"returns": (float("nan"), 0., 0.)}, {"done": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ac_trainer(cfg, **kwargs)._imagination_rollout(z, z)

    def test_absolute_objective_true_initial_maxdd_and_no_95pct_clipping(self):
        trainer = ac_trainer()
        trainer.reward_ema.update(torch.tensor([[-1., 1.]]))
        rewards, diag = trainer._benchmark_absolute_constraint_rewards(
            strategy_returns=tensor([[-.1, .2]]), benchmark_returns=tensor([[0., 0.]]),
            next_inventory=tensor([[0., 0.]]), rewards_norm=tensor([[0., 0.]]), advantage0=None)
        self.assertAlmostEqual(diag["bac_terminal_dd_delta"], .1, places=14)
        self.assertAlmostEqual(diag["bac_terminal_excess"], math.log(.9 * 1.2), places=14)
        _, extreme = trainer._benchmark_absolute_constraint_rewards(
            strategy_returns=tensor([[-.99]]), benchmark_returns=tensor([[0.]]),
            next_inventory=tensor([[0.]]), rewards_norm=tensor([[0.]]), advantage0=None)
        self.assertAlmostEqual(extreme["bac_terminal_excess"], math.log(.01), places=13)
        self.assertTrue(torch.isfinite(rewards).all())

    def test_synthetic_train_step_does_not_generate_second_benchmark_path(self):
        trainer = ac_trainer(targets=(.92, 1.08, 1.))
        z = torch.zeros(2, 2)
        with patch.object(trainer, "_benchmark_rollout_rewards", side_effect=AssertionError("second market path")):
            result = trainer.train_step(z, z)
        self.assertEqual(len(trainer.ensemble.calls), 3)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))

    def test_synthetic_critic_pretrain_uses_actual_initial_bh_and_shared_path(self):
        trainer = ac_trainer(targets=(1., 1., 1.))
        with patch.object(trainer, "_benchmark_rollout_rewards", side_effect=AssertionError("second market path")):
            trainer.pretrain_critic([{"z": np.zeros((3, 2), dtype=np.float32),
                                      "h": np.zeros((3, 2), dtype=np.float32)}],
                                     n_steps=1, batch_size=2)
        for inventory in trainer.actor.inventories:
            torch.testing.assert_close(inventory, torch.zeros_like(inventory), rtol=0, atol=0)
        self.assertEqual(len(trainer.ensemble.calls), 3)

    def test_independent_market_benchmark_generation_is_rejected(self):
        trainer = ac_trainer()
        with self.assertRaises(ValueError):
            trainer._benchmark_rollout_rewards(torch.zeros(2, 2), torch.zeros(2, 2))

    def test_market_ac_checkpoint_refuses_legacy_or_different_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "ac.pt")
            trainer = ac_trainer()
            trainer.save(path)
            trainer.load(path)
            different = config()
            different["costs"]["borrow_annual"] = .2
            with self.assertRaises(ValueError):
                ac_trainer(different).load(path)
            legacy_cfg = config()
            legacy_cfg["world_model"]["reward_mode"] = "excess_bh"
            legacy_cfg["ac"]["reward_objective"] = "relative_constraint"
            legacy = ac_trainer(legacy_cfg)
            legacy.save(path)
            self.assertNotIn("market_reward_contract", torch.load(path, weights_only=False))
            legacy.load(path)
            with self.assertRaises(ValueError):
                trainer.load(path)

    def test_legacy_rollout_still_passes_actor_positions_and_done(self):
        cfg = config()
        cfg["world_model"]["reward_mode"] = "excess_bh"
        cfg["ac"]["reward_objective"] = "relative_constraint"
        trainer = ac_trainer(cfg, targets=(.8, .9, 1.1), done=.4)
        result = trainer._imagination_rollout(torch.zeros(2, 2), torch.zeros(2, 2))
        self.assertNotIn("benchmark_rewards", result)
        for i, (action, _) in enumerate(trainer.ensemble.calls):
            torch.testing.assert_close(action, torch.full_like(action, (.8, .9, 1.1)[i]))
        torch.testing.assert_close(result["dones"], torch.full((2, 3), .4))


if __name__ == "__main__":
    unittest.main()

class MarketControllerStateTests(unittest.TestCase):
    def test_full4_duration_fill_and_passive_drift_are_separate(self):
        from unidream.actor_critic.actor import Actor
        actor = Actor(2, 2, 4, hidden_dim=4, n_layers=1, inventory_dim=4)
        actor.benchmark_position = 1.0
        actor.state_hold_scale = 10.0
        seen = []
        def get_action(z, h, inventory=None, **kwargs):
            seen.append(inventory.detach().clone())
            # First actual buy. Later intent exactly current passive exposure.
            target = z.new_full((len(z), 1), .88) if len(seen) == 1 else inventory[:, :1] + 1
            return target, z.new_zeros(len(z)), z.new_zeros(len(z))
        actor.get_action = get_action
        trainer = ImagACTrainer(actor, Critic(2, 2, hidden_dim=4, n_layers=1, n_bins=5),
                                _MarketSpy((.2, -.1, .02)), config(), device="cpu")
        initial = torch.tensor([[-.2, -.03, .4, .2]])
        out = trainer._market_imagination_rollout(torch.zeros(1, 2), torch.zeros(1, 2), inventory0=initial)
        torch.testing.assert_close(out["controller_states"], torch.stack(seen, dim=1))
        torch.testing.assert_close(seen[0], initial)
        self.assertAlmostEqual(float(seen[1][0, 1]), .08, places=6)
        self.assertEqual(float(seen[1][0, 2]), 0.)
        # Drift changed exposure, but no second fill was executed.
        self.assertNotAlmostEqual(float(seen[2][0, 0]), float(seen[1][0, 0]), places=5)
        self.assertEqual(float(seen[2][0, 1]), 0.)
        self.assertAlmostEqual(float(seen[2][0, 2]), 1 / actor._state_hold_scale(), places=6)
        self.assertGreater(float(seen[2][0, 3]), float(seen[1][0, 3]))
        self.assertEqual(float(out["trade_values"][0, 1]), 0.)

    def test_full4_initial_state_and_legacy_rejection(self):
        trainer = ac_trainer()
        with self.assertRaisesRegex(ValueError, "B, B x 1 or B x 4"):
            trainer._market_imagination_rollout(torch.zeros(1, 2), torch.zeros(1, 2), inventory0=torch.zeros(1, 3))
        with self.assertRaisesRegex(ValueError, "finite B x 4"):
            trainer.train_step(torch.zeros(1, 2), torch.zeros(1, 2), controller_state0=torch.zeros(1, 3))
        trainer.market_reward_mode = False
        with self.assertRaisesRegex(ValueError, "only supported"):
            trainer.train_step(torch.zeros(1, 2), torch.zeros(1, 2), controller_state0=torch.zeros(1, 4))


class MarketFixedContextTests(unittest.TestCase):
    def test_generated_steps_replace_old_context_rows(self):
        cfg = config(); cfg['data']['seq_len'] = 64; cfg['ac']['horizon'] = 4
        trainer = ac_trainer(cfg)
        past = torch.arange(63 * 2, dtype=torch.float32).reshape(1, 63, 2)
        trainer._market_imagination_rollout(torch.zeros(1, 2), torch.zeros(1, 2), past_zs=past)
        self.assertEqual(len(trainer.ensemble.calls), 4)
        for action, context in trainer.ensemble.calls:
            self.assertEqual(tuple(context.shape), (1, 63, 1))
            torch.testing.assert_close(context, torch.ones_like(context), rtol=0, atol=0)
