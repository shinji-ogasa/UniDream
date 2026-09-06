from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from unidream.actor_critic.actor import Actor
from unidream.actor_critic.imagination_ac import ImagACTrainer
from unidream.experiments import wm_rl_training as mod


def config():
    return {"data": {"interval": "15m", "seq_len": 64}, "reward": {"benchmark_position": 1.0},
        "world_model": {"reward_mode": "market_log_return", "action_context": "actionless",
            "train_sequence_length": 128, "max_seq_len": 128, "batch_size": 32,
            "max_steps": 700, "idm_scale": 0, "reward_scale": 1},
        "ac": {"reward_objective": "benchmark_absolute_constraint", "controller_state_dim": 4,
            "regime_dim": 0, "horizon": 4, "max_steps": 300, "alpha_init": .35,
            "use_wm_predictive_state": True, "wm_predictive_state_heads": list(mod.HEADS),
            "actor_hidden": 4, "ac_layers": 1, "batch_size": 2, "critic_hidden": 4,
            "advantage_input_mode": "concat", "max_position_step": .08},
        "bc": {"n_epochs": 5, "batch_size": 512, "sirl_hidden": 0,
            "benchmark_overlay_teacher_mode": "lowfreq_wm_overlay", "benchmark_overlay_teacher": True,
            "benchmark_overlay_lowfreq_base": 0, "benchmark_overlay_lowfreq_min_hold": 32,
            "sample_quality_mode": "none", "chunk_size": 4},
        "actions": {"values": [0., .5, 1., 1.25]}}


def actor():
    a = Actor(2, 2, 4, hidden_dim=4, n_layers=1, inventory_dim=4, advantage_dim=42)
    a.benchmark_position = 1.0
    a.target_values = np.array([0, .5, 1, 1.25], dtype=np.float32)
    return a


class PureTrainingSupportTests(unittest.TestCase):
    def test_segments_do_not_compress_gaps(self):
        self.assertEqual(mod.contiguous_segments(np.array([1, 1, 0, 1, 0], bool)), [(0, 2), (3, 4)])
        for bad in ([1, 0], np.array([[True]])):
            with self.assertRaises(ValueError):
                mod.contiguous_segments(bad)

    def test_normalizer_T_only_and_off_support_nan(self):
        raw = np.arange(6 * 42, dtype=np.float32).reshape(6, 42)
        available = np.array([1, 1, 1, 1, 1, 0], bool)
        fit = np.array([1, 1, 1, 0, 0, 0], bool)
        names = [f"f{i}" for i in range(42)]
        first, normal = mod.normalize_auxiliary(raw, fit_mask=fit, available=available, names=names, ac_cfg={})
        altered = raw.copy(); altered[3:5] = 100000; altered[5] = np.nan
        second, normal2 = mod.normalize_auxiliary(altered, fit_mask=fit, available=available, names=names, ac_cfg={})
        self.assertEqual(normal, normal2)
        np.testing.assert_array_equal(first[:3], second[:3])
        np.testing.assert_array_equal(normal["mean"], raw[:3].mean(0))
        self.assertTrue(np.isnan(second[5]).all())
        self.assertTrue((abs(second[3:5]) <= 5).all())
        with self.assertRaises(ValueError):
            mod.normalize_auxiliary(raw, fit_mask=fit.astype(int), available=available, names=names, ac_cfg={})
        altered[0, 0] = np.nan
        with self.assertRaises(ValueError):
            mod.normalize_auxiliary(altered, fit_mask=fit, available=available, names=names, ac_cfg={})

    def test_segmented_teacher_matches_independent_calls_and_resets(self):
        n = 90; mask = np.ones(n, bool); mask[40:45] = False
        aux = np.zeros((n, 42), np.float32); aux[:, 10:19] = 2
        returns = np.full(n, -.01)
        cfg = config()
        output = mod.build_segmented_teacher(returns=returns, auxiliary=aux, eligible=mask, cfg=cfg)
        for start, end in [(0, 40), (45, 90)]:
            expected = mod.apply_benchmark_overlay_teacher(np.ones(end-start, np.float32),
                bc_cfg=cfg['bc'], ac_cfg=cfg['ac'], reward_cfg=cfg['reward'],
                advantage_values=aux[start:end], returns=returns[start:end])
            np.testing.assert_array_equal(output[start:end], expected)
        self.assertTrue(np.isnan(output[40:45]).all())
        altered = aux.copy(); altered[:40] = -999
        second = mod.build_segmented_teacher(returns=returns, auxiliary=altered, eligible=mask, cfg=cfg)
        np.testing.assert_array_equal(output[45:], second[45:])

    def test_controller_and_chunks_keep_original_origins(self):
        positions = np.array([.9, .9, .9, 1., .9, np.nan, .8, .8, .8, .8, .8], np.float32)
        eligible = np.isfinite(positions)
        states = mod.segmented_controller_states(actor(), positions, eligible)
        np.testing.assert_array_equal(states[6], np.zeros(4))
        self.assertTrue(np.isnan(states[5]).all())
        origins, targets = mod.chunk_origins_and_targets(positions, states, eligible, chunk_size=4)
        np.testing.assert_array_equal(origins, [0, 6])
        np.testing.assert_array_equal(targets, [0, 6])
        # Already holding .9 at origin; first switch inside next chunk is +1.
        states[0, 0] = -.1
        _, targets = mod.chunk_origins_and_targets(positions, states, eligible, chunk_size=4)
        self.assertEqual(targets[0], 3)

    def test_anchor_override_keeps_full_state_gap_reset_and_origin_ids(self):
        fake = _FakeAC(actor(), None, None, config())
        states = np.array([[0, 0, 0, 0], [-.1, -.1, .4, .4], [0, 0, 0, 0], [-.2, -.2, 0, .3]], np.float32)
        positions = np.array([.9, .9, .8, .8], np.float32)
        states[1, 0], states[3, 0] = positions[0] - np.float32(1), positions[2] - np.float32(1)
        meta = mod.bind_controller_anchor_bank(fake, z=np.zeros((4, 2)), h=np.zeros((4, 2)),
            positions=positions, states=states, auxiliary=np.zeros((4, 42)), origins=np.array([10, 11, 80, 81]))
        np.testing.assert_array_equal(fake._oracle_inventory.numpy(), states)
        np.testing.assert_array_equal(fake._oracle_anchor_inventory.numpy(), states[[0, 3]])
        self.assertEqual(meta['anchor_origin_indices'], [10, 81])
        self.assertEqual(meta['trade_positive'], 2)
        self.assertEqual(float(fake._oracle_trade_pos_weight), 1.)

    def test_config_fails_closed_without_expanding_family(self):
        cfg = config()
        resolved = mod._validate_cfg(cfg, None)
        self.assertTrue(resolved['world_model']['require_target_gradient_coverage'])
        self.assertEqual(resolved['ac']['advantage_dim'], 42)
        self.assertNotIn('advantage_dim', cfg['ac'])
        self.assertEqual(list(mod.AC_ARMS), ['ac_decay_dd25', 'ac_anchor_dd25', 'ac_anchor_dd50'])
        for section, name, val in [('bc', 'transition_advantage_relabel', True), ('bc', 'sample_quality_mode', 'outcome_edge'),
                                   ('ac', 'regime_dim', 3), ('world_model', 'train_sequence_length', 64),
                                   ('bc', 'self_condition_prob', .2)]:
            bad = copy.deepcopy(cfg); bad[section][name] = val
            with self.subTest(name=name), self.assertRaises(ValueError):
                mod._validate_cfg(bad, None)


class _Ensemble(nn.Module):
    def __init__(self):
        super().__init__(); self.param = nn.Parameter(torch.tensor(.5))
    def get_z_dim(self): return 2
    def get_d_model(self): return 2
    def get_bins(self): return torch.arange(5.)


class _FakeWM:
    calls = []
    global_step = 700
    def __init__(self, ensemble, cfg, device): self.ensemble = ensemble
    def train_on_dataset(self, dataset, **kwargs):
        self.calls.append((dataset, kwargs))
        Path(kwargs['checkpoint_path']).write_bytes(b'fake-unfitted-WM')
        return [{'loss': 0.0}]
    def target_gradient_coverage_passes(self): return True
    def predictive_feature_names(self): return [f'f{i}' for i in range(42)]
    def predict_auxiliary_from_encoded(self, z, h, features, **kwargs):
        return {name: np.tile(z[:, :1], (1, 7 if name == 'position_utility' else 5)) for name in mod.HEADS}


class _NoOptimizer:
    def zero_grad(self): pass
    def step(self): pass
    def state_dict(self): return {}


class _FakeBC:
    calls = []
    def __init__(self, actor, z_dim, h_dim, device, **kwargs):
        self.actor = actor; self.optimizer = _NoOptimizer()
    def _bc_loss(self, z, h, positions, inventory=None, advantage=None, **kwargs):
        self.calls.append((z.detach().numpy(), inventory.detach().numpy(), advantage.detach().numpy()))
        return next(self.actor.parameters()).sum() * 0  # no training/parameter mutation


class _FakeAC:
    initial_states = []; histories = []; steps = []
    set_oracle_data = ImagACTrainer.set_oracle_data
    def __init__(self, actor, critic, ensemble, cfg, device='cpu'):
        self.actor = actor; self.device = torch.device('cpu'); self.benchmark_position = 1.
        self.abs_min_position = .5; self.abs_max_position = 1.12; self.nn_anchor_bank_size = 2
        self.initial_states.append({k:v.detach().clone() for k,v in actor.state_dict().items()})
        self.steps.append(0)
        self.which = len(self.steps) - 1
    def train_step(self, z, h, past_zs=None, past_as=None, advantage0=None, controller_state0=None, **kwargs):
        self.steps[self.which] += 1
        assert past_zs.shape[1] == 63 and controller_state0.shape[-1] == 4 and advantage0.shape[-1] == 42
        assert torch.isfinite(past_zs).all()
        assert torch.all(torch.diff(past_zs[:, :, 0], dim=1) == 1)
        assert torch.all(z[:, 0] - past_zs[:, -1, 0] == 1)
        assert torch.all(past_as == 1)
        return {'actor_loss': 0.0}
    def save(self, path): Path(path).write_bytes(b'fake-unfitted-AC')


def _fake_encode(ensemble, features, timestamps, full_feature_eligible, **kwargs):
    mask = mod.sequence_masks(timestamps, feature_eligible=full_feature_eligible, seq_len=64)['endpoint_eligible']
    z = np.full((len(mask), 2), np.nan, np.float32); z[mask] = features[mask, :2]
    return {'z': z, 'h': z.copy(), 'available': mask}


class MockedOrchestrationTests(unittest.TestCase):
    def inputs(self):
        index = pd.date_range('2020-01-01', periods=700, freq='15min', tz='UTC')
        features = pd.DataFrame(np.repeat(np.arange(700, dtype=np.float32)[:,None], 2, 1), index=index, columns=['a','b'])
        fm = np.ones(700, bool); fm[240:250] = False
        train = np.arange(700) < 500
        return dict(features=features, returns=pd.Series(np.ones(700)*.01, index=index),
                    feature_eligible=fm, target_eligible=np.ones(700, bool), train_mask=train,
                    wm_val_mask=None, cfg=config())

    def test_mocked_pipeline_never_fits_and_reuses_identical_BC_fork(self):
        _FakeWM.calls.clear(); _FakeBC.calls.clear(); _FakeAC.initial_states.clear(); _FakeAC.steps.clear()
        with tempfile.TemporaryDirectory() as tmp, patch.object(mod, 'build_ensemble', return_value=_Ensemble()), \
             patch.object(mod, 'WorldModelTrainer', _FakeWM), patch.object(mod, 'BCPretrainer', _FakeBC), \
             patch.object(mod, 'ImagACTrainer', _FakeAC), patch.object(mod, 'encode_fixed_context', _fake_encode):
            out = mod.train_wm_bc_ac(**self.inputs(), output_dir=Path(tmp)/'new')
            self.assertEqual(_FakeAC.steps, [300, 300, 300])
            self.assertEqual(_FakeWM.calls[0][0].seq_len, 128)
            self.assertEqual(_FakeWM.calls[0][1]['max_steps'], 700)
            self.assertTrue(all(_FakeWM.calls[0][0]._valid_starts < 500-127))
            for state in _FakeAC.initial_states[1:]:
                for key, value in state.items():
                    torch.testing.assert_close(value, _FakeAC.initial_states[0][key], rtol=0, atol=0)
            self.assertEqual(set(out['ac_actors']), set(mod.AC_ARMS))
            self.assertEqual(out['auxiliary']['standardized'].shape, (700, 42))
            self.assertEqual(out['training_provenance']['regime_dim'], 0)
            self.assertTrue((Path(tmp)/'new/bc_actor_full.pt').exists())
            self.assertEqual(len(list((Path(tmp)/'new').glob('*_actor_full.pt'))), 4)
            self.assertEqual(out['auxiliary']['normalizer']['fit_count'], int((out['encoded']['available'] & self.inputs()['train_mask']).sum()))
            # Existing directory is never resumed or overwritten.
            with self.assertRaises(FileExistsError):
                mod.train_wm_bc_ac(**self.inputs(), output_dir=Path(tmp)/'new')

    def test_no_sequence_no_training_and_invalid_validation_fails(self):
        inputs = self.inputs(); inputs['feature_eligible'][::100] = False
        with tempfile.TemporaryDirectory() as tmp, patch.object(mod, 'build_ensemble') as build:
            with self.assertRaisesRegex(ValueError, 'full128'):
                mod.train_wm_bc_ac(**inputs, output_dir=Path(tmp)/'new')
            build.assert_not_called()
        inputs = self.inputs(); inputs['wm_val_mask'] = ~inputs['train_mask']
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'endpoint700'):
            mod.train_wm_bc_ac(**inputs, output_dir=Path(tmp)/'new')

class NativeLossWiringTests(unittest.TestCase):
    def test_native_BC_and_AC_losses_accept_full4_and42_without_parameter_updates(self):
        from unidream.actor_critic.bc_pretrain import BCPretrainer
        from unidream.actor_critic.critic import Critic
        from tests.test_market_wm_reward import _MarketSpy
        cfg = config(); cfg['world_model']['done_scale'] = 0
        cfg['ac'].update(residual_controller=True, abs_min_position=.5, abs_max_position=1.12,
                         residual_min_overlay=-.5, residual_max_overlay=.12, advantage_dim=42,
                         market_deadband=.01, prior_kl_coef=.001, prior_trade_coef=.001,
                         prior_band_coef=.002, prior_flow_coef=.004)
        ensemble = _MarketSpy()
        setup = mod.prepare_bc_setup(ensemble=_Ensemble(), oracle_action_values=np.array([0, .5, 1, 1.25], np.float32),
            oracle_positions=np.array([1, .92, .92, 1], np.float32), oracle_values=None,
            train_regime_probs=None, outcome_edge=None, ac_cfg=cfg['ac'], bc_cfg=cfg['bc'],
            reward_cfg=cfg['reward'], oracle_teacher_mode='lowfreq_wm_overlay')
        a = setup['actor']
        z, h, states, aux = torch.zeros(2, 2), torch.zeros(2, 2), torch.zeros(2, 4), torch.zeros(2, 42)
        states[:, 0] = torch.tensor([-.08, .10])
        states[:, 1] = torch.tensor([-.02, .03])
        states[:, 2:] = .25
        bc = BCPretrainer(a, 2, 2, sirl_hidden=0)
        loss = bc._bc_loss(z, h, torch.tensor([1., .92]), inventory=states, advantage=aux)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters()))
        before = {k: v.detach().clone() for k, v in a.state_dict().items()}
        ac = ImagACTrainer(a, Critic(2, 2, hidden_dim=4, n_layers=1, n_bins=5), ensemble, cfg)
        mod.bind_controller_anchor_bank(ac, z=z.numpy(), h=h.numpy(), positions=np.array([1., .92], np.float32),
            states=states.numpy(), auxiliary=aux.numpy(), origins=np.array([100, 500]))
        with patch.object(ac.actor, 'soft_execute_controller', wraps=ac.actor.soft_execute_controller) as cur_execute, \
             patch.object(ac.actor_prior, 'soft_execute_controller', wraps=ac.actor_prior.soft_execute_controller) as prior_execute:
            anchor_loss = ac._prior_anchor_loss(z, h, states, advantage=aux)
        self.assertTrue(torch.isfinite(anchor_loss))
        self.assertTrue(cur_execute.called and prior_execute.called)
        for calls in (cur_execute.call_args_list, prior_execute.call_args_list):
            for call in calls:
                self.assertEqual(tuple(call.kwargs['current_inventory'].shape), (2,))
                torch.testing.assert_close(call.kwargs['current_inventory'], states[:, 0], rtol=0, atol=0)
        with patch.object(ac.actor_optimizer, 'step'), patch.object(ac.critic_optimizer, 'step'):
            result = ac.train_step(z, h, past_zs=torch.zeros(2, 63, 2), past_as=torch.ones(2, 63, 1),
                                   controller_state0=states, advantage0=aux)
        self.assertTrue(np.isfinite(result['actor_loss']))
        self.assertEqual(ac.global_step, 1)
        for key, value in before.items():
            torch.testing.assert_close(value, a.state_dict()[key], rtol=0, atol=0)
