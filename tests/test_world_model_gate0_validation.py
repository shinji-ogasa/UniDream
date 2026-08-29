from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from unidream.data.dataset import SequenceDataset
from unidream.world_model import train_wm


class _ModelMeta(nn.Module):
    def __init__(self, obs_dim: int = 2):
        super().__init__()
        self.obs_dim = obs_dim


class _SmallEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([_ModelMeta()])
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.loss_calls = []
        self.batch_starts = []

    def get_z_dim(self) -> int:
        return 2

    def get_d_model(self) -> int:
        return 2

    def compute_losses(self, obs, actions, rewards, dones, **kwargs):
        self.loss_calls.append(int(obs.shape[0]))
        self.batch_starts.append(float(obs[0, 0, 0]))
        zero = obs.mean() * 0.0
        loss = self.weight.square() + zero
        return {"loss": loss, "base_loss": loss, "disagreement": zero}

    def encode(self, obs):
        return obs[..., :2], None

    def forward(self, z, actions):
        return {"h": z}


class _BestStateProbe(train_wm.WorldModelTrainer):
    def __init__(self, *args, **kwargs):
        self.validation_calls = 0
        self.best_snapshot = None
        self.final_snapshot_before_restore = None
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def _eval_loss(self, dataset, n_batches=None):
        if self.validation_calls == 0:
            self.best_snapshot = self._capture_model_state()
            value = 0.0
        else:
            self.final_snapshot_before_restore = self._capture_model_state()
            value = 1.0
        self.validation_calls += 1
        return value


class WorldModelGate0ValidationTest(unittest.TestCase):
    def test_validation_is_ordered_and_full_by_default(self) -> None:
        ensemble = _SmallEnsemble()
        trainer = train_wm.WorldModelTrainer(
            ensemble,
            {
                "world_model": {"batch_size": 2},
                "logging": {"log_interval": 1},
            },
            device="cpu",
        )
        dataset = SequenceDataset(
            np.arange(6, dtype=np.float32).reshape(6, 1),
            seq_len=2,
            returns=np.zeros(6, dtype=np.float32),
        )
        original_loader = train_wm.DataLoader
        loader_calls = []

        def recording_loader(*args, **kwargs):
            loader_calls.append(kwargs.copy())
            return original_loader(*args, **kwargs)

        with patch.object(train_wm, "DataLoader", side_effect=recording_loader):
            trainer._eval_loss(dataset)

        self.assertEqual(len(loader_calls), 1)
        self.assertFalse(loader_calls[0]["shuffle"])
        # len(dataset) == 5 and batch_size == 2, so all three batches must run.
        self.assertEqual(len(ensemble.loss_calls), 3)
        self.assertEqual(ensemble.batch_starts, [0.0, 2.0, 4.0])
        ensemble.loss_calls.clear()
        ensemble.batch_starts.clear()
        trainer._eval_loss(dataset, n_batches=2)
        self.assertEqual(len(ensemble.loss_calls), 2)
        self.assertEqual(ensemble.batch_starts, [0.0, 2.0])

    def test_best_restore_is_coherent_across_ensemble_and_all_active_heads(self) -> None:
        cfg = {
            "actions": {"values": [0.0, 1.0], "n": 2},
            "reward": {"mode": "absolute", "benchmark_position": 1.0},
            "world_model": {
                "action_context": "oracle",
                "batch_size": 1,
                "max_steps": 2,
                "idm_scale": 1.0,
                "return_scale": 1.0,
                "return_horizons": [1],
                "vol_scale": 1.0,
                "drawdown_scale": 1.0,
                "crash_scale": 1.0,
                "drawdown_excess_scale": 1.0,
                "position_utility_scale": 1.0,
                "position_utility_positions": [0.0, 1.0],
                "position_utility_horizon": 1,
                "overweight_advantage_scale": 1.0,
                "recovery_scale": 1.0,
                "risk_horizons": [1],
                "regime_aux_scale": 1.0,
                "regime_dim": 2,
            },
            "logging": {"log_interval": 1},
        }
        trainer = _BestStateProbe(_SmallEnsemble(), cfg, device="cpu")
        features = np.arange(12, dtype=np.float32).reshape(6, 2) / 10.0
        dataset = SequenceDataset(
            features,
            seq_len=3,
            actions=np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
            returns=np.asarray([0.01, -0.02, 0.03, 0.01, -0.01, 0.02], dtype=np.float32),
            regime_probs=np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=np.float32,
            ),
        )

        trainer.train_on_dataset(dataset, val_dataset=dataset, checkpoint_path=None)

        self.assertEqual(trainer.validation_calls, 2)
        self.assertIsNotNone(trainer.best_snapshot)
        self.assertIsNotNone(trainer.final_snapshot_before_restore)
        active_names = ["ensemble", *trainer._active_auxiliary_heads()]
        self.assertEqual(
            set(active_names),
            {
                "ensemble",
                "idm_head",
                "return_head",
                "vol_head",
                "drawdown_head",
                "crash_head",
                "drawdown_excess_head",
                "position_utility_head",
                "overweight_advantage_head",
                "recovery_head",
                "regime_head",
            },
        )

        restored = trainer._capture_model_state()
        for name in active_names:
            for key, expected in trainer.best_snapshot[name].items():
                torch.testing.assert_close(restored[name][key], expected)

        changed = any(
            not torch.equal(trainer.final_snapshot_before_restore[name][key], expected)
            for name in active_names
            for key, expected in trainer.best_snapshot[name].items()
        )
        self.assertTrue(changed, "the second step should differ from the best snapshot")


if __name__ == "__main__":
    unittest.main()
