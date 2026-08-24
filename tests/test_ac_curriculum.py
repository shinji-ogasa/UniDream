from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from unidream.actor_critic.imagination_ac import ImagACTrainer
from unidream.actor_critic.actor import Actor
from unidream.experiments.ac_stage import _apply_curriculum_stage
from unidream.experiments.checkpoint_eval import load_actor_state_checkpoint


class ACCurriculumTest(unittest.TestCase):
    def test_ac_residual_adapter_is_zero_init_and_bounded(self) -> None:
        actor = Actor(z_dim=2, h_dim=3, hidden_dim=4, n_layers=1)
        actor.use_ac_residual_adapter = True
        actor.ac_residual_adapter_scale = 0.05
        actor.benchmark_position = 1.0
        actor.abs_min_position = 0.5
        actor.abs_max_position = 1.12
        actor.residual_min_overlay = -0.5
        actor.residual_max_overlay = 0.12
        target = torch.tensor([0.0, 0.11])
        hidden = torch.ones(2, 4)

        unchanged = actor._apply_ac_residual_adapter(target, hidden)
        torch.testing.assert_close(unchanged, target)

        with torch.no_grad():
            actor.ac_residual_adapter.bias.fill_(10.0)
        shifted = actor._apply_ac_residual_adapter(target, hidden)
        self.assertAlmostEqual(float(shifted[0].detach()), 0.05, places=5)
        self.assertAlmostEqual(float(shifted[1].detach()), 0.12, places=5)

    def test_alpha_schedule_uses_stage_local_steps(self) -> None:
        trainer = ImagACTrainer.__new__(ImagACTrainer)
        trainer.global_step = 120
        trainer.alpha_init = 0.8
        trainer.alpha_final = 0.2
        trainer.alpha_decay_steps = 100
        trainer._alpha_speed = 1.0
        trainer._last_val_sharpe = 1.0
        trainer._max_alpha_t = 100.0
        trainer._alpha_stage_start_step = 0

        trainer.begin_alpha_stage()
        self.assertEqual(trainer._get_alpha(), 0.8)
        trainer.global_step = 170
        self.assertAlmostEqual(trainer._get_alpha(), 0.5)
        trainer.global_step = 220
        self.assertAlmostEqual(trainer._get_alpha(), 0.2)

    def test_curriculum_syncs_actor_and_trainer_bounds(self) -> None:
        actor = torch.nn.Linear(2, 1)
        actor.abs_min_position = 0.5
        actor.abs_max_position = 1.12
        trainer = SimpleNamespace(
            actor_optimizer=Mock(param_groups=[{"lr": 3e-5}]),
            critic_only=False,
            trainable_actor_prefixes=(),
            abs_min_position=0.5,
            abs_max_position=1.12,
            actor_runtime_overrides={},
            actor_runtime_defaults={},
            global_step=60,
            begin_alpha_stage=Mock(),
        )

        _apply_curriculum_stage(
            trainer,
            actor,
            {
                "actor": {
                    "abs_min_position": 0.3,
                    "abs_max_position": 1.25,
                },
            },
            {},
        )

        self.assertEqual(actor.abs_min_position, 0.3)
        self.assertEqual(actor.abs_max_position, 1.25)
        self.assertEqual(trainer.abs_min_position, 0.3)
        self.assertEqual(trainer.abs_max_position, 1.25)
        self.assertEqual(
            trainer.actor_runtime_overrides,
            {"abs_min_position": 0.3, "abs_max_position": 1.25},
        )
        self.assertEqual(
            trainer.actor_runtime_defaults,
            {"abs_min_position": 0.5, "abs_max_position": 1.12},
        )
        trainer.begin_alpha_stage.assert_called_once_with()

    def test_shared_checkpoint_loader_restores_actor_runtime_overrides(self) -> None:
        source = torch.nn.Linear(2, 1)
        target = torch.nn.Linear(2, 1)
        target.abs_min_position = 0.5
        target.abs_max_position = 1.12
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ac.pt"
            torch.save(
                {
                    "actor": source.state_dict(),
                    "actor_runtime_overrides": {
                        "abs_min_position": 0.3,
                        "abs_max_position": 1.25,
                    },
                },
                path,
            )
            load_actor_state_checkpoint(target, path, "cpu")

        self.assertEqual(target.abs_min_position, 0.3)
        self.assertEqual(target.abs_max_position, 1.25)

    def test_runtime_restore_clears_later_stage_overrides(self) -> None:
        trainer = ImagACTrainer.__new__(ImagACTrainer)
        trainer.actor = SimpleNamespace(abs_min_position=0.3, abs_max_position=1.25)
        trainer.abs_min_position = 0.3
        trainer.abs_max_position = 1.25
        trainer.actor_runtime_defaults = {
            "abs_min_position": 0.5,
            "abs_max_position": 1.12,
        }
        trainer.actor_runtime_overrides = {
            "abs_min_position": 0.3,
            "abs_max_position": 1.25,
        }

        trainer._restore_actor_runtime_config(defaults={}, overrides={})

        self.assertEqual(trainer.actor.abs_min_position, 0.5)
        self.assertEqual(trainer.actor.abs_max_position, 1.12)
        self.assertEqual(trainer.abs_min_position, 0.5)
        self.assertEqual(trainer.abs_max_position, 1.12)
        self.assertEqual(trainer.actor_runtime_overrides, {})


if __name__ == "__main__":
    unittest.main()
