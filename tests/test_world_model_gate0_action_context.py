from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from unidream.experiments import wm_stage
from unidream.world_model.train_wm import world_model_action_context


class _FakeWFO:
    train_features = np.zeros((5, 2), dtype=np.float32)
    val_features = np.zeros((4, 2), dtype=np.float32)
    val_returns = np.zeros(4, dtype=np.float32)


class _FakeTrainer:
    def __init__(self, ensemble, cfg, device):
        self.ensemble = ensemble
        self.cfg = cfg
        self.device = device
        self.checkpoint_metadata = {}
        self.train_dataset = None
        self.val_dataset = None

    def train_on_dataset(self, train_dataset, val_dataset, checkpoint_path):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset


class WorldModelGate0ActionContextTest(unittest.TestCase):
    def test_plan011_configs_select_actionless_context(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in (
            "plan011_overlay_actor_v31_relative_constraint_ac.yaml",
            "plan011_overlay_actor_v31_holdout.yaml",
        ):
            with self.subTest(name=name):
                cfg = yaml.safe_load((root / "configs" / name).read_text())
                self.assertEqual(world_model_action_context(cfg), "actionless")

    def test_actionless_stage_does_not_attach_future_oracle_actions(self) -> None:
        cfg = {
            "world_model": {"action_context": "actionless"},
            "data": {"seq_len": 2},
        }
        oracle = np.asarray([0.0, 0.5, 1.0, 1.25, 0.0], dtype=np.float32)
        val_oracle = np.asarray([1.25, 1.0, 0.5, 0.0], dtype=np.float32)
        datasets = []

        def capture_dataset(*args, **kwargs):
            datasets.append((args, kwargs))
            return object()

        with patch.object(wm_stage, "build_ensemble", return_value=object()), patch.object(
            wm_stage, "WorldModelTrainer", _FakeTrainer
        ), patch.object(wm_stage, "SequenceDataset", side_effect=capture_dataset):
            ensemble, trainer = wm_stage.prepare_world_model_stage(
                obs_dim=2,
                cfg=cfg,
                device="cpu",
                wm_path="unused.pt",
                wfo_dataset=_FakeWFO(),
                oracle_positions=oracle,
                val_oracle_positions=val_oracle,
                train_returns=np.zeros(5, dtype=np.float32),
                log_ts=lambda: "00:00:00",
            )

        self.assertIsNotNone(ensemble)
        self.assertIsInstance(trainer, _FakeTrainer)
        self.assertEqual(len(datasets), 2)
        self.assertIsNone(datasets[0][1]["actions"])
        self.assertIsNone(datasets[1][1]["actions"])
        self.assertIsNotNone(trainer.train_dataset)
        self.assertIsNotNone(trainer.val_dataset)

    def test_oracle_context_remains_available_for_legacy_callers(self) -> None:
        cfg = {"world_model": {"action_context": "oracle"}}
        self.assertEqual(world_model_action_context(cfg), "oracle")


if __name__ == "__main__":
    unittest.main()
