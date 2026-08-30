from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from unidream.data.dataset import SequenceDataset
from unidream.world_model.train_wm import (
    TargetGradientCoverageError,
    WorldModelTrainer,
)


class _ModelMeta(nn.Module):
    def __init__(self, obs_dim: int = 2):
        super().__init__()
        self.obs_dim = obs_dim


class _CoverageEnsemble(nn.Module):
    """Small differentiable ensemble sufficient to exercise auxiliary heads."""

    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([_ModelMeta()])
        self.weight = nn.Parameter(torch.tensor(0.5))

    def get_z_dim(self) -> int:
        return 2

    def get_d_model(self) -> int:
        return 2

    def compute_losses(self, obs, actions, rewards, dones, **kwargs):
        zero = obs.mean() * 0.0
        loss = self.weight.square() + zero
        return {"loss": loss, "base_loss": loss, "disagreement": zero}

    def encode(self, obs):
        return obs[..., :2], None

    def forward(self, z, actions):
        return {"h": z}


class WorldModelTargetCoverageTest(unittest.TestCase):
    def _dataset(self) -> SequenceDataset:
        rng = np.random.default_rng(7)
        return SequenceDataset(
            rng.normal(size=(64, 2)).astype(np.float32),
            seq_len=64,
            returns=rng.normal(size=64).astype(np.float32),
        )

    def _trainer(
        self,
        *,
        require_target_gradient_coverage: bool = False,
        conditional: bool = False,
    ) -> WorldModelTrainer:
        config = {
            "world_model": {
                "action_context": "actionless",
                "batch_size": 1,
                "max_steps": 1,
                "return_scale": 1.0,
                "return_horizons": [4, 64],
                "return_include_current": False,
                "position_utility_scale": 1.0,
                "position_utility_positions": [0.5, 1.0],
                "position_utility_horizon": 64,
            },
            "logging": {"log_interval": 1},
        }
        if require_target_gradient_coverage:
            config["world_model"]["require_target_gradient_coverage"] = True
        if conditional:
            config["conditional_oracle_path"] = True
        return WorldModelTrainer(
            _CoverageEnsemble(),
            config,
            device="cpu",
        )

    def test_zero_target_horizon_is_blocked_and_written_machine_readably(self) -> None:
        trainer = self._trainer()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / "world_model.pt")
            trainer.train_on_dataset(self._dataset(), max_steps=1, checkpoint_path=checkpoint)
            rows = trainer.target_gradient_coverage()
            artifact = Path(tmp) / "target_gradient_coverage.jsonl"
            self.assertTrue(artifact.exists())
            self.assertTrue(Path(checkpoint).exists())
            self.assertFalse(Path(f"{checkpoint}.blocked.json").exists())
            file_rows = [json.loads(line) for line in artifact.read_text().splitlines()]
            self.assertEqual(len(file_rows), len(rows))

        by_key = {(row["head"], row["horizon"], row["output_index"]): row for row in rows}
        return_h4 = by_key[("return", 4, 0)]
        self.assertGreater(return_h4["target_count"], 0)
        self.assertGreater(return_h4["nonzero_gradient_steps"], 0)
        self.assertEqual(return_h4["status"], "pass")

        return_h64 = by_key[("return", 64, 1)]
        self.assertEqual(return_h64["target_count"], 0)
        self.assertEqual(return_h64["mask_fraction"], 0.0)
        self.assertEqual(return_h64["status"], "block")
        self.assertEqual(return_h64["block_reason"], "zero_valid_targets")

        utility_rows = [row for row in rows if row["head"] == "position_utility"]
        self.assertEqual(len(utility_rows), 2)
        self.assertTrue(all(row["horizon"] == 64 for row in utility_rows))
        self.assertTrue(all(row["target_count"] == 0 for row in utility_rows))
        self.assertTrue(all(row["status"] == "block" for row in utility_rows))
        self.assertFalse(trainer.target_gradient_coverage_passes())

    def test_output_specific_gradient_does_not_hide_zero_masked_sibling(self) -> None:
        trainer = self._trainer()
        trainer.train_on_dataset(self._dataset(), max_steps=1)
        rows = trainer.target_gradient_coverage()
        h64 = next(
            row
            for row in rows
            if row["head"] == "return" and row["horizon"] == 64
        )
        # The return head shares hidden layers with h4, but its h64 projection
        # row receives no gradient when h64 has no valid target.
        self.assertEqual(h64["nonzero_gradient_steps"], 0)
        self.assertEqual(h64["gradient_coverage"], 0.0)

    def test_required_coverage_gate_writes_artifact_and_blocks_checkpoint(self) -> None:
        for trainer_kwargs in (
            {"require_target_gradient_coverage": True},
            {"conditional": True},
        ):
            with self.subTest(**trainer_kwargs):
                trainer = self._trainer(**trainer_kwargs)
                with tempfile.TemporaryDirectory() as tmp:
                    checkpoint = str(Path(tmp) / "world_model.pt")
                    with self.assertRaises(TargetGradientCoverageError):
                        trainer.train_on_dataset(
                            self._dataset(),
                            max_steps=1,
                            checkpoint_path=checkpoint,
                        )
                    artifact = Path(tmp) / "target_gradient_coverage.jsonl"
                    marker = Path(f"{checkpoint}.blocked.json")
                    self.assertTrue(artifact.exists())
                    self.assertTrue(marker.exists())
                    marker_payload = json.loads(marker.read_text())
                    self.assertEqual(marker_payload["status"], "blocked")
                    self.assertFalse(marker_payload["promotable"])
                    self.assertTrue(Path(checkpoint).exists())

    def test_coverage_context_and_head_steps_are_not_credited_when_head_skipped(self) -> None:
        rng = np.random.default_rng(8)
        dataset_without_returns = SequenceDataset(
            rng.normal(size=(64, 2)).astype(np.float32),
            seq_len=64,
        )
        trainer = self._trainer()
        trainer.train_on_dataset(
            dataset_without_returns,
            max_steps=1,
            coverage_context={"run": "run-1", "fold": 3, "phase": "train"},
        )
        h4 = next(
            row
            for row in trainer.target_gradient_coverage()
            if row["head"] == "return" and row["horizon"] == 4
        )
        self.assertEqual(h4["valid_targets"], 0)
        self.assertEqual(h4["gradient_steps"], 0)
        self.assertEqual(h4["finite_loss_steps"], 0)
        self.assertEqual(h4["nonzero_gradient_steps"], 0)
        self.assertEqual(h4["run"], "run-1")
        self.assertEqual(h4["fold"], 3)
        self.assertEqual(h4["phase"], "train")


if __name__ == "__main__":
    unittest.main()
