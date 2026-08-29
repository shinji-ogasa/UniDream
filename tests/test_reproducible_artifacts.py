from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from unidream.experiments.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    apply_actor_inference_settings,
    atomic_torch_save,
    checkpoint_metadata_for_fold,
    snapshot_actor_inference_settings,
    validate_checkpoint_metadata,
)
from unidream.experiments.ac_stage import _cleanup_auxiliary_checkpoints
from unidream.experiments.runtime import load_training_features


class ReproducibleArtifactsTest(unittest.TestCase):
    def test_cache_hit_validates_and_records_the_data_contract(self) -> None:
        columns = [
            "open_ret", "high_ret", "low_ret", "close_ret", "vol_ret",
            "RSI_14", "macd", "macd_signal", "atr_norm_ret", "atr",
            "rv_4", "rv_16", "rv_96", "funding_rate", "basis",
            "basis_mom", "basis_abs",
        ]
        index = pd.date_range("2024-01-01", periods=4, freq="15min")
        features = pd.DataFrame(
            np.arange(len(index) * len(columns), dtype=np.float64).reshape(len(index), -1),
            index=index,
            columns=columns,
        )
        returns = pd.Series(np.linspace(0.0, 0.003, len(index)), index=index, name="returns")

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            tag = "test_v3"
            features.to_parquet(cache_dir / f"{tag}_features.parquet")
            returns.to_frame().to_parquet(cache_dir / f"{tag}_returns.parquet")

            loaded_features, loaded_returns = load_training_features(
                symbol="BTCUSDT",
                interval="15m",
                start="2024-01-01",
                end="2024-01-02",
                zscore_window=60,
                cache_dir=str(cache_dir),
                cache_tag=tag,
                include_funding=True,
                include_oi=False,
                include_mark=True,
            )

            pd.testing.assert_frame_equal(loaded_features, features, check_freq=False)
            pd.testing.assert_series_equal(loaded_returns, returns, check_freq=False)
            metadata = json.loads((cache_dir / f"{tag}_metadata.json").read_text())
            self.assertEqual(metadata["provenance"], "legacy_unverified")
            self.assertEqual(metadata["parameters"]["include_mark"], True)

    def test_checkpoint_metadata_is_bound_to_the_run(self) -> None:
        manifest = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": "run-123",
            "config_path": "/tmp/config.yaml",
            "config_sha256": "config-sha",
            "source_sha256": "source-sha",
            "data": {
                "fingerprint_sha256": "data-sha",
                "cache_tag": "cache-v3",
                "columns": ["close_ret"],
                "rows": 10,
            },
            "seed": 7,
            "device": "cpu",
            "deterministic_algorithms": True,
        }
        metadata = checkpoint_metadata_for_fold(manifest, fold_idx=3, stage="ac")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ac.pt"
            atomic_torch_save({"checkpoint_metadata": metadata}, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint_metadata(
                payload["checkpoint_metadata"],
                manifest=manifest,
                fold_idx=3,
                stage="ac",
                path=path,
            )
            self.assertEqual(list(Path(temp_dir).glob("*.tmp-*")), [])
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                validate_checkpoint_metadata(
                    {**metadata, "source_sha256": "different"},
                    manifest=manifest,
                    fold_idx=3,
                    stage="ac",
                    path=path,
                )

    def test_actor_inference_settings_round_trip(self) -> None:
        actor = SimpleNamespace(
            infer_adjust_rate_scale=np.float32(0.7),
            infer_advantage_level=0.25,
            target_values=np.asarray([0.5, 1.0, 1.25], dtype=np.float32),
            support_transition_counts=np.asarray([[1, 2]], dtype=np.int64),
        )
        settings = snapshot_actor_inference_settings(actor)
        actor.infer_adjust_rate_scale = 1.0
        actor.infer_advantage_level = 0.0
        actor.target_values = [0.0]
        actor.support_transition_counts = None
        apply_actor_inference_settings(actor, settings)
        self.assertAlmostEqual(actor.infer_adjust_rate_scale, 0.7)
        self.assertEqual(actor.infer_advantage_level, 0.25)
        self.assertEqual(actor.target_values, [0.5, 1.0, 1.25])
        self.assertEqual(actor.support_transition_counts, [[1, 2]])

    def test_auxiliary_ac_checkpoints_are_removed_but_final_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final = root / "ac.pt"
            for name in ("ac.pt", "ac_best.pt", "ac_fire_best.pt", "custom_step1.pt", "ac_stage_01.pt"):
                (root / name).write_bytes(b"checkpoint")
            _cleanup_auxiliary_checkpoints(
                str(final),
                step_checkpoint_prefix="custom_step",
            )
            self.assertTrue(final.exists())
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["ac.pt"],
            )


if __name__ == "__main__":
    unittest.main()
