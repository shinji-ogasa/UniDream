from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unidream.cli.train import build_parser
from unidream.experiments.run_config import (
    finalize_run_manifest,
    load_training_run_config,
    prepare_run_directory,
    write_run_manifest,
)


def _config(root: Path) -> dict:
    return {
        "run": {
            "start": "2018-01-01",
            "end": "2024-01-01",
            "folds": [0, 2],
            "clean_checkpoint_dir": True,
            "deterministic_algorithms": True,
        },
        "data": {
            "include_funding": True,
            "include_oi": False,
            "include_mark": True,
        },
        "logging": {
            "checkpoint_dir": str(root / "run"),
            "cache_dir": str(root / "cache"),
        },
    }


class StrictTrainingConfigTest(unittest.TestCase):
    def test_parser_exposes_only_reproducible_entrypoint_arguments(self) -> None:
        parser = build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.config, "configs/trading.yaml")
        self.assertEqual(defaults.seed, 7)
        self.assertEqual(defaults.device, "auto")
        args = parser.parse_args([
            "--config",
            "configs/trading.yaml",
            "--seed",
            "7",
            "--device",
            "cpu",
        ])
        self.assertEqual(args.seed, 7)
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--config",
                "configs/trading.yaml",
                "--seed",
                "7",
                "--device",
                "cpu",
                "--resume",
            ])

    def test_run_config_is_strict_and_typed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run = load_training_run_config(_config(Path(temp_dir)))
            self.assertEqual(run.folds, (0, 2))
            self.assertTrue(run.deterministic_algorithms)

    def test_removed_checkpoint_compatibility_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _config(Path(temp_dir))
            cfg["world_model"] = {"init_checkpoint": "old.pt"}
            with self.assertRaisesRegex(ValueError, "warm-start/resume"):
                load_training_run_config(cfg)

    def test_removed_non_mainline_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _config(Path(temp_dir))
            cfg["plan004_residual_bc_ac"] = {"enabled": False}
            with self.assertRaisesRegex(ValueError, "removed from the strict training pipeline"):
                load_training_run_config(cfg)

    def test_prepare_run_directory_removes_stale_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _config(Path(temp_dir))
            run = load_training_run_config(cfg)
            run.checkpoint_dir.mkdir(parents=True)
            stale = run.checkpoint_dir / "fold_0" / "ac_best.pt"
            stale.parent.mkdir()
            stale.write_bytes(b"stale")
            prepare_run_directory(run, cfg)
            self.assertFalse(stale.exists())
            resolved = yaml.safe_load((run.checkpoint_dir / "resolved_config.yaml").read_text())
            self.assertEqual(resolved["run"]["folds"], [0, 2])

    def test_finalize_marks_a_run_incomplete_when_required_artifacts_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = _config(root)
            run = load_training_run_config(cfg)
            prepare_run_directory(run, cfg)
            index = pd.date_range("2024-01-01", periods=2, freq="15min")
            features = pd.DataFrame({"close_ret": [0.1, 0.2]}, index=index)
            returns = pd.Series(np.array([0.01, 0.02]), index=index, name="returns")
            write_run_manifest(
                run_cfg=run,
                cfg=cfg,
                config_path="config.yaml",
                seed=7,
                device="cpu",
                active_cost_profile="default",
                features_df=features,
                raw_returns=returns,
                selected_folds=[0],
                cache_tag="test-v3",
                cache_contract_version=1,
            )
            with self.assertRaisesRegex(RuntimeError, "required checkpoint artifacts"):
                finalize_run_manifest(run, {0: {}})
            manifest = yaml.safe_load((run.checkpoint_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "incomplete")
            self.assertFalse(manifest["completed"])


if __name__ == "__main__":
    unittest.main()
