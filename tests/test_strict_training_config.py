from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from unidream.cli.train import build_parser
from unidream.experiments.run_config import load_training_run_config, prepare_run_directory


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

    def test_removed_plan004_mainline_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _config(Path(temp_dir))
            cfg["plan004_residual_bc_ac"] = {"enabled": False}
            with self.assertRaisesRegex(ValueError, "removed from the main training pipeline"):
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


if __name__ == "__main__":
    unittest.main()
