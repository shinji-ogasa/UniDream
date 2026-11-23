"""Integration boundaries for the chronological WM ablation runner."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_technical_ablation as runner
from unidream.experiments.wm_ablation_evaluation import build_future_targets
from unidream.experiments.wm_ablation_features import TA_COLUMNS


def synthetic_arrays(n=180):
    position = np.arange(n)
    close = 100. * np.exp(.0003 * position + .015 * np.sin(position / 7.))
    open_ = np.r_[close[0], close[:-1]]
    raw = np.stack((open_, np.maximum(open_, close) * 1.002,
                    np.minimum(open_, close) * .998, close,
                    10. + position % 13), axis=-1)
    index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    targets = build_future_targets(index, close)
    return {
        "raw": raw,
        "market": np.repeat((position * .001)[:, None], len(runner.MARKET_COLUMNS), axis=1).astype(np.float32),
        "index_ns": index.asi8,
        "returns": targets.returns,
        "variance": targets.variance,
        "downside": targets.downside,
        "crash": targets.crash,
        "target_mask": targets.mask,
        "baseline_variance": targets.baseline_variance,
    }


class WMTechnicalAblationRunnerTests(unittest.TestCase):
    def test_context_uses_completed_bars_and_does_not_compress_missing_rows(self):
        valid = np.ones(12, dtype=bool)
        valid[6] = False
        selected = runner._context_mask(valid, 4)
        np.testing.assert_array_equal(np.flatnonzero(selected), [4, 5, 6, 11])
        arrays = synthetic_arrays()
        decisions = np.array([8, 12])
        positions, raw = runner._windows(arrays["raw"], decisions, 4)
        np.testing.assert_array_equal(positions, [[4, 5, 6, 7], [8, 9, 10, 11]])
        np.testing.assert_array_equal(raw[:, -1], arrays["raw"][decisions - 1])

    def test_normalization_precedes_channel_mask_and_base_target_is_common(self):
        arrays = synthetic_arrays()
        decisions = np.array([16, 20])
        width = 5 + len(runner.MARKET_COLUMNS) + len(TA_COLUMNS)
        fold = {"mean": np.linspace(-1., 1., width).astype(np.float32),
                "std": np.linspace(.5, 1.5, width).astype(np.float32)}
        outputs = {group: runner._input(arrays, decisions, fold, group, 8) for group in "ABCD"}
        full, common_base = outputs["D"]
        for group, (features, base) in outputs.items():
            mask = runner.channel_mask(group).astype(bool)
            self.assertEqual(features.shape, full.shape)
            np.testing.assert_array_equal(features[:, :, mask], full[:, :, mask])
            np.testing.assert_array_equal(features[:, :, ~mask], 0.)
            np.testing.assert_array_equal(base, common_base)
        self.assertEqual(int(runner.channel_mask("A").sum()), 5)
        self.assertEqual(int(runner.channel_mask("B").sum()), 5 + len(TA_COLUMNS))
        self.assertEqual(int(runner.channel_mask("C").sum()), 5 + len(runner.MARKET_COLUMNS))

    def test_current_and_future_data_cannot_affect_a_decision(self):
        arrays = synthetic_arrays()
        decisions = np.array([80])
        before = runner._features(arrays["raw"], arrays["market"], decisions, 32)
        changed = {name: values.copy() for name, values in arrays.items()}
        changed["raw"][80:, :4] *= 5.0
        changed["raw"][80:, 4] *= 7.0
        changed["market"][80:] *= 100.0
        after = runner._features(changed["raw"], changed["market"], decisions, 32)
        np.testing.assert_array_equal(before, after)

    def test_frozen_prepared_split_and_scaler_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            prepared = out / "prepared"
            prepared.mkdir()
            config_path = out / "config.yaml"
            raw_path, availability_path = out / "raw.parquet", out / "availability.parquet"
            config_path.write_text("frozen config")
            raw_path.write_bytes(b"frozen raw")
            availability_path.write_bytes(b"frozen availability")
            fold_path = prepared / "fold_00.npz"
            np.savez(fold_path, mean=np.zeros(5), train=np.array([1, 2, 3]))
            cfg = {"output_dir": str(out), "config_path": str(config_path),
                   "raw_data": str(raw_path), "availability": str(availability_path)}
            manifest = {
                "config_sha256": runner._sha(config_path),
                "source_sha256": {"fixture.py": "frozen-source"},
                "raw_data_sha256": runner._sha(raw_path),
                "availability_sha256": runner._sha(availability_path),
                "prepared_sha256": {"prepared/fold_00.npz": runner._sha(fold_path)},
            }
            (out / "preregistration.json").write_text(json.dumps(manifest))
            with patch.object(runner, "_source_hashes", return_value=manifest["source_sha256"]):
                self.assertEqual(runner._verify_freeze(cfg), manifest)
                np.savez(fold_path, mean=np.ones(5), train=np.array([1, 2, 4]))
                with self.assertRaisesRegex(ValueError, "prepared split/scaler/target"):
                    runner._verify_freeze(cfg)

    def test_checkpoint_is_fixed_before_any_test_predictions(self):
        # This is a two-update CPU integration check on synthetic candles,
        # not a training arm or a result used by the research experiment.
        old_threads = torch.get_num_threads()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                out = Path(temporary)
                prepared = out / "prepared"
                prepared.mkdir()
                arrays = synthetic_arrays()
                np.savez(prepared / "arrays.npz", **arrays)
                width = len(runner.channel_mask("D"))
                train = np.arange(12, 30)
                val = np.array([102, 104])
                test = np.array([110, 112])
                np.savez(prepared / "fold_00.npz", train=train, val=val, test=test,
                         selection_val=val[:1], mean=np.zeros(width, dtype=np.float32),
                         std=np.ones(width, dtype=np.float32))
                fold = {"train_start": str(pd.Timestamp(arrays["index_ns"][0], tz="UTC")),
                        "train_end": str(pd.Timestamp(arrays["index_ns"][100], tz="UTC")),
                        "scales": {"return": [.01, .02, .03], "variance": [.001, .002, .003], "downside": [.01, .02, .03]}}
                (prepared / "folds.json").write_text(json.dumps([fold]))
                (out / "preregistration.json").write_text('{"synthetic_verification":true}')
                config = {
                    "output_dir": str(out), "device": "cpu", "cpu_threads_per_worker": 1,
                    "model": {"context": 8, "n_categoricals": 3, "n_classes": 4,
                              "d_model": 16, "n_heads": 2, "n_layers": 1,
                              "d_ff": 32, "encoder_hidden": 16, "dropout": .1},
                    "training": {"max_steps": 2, "batch_size": 2, "learning_rate": .001,
                                 "weight_decay": .0001, "gradient_clip": 10.,
                                 "validation_every": 1, "evaluation_batch_size": 2},
                }
                task = {"job": "synthetic_boundary_check", "fold": 0, "seed": 7, "group": "A"}
                directory = out / "runs" / task["job"]
                checkpoint = directory / "research_model.pt"
                actual_predict = runner._predict
                calls = []

                def observe_predictions(model, data, ids, fold_data, group, cfg):
                    calls.append((ids.copy(), checkpoint.exists()))
                    return actual_predict(model, data, ids, fold_data, group, cfg)

                runner._arrays.cache_clear()
                with patch.object(runner, "_load_config", return_value=config), \
                     patch.object(runner, "_predict", side_effect=observe_predictions), \
                     redirect_stdout(io.StringIO()):
                    result = runner._train_task("synthetic.yaml", task)
                self.assertEqual(result["steps"], 2)
                self.assertEqual(len(calls), 4)
                self.assertEqual([exists for _, exists in calls], [False, False, True, True])
                np.testing.assert_array_equal(calls[-1][0], test)
                stored = torch.load(checkpoint, weights_only=False)
                self.assertIn(stored["best_step"], (1, 2))
                self.assertEqual(stored["format"], "wm_feature_ablation_research_v1")
                self.assertTrue((directory / "test_predictions.parquet").exists())
        finally:
            runner._arrays.cache_clear()
            torch.set_num_threads(old_threads)
            torch.use_deterministic_algorithms(old_deterministic)


if __name__ == "__main__":
    unittest.main()
