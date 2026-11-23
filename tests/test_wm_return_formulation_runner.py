"""Train-label, token-time and Val/Test isolation contracts of the new runner."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from unidream.experiments import wm_return_formulation as runner


BAR_NS = 15 * 60 * 1_000_000_000


class ReturnFormulationRunnerTests(unittest.TestCase):
    def test_threshold_rms_and_priors_ignore_non_train_values(self):
        values = np.array([-.04, 1000., 0., -1000., .04])
        ids = np.array([0, 2, 4])
        contract = runner.train_label_contract(values, ids, k=.5)
        expected_rms = float(np.sqrt(np.mean(np.square(values[ids]))))
        self.assertAlmostEqual(contract["rms"], expected_rms)
        self.assertAlmostEqual(contract["theta"], .5 * expected_rms)
        np.testing.assert_allclose(contract["priors"], np.full(3, 1. / 3.))
        changed = values.copy()
        changed[[1, 3]] = [np.nan, np.inf]
        replay = runner.train_label_contract(changed, ids, k=.5)
        self.assertEqual(contract, replay)

    def test_token_mask_purges_train_start_and_strict_future_label_end(self):
        index = np.arange(12, dtype=np.int64) * BAR_NS
        ends = (index + 2 * BAR_NS)[:, None]
        valid = np.ones((len(index), 1), dtype=bool)
        valid[5, 0] = False
        arrays = {"index_ns": index, "label_end_ns": ends, "mask": valid}
        fold = {"train_start": pd.Timestamp(2 * BAR_NS, tz="UTC").isoformat(),
                "train_end": pd.Timestamp(8 * BAR_NS, tz="UTC").isoformat()}
        positions = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        mask = runner.training_token_mask(arrays, positions, fold)
        expected = np.array([[[False], [False], [True], [True], [True], [False], [False], [False]]])
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(mask.dtype, np.bool_)
        # Label-end equality at Train end is deliberately excluded.
        self.assertEqual(ends[6, 0], pd.Timestamp(fold["train_end"]).value)
        self.assertFalse(mask[0, 6, 0])

    def test_middle_val_calibration_excludes_other_splits_and_rejects_duplicates(self):
        frame = pd.DataFrame({"decision_id": [1, 2, 3, 4, 5],
                              "split": ["earlystop", "calibration", "calibration", "selection", "test"],
                              "actual_return": [1., 2., 3., 4., 5.]})
        selected = runner.calibration_middle_month(frame)
        np.testing.assert_array_equal(selected.decision_id, [2, 3])
        self.assertEqual(set(selected.split), {"calibration"})
        duplicate = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
        with self.assertRaises(ValueError):
            runner.calibration_middle_month(duplicate)
        with self.assertRaises(ValueError):
            runner.calibration_middle_month(frame[frame.split != "calibration"])

    def test_calendar_validation_months_are_disjoint_and_purge_label_end_equality(self):
        index = pd.date_range("2024-01-01", "2024-04-01", freq="15min", inclusive="left", tz="UTC").asi8
        ids = np.arange(len(index))
        parts = runner.parent.validation_parts(ids, index, "2024-01-01T00:00:00+00:00", 16)
        self.assertEqual(set(parts), {"earlystop", "calibration", "selection"})
        boundaries = pd.date_range("2024-01-01", periods=4, freq="MS", tz="UTC").asi8
        for number, name in enumerate(("earlystop", "calibration", "selection")):
            times = index[parts[name]]
            self.assertTrue((times >= boundaries[number]).all())
            self.assertTrue((times + 16 * BAR_NS < boundaries[number + 1]).all())
            equality = np.flatnonzero(index + 16 * BAR_NS == boundaries[number + 1])
            self.assertFalse(np.isin(equality, parts[name]).any())
        self.assertEqual(len(np.unique(np.concatenate(list(parts.values())))), sum(map(len, parts.values())))

    def test_fixed_yaml_rejects_changed_scope_threshold_or_test_selection(self):
        original = yaml.safe_load(runner.DEFAULT_CONFIG.read_text())
        valid = runner.load_config(runner.DEFAULT_CONFIG)
        self.assertTrue(Path(valid["parent_config"]).is_absolute())
        self.assertTrue(Path(valid["output_dir"]).is_absolute())
        changes = {"groups": ["A"], "seeds": [7, 17], "horizons": [4],
                   "folds": list(range(14)), "test_selection_allowed": True,
                   "threshold_train_rms_multiplier": .6, "validation_partition": "random"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yaml"
            for name, value in changes.items():
                altered = copy.deepcopy(original)
                altered[name] = value
                path.write_text(yaml.safe_dump(altered))
                with self.subTest(setting=name), self.assertRaises(ValueError):
                    runner.load_config(path)

    def test_parent_barrier_requires_complete_frozen_fit_manifest_and_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            registration = {"neural_tasks": [{"job": "neural_fixture"}],
                            "classical_tasks": [{"job": "classical_fixture"}]}
            (out / "preregistration.json").write_text(json.dumps(registration))
            (out / "validation_choices.json").write_text('{"selection":"Val only"}')
            registration_hash = runner._sha(out / "preregistration.json")
            manifest = {}
            for kind, task_key in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
                directory = out / kind / registration[task_key][0]["job"]
                directory.mkdir(parents=True)
                model = "research_model.pt" if kind == "neural" else "models.pkl"
                key = "checkpoint_sha256" if kind == "neural" else "model_sha256"
                (directory / model).write_bytes(b"frozen model fixture")
                (directory / "validation_predictions.parquet").write_bytes(b"frozen Val fixture")
                fit = {"registration_sha256": registration_hash,
                       key: runner._sha(directory / model),
                       "validation_predictions_sha256": runner._sha(directory / "validation_predictions.parquet")}
                (directory / "fit.json").write_text(json.dumps(fit))
                manifest[str((directory / "fit.json").relative_to(out))] = runner._sha(directory / "fit.json")
            barrier = {"registration_sha256": registration_hash,
                       "validation_choices_sha256": runner._sha(out / "validation_choices.json"),
                       "frozen_fit_sha256": manifest, "test_selection_allowed": False}
            cfg = {"parent_config": "fixture_parent.yaml"}
            with patch.object(runner.parent, "load_config", return_value={"output_dir": str(out)}):
                with self.assertRaisesRegex(ValueError, "closed"):
                    runner.enforce_parent_test_barrier(cfg)
                (out / "test_barrier.json").write_text(json.dumps(barrier))
                actual_cfg, actual_barrier = runner.enforce_parent_test_barrier(cfg)
                self.assertEqual(actual_cfg["output_dir"], str(out))
                self.assertEqual(actual_barrier, barrier)
                incomplete = copy.deepcopy(barrier)
                incomplete["frozen_fit_sha256"].pop(next(iter(manifest)))
                (out / "test_barrier.json").write_text(json.dumps(incomplete))
                with self.assertRaisesRegex(ValueError, "complete fit manifest"):
                    runner.enforce_parent_test_barrier(cfg)
                (out / "test_barrier.json").write_text(json.dumps(barrier))
                changed_fit = out / next(iter(manifest))
                changed_fit.write_text(changed_fit.read_text() + " ")
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    runner.enforce_parent_test_barrier(cfg)

    def test_combined_test_report_calls_global_barrier_before_opening_predictions(self):
        with patch.object(runner, "verify", return_value=({}, {})), \
                patch.object(runner, "open_test_barrier", side_effect=ValueError("global Test closed")) as barrier, \
                patch.object(runner.pd, "read_parquet") as read:
            with self.assertRaisesRegex(ValueError, "global Test closed"):
                runner.combined_predictions({"output_dir": "unused"}, include_test=True)
            barrier.assert_called_once()
            read.assert_not_called()

    def test_parent_reference_readiness_waits_for_completion_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg, reg = {"output_dir": temporary}, {"folds": [{"fold": 0}]}
            directories = set()
            for family in ("linear", "lightgbm", "direct", "wm"):
                path = runner.regression_path(cfg, 0, family, "calibration")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"writer has begun, but not completed")
                directories.add(path.parent)
            self.assertFalse(runner.reference_ready(cfg, reg))
            for directory in directories:
                (directory / "fit.json").write_text(json.dumps({
                    "validation_predictions_sha256": runner._sha(directory / "validation_predictions.parquet")}))
            self.assertTrue(runner.reference_ready(cfg, reg))
            for directory in directories:
                (directory / "test_predictions.parquet").write_bytes(b"Test writer in progress")
            self.assertFalse(runner.reference_ready(cfg, reg, test=True))
            for directory in directories:
                (directory / "evaluation.json").write_text('{"completed":true}')
            self.assertTrue(runner.reference_ready(cfg, reg, test=True))

    def test_changed_parent_reference_is_rejected_before_parquet_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = {"output_dir": temporary}
            path = runner.regression_path(cfg, 0, "linear", "calibration")
            path.parent.mkdir(parents=True)
            path.write_bytes(b"changed bytes")
            (path.parent / "fit.json").write_text('{"validation_predictions_sha256":"incorrect"}')
            with patch.object(runner.pd, "read_parquet") as read:
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    runner.read_regression(cfg, 0, "linear", "calibration")
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
