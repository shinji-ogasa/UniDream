import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_wm_tail_transfer_training import fixture
from unidream.experiments import wm_tail_transfer_training as training


class UnavailableForecastTests(unittest.TestCase):
    def _sparse(self, root):
        cfg, data, ids, ranges, regsha = fixture(root)
        data["valid_D06"] = np.ones(len(data["index_ns"]), bool)
        data["valid_D07"] = np.zeros(len(data["index_ns"]), bool)
        data["valid_D07"][[20, 40]] = True
        data["label_end_ns"] = data["index_ns"][:, None] + 900_000_000_000
        task = {"job": "sparse", "role": "benchmark", "fold": 0, "seed": 7,
                "group": "D06", "cohort": "liquidation_common", "model": "linear",
                "objectives": ["return"]}
        return cfg, data, ranges, regsha, task

    def test_sparse_empty_month_records_no_fit_and_no_placeholder_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ranges, regsha, task = self._sparse(tmp)
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch.object(training, "_ranges", return_value=({"fold": 0}, ranges)), \
                 patch.object(training, "normalization", side_effect=AssertionError("must not normalize unavailable data")):
                result = training.train(cfg, task, regsha, arrays=data)
            self.assertTrue(result["complete"])
            self.assertFalse(result["available"])
            self.assertEqual(result["status"], "unavailable_data")
            self.assertEqual(result["partition_rows"], {"train": 2, "earlystop": 0, "calibration": 0, "selection": 0})
            self.assertEqual(result["actual_steps"], 0)
            self.assertIsNone(result["validation_score"])
            self.assertEqual(result["artifact_sha256"], {})
            self.assertFalse(result["test_read"])
            self.assertEqual([p.name for p in (Path(tmp) / "fits/sparse").iterdir()], ["fit.json"])
            self.assertEqual(json.loads((Path(tmp) / "fits/sparse/fit.json").read_text()), result)
            # A completed N/A condition is restartable without loading arrays.
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch.object(training.data_module, "load_data", side_effect=AssertionError("must reuse N/A marker")):
                self.assertEqual(training.train(cfg, task, regsha), result)

    def test_missing_input_group_remains_a_real_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ranges, regsha, task = self._sparse(tmp)
            del data["valid_D06"]
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch.object(training, "_ranges", return_value=({"fold": 0}, ranges)):
                with self.assertRaises(KeyError):
                    training.train(cfg, task, regsha, arrays=data)
            self.assertFalse((Path(tmp) / "fits/sparse/fit.json").exists())

    def test_overlapping_observed_train_val_remains_error_with_other_months_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ranges, regsha, task = self._sparse(tmp)
            data["valid_D07"][:] = False
            data["valid_D07"][50] = True
            dates = data["index_ns"]
            import pandas as pd
            def stamp(i):
                return pd.Timestamp(int(dates[i]), tz="UTC").isoformat()
            ranges = {"train": (stamp(8), stamp(90)), "earlystop": (stamp(40), stamp(100)),
                      "calibration": (stamp(120), stamp(150)), "selection": (stamp(160), stamp(180))}
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch.object(training, "_ranges", return_value=({"fold": 0}, ranges)):
                with self.assertRaisesRegex(ValueError, "Train/Val overlap"):
                    training.train(cfg, task, regsha, arrays=data)
            self.assertFalse((Path(tmp) / "fits/sparse/fit.json").exists())

    def test_successful_fit_explicitly_marks_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ids, ranges, regsha = fixture(tmp)
            task = {"job": "available", "role": "benchmark", "fold": 0, "seed": 7,
                    "group": "D03", "model": "linear", "objectives": ["return"]}
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch.object(training, "split_arrays", return_value=({"fold": 0}, ranges, ids)):
                result = training.train(cfg, task, regsha, arrays=data)
            self.assertTrue(result["available"])
            self.assertEqual(result["status"], "fitted")
            self.assertTrue(result["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
