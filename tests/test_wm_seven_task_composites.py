import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from unidream.experiments.wm_seven_task_composites import (
    derive_composites, preregister_composites, write_composite_report,
)


CONSTANTS = {"advantage_delta": .06, "dd_penalty": .35, "recovery_dd_penalty": .5, "transition_cost": .00055}


class SevenTaskCompositesTest(unittest.TestCase):
    @staticmethod
    def register(directory):
        out = Path(directory)
        (out / "preregistration.json").write_text(json.dumps({"config": {"targets": CONSTANTS}}))
        return out, preregister_composites(out)

    def test_fixed_formulas_floor_clipping_and_nonlinear_expectation(self):
        result = derive_composites(np.array([.01, -.02]), np.array([.0001, 0.]), np.array([0., .03]), CONSTANTS)
        np.testing.assert_allclose(result["advantage"], .06 * (np.array([.01, -.02]) - .35 * np.array([0., .03]) - .00055))
        self.assertAlmostEqual(result["recovery"][0], .01 / (.01 + 1e-6) / 5)
        self.assertEqual(result["recovery"][1], -1)
        components = derive_composites(np.array([.01, .01]), np.array([.0001, .0009]), np.zeros(2), CONSTANTS)
        means = derive_composites(np.array([.01]), np.array([.0005]), np.zeros(1), CONSTANTS)
        self.assertNotAlmostEqual(components["recovery"].mean(), means["recovery"][0])

    def test_registration_is_before_test_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            out, frozen = self.register(directory)
            self.assertTrue(frozen["test_barrier_absent_at_registration"])
            self.assertFalse(frozen["primary_seven_task_family_changed"])
            (out / "test_barrier.json").write_text("{}")
            self.assertEqual(preregister_composites(out), frozen)
            frozen["constants"]["dd_penalty"] = .99
            (out / "composites/preregistration.json").write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, "constants changed"):
                preregister_composites(out)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "preregistration.json").write_text(json.dumps({"config": {"targets": CONSTANTS}}))
            (out / "test_barrier.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "before the Test barrier"):
                preregister_composites(out)

    def test_identical_cohort_and_mean_per_seed_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            out, _ = self.register(directory)
            frames = []
            actual_advantage = np.array([-.001, 0., .001])
            for seed, multiplier in ((7, 1), (17, -2)):
                frame = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=3, freq="15min", tz="UTC"),
                    "decision_id": [100, 101, 102], "group": "D", "model": "wm", "fold": 0,
                    "seed": seed, "split": "test", "horizon": 16})
                for prefix in ("actual_", "pred_"):
                    for task in ("return", "variance", "downside", "crash", "dd_excess", "advantage", "recovery"):
                        frame[prefix + task] = .0001 if task == "variance" else 0.
                frame["actual_advantage"] = actual_advantage
                frame["pred_advantage"] = actual_advantage * multiplier
                frame["pred_return"] = np.array([-.01, 0., .01])
                frames.append(frame)
            # Variance-only baselines are excluded, rather than allowed to
            # change the comparison cohort through NaN filtering.
            baseline = frames[0].copy()
            baseline["model"] = "garch"
            baseline["pred_advantage"] = np.nan
            pd.concat([*frames, baseline], ignore_index=True).to_parquet(out / "all_predictions.parquet", index=False)
            completion = write_composite_report(out)
            self.assertEqual(completion["prediction_rows"], 6)
            metrics = pd.read_csv(out / "composites/metrics_pooled.csv")
            row = metrics.loc[(metrics.task == "advantage") & (metrics.method == "direct_head")].iloc[0]
            self.assertAlmostEqual(row.pearson_ic_mean_over_seeds, 0.)
            self.assertAlmostEqual(row.spearman_ic_mean_over_seeds, 0.)
            pooled_corr = pearsonr(np.tile(actual_advantage, 2), np.r_[actual_advantage, -2 * actual_advantage]).statistic
            self.assertNotAlmostEqual(pooled_corr, row.pearson_ic_mean_over_seeds)
            expected_mse = np.square(np.r_[np.zeros(3), -3 * actual_advantage]).mean()
            self.assertAlmostEqual(row.rmse, np.sqrt(expected_mse))
            derived = pd.read_parquet(out / "composites/derived_predictions.parquet")
            np.testing.assert_array_equal(derived.actual_advantage.to_numpy(), np.tile(actual_advantage, 2))
            self.assertIn("f(E[...]) ≠ E[f(...)]", (out / "composites/report.html").read_text())
            json.loads((out / "composites/completion.json").read_text())


if __name__ == "__main__":
    unittest.main()
