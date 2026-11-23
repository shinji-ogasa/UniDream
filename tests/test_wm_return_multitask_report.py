import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_return_multitask_evaluation import ARMS, MODELS
from unidream.experiments.wm_return_multitask_report import (
    daily_statistics, validate_multitask_predictions, write_multitask_report,
)


def prediction_fixture(days=9, splits=("selection", "test")):
    frames = []
    n = days * 96
    labels = np.arange(n) % 3
    actual = np.array([-.02, 0., .02])[labels] * (1 + np.arange(n) // 96 / 50)
    for split in splits:
        times = pd.date_range("2024-01-01" if split == "selection" else "2024-04-01", periods=n, freq="15min", tz="UTC")
        for model in MODELS:
            for arm in ARMS:
                frame = pd.DataFrame({"fold": 0, "group": "D", "seed": 7, "model": model, "arm": arm,
                    "split": split, "timestamp": times, "decision_id": np.arange(n),
                    "actual_return": actual, "actual_class": labels, "theta": .01, "train_rms": .02,
                    "train_prior_0": .2, "train_prior_1": .6, "train_prior_2": .2,
                    "train_class_return_mean_0": -.025, "train_class_return_mean_1": .001,
                    "train_class_return_mean_2": .025})
                frame["pred_return"] = actual * (.3 if arm == "reg_only" else .6) if arm != "class_only" else np.nan
                frame[["p0", "p1", "p2"]] = (np.nan if arm == "reg_only" else
                    np.tile([.1, .8, .1], (n, 1)) if arm == "class_only" else .1 + .7 * np.eye(3)[labels])
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class ReturnMultitaskReportTests(unittest.TestCase):
    def test_all_arms_share_exact_cohort_and_inactive_heads_are_not_native(self):
        frame = prediction_fixture(1)
        self.assertEqual(len(validate_multitask_predictions(frame)), len(frame))
        changed = frame.copy()
        changed.loc[(changed.arm == "class_only"), "pred_return"] = 0.
        with self.assertRaisesRegex(ValueError, "trained regression"):
            validate_multitask_predictions(changed)
        changed = frame.copy()
        changed.loc[0, "actual_return"] -= .001
        with self.assertRaisesRegex(ValueError, "identical decisions"):
            validate_multitask_predictions(changed)
        with self.assertRaises(ValueError):
            validate_multitask_predictions(frame.iloc[1:])

    def test_train_contract_drift_and_label_mismatch_are_rejected(self):
        frame = prediction_fixture(1)
        changed = frame.copy(); changed.loc[0, "train_class_return_mean_0"] -= .001
        with self.assertRaisesRegex(ValueError, "immutable Train"):
            validate_multitask_predictions(changed)
        changed = frame.copy(); changed.loc[0, "actual_class"] = 2
        with self.assertRaisesRegex(ValueError, "actual classes"):
            validate_multitask_predictions(changed)
        changed = frame.copy(); changed.loc[0, "theta"] = .02
        with self.assertRaisesRegex(ValueError, "half-Train-RMS"):
            validate_multitask_predictions(changed)

    def test_native_and_auxiliary_reports_and_daily_stats_are_consistent(self):
        frame = prediction_fixture(2)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            summary = write_multitask_report(frame, out, {"folds": [0]})
            table = pd.read_csv(out / "metrics.csv")
            self.assertEqual(len(table), 24)
            self.assertEqual(table.native.sum(), 16)
            aux = table[table.variant == "auxiliary_scalar_threshold"]
            self.assertTrue(aux.log_loss.isna().all())
            self.assertTrue(aux.multiclass_brier.isna().all())
            self.assertEqual(set(table[table.native & (table.task == "regression")].arm), {"reg_only", "joint"})
            self.assertEqual(set(table[table.native & (table.task == "classification")].arm), {"class_only", "joint"})
            day = pd.read_csv(out / "day_statistics.csv")
            panel = frame[(frame.split == "test") & (frame.model == "wm") & (frame.arm == "joint")]
            daily = day[(day.split == "test") & (day.model == "wm") & (day.arm == "joint")]
            self.assertAlmostEqual(daily.sse_sum.sum(), np.square(panel.actual_return - panel.pred_return).sum())
            self.assertEqual(daily.prior_correct_c0.sum(), (panel.actual_class == 0).sum())
            self.assertTrue(summary["historical_seen_test"])
            self.assertFalse(summary["seed_stability_claim"])
            json.dumps(summary, allow_nan=False)
            json.loads((out / "report.json").read_text())

    def test_missing_bars_split_same_day_into_separate_segments(self):
        frame = prediction_fixture(1, ("test",))
        frame = frame[frame.decision_id != 10]
        daily = daily_statistics(validate_multitask_predictions(frame))
        self.assertEqual(len(daily), 12)
        self.assertEqual(set(daily.segment_id), {0, 1})
        self.assertTrue(daily.groupby(["model", "arm"]).n_rows.sum().eq(95).all())

    def test_report_to_statistics_uses_exact_same_native_metrics(self):
        from unidream.experiments.wm_return_multitask_statistics import write_multitask_statistics
        frame = prediction_fixture(9, ("test",))
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            cfg = {"folds": [0], "statistics": {"bootstrap_replicates": 30}}
            write_multitask_report(frame, out / "final", cfg)
            summary = write_multitask_statistics(out, cfg)
            metrics = pd.read_csv(out / "final/metrics.csv")
            inference = pd.read_csv(out / "final/multitask_inference.csv")
            self.assertEqual(summary["n_primary_available"], 4)
            for row in inference.itertuples():
                metric = metrics[(metrics.model == row.model) & (metrics.task == row.task) & metrics.native]
                base = metric[metric.arm == row.baseline_arm].iloc[0]
                joint = metric[metric.arm == "joint"].iloc[0]
                self.assertAlmostEqual(row.observed_baseline, base[row.metric], places=12)
                self.assertAlmostEqual(row.observed_joint, joint[row.metric], places=12)


if __name__ == "__main__":
    unittest.main()
