import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_seven_task_evaluation import TASKS, score_seven
from unidream.experiments.wm_seven_task_report import paired_loss_inference, write_report


class SevenTaskReportTest(unittest.TestCase):
    @staticmethod
    def predictions(days=9):
        index = pd.date_range("2023-01-01", periods=days * 96, freq="15min", tz="UTC")
        frames = []
        for model in ("direct", "wm"):
            for seed in (7, 17):
                frame = pd.DataFrame({"timestamp": index, "decision_id": np.arange(len(index)), "fold": 0,
                    "seed": seed, "group": "D", "model": model, "split": "test", "horizon": 16})
                for task in TASKS:
                    actual = np.full(len(index), .01)
                    if task == "variance":
                        actual[:] = .0001
                    elif task == "crash":
                        actual[:] = 0
                    frame["actual_" + task] = actual
                    # Both direct seeds average to the true return.  The
                    # individual squared errors are nevertheless nonzero.
                    error = 0.02 if seed == 7 else -0.02
                    predicted = actual.copy() if model == "wm" else actual + error
                    if task == "variance":
                        predicted = actual * (2 if model == "direct" else 1)
                    elif task == "crash":
                        predicted = np.full(len(index), .2 if model == "direct" else .1)
                    frame["pred_" + task] = predicted
                frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def test_bootstrap_averages_losses_not_predictions_or_seed_replications(self):
        frame = self.predictions()
        result = paired_loss_inference(frame, ("D", "direct"), ("D", "wm"), block_days=(1, 7, 28), replicates=29)
        returned = result.loc[result.task == "return"]
        np.testing.assert_allclose(returned.estimate, 0.02 ** 2)
        self.assertTrue((result.n_paired_timestamps == 9 * 96).all())
        self.assertTrue((result.n_seeds_reference == 2).all())
        one_seed = paired_loss_inference(frame.loc[frame.seed == 7], ("D", "direct"), ("D", "wm"),
                                        block_days=(1, 7, 28), replicates=29)
        np.testing.assert_allclose(result.estimate, one_seed.estimate)
        np.testing.assert_allclose(result.ci_low, one_seed.ci_low, equal_nan=True)
        long_block = returned.loc[returned.block_days == 28].iloc[0]
        self.assertFalse(long_block.inference_available)
        self.assertTrue(pd.isna(long_block.p_value))

    def test_gap_and_fold_boundaries_cannot_be_bridged(self):
        frame = self.predictions(days=6)
        frame = frame.loc[frame.timestamp != pd.Timestamp("2023-01-03T12:00Z")].copy()
        frame.loc[frame.timestamp >= pd.Timestamp("2023-01-05T00:00Z"), "fold"] = 1
        result = paired_loss_inference(frame, ("D", "direct"), ("D", "wm"), block_days=(7,), replicates=19)
        self.assertTrue((result.n_segments == 3).all())
        self.assertTrue((result.n_segments_shorter_than_block == 3).all())
        self.assertFalse(result.inference_available.any())

    def test_pairing_rejects_incomplete_seed_or_changed_outcomes(self):
        frame = self.predictions(days=2)
        with self.assertRaisesRegex(ValueError, "complete seed set"):
            paired_loss_inference(frame.iloc[1:], ("D", "direct"), ("D", "wm"), replicates=9)
        use = frame.model == "wm"
        frame.loc[use, "actual_return"] += .01
        with self.assertRaisesRegex(ValueError, "identical outcomes"):
            paired_loss_inference(frame, ("D", "direct"), ("D", "wm"), replicates=9)

    def test_report_preserves_variance_only_baseline_and_primary_family(self):
        predictions = self.predictions()
        baseline = predictions.loc[(predictions.model == "direct") & (predictions.seed == 7)].copy()
        baseline["model"] = "past_variance"
        baseline["seed"] = 0
        for task in TASKS:
            if task != "variance":
                baseline["pred_" + task] = np.nan
        predictions = pd.concat([predictions, baseline], ignore_index=True)
        pooled, folded = [], []
        for keys, frame in predictions.groupby(["split", "group", "model"]):
            if keys[2] == "past_variance":
                scores = {"variance": {"qlike": .2, "rmse": .0001, "calibration_mean_ratio": 2}}
            else:
                full = score_seven({task: frame["actual_" + task].to_numpy() for task in TASKS},
                                   {task: frame["pred_" + task].to_numpy() for task in TASKS})
                scores = {task: full[task] for task in TASKS}
            for task, score in scores.items():
                scalar = {key: value for key, value in score.items() if not isinstance(value, (dict, list))}
                base = dict(zip(("split", "group", "model"), keys))
                pooled.append({**base, "task": task, **scalar})
                for seed in frame.seed.unique():
                    folded.append({**base, "task": task, "fold": 0, "seed": int(seed), **scalar})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            predictions.to_parquet(out / "all_predictions.parquet", index=False)
            pd.DataFrame(pooled).to_csv(out / "pooled_metrics.csv", index=False)
            pd.DataFrame(folded).to_csv(out / "fold_metrics.csv", index=False)
            registration = {"config": {"primary_horizon": 16, "statistics": {"bootstrap_replicates": 19}},
                            "limitations": ["synthetic fixture"]}
            (out / "preregistration.json").write_text(json.dumps(registration))
            completion = write_report(out)
            inference = pd.read_csv(out / "inference.csv")
            primary = inference.loc[inference.primary_inference]
            self.assertEqual(len(primary), 7)
            self.assertEqual(set(primary.task), set(TASKS))
            self.assertTrue(primary.p_holm_seven_tasks.notna().all())
            self.assertTrue(inference.loc[~inference.primary_inference, "p_holm_seven_tasks"].isna().all())
            readable = pd.read_csv(out / "readable_7task_D.csv")
            row = readable.loc[readable.model == "past_variance"].iloc[0]
            self.assertAlmostEqual(row.variance_qlike, .2)
            self.assertTrue(pd.isna(row.return_rmse))
            self.assertEqual(completion["test_unique_timestamps"], 9 * 96)
            self.assertIn("過去に確認済み", (out / "report.html").read_text())
            contract = json.loads((out / "prediction_contract.json").read_text())
            self.assertTrue(contract["historical_seen_test"])
            self.assertIn("never average predictions first", contract["seed_handling"])


if __name__ == "__main__":
    unittest.main()
