import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_classification_evaluation import balanced_prior_predictions
from unidream.experiments.wm_return_formulation_report import (
    FAMILIES, VARIANTS, validate_matched_predictions, write_formulation_report,
)


class ReturnFormulationReportTest(unittest.TestCase):
    @staticmethod
    def predictions(days=9, splits=("selection", "test")):
        frames = []
        n = days * 96
        labels = np.arange(n) % 3
        actual_returns = np.array([-.02, 0., .02])[labels]
        for split in splits:
            times = pd.date_range("2022-12-01" if split == "selection" else "2023-01-01", periods=n, freq="15min", tz="UTC")
            for family in FAMILIES:
                for formulation, variant in VARIANTS:
                    frame = pd.DataFrame({"timestamp": times, "fold": 0, "seed": 7, "decision_id": np.arange(n),
                        "family": family, "split": split, "actual_return": actual_returns, "theta": .01,
                        "actual": labels, "formulation": formulation, "variant": variant,
                        "train_prior_0": .2, "train_prior_1": .6, "train_prior_2": .2})
                    frame["scalar_prediction"] = 0. if formulation == "regression" else np.nan
                    if variant == "scalar_threshold":
                        frame[["p0", "p1", "p2"]] = np.nan
                        frame["pred"] = 1
                        frame["pred_raw"] = 1
                    else:
                        probs = np.tile([.1, .8, .1], (n, 1)) if formulation == "regression" else .1 + np.eye(3)[labels] * .7
                        frame[["p0", "p1", "p2"]] = probs
                        frame["pred"] = balanced_prior_predictions(probs, np.array([.2, .6, .2]))
                        frame["pred_raw"] = probs.argmax(axis=1)
                    frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def test_matched_cohort_enforces_shared_outcomes_and_all_variants(self):
        frame = self.predictions(days=1)
        self.assertEqual(len(validate_matched_predictions(frame)), len(frame))
        with self.assertRaisesRegex(ValueError, "identical fold/time"):
            validate_matched_predictions(frame.iloc[1:])
        changed = frame.copy()
        changed.loc[0, "theta"] = .009
        with self.assertRaisesRegex(ValueError, "identical labels"):
            validate_matched_predictions(changed)
        changed = frame.loc[~((frame.family == "wm") & (frame.variant == "scalar_threshold"))]
        with self.assertRaisesRegex(ValueError, "all four variants"):
            validate_matched_predictions(changed)

    def test_saved_decisions_must_follow_registered_rule(self):
        frame = self.predictions(days=1)
        frame.loc[frame.variant == "calibrated", "pred"] = 0
        with self.assertRaisesRegex(ValueError, "p/Train prior"):
            validate_matched_predictions(frame)

    def test_report_primary_holm_family_and_no_probability_loss_for_scalar(self):
        frame = self.predictions()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            summary = write_formulation_report(frame, out, {"statistics": {"bootstrap_replicates": 29}})
            inference = pd.read_csv(out / "formulation_inference.csv")
            primary = inference.loc[inference.primary_family_test]
            self.assertEqual(len(primary), 4)
            self.assertEqual(set(primary.family), set(FAMILIES))
            np.testing.assert_allclose(primary.estimate, 2 / 3)
            self.assertTrue(primary.p_holm_four_families.notna().all())
            self.assertTrue(inference.loc[~inference.primary_family_test, "p_holm_four_families"].isna().all())
            long_blocks = inference.loc[inference.block_days == 28]
            self.assertFalse(long_blocks.inference_available.any())
            metrics = pd.read_csv(out / "formulation_metrics.csv")
            scalar = metrics.loc[metrics.variant == "scalar_threshold"]
            self.assertTrue(scalar.log_loss.isna().all())
            self.assertTrue(scalar.multiclass_brier.isna().all())
            probabilities = metrics.loc[metrics.variant != "scalar_threshold"]
            self.assertTrue(probabilities.log_loss.notna().all())
            self.assertEqual(summary["test_unique_timestamps"], 9 * 96)
            json.dumps(summary, allow_nan=False)
            json.loads((out / "formulation_metrics.json").read_text())
            contract = json.loads((out / "formulation_report_contract.json").read_text())
            self.assertTrue(contract["historical_seen_test"])
            self.assertFalse(contract["seed_stability_claim"])
            self.assertIn("seed 7", (out / "report.md").read_text())

    def test_gap_and_fold_boundaries_prohibit_spurious_long_block_inference(self):
        frame = self.predictions(days=6, splits=("test",))
        frame = frame.loc[frame.timestamp != pd.Timestamp("2023-01-03T12:00Z")].copy()
        frame.loc[frame.timestamp >= pd.Timestamp("2023-01-05T00:00Z"), "fold"] = 1
        with tempfile.TemporaryDirectory() as directory:
            write_formulation_report(frame, directory, {"statistics": {"bootstrap_replicates": 19, "block_days": [7]}})
            inference = pd.read_csv(Path(directory) / "formulation_inference.csv")
            self.assertTrue((inference.n_segments == 3).all())
            self.assertFalse(inference.inference_available.any())
            self.assertTrue(inference.p_holm_four_families.isna().all())


if __name__ == "__main__":
    unittest.main()
