import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_return_multitask_statistics import paired_native_inference, write_multitask_statistics


def daily_fixture(days=10, fold=0, segment=0, start="2024-01-01"):
    rows = []
    for model in ("direct", "wm"):
        for arm in ("reg_only", "class_only", "joint"):
            for day, date in enumerate(pd.date_range(start, periods=days, freq="D", tz="UTC")):
                y2 = .02 + day * .001
                row = {"split": "test", "group": "D", "model": model, "arm": arm, "fold": fold, "seed": 7,
                    "utc_day": date, "segment_id": segment, "n_rows": 60, "actual_sum": 0., "actual2_sum": y2,
                    "actual_c0": 20, "actual_c1": 20, "actual_c2": 20}
                weight = .2 if arm == "reg_only" else .5
                row.update(pred_sum=0., pred2_sum=weight ** 2 * y2,
                           actual_pred_sum=weight * y2, sse_sum=(1 - weight) ** 2 * y2)
                correct = 10 if arm == "class_only" else 15 + day % 2
                for c in range(3):
                    row[f"prior_correct_c{c}"] = correct
                    row[f"raw_correct_c{c}"] = correct - 1
                row.update(log_loss_sum=60 * (1. if arm == "class_only" else .8),
                           brier_sum=60 * (.7 if arm == "class_only" else .5))
                rows.append(row)
    return pd.DataFrame(rows)


class ReturnMultitaskStatisticsTests(unittest.TestCase):
    def test_exact_pooled_native_effects_and_undefined_constant_predictor(self):
        frame = daily_fixture()
        reg = paired_native_inference(frame, "direct", "regression", block_days=(1, 7, 28), replicates=40)
        r2 = [r for r in reg if r["metric"] == "zero_oos_r2"]
        for row in r2:
            self.assertAlmostEqual(row["estimate"], .64 - .25)
            self.assertEqual(row["n_paired_rows"], 600)
        self.assertTrue(r2[0]["inference_available"])
        self.assertFalse(r2[-1]["inference_available"])
        self.assertIsNone(r2[-1]["p_value"])
        cls = paired_native_inference(frame, "direct", "classification", block_days=(7,), replicates=40)
        values = {r["metric"]: r for r in cls}
        self.assertAlmostEqual(values["balanced_accuracy"]["estimate"], .275)
        self.assertAlmostEqual(values["log_loss"]["estimate"], .2)
        self.assertAlmostEqual(values["multiclass_brier"]["estimate"], .2)
        altered = frame.copy()
        altered.loc[altered.arm == "reg_only", ["pred_sum", "pred2_sum", "actual_pred_sum"]] = 0.
        reg = paired_native_inference(altered, "direct", "regression", block_days=(1,), replicates=20)
        ic = next(row for row in reg if row["metric"] == "pearson_ic")
        self.assertIsNone(ic["estimate"])
        self.assertFalse(ic["inference_available"])

    def test_short_segments_keep_observations_but_do_not_invent_block_support(self):
        frame = pd.concat([daily_fixture(days=4), daily_fixture(days=4, fold=1, start="2024-02-01")])
        rows = paired_native_inference(frame, "wm", "regression", block_days=(7,), replicates=20)
        for row in rows:
            self.assertEqual(row["n_paired_rows"], 480)
            self.assertEqual(row["n_resampleable_segments"], 0)
            self.assertEqual(row["n_nonresampleable_rows"], 480)
            self.assertIsNone(row["ci_low"])
        mixed = pd.concat([daily_fixture(days=9), daily_fixture(days=1, segment=1, start="2024-02-01")])
        row = paired_native_inference(mixed, "wm", "classification", block_days=(7,), replicates=20)[0]
        self.assertEqual(row["n_nonresampleable_rows"], 60)
        self.assertEqual(row["n_resampleable_rows"], 540)
        self.assertTrue(row["inference_available"])

    def test_strict_pairing_and_fold_gap_boundaries_and_no_selection_input(self):
        frame = daily_fixture()
        first = paired_native_inference(frame, "wm", "classification", block_days=(1,), replicates=20)
        val = frame.copy(); val["split"] = "selection"; val["actual_sum"] = np.nan
        self.assertEqual(first, paired_native_inference(pd.concat([frame, val]), "wm", "classification", block_days=(1,), replicates=20))
        altered = frame.copy(); altered.loc[altered.index[0], "actual_sum"] += 1
        with self.assertRaisesRegex(ValueError, "identical outcomes"):
            paired_native_inference(altered, "direct", "regression", replicates=20)
        altered = frame.copy(); altered.loc[altered.utc_day >= pd.Timestamp("2024-01-05", tz="UTC"), "utc_day"] += pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "cannot bridge"):
            paired_native_inference(altered, "direct", "regression", replicates=20)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_native_inference(pd.concat([frame, frame.iloc[[0]]]), "direct", "regression", replicates=20)

    def test_writer_all_native_comparisons_and_holm_family_four(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary); (out / "final").mkdir()
            daily_fixture().to_csv(out / "final/day_statistics.csv", index=False)
            result = write_multitask_statistics(out, {"folds": [0], "statistics": {"bootstrap_replicates": 40}})
            self.assertEqual(result["n_primary_comparisons"], 4)
            table = pd.read_csv(out / "final/multitask_inference.csv")
            self.assertEqual(len(table), 42)
            primary = table[table.primary_inference]
            self.assertEqual(len(primary), 4)
            self.assertEqual(set(primary.block_days), {7})
            self.assertTrue((primary.p_holm_4 >= primary.p_value).all())
            self.assertTrue(table.loc[~table.primary_inference, "p_holm_4"].isna().all())
            json.dumps(result, allow_nan=False)
            json.loads((out / "statistics_completion.json").read_text())

    def test_absent_class_replicates_are_undefined_not_redefined_as_binary(self):
        frame = daily_fixture(days=5)
        frame["actual_c2"] = 0
        frame["prior_correct_c2"] = frame["raw_correct_c2"] = 0
        first = frame.utc_day.min()
        frame.loc[frame.utc_day == first, "actual_c2"] = 1
        frame.loc[(frame.utc_day == first) & (frame.arm == "joint"), "prior_correct_c2"] = 1
        frame["n_rows"] = frame[["actual_c0", "actual_c1", "actual_c2"]].sum(1)
        rows = paired_native_inference(frame, "wm", "classification", block_days=(1,), replicates=200)
        ba = next(row for row in rows if row["metric"] == "balanced_accuracy")
        self.assertGreater(ba["n_undefined_replicates"], 0)
        self.assertGreater(ba["n_valid_replicates"], 1)
        self.assertEqual(ba["n_valid_replicates"] + ba["n_undefined_replicates"], 200)
        self.assertTrue(ba["inference_available"])


if __name__ == "__main__":
    unittest.main()
