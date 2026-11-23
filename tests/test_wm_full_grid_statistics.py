import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unidream.experiments import wm_full_grid_statistics as statistics


def daily_frame(task="return", days=10, *, fold=0, segment=0, start="2024-01-01"):
    classes = 2 if task == "crash" else 3
    rows = []
    for day, date in enumerate(pd.date_range(start, periods=days, freq="D", tz="UTC")):
        support = [20, 10, 5] if classes == 3 else [20, 10, 0]
        for formulation in ("regression", "classification"):
            correct = ([12, 5 + day % 2, 2] if formulation == "regression"
                       else [15 + day % 3, 7, 3])
            if classes == 2:
                correct[2] = 0
            row = {"split": "test", "group": "A", "model": "linear", "task": task,
                   "formulation": formulation, "fold": fold, "seed": 7,
                   "utc_day": date, "segment_id": segment,
                   "n_classes": classes, "n_rows": sum(support)}
            row.update({f"actual_c{i}": support[i] for i in range(3)})
            row.update({f"correct_c{i}": correct[i] for i in range(3)})
            rows.append(row)
    return pd.DataFrame(rows)


def independent_difference(frame):
    values = {}
    selected = frame[frame.split == "test"]
    k = int(selected.n_classes.iloc[0])
    for formulation, part in selected.groupby("formulation"):
        correct = np.array([part[f"correct_c{i}"].sum() for i in range(k)], dtype=float)
        support = np.array([part[f"actual_c{i}"].sum() for i in range(k)], dtype=float)
        values[formulation] = float(np.mean(correct / support))
    return values["classification"] - values["regression"]


class FullGridStatisticsTests(unittest.TestCase):
    def test_binary_and_ternary_use_fixed_correct_class_universe(self):
        for task, expected_k in (("crash", 2), ("return", 3)):
            frame = daily_frame(task)
            result = statistics.paired_day_count_inference(frame, block_days=(1, 7, 28), replicates=80, random_seed=9)
            for row in result:
                self.assertEqual(row["n_classes"], expected_k)
                self.assertAlmostEqual(row["estimate"], independent_difference(frame), places=13)
                self.assertEqual(row["n_paired_rows"], frame[frame.formulation == "classification"].n_rows.sum())
            self.assertTrue(result[0]["inference_available"])
            self.assertIsNone(result[-1]["ci_low"])
            self.assertIsNone(result[-1]["p_value"])

    def test_prefix_sampling_exactly_matches_explicit_non_circular_block_indices(self):
        values = np.arange(11 * 4).reshape(11, 4).astype(float)
        days, replicates = 3, 20
        actual = statistics._block_sums(values, days, replicates, np.random.default_rng(81))
        rng = np.random.default_rng(81)
        starts = rng.integers(0, len(values) - days + 1, size=(replicates, 4))
        indices = (starts[:, :, None] + np.arange(days)).reshape(replicates, -1)[:, :len(values)]
        expected = values[indices].sum(1)
        np.testing.assert_array_equal(actual, expected)

    def test_short_segments_and_fold_boundaries_never_create_artificial_block_support(self):
        frame = pd.concat([daily_frame(days=4),
                           daily_frame(days=4, segment=1, start="2024-01-11"),
                           daily_frame(days=4, fold=1, start="2024-02-01")], ignore_index=True)
        result = statistics.paired_day_count_inference(frame, block_days=(7,), replicates=40)[0]
        self.assertEqual(result["n_segments"], 3)
        self.assertEqual(result["n_resampleable_segments"], 0)
        self.assertEqual(result["n_segments_shorter_than_block"], 3)
        self.assertEqual(result["resampleable_row_fraction"], 0.)
        self.assertEqual(result["n_nonresampleable_rows"], result["n_paired_rows"])
        self.assertAlmostEqual(result["estimate"], independent_difference(frame))
        self.assertFalse(result["inference_available"])
        self.assertIsNone(result["p_value"])
        self.assertIsNone(result["ci_low"])

    def test_short_segment_stays_in_point_estimate_and_all_bootstrap_replicates(self):
        long = daily_frame(days=9)
        short = daily_frame(days=1, segment=1, start="2024-02-01")
        for category in range(3):
            short.loc[short.formulation == "classification", f"correct_c{category}"] = 0
            short.loc[short.formulation == "regression", f"correct_c{category}"] = short.loc[
                short.formulation == "regression", f"actual_c{category}"]
        frame = pd.concat([long, short], ignore_index=True)
        result = statistics.paired_day_count_inference(frame, block_days=(7,), replicates=100)[0]
        self.assertAlmostEqual(result["estimate"], independent_difference(frame))
        self.assertNotAlmostEqual(result["estimate"], independent_difference(long))
        self.assertEqual(result["n_nonresampleable_rows"], 35)
        self.assertEqual(result["n_resampleable_rows"], 315)
        self.assertTrue(result["inference_available"])
        self.assertIn("replayed fixed", result["unsupported_segment_policy"])

    def test_selection_rows_are_not_used_and_resampling_is_deterministic(self):
        frame = daily_frame()
        selection = frame.copy()
        selection["split"] = "selection"
        selection["actual_c0"] = np.nan
        first = statistics.paired_day_count_inference(frame, block_days=(1, 7), replicates=40, random_seed=7)
        second = statistics.paired_day_count_inference(pd.concat([frame, selection]), block_days=(1, 7), replicates=40, random_seed=7)
        self.assertEqual(first, second)
        # Identical paired predictions give exact zero differences and P=1.
        identical = frame.copy()
        for category in range(3):
            values = identical.loc[identical.formulation == "regression", f"correct_c{category}"].to_numpy()
            identical.loc[identical.formulation == "classification", f"correct_c{category}"] = values
        result = statistics.paired_day_count_inference(identical, block_days=(1,), replicates=40)[0]
        self.assertEqual(result["estimate"], 0.)
        self.assertEqual(result["ci_low"], 0.)
        self.assertEqual(result["ci_high"], 0.)
        self.assertEqual(result["p_value"], 1.)

    def test_absent_class_bootstrap_replicates_are_counted_not_redefined_as_binary(self):
        frame = daily_frame(days=5)
        frame["actual_c2"] = 0
        frame["correct_c2"] = 0
        first_day = frame.utc_day.min()
        frame.loc[frame.utc_day == first_day, "actual_c2"] = 1
        frame.loc[(frame.utc_day == first_day) & (frame.formulation == "classification"), "correct_c2"] = 1
        frame["n_rows"] = frame[["actual_c0", "actual_c1", "actual_c2"]].sum(1)
        result = statistics.paired_day_count_inference(frame, block_days=(1,), replicates=200, random_seed=83)[0]
        self.assertEqual(result["n_classes"], 3)
        self.assertGreater(result["n_undefined_replicates"], 0)
        self.assertGreater(result["n_valid_replicates"], 1)
        self.assertEqual(result["n_valid_replicates"] + result["n_undefined_replicates"], 200)
        self.assertTrue(result["inference_available"])

    def test_pairing_mismatches_duplicates_and_day_gaps_are_rejected(self):
        frame = daily_frame()
        altered = frame.copy()
        altered.loc[0, "actual_c0"] += 1
        altered.loc[0, "actual_c1"] -= 1
        with self.assertRaisesRegex(ValueError, "identical class/day support"):
            statistics.paired_day_count_inference(altered, block_days=(1,), replicates=10)
        with self.assertRaisesRegex(ValueError, "identical fold/segment/day"):
            statistics.paired_day_count_inference(frame.iloc[1:], block_days=(1,), replicates=10)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            statistics.paired_day_count_inference(pd.concat([frame, frame.iloc[[0]]]), block_days=(1,), replicates=10)
        altered = frame.copy()
        altered.loc[altered.utc_day >= pd.Timestamp("2024-01-05", tz="UTC"), "utc_day"] += pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "cannot bridge"):
            statistics.paired_day_count_inference(altered, block_days=(1,), replicates=10)

    def test_writer_requires_all112_comparisons_and_preserves_full_holm_family(self):
        frames = []
        for group in statistics.GROUPS:
            for model in statistics.MODELS:
                for task in statistics.TASKS:
                    frame = daily_frame(task, days=9)
                    frame["group"], frame["model"] = group, model
                    frames.append(frame)
        config = {"folds": [0], "seeds": [7], "statistics": {"bootstrap_replicates": 12,
                  "block_days": [1, 7, 28], "primary_block_days": 7, "seed": 7}}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "final").mkdir()
            source = out / "final/day_classification_counts.csv"
            pd.concat(frames, ignore_index=True).to_csv(source, index=False)
            summary = statistics.write_grid_statistics(out, config)
            self.assertEqual(summary["n_comparisons"], 112)
            self.assertEqual(summary["n_rows"], 336)
            self.assertTrue(summary["complete"])
            report = pd.read_csv(out / "final/formulation_inference.csv")
            self.assertEqual(report.holm_family_size.unique().tolist(), [112])
            primary = report[report.primary_inference]
            self.assertEqual(len(primary), 112)
            self.assertTrue((primary.p_holm_112 >= primary.p_value).all())
            self.assertTrue(report.loc[report.block_days == 28, "p_value"].isna().all())
            marker = json.loads((out / "statistics_completion.json").read_text())
            self.assertEqual(marker["daily_counts_sha256"], statistics._sha(source))
            frame.iloc[:2].to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "all112"):
                statistics.write_grid_statistics(out, config)


if __name__ == "__main__":
    unittest.main()
