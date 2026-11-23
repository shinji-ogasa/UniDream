import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_classification_complementarity import (
    MODELS, align_predictions, complementarity_tables,
)


class ComplementarityCountsTest(unittest.TestCase):
    def test_selection_oracle_is_union_and_jaccard_uses_correct_sets(self):
        actual = np.array([0, 0, 1, 1, 2, 2])
        a = np.array([0, 1, 1, 0, 2, 0])
        b = np.array([1, 0, 1, 0, 0, 2])
        result = complementarity_tables(actual, np.column_stack([a, b]), models=("a", "b"))
        pair = result["pairs"][0]
        self.assertEqual([pair[k] for k in ("both_correct_n", "a_only_correct_n", "b_only_correct_n", "both_wrong_n")], [1, 2, 2, 1])
        self.assertAlmostEqual(pair["correct_set_jaccard"], 1 / 5)
        self.assertAlmostEqual(pair["rescue_b_given_a_wrong"], 2 / 3)
        self.assertAlmostEqual(pair["oracle_balanced_accuracy"], 5 / 6)
        self.assertAlmostEqual(pair["oracle_gain_over_best_single_ba"], 1 / 3)

    def test_balanced_accuracy_is_not_accuracy_on_imbalanced_classes(self):
        actual = np.array([0, *([1] * 8), 2])
        result = complementarity_tables(actual, np.ones((10, 2), dtype=int), models=("a", "b"))
        pair = result["pairs"][0]
        self.assertAlmostEqual(pair["oracle_accuracy"], .8)
        self.assertAlmostEqual(pair["oracle_balanced_accuracy"], 1 / 3)
        self.assertEqual(pair["oracle_gain_over_best_single_ba"], 0)
        self.assertEqual(pair["correct_set_jaccard"], 1)
        self.assertEqual(pair["rescue_b_given_a_wrong"], 0)

    def test_all_subsets_and_patterns_partition_rows_and_oracle_is_monotonic(self):
        actual = np.array([0, 1, 2, 0, 1, 2])
        pred = np.column_stack([actual, (actual + 1) % 3, (actual + 2) % 3, np.ones(6)])
        tables = complementarity_tables(actual, pred)
        self.assertEqual(len(tables["subsets"]), 15)
        self.assertEqual(len(tables["pairs"]), 6)
        self.assertEqual(len(tables["patterns"]), 16)
        self.assertEqual(sum(r["n"] for r in tables["patterns"]), 6)
        for row in tables["subsets"]:
            self.assertGreaterEqual(row["oracle_gain_over_best_single_ba"], -1e-12)
        # Constant classes have a perfect selection oracle without predictive information.
        constant = complementarity_tables(actual, np.tile([0, 1, 2], (6, 1)), models=("a", "b", "c"))
        self.assertAlmostEqual(constant["subsets"][-1]["oracle_balanced_accuracy"], 1)

    def test_absent_classes_and_invalid_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            complementarity_tables([0, 1], np.zeros((2, 4)))
        with self.assertRaises(ValueError):
            complementarity_tables([0, 1, 2], np.full((3, 4), 3))

    def test_undefined_jaccard_and_rescue_are_explicit(self):
        actual = np.array([0, 1, 2])
        none = complementarity_tables(actual, np.tile((actual + 1)[:, None] % 3, (1, 2)), models=("a", "b"))["pairs"][0]
        self.assertIsNone(none["correct_set_jaccard"])
        all_correct = complementarity_tables(actual, np.tile(actual[:, None], (1, 2)), models=("a", "b"))["pairs"][0]
        self.assertIsNone(all_correct["rescue_b_given_a_wrong"])


class ComplementarityAlignmentTest(unittest.TestCase):
    @staticmethod
    def frame():
        rows = []
        for split in ("val", "test"):
            for model in MODELS:
                for seed in ((0,) if model in ("logistic", "lightgbm") else (7, 17, 27)):
                    for c in range(3):
                        # Different neural seeds intentionally give different answers.
                        pred = c if seed in (0, 7) else (c + 1) % 3
                        probs = np.full(3, .1)
                        probs[pred] = .8
                        rows.append({"condition": "fixed", "model": model, "fold": 0, "seed": seed,
                            "split": split, "decision_id": c, "timestamp": pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(minutes=15 * c),
                            "actual": c, "theta": .01, "return": [-.02, 0, .02][c],
                            "p0": probs[0], "p1": probs[1], "p2": probs[2], "pred": pred, "pred_raw": pred})
        return pd.DataFrame(rows).sample(frac=1, random_state=3).reset_index(drop=True)

    def test_seeds_are_paired_not_unioned_and_classical_is_broadcast(self):
        aligned = align_predictions(self.frame(), (7, 17, 27), {0: np.full(3, 1 / 3)})
        self.assertEqual(len(aligned), 18)
        self.assertTrue((aligned.source_seed_lightgbm == 0).all())
        self.assertTrue((aligned.source_seed_wm == aligned.seed).all())
        self.assertAlmostEqual(aligned.correct_wm.mean(), 1 / 3)
        np.testing.assert_array_equal(aligned.correct_wm, aligned.correct_direct)

    def test_missing_dates_and_duplicate_keys_are_rejected(self):
        frame = self.frame()
        with self.assertRaisesRegex(ValueError, "coverage"):
            align_predictions(frame.iloc[1:], (7, 17, 27), {0: np.full(3, 1 / 3)})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            align_predictions(pd.concat([frame, frame.iloc[:1]]), (7, 17, 27), {0: np.full(3, 1 / 3)})

    def test_mismatched_truth_and_saved_decisions_are_rejected(self):
        frame = self.frame()
        index = frame.index[frame.model == "wm"][0]
        changed = frame.copy()
        changed.loc[index, "actual"] = (changed.loc[index, "actual"] + 1) % 3
        with self.assertRaisesRegex(ValueError, "truth"):
            align_predictions(changed, (7, 17, 27), {0: np.full(3, 1 / 3)})
        changed = frame.copy()
        changed.loc[index, "pred"] = (changed.loc[index, "pred"] + 1) % 3
        with self.assertRaisesRegex(ValueError, "prior-adjusted"):
            align_predictions(changed, (7, 17, 27), {0: np.full(3, 1 / 3)})


if __name__ == "__main__":
    unittest.main()
