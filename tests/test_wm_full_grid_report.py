"""Full-cell exports, fold-specific pooling and immutable-source report gates."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_full_grid_evaluation import TASKS, make_task_labels, score_grid
from unidream.experiments.wm_full_grid_report import _sha, write_full_grid_report


def contract(fold):
    multiplier = fold + 1
    result = {}
    for task in TASKS:
        kind = "binary_event" if task == "crash" else "zero_positive_median" if task == "dd_excess" else "train_tertiles" if task in ("variance", "downside") else "signed_rms"
        bounds = {"return": [-.005 * multiplier, .005 * multiplier],
                  "variance": [.001 * multiplier, .002 * multiplier],
                  "downside": [.01 * multiplier, .02 * multiplier], "crash": [.5],
                  "dd_excess": [0., .01 * multiplier],
                  "advantage": [-.001 * multiplier, .001 * multiplier],
                  "recovery": [-.1 * multiplier, .1 * multiplier]}[task]
        result[task] = {"kind": kind, "bounds": bounds, "n_classes": 2 if task == "crash" else 3,
                        "priors": [.7, .3] if task == "crash" else ([.2, .6, .2] if fold == 0 else [.3, .4, .3])}
    return result


def actual_values():
    return {"return": np.array([-.04, -.01, 0., .01, .03, .05]),
            "variance": np.array([.0001, .0005, .0015, .0025, .004, .008]),
            "downside": np.array([0., .005, .015, .025, .04, .08]),
            "crash": np.array([0., 1., 0., 1., 0., 1.]),
            "dd_excess": np.array([0., 0., .005, .015, .025, .04]),
            "advantage": np.array([-.01, -.002, 0., .002, .005, .01]),
            "recovery": np.array([-.8, -.15, 0., .15, .4, .8])}


def panel(fold, group, model, formulation, split):
    c = contract(fold)
    timestamp = pd.Timestamp("2024-03-01", tz="UTC") + pd.DateOffset(months=3 * fold + (1 if split == "test" else 0))
    # One missing 15-minute bar creates two separately identifiable segments.
    offsets = np.array([0, 1, 2, 4, 5, 6])
    value = pd.DataFrame({"decision_id": fold * 100 + (1000 if split == "test" else 0) + offsets,
        "timestamp": timestamp + pd.to_timedelta(offsets * 15, unit="min"), "fold": fold,
        "group": group, "seed": 7, "model": model, "formulation": formulation, "split": split})
    for task, actual in actual_values().items():
        labels = make_task_labels(actual, task, c)
        value["actual_" + task] = actual
        value["label_" + task] = labels
        value["scalar_" + task] = (.012 if task == "return" else np.where(actual > .5, .9, .1) if task == "crash" else actual) if formulation == "regression" else np.nan
        classes = 2 if task == "crash" else 3
        probability = .8 * np.eye(classes)[labels] + .2 / classes
        if task == "return":
            probability = np.tile([.25, .5, .25], (len(actual), 1))
        for index in range(3):
            value[f"p_{task}_{index}"] = probability[:, index] if formulation == "classification" and index < classes else np.nan
    return value


class FullGridReportTests(unittest.TestCase):
    def fixture(self, root, with_test=False):
        neural, classical = [], []
        for fold in range(2):
            for group in "ABCD":
                for model in ("direct", "wm"):
                    for formulation in ("regression", "classification"):
                        neural.append({"fold": fold, "group": group, "model": model, "formulation": formulation,
                                       "seed": 7, "job": f"fold_{fold:02d}_{group}_{model}_{formulation}_s7"})
                classical.append({"fold": fold, "group": group, "job": f"fold_{fold:02d}_{group}"})
        scales = {task: [1.] for task in TASKS}
        reg = {"folds": [{"fold": fold, "label_contract": contract(fold), "scales": scales} for fold in range(2)],
               "neural_tasks": neural, "classical_tasks": classical}
        (root / "preregistration.json").write_text(json.dumps(reg))
        reg_hash = _sha(root / "preregistration.json")
        fits = {}
        for kind, tasks in (("neural", neural), ("classical", classical)):
            for task in tasks:
                directory = root / kind / task["job"]
                directory.mkdir(parents=True)
                variants = [(task["model"], task["formulation"])] if kind == "neural" else [(model, formulation) for model in ("linear", "lightgbm") for formulation in ("regression", "classification")]
                for split in (["selection", "test"] if with_test else ["selection"]):
                    frames = [panel(task["fold"], task["group"], model, formulation, split) for model, formulation in variants]
                    path = directory / ("test_predictions.parquet" if split == "test" else "validation_predictions.parquet")
                    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
                    if split == "test":
                        (directory / "evaluation.json").write_text(json.dumps({"prediction_sha256": _sha(path)}))
                fit = {"registration_sha256": reg_hash, "validation_predictions_sha256": _sha(directory / "validation_predictions.parquet")}
                (directory / "fit.json").write_text(json.dumps(fit))
                fits[str((directory / "fit.json").relative_to(root))] = _sha(directory / "fit.json")
        if with_test:
            (root / "validation_choices.json").write_text('{"frozen": true}')
            (root / "test_barrier.json").write_text(json.dumps({"registration_sha256": reg_hash,
                "frozen_fit_sha256": fits, "validation_choices_sha256": _sha(root / "validation_choices.json")}))
        return {"output_dir": str(root), "groups": list("ABCD")}

    def test_selection_export_has_all_cells_and_uses_each_folds_map_before_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            summary = write_full_grid_report(root, cfg)
            self.assertEqual(summary["cells_per_split"], {"selection": 224})
            self.assertEqual(summary["n_fold_metric_rows"], 448)
            table = pd.read_csv(root / "selection/results_224.csv")
            self.assertEqual(len(table), 224)
            self.assertEqual(set(table.n_folds), {2})
            self.assertEqual(set(table.n_rows), {12})
            self.assertEqual(set(table.n_seeds), {1})
            details = json.loads((root / "selection/results_224.json").read_text())
            for formulation in ("regression", "classification"):
                selected = next(row for row in details if row["group"] == "A" and row["model"] == "linear" and row["task"] == "return" and row["formulation"] == formulation)
                folded = []
                for fold in range(2):
                    frame = panel(fold, "A", "linear", formulation, "selection")
                    forecast = frame.scalar_return.to_numpy() if formulation == "regression" else frame[["p_return_0", "p_return_1", "p_return_2"]].to_numpy()
                    folded.append(score_grid("return", frame.actual_return.to_numpy(), forecast, formulation, contract(fold), None))
                expected_confusion = sum(np.array(row["classification"]["confusion_matrix"]) for row in folded)
                np.testing.assert_array_equal(selected["classification"]["confusion_matrix"], expected_confusion)
                expected_fold_ba = np.mean([row["classification"]["balanced_accuracy"] for row in folded])
                self.assertAlmostEqual(selected["fold_mean_balanced_accuracy"], expected_fold_ba)
            self.assertTrue((root / "selection/index.html").exists())

    def test_daily_counts_preserve_gaps_and_reproduce_pooled_confusion_diagonal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            write_full_grid_report(root, cfg)
            daily = pd.read_csv(root / "selection/day_classification_counts.csv")
            self.assertEqual(set(daily.segment_id), {0, 1})
            self.assertEqual(int(daily.n_rows.sum()), 224 * 12)
            for index in range(3):
                self.assertTrue((daily[f"correct_c{index}"] <= daily[f"actual_c{index}"]).all())
            crash = daily.loc[daily.task == "crash"]
            self.assertTrue((crash.actual_c2 == 0).all())
            self.assertTrue((crash.correct_c2 == 0).all())
            details = json.loads((root / "selection/results_224.json").read_text())
            chosen = next(row for row in details if row["group"] == "A" and row["model"] == "linear" and row["task"] == "return" and row["formulation"] == "classification")
            rows = daily.loc[(daily.group == "A") & (daily.model == "linear") & (daily.task == "return") & (daily.formulation == "classification")]
            matrix = np.array(chosen["classification"]["confusion_matrix"])
            np.testing.assert_array_equal([rows[f"actual_c{i}"].sum() for i in range(3)], matrix.sum(1))
            np.testing.assert_array_equal([rows[f"correct_c{i}"].sum() for i in range(3)], matrix.diagonal())

    def test_missing_fit_keeps_report_closed_and_test_requires_frozen_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "Test barrier is closed"):
                write_full_grid_report(root, cfg, include_test=True)
            next((root / "neural").glob("*/fit.json")).unlink()
            with self.assertRaisesRegex(ValueError, "requires every registered fit"):
                write_full_grid_report(root, cfg)

    def test_cohort_changes_cannot_be_hidden_by_updating_a_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root)
            directory = root / "neural/fold_00_A_direct_regression_s7"
            path = directory / "validation_predictions.parquet"
            pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
            fit = json.loads((directory / "fit.json").read_text())
            fit["validation_predictions_sha256"] = _sha(path)
            (directory / "fit.json").write_text(json.dumps(fit))
            with self.assertRaisesRegex(ValueError, "same outcomes, labels and time cohort"):
                write_full_grid_report(root, cfg)

    def test_final_export_keeps_selection_and_test_separate_and_checks_test_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.fixture(root, with_test=True)
            summary = write_full_grid_report(root, cfg, include_test=True)
            self.assertEqual(summary["cells_per_split"], {"selection": 224, "test": 224})
            table = pd.read_csv(root / "final/results_224.csv")
            self.assertEqual(len(table), 224)
            self.assertEqual(set(table.split), {"test"})
            self.assertEqual(summary["n_result_rows"], 224)
            self.assertEqual(len(json.loads((root / "final/results_224.json").read_text())), 224)
            selection = pd.read_csv(root / "final/selection_results_224.csv")
            self.assertEqual(len(selection), 224)
            self.assertEqual(set(selection.split), {"selection"})
            combined = pd.read_csv(root / "final/all_split_metrics.csv")
            self.assertEqual(len(combined), 448)
            self.assertEqual(combined.groupby("split").size().to_dict(), {"selection": 224, "test": 224})
            path = next((root / "neural").glob("*/test_predictions.parquet"))
            frame = pd.read_parquet(path)
            frame.loc[0, "scalar_return"] = .123
            frame.to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "frozen Test predictions changed"):
                write_full_grid_report(root, cfg, include_test=True)


if __name__ == "__main__":
    unittest.main()
