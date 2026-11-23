"""Full336 accounting, dual native scores, cohort gates and exact controls."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import test_wm_full_grid_report as controls
from unidream.experiments.wm_full_grid_report import write_full_grid_report, _sha
from unidream.experiments.wm_joint_grid_report import write_joint_grid_report
from unidream.experiments.wm_joint_grid_statistics import write_joint_grid_statistics


def fixture(root, with_test=True):
    parent = root / "parent"; parent.mkdir()
    cfg = controls.FullGridReportTests().fixture(parent, with_test=with_test)
    write_full_grid_report(parent, cfg, include_test=with_test)
    joint = root / "joint"; joint.mkdir()
    p = json.loads((parent / "preregistration.json").read_text())
    neural = [{**x, "formulation": "joint", "job": x["job"].replace("regression", "joint")}
              for x in p["neural_tasks"] if x["formulation"] == "regression"]
    classical = p["classical_tasks"]
    registration = {"folds": p["folds"], "neural_tasks": neural, "classical_tasks": classical,
        "parent_registration_sha256": _sha(parent / "preregistration.json")}
    (joint / "preregistration.json").write_text(json.dumps(registration))
    reg_sha = _sha(joint / "preregistration.json")
    fits = {}
    for kind, tasks in (("neural", neural), ("classical", classical)):
        for task in tasks:
            destination = joint / kind / task["job"]; destination.mkdir(parents=True)
            models = [task["model"]] if kind == "neural" else ["linear", "lightgbm"]
            for split in (["selection", "test"] if with_test else ["selection"]):
                frames = []
                for model in models:
                    frame = controls.panel(task["fold"], task["group"], model, "regression", split)
                    probability = controls.panel(task["fold"], task["group"], model, "classification", split)
                    frame["formulation"] = "joint"
                    for name in probability.columns:
                        if name.startswith("p_"):
                            frame[name] = probability[name]
                    if kind == "neural":
                        frame["scalar_return"] = .8 * frame.actual_return
                    frames.append(frame)
                target = destination / ("test_predictions.parquet" if split == "test" else "validation_predictions.parquet")
                pd.concat(frames, ignore_index=True).to_parquet(target, index=False)
                if split == "test":
                    (destination / "evaluation.json").write_text(json.dumps({"prediction_sha256": _sha(target)}))
            (destination / "fit.json").write_text(json.dumps({"registration_sha256": reg_sha,
                "validation_predictions_sha256": _sha(destination / "validation_predictions.parquet")}))
            fits[str((destination / "fit.json").relative_to(joint))] = _sha(destination / "fit.json")
    if with_test:
        (joint / "validation_choices.json").write_text('{"frozen": true}')
        (joint / "test_barrier.json").write_text(json.dumps({"registration_sha256": reg_sha,
            "frozen_fit_sha256": fits, "validation_choices_sha256": _sha(joint / "validation_choices.json"),
            "parent_results_224_sha256": _sha(parent / "final/results_224.csv")}))
    return joint, {"output_dir": str(joint), "parent_output_dir": str(parent), "groups": list("ABCD"),
        "folds": [0, 1], "seeds": [7], "statistics": {"bootstrap_replicates": 20, "block_days": [1, 7, 28], "primary_block_days": 7}}


class JointGridReportTests(unittest.TestCase):
    def test_selection_has112_joint_cells_and_never_requires_parent_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = fixture(Path(tmp), with_test=False)
            result = write_joint_grid_report(root, cfg)
            self.assertEqual(result["n_result_rows"], 112)
            self.assertFalse((root / "final/results_336.csv").exists())
            table = pd.read_csv(root / "selection/joint_results_112.csv")
            self.assertEqual(int(table.shared_parameters.sum()), 56)
            self.assertTrue(np.isfinite(table.regression_primary_value).all())
            self.assertTrue(np.isfinite(table.classification_balanced_accuracy).all())

    def test_final_preserves224_original_cells_and_marks56_algebraic_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = fixture(Path(tmp))
            result = write_joint_grid_report(root, cfg, include_test=True)
            table = pd.read_csv(root / "final/results_336.csv")
            self.assertEqual(result["n_result_rows"], 336)
            self.assertEqual(len(table.drop_duplicates(["group", "model", "task", "formulation"])), 336)
            old = pd.read_csv(Path(cfg["parent_output_dir"]) / "final/results_224.csv")
            pd.testing.assert_frame_equal(table.loc[table.formulation != "joint", old.columns].reset_index(drop=True), old,
                check_dtype=False)
            joint = table.loc[table.formulation == "joint"]
            classical = joint.loc[joint.model.isin(["linear", "lightgbm"])]
            self.assertEqual(len(classical), 56)
            self.assertTrue((classical.fresh_training == False).all())
            self.assertTrue((classical.shared_parameters == False).all())
            self.assertTrue((classical.control_only == True).all())
            self.assertEqual(set(classical.training_origin), {"frozen224_regression_and_classification_fits"})
            contract = json.loads((root / "final/report_contract.json").read_text())
            self.assertEqual(contract["distinct_empirical_prediction_cells"], 280)
            self.assertFalse(contract["all_336_are_fresh_training"])

    def test_open_test_requires_both_own_and_parent_frozen_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = fixture(Path(tmp))
            (Path(cfg["parent_output_dir"]) / "test_barrier.json").unlink()
            with self.assertRaisesRegex(ValueError, "Test barrier is closed"):
                write_joint_grid_report(root, cfg, include_test=True)

    def test_hash_resigning_cannot_hide_different_joint_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = fixture(Path(tmp), with_test=False)
            destination = next((root / "neural").iterdir())
            path = destination / "validation_predictions.parquet"
            pd.read_parquet(path).iloc[:-1].to_parquet(path, index=False)
            fit = json.loads((destination / "fit.json").read_text())
            fit["validation_predictions_sha256"] = _sha(path)
            (destination / "fit.json").write_text(json.dumps(fit))
            with self.assertRaisesRegex(ValueError, "identical full outcome/time cohort"):
                write_joint_grid_report(root, cfg)

    def test_parent224_csv_is_bound_to_own_frozen_test_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = fixture(Path(tmp))
            path = Path(cfg["parent_output_dir"]) / "final/results_224.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "primary_value"] += .01
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "changed after the joint Test barrier"):
                write_joint_grid_report(root, cfg, include_test=True)

    def test_complete_inference_has112_neural_contrasts_and112_exact_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = fixture(Path(tmp))
            write_joint_grid_report(root, cfg, include_test=True)
            result = write_joint_grid_statistics(root, cfg)
            self.assertEqual(result["n_primary_comparisons"], 112)
            self.assertEqual(result["n_deterministic_classical_controls"], 112)
            table = pd.read_csv(root / "final/joint_inference.csv")
            self.assertEqual(len(table), 224 * 3)
            control = table.loc[table.control_only]
            self.assertTrue(control.estimate.eq(0).all())
            self.assertTrue(control.p_value.isna().all())
            self.assertTrue(control.replicates.eq(0).all())
            rank = table.loc[(table.task.isin(["advantage", "recovery"])) & (table["head"] == "regression")]
            self.assertTrue(rank.inference_metric.eq("rmse").all())
            self.assertTrue(rank.report_primary_metric.eq("spearman_ic").all())
            self.assertTrue(rank.secondary_diagnostic.all())

    def test_changed_classical_control_cannot_be_claimed_as_joint_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, cfg = fixture(Path(tmp))
            write_joint_grid_report(root, cfg, include_test=True)
            path = root / "final/day_native_statistics.csv"
            table = pd.read_csv(path)
            selection = (table.model == "linear") & (table.formulation == "joint")
            table.loc[selection, "sum_squared_error"] += 1
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly reuse regression outputs"):
                write_joint_grid_statistics(root, cfg)


if __name__ == "__main__":
    unittest.main()
