"""Scientific accounting and unambiguous presentation of baseline inference."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_significance_report import (
    GROUPS, MODELS, TASKS, FORMULATIONS, KEYS, write_significance_report,
)


NATIVE = {"return": "zero_oos_r2", "variance": "qlike", "downside": "mae", "crash": "brier",
    "dd_excess": "mae", "advantage": "spearman_ic", "recovery": "spearman_ic"}


def fixture(root):
    final = root / "final"
    final.mkdir()
    points, inference = [], []
    for group in GROUPS:
        for model in MODELS:
            for task in TASKS:
                for formulation in FORMULATIONS:
                    base = dict(zip(KEYS, (group, model, task, formulation)))
                    points.append({**base, "primary_metric": "balanced_accuracy" if formulation != "regression" else NATIVE[task],
                        "primary_value": .6, "balanced_accuracy": .4 if formulation == "regression" else .6, "n_rows": 124488,
                        "regression_primary_value": .6 if formulation == "joint" else np.nan,
                        "classification_primary_value": .6 if formulation == "joint" else np.nan,
                        "control_only": model in ("linear", "lightgbm") and formulation == "joint"})
                    heads = ("regression", "classification") if formulation == "joint" else (formulation,)
                    for head in heads:
                        inference.append({**base, "head": head,
                            "metric": "balanced_accuracy" if head == "classification" else NATIVE[task],
                            "role": "primary", "family": "primary", "baseline": "causal_reference",
                            "split": "test", "block_days": 7, "model_estimate": .6,
                            "model_ci_low": .55, "model_ci_high": .65, "baseline_estimate": .5,
                            "difference": .1, "improvement": .1, "improvement_ci_low": .05,
                            "improvement_ci_high": .15, "p_value": .001, "p_holm_primary448": .02,
                            "p_holm_global": .03, "inference_available": True})
                    if formulation == "regression":
                        inference.append({**base, "head": "auxiliary", "metric": "balanced_accuracy",
                            "role": "auxiliary", "family": "ba_chance", "baseline": "chance_1_over_K",
                            "split": "test", "block_days": 7, "model_estimate": .4, "model_ci_low": .37,
                            "model_ci_high": .43, "baseline_estimate": 1 / (2 if task == "crash" else 3),
                            "improvement": .02, "improvement_ci_low": -.01, "improvement_ci_high": .05,
                            "p_value": .3, "p_holm_global": 1., "inference_available": True})
    pd.DataFrame(points).to_csv(root / "points.csv", index=False)
    pd.DataFrame(inference).to_csv(final / "all_inference.csv", index=False)
    return {"point_table_path": str(root / "points.csv"), "statistics": {"primary_block_days": 7}}, final


class SignificanceReportTests(unittest.TestCase):
    def test336_cells_preserve_two_joint_heads_and112_scalar_BA_proxies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, final = fixture(root)
            result = write_significance_report(root, cfg)
            table = pd.read_csv(final / "results_336_significance.csv")
            self.assertEqual(result["cells"], 336)
            self.assertEqual(result["n_primary_tests"], 448)
            self.assertEqual(len(table.drop_duplicates(list(KEYS))), 336)
            joint = table.loc[table.formulation.eq("joint")]
            self.assertEqual(int(joint.control_only.sum()), 56)
            self.assertTrue(joint.scalar_primary_ci_low.notna().all())
            self.assertTrue(joint.class_BA_ci_low.notna().all())
            reg = table.loc[table.formulation.eq("regression")]
            self.assertTrue(reg.class_BA_is_auxiliary_scalar_proxy.all())
            self.assertTrue(reg.classification_decision_origin.str.contains("scalar_forecast").all())
            self.assertTrue(np.allclose(reg.class_BA_ci_low, .37))
            self.assertTrue(table.loc[table.formulation.eq("classification"), "scalar_primary_ci_low"].isna().all())

    def test_significance_uses_direction_and_global_p_not_CI_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, final = fixture(root)
            rows = pd.read_csv(final / "all_inference.csv")
            rows.loc[rows.family.eq("primary"), "p_holm_global"] = .1
            negative = rows.family.eq("primary") & rows.task.eq("variance")
            rows.loc[negative, "improvement"] = -.1
            rows.loc[negative, "p_holm_global"] = .001
            rows.to_csv(final / "all_inference.csv", index=False)
            write_significance_report(root, cfg)
            summary = json.loads((final / "summary.json").read_text())
            self.assertEqual(summary["n_significant_improvements_global"], 0)
            self.assertGreater(summary["n_significant_improvements_holm448"], 0)
            self.assertFalse(summary["independently_unused_holdout"])
            self.assertTrue(summary["single_training_seed"])

    def test_missing_or_duplicate_primary_comparison_is_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, final = fixture(root)
                rows = pd.read_csv(final / "all_inference.csv")
                rows = pd.concat([rows, rows.iloc[[0]]]) if duplicate else rows.iloc[1:]
                rows.to_csv(final / "all_inference.csv", index=False)
                with self.assertRaisesRegex(ValueError, "448"):
                    write_significance_report(root, cfg)

    def test_missing336_condition_is_rejected_without_completing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, final = fixture(root)
            pd.read_csv(root / "points.csv").iloc[:-1].to_csv(root / "points.csv", index=False)
            with self.assertRaisesRegex(ValueError, "336"):
                write_significance_report(root, cfg)
            self.assertFalse((final / "report_completion.json").exists())

    def test_unavailable_primary_is_retained_with_reason_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, final = fixture(root)
            rows = pd.read_csv(final / "all_inference.csv")
            for field in ("model_ci_low", "model_ci_high", "improvement_ci_low", "improvement_ci_high", "p_value"):
                rows.loc[0, field] = np.nan
            rows.loc[0, "inference_available"] = False
            rows.loc[0, "inference_unavailable_reason"] = "no supported varying block segment"
            rows.loc[0, "p_holm_primary448"] = 1.
            rows.loc[0, "p_holm_global"] = 1.
            rows.to_csv(final / "all_inference.csv", index=False)
            write_significance_report(root, cfg)
            table = pd.read_csv(final / "results_336_significance.csv")
            row = table.iloc[0]
            self.assertEqual(len(table), 336)
            self.assertFalse(row.scalar_primary_inference_available)
            self.assertIn("no supported", row.scalar_primary_inference_unavailable_reason)
            self.assertFalse(row.scalar_primary_significant_improvement_global)

    def test_absolute_IC_intervals_are_exported_without_recomputing_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, final = fixture(root)
            rows = pd.read_csv(final / "all_inference.csv")
            extra = rows.iloc[0].copy()
            extra.update({"role": "auxiliary", "family": np.nan, "metric": "pearson_ic", "baseline": "none",
                "model_estimate": .07, "model_ci_low": .02, "model_ci_high": .11})
            pd.concat([rows, extra.to_frame().T]).to_csv(final / "all_inference.csv", index=False)
            write_significance_report(root, cfg)
            table = pd.read_csv(final / "results_336_significance.csv")
            self.assertAlmostEqual(table.iloc[0].native_pearson_ic_ci_low, .02)
            self.assertAlmostEqual(table.iloc[0].native_pearson_ic_ci_high, .11)

    def test_variance_secondary_comparisons_do_not_replace_registered_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, final = fixture(root)
            rows = pd.read_csv(final / "all_inference.csv")
            extra = rows.loc[rows.task.eq("variance") & rows["head"].eq("classification")].iloc[0].copy()
            extra.update({"role": "auxiliary", "family": "variance_BA", "baseline": "past_variance", "model_estimate": .99})
            pd.concat([rows, extra.to_frame().T]).to_csv(final / "all_inference.csv", index=False)
            write_significance_report(root, cfg)
            table = pd.read_csv(final / "results_336_significance.csv")
            row = table.loc[(table.task == "variance") & (table.formulation == "classification")].iloc[0]
            self.assertEqual(row.class_BA_baseline, "causal_reference")
            self.assertAlmostEqual(row.class_BA_estimate, .6)

    def test_confidence_intervals_and_p_values_are_validated(self):
        for field, value, message in (("p_holm_primary448", 1.5, "P values"),
                                      ("model_ci_low", .9, "reversed"),
                                      ("model_ci_low", np.nan, "finite")):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, final = fixture(root)
                rows = pd.read_csv(final / "all_inference.csv")
                rows.loc[0, field] = value
                rows.to_csv(final / "all_inference.csv", index=False)
                with self.assertRaisesRegex(ValueError, message):
                    write_significance_report(root, cfg)

    def test_only_Test_and_seven_day_primary_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, final = fixture(root)
            rows = pd.read_csv(final / "all_inference.csv")
            rows.loc[0, "split"] = "selection"
            rows.to_csv(final / "all_inference.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Test"):
                write_significance_report(root, cfg)

    def test_report_does_not_modify_either_input_and_preserves_audit_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, final = fixture(root)
            before_points = (root / "points.csv").read_bytes()
            before_inference = (final / "all_inference.csv").read_bytes()
            write_significance_report(root, cfg)
            self.assertEqual(before_points, (root / "points.csv").read_bytes())
            self.assertEqual(before_inference, (final / "all_inference.csv").read_bytes())
            contract = json.loads((final / "report_contract.json").read_text())
            self.assertTrue(contract["no_models_fitted"])
            self.assertTrue(contract["no_p_values_adjusted_by_report"])
            self.assertEqual(len(contract["source_results_336_sha256"]), 64)
            self.assertIn("探索的", (final / "index.html").read_text())
            self.assertIn("同時信頼区間ではありません", (final / "index.html").read_text())

    def test_new_estimates_must_reproduce_the_frozen_point_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, final = fixture(root)
            rows = pd.read_csv(final / "all_inference.csv")
            rows.loc[0, "model_estimate"] = .7
            rows.to_csv(final / "all_inference.csv", index=False)
            with self.assertRaisesRegex(ValueError, "frozen336"):
                write_significance_report(root, cfg)


if __name__ == "__main__":
    unittest.main()
