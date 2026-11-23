"""Expanded-family inference without modifying or re-fitting frozen studies."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from unidream.experiments import wm_lstm_significance_benchmark as addon


def cfg_fixture(root):
    raw = yaml.safe_load(addon.DEFAULT_CONFIG.read_text())
    raw["output_dir"] = str(root / "addon")
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return addon.load_config(path)


def records_fixture():
    records = []
    for group, model, task, form, head, family, metric, filter_name in sorted(addon.expected_slots(addon.ALL_MODELS)):
        baseline = filter_name or "fixed_chance" if family == "ba_chance" else filter_name or "causal_simple"
        estimate = .4 if form == "regression" and family == "ba_chance" else .6
        row = dict(group=group, model=model, task=task, formulation=form, head=head, family=family,
            metric=metric, baseline=baseline, block_days=7, role="primary" if family == "primary" else "auxiliary",
            estimate=estimate, ci_low=estimate - .03, ci_high=estimate + .03,
            baseline_estimate=.3, improvement=.1, improvement_ci_low=.05, improvement_ci_high=.15,
            p_value=1e-6, inference_available=True)
        row["hypothesis_id"] = "/".join(str(row[key]) for key in (*addon.KEYS, "head", "family", "baseline", "metric"))
        if model != "lstm":
            row.update(p_holm_primary448=.001, p_holm_global=.002)
        records.append(row)
    return records


def parent_fixture(root):
    sroot, lroot = root / "sig", root / "lstm"
    addon.atomic_json(sroot / "preregistration.json", {"old": "sig"})
    addon.atomic_json(lroot / "preregistration.json", {"old": "lstm"})
    return ({"output_dir": str(sroot)}, {"global_family_size": 1056, "primary_family_size": 448,
        "rank_backend": {"path": "frozen_rank.dylib", "binary_sha256": "frozen"}},
        {"output_dir": str(lroot)}, {"grid": {"cells": 84}})


class LSTMInferenceExtensionTests(unittest.TestCase):
    def test200_physical_jobs_and264_new_hypotheses_are_exact(self):
        jobs = addon.make_jobs()
        self.assertEqual(len(jobs), 200)
        self.assertEqual(len({task["job_id"] for task in jobs}), 200)
        self.assertEqual({task["model"] for task in jobs}, {"lstm"})
        self.assertEqual(Counter(task["kind"] for task in jobs), {
            "main": 84, "ic_aux": 48, "variance_ba": 36, "variance_qlike": 16, "crash_brier": 16})
        self.assertEqual(Counter(slot[5] for slot in addon.expected_slots()), addon.FAMILY_COUNTS)
        self.assertEqual(len(addon.expected_slots(addon.ALL_MODELS)), 1320)

    def test_config_rejects_weak_bootstrap_or_reduced_comparison_family(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = cfg_fixture(Path(temp))
            raw = yaml.safe_load(Path(cfg["config_path"]).read_text())
            raw["statistics"]["combined_global_family_size"] = 264
            Path(cfg["config_path"]).write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(ValueError, "1320"):
                addon.load_config(cfg["config_path"])

    def test_expanded_Holm_retains_old_raw_p_and_does_not_mutate_input(self):
        records = records_fixture()
        original = copy.deepcopy(records)
        old = next(row for row in records if row["model"] != "lstm" and row["family"] == "primary")
        old["p_value"] = 1 / 100001
        original = copy.deepcopy(records)
        result = addon.adjust_expanded(records)
        self.assertEqual(records, original)
        selected = next(row for row in result if row["hypothesis_id"] == old["hypothesis_id"])
        self.assertEqual(selected["p_value"], old["p_value"])
        self.assertEqual(selected["historical_p_holm_primary448"], .001)
        self.assertEqual(selected["historical_p_holm_global1056"], .002)
        self.assertIn("p_holm_primary560", selected)
        self.assertIn("p_holm_global1320", selected)
        self.assertEqual(sum(row["p_holm_primary560"] is not None for row in result), 560)

    def test_missing_or_duplicate_expanded_hypothesis_is_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                records = records_fixture()
                records = [*records, records[0]] if duplicate else records[:-1]
                with self.assertRaisesRegex(ValueError, "1320"):
                    addon.adjust_expanded(records)

    def test_unavailable_p_is_one_and_IC_has_no_hidden_hypothesis(self):
        records = records_fixture()
        records[0]["p_value"] = None
        unavailable_id = records[0]["hypothesis_id"]
        ic = {**records[0], "family": "ic_ci", "metric": "pearson_ic", "p_value": None,
            "hypothesis_id": "absolute_IC_only", "improvement": None}
        sensitivity = {**records[1], "block_days": 28}
        result = addon.adjust_expanded([*records, ic, sensitivity])
        self.assertEqual(next(row for row in result if row["hypothesis_id"] == unavailable_id)["p_holm_global1320"], 1.)
        self.assertIsNone(result[-1]["p_holm_global1320"])
        self.assertIsNone(result[-2]["p_holm_global1320"])
        self.assertIsNone(result[-2]["p_value"])

    def test_pre_registration_precedes_Test_and_refuses_already_open_LSTM(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = cfg_fixture(root)
            parents = parent_fixture(root)
            with patch.object(addon, "_parents", return_value=parents), \
                 patch.object(addon, "source_hashes", return_value={"new.py": "frozen"}), \
                 patch.object(addon, "build_cache", side_effect=AssertionError("Test opened during registration")):
                reg = addon.prepare(cfg)
                self.assertTrue(reg["registered_before_lstm_test"])
                self.assertEqual(reg["combined_global_hypotheses"], 1320)
                self.assertEqual(addon.verify(cfg), reg)
                addon.atomic_json(Path(parents[2]["output_dir"]) / "test_barrier.json", {"opened_utc": "2000-01-01T00:00:00Z"})
                with self.assertRaisesRegex(ValueError, "before LSTM Test"):
                    addon.verify(cfg)
                Path(cfg["output_dir"], "preregistration.json").unlink()
                with self.assertRaisesRegex(ValueError, "before the LSTM Test barrier"):
                    addon.prepare(cfg)

    def test_Test_cache_requires_own_registration_and_completed_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = cfg_fixture(root)
            with patch.object(addon.pd, "read_parquet", side_effect=AssertionError("Test read too early")):
                with self.assertRaisesRegex(ValueError, "registration"):
                    addon.build_cache(cfg, {})
                addon.atomic_json(Path(cfg["output_dir"]) / "preregistration.json", {})
                with patch.object(addon, "dependencies_ready", return_value=False):
                    with self.assertRaisesRegex(ValueError, "complete before Test"):
                        addon.build_cache(cfg, {})

    def test_dependencies_wait_for_both_complete_markers_and_match_registration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = cfg_fixture(root)
            sroot, lroot = root / "sig", root / "lstm"
            reg = {"significance_registration_sha256": "sig", "lstm_registration_sha256": "lstm"}
            with patch.object(addon.original, "load_config", return_value={"output_dir": str(sroot)}), \
                 patch.object(addon.lstm, "load_config", return_value={"output_dir": str(lroot)}):
                self.assertFalse(addon.dependencies_ready(cfg, reg))
                addon.atomic_json(sroot / "completion.json", {"complete": True, "status": "complete", "registration_sha256": "sig"})
                self.assertFalse(addon.dependencies_ready(cfg, reg))
                addon.atomic_json(lroot / "completion.json", {"complete": True, "status": "complete", "registration_sha256": "different"})
                with self.assertRaisesRegex(ValueError, "different registration"):
                    addon.dependencies_ready(cfg, reg)

    def test_job_reuse_rejects_manifest_from_another_addon_before_calculation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = cfg_fixture(root)
            out = Path(cfg["output_dir"])
            task = addon.make_jobs()[0]
            addon.atomic_json(out / "preregistration.json", {"own": "registered"})
            addon.atomic_json(out / "prepared/cache_manifest.json", {"complete": True, "registration_sha256": "wrong"})
            with patch.object(addon, "verify", return_value={"jobs": [task]}), \
                 patch.object(addon, "calculate_job_records", side_effect=AssertionError("calculated stale cache")):
                with self.assertRaisesRegex(ValueError, "different addon registration"):
                    addon.run_job(cfg["config_path"], task)

    def test_either_parent_failure_is_reported_even_if_other_is_still_running(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = cfg_fixture(root)
            sroot, lroot = root / "sig", root / "lstm"
            reg = {"significance_registration_sha256": "sig", "lstm_registration_sha256": "lstm"}
            addon.atomic_json(lroot / "failure.json", {"error": "synthetic failed LSTM task"})
            with patch.object(addon.original, "load_config", return_value={"output_dir": str(sroot)}), \
                 patch.object(addon.lstm, "load_config", return_value={"output_dir": str(lroot)}):
                with self.assertRaisesRegex(addon.DependencyFailure, "synthetic failed LSTM task"):
                    addon.dependencies_ready(cfg, reg)
                addon.atomic_json(lroot / "completion.json", {"complete": True, "status": "complete", "registration_sha256": "lstm"})
                # A successful immutable completion takes priority over an
                # archived/unremoved historical failure marker.
                self.assertFalse(addon.dependencies_ready(cfg, reg))

    def test_small_original_metrics_survive_round_trip_without_JSON_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old.csv"
            value = 1.234567891234567e-4
            p = 1 / 100001
            pd.DataFrame([{"estimate": value, "p_value": p, "missing": np.nan}]).to_csv(path, index=False)
            restored = addon.original._clean(pd.read_csv(path, float_precision="round_trip").to_dict("records"))
            self.assertEqual(restored[0]["estimate"], value)
            self.assertEqual(restored[0]["p_value"], p)
            self.assertIsNone(restored[0]["missing"])

    def test420_report_keeps_both_joint_heads_and84_LSTM_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "final").mkdir()
            rows = addon.adjust_expanded(records_fixture())
            frame = pd.DataFrame(rows)
            frame.to_csv(root / "final/all_inference.csv", index=False)
            points = []
            for group in "ABCD":
                for model in addon.ALL_MODELS:
                    for task in addon.TASKS:
                        for form in addon.FORMULATIONS:
                            points.append(dict(group=group, model=model, task=task, formulation=form,
                                primary_metric="balanced_accuracy" if form != "regression" else addon.NATIVE_PRIMARY[task],
                                primary_value=.6, balanced_accuracy=.4 if form == "regression" else .6,
                                regression_primary_value=.6 if form == "joint" else np.nan,
                                classification_primary_value=.6 if form == "joint" else np.nan))
            source = root / "points420.csv"
            pd.DataFrame(points).to_csv(source, index=False)
            before = source.read_bytes()
            report = addon.write_report(root, source, frame)
            summary = pd.read_csv(root / "final/results_420_significance.csv")
            self.assertEqual(report["cells"], 420)
            self.assertEqual(len(summary.loc[summary.model.eq("lstm")]), 84)
            self.assertTrue(summary.loc[summary.formulation.eq("joint"), "scalar_primary_ci_low"].notna().all())
            self.assertTrue(summary.loc[summary.formulation.eq("joint"), "class_BA_ci_low"].notna().all())
            self.assertEqual(source.read_bytes(), before)
            self.assertIn("直接の優位性検定ではありません", (root / "final/index.html").read_text())
            self.assertFalse(report["baseline_refitted"])


if __name__ == "__main__":
    unittest.main()
