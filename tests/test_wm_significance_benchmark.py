"""Freeze, resume, alias accounting and multiplicity integration contracts."""
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from unidream.experiments import wm_significance_benchmark as runner
from unidream.experiments.wm_significance_jobs import make_jobs


def cfg_fixture(root):
    raw = yaml.safe_load(runner.DEFAULT_CONFIG.read_text())
    raw["output_dir"] = str(root / "study")
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return runner.load_config(path)


def records_fixture(include_aliases=True):
    records = []
    for group, model, task, formulation, head, family, metric, filter_name in sorted(runner.expected_slots()):
        if not include_aliases and model in ("linear", "lightgbm") and formulation == "joint":
            continue
        baseline = filter_name or ("fixed_chance" if family == "ba_chance" else "train_crash_prior" if family == "crash_Brier" else "causal_simple")
        row = dict(group=group, model=model, task=task, formulation=formulation, head=head, family=family,
            metric=metric, baseline=baseline, block_days=7, p_value=.001, improvement=.1)
        row["hypothesis_id"] = runner._hypothesis_id(row)
        records.append(row)
    return records


class SignificanceBenchmarkTests(unittest.TestCase):
    def test_exact_physical_jobs_and_global_families(self):
        cfg = runner.load_config()
        self.assertEqual(cfg["runtime"]["workers"], 12)
        self.assertEqual(len(make_jobs(cfg)), 640)
        self.assertEqual(len(runner.expected_slots()), 1056)
        self.assertEqual(Counter(row[5] for row in runner.expected_slots()), runner.FAMILIES)

    def test_config_rejects_small_monte_carlo_and_unseen_holdout_claim(self):
        for section, field, value in (("statistics", "bootstrap_replicates", 10000),
                                      (None, "independently_unused_holdout", True),
                                      (None, "test_selection_allowed", True)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = cfg_fixture(root)
                raw = yaml.safe_load(Path(cfg["config_path"]).read_text())
                (raw[section] if section else raw)[field] = value
                Path(cfg["config_path"]).write_text(yaml.safe_dump(raw))
                with self.assertRaisesRegex(ValueError, "fixed336"):
                    runner.load_config(cfg["config_path"])

    def test_joint_alias_expansion_restores_all1056_slots_without_new_models(self):
        source = records_fixture(include_aliases=False)
        records = runner.expand_control_aliases(source)
        self.assertEqual(len(source), 832)
        self.assertEqual(len(records), 1056)
        aliases = [row for row in records if row.get("control_alias")]
        self.assertEqual(len(aliases), 224)
        self.assertTrue(all(row["formulation"] == "joint" and row["shared_parameters"] is False for row in aliases))
        self.assertTrue(all(row["alias_source_formulation"] == row["head"] for row in aliases if row["family"] in ("primary", "crash_Brier")))
        self.assertEqual(len({row["hypothesis_id"] for row in records}), 1056)
        self.assertEqual(len(runner.adjust_families(records)), 1056)

    def test_fixed_families_count_unavailable_as_one_and_keep_direction(self):
        records = records_fixture()
        primary = next(row for row in records if row["family"] == "primary")
        primary["p_value"] = None
        another = next(row for row in records if row["family"] == "primary" and row is not primary)
        another["p_value"] = 1e-10
        another["improvement"] = -.1
        output = runner.adjust_families(records)
        by_id = {row["hypothesis_id"]: row for row in output}
        self.assertEqual(by_id[primary["hypothesis_id"]]["p_holm_primary448"], 1.)
        self.assertEqual(by_id[primary["hypothesis_id"]]["p_holm_global"], 1.)
        self.assertFalse(by_id[another["hypothesis_id"]]["significant_improvement_global"])
        self.assertIsNone(next(row for row in output if row["family"] == "ba_chance")["p_holm_primary448"])

    def test_missing_or_duplicate_global_hypothesis_is_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                rows = records_fixture()
                rows = rows + [rows[0]] if duplicate else rows[:-1]
                with self.assertRaisesRegex(ValueError, "1056"):
                    runner.adjust_families(rows)

    def test_sensitivities_and_IC_are_not_in_the_global_hypothesis_family(self):
        records = records_fixture()
        sensitivity = {**records[0], "block_days": 1}
        ic = {**records[0], "family": "ic_ci", "baseline": "none", "metric": "pearson_ic", "p_value": None}
        ic["hypothesis_id"] = runner._hypothesis_id(ic)
        output = runner.adjust_families([*records, sensitivity, ic])
        self.assertIsNone(output[-1]["p_holm_global"])
        self.assertIsNone(output[-2]["p_holm_global"])

    def test_registration_is_frozen_before_any_cache_or_Test_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = cfg_fixture(root)
            binary = root / "rank.dylib"
            binary.write_bytes(b"synthetic compiled rank helper")
            frozen = root / "old_selector.json"
            frozen.write_text('{"Test_used":false}')
            old = {"manifest": {str(frozen): runner._sha(frozen)}, "folds": []}
            with patch.object(runner, "_old_contract", return_value=old), \
                 patch.object(runner, "source_hashes", return_value={"frozen.py": "same"}), \
                 patch("unidream.experiments.wm_significance_statistics.compile_rank_backend", return_value=binary), \
                 patch.object(runner.subprocess, "run", return_value=SimpleNamespace(stdout="clang version synthetic\n")), \
                 patch("unidream.experiments.wm_significance_cache.build_cache", side_effect=AssertionError("Test cache opened during registration")):
                registration = runner.prepare(cfg)
                self.assertTrue(registration["historical_seen_test"])
                self.assertFalse(registration["independently_unused_holdout"])
                self.assertEqual(len(registration["jobs"]), 640)
                self.assertEqual(runner.verify(cfg), registration)
                with self.assertRaisesRegex(ValueError, "immutable"):
                    runner.prepare(cfg)
                frozen.write_text('{"Test_used":true}')
                with self.assertRaisesRegex(ValueError, "prerequisite artifact changed"):
                    runner.verify(cfg)

    def test_registered_code_and_binary_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = cfg_fixture(root)
            out = Path(cfg["output_dir"])
            out.mkdir()
            binary = root / "rank.dylib"
            binary.write_bytes(b"original")
            registration = {"config_sha256": runner._sha(cfg["config_path"]), "source_sha256": {"a": "b"},
                "rank_backend": {"path": str(binary), "binary_sha256": runner._sha(binary)}, "old_contract": {"manifest": {}}}
            runner.atomic_json(out / "preregistration.json", registration)
            with patch.object(runner, "source_hashes", return_value={"a": "changed"}):
                with self.assertRaisesRegex(ValueError, "source/config changed"):
                    runner.verify(cfg)
            binary.write_bytes(b"modified")
            with patch.object(runner, "source_hashes", return_value={"a": "b"}):
                with self.assertRaisesRegex(ValueError, "rank binary changed"):
                    runner.verify(cfg)

    def test_resume_rejects_a_job_from_different_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = cfg_fixture(root)
            out = Path(cfg["output_dir"])
            runner.atomic_json(out / "preregistration.json", {})
            runner.atomic_json(out / "prepared/cache_manifest.json", {})
            task = make_jobs()[0]
            payload = {"complete": True, "task": task, "registration_sha256": runner._sha(out / "preregistration.json"),
                "cache_manifest_sha256": "different", "records": [{"estimate": .1}]}
            runner.atomic_json(out / "jobs" / (task["job_id"] + ".json"), payload)
            with self.assertRaisesRegex(ValueError, "different immutable data"):
                runner._job_payload(cfg, task)
            self.assertIsNone(runner._job_payload(cfg, make_jobs()[1]))

    def test_campaign_lock_prevents_second_coordinator_and_recovers_after_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = "from unidream.experiments.wm_significance_benchmark import campaign_lock\nimport sys\nwith campaign_lock(sys.argv[1]):\n print('acquired')\n"
            with runner.campaign_lock(root):
                other = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True, text=True)
                self.assertNotEqual(other.returncode, 0)
                self.assertIn("holds the campaign lock", other.stderr)
            with runner.campaign_lock(root):
                self.assertEqual(json.loads((root / "coordinator.json").read_text())["lock"], "flock")


if __name__ == "__main__":
    unittest.main()
