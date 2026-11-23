"""Coordinator contracts that prevent partial-grid Test and stale selections."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from unidream.experiments import wm_seed_robustness_benchmark as benchmark
from unidream.experiments.wm_policy_ablation_policy import sha, write_json


class IndependentSeedGridTests(unittest.TestCase):
    def config(self, root):
        return {"output_dir": str(root), "release_output_dir": str(root / "old"),
                "new_seeds": benchmark.SEEDS[3:], "reused_seeds": benchmark.SEEDS[:3],
                "seeds": benchmark.SEEDS, "folds": list(range(6)), "conditions": benchmark.CONDITIONS,
                "condition_definitions": {"C_floor_gradual": {"arm": "C", "condition": "floor_gradual"},
                    "A_floor_gradual": {"arm": "A", "condition": "floor_gradual"},
                    "C_control": {"arm": "C", "condition": "control"}}}

    def test_new_seed_changes_every_stage_not_only_continuation(self):
        cfg = self.config(Path("/tmp/study"))
        tasks = benchmark.jobs(cfg)
        counts = {kind: sum(t["kind"] == kind for t in tasks) for kind in ("wm", "policy", "continuation")}
        self.assertEqual(counts, {"wm": 42, "policy": 84, "continuation": 126})
        all_ids = {t["job"] for t in tasks}
        self.assertEqual(len(all_ids), 252)
        for seed in cfg["new_seeds"]:
            for fold in cfg["folds"]:
                part = [t for t in tasks if t["seed"] == seed and t["fold"] == fold]
                self.assertEqual(len(part), 6)
                self.assertTrue(all(set(t["dependencies"]) <= all_ids for t in part))
                c = next(t for t in part if t["kind"] == "policy" and t["arm"] == "C")
                self.assertEqual(c["dependencies"], [f"wm_fold_{fold:02d}_s{seed}"])
        reused = benchmark.reused_jobs(cfg)
        self.assertEqual(len(reused), 54)
        self.assertEqual({t["seed"] for t in reused}, {7, 19, 42})

    def test_partial_test_barrier_or_changed_choices_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = self.config(root)
            tasks = benchmark.jobs(cfg)
            write_json(root / "preregistration.json", {"fixture": True})
            write_json(root / "validation_choices.json", {"fixture": True})
            write_json(root / "prepared/cache_manifest.json", {"fixture": True})
            barrier = {"complete": True, "registration_sha256": sha(root / "preregistration.json"),
                "test_selection_allowed": False, "frozen_fit_sha256": {},
                "validation_choices_sha256": sha(root / "validation_choices.json"),
                "cache_manifest_sha256": sha(root / "prepared/cache_manifest.json")}
            write_json(root / "test_barrier.json", barrier)
            with patch.object(benchmark, "verify", return_value={"jobs": tasks}):
                with self.assertRaisesRegex(ValueError, "all new fits"):
                    benchmark.verify_test_barrier(cfg)

    def test_barrier_detects_changed_fit_even_with_same_complete_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = self.config(root)
            task = {"job": "one", "fit_path": "fit.json"}
            write_json(root / "preregistration.json", {"fixture": True})
            write_json(root / "fit.json", {"complete": True, "selected_update": 125})
            write_json(root / "validation_choices.json", {"fixture": True})
            write_json(root / "prepared/cache_manifest.json", {"fixture": True})
            frozen = sha(root / "fit.json")
            write_json(root / "test_barrier.json", {"complete": True,
                "registration_sha256": sha(root / "preregistration.json"), "test_selection_allowed": False,
                "frozen_fit_sha256": {"fit.json": frozen},
                "validation_choices_sha256": sha(root / "validation_choices.json"),
                "cache_manifest_sha256": sha(root / "prepared/cache_manifest.json")})
            write_json(root / "fit.json", {"complete": True, "selected_update": 200})
            with patch.object(benchmark, "verify", return_value={"jobs": [task]}):
                with self.assertRaisesRegex(ValueError, "fit changed"):
                    benchmark.verify_test_barrier(cfg)

    def test_freeze_checks_all_tasks_before_writing_test_barrier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = self.config(root)
            task = {"job": "missing", "fit_path": "missing/fit.json"}
            with patch.object(benchmark, "verify", return_value={"jobs": [task]}):
                with self.assertRaises(FileNotFoundError):
                    benchmark.freeze_test_barrier(cfg)
            self.assertFalse((root / "test_barrier.json").exists())

    def test_failed_split_audit_cannot_start_any_training(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = self.config(Path(temp))
            cfg["source_config"] = "source-config"
            with (patch.object(benchmark, "register", return_value={"jobs": []}),
                  patch.object(benchmark.parent, "load_config", return_value={}),
                  patch("unidream.experiments.wm_seed_robustness_audit.audit_source", return_value={"passed": False, "leakage_found": True}),
                  patch.object(benchmark, "run_pool") as pool):
                with self.assertRaisesRegex(ValueError, "audit failed"):
                    benchmark.run(cfg)
                pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
