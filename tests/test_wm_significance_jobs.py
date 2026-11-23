from __future__ import annotations

from collections import Counter
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments.wm_significance_jobs import (
    TASKS, _claim, _clean, calculate_job_records, make_jobs, run_job,
)


def fixture(directory):
    n = 16 * 96
    rng = np.random.default_rng(29)
    actual = rng.normal(size=(n, 7)) * .01
    actual[:, 1] = np.exp(rng.normal(size=n)) * .0001
    actual[:, 2] = np.abs(actual[:, 2])
    actual[:, 3] = rng.binomial(1, .2, n)
    actual[:, 4] = np.maximum(actual[:, 2] - .012, 0)
    actual[:, 6] = np.tanh(actual[:, 6] * 10)
    labels = np.tile((np.arange(n) % 3)[:, None], (1, 7))
    labels[:, 3] = actual[:, 3]
    scalar = np.tile(np.mean(actual, axis=0), (n, 1))
    scalar[:, 0] = 0
    scalar[:, 5:] += rng.normal(size=(n, 2)) * .001
    baseline = {"fold": np.repeat([0, 1], n // 2), "decision_id": np.arange(n),
        "index_ns": pd.date_range("2020-01-01", periods=n, freq="15min", tz="UTC").astype("int64").to_numpy(),
        "actual": actual, "labels": labels, "scalar": scalar, "classes": np.zeros((n, 7), dtype=int),
        "constant_crash_p": np.full(n, .2), "filter_variance": np.tile([.0001, .00012, .00015], (n, 1)),
        "filter_classes": np.tile([0, 1, 2], (n, 1))}
    forecast = actual.copy()
    forecast[:, 3] = .1 + .8 * actual[:, 3]
    model = {"classes": labels, "scalar": forecast, "crash_p": .1 + .8 * actual[:, 3]}
    cfg = {"output_dir": directory, "statistics": {"block_days": [1, 7, 28],
        "bootstrap_replicates": 13, "sensitivity_replicates": 7, "ic_replicates": 9, "seed": 17}}
    return cfg, baseline, model


def select_job(kind="main", task="return", formulation="regression", **extra):
    return next(job for job in make_jobs() if job["kind"] == kind and job["task"] == task
                and job["group"] == "A" and job["model"] == "direct" and job["formulation"] == formulation
                and all(job.get(key) == value for key, value in extra.items()))


class GridJobsTests(unittest.TestCase):
    def test_fixed_grid_has640_unique_jobs_and_no_fake_classical_joint_training(self):
        jobs = make_jobs()
        self.assertEqual(len(jobs), 640)
        self.assertEqual(len({job["job_id"] for job in jobs}), 640)
        self.assertEqual(Counter(job["kind"] for job in jobs), {
            "main": 280, "ic_aux": 144, "variance_ba": 120, "variance_qlike": 48, "crash_brier": 48})
        self.assertFalse(any(job["model"] in ("linear", "lightgbm") and job["formulation"] == "joint" for job in jobs))

    def test_regression_main_has_native_primary_and_explicit_chance_proxy_only(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, baselines, panel = fixture(directory)
            records = calculate_job_records(cfg, select_job(), baselines, panel)
        primary = [row for row in records if row["role"] == "primary"]
        chance = [row for row in records if row["family"] == "ba_chance"]
        self.assertEqual(len(primary), 3)
        self.assertEqual({row["metric"] for row in primary}, {"zero_oos_r2"})
        self.assertEqual(len(chance), 3)
        self.assertTrue(all(row["class_decision_is_scalar_proxy"] for row in chance))
        self.assertTrue(all(row["baseline_estimate"] == 1 / 3 for row in chance))
        self.assertTrue(all(row["baseline"] == "fixed_chance" for row in chance))
        self.assertEqual({row["replicates"] for row in records if row["block_days"] == 7}, {13})
        self.assertEqual({row["replicates"] for row in records if row["block_days"] != 7}, {7})

    def test_classification_ignores_unavailable_native_scalar_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, baselines, panel = fixture(directory)
            panel["scalar"][:] = np.nan
            rows = calculate_job_records(cfg, select_job(formulation="classification"), baselines, panel)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["metric"] for row in rows}, {"balanced_accuracy"})
        self.assertEqual(sum(row["role"] == "primary" for row in rows), 3)

    def test_joint_keeps_two_native_heads_and_single_chance_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, baselines, panel = fixture(directory)
            rows = calculate_job_records(cfg, select_job(formulation="joint"), baselines, panel)
        self.assertEqual(len(rows), 9)
        self.assertEqual({row["head"] for row in rows if row["role"] == "primary"}, {"classification", "regression"})
        selected = [row["hypothesis_id"] for row in rows if row["block_days"] == 7]
        self.assertEqual(len(selected), len(set(selected)))

    def test_ic_aux_is_confidence_interval_only_with_no_hidden_p_values(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, baselines, panel = fixture(directory)
            rows = calculate_job_records(cfg, select_job(kind="ic_aux"), baselines, panel)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["metric"] for row in rows}, {"pearson_ic", "spearman_ic"})
        self.assertTrue(all(row["p_value"] is None and row["improvement"] is None for row in rows))
        self.assertTrue(all(row["replicates"] == 9 for row in rows))
        self.assertTrue(all(row["family"] == "ic_ci" for row in rows))

    def test_native_spearman_is_not_repeated_in_ic_aux(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, baselines, panel = fixture(directory)
            rows = calculate_job_records(cfg, select_job(kind="ic_aux", task="advantage"), baselines, panel)
        self.assertEqual({row["metric"] for row in rows}, {"pearson_ic"})

    def test_variance_classification_uses_registered_filter_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, baselines, panel = fixture(directory)
            rows = calculate_job_records(cfg, select_job(kind="variance_ba", task="variance", filter="garch"), baselines, panel)
        self.assertEqual({row["family"] for row in rows}, {"variance_BA"})
        self.assertEqual({row["baseline"] for row in rows}, {"garch"})
        self.assertTrue(all(abs(row["baseline_estimate"] - 1 / 3) < 1e-14 for row in rows))

    def test_secondary_variance_qlike_uses_frozen_filter_points(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, baselines, panel = fixture(directory)
            rows = calculate_job_records(cfg, select_job(kind="variance_qlike", task="variance", filter="past_variance"), baselines, panel)
        self.assertEqual({row["family"] for row in rows}, {"variance_QLIKE"})
        self.assertTrue(all(row["improvement"] > 0 for row in rows))

    def test_crash_probability_head_and_point_head_are_distinct_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, baselines, panel = fixture(directory)
            native = calculate_job_records(cfg, select_job(kind="crash_brier", task="crash", formulation="joint", head="regression"), baselines, panel)
            probability = calculate_job_records(cfg, select_job(kind="crash_brier", task="crash", formulation="joint", head="classification"), baselines, panel)
        self.assertTrue(all(abs(row["estimate"] - .01) < 1e-12 for row in native + probability))
        self.assertNotEqual(native[1]["hypothesis_id"], probability[1]["hypothesis_id"])

    def test_strict_json_cleaner_and_live_claim(self):
        text = json.dumps(_clean({"a": np.float64(np.nan), "b": np.int64(2)}), allow_nan=False)
        self.assertEqual(json.loads(text), {"a": None, "b": 2})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim.lock"
            with _claim(path):
                with self.assertRaisesRegex(RuntimeError, "already claimed"):
                    with _claim(path):
                        pass
            self.assertTrue(path.exists())
            # The same persistent inode is immediately reusable on unlock.
            with _claim(path):
                self.assertEqual(json.loads(path.read_text())["pid"], os.getpid())

    def test_resume_verifies_prediction_cache_before_reusing_complete_job(self):
        from unidream.experiments import wm_significance_benchmark as runner
        with tempfile.TemporaryDirectory() as directory:
            cfg, baselines, panel = fixture(directory)
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            registration = root / "preregistration.json"
            registration.write_text('{"synthetic_test": true}\n')
            digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            baseline_path, panel_path = prepared / "baseline.npz", prepared / "A_direct_regression.npz"
            np.savez(baseline_path, **baselines)
            np.savez(panel_path, **panel)
            manifest = {"registration_sha256": digest(registration), "n_rows": len(baselines["fold"]),
                "task_order": list(TASKS), "filter_order": ["past_variance", "ewma", "garch"],
                "baseline_path": str(baseline_path), "baseline_sha256": digest(baseline_path),
                "panels": {"A_direct_regression": {"path": str(panel_path), "sha256": digest(panel_path)}}}
            (prepared / "cache_manifest.json").write_text(json.dumps(manifest))
            with patch.object(runner, "load_config", return_value=cfg), patch.object(runner, "verify", return_value={}):
                first = run_job("unused_synthetic_config", select_job())
                second = run_job("unused_synthetic_config", select_job())
                self.assertEqual(first, second)
                self.assertTrue(first["complete"])
                with baseline_path.open("ab") as stream:
                    stream.write(b"changed-cache-after-completion")
                with self.assertRaisesRegex(ValueError, "integrity"):
                    run_job("unused_synthetic_config", select_job())

    def test_stale_pid_metadata_is_ignored_without_a_kernel_holder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stale.lock"
            path.write_text(json.dumps({"pid": 2147483647}))
            with _claim(path):
                self.assertTrue(path.exists())
            self.assertTrue(path.exists())

    def test_reused_live_pid_metadata_never_prevents_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reused_pid.lock"
            # Simulate an old claim whose PID now names an unrelated live
            # process. A PID-existence guard would incorrectly reject this.
            path.write_text(json.dumps({"pid": os.getpid()}))
            before = path.stat().st_ino
            with _claim(path):
                self.assertEqual(json.loads(path.read_text())["pid"], os.getpid())
            self.assertEqual(path.stat().st_ino, before)


if __name__ == "__main__":
    unittest.main()
