"""Post-hoc runner integration checks using synthetic executed ledgers only."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments import wm_inference_robust_benchmark as benchmark
from unidream.experiments.wm_policy_ablation_evaluation import (
    accounting, bounded_requests, execute_positions, reset_mask,
)
from unidream.experiments.wm_policy_bh_inference import infer_bh


def fixture(root, days=20):
    n = days * 96
    timestamp = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    timestamp += pd.to_timedelta((np.arange(n) >= n // 2) * 86400, unit="s")
    returns = np.random.default_rng(13).normal(.00002, .001, n)
    folds = np.zeros(n, dtype=int)
    original_cfg = {"accounting": accounting()}
    bh = execute_positions(returns, np.ones(n), timestamp, folds, original_cfg)
    bh["decision_id"] = np.arange(n)
    bh["baseline"] = "bh"
    paths = []
    for seed in (7, 19, 42):
        for stage in ("bc", "rl"):
            for arm in "ABCD":
                targets = np.full(n, .8 + .03 * (ord(arm) - ord("A")))
                requests = bounded_requests(targets, timestamp, folds, original_cfg)
                path = execute_positions(returns, requests, timestamp, folds, original_cfg)
                path["decision_id"] = np.arange(n)
                paths.append(path.assign(seed=seed, arm=arm, stage=stage))
    policy = pd.concat(paths, ignore_index=True)
    final = root / "original" / "final"
    final.mkdir(parents=True)
    policy.to_parquet(final / "executed_policy_ledger.parquet", index=False)
    bh.to_parquet(final / "executed_baseline_ledger.parquet", index=False)
    cfg = {"source_config": "synthetic", "previous_inference_config": "synthetic",
        "output_dir": str(root / "new"), "folds": [0], "seeds": [7, 19, 42], "arms": list("ABCD"),
        "stages": ["bc", "rl"], "benchmarks": ["bh", "fixed50"],
        "condition_names": [f"{stage}_{arm}" for stage in ("bc", "rl") for arm in "ABCD"],
        "accounting": {"annual_bars": 35040}, "statistics": {"expected_market_rows": n,
        "bootstrap_replicates": 199, "block_days": [1, 7, 28], "seed": 17}}
    registration = {"sources": {"source_root": str(root / "original"), "prior_bh_output_dir": str(root / "prior")}}
    return policy, bh, cfg, original_cfg, registration


class RobustBenchmarkTests(unittest.TestCase):
    def test_real_execution_panel_counts_three_seeds_and_gap_reset_half_control(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy, bh, cfg, original_cfg, registration = fixture(root)
            source = root / "original" / "final" / "executed_policy_ledger.parquet"
            before = benchmark.sha(source)
            with patch("unidream.experiments.wm_policy_ablation_benchmark.load_config", return_value=original_cfg):
                panel, descriptions, folds, controls, _, audit = benchmark.build_panel(cfg, registration)
            self.assertEqual(panel["sum_bh"].shape, (20, 8, 3))
            self.assertEqual(panel["counts"].sum(), len(bh))
            self.assertEqual(audit["panel"]["n_unique_market_rows"], len(bh))
            self.assertEqual(audit["panel"]["n_training_seeds"], 3)
            self.assertEqual(len(descriptions), 48)
            self.assertEqual(len(folds), 48)
            self.assertEqual(audit["panel"]["n_segments"], 2)
            resets = reset_mask(bh.timestamp, bh.fold)
            half = controls["fixed50"]
            np.testing.assert_array_equal(half.reset, resets)
            np.testing.assert_allclose(half.executed_position[resets], 1)
            np.testing.assert_allclose(half.cost_fraction[resets], 0)
            self.assertEqual(half.executed_position.min(), .5)
            for j, condition in enumerate(cfg["condition_names"]):
                stage, arm = condition.split("_")
                for s, seed in enumerate(cfg["seeds"]):
                    part = policy.loc[policy.stage.eq(stage) & policy.arm.eq(arm) & policy.seed.eq(seed)]
                    self.assertAlmostEqual(panel["sum_bh"][:, j, s].sum(),
                        (part.net_log_return.to_numpy() - bh.net_log_return.to_numpy()).sum())
            self.assertEqual(benchmark.sha(source), before)

    def test_panel_fails_closed_for_cohort_or_fee_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy, _, cfg, original_cfg, registration = fixture(root, days=8)
            source = root / "original/final/executed_policy_ledger.parquet"
            for column in ("bar_log_return", "cost_fraction", "net_log_return"):
                wrong = policy.copy(); wrong.loc[0, column] += .01
                wrong.to_parquet(source, index=False)
                with patch("unidream.experiments.wm_policy_ablation_benchmark.load_config", return_value=original_cfg):
                    with self.subTest(column=column), self.assertRaises(ValueError):
                        benchmark.build_panel(cfg, registration)

    def test_original_legacy_inference_roundtrip_is_replayed_without_method_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy, bh, cfg, _, registration = fixture(root)
            prior = root / "prior"; prior.mkdir()
            before = infer_bh(policy, bh, cfg)
            before.to_csv(prior / "inference.csv", index=False)
            with patch("unidream.experiments.wm_policy_bh_benchmark.load_config", return_value=cfg):
                joined, audit = benchmark._original_replay(policy, bh, cfg, registration)
            self.assertEqual(len(joined), 24)
            self.assertLessEqual(audit["max_absolute_difference"], 1e-12)
            self.assertTrue(audit["explicit_centering_present"])
            before.loc[0, "p_value"] = .12345
            before.to_csv(prior / "inference.csv", index=False)
            with patch("unidream.experiments.wm_policy_bh_benchmark.load_config", return_value=cfg):
                with self.assertRaisesRegex(ValueError, "cannot be reproduced"):
                    benchmark._original_replay(policy, bh, cfg, registration)

    def test_register_before_calculate_and_new_namespace_preserves_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frozen = root / "old_result.csv"; frozen.write_bytes(b"immutable original outcomes\n")
            frozen_sha = benchmark.sha(frozen)
            out = root / "new_result"
            cfg = {"output_dir": str(out)}
            bound = {"immutable_source": frozen_sha}
            def calculate(given, registration):
                self.assertTrue((out / "posthoc_registration.json").exists())
                self.assertEqual(registration["sources"], bound)
                final = out / "final"; final.mkdir()
                (final / "synthetic.csv").write_text("synthetic_only\n1\n")
                (out / "prepared_panel.npz").write_bytes(b"synthetic only")
                return final
            report = types.SimpleNamespace(write_report=lambda *args: {"synthetic_only": True})
            with patch.object(benchmark, "sources", return_value=bound), \
                    patch.object(benchmark, "source_hashes", return_value={"new.py": "abc"}), \
                    patch.object(benchmark.importlib.metadata, "version", return_value="test"), \
                    patch.object(benchmark, "calculate", side_effect=calculate) as compute, \
                    patch.dict("sys.modules", {"unidream.experiments.wm_inference_robust_report": report}):
                done = benchmark.run(cfg)
                self.assertTrue(done["complete"])
                self.assertEqual(benchmark.run(cfg), done)
                self.assertEqual(compute.call_count, 1)
            self.assertEqual(benchmark.sha(frozen), frozen_sha)
            self.assertFalse((root / "posthoc_registration.json").exists())

    def test_registration_rejects_modified_sources_config_and_completed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = {"output_dir": temp}
            with patch.object(benchmark, "sources", return_value={"old": "a"}), \
                    patch.object(benchmark, "source_hashes", return_value={"new": "b"}), \
                    patch.object(benchmark.importlib.metadata, "version", return_value="test"):
                benchmark.register(cfg)
                with self.assertRaisesRegex(ValueError, "changed"):
                    benchmark.register({**cfg, "altered_test_selection": True})
                with patch.object(benchmark, "source_hashes", return_value={"new": "changed"}):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        benchmark.register(cfg)
                with patch.object(benchmark, "sources", return_value={"old": "changed"}):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        benchmark.register(cfg)

    def test_sensitivity_holm_keeps_all_candidates_without_selecting_best_method(self):
        frame = pd.DataFrame({"benchmark": ["bh"] * 4,
            "method": ["legacy", "legacy", "circular", "circular"],
            "block": [7] * 4, "p_value": [.01, .04, .001, np.nan]})
        benchmark._holm_column(frame, ["benchmark", "method", "block"], "q")
        np.testing.assert_allclose(frame.q, [.02, .04, .002, 1.])
        self.assertEqual(len(frame), 4)

    def test_source_hash_namespace_excludes_new_writes_to_frozen_old_globs(self):
        cfg = benchmark.load_config()
        sources = benchmark.source_hashes(cfg)
        self.assertIn("unidream/experiments/wm_inference_robust_benchmark.py", sources)
        self.assertIn("tests/test_wm_inference_robust_benchmark.py", sources)
        self.assertIn("unidream/experiments/wm_policy_bh_inference.py", sources)
        self.assertNotIn("unidream/experiments/wm_policy_ablation_policy.py", sources)
        self.assertTrue(all(len(value) == 64 for value in sources.values()))

    def test_ablation_twelve_contrasts_cancel_identical_benchmark(self):
        rng = np.random.default_rng(31)
        names = [f"{stage}_{arm}" for stage in ("bc", "rl") for arm in "ABCD"]
        policy = rng.normal(size=(40, 8, 3))
        bh = rng.normal(size=(40, 1, 1))
        actual, labels = benchmark.ablation_panel(policy - bh, names)
        direct, direct_labels = benchmark.ablation_panel(policy, names)
        self.assertEqual(actual.shape, (40, 12, 3))
        self.assertEqual(labels, direct_labels)
        np.testing.assert_allclose(actual, direct, atol=1e-14)
        for i, label in enumerate(labels):
            left, right = label.split("_minus_")
            np.testing.assert_allclose(actual[:, i], policy[:, names.index(left)] - policy[:, names.index(right)], atol=1e-14)

    def test_calculate_and_report_end_to_end_on_synthetic_panel(self):
        from unidream.experiments.wm_inference_robust_report import write_report
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy, bh, toy, original_cfg, registration = fixture(root, days=40)
            with patch("unidream.experiments.wm_policy_ablation_benchmark.load_config", return_value=original_cfg):
                prepared = benchmark.build_panel(toy, registration)
            cfg = benchmark.load_config()
            cfg.update({key: toy[key] for key in ("output_dir", "folds", "seeds", "condition_names")})
            cfg["statistics"].update(bootstrap_replicates=199, expected_market_rows=len(bh))
            # Six geometric-block calibration segments each need >7 days.
            cfg["calibration"].update(outer_replicates=2, inner_replicates=19, days=48, burnin=30,
                                      segment_layouts=["single", "six_equal"])
            out = Path(cfg["output_dir"]); out.mkdir()
            benchmark.write_json(out / "posthoc_registration.json", {"config": cfg, "synthetic_only": True})
            prior = root / "prior"; prior.mkdir()
            previous = infer_bh(policy, bh, toy)
            previous.to_csv(prior / "inference.csv", index=False)
            old12 = pd.DataFrame({"comparison": [f"old_{i}" for i in range(12)],
                                  "p_value": np.linspace(.1, .9, 12), "primary_inference": True})
            old12.to_csv(root / "original/final/paired_inference.csv", index=False)
            replay_audit = {"max_absolute_difference": 0., "synthetic_only": True}
            with patch.object(benchmark, "build_panel", return_value=prepared), \
                    patch.object(benchmark, "_original_replay", return_value=(previous, replay_audit)), \
                    patch.object(benchmark, "_automatic_blocks", return_value=pd.DataFrame({"synthetic_only": [True]})):
                final = benchmark.calculate(cfg, registration)
            report = write_report(out, cfg)
            self.assertTrue(report)
            self.assertTrue((final / "index.html").exists())
            self.assertTrue((final / "primary_forest.png").exists())
            self.assertTrue((final / "primary_forest.svg").exists())
            summary = pd.read_csv(final / "summary.csv")
            ablation = pd.read_csv(final / "ablation_primary.csv")
            expanded = pd.read_csv(final / "expanded_family20.csv")
            self.assertEqual(len(summary), 16)
            self.assertEqual(len(ablation), 12)
            self.assertEqual(len(expanded), 20)
            self.assertEqual(len(pd.read_csv(final / "calibration.csv")), 32)
            self.assertTrue(summary.loc[summary.benchmark.eq("bh"), "p_holm20_prior12_sensitivity"].notna().all())
            self.assertTrue(pd.read_csv(final / "bootstrap_inference.csv").p_holm_all_bootstrap_choices.between(0, 1).all())
            self.assertTrue((final / "power.csv").exists())
            self.assertTrue((final / "required_years.csv").exists())


if __name__ == "__main__":
    unittest.main()
