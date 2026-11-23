"""Toy-ledger tests for the isolated post-hoc B&H analysis; no real Test reads."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from unidream.experiments import wm_policy_bh_benchmark as study
from unidream.experiments import wm_policy_bh_inference as inference_module
from unidream.experiments.wm_significance_statistics import holm_fixed_family


class PolicyBHBenchmarkTests(unittest.TestCase):
    def setUp(self):
        # Keep config inside ROOT so production source_hashes can be exercised.
        self.temp = tempfile.TemporaryDirectory(dir=study.ROOT, prefix=".test-policy-bh-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        template = yaml.safe_load(study.DEFAULT_CONFIG.read_text())
        template["output_dir"] = str(self.root / "new-output")
        template["source_config"] = str(self.root / "source.yaml")
        self.config_path = self.root / "analysis.yaml"
        self.config_path.write_text(yaml.safe_dump(template))
        self.cfg = study.load_config(self.config_path)
        self.source = self.root / "original"
        self.sources = {"source_root": str(self.source), "source_artifact_sha256": {"old-results.csv": "unchanged"},
            "expected_market_rows": 4}
        times = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
        bar = np.array([.005, -.002, .003, .001])
        self.bh = pd.DataFrame({"baseline": "bh", "fold": 0, "decision_id": np.arange(4) + 100,
            "timestamp": times, "bar_log_return": bar, "executed_position": 1.,
            "cost_fraction": 0., "borrow_fraction": 0., "net_log_return": bar})
        panels = []
        for stage in ("bc", "rl"):
            for arm in "ABCD":
                for seed, shift in ((7, -.01), (19, .002), (42, .018)):
                    panel = self.bh.drop(columns="baseline").copy()
                    panel["stage"], panel["arm"], panel["seed"] = stage, arm, seed
                    panel["net_log_return"] += shift
                    panel["executed_position"] = .9
                    panels.append(panel)
        self.policies = pd.concat(panels, ignore_index=True)
        rows = []
        for block in (1, 7, 28):
            for stage in ("bc", "rl"):
                for arm in "ABCD":
                    rows.append({"stage": stage, "arm": arm, "comparison": f"{stage}_{arm}_minus_bh",
                        "block_days": block, "primary_inference": block == 7, "p_value": .02,
                        "p_holm_8": .16 if block == 7 else np.nan, "estimate": .1,
                        "ci_low": -.1, "ci_high": .3, "inference_available": True,
                        "significant_improvement": False})
        self.inference = pd.DataFrame(rows)
        self.original = pd.DataFrame({"comparison": [f"old_{i}" for i in range(12)],
            "primary_inference": True, "p_value": np.linspace(.001, .012, 12)})

    def mocked_run(self, infer=None, sources=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(study, "original_sources", side_effect=sources,
            return_value=self.sources if sources is None else None))
        stack.enter_context(patch.object(study, "source_hashes", return_value={"new-source.py": "fixed"}))
        stack.enter_context(patch.object(study.pd, "read_parquet", side_effect=lambda path:
            self.policies.copy() if Path(path).name == "executed_policy_ledger.parquet" else self.bh.copy()))
        stack.enter_context(patch.object(study.pd, "read_csv", return_value=self.original.copy()))
        mocked = stack.enter_context(patch.object(inference_module, "infer_bh",
            side_effect=infer, return_value=self.inference.copy() if infer is None else None))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return mocked

    def test_registration_exists_before_any_new_inference(self):
        observed = []
        def infer(*args):
            path = Path(self.cfg["output_dir"]) / "posthoc_registration.json"
            self.assertTrue(path.exists())
            registration = json.loads(path.read_text())
            self.assertTrue(registration["post_hoc"])
            self.assertTrue(registration["original_test_results_previously_seen"])
            self.assertFalse(registration["independently_unused_holdout"])
            observed.append(path.read_bytes())
            return self.inference.copy()
        mocked = self.mocked_run(infer=infer)
        result = study.run(self.cfg)
        self.assertTrue(result["complete"])
        self.assertEqual(result["n_primary_comparisons"], 8)
        self.assertEqual(result["n_total_inference_rows"], 24)
        self.assertEqual(len(observed), 1)
        mocked.assert_called_once()

    def test_existing_registration_rejects_configuration_or_source_mutation(self):
        with patch.object(study, "original_sources", return_value=self.sources), \
                patch.object(study, "source_hashes", return_value={"new": "one"}):
            first = study.register(self.cfg)
            self.assertEqual(study.register(self.cfg), first)
            changed = copy.deepcopy(self.cfg)
            changed["statistics"]["seed"] += 1
            with self.assertRaisesRegex(ValueError, "frozen BH analysis"):
                study.register(changed)
        with patch.object(study, "original_sources", return_value=self.sources), \
                patch.object(study, "source_hashes", return_value={"new": "two"}):
            with self.assertRaisesRegex(ValueError, "frozen BH analysis"):
                study.register(self.cfg)

    def test_frozen_original_namespace_and_source_glob_are_separate(self):
        from unidream.experiments import wm_policy_ablation_benchmark as parent
        new = study.source_hashes(self.cfg)
        self.assertTrue(any("wm_policy_bh_benchmark.py" in name for name in new))
        self.assertTrue(any("test_wm_policy_bh_benchmark.py" in name for name in new))
        self.assertFalse(any("wm_policy_ablation_benchmark.py" in name for name in new))
        self.assertFalse(any("test_wm_policy_ablation_" in name for name in new))
        # Calling the original source manifest hashes source/config only.
        original_cfg = {"config_path": str(self.config_path)}
        old = parent.source_hashes(original_cfg)
        self.assertTrue(any("wm_policy_ablation_benchmark.py" in name for name in old))
        self.assertFalse(any("wm_policy_bh_" in name for name in old))

    def test_bh_is_exact_actual_one_x_matching_market_return(self):
        result = study._validate_bh(self.bh.sample(frac=1, random_state=4))
        np.testing.assert_array_equal(result.decision_id, self.bh.decision_id)
        for column in ("executed_position", "cost_fraction", "borrow_fraction", "net_log_return"):
            bad = self.bh.copy()
            bad.loc[0, column] += .001
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "actual1x"):
                study._validate_bh(bad)
        with self.assertRaisesRegex(ValueError, "actual1x"):
            study._validate_bh(pd.concat([self.bh, self.bh.iloc[[0]]]))

    def test_holm20_retains_twelve_original_and_eight_new_hypotheses(self):
        result, family = study.extended_correction(self.inference, self.original)
        self.assertEqual(len(family), 20)
        self.assertEqual(family.groupby("family").size().to_dict(), {"original": 12, "bh": 8})
        primary = result.loc[result.primary_inference]
        expected = holm_fixed_family({f"o{i}": float(p) for i, p in enumerate(self.original.p_value)} |
            {f"n{i}": .02 for i in range(8)}, family_size=20)
        np.testing.assert_allclose(primary.p_holm_20_sensitivity, [expected[f"n{i}"] for i in range(8)])
        self.assertTrue(result.loc[~result.primary_inference, "p_holm_20_sensitivity"].isna().all())
        self.assertFalse(primary.significant_improvement_holm20.any())
        with self.assertRaisesRegex(ValueError, "exactly all8"):
            study.extended_correction(self.inference, self.original.iloc[:-1])

    def test_missing_original_p_is_retained_as_one_not_dropped(self):
        original = self.original.copy()
        original.loc[0, "p_value"] = np.nan
        _, family = study.extended_correction(self.inference, original)
        row = family.loc[family.family.eq("original") & family.comparison.eq("old_0")]
        self.assertEqual(float(row.p_value.iloc[0]), 1.)
        self.assertEqual(len(family), 20)

    def test_descriptive_arithmetic_seed_returns_are_distinct_from_log_growth(self):
        seeds, folds, aggregate = study.descriptive_tables(self.policies, self.bh, self.cfg)
        example = seeds.loc[seeds.stage.eq("bc") & seeds.arm.eq("A")]
        summary = aggregate.loc[aggregate.stage.eq("bc") & aggregate.arm.eq("A")].iloc[0]
        arithmetic = example.policy_net_total_return.mean()
        geometric = np.expm1(np.log1p(example.policy_net_total_return).mean())
        self.assertAlmostEqual(summary.policy_net_total_return, arithmetic)
        self.assertGreater(abs(arithmetic-geometric), .0001)
        self.assertAlmostEqual(summary.cumulative_return_difference_descriptive,
            arithmetic-np.expm1(self.bh.net_log_return.sum()))
        np.testing.assert_allclose(example.annualized_log_growth_difference,
            [((self.policies.loc[(self.policies.stage == "bc") & (self.policies.arm == "A") &
                (self.policies.seed == seed), "net_log_return"].mean() - self.bh.net_log_return.mean()) *
                self.cfg["accounting"]["annual_bars"]) for seed in example.seed])
        self.assertEqual(len(folds), 24)

    def test_input_change_during_inference_prevents_completion(self):
        changed = copy.deepcopy(self.sources)
        changed["source_artifact_sha256"]["old-results.csv"] = "changed"
        self.mocked_run(sources=[self.sources, changed])
        with self.assertRaisesRegex(ValueError, "changed during"):
            study.run(self.cfg)
        self.assertFalse((Path(self.cfg["output_dir"]) / "completion.json").exists())

    def test_completed_analysis_is_cached_immutable_and_never_reinferred(self):
        mocked = self.mocked_run()
        first = study.run(self.cfg)
        completion = Path(self.cfg["output_dir"]) / "completion.json"
        before = completion.read_bytes()
        second = study.run(self.cfg)
        self.assertEqual(first, second)
        self.assertEqual(before, completion.read_bytes())
        mocked.assert_called_once()
        (Path(self.cfg["output_dir"]) / "summary.csv").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "completed BH output changed"):
            study.run(self.cfg)
        self.assertEqual(before, completion.read_bytes())

    def test_registration_failure_prevents_inference_or_input_loading(self):
        with patch.object(study, "register", side_effect=ValueError("foreign source")), \
                patch.object(study.pd, "read_parquet") as load, \
                patch.object(inference_module, "infer_bh") as infer:
            with self.assertRaisesRegex(ValueError, "foreign source"):
                study.run(self.cfg)
            load.assert_not_called()
            infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
