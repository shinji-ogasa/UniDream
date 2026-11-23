"""Synthetic, causal-accounting checks for the continued-RL report contract."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments import wm_rl_release_evaluation as evaluation
# Load the real guard/import graph before temporary sys.modules stubs. Reloading
# a removed torch module after a patch.dict context is unsafe for its C bindings.
from unidream.experiments import wm_rl_release_benchmark as release_benchmark
from unidream.experiments import wm_policy_ablation_benchmark as original_benchmark
from unidream.experiments.wm_policy_ablation_evaluation import (
    accounting, bounded_requests, execute_positions, reset_mask,
)


def fixture(days=40, seeds=(7, 19, 42)):
    n = days * 96
    time = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    time += pd.to_timedelta((np.arange(n) >= n // 2) * 86400, unit="s")
    returns = np.random.default_rng(21).normal(.00003, .001, n)
    configs = {condition: {"accounting": accounting()} for condition in evaluation.CONDITIONS}
    for condition in ("floor_abrupt", "floor_gradual"):
        configs[condition]["accounting"]["position_min"] = 0.
    cfg = {"arms": list(evaluation.ARMS), "conditions": list(evaluation.CONDITIONS), "folds": [0],
        "seeds": list(seeds), "accounting": {"annual_bars": 35040}, "statistics": {
        "expected_market_rows": n, "bootstrap_replicates": 99, "block_days": [7, 14, 30],
        "methods": ["circular"], "hac_lags": [6, 13, 29], "seed": 17}}
    frames, original, targets = [], [], {}
    for arm in evaluation.ARMS:
        for seed in seeds:
            request = bounded_requests(np.full(n, .8), time, np.zeros(n), configs["control"])
            path = execute_positions(returns, request, time, np.zeros(n), configs["control"])
            path["decision_id"] = np.arange(n)
            original.append(path.assign(arm=arm, seed=seed, stage="rl"))
            for condition in evaluation.CONDITIONS:
                target = .25 if condition.startswith("floor_") else .7
                target += (.03 if arm == "C" else 0) + (.01 if condition.endswith("gradual") else 0)
                request = bounded_requests(np.full(n, target), time, np.zeros(n), configs[condition])
                path = execute_positions(returns, request, time, np.zeros(n), configs[condition])
                path["decision_id"] = np.arange(n)
                frames.append(path.assign(arm=arm, seed=seed, condition=condition))
                # Chosen independently of Test exposure, with a distinct target
                # that makes accidental Test matching observable in the test.
                targets[(0, seed, arm, condition)] = target + .04
    return pd.concat(frames, ignore_index=True), cfg, configs, targets, pd.concat(original, ignore_index=True)


class ReleaseEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.cfg, cls.configs, cls.targets, cls.original = fixture()
        cls.result = evaluation.evaluate_ledgers(cls.frame, cls.cfg, cls.configs, cls.targets,
                                                 known_original=cls.original)

    def test_full_registered_grid_and_same_market_counts_not_seed_multiplication(self):
        result = self.result
        self.assertEqual(len(result["aggregate_metrics"]), 10)
        self.assertEqual(len(result["pooled_metrics"]), 30)
        self.assertEqual(len(result["fold_metrics"]), 30)
        self.assertEqual(len(result["primary_inference"]), 27)
        self.assertEqual(len(result["hac_inference"]), 81)
        self.assertEqual(result["daily_panel"]["sums"].shape, (40, 27, 3))
        self.assertEqual(result["audit"]["n_unique_market_rows"], 40 * 96)
        self.assertEqual(result["audit"]["n_training_seeds"], 3)
        self.assertEqual(result["audit"]["n_reset_segments"], 2)
        self.assertFalse(result["audit"]["independently_unused_holdout"])
        self.assertFalse(result["audit"]["seed_market_independence"])

    def test_all_12_5_10_hypotheses_and_17_27_corrections_retained(self):
        rows = self.result["primary_inference"]
        self.assertEqual(rows.groupby("family").size().to_dict(), {
            "release_primary": 12, "wm_secondary": 5, "risk_matched_secondary": 10})
        self.assertTrue(rows.p_holm_family.between(0, 1).all())
        self.assertTrue(rows.p_holm_all27.ge(rows.p_holm_family).all())
        self.assertTrue(rows.loc[~rows.family.eq("risk_matched_secondary"), "p_holm_all17"].notna().all())
        self.assertTrue(rows.loc[rows.family.eq("risk_matched_secondary"), "p_holm_all17"].isna().all())
        for row in rows.itertuples():
            expected = row.p_holm_family < .05
            self.assertEqual(row.significant_difference, expected)
        # 30-day intervals are unavailable: both synthetic segments have20days.
        sensitivity = self.result["inference"].loc[lambda f: f.block_days.eq(30)]
        self.assertFalse(sensitivity.inference_available.any())
        self.assertTrue(sensitivity.p_holm_family.eq(1).all())

    def test_floor_cash_metrics_and_static_zero_pay_the_same_execution_costs(self):
        pooled = self.result["pooled_metrics"]
        self.assertTrue(pooled.loc[pooled.condition.str.startswith("floor_"), "final_position_min"].eq(0).all())
        self.assertTrue(pooled.loc[~pooled.condition.str.startswith("floor_"), "final_position_min"].eq(.5).all())
        static = self.result["static_ledger"]
        cash = static.loc[static.baseline.eq("static0")]
        resets = reset_mask(cash.timestamp, cash.fold)
        np.testing.assert_allclose(cash.executed_position.to_numpy()[resets], 1)
        np.testing.assert_allclose(cash.cost_fraction.to_numpy()[resets], 0)
        self.assertEqual(cash.executed_position.min(), 0)
        self.assertGreater(cash.cost_fraction.sum(), 0)
        self.assertGreater(self.result["static_metrics"].loc[lambda f: f.baseline.eq("static0"), "near_cash_fraction"].iloc[0], .9)
        bh = static.loc[static.baseline.eq("bh")]
        np.testing.assert_allclose(bh.net_log_return, bh.bar_log_return, atol=1e-15)

    def test_val_exposure_matching_uses_fixed_provided_target_and_paired_market(self):
        control = self.result["matched_ledger"]
        path = control.loc[control.arm.eq("A") & control.condition.eq("control") & control.seed.eq(7)]
        request = path.requested_position.to_numpy()
        np.testing.assert_allclose(request[20:100], self.targets[(0, 7, "A", "control")])
        policy = self.frame.loc[self.frame.arm.eq("A") & self.frame.condition.eq("control") & self.frame.seed.eq(7)]
        self.assertGreater(abs(path.executed_position.mean() - policy.executed_position.mean()), .03)
        np.testing.assert_array_equal(path.decision_id, policy.decision_id)
        np.testing.assert_allclose(path.bar_log_return, policy.bar_log_return)
        self.assertEqual(len(self.result["known_original_metrics"]), 6)

    def test_daily_point_estimates_are_exactly_paired_net_log_differences(self):
        row = self.result["primary_inference"].loc[lambda f: f.condition.eq("A_floor_abrupt_minus_control")].iloc[0]
        def choose(condition):
            return self.frame.loc[self.frame.arm.eq("A") & self.frame.condition.eq(condition) & self.frame.seed.eq(7)]
        direct = (choose("floor_abrupt").net_log_return.to_numpy() - choose("control").net_log_return.to_numpy()).mean() * 35040
        self.assertAlmostEqual(row.estimate, direct, places=12)
        self.assertTrue(self.result["inference"].null_data_vs_subtract_estimate_max_error.lt(1e-12).all())

    def test_cohort_accounting_missing_seed_and_future_fitted_targets_fail_closed(self):
        variants = []
        for column in ("bar_log_return", "cost_fraction", "net_log_return"):
            wrong = self.frame.copy(); wrong.loc[0, column] += .01; variants.append(wrong)
        variants.append(self.frame.iloc[1:])
        variants.append(self.frame.loc[~(self.frame.arm.eq("C") & self.frame.condition.eq("control"))])
        for wrong in variants:
            with self.subTest(rows=len(wrong)), self.assertRaises(ValueError):
                evaluation.evaluate_ledgers(wrong, self.cfg, self.configs, self.targets)
        targets = dict(self.targets); targets[(0, 7, "A", "control")] = 0.
        with self.assertRaisesRegex(ValueError, "Val exposure"):
            evaluation.evaluate_ledgers(self.frame, self.cfg, self.configs, targets)

    def test_aggregate_barrier_checked_before_any_test_parquet_access(self):
        policy_stub = types.SimpleNamespace(condition_cfg=lambda *args: {})
        with patch.dict("sys.modules", {"unidream.experiments.wm_rl_release_policy": policy_stub}), \
                patch("unidream.experiments.wm_rl_release_benchmark.verify_test_barrier", side_effect=ValueError("barrier missing")), \
                patch.object(pd, "read_parquet") as read:
            with self.assertRaisesRegex(ValueError, "barrier missing"):
                evaluation.aggregate({"output_dir": "/not_read"}, "registration")
            read.assert_not_called()

    def test_report_exports_registered_family_counts_and_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            evaluation._write_report(path, self.result)
            text = (path / "index.html").read_text()
            for marker in ("主比較12", "WMの追加効果5", "対照との10比較", "未使用Holdout", "3seed", "Testで調整しません"):
                self.assertIn(marker, text)

    def test_aggregate_marker_fit_ledger_to_report_synthetic_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); old = root / "original"; new = root / "continued"
            (old / "final").mkdir(parents=True); new.mkdir()
            self.original.to_parquet(old / "final/executed_policy_ledger.parquet", index=False)
            (new / "preregistration.json").write_text('{"synthetic_only":true}\n')
            (new / "selection_barrier.json").write_text('{"synthetic_only":true}\n')
            registration_sha = evaluation._sha(new / "preregistration.json")
            barrier_sha = evaluation._sha(new / "selection_barrier.json")
            for (fold, seed, arm, condition), part in self.frame.groupby(["fold", "seed", "arm", "condition"]):
                job = evaluation._job(int(fold), arm, int(seed), condition)
                directory = new / "evaluations" / job; directory.mkdir(parents=True)
                fit_dir = new / "neural" / job; fit_dir.mkdir(parents=True)
                ledger_path = directory / "executed_ledger.parquet"
                part.to_parquet(ledger_path, index=False)
                marker = {"complete": True, "registration_sha256": registration_sha, "barrier_sha256": barrier_sha,
                    "job": job, "artifact_sha256": {"executed_ledger.parquet": evaluation._sha(ledger_path)}}
                (directory / "evaluation.json").write_text(json.dumps(marker))
                fit = {"selected_update": 150, "updates": 200,
                    "full_validation_metrics": {"mean_exposure": self.targets[(fold, seed, arm, condition)]},
                    "validation_metrics": {"mean_exposure": 1.12}}
                (fit_dir / "fit.json").write_text(json.dumps(fit))
            cfg = {**self.cfg, "output_dir": str(new), "source_config": "synthetic"}
            old_cfg = {"output_dir": str(old), "accounting": accounting()}
            stub = types.SimpleNamespace(condition_cfg=lambda config, condition: self.configs[condition])
            with patch.dict("sys.modules", {"unidream.experiments.wm_rl_release_policy": stub}), \
                    patch("unidream.experiments.wm_rl_release_benchmark.verify_test_barrier", return_value={"synthetic_only": True}) as guard, \
                    patch("unidream.experiments.wm_policy_ablation_benchmark.load_config", return_value=old_cfg):
                report = evaluation.aggregate(cfg, registration_sha)
            self.assertEqual(guard.call_count, 2)
            self.assertTrue(report["complete"])
            self.assertEqual(report["n_release_fits"], 30)
            self.assertEqual(report["n_market_rows"], 40 * 96)
            self.assertTrue(Path(report["report_path"]).exists())
            self.assertEqual(len(pd.read_csv(new / "final/primary_inference.csv")), 27)
            selections = pd.read_csv(new / "final/selection_metrics.csv")
            self.assertTrue(selections.target_source.eq("full_validation_metrics").all())
            self.assertFalse(selections.test_used_for_selection.any())
            for name, digest in report["artifact_sha256"].items():
                self.assertEqual(evaluation._sha(new / "final" / name), digest)


if __name__ == "__main__":
    unittest.main()
