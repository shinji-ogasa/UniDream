"""Complete BC/RL accounting, common benchmark masks and provenance gates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_policy_ablation_evaluation import ARMS, STAGES
from unidream.experiments.wm_policy_ablation_report import (
    _sha, _gate, verify_normalizers, write_policy_report,
)


def fixture(root, include_test=False):
    cfg = {"folds": list(range(6)), "seeds": [7, 19, 42],
        "statistics": {"bootstrap_replicates": 9, "block_days": [1, 7, 28], "seed": 23}}
    n = 6 * 96
    timestamps = pd.date_range("2025-04-01", periods=n, freq="15min", tz="UTC")
    returns = np.random.default_rng(21).normal(.0001, .001, n)
    common = pd.DataFrame({"fold": np.repeat(np.arange(6), 96), "decision_id": np.arange(n),
        "timestamp": timestamps, "bar_log_return": returns})
    policies = []
    normalizers = []
    fits = {}
    for seed in cfg["seeds"]:
        for arm in ARMS:
            for fold in cfg["folds"]:
                source = hashlib.sha256(f"purged_Train_ids_fold{fold}".encode()).hexdigest()
                normalizers.append({"fold": fold, "seed": seed, "arm": arm, "source_id": source,
                    "expected_source_id": source, "fit_split": "train"})
                path = root / "neural" / f"fold_{fold:02d}_{arm}_s{seed}" / "fit.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"frozen":true}')
                fits[str(path.relative_to(root))] = _sha(path)
            for stage in STAGES:
                panel = common.copy()
                panel["seed"], panel["arm"], panel["stage"] = seed, arm, stage
                panel["requested_position"] = 1 + .01 * (ord(arm) - ord("A")) + (.01 if stage == "rl" else 0)
                panel["split"] = "test" if include_test else "selection"
                policies.append(panel)
    controls = []
    for name, position in (("val_constant", .9), ("ewma_voltarget", .85)):
        panel = common.copy()
        panel["baseline"], panel["target_position"] = name, position
        controls.append(panel)
    (root / "preregistration.json").write_text('{"analysis":"conditional_saved_forecast_cohort"}')
    if include_test:
        (root / "validation_choices.json").write_text('{"test_selection_allowed":false}')
        (root / "test_barrier.json").write_text(json.dumps({"registration_sha256": _sha(root / "preregistration.json"),
            "validation_choices_sha256": _sha(root / "validation_choices.json"), "frozen_fit_sha256": fits}))
    return cfg, pd.concat(policies, ignore_index=True), {"normalizers": normalizers, "data_source_sha256": "synthetic"}, pd.concat(controls, ignore_index=True)


class PolicyReportTests(unittest.TestCase):
    def test_full144_fold24_seed8_aggregate_rows_and12_primary_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, policies, provenance, controls = fixture(root, include_test=True)
            result = write_policy_report(root, cfg, policies, include_test=True, provenance=provenance, baseline_ledger=controls)
            self.assertEqual(result["fold_rows"], 144)
            self.assertEqual(result["pooled_conditions"], 24)
            self.assertEqual(result["aggregate_conditions"], 8)
            self.assertEqual(result["n_primary_hypotheses"], 12)
            self.assertEqual(len(pd.read_csv(root / "final/fold_metrics.csv")), 144)
            self.assertEqual(len(pd.read_csv(root / "final/paired_inference.csv")), 36)
            controls = pd.read_csv(root / "final/baseline_metrics.csv")
            self.assertEqual(set(controls.baseline), {"bh", "constant", "ewma_vol"})
            contract = json.loads((root / "final/report_contract.json").read_text())
            self.assertTrue(contract["normalizer_audit"]["verified"])
            self.assertFalse(contract["full_calendar_pnl_claim"])
            self.assertFalse(contract["independently_unused_holdout"])
            self.assertFalse(contract["seed_market_independence_assumed"])
            html = (root / "final/index.html").read_text()
            self.assertIn("特徴選択", html)
            self.assertIn("全カレンダー", html)

    def test_missing_global_barrier_rejects_Test_before_ledger_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, _, provenance, _ = fixture(root)
            with self.assertRaisesRegex(ValueError, "global Test barrier is closed"):
                write_policy_report(root, cfg, include_test=True, provenance=provenance)

    def test_normalizer_must_match_purged_Train_source_for_all_72_pipelines(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg, _, provenance, _ = fixture(Path(temp))
            self.assertEqual(verify_normalizers(provenance, cfg)["n_pipeline_normalizers"], 72)
            provenance["normalizers"][0]["fit_split"] = "test"
            with self.assertRaisesRegex(ValueError, "Train decisions only"):
                verify_normalizers(provenance, cfg)
            provenance["normalizers"][0]["fit_split"] = "train"
            provenance["normalizers"][0]["source_id"] = "another_fold"
            with self.assertRaisesRegex(ValueError, "Train decisions only"):
                verify_normalizers(provenance, cfg)
            provenance["normalizers"].pop()
            with self.assertRaisesRegex(ValueError, "every registered"):
                verify_normalizers(provenance, cfg)

    def test_baseline_cannot_use_a_different_time_or_return_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, policies, provenance, controls = fixture(root)
            controls.loc[0, "bar_log_return"] += .01
            with self.assertRaisesRegex(ValueError, "exactly the same bars"):
                write_policy_report(root, cfg, policies, provenance=provenance, baseline_ledger=controls)
            self.assertFalse((root / "selection/report_completion.json").exists())

    def test_Test_rows_never_enter_validation_reporting_or_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, policies, provenance, controls = fixture(root, include_test=True)
            with self.assertRaisesRegex(ValueError, "Test rows cannot enter validation"):
                write_policy_report(root, cfg, policies, provenance=provenance, baseline_ledger=controls)

    def test_Val_report_does_not_require_or_open_Test_barrier(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, policies, provenance, controls = fixture(root)
            result = write_policy_report(root, cfg, policies, provenance=provenance, baseline_ledger=controls)
            self.assertFalse(result["include_test"])
            self.assertFalse((root / "test_barrier.json").exists())
            self.assertFalse((root / "final").exists())

    def test_fit_or_Val_mutation_invalidates_frozen_Test_reporting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, _, _, _ = fixture(root, include_test=True)
            path = next((root / "neural").glob("*/fit.json"))
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "fit changed"):
                _gate(root, cfg, True)
            (root / "validation_choices.json").write_text('{"test_selection_allowed":true}')
            with self.assertRaisesRegex(ValueError, "Val choices changed"):
                _gate(root, cfg, True)

    def test_existing_immutable_ledger_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg, policies, provenance, controls = fixture(root)
            result = write_policy_report(root, cfg, policies, provenance=provenance, baseline_ledger=controls)
            original = (root / "selection/policy_ledger.parquet").read_bytes()
            policies.loc[0, "requested_position"] = 1.01
            with self.assertRaisesRegex(ValueError, "immutable input artifact"):
                write_policy_report(root, cfg, policies, provenance=provenance, baseline_ledger=controls)
            self.assertEqual((root / "selection/policy_ledger.parquet").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
