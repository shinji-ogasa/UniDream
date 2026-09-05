import copy
from pathlib import Path
import unittest

import numpy as np
import yaml

from unidream.experiments.oracle_frozen_procedure_parity import (
    FOLDS, MEANS, POLICIES, compare_array, compare_tree, validate_config, validate_completed_fold,
)


class OracleFrozenProcedureParityTests(unittest.TestCase):
    def test_resume_cannot_pass_empty_or_missing_artifact_manifests(self):
        saved = {"registration_sha256": "frozen", "rows": [], "scores": [],
                 "max_absolute_differences": {}, "artifact_sha256": {}}
        with self.assertRaisesRegex(ValueError, "family"):
            validate_completed_fold(saved, 5, Path("absent"), "frozen", {}, {}, {})
        saved["rows"] = [{"fold": 5, "candidate_id": cid} for cid in POLICIES]
        saved["scores"] = [{"fold": 5, "mean_id": mid} for mid in MEANS]
        with self.assertRaisesRegex(ValueError, "inventory"):
            validate_completed_fold(saved, 5, Path("absent"), "frozen", {}, {}, {})

    def test_frozen_inventory_and_no_new_interval_or_looser_tolerance(self):
        path = Path(__file__).parents[1] / "configs/oracle_frozen_procedure_parity_20260906.yaml"
        cfg = yaml.safe_load(path.read_text()); validate_config(cfg)
        self.assertEqual(len(FOLDS) * len(POLICIES), 96)
        self.assertEqual(len(FOLDS) * len(MEANS), 40)
        for key, value in {"calendar_evaluation_folds": list(range(26, 34)),
                "development_folds": list(range(15, 24)), "forecast_atol": .001,
                "targets_must_match_exactly": False, "selection_permitted": True,
                "additional_periods_permitted": True, "expected_source_artifacts": 920}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**cfg, key: value})
        changed = copy.deepcopy(cfg); changed["metadata_bindings"].pop(next(iter(changed["metadata_bindings"])))
        with self.assertRaises(ValueError): validate_config(changed)

    def test_forecast_tolerance_cannot_hide_changed_nan_or_inf_support(self):
        expected = np.array([.01, np.nan, .02])
        got = expected.copy(); got[0] += 1e-15
        self.assertGreater(compare_array(got, expected, name="forecast"), 0)
        for bad in (np.array([.01, .0, .02]), np.array([.01, np.nan, np.inf]),
                    np.array([.01, np.nan]), expected.astype(complex), np.array([.02, np.nan, .02])):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                compare_array(bad, expected, name="forecast")

    def test_exact_target_nan_semantics_and_mask_dtype_are_preserved(self):
        target = np.array([1., np.nan, .92])
        self.assertEqual(compare_array(target.copy(), target, name="target", exact=True), 0.)
        changed = target.copy(); changed[0] += 1e-15
        with self.assertRaises(ValueError): compare_array(changed, target, name="target", exact=True)
        with self.assertRaises(ValueError):
            compare_array(np.array([1, 0]), np.array([True, False]), name="mask", exact=True)
        with self.assertRaises(ValueError):
            compare_array(np.array([1., 0.]), np.array([1, 0]), name="timestamp", exact=True)

    def test_full_trace_fields_discrete_counts_and_numeric_values_are_compared(self):
        trace = {"count": 2, "decision_trace": {"targets": [None, 1.0],
                 "reasons": ["learned", "forecast_unavailable"], "known_nav": [1., 1.1]}}
        self.assertEqual(compare_tree(copy.deepcopy(trace), trace, name="trace"), 0.)
        for change in ("missing", "extra", "reason", "counttype", "length", "large_numeric", "none"):
            bad = copy.deepcopy(trace)
            if change == "missing": bad.pop("count")
            elif change == "extra": bad["new_field"] = 1
            elif change == "reason": bad["decision_trace"]["reasons"][1] = "learned"
            elif change == "counttype": bad["count"] = 2.
            elif change == "length": bad["decision_trace"]["targets"].pop()
            elif change == "large_numeric": bad["decision_trace"]["known_nav"][1] += .001
            else: bad["decision_trace"]["targets"][0] = 0.
            with self.subTest(change=change), self.assertRaises(ValueError): compare_tree(bad, trace, name="trace")


if __name__ == "__main__":
    unittest.main()
