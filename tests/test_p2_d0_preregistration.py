"""Contract-only tests for the P2-D0 preregistration boundary.

These tests inspect fixed schema, hashes, registries, and the authenticated
runtime call order. They do not load data, fit models, score forecasts, or
start an outer operation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from unidream.experiments import p2_d0_prereg as p2


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "docs" / "experiments" / "p2_d0_prereg_manifest.json"


def _read_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _rebind_digest(payload: dict) -> None:
    payload["manifest_sha256"] = p2.canonical_manifest_sha256(payload)


class P2D0PreregistrationTests(unittest.TestCase):
    def test_fixed_manifest_is_result_free_and_deep_frozen(self) -> None:
        payload = _read_manifest()
        self.assertEqual(payload["manifest_sha256"], "a0ac7357abadb4b459f0687b12fb5926089fe9e1bd0987990ede82750b952cd2")
        self.assertFalse(payload["results_observed"])
        self.assertEqual(payload["base_revision"], p2.REGISTERED_BASE_REVISION)
        self.assertEqual(
            payload["amends_manifest_sha256"],
            "d1854827bd4aa204cc2b5cde375edf62583bf0d164b39e8ac25a6c10ad7dc0c4",
        )
        p2.validate_fixed_manifest(payload, root=ROOT)
        frozen = p2.load_fixed_manifest()
        self.assertIsInstance(frozen, type(__import__("types").MappingProxyType({})))
        self.assertEqual(frozen["manifest_sha256"], payload["manifest_sha256"])
        with self.assertRaises(TypeError):
            frozen["results_observed"] = True  # type: ignore[index]

    def test_manifest_forgery_and_contract_mutations_are_rejected(self) -> None:
        base = _read_manifest()
        mutations = {
            "manifest_sha256": "0" * 64,
            "results_observed": True,
            "amends_manifest_sha256": "1" * 64,
            "common.feature_arms.ohlcv13.columns": list(reversed(p2.OHLCV13_COLUMNS)),
            "common.feature_arms.shared_common_row_rule": "use separate arm rows",
            "common.target_contract.following_edge": "required",
            "common.split_contract.folds.outer_validation.end": "2023-01-02T00:00:00Z",
            "common.models.hist_gradient_boosting.max_depth": 8,
            "common.coverage_contract.thresholds.common_row_fraction_min": 0.0,
            "common.statistical_contract.family_size": 13,
            "common.historical_boundary.untouched_claim_forbidden": False,
            "common.runner_contract.outer_operation_policy.max_runs": 2,
            "common.runtime_contract.frozen_schema_digest": "2" * 64,
        }
        for path, value in mutations.items():
            with self.subTest(path=path):
                candidate = copy.deepcopy(base)
                parts = path.split(".")
                cursor = candidate
                for part in parts[:-1]:
                    cursor = cursor[part]
                cursor[parts[-1]] = value
                if path not in {"manifest_sha256", "results_observed", "amends_manifest_sha256"}:
                    _rebind_digest(candidate)
                with self.assertRaises(p2.P2D0PreregistrationError):
                    p2.validate_fixed_manifest(candidate, root=ROOT)

    def test_feature_target_split_and_historical_boundaries_are_exact(self) -> None:
        manifest = _read_manifest()
        common = manifest["common"]
        arms = common["feature_arms"]
        self.assertEqual(tuple(arms["full17"]["columns"]), p2.FULL17_COLUMNS)
        self.assertEqual(tuple(arms["ohlcv13"]["columns"]), p2.OHLCV13_COLUMNS)
        self.assertEqual(arms["ohlcv13"]["columns"], arms["full17"]["columns"][:13])
        self.assertIn("exactly the intersection", arms["shared_common_row_rule"])
        self.assertIn("identical timestamp rows", arms["shared_common_row_rule"])
        self.assertIn("neither arm may recover", arms["shared_common_row_rule"])
        self.assertEqual(common["forecast_horizons"], [1, 4, 8, 16])
        self.assertEqual(common["forbidden_horizons"], [64])
        self.assertFalse(common["utility_head"])
        self.assertEqual(common["target_contract"]["target_formula"], "y[t,h] = sum(return[t+1:t+h+1])")
        self.assertEqual(common["target_contract"]["required_edges"], "every exact 15m edge t->t+1 through t+h-1->t+h")
        self.assertEqual(common["target_contract"]["following_edge"], "t+h->t+h+1 is not required")
        self.assertEqual(common["context_contract"]["window"], "current-inclusive [t-63,t]")
        self.assertEqual(common["split_contract"]["purge_bars"], 16)
        self.assertEqual(common["minimum_history_rows"], 16384)
        self.assertEqual(
            common["split_contract"]["folds"]["historical_report_only"],
            {
                "split_id": "historical_report_2023",
                "start": "2023-01-01T00:00:00Z",
                "end": "2024-01-01T00:00:00Z",
                "role": "report_only",
            },
        )
        self.assertTrue(common["historical_boundary"]["untouched_claim_forbidden"])
        self.assertTrue(common["historical_boundary"]["future_holdout_required"])

    def test_machine_readable_models_coverage_holm_and_once_policy(self) -> None:
        manifest = _read_manifest()
        common = manifest["common"]
        self.assertEqual(
            common["models"]["zero_return"]["binary_prediction"],
            "class-1 probability 0.5",
        )
        self.assertEqual(
            common["models"]["persistence_last_observed"]["binary_prediction"],
            "1-eps when return[t] > 0, otherwise eps, eps=1e-6",
        )
        hgb = common["models"]["hist_gradient_boosting"]
        self.assertEqual(
            {key: hgb[key] for key in ("max_iter", "max_leaf_nodes", "max_depth", "min_samples_leaf", "early_stopping", "deep_model")},
            {
                "max_iter": 200,
                "max_leaf_nodes": 15,
                "max_depth": 4,
                "min_samples_leaf": 64,
                "early_stopping": False,
                "deep_model": False,
            },
        )
        self.assertEqual(common["coverage_contract"]["thresholds"]["common_row_fraction_min"], 0.9)
        self.assertEqual(common["coverage_contract"]["thresholds"]["finite_prediction_fraction_min"], 0.95)
        self.assertEqual(common["coverage_contract"]["thresholds"]["scored_action_fraction_min"], 0.8)
        self.assertIn("every primary comparison cell", common["coverage_contract"]["all_na_rule"])
        self.assertEqual(common["statistical_contract"]["family_size"], 14)
        self.assertEqual(common["statistical_contract"]["multiplicity_method"], "Holm-Bonferroni")
        self.assertEqual(
            common["runner_contract"]["outer_operation_policy"]["mode"],
            "report_only",
        )
        self.assertEqual(common["runner_contract"]["outer_operation_policy"]["max_runs"], 1)
        self.assertTrue(common["runner_contract"]["d1_excluded"])
        self.assertEqual(common["runner_contract"]["action_primitive_execution_status"], "blocked_not_implemented")

    def test_registry_hashes_and_rows_are_result_free(self) -> None:
        manifest = p2.load_fixed_manifest()
        p2.validate_pinned_artifacts(manifest)
        common = manifest["common"]
        trial_ref = common["trial_registry"]
        comparison_ref = common["primary_comparison_registry"]
        self.assertEqual(trial_ref["record_count"], 10)
        self.assertEqual(comparison_ref["record_count"], 14)
        self.assertEqual(comparison_ref["family_size"], 14)
        self.assertTrue(all(
            json.loads(line)["results_observed"] is False
            for line in (ROOT / trial_ref["path"]).read_text(encoding="utf-8").splitlines()
        ))
        comparisons = [
            json.loads(line)
            for line in (ROOT / comparison_ref["path"]).read_text(encoding="utf-8").splitlines()
        ]
        self.assertTrue(all(row["primary"] is True for row in comparisons))
        self.assertEqual(
            {row["support_id"] for row in comparisons},
            {"outer_validation_2022"},
        )
        self.assertEqual(
            {row["horizon"] for row in comparisons if row["cost_mode"] == "on"},
            {4},
        )
        self.assertTrue(all("result" not in row and "score" not in row for row in comparisons))

    def test_authenticated_runtime_is_the_only_production_boundary(self) -> None:
        self.assertEqual(
            p2.P2_D0_RUNTIME_VALIDATION_ENTRYPOINT,
            "unidream.experiments.p2_d0_prereg.load_authenticated_v4_runtime",
        )
        self.assertNotIn("validate_v4_runtime_inputs", p2.__all__)
        fake_result = {
            "p1_runtime_validation_entrypoint": p2.P1_V4_RUNTIME_VALIDATION_ENTRYPOINT,
            "p1_manifest_sha256": p2.P1_REGISTERED_MANIFEST_SHA256,
            "p1_results_observed": False,
            "v4_runtime_validation_status": "passed",
            "v4_runtime_provenance_disposition": "authenticated",
            "v4_feature_path": "/tmp/features.parquet",
            "v4_returns_path": "/tmp/returns.parquet",
            "v4_availability_path": "/tmp/availability.parquet",
        }
        with patch(
            "unidream.experiments.runtime.validate_p1_v4_runtime_inputs",
            return_value=fake_result,
        ) as wrapper:
            result = p2.load_authenticated_v4_runtime()
        wrapper.assert_called_once_with()
        self.assertIsInstance(result, type(__import__("types").MappingProxyType({})))
        self.assertEqual(result["p2_manifest_sha256"], p2.REGISTERED_MANIFEST_SHA256)
        self.assertEqual(result["v4_runtime_validation_status"], "passed")
        self.assertNotIn("features", result)
        self.assertNotIn("returns", result)

    def test_p2_runtime_identity_is_pinned_separately_from_p1(self) -> None:
        manifest = _read_manifest()
        runtime = manifest["common"]["runtime_contract"]
        self.assertEqual(runtime["p2_runtime_entrypoint"], p2.P2_D0_RUNTIME_VALIDATION_ENTRYPOINT)
        self.assertEqual(runtime["runtime_validation_entrypoint"], p2.P1_V4_RUNTIME_VALIDATION_ENTRYPOINT)
        self.assertNotEqual(p2.REGISTERED_MANIFEST_SHA256, p2.P1_REGISTERED_MANIFEST_SHA256)
        self.assertFalse(manifest["results_observed"])
        self.assertEqual(manifest["amendment_history"][0]["results_observed"], False)


if __name__ == "__main__":
    unittest.main()
