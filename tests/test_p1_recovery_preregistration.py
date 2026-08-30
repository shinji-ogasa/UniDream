"""Contract-only tests for the preregistered P1 recovery protocol.

These tests inspect schema, immutable field pins, and registry structure only;
they deliberately do not generate data, fit models, or inspect outcomes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from unidream.experiments.p1_recovery_prereg import (
    P1PreregistrationError,
    canonical_json_sha256,
    canonical_manifest_sha256,
    exact_file_sha256,
    load_fixed_manifest,
    validate_pinned_artifacts,
    validate_fixed_manifest,
)


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "docs" / "experiments" / "p1_recovery_prereg_manifest.json"


def _read_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _set_path(payload: dict, path: str, value: object) -> None:
    parts = path.split(".")
    cursor = payload
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _review_manifest(payload: dict) -> None:
    """Validate a working copy against the registered digest."""
    payload["manifest_sha256"] = canonical_manifest_sha256(payload)
    validate_fixed_manifest(
        payload,
        expected_digest=payload["manifest_sha256"],
    )


class P1PreregistrationTests(unittest.TestCase):
    def test_manifest_json_and_fixed_schema(self) -> None:
        payload = _read_manifest()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["status"], "preregistered")
        self.assertEqual(payload["common"]["feature_columns"], [
            "open_ret", "high_ret", "low_ret", "close_ret", "vol_ret",
            "RSI_14", "macd", "macd_signal", "atr_norm_ret", "atr",
            "rv_4", "rv_16", "rv_96", "funding_rate", "basis",
            "basis_mom", "basis_abs",
        ])
        validate_fixed_manifest(payload)
        validate_pinned_artifacts(payload, root=ROOT)

    def test_critical_fields_cannot_be_omitted_or_altered(self) -> None:
        base = _read_manifest()
        _review_manifest(copy.deepcopy(base))

        missing = copy.deepcopy(base)
        del missing["common"]["oof"]
        with self.assertRaises(P1PreregistrationError):
            _review_manifest(missing)

        for field in ("amends_manifest_sha256", "amendment_reason", "results_observed"):
            missing = copy.deepcopy(base)
            del missing[field]
            with self.subTest(missing=field):
                with self.assertRaises(P1PreregistrationError):
                    _review_manifest(missing)

        mutations = {
            "amends_manifest_sha256": "wrong-prior-digest",
            "amendment_reason": "post-result revision",
            "results_observed": True,
            "common.target_end_formula": "target_end[t,h] = t + h",
            "common.sequence_context_bars": 32,
            "common.model_input_rule": "flatten the 64-bar context",
            "common.binary_label_rule": "label=1 for y>=0",
            "common.learned_fit_contract.feature_scaler": "global scaler",
            "common.split_end_rule": "allow target tails to cross split boundary",
            "common.evaluation_split_state_policy": "carry inventory across all splits",
            "common.oof.min_history_rows": 2048,
            "common.oof.origin_schedule.step": 512,
            "common.oof.primary_inferential_support.fit_prefix_range": [0, 89999],
            "common.oof.primary_inferential_support.prediction_range": [80000, 90000],
            "common.oof.outer_report_operation.origin": 110000,
            "common.availability.outcome_label_row_rule": "all three sidecar flags are required on target bars",
            "common.action_contract.commitment_bars": 8,
            "common.action_contract.argmax_selector": "other.selector",
            "common.action_contract.argmax_tie_rule": "first candidate wins",
            "common.action_contract.benchmark_hold_path": "candidate-local hold",
            "common.v4_load_contract.feature_path": "checkpoints/data_cache/other_features.parquet",
            "common.v4_load_contract.metadata_path": "checkpoints/data_cache/local_metadata.json",
            "common.v4_load_contract.require_explicit_paths": False,
            "common.v4_load_contract.known_cache_local_snapshot.source_provenance_digest": "wrong-revision",
            "common.runner_contract.outer_test_selection_allowed": True,
            "common.models.ridge.solver": "auto",
            "common.metrics.coverage_definitions.context_fraction": "context_complete / all_rows",
            "common.gates.block_bootstrap.invalid_replicate_policy": "drop N/A rows and compact",
            "common.gates.high_snr_recovery.utility_per_seed_rule": "aggregate only",
            "common.gates.high_snr_recovery.clairvoyant_rule": "clairvoyant is report-only",
            "synthetic_contract.n_rows": 20000,
            "synthetic_contract.raw_n_rows": 120000,
            "synthetic_contract.draw_order": ["epsilon first"],
            "synthetic_contract.availability.gap_block_count": 100,
            "synthetic_contract.outer_report_operation.refit_origins": [110000],
            "common.metrics.primary_support_policy.s3.origin_raw": 104529,
            "scenarios.S3.signal.source_feature": "hidden_z",
            "scenarios.S3.signal.prefix_eligibility": "target rows only",
            "scenarios.S3.seeds": [20260830, 20260831],
            "scenarios.S3.raw_body_indices.2023-01-01T00:00:00Z": 142491,
            "scenarios.S3.primary_inferential_operation.prediction_raw_range": [102492, 139568],
            "scenarios.S3.excluded_common_schedule_origin_raw_index": 142493,
            "provenance.v4_parent.feature_rows": 173110,
        }
        for path, value in mutations.items():
            with self.subTest(path=path):
                mutated = copy.deepcopy(base)
                _set_path(mutated, path, value)
                with self.assertRaises(P1PreregistrationError):
                    _review_manifest(mutated)

    def test_reporting_arm_ledger_is_complete_and_unique(self) -> None:
        manifest = _read_manifest()
        path = ROOT / "docs" / "experiments" / "p1_recovery_trial_registry.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 56)
        self.assertEqual(len({row["trial_id"] for row in rows}), 56)
        self.assertTrue(all(row["primary"] is True for row in rows))
        self.assertEqual({row["scenario_id"] for row in rows}, {
            "S0", "S1", "S2-high", "S2-medium", "S2-low", "S3",
        })
        self.assertEqual({row["model_id"] for row in rows}, {
            "zero_return", "persistence_last_observed", "ridge", "logistic",
        })
        self.assertEqual({row["cost_mode"] for row in rows}, {"off", "on"})
        self.assertEqual(
            exact_file_sha256(path),
            manifest["common"]["trial_registry"]["sha256"],
        )

    def test_primary_comparison_registry_defines_multiplicity_family(self) -> None:
        manifest = _read_manifest()
        ref = manifest["common"]["primary_comparison_registry"]
        path = ROOT / ref["path"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        required = set(ref["required_fields"])
        self.assertTrue({"support_id", "support_range", "support_role"} <= required)
        self.assertEqual(len(rows), ref["family_size"])
        self.assertEqual(exact_file_sha256(path), ref["sha256"])
        self.assertEqual(len({row["comparison_id"] for row in rows}), ref["family_size"])
        self.assertTrue(all(row["primary"] is True for row in rows))
        self.assertTrue(all(required <= set(row) for row in rows))
        expected_support = {
            "S0": ("synthetic_validation", [90000, 100000]),
            "S1": ("synthetic_validation", [90000, 100000]),
            "S2": ("synthetic_validation", [90000, 100000]),
            "S3": ("s3_validation", [104528, 139568]),
        }
        self.assertTrue(all(
            (row["support_id"], row["support_range"], row["support_role"])
            == (*expected_support[row["scenario_id"]], "primary_inferential_gate")
            for row in rows
        ))
        self.assertTrue(all(row["horizon"] == 4 for row in rows))
        self.assertTrue(all(row["cost_mode"] in {"off", "on"} for row in rows))
        arm_path = ROOT / manifest["common"]["trial_registry"]["path"]
        arm_ids = {
            json.loads(line)["trial_id"]
            for line in arm_path.read_text(encoding="utf-8").splitlines()
        }
        self.assertTrue(all(
            row["candidate_id"] in arm_ids
            and (
                row["baseline_id"] in arm_ids
                or row["baseline_id"].endswith("__benchmark_hold__off")
            )
            for row in rows
        ))

    def test_action_contract_is_canonical_and_separate_from_manifest_hash(self) -> None:
        manifest = _read_manifest()
        ref = manifest["common"]["action_execution_contract_reference"]
        path = ROOT / ref["path"]
        contract = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(contract["h_decision"], 4)
        self.assertEqual(contract["commitment_bars"], 4)
        self.assertEqual(contract["execution_delay_bars"], 1)
        self.assertEqual(contract["feature_unavailable_policy"], "hold_and_score_commitment")
        self.assertEqual(contract["outcome_unavailable_policy"], "exclude_block")
        self.assertEqual(contract["execution_skip_policy"], "hold_commitment")
        self.assertEqual(contract["return_unit"], "additive_log_return")
        self.assertEqual(canonical_json_sha256(contract), ref["sha256"])
        mode_refs = manifest["common"]["cost_mode_contracts"]
        off = json.loads((ROOT / mode_refs["off"]["path"]).read_text(encoding="utf-8"))
        self.assertEqual(canonical_json_sha256(off), mode_refs["off"]["sha256"])
        self.assertEqual(mode_refs["on"]["sha256"], ref["sha256"])
        self.assertEqual(
            mode_refs["off"]["sha256"],
            manifest["common"]["cost_modes"]["off"]["contract_hash"],
        )
        self.assertEqual(
            mode_refs["on"]["sha256"],
            manifest["common"]["cost_modes"]["on"]["contract_hash"],
        )
        cost_only = {"spread_bps", "slippage_bps", "fee_rate", "transition_cost_rate"}
        self.assertEqual(
            {key: off[key] for key in off if key not in cost_only},
            {key: contract[key] for key in contract if key not in cost_only},
        )
        self.assertTrue(all(off[key] == 0.0 for key in cost_only))

    def test_s1_recovery_registry_requires_each_seed_and_clairvoyant_sanity(self) -> None:
        manifest = _read_manifest()
        path = ROOT / manifest["common"]["primary_comparison_registry"]["path"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        row = next(
            item for item in rows
            if item["comparison_id"] == "S1__ridge__utility_vs_hold__cost_on"
        )
        self.assertEqual(row["candidate_id"], "S1__ridge__on")
        self.assertEqual(row["baseline_id"], "S1__benchmark_hold__off")
        self.assertEqual(row["support_id"], "synthetic_validation")
        self.assertEqual(row["support_range"], [90000, 100000])
        self.assertIn("all ten seed-level validation utility deltas > 0", row["gate"])
        self.assertIn("every seed clairvoyant value > Ridge value", row["gate"])
        high_snr = manifest["common"]["gates"]["high_snr_recovery"]
        self.assertIn("every seed", high_snr["utility_per_seed_rule"])
        self.assertIn("strictly greater", high_snr["clairvoyant_rule"])

    def test_v4_loader_requires_explicit_bodies_and_frozen_metadata(self) -> None:
        manifest = _read_manifest()
        contract = manifest["common"]["v4_load_contract"]
        parent = manifest["provenance"]["v4_parent"]
        self.assertEqual(contract["loader"], "unidream.data.cache_v4.load_cache_v4")
        self.assertTrue(contract["require_explicit_paths"])
        self.assertEqual(contract["cache_dir_cache_tag_fallback"], "forbidden")
        self.assertEqual(contract["cache_tag"], parent["cache_tag"])
        self.assertEqual(contract["metadata_path"], parent["metadata_path"])
        self.assertEqual(contract["frozen_metadata_sha256"], parent["metadata_sha256"])
        self.assertEqual(contract["frozen_source_provenance_digest"], parent["source_provenance_digest"])
        self.assertEqual(contract["frozen_schema_digest"], parent["schema_digest"])
        self.assertEqual(contract["frozen_content_digests"], parent["content_digests"])
        local = contract["known_cache_local_snapshot"]
        self.assertEqual(local["metadata_sha256"], "bade1775884cd22c8675af225b429976aa6b2c60b859b4a591c76f8a87d17450")
        self.assertEqual(local["source_provenance_digest"], "1e78ccf3162567e799b05a1c25dbe12a1c4c37e8e5a2abf2f9b95a70c380e2db")
        self.assertEqual(local["schema_digest"], contract["frozen_schema_digest"])
        self.assertEqual(local["content_digests"], contract["frozen_content_digests"])
        self.assertEqual(local["rows"], parent["feature_rows"])
        self.assertEqual(local["sidecar_rows"], parent["sidecar_rows"])
        self.assertIn("differ", local["difference_from_frozen"])
        self.assertNotEqual(contract["cache_local_metadata_path"], contract["metadata_path"])
        self.assertIn("never pass cache-local metadata as metadata_path", contract["cache_local_metadata_policy"])
        self.assertIn("do not hide", contract["cache_local_frozen_difference_policy"])
        self.assertTrue({
            "v4_feature_path",
            "v4_returns_path",
            "v4_availability_path",
            "v4_frozen_metadata_path",
            "v4_frozen_metadata_sha256",
            "v4_cache_local_metadata_path",
            "v4_cache_local_metadata_sha256",
            "v4_cache_local_source_provenance_digest",
        }.issubset(set(contract["artifact_echo_fields"])))
        frozen_metadata = json.loads(
            (ROOT / contract["metadata_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(exact_file_sha256(ROOT / contract["metadata_path"]), contract["frozen_metadata_sha256"])
        self.assertEqual(frozen_metadata["content_digests"], contract["frozen_content_digests"])

    def test_production_loader_succeeds_and_freezes_pinned_manifest(self) -> None:
        manifest = load_fixed_manifest()
        self.assertEqual(manifest["manifest_sha256"], "5f8dbd798cf6dc44e15c94b45bc49081c1f7eefea2b89369b682e8e1c7f5d0cc")
        with self.assertRaises(TypeError):
            manifest["common"] = {}  # type: ignore[index]

    def test_recomputed_one_field_mutation_fails_registered_digest(self) -> None:
        payload = _read_manifest()
        payload["common"]["return_unit"] = "simple_return"
        payload["manifest_sha256"] = canonical_manifest_sha256(payload)
        with self.assertRaises(P1PreregistrationError):
            validate_fixed_manifest(payload, expected_digest=payload["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
