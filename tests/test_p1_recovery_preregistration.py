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
    REGISTERED_MANIFEST_SHA256,
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
        self.assertEqual(
            payload["amends_manifest_sha256"],
            "d1854827bd4aa204cc2b5cde375edf62583bf0d164b39e8ac25a6c10ad7dc0c4",
        )
        self.assertEqual(
            payload["amendment_reason"],
            "pre-result execution authorization after action/MBB implementation audit",
        )
        self.assertEqual(len(payload["amendment_history"]), 5)
        self.assertFalse(payload["results_observed"])
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

        for field in (
            "amends_manifest_sha256",
            "amendment_reason",
            "amendment_history",
            "results_observed",
        ):
            missing = copy.deepcopy(base)
            del missing[field]
            with self.subTest(missing=field):
                with self.assertRaises(P1PreregistrationError):
                    _review_manifest(missing)

        mutations = {
            "amends_manifest_sha256": "wrong-prior-digest",
            "amendment_reason": "post-result revision",
            "amendment_history": [],
            "results_observed": True,
            "common.target_end_formula": "target_end[t,h] = t + h",
            "common.sequence_context_bars": 32,
            "common.model_input_rule": "flatten the 64-bar context",
            "common.binary_label_rule": "label=1 for y>=0",
            "common.learned_fit_contract.feature_scaler": "global scaler",
            "common.split_end_rule": "allow target tails to cross split boundary",
            "common.evaluation_split_state_policy": "carry inventory across all splits",
            "common.index_range_contract": "inclusive ranges",
            "common.oof.min_history_rows": 2048,
            "common.oof.min_history_rule": "count raw prefix rows",
            "common.oof.range_semantics": "inclusive end",
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
            "common.v4_load_contract.body_validation_policy": "trust cache directory",
            "common.v4_load_contract.missing_unknown_mismatch_policy": "warn and continue",
            "common.v4_load_contract.known_cache_local_snapshot.source_provenance_digest": "wrong-revision",
            "common.runner_contract.outer_test_selection_allowed": True,
            "common.models.ridge.solver": "auto",
            "common.metrics.coverage_definitions.context_fraction": "context_complete / all_rows",
            "common.gates.block_bootstrap.invalid_replicate_policy": "drop N/A rows and compact",
            "common.gates.block_bootstrap.action_primitive_record_fields": "candidate utility only",
            "common.gates.block_bootstrap.action_bootstrap_replay_policy": "replay policy over bootstrap rows",
            "common.gates.block_bootstrap.rng_lifecycle": "reseed each replicate",
            "common.gates.block_bootstrap.quantile_method": "nearest",
            "common.gates.block_bootstrap.denominator_policy": "omit failed rows",
            "common.gates.high_snr_recovery.utility_per_seed_rule": "aggregate only",
            "common.gates.high_snr_recovery.clairvoyant_rule": "clairvoyant is report-only",
            "synthetic_contract.n_rows": 20000,
            "synthetic_contract.raw_n_rows": 120000,
            "synthetic_contract.draw_order": ["epsilon first"],
            "synthetic_contract.random_distribution": "not standard normal",
            "synthetic_contract.random_dtype": "float32",
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
        self.assertTrue({"support_id", "support_range", "support_range_semantics", "support_role"} <= required)
        self.assertEqual(ref["action_required_fields"], [
            "action_bootstrap_replay_policy",
            "action_primitive_hash_fields",
        ])
        self.assertEqual(len(rows), ref["family_size"])
        self.assertEqual(exact_file_sha256(path), ref["sha256"])
        self.assertEqual(len({row["comparison_id"] for row in rows}), ref["family_size"])
        self.assertTrue(all(row["primary"] is True for row in rows))
        action_rows = [row for row in rows if "action_bootstrap_replay_policy" in row]
        self.assertEqual(len(action_rows), 10)
        self.assertTrue(all(
            row["action_primitive_hash_fields"] == [
                "action_primitive_payload_sha256",
                "action_primitive_schema_sha256",
                "action_primitive_content_sha256",
            ]
            for row in action_rows
        ))
        self.assertTrue(all(required <= set(row) for row in rows))
        self.assertTrue(all(
            row["support_range_semantics"] == "zero-based [start,end) right-exclusive; end excluded"
            for row in rows
        ))
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
        action_ids = {
            "S0__ridge__utility_vs_hold__cost_on",
            "S0__persistence__utility_vs_hold__cost_on",
            "S1__ridge__utility_vs_hold__cost_on",
            "S2__high_vs_medium__ridge__normalized_regret__cost_on",
            "S2__high_vs_medium__ridge__utility__cost_on",
            "S2__high_vs_medium__ridge__agreement__cost_on",
            "S2__medium_vs_low__ridge__normalized_regret__cost_on",
            "S2__medium_vs_low__ridge__utility__cost_on",
            "S2__medium_vs_low__ridge__agreement__cost_on",
            "S3__injected_vs_control__ridge__utility__cost_on",
        }
        replay = "resample stored canonical action block record indices and recompute declared means/sums/ratios/DiD; never replay policy state over a resampled or nonchronological sequence"
        self.assertTrue(all(
            ("action_bootstrap_replay_policy" in row) == (row["comparison_id"] in action_ids)
            and (row.get("action_bootstrap_replay_policy") in {None, replay})
            for row in rows
        ))
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

    def test_primary_comparison_semantic_tuples_are_exact(self) -> None:
        manifest = _read_manifest()
        path = ROOT / manifest["common"]["primary_comparison_registry"]["path"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        s0_gate = "Holm-rank-adjusted direction-aware lower percentile <= 0 for every fixed block length; positive-edge Holm rejection is false; never promote"
        s1_mse_gate = "Holm-adjusted one-sided paired bootstrap p <= 0.05 and direction-aware point delta < 0"
        s1_utility_gate = "all ten seed-level validation utility deltas > 0 and non-N/A; every seed on the identical scored mask has mean realized same-state clairvoyant net utility/value strictly greater than Ridge mean realized net utility/value; aggregate Holm-adjusted one-sided paired bootstrap p <= 0.05 and favorable point delta > 0"
        s3_gate = "Holm-adjusted one-sided paired bootstrap p <= 0.05 and favorable point delta > 0"
        ge_high = "Holm-adjusted monotonic contrast p <= 0.05 and median paired contrast high-medium >= -1e-12"
        le_high = "Holm-adjusted monotonic contrast p <= 0.05 and median paired contrast high-medium <= 1e-12"
        ge_low = "Holm-adjusted monotonic contrast p <= 0.05 and median paired contrast medium-low >= -1e-12"
        le_low = "Holm-adjusted monotonic contrast p <= 0.05 and median paired contrast medium-low <= 1e-12"
        expected = {
            "S0__ridge__utility_vs_hold__cost_on": ("S0__ridge__on", "S0__benchmark_hold__off", "paired_net_utility_delta_vs_hold", "on", "non_positive", s0_gate, "synthetic_validation", [90000, 100000]),
            "S0__persistence__utility_vs_hold__cost_on": ("S0__persistence_last_observed__on", "S0__benchmark_hold__off", "paired_net_utility_delta_vs_hold", "on", "non_positive", s0_gate, "synthetic_validation", [90000, 100000]),
            "S1__ridge__mse_vs_zero__cost_off": ("S1__ridge__off", "S1__zero_return__off", "mse_delta_vs_baseline", "off", "negative", s1_mse_gate, "synthetic_validation", [90000, 100000]),
            "S1__ridge__utility_vs_hold__cost_on": ("S1__ridge__on", "S1__benchmark_hold__off", "paired_net_utility_delta_vs_hold", "on", "positive", s1_utility_gate, "synthetic_validation", [90000, 100000]),
            "S2__high_vs_medium__ridge__mse_skill__cost_off": ("S2-high__ridge__off", "S2-medium__ridge__off", "forecast_mse_skill_vs_zero", "off", "high_ge_medium", ge_high, "synthetic_validation", [90000, 100000]),
            "S2__high_vs_medium__ridge__normalized_regret__cost_on": ("S2-high__ridge__on", "S2-medium__ridge__on", "normalized_action_regret", "on", "high_le_medium", le_high, "synthetic_validation", [90000, 100000]),
            "S2__high_vs_medium__ridge__utility__cost_on": ("S2-high__ridge__on", "S2-medium__ridge__on", "s2_timing_net_utility_delta", "on", "high_ge_medium", ge_high, "synthetic_validation", [90000, 100000]),
            "S2__high_vs_medium__ridge__agreement__cost_on": ("S2-high__ridge__on", "S2-medium__ridge__on", "feasible_action_agreement", "on", "high_ge_medium", ge_high, "synthetic_validation", [90000, 100000]),
            "S2__high_vs_medium__logistic__log_loss__cost_off": ("S2-high__logistic__off", "S2-medium__logistic__off", "log_loss", "off", "high_le_medium", le_high, "synthetic_validation", [90000, 100000]),
            "S2__medium_vs_low__ridge__mse_skill__cost_off": ("S2-medium__ridge__off", "S2-low__ridge__off", "forecast_mse_skill_vs_zero", "off", "medium_ge_low", ge_low, "synthetic_validation", [90000, 100000]),
            "S2__medium_vs_low__ridge__normalized_regret__cost_on": ("S2-medium__ridge__on", "S2-low__ridge__on", "normalized_action_regret", "on", "medium_le_low", le_low, "synthetic_validation", [90000, 100000]),
            "S2__medium_vs_low__ridge__utility__cost_on": ("S2-medium__ridge__on", "S2-low__ridge__on", "s2_timing_net_utility_delta", "on", "medium_ge_low", ge_low, "synthetic_validation", [90000, 100000]),
            "S2__medium_vs_low__ridge__agreement__cost_on": ("S2-medium__ridge__on", "S2-low__ridge__on", "feasible_action_agreement", "on", "medium_ge_low", ge_low, "synthetic_validation", [90000, 100000]),
            "S2__medium_vs_low__logistic__log_loss__cost_off": ("S2-medium__logistic__off", "S2-low__logistic__off", "log_loss", "off", "medium_le_low", le_low, "synthetic_validation", [90000, 100000]),
            "S3__injected_vs_control__ridge__mse_skill_did__cost_off": ("S3-injected__ridge__off", "S3-control__ridge__off", "s3_mse_skill_difference_in_differences", "off", "positive", s3_gate, "s3_validation", [104528, 139568]),
            "S3__injected_vs_control__ridge__utility__cost_on": ("S3-injected__ridge__on", "S3-control__ridge__on", "s3_timing_net_utility_difference_in_differences", "on", "positive", s3_gate, "s3_validation", [104528, 139568]),
        }
        self.assertEqual({row["comparison_id"] for row in rows}, set(expected))
        for row in rows:
            with self.subTest(comparison_id=row["comparison_id"]):
                actual = (
                    row["candidate_id"], row["baseline_id"], row["metric"],
                    row["cost_mode"], row["direction"], row["gate"],
                    row["support_id"], row["support_range"],
                )
                self.assertEqual(actual, expected[row["comparison_id"]])
                self.assertEqual(
                    row["support_range_semantics"],
                    "zero-based [start,end) right-exclusive; end excluded",
                )
        action_ids = {
            "S0__ridge__utility_vs_hold__cost_on",
            "S0__persistence__utility_vs_hold__cost_on",
            "S1__ridge__utility_vs_hold__cost_on",
            "S2__high_vs_medium__ridge__normalized_regret__cost_on",
            "S2__high_vs_medium__ridge__utility__cost_on",
            "S2__high_vs_medium__ridge__agreement__cost_on",
            "S2__medium_vs_low__ridge__normalized_regret__cost_on",
            "S2__medium_vs_low__ridge__utility__cost_on",
            "S2__medium_vs_low__ridge__agreement__cost_on",
            "S3__injected_vs_control__ridge__utility__cost_on",
        }
        replay = "resample stored canonical action block record indices and recompute declared means/sums/ratios/DiD; never replay policy state over a resampled or nonchronological sequence"
        for row in rows:
            if row["comparison_id"] in action_ids:
                self.assertEqual(row["action_bootstrap_replay_policy"], replay)
            else:
                self.assertNotIn("action_bootstrap_replay_policy", row)

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

    def test_dgp_and_action_bootstrap_lifecycle_are_pinned(self) -> None:
        manifest = _read_manifest()
        synthetic = manifest["synthetic_contract"]
        self.assertEqual(
            synthetic["random_generator"],
            "np.random.default_rng(seed + 100).standard_normal",
        )
        self.assertEqual(
            synthetic["random_distribution"],
            "z0, every xi entry, every noise_features entry, and every epsilon entry are mutually independent iid standard normal N(0,1) draws",
        )
        self.assertEqual(
            synthetic["random_independence"],
            "z0, xi, noise_features, and epsilon are mutually independent; entries within each vector or matrix are iid",
        )
        self.assertEqual(synthetic["random_dtype"], "float64")
        self.assertEqual(
            synthetic["availability"]["start_sampling_api"],
            "rng=np.random.default_rng(seed+50000+source_offset); relative=rng.choice(119998-512,size=40,replace=False,shuffle=True); starts=np.asarray(relative,dtype=np.int64)+512",
        )
        self.assertIn("never sort", synthetic["availability"]["start_order_policy"])
        self.assertIn("union", synthetic["availability"]["interval_union_policy"])
        bootstrap = manifest["common"]["gates"]["block_bootstrap"]
        self.assertEqual(
            bootstrap["action_bootstrap_replay_policy"],
            "resample stored canonical action block record indices and recompute declared means/sums/ratios/DiD; never replay policy state over a resampled or nonchronological sequence",
        )
        self.assertIn("selected_delta", bootstrap["action_primitive_record_fields"])
        self.assertIn("selected_position", bootstrap["action_primitive_record_fields"])
        self.assertIn("previous_position", bootstrap["action_primitive_record_fields"])
        self.assertIn("turnover", bootstrap["action_primitive_record_fields"])
        self.assertIn("active_indicator", bootstrap["action_primitive_record_fields"])
        self.assertIn("common_mask", bootstrap["action_primitive_record_fields"])
        self.assertEqual(
            bootstrap["action_primitive_record_fields"],
            bootstrap["action_primitive_schema"]["record_fields"],
        )
        self.assertEqual(
            bootstrap["rng_lifecycle"],
            "for each unit/support/seed/L create np.random.default_rng(derived_seed) exactly once, then draw all replicate starts in replicate order b=0..1999; do not reinitialize per replicate, arm, or comparison",
        )
        self.assertEqual(bootstrap["replicate_order"], "b=0,1,...,1999 in ascending order")
        self.assertEqual(bootstrap["quantile_method"], "np.quantile(values, q, method='linear')")
        self.assertIn("denominator is zero or nonpositive", bootstrap["denominator_policy"])
        self.assertEqual(
            bootstrap["mbb_draw_api"],
            "starts=rng.integers(low=0,high=n-L+1,size=ceil(n/L),endpoint=False,dtype=np.int64)",
        )
        self.assertEqual(
            bootstrap["mbb_index_materialization"],
            "indices = starts[:,None] + np.arange(L,dtype=np.int64); flatten in C order and take the first n indices",
        )
        schema = bootstrap["action_primitive_schema"]
        self.assertEqual(schema["index_dtype"], "int64")
        self.assertEqual(schema["mask_dtype"], "bool")
        self.assertEqual(schema["value_dtype"], "float64")
        self.assertEqual(
            schema["index_fields"],
            ["primitive_index", "decision_index", "fill_index", "end_index"],
        )
        self.assertIn("cost_contract_hash", schema["arm_id_fields"])
        self.assertIn("action_primitive_payload_sha256", schema["hash_fields"])
        self.assertEqual(
            schema["external_schema_sha256"],
            "d0520b3dbc3c444e2efe5a55e175e96b662f97fb404d901ea51e1c32e5bb9955",
        )
        self.assertIn("float64", schema["canonical_serialization"]["value_encoding"])
        self.assertIn("C order", schema["canonical_serialization"]["shape"])
        self.assertIn("full-grid", schema["canonical_serialization"]["row_order"])
        self.assertIn("payload", schema["hash_scopes"]["action_primitive_payload_sha256"])

    def test_ranges_history_and_v4_runtime_policy_are_pinned(self) -> None:
        manifest = _read_manifest()
        common = manifest["common"]
        self.assertEqual(
            common["index_range_contract"],
            "all numeric split_range, support_range, fit_prefix_range, prediction_range, fit_raw_range, prediction_raw_range, and body index ranges are zero-based [start,end) right-exclusive; end is excluded and the origin row is never admitted to its fit prefix",
        )
        self.assertIn("eligible train-mask row count", common["oof"]["min_history_rule"])
        self.assertIn("if count < 16384", common["oof"]["min_history_rule"])
        self.assertIn("origin 90000 is excluded", common["oof"]["primary_inferential_support"]["range_semantics"])
        v4 = common["v4_load_contract"]
        self.assertIn("all explicit feature, returns, availability, and frozen metadata paths", v4["body_validation_policy"])
        self.assertIn("content/schema/cache-tag/row-count", v4["source_provenance_difference_policy"])
        self.assertIn("blocks S3", v4["missing_unknown_mismatch_policy"])
        self.assertTrue(v4["promotion_disposition_required"])
        self.assertEqual(
            v4["runtime_validation_entrypoint"],
            "unidream.experiments.runtime.validate_p1_v4_runtime_inputs",
        )
        self.assertEqual(
            v4["runtime_body_validator_entrypoint"],
            "unidream.experiments.runtime.validate_v4_runtime_inputs",
        )
        self.assertIn("load_fixed_manifest first", v4["runtime_authentication_policy"])
        self.assertTrue(v4["runtime_validation_required_before_fit_or_score"])
        self.assertEqual(
            v4["runtime_disposition_fields"],
            ["status", "reason", "body_match", "source_provenance_match"],
        )

    def test_exact_availability_coordinates_and_window_edges_are_pinned(self) -> None:
        manifest = _read_manifest()
        availability = manifest["common"]["availability"]
        self.assertIn("t >= 63", availability["context_window_rule"])
        self.assertIn("[t-63,t]", availability["context_window_rule"])
        self.assertIn("only X[t]", availability["context_window_rule"])
        self.assertIn("t+h-1->t+h", availability["target_window_rule"])
        self.assertIn("t+h->t+h+1 is not required", availability["target_window_rule"])
        self.assertIn("target_end=t+h+1 is exclusive", availability["target_window_rule"])

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
        self.assertIn(
            "every seed on the identical scored mask has mean realized same-state clairvoyant net utility/value strictly greater than Ridge mean realized net utility/value",
            row["gate"],
        )
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
        self.assertEqual(manifest["manifest_sha256"], REGISTERED_MANIFEST_SHA256)
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
