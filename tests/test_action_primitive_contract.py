"""Contract-only tests for the staged P1 action primitive boundary."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from unidream.eval.action_execution import ActionExecutionContract
from unidream.experiments.action_primitives import (
    ACTION_PRIMITIVE_COST_CONTRACT_SHA256,
    ACTION_PRIMITIVE_RECORD_FIELDS,
    ActionPrimitiveContractError,
    ActionPrimitiveImplementationBlocked,
    action_primitive_content_sha256,
    action_primitive_payload_sha256,
    build_action_primitive_grid,
    canonical_action_primitive_schema_sha256,
    produce_action_primitive_grid,
    require_action_primitive_implementation,
    run_action_primitive_mbb,
    validate_action_primitive_semantics,
    validate_action_primitive_records,
)


ROOT = Path(__file__).parents[1]


def _record(index: int, *, common: bool = True) -> dict[str, object]:
    values: dict[str, object] = {
        "primitive_index": index,
        "decision_index": index * 4,
        "fill_index": index * 4 + 1,
        "end_index": index * 4 + 4,
        "previous_position": 1.0,
        "selected_delta": 0.0,
        "selected_position": 1.0,
        "candidate_utility": float("nan") if not common else 0.1,
        "benchmark_hold_utility": 0.0,
        "same_state_local_hold_utility": 0.0,
        "clairvoyant_utility": 0.2,
        "regret": 0.1,
        "opportunity": 0.2,
        "agreement": 1.0,
        "turnover": 0.0,
        "active_indicator": 0.0,
        "origin_eligible_mask": common,
        "forecast_finite_mask": common,
        "fill_complete_mask": common,
        "outcome_complete_mask": common,
        "scored_action_mask": common,
        "common_mask": common,
        "scenario_id": "S1",
        "seed": 20260830,
        "split_id": "validation",
        "support_id": "synthetic_validation",
        "model_id": "ridge",
        "cost_mode": "on",
        "cost_contract_hash": "a" * 64,
    }
    assert tuple(values) == ACTION_PRIMITIVE_RECORD_FIELDS
    return values


class ActionPrimitiveContractTests(unittest.TestCase):
    def test_external_schema_is_pinned_and_record_fields_match(self) -> None:
        path = ROOT / "docs" / "experiments" / "action_primitive_schema.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(schema["record_fields"], list(ACTION_PRIMITIVE_RECORD_FIELDS))
        self.assertEqual(
            canonical_action_primitive_schema_sha256(schema),
            "d0520b3dbc3c444e2efe5a55e175e96b662f97fb404d901ea51e1c32e5bb9955",
        )

    def test_hashes_cover_full_grid_masks_and_canonical_nan(self) -> None:
        records = [_record(0, common=False), _record(1)]
        schema = json.loads(
            (ROOT / "docs" / "experiments" / "action_primitive_schema.json").read_text(
                encoding="utf-8"
            )
        )
        schema_sha256 = "d0520b3dbc3c444e2efe5a55e175e96b662f97fb404d901ea51e1c32e5bb9955"
        content_sha256 = action_primitive_content_sha256(records)
        payload_sha256 = action_primitive_payload_sha256(
            records,
            schema_sha256=schema_sha256,
            content_sha256=content_sha256,
        )
        result = validate_action_primitive_records(
            records,
            schema=schema,
            expected_schema_sha256=schema_sha256,
            expected_content_sha256=content_sha256,
            expected_payload_sha256=payload_sha256,
        )
        self.assertEqual(result["record_count"], 2)
        self.assertIn("common_mask", result["record_fields"])
        changed = copy.deepcopy(records)
        changed[0]["common_mask"] = True
        self.assertNotEqual(
            action_primitive_content_sha256(records),
            action_primitive_content_sha256(changed),
        )
        reordered = [records[1], records[0]]
        with self.assertRaisesRegex(ActionPrimitiveContractError, "primitive_index"):
            action_primitive_content_sha256(reordered)

    def test_missing_common_mask_and_bootstrap_are_blocked(self) -> None:
        broken = _record(0)
        del broken["common_mask"]
        with self.assertRaisesRegex(ActionPrimitiveContractError, "missing=common_mask"):
            action_primitive_content_sha256([broken])
        with self.assertRaises(ActionPrimitiveImplementationBlocked):
            require_action_primitive_implementation()
        with self.assertRaises(ActionPrimitiveImplementationBlocked):
            run_action_primitive_mbb([broken], block_length=16)

    @staticmethod
    def _fixture(*, cost_mode: str = "on") -> dict[str, object]:
        contract = ActionExecutionContract.canonical()
        if cost_mode == "off":
            contract = ActionExecutionContract.from_config(
                json.loads(
                    (ROOT / "docs" / "experiments" / "action_execution_contract_cost_off.json").read_text(
                        encoding="utf-8"
                    )
                ),
                require_canonical=False,
            )
        n_bars = 17
        returns = np.asarray(
            [
                0.001,
                -0.0005,
                0.002,
                0.001,
                -0.001,
                0.002,
                0.003,
                -0.002,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
            ],
            dtype=np.float64,
        )
        scores = np.full(n_bars, np.nan, dtype=np.float64)
        scores[[0, 4, 8, 12]] = (-1.0, 0.0, 1.0, 0.0)
        decision_eligible = np.ones(n_bars, dtype=bool)
        score_eligible = np.ones(n_bars, dtype=bool)
        # The second scheduled block has a delayed fill gap.  Its physical
        # row must remain in the artifact, with false masks and no state move.
        score_eligible[5] = False
        return produce_action_primitive_grid(
            returns=returns,
            decision_block_scores=scores,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
            scenario_id="S1",
            seed=20260830,
            split_id="validation",
            support_id="synthetic_validation",
            model_id="ridge",
            cost_mode=cost_mode,
            cost_contract_hash=contract.contract_hash,
        )

    @staticmethod
    def _fixture_returns() -> np.ndarray:
        return np.asarray(
            [
                0.001,
                -0.0005,
                0.002,
                0.001,
                -0.001,
                0.002,
                0.003,
                -0.002,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
                0.001,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _refresh_hashes(artifact: dict[str, object]) -> None:
        header = artifact["header"]
        records = artifact["records"]
        assert isinstance(header, dict)
        assert isinstance(records, list)
        content = action_primitive_content_sha256(records)
        payload = action_primitive_payload_sha256(
            records,
            schema_sha256=header["action_primitive_schema_sha256"],
            content_sha256=content,
        )
        header["action_primitive_content_sha256"] = content
        header["action_primitive_payload_sha256"] = payload
        artifact["action_primitive_content_sha256"] = content
        artifact["action_primitive_payload_sha256"] = payload

    def test_deterministic_producer_preserves_grid_masks_state_and_cost_off(self) -> None:
        artifact = self._fixture()
        header = artifact["header"]
        records = artifact["records"]
        self.assertEqual(header["record_count"], 4)
        self.assertEqual(header["support_start"], 0)
        self.assertEqual(header["support_range"], [0, 17])
        self.assertEqual(
            header["source_role"],
            "deterministic_fixture_realized_return_inputs",
        )
        self.assertEqual([row["decision_index"] for row in records], [0, 4, 8, 12])
        self.assertEqual([row["fill_index"] for row in records], [1, 5, 9, 13])
        self.assertEqual([row["end_index"] for row in records], [4, 8, 12, 16])
        self.assertEqual(records[0]["selected_delta"], -0.08)
        self.assertAlmostEqual(records[0]["selected_position"], 0.92)
        self.assertAlmostEqual(records[0]["turnover"], 0.08)
        self.assertEqual(records[1]["fill_complete_mask"], False)
        self.assertEqual(records[1]["outcome_complete_mask"], False)
        self.assertEqual(records[1]["scored_action_mask"], False)
        self.assertEqual(records[1]["common_mask"], False)
        self.assertEqual(records[1]["selected_delta"], 0.0)
        self.assertEqual(records[1]["previous_position"], records[0]["selected_position"])
        self.assertTrue(np.isnan(records[1]["candidate_utility"]))
        self.assertEqual(records[2]["previous_position"], records[1]["selected_position"])
        self.assertEqual(header["schedule"]["target_rule"], "decision t -> fill t+1 -> returns[t+1:t+5]")
        self.assertEqual(header["contract_hash"], ACTION_PRIMITIVE_COST_CONTRACT_SHA256["on"])
        result = validate_action_primitive_semantics(
            artifact,
            realized_returns=self._fixture_returns(),
        )
        self.assertEqual(result["semantic_validation_status"], "passed")
        with self.assertRaisesRegex(ActionPrimitiveContractError, "production validation"):
            validate_action_primitive_semantics(
                artifact,
                realized_returns=self._fixture_returns(),
                require_production=True,
            )

        off = self._fixture(cost_mode="off")
        self.assertEqual(
            off["header"]["contract_hash"],
            ACTION_PRIMITIVE_COST_CONTRACT_SHA256["off"],
        )
        self.assertEqual(off["header"]["cost"]["transition_cost_rate"], 0.0)
        self.assertEqual(
            validate_action_primitive_semantics(
                off,
                realized_returns=self._fixture_returns(),
            )["cost_mode"],
            "off",
        )

    def test_direct_stored_actions_and_alias_builder_are_deterministic(self) -> None:
        contract = ActionExecutionContract.canonical()
        returns = np.full(13, 0.001, dtype=np.float64)
        artifact = build_action_primitive_grid(
            returns=returns,
            selected_deltas=np.asarray([-0.08, 0.0, 0.08], dtype=np.float64),
            decision_eligible=np.ones(13, dtype=bool),
            score_eligible=np.ones(13, dtype=bool),
            forecast_finite_mask=np.ones(3, dtype=bool),
            scenario_id="S0",
            seed=1,
            split_id="validation",
            support_id="synthetic_validation",
            model_id="persistence",
            cost_mode="on",
            cost_contract_hash=contract.contract_hash,
        )
        self.assertEqual([r["selected_delta"] for r in artifact["records"]], [-0.08, 0.0, 0.08])
        self.assertEqual(
            validate_action_primitive_semantics(
                artifact,
                realized_returns=np.full(13, 0.001, dtype=np.float64),
            )["record_count"],
            3,
        )

    def test_semantic_validator_rejects_hash_repaired_state_mask_nan_and_cost_forgery(self) -> None:
        artifact = self._fixture()
        tampered = copy.deepcopy(artifact)
        tampered["records"][0]["turnover"] = 0.0
        self._refresh_hashes(tampered)
        with self.assertRaisesRegex(ActionPrimitiveContractError, "turnover"):
            validate_action_primitive_semantics(
                tampered,
                realized_returns=self._fixture_returns(),
            )

        tampered = copy.deepcopy(artifact)
        tampered["records"][1]["common_mask"] = True
        self._refresh_hashes(tampered)
        with self.assertRaisesRegex(ActionPrimitiveContractError, "common_mask"):
            validate_action_primitive_semantics(
                tampered,
                realized_returns=self._fixture_returns(),
            )

        tampered = copy.deepcopy(artifact)
        tampered["records"][0]["selected_delta"] = float("nan")
        self._refresh_hashes(tampered)
        with self.assertRaises(ActionPrimitiveContractError):
            validate_action_primitive_semantics(
                tampered,
                realized_returns=self._fixture_returns(),
            )

        tampered = copy.deepcopy(artifact)
        tampered["records"][0]["model_id"] = "forged-model"
        self._refresh_hashes(tampered)
        with self.assertRaisesRegex(ActionPrimitiveContractError, "model_id"):
            validate_action_primitive_semantics(
                tampered,
                realized_returns=self._fixture_returns(),
            )

        with self.assertRaisesRegex(ActionPrimitiveContractError, "cost_contract_hash"):
            self._fixture_with_cost_hash(ACTION_PRIMITIVE_COST_CONTRACT_SHA256["off"])

    def _fixture_with_cost_hash(self, cost_hash: str) -> None:
        contract = ActionExecutionContract.canonical()
        produce_action_primitive_grid(
            returns=np.full(5, 0.001, dtype=np.float64),
            decision_block_scores=np.asarray([0.0, np.nan, np.nan, np.nan, np.nan]),
            decision_eligible=np.ones(5, dtype=bool),
            score_eligible=np.ones(5, dtype=bool),
            scenario_id="S",
            seed=1,
            split_id="v",
            support_id="sp",
            model_id="m",
            cost_mode="on",
            cost_contract_hash=cost_hash,
            contract=contract,
        )

    def test_grid_order_requires_one_global_start_and_four_bar_spacing(self) -> None:
        for starts in ((-4, 0), (1, 6), (0, 4, 100)):
            with self.subTest(starts=starts):
                records = [_record(index) for index in range(len(starts))]
                for record, decision_index in zip(records, starts):
                    record["decision_index"] = decision_index
                    record["fill_index"] = decision_index + 1
                    record["end_index"] = decision_index + 4
                with self.assertRaisesRegex(ActionPrimitiveContractError, "global support start"):
                    action_primitive_content_sha256(records)

        shifted = [_record(index) for index in range(2)]
        for record, decision_index in zip(shifted, (90_000, 90_004)):
            record["decision_index"] = decision_index
            record["fill_index"] = decision_index + 1
            record["end_index"] = decision_index + 4
        self.assertEqual(len(action_primitive_content_sha256(shifted)), 64)

    def test_production_artifact_binds_registered_global_support(self) -> None:
        contract = ActionExecutionContract.canonical()
        n_bars = 10_000
        scores = np.full(n_bars, np.nan, dtype=np.float64)
        scores[np.arange(0, n_bars - 4, 4, dtype=np.int64)] = 0.001
        artifact = produce_action_primitive_grid(
            returns=np.full(n_bars, 0.0001, dtype=np.float64),
            support_start=90_000,
            decision_block_scores=scores,
            decision_eligible=np.ones(n_bars, dtype=bool),
            score_eligible=np.ones(n_bars, dtype=bool),
            scenario_id="S1",
            seed=20260830,
            split_id="validation",
            support_id="synthetic_validation",
            model_id="ridge",
            cost_mode="on",
            cost_contract_hash=contract.contract_hash,
            require_production=True,
        )
        self.assertEqual(artifact["header"]["source_role"], "validated_stored_action_inputs")
        self.assertEqual(artifact["records"][0]["decision_index"], 90_000)
        self.assertEqual(artifact["records"][-1]["decision_index"], 99_992)
        self.assertEqual(
            validate_action_primitive_semantics(
                artifact,
                realized_returns=np.full(n_bars, 0.0001, dtype=np.float64),
                require_production=True,
            )["semantic_validation_status"],
            "passed",
        )
        with self.assertRaisesRegex(ActionPrimitiveContractError, "preregistered support range"):
            produce_action_primitive_grid(
                returns=np.full(17, 0.0001, dtype=np.float64),
                support_start=0,
                decision_block_scores=np.zeros(4, dtype=np.float64),
                decision_eligible=np.ones(17, dtype=bool),
                score_eligible=np.ones(17, dtype=bool),
                scenario_id="S1",
                seed=20260830,
                split_id="validation",
                support_id="synthetic_validation",
                model_id="ridge",
                cost_mode="on",
                cost_contract_hash=contract.contract_hash,
                require_production=True,
            )

    def test_validator_is_fail_closed_for_schema_empty_rows_and_omitted_hashes(self) -> None:
        records = [_record(0)]
        schema = json.loads(
            (ROOT / "docs" / "experiments" / "action_primitive_schema.json").read_text(
                encoding="utf-8"
            )
        )
        schema_sha256 = "d0520b3dbc3c444e2efe5a55e175e96b662f97fb404d901ea51e1c32e5bb9955"
        content_sha256 = action_primitive_content_sha256(records)
        payload_sha256 = action_primitive_payload_sha256(
            records,
            schema_sha256=schema_sha256,
            content_sha256=content_sha256,
        )
        expected_hashes = {
            "expected_schema_sha256": schema_sha256,
            "expected_content_sha256": content_sha256,
            "expected_payload_sha256": payload_sha256,
        }
        for missing_field in expected_hashes:
            with self.subTest(missing_field=missing_field):
                omitted = dict(expected_hashes)
                omitted.pop(missing_field)
                with self.assertRaisesRegex(ActionPrimitiveContractError, "is required"):
                    validate_action_primitive_records(
                        records,
                        schema=schema,
                        **omitted,
                    )
        with self.assertRaisesRegex(ActionPrimitiveContractError, "schema mapping is required"):
            validate_action_primitive_records(
                records,
                expected_schema_sha256=schema_sha256,
                expected_content_sha256=content_sha256,
                expected_payload_sha256=payload_sha256,
            )
        with self.assertRaisesRegex(ActionPrimitiveContractError, "independently pinned"):
            validate_action_primitive_records(
                records,
                schema=schema,
                expected_schema_sha256="a" * 64,
                expected_content_sha256=content_sha256,
                expected_payload_sha256=payload_sha256,
            )
        forged_schema = copy.deepcopy(schema)
        forged_schema["forged"] = True
        with self.assertRaisesRegex(ActionPrimitiveContractError, "external schema SHA-256 mismatch"):
            validate_action_primitive_records(
                records,
                schema=forged_schema,
                expected_schema_sha256=schema_sha256,
                expected_content_sha256=content_sha256,
                expected_payload_sha256=payload_sha256,
            )
        with self.assertRaisesRegex(ActionPrimitiveContractError, "expected external schema SHA-256 is required"):
            validate_action_primitive_records(records, schema=schema)
        with self.assertRaisesRegex(ActionPrimitiveContractError, "at least one full-grid row"):
            validate_action_primitive_records(
                [],
                schema=schema,
                expected_schema_sha256=schema_sha256,
                expected_content_sha256=content_sha256,
                expected_payload_sha256=payload_sha256,
            )


if __name__ == "__main__":
    unittest.main()
