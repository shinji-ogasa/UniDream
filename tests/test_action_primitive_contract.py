"""Contract-only tests for the staged P1 action primitive boundary."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from unidream.experiments.action_primitives import (
    ACTION_PRIMITIVE_RECORD_FIELDS,
    ActionPrimitiveContractError,
    ActionPrimitiveImplementationBlocked,
    action_primitive_content_sha256,
    action_primitive_payload_sha256,
    build_action_primitive_grid,
    canonical_action_primitive_schema_sha256,
    require_action_primitive_implementation,
    run_action_primitive_mbb,
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

    def test_missing_common_mask_and_producer_are_blocked(self) -> None:
        broken = _record(0)
        del broken["common_mask"]
        with self.assertRaisesRegex(ActionPrimitiveContractError, "missing=common_mask"):
            action_primitive_content_sha256([broken])
        with self.assertRaises(ActionPrimitiveImplementationBlocked):
            require_action_primitive_implementation()
        with self.assertRaises(ActionPrimitiveImplementationBlocked):
            build_action_primitive_grid([broken])
        with self.assertRaises(ActionPrimitiveImplementationBlocked):
            run_action_primitive_mbb([broken], block_length=16)

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
        with self.assertRaisesRegex(ActionPrimitiveContractError, "payload SHA-256 is required"):
            validate_action_primitive_records(
                records,
                schema=schema,
                expected_schema_sha256=schema_sha256,
                expected_content_sha256=content_sha256,
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
