"""Authenticated forecast-to-action adapter contract tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.test_p1_validation_forecast_artifact import _fixture_artifact
from unidream.eval.action_execution import ActionExecutionContract, complete_decision_starts
from unidream.experiments import p1_validation_forecast as forecast
from unidream.experiments.action_primitives import (
    ACTION_PRIMITIVE_HASH_FIELDS,
    ActionPrimitiveContractError,
    expected_authenticated_action_metadata,
    produce_authenticated_action_primitive_grid,
    validate_action_primitive_semantics,
)


class P1AuthenticatedActionAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = forecast.authenticate_p1_forecast_contract()

    def _load(self, root: Path):
        payload = _fixture_artifact(self.contract)
        metadata = forecast.expected_metadata_for_arm(
            self.contract,
            "S1",
            "known_high_snr_dgp",
            20260830,
        )
        path = root / "forecast.json"
        digest = forecast.save_p1_forecast_artifact(
            path,
            payload,
            expected_metadata=metadata,
        )
        return forecast.load_p1_forecast_artifact(
            path,
            expected_file_sha256=digest,
            expected_metadata=metadata,
        )

    def test_model_selection_is_explicit_and_every_action_arm_is_registry_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded = self._load(Path(directory))
            with self.assertRaisesRegex(forecast.P1ForecastError, "ambiguous"):
                _ = loaded.action_source
            with self.assertRaisesRegex(forecast.P1ForecastError, "model_id"):
                loaded.as_action_source("logistic")

            action_hashes: set[str] = set()
            for model_id in forecast.P1_ACTION_MODEL_IDS:
                with self.subTest(model_id=model_id):
                    source = loaded.as_action_source(model_id)
                    self.assertEqual(source.model_id, model_id)
                    paired = np.ones(
                        len(
                            complete_decision_starts(
                                len(source.timestamps),
                                ActionExecutionContract.canonical(),
                            )
                        ),
                        dtype=np.bool_,
                    )
                    expected = expected_authenticated_action_metadata(
                        source,
                        cost_mode="on",
                        paired_common_mask=paired,
                    )
                    artifact = produce_authenticated_action_primitive_grid(
                        action_source=source,
                        cost_mode="on",
                        paired_common_mask=paired,
                        expected_metadata=expected,
                    )
                    header = artifact["header"]
                    self.assertEqual(header["model_id"], model_id)
                    self.assertEqual(header["trial_id"], expected["trial_id"])
                    self.assertEqual(
                        header["source_binding"]["capability_binding_sha256"],
                        source.binding_sha256,
                    )
                    self.assertEqual(
                        set(ACTION_PRIMITIVE_HASH_FIELDS),
                        set(artifact) - {"header", "records"},
                    )
                    self.assertTrue(all(row["common_mask"] for row in artifact["records"]))
                    action_hashes.add(artifact["action_primitive_content_sha256"])
            self.assertEqual(len(action_hashes), len(forecast.P1_ACTION_MODEL_IDS))

    def test_forged_capability_metadata_raw_inputs_and_self_hashes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            loaded = self._load(Path(directory))
            source = loaded.as_action_source("ridge")
            paired = np.ones(
                len(
                    complete_decision_starts(
                        len(source.timestamps),
                        ActionExecutionContract.canonical(),
                    )
                ),
                dtype=np.bool_,
            )
            expected = dict(
                expected_authenticated_action_metadata(
                    source,
                    cost_mode="on",
                    paired_common_mask=paired,
                )
            )
            artifact = produce_authenticated_action_primitive_grid(
                action_source=source,
                cost_mode="on",
                paired_common_mask=paired,
                expected_metadata=expected,
            )
            with self.assertRaisesRegex(ActionPrimitiveContractError, "sealed"):
                validate_action_primitive_semantics(
                    artifact,
                    expected_metadata=expected,
                    expected_source_binding=artifact["header"]["source_binding"],
                    realized_returns=source.realized_returns,
                    decision_block_scores=source.forecast_h4,
                    decision_eligible=source.origin_mask,
                    score_eligible=source.bar_available,
                    expected_common_mask=paired,
                    require_production=False,
                )
            forged = replace(source)
            with self.assertRaisesRegex(ActionPrimitiveContractError, "authenticated"):
                produce_authenticated_action_primitive_grid(
                    action_source=forged,
                    cost_mode="on",
                    paired_common_mask=paired,
                    expected_metadata=expected,
                )
            wrong = dict(expected)
            wrong["model_id"] = "zero_return"
            with self.assertRaisesRegex(ActionPrimitiveContractError, "paired_common_mask"):
                produce_authenticated_action_primitive_grid(
                    action_source=source,
                    cost_mode="on",
                    expected_metadata=expected,
                )
            with self.assertRaisesRegex(ActionPrimitiveContractError, "expected_metadata"):
                produce_authenticated_action_primitive_grid(
                    action_source=source,
                    cost_mode="on",
                    paired_common_mask=paired,
                    expected_metadata=wrong,
                )
            with self.assertRaisesRegex(ActionPrimitiveContractError, "own action output"):
                produce_authenticated_action_primitive_grid(
                    action_source=source,
                    cost_mode="on",
                    paired_common_mask=paired,
                    expected_metadata=expected,
                    expected_output_hashes={field: "0" * 64 for field in ACTION_PRIMITIVE_HASH_FIELDS},
                )
            with self.assertRaisesRegex(ActionPrimitiveContractError, "raw v4 runtime"):
                produce_authenticated_action_primitive_grid(
                    action_source=source,
                    cost_mode="on",
                    paired_common_mask=paired,
                    expected_metadata=expected,
                    returns=source.realized_returns,
                )


if __name__ == "__main__":
    unittest.main()
