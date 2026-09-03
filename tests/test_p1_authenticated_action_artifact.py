"""Identity-sealed action artifact and typed MBB-input boundary tests."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

import numpy as np

import unidream.experiments.p1_action_artifact as action_artifact
import tests.test_p1_action_artifact as base_action_tests
from unidream.eval.action_execution import ActionExecutionContract
from unidream.experiments.p1_mbb import (
    bootstrap_p1_action_metric,
    build_p1_mbb_index_artifact,
    load_p1_mbb_index_artifact,
    p1_mask_sha256,
    save_p1_mbb_index_artifact,
)


class AuthenticatedP1ActionArtifactTests(unittest.TestCase):
    @classmethod
    def _production_case(cls):
        artifact = base_action_tests.P1ActionArtifactTests._artifact()
        returns, scores, decision, score = base_action_tests.P1ActionArtifactTests._inputs()
        header = artifact["header"]
        arm = header["arm_metadata"]
        source_hashes = {
            field_name: "a" * 64
            for field_name in action_artifact.P1_ACTION_SOURCE_BINDING_HASH_FIELDS
            if field_name != "source_body_sha256"
        }
        source = SimpleNamespace(
            scenario_id=arm["scenario_id"],
            arm=arm["scenario_id"],
            seed=arm["seed"],
            model_id=arm["model_id"],
            split_id=arm["split_id"],
            support_id=arm["support_id"],
            support_range=tuple(header["support_range"]),
            fit_origin=0,
            prereg_results_observed=False,
            validation_results_observed=True,
            outer_results_observed=False,
            validation_status="passed",
            promotion_allowed=True,
            binding_sha256="b" * 64,
            source_hashes=source_hashes,
            realized_returns=returns,
            forecast_h4=scores,
            origin_mask=decision,
            bar_available=score,
        )
        source_binding = {
            "schema_id": "p1-forecast-action-source-binding-v1",
            "source_role": "authenticated_p1_forecast_action_source",
            "scenario_id": arm["scenario_id"],
            "arm": source.arm,
            "seed": arm["seed"],
            "model_id": arm["model_id"],
            "split_id": arm["split_id"],
            "support_id": arm["support_id"],
            "support_range": list(header["support_range"]),
            "fit_origin": 0,
            "prereg_results_observed": False,
            "validation_results_observed": True,
            "outer_results_observed": False,
            "validation_status": "passed",
            "promotion_allowed": True,
            "capability_binding_sha256": source.binding_sha256,
            "source_hashes": source_hashes,
        }
        header.update(
            {
                "source_role": "validated_stored_action_inputs",
                "action_primitive_producer_status": "validated_production_input",
                "metric_source": "recomputed_from_realized_returns",
                "trial_id": "fixture-trial",
                "source_binding": source_binding,
                "source_binding_sha256": action_artifact._source_binding_sha256(
                    source_binding
                ),
                "paired_common_mask_sha256": hashlib.sha256(
                    np.ones(4, dtype=np.bool_).tobytes(order="C")
                ).hexdigest(),
            }
        )
        metadata = {
            field_name: arm[field_name] if field_name in arm else header[field_name]
            for field_name in (
                *action_artifact.ACTION_PRIMITIVE_ARM_FIELDS,
                "support_start",
                "support_range",
                "trial_id",
                "source_binding_sha256",
                "paired_common_mask_sha256",
            )
        }
        expected_hashes = {
            field_name: artifact[field_name]
            for field_name in action_artifact.ACTION_PRIMITIVE_HASH_FIELDS
        }
        return artifact, source, source_binding, metadata, expected_hashes, (returns, scores, decision, score)

    def _save_authenticated(self, directory: str):
        (
            artifact,
            source,
            source_binding,
            metadata,
            expected_hashes,
            arrays,
        ) = self._production_case()
        path = Path(directory) / "action.json"
        kwargs = {
            "expected_metadata": metadata,
            "expected_hashes": expected_hashes,
            "expected_source_binding": source_binding,
            "authenticated_action_source": source,
            "realized_returns": arrays[0],
            "decision_block_scores": arrays[1],
            "decision_eligible": arrays[2],
            "score_eligible": arrays[3],
            "expected_common_mask": np.ones(4, dtype=np.bool_),
        }
        # This branch predates the action-source adapter in action_primitives;
        # the test supplies the adapter's successful semantic result while
        # exercising this module's independent source/hash/capability checks.
        with patch.object(
            action_artifact,
            "_require_authenticated_forecast_source",
            return_value=source,
        ), patch.object(
            action_artifact,
            "validate_action_primitive_semantics",
            return_value={"semantic_validation_status": "passed"},
        ):
            digest = action_artifact.save_p1_action_artifact(path, artifact, **kwargs)
            loaded = action_artifact.load_p1_action_artifact(
                path,
                expected_file_sha256=digest,
                **kwargs,
            )
        return path, digest, loaded, kwargs

    def test_production_load_is_identity_sealed_and_exposes_field_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest, loaded, _ = self._save_authenticated(directory)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(loaded.is_authenticated)
            self.assertIs(
                action_artifact.require_authenticated_loaded_action_artifact(loaded),
                loaded,
            )
            mbb_input = loaded.as_mbb_input()
            selected = mbb_input.select_metric("policy_utility_delta")
            self.assertEqual(
                set(selected.metric_values),
                {"candidate_utility", "benchmark_hold_utility"},
            )
            expected_mask = np.asarray(
                [
                    row["outcome_complete_mask"] and row["common_mask"]
                    for row in loaded.artifact["records"]
                ],
                dtype=np.bool_,
            )
            np.testing.assert_array_equal(selected.effective_mask, expected_mask)
            self.assertFalse(selected.effective_mask.flags.writeable)
            self.assertEqual(selected.provenance["file_sha256"], digest)
            self.assertNotIn("file_sha256", json.loads(path.read_text())["header"])

    def test_wrong_pinned_action_hash_and_source_binding_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, digest, _, kwargs = self._save_authenticated(directory)
            wrong_hashes = dict(kwargs["expected_hashes"])
            field_name = action_artifact.ACTION_PRIMITIVE_HASH_FIELDS[0]
            wrong_hashes[field_name] = "0" * 64
            with patch.object(
                action_artifact,
                "_require_authenticated_forecast_source",
                return_value=kwargs["authenticated_action_source"],
            ), patch.object(
                action_artifact,
                "validate_action_primitive_semantics",
                return_value={"semantic_validation_status": "passed"},
            ), self.assertRaisesRegex(
                action_artifact.P1ActionArtifactError, "external expected digest"
            ):
                action_artifact.load_p1_action_artifact(
                    path,
                    expected_file_sha256=digest,
                    **{**kwargs, "expected_hashes": wrong_hashes},
                )
            wrong_binding = dict(kwargs["expected_source_binding"])
            wrong_binding["source_role"] = "forged"
            with patch.object(
                action_artifact,
                "_require_authenticated_forecast_source",
                return_value=kwargs["authenticated_action_source"],
            ), self.assertRaisesRegex(
                action_artifact.P1ActionArtifactError, "source_binding"
            ):
                action_artifact.load_p1_action_artifact(
                    path,
                    expected_file_sha256=digest,
                    **{**kwargs, "expected_source_binding": wrong_binding},
                )

    def test_production_raw_arrays_without_sealed_source_are_blocked(self) -> None:
        (
            artifact,
            _source,
            source_binding,
            metadata,
            expected_hashes,
            arrays,
        ) = self._production_case()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                action_artifact.P1ActionArtifactError, "sealed ForecastActionSource"
            ):
                action_artifact.save_p1_action_artifact(
                    Path(directory) / "action.json",
                    artifact,
                    expected_metadata=metadata,
                    expected_hashes=expected_hashes,
                    expected_source_binding=source_binding,
                    realized_returns=arrays[0],
                    decision_block_scores=arrays[1],
                    decision_eligible=arrays[2],
                    score_eligible=arrays[3],
                    expected_common_mask=np.ones(4, dtype=np.bool_),
                )

    def test_fixture_and_reconstructed_loaded_objects_cannot_promote(self) -> None:
        artifact = base_action_tests.P1ActionArtifactTests._artifact()
        returns, scores, decision, score = base_action_tests.P1ActionArtifactTests._inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            digest = action_artifact.save_p1_action_artifact(
                path,
                artifact,
                realized_returns=returns,
                decision_block_scores=scores,
                decision_eligible=decision,
                score_eligible=score,
                require_production=False,
            )
            loaded = action_artifact.load_p1_action_artifact(
                path,
                expected_file_sha256=digest,
                realized_returns=returns,
                decision_block_scores=scores,
                decision_eligible=decision,
                score_eligible=score,
                require_production=False,
            )
            self.assertFalse(loaded.is_authenticated)
            with self.assertRaisesRegex(action_artifact.P1ActionArtifactError, "identity-sealed"):
                action_artifact.require_authenticated_loaded_action_artifact(loaded)
            reconstructed = replace(loaded, _production_seal=action_artifact._P1_ACTION_ARTIFACT_SEAL)
            self.assertFalse(reconstructed.is_authenticated)
            with self.assertRaises(action_artifact.P1ActionArtifactError):
                action_artifact.require_authenticated_loaded_action_artifact(reconstructed)

    def test_authenticated_action_capability_can_feed_field_specific_mbb(self) -> None:
        # Keep this regression small while still satisfying the fixed L=8
        # primitive-grid index.  The semantic validator is patched exactly as
        # in the capability tests above; the action-artifact loader's identity
        # seal remains real and is what the MBB adapter must authenticate.
        from unidream.experiments.action_primitives import produce_action_primitive_grid

        n_bars = 33
        returns = np.full(n_bars, 0.001, dtype=np.float64)
        scores = np.full(n_bars, np.nan, dtype=np.float64)
        scores[::4] = 0.01
        decision = np.ones(n_bars, dtype=np.bool_)
        score = np.ones(n_bars, dtype=np.bool_)
        contract = ActionExecutionContract.canonical()
        artifact = produce_action_primitive_grid(
            returns=returns,
            decision_block_scores=scores,
            decision_eligible=decision,
            score_eligible=score,
            scenario_id="fixture",
            seed=1,
            split_id="fixture",
            support_id="fixture",
            model_id="ridge",
            cost_mode="on",
            cost_contract_hash=contract.contract_hash,
        )
        record_count = len(artifact["records"])
        source_hashes = {
            field_name: "a" * 64
            for field_name in action_artifact.P1_ACTION_SOURCE_BINDING_HASH_FIELDS
            if field_name != "source_body_sha256"
        }
        source = SimpleNamespace(
            scenario_id="fixture",
            arm="fixture",
            seed=1,
            model_id="ridge",
            split_id="fixture",
            support_id="fixture",
            support_range=(0, n_bars),
            fit_origin=0,
            prereg_results_observed=False,
            validation_results_observed=True,
            outer_results_observed=False,
            validation_status="passed",
            promotion_allowed=True,
            binding_sha256="b" * 64,
            source_hashes=source_hashes,
            realized_returns=returns,
            forecast_h4=scores,
            origin_mask=decision,
            bar_available=score,
        )
        source_binding = {
            "schema_id": "p1-forecast-action-source-binding-v1",
            "source_role": "authenticated_p1_forecast_action_source",
            "scenario_id": "fixture",
            "arm": "fixture",
            "seed": 1,
            "model_id": "ridge",
            "split_id": "fixture",
            "support_id": "fixture",
            "support_range": [0, n_bars],
            "fit_origin": 0,
            "prereg_results_observed": False,
            "validation_results_observed": True,
            "outer_results_observed": False,
            "validation_status": "passed",
            "promotion_allowed": True,
            "capability_binding_sha256": "b" * 64,
            "source_hashes": source_hashes,
        }
        artifact["header"].update(
            {
                "source_role": "validated_stored_action_inputs",
                "action_primitive_producer_status": "validated_production_input",
                "metric_source": "recomputed_from_realized_returns",
                "trial_id": "fixture-trial",
                "source_binding": source_binding,
                "source_binding_sha256": action_artifact._source_binding_sha256(source_binding),
                "paired_common_mask_sha256": hashlib.sha256(
                    np.ones(record_count, dtype=np.bool_).tobytes(order="C")
                ).hexdigest(),
            }
        )
        metadata = {
            field_name: artifact["header"][field_name]
            for field_name in (
                *action_artifact.ACTION_PRIMITIVE_ARM_FIELDS,
                "support_start",
                "support_range",
                "trial_id",
                "source_binding_sha256",
                "paired_common_mask_sha256",
            )
        }
        expected_hashes = {
            field_name: artifact[field_name]
            for field_name in action_artifact.ACTION_PRIMITIVE_HASH_FIELDS
        }
        index = build_p1_mbb_index_artifact(
            record_count,
            unit="synthetic_action",
            support_id="synthetic_validation",
            seed_ordinal=0,
            block_length=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            action_path = root / "action.json"
            kwargs = {
                "expected_metadata": metadata,
                "expected_hashes": expected_hashes,
                "expected_source_binding": source_binding,
                "authenticated_action_source": source,
                "realized_returns": returns,
                "decision_block_scores": scores,
                "decision_eligible": decision,
                "score_eligible": score,
                "expected_common_mask": np.ones(record_count, dtype=np.bool_),
            }
            with patch.object(
                action_artifact,
                "_require_authenticated_forecast_source",
                return_value=source,
            ), patch.object(
                action_artifact,
                "validate_action_primitive_semantics",
                return_value={"semantic_validation_status": "passed"},
            ):
                action_file_sha = action_artifact.save_p1_action_artifact(
                    action_path,
                    artifact,
                    **kwargs,
                )
                loaded = action_artifact.load_p1_action_artifact(
                    action_path,
                    expected_file_sha256=action_file_sha,
                    **kwargs,
                )
            index_path = root / "index.npz"
            index_file_sha = save_p1_mbb_index_artifact(index_path, index)
            index = load_p1_mbb_index_artifact(
                index_path,
                expected_artifact_sha256=index.artifact_sha256,
                expected_file_sha256=index_file_sha,
            )
            common = np.ones(record_count, dtype=np.bool_)
            expected = {
                field_name: expected_hashes[field_name]
                for field_name in action_artifact.ACTION_PRIMITIVE_HASH_FIELDS
            }
            expected.update(
                {
                    "source_result_sha256": "a" * 64,
                    "source_action_file_sha256": action_file_sha,
                }
            )
            result = bootstrap_p1_action_metric(
                "policy_utility_delta",
                artifact=index,
                candidate_action_artifact=loaded,
                candidate_expected=expected,
                common_mask=common,
                expected_common_mask_sha256=p1_mask_sha256(common),
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["metric"], "policy_utility_delta")
            self.assertTrue(np.isfinite(result["bootstrap_values"]).all())


if __name__ == "__main__":
    unittest.main()
