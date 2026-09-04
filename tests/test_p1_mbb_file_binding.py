"""Exact-file persistence and source-binding tests for P1 MBB."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from unidream.eval.action_execution import ActionExecutionContract
from unidream.experiments.action_primitives import produce_action_primitive_grid
from unidream.experiments.p1_action_artifact import (
    LoadedP1ActionArtifact,
    load_p1_action_artifact,
    save_p1_action_artifact,
)
from unidream.experiments.p1_mbb import (
    P1MBBError,
    P1MBBIndexArtifact,
    P1MBBResultArtifact,
    build_p1_mbb_index_artifact,
    bootstrap_p1_metric,
    load_p1_mbb_index_artifact,
    load_p1_mbb_result,
    p1_mask_sha256,
    save_p1_mbb_index_artifact,
    save_p1_mbb_result_artifact,
)


class P1MBBFileBindingTests(unittest.TestCase):
    @staticmethod
    def _index() -> P1MBBIndexArtifact:
        return build_p1_mbb_index_artifact(
            19,
            unit="synthetic_forecast",
            support_id="synthetic_validation",
            seed_ordinal=0,
            block_length=8,
        )

    @staticmethod
    def _forecast_result(index: P1MBBIndexArtifact) -> P1MBBResultArtifact:
        mask = np.ones(index.n, dtype=np.bool_)
        digest = p1_mask_sha256(mask)
        result = bootstrap_p1_metric(
            "mse_delta",
            artifact=index,
            mask=mask,
            candidate_mask=mask,
            baseline_mask=mask,
            candidate_se=np.ones(index.n, dtype="<f8"),
            baseline_se=np.full(index.n, 2.0, dtype="<f8"),
            provenance={
                "kind": "forecast",
                "common_mask_sha256": digest,
                "common_mask_field": "common_mask",
                "forecast_artifact_sha256": "a" * 64,
                "forecast_result_sha256": "b" * 64,
            },
            expected_common_mask_sha256=digest,
            expected_common_mask_field="common_mask",
            expected_forecast_artifact_sha256="a" * 64,
            expected_forecast_result_sha256="b" * 64,
        )
        return P1MBBResultArtifact.from_result_production(result)

    def _loaded_index(self, root: Path) -> tuple[P1MBBIndexArtifact, str]:
        source = self._index()
        path = root / "index.npz"
        file_sha256 = save_p1_mbb_index_artifact(path, source)
        loaded = load_p1_mbb_index_artifact(
            path,
            expected_artifact_sha256=source.artifact_sha256,
            expected_file_sha256=file_sha256,
        )
        return loaded, file_sha256

    @staticmethod
    def _fixture_loaded_action(root: Path) -> LoadedP1ActionArtifact:
        returns = np.asarray(
            [
                0.0,
                0.001,
                -0.001,
                0.002,
                0.001,
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
                0.001,
                0.001,
            ],
            dtype="<f8",
        )
        scores = np.full(17, np.nan, dtype="<f8")
        scores[[0, 4, 8, 12]] = (-0.01, 0.01, 0.0, 0.01)
        decision = np.ones(17, dtype=np.bool_)
        score = np.ones(17, dtype=np.bool_)
        score[5] = False
        contract = ActionExecutionContract.canonical()
        action = produce_action_primitive_grid(
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
        path = root / "action.json"
        file_sha256 = save_p1_action_artifact(
            path,
            action,
            realized_returns=returns,
            decision_block_scores=scores,
            decision_eligible=decision,
            score_eligible=score,
            require_production=False,
        )
        return load_p1_action_artifact(
            path,
            expected_file_sha256=file_sha256,
            realized_returns=returns,
            decision_block_scores=scores,
            decision_eligible=decision,
            score_eligible=score,
            require_production=False,
        )

    def test_save_returns_exact_post_write_sha_without_self_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._index()
            path = root / "index.npz"
            file_sha256 = save_p1_mbb_index_artifact(path, source)
            self.assertEqual(file_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertNotEqual(file_sha256, source.artifact_sha256)
            self.assertNotIn("file_sha256", source.to_dict())

            result = self._forecast_result(self._loaded_index(root)[0])
            result_path = root / "result.npz"
            result_file_sha256 = save_p1_mbb_result_artifact(result_path, result)
            self.assertEqual(
                result_file_sha256,
                hashlib.sha256(result_path.read_bytes()).hexdigest(),
            )
            self.assertNotEqual(result_file_sha256, result.result_sha256)
            self.assertNotIn("file_sha256", result.to_dict())

    def test_missing_or_wrong_expected_file_sha_and_byte_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._index()
            path = root / "index.npz"
            expected_file_sha256 = save_p1_mbb_index_artifact(path, source)
            with self.assertRaisesRegex(P1MBBError, "expected_file_sha256"):
                load_p1_mbb_index_artifact(
                    path,
                    expected_artifact_sha256=source.artifact_sha256,
                )
            with self.assertRaisesRegex(P1MBBError, "file SHA-256 mismatch"):
                load_p1_mbb_index_artifact(
                    path,
                    expected_artifact_sha256=source.artifact_sha256,
                    expected_file_sha256="0" * 64,
                )

            encoded = bytearray(path.read_bytes())
            encoded[-1] ^= 0x01
            path.write_bytes(encoded)
            with self.assertRaisesRegex(P1MBBError, "file SHA-256 mismatch"):
                load_p1_mbb_index_artifact(
                    path,
                    expected_artifact_sha256=source.artifact_sha256,
                    expected_file_sha256=expected_file_sha256,
                )

    def test_result_replay_and_byte_tamper_require_both_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded_index, _ = self._loaded_index(root)
            result = self._forecast_result(loaded_index)
            path = root / "result.npz"
            result_file_sha256 = save_p1_mbb_result_artifact(path, result)
            with self.assertRaisesRegex(P1MBBError, "expected_file_sha256"):
                load_p1_mbb_result(path, expected_result_sha256=result.result_sha256)
            with self.assertRaisesRegex(P1MBBError, "file SHA-256 mismatch"):
                load_p1_mbb_result(
                    path,
                    expected_result_sha256=result.result_sha256,
                    expected_file_sha256="f" * 64,
                )
            encoded = bytearray(path.read_bytes())
            encoded[0] ^= 0x01
            path.write_bytes(encoded)
            with self.assertRaisesRegex(P1MBBError, "file SHA-256 mismatch"):
                load_p1_mbb_result(
                    path,
                    expected_result_sha256=result.result_sha256,
                    expected_file_sha256=result_file_sha256,
                )

    def test_mismatched_source_action_binding_and_unsealed_typed_input_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded_index, _ = self._loaded_index(root)
            mask = np.ones(loaded_index.n, dtype=np.bool_)
            common_sha256 = p1_mask_sha256(mask)
            action_provenance = {
                "kind": "action",
                "common_mask_sha256": common_sha256,
                "common_mask_field": "common_mask",
                "action_primitive_payload_sha256": "1" * 64,
                "action_primitive_schema_sha256": "2" * 64,
                "action_primitive_content_sha256": "3" * 64,
                "source_result_sha256": "4" * 64,
                "source_action_file_sha256": "5" * 64,
            }
            kwargs = {
                "artifact": loaded_index,
                "mask": mask,
                "candidate_mask": mask,
                "baseline_mask": mask,
                "candidate_utility": np.arange(loaded_index.n, dtype="<f8"),
                "benchmark_hold_utility": np.zeros(loaded_index.n, dtype="<f8"),
                "provenance": action_provenance,
                "expected_common_mask_sha256": common_sha256,
                "expected_common_mask_field": "common_mask",
                "expected_source_result_sha256": "4" * 64,
                "expected_action_primitive_payload_sha256": "1" * 64,
                "expected_action_primitive_schema_sha256": "2" * 64,
                "expected_action_primitive_content_sha256": "3" * 64,
                "expected_source_action_file_sha256": "5" * 64,
            }
            mismatched = dict(kwargs)
            mismatched["expected_source_action_file_sha256"] = "6" * 64
            with self.assertRaisesRegex(P1MBBError, "authenticated action capability"):
                bootstrap_p1_metric("policy_utility_delta", **mismatched)

            # A fixture-loaded or directly constructed action object has no
            # identity-sealed production marker and is intentionally rejected.
            fixture_typed = self._fixture_loaded_action(root)
            typed_kwargs = dict(kwargs)
            typed_kwargs["source_action_artifact"] = fixture_typed
            typed_kwargs["expected_action_primitive_payload_sha256"] = fixture_typed.artifact[
                "action_primitive_payload_sha256"
            ]
            typed_kwargs["expected_action_primitive_schema_sha256"] = fixture_typed.artifact[
                "action_primitive_schema_sha256"
            ]
            typed_kwargs["expected_action_primitive_content_sha256"] = fixture_typed.artifact[
                "action_primitive_content_sha256"
            ]
            typed_kwargs["provenance"] = dict(action_provenance)
            typed_kwargs["provenance"]["action_primitive_payload_sha256"] = typed_kwargs[
                "expected_action_primitive_payload_sha256"
            ]
            typed_kwargs["provenance"]["action_primitive_schema_sha256"] = typed_kwargs[
                "expected_action_primitive_schema_sha256"
            ]
            typed_kwargs["provenance"]["action_primitive_content_sha256"] = typed_kwargs[
                "expected_action_primitive_content_sha256"
            ]
            with self.assertRaisesRegex(P1MBBError, "identity-authenticated"):
                bootstrap_p1_metric("policy_utility_delta", **typed_kwargs)

            direct_typed = LoadedP1ActionArtifact(
                root / "direct.json",
                fixture_typed.file_sha256,
                fixture_typed.artifact,
                fixture_typed.validation,
            )
            typed_kwargs["source_action_artifact"] = direct_typed
            with self.assertRaisesRegex(P1MBBError, "identity-authenticated"):
                bootstrap_p1_metric("policy_utility_delta", **typed_kwargs)


if __name__ == "__main__":
    unittest.main()
