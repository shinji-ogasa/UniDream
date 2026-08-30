"""Persistence and external-binding tests for P1 action primitives."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from unidream.eval.action_execution import ActionExecutionContract
from unidream.experiments.action_primitives import produce_action_primitive_grid
from unidream.experiments.p1_action_artifact import (
    P1ActionArtifactError,
    load_p1_action_artifact,
    save_p1_action_artifact,
)


class P1ActionArtifactTests(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            dtype=np.float64,
        )
        scores = np.full(17, np.nan, dtype=np.float64)
        scores[[0, 4, 8, 12]] = (-0.01, 0.01, 0.0, 0.01)
        decision = np.ones(17, dtype=bool)
        score = np.ones(17, dtype=bool)
        score[5] = False
        return returns, scores, decision, score

    @classmethod
    def _artifact(cls) -> dict[str, object]:
        returns, scores, decision, score = cls._inputs()
        contract = ActionExecutionContract.canonical()
        return produce_action_primitive_grid(
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

    def test_round_trip_requires_external_file_digest_and_restores_nan(self) -> None:
        artifact = self._artifact()
        returns, scores, decision, score = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.json"
            digest = save_p1_action_artifact(
                path,
                artifact,
                realized_returns=returns,
                decision_block_scores=scores,
                decision_eligible=decision,
                score_eligible=score,
                require_production=False,
            )
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("NaN", raw)
            self.assertIn("null", raw)
            loaded = load_p1_action_artifact(
                path,
                expected_file_sha256=digest,
                realized_returns=returns,
                decision_block_scores=scores,
                decision_eligible=decision,
                score_eligible=score,
                require_production=False,
            )
            self.assertEqual(loaded.file_sha256, digest)
            self.assertEqual(loaded.validation["semantic_validation_status"], "passed")
            self.assertTrue(np.isnan(loaded.artifact["records"][1]["candidate_utility"]))
            with self.assertRaisesRegex(P1ActionArtifactError, "file SHA-256 mismatch"):
                load_p1_action_artifact(
                    path,
                    expected_file_sha256="0" * 64,
                    require_production=False,
                )

    def test_rehashed_header_tamper_and_symlink_fail_closed(self) -> None:
        artifact = self._artifact()
        returns, scores, decision, score = self._inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "action.json"
            save_p1_action_artifact(
                path,
                artifact,
                realized_returns=returns,
                decision_block_scores=scores,
                decision_eligible=decision,
                score_eligible=score,
                require_production=False,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["header"]["support_start"] = 4
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            path.write_bytes(encoded)
            digest = hashlib.sha256(encoded).hexdigest()
            with self.assertRaisesRegex(P1ActionArtifactError, "semantic validation"):
                load_p1_action_artifact(
                    path,
                    expected_file_sha256=digest,
                    realized_returns=returns,
                    decision_block_scores=scores,
                    decision_eligible=decision,
                    score_eligible=score,
                    require_production=False,
                )
            payload["header"]["forged_provenance"] = {"result": "self-bound"}
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            path.write_bytes(encoded)
            with self.assertRaisesRegex(P1ActionArtifactError, "header fields"):
                load_p1_action_artifact(
                    path,
                    expected_file_sha256=hashlib.sha256(encoded).hexdigest(),
                    require_production=False,
                )
            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(P1ActionArtifactError, "non-symlink"):
                load_p1_action_artifact(
                    link,
                    expected_file_sha256=digest,
                    require_production=False,
                )

    def test_production_write_requires_external_metadata_and_sources(self) -> None:
        artifact = self._artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "action.json"
            with self.assertRaisesRegex(P1ActionArtifactError, "expected_metadata"):
                save_p1_action_artifact(path, artifact, require_production=True)


if __name__ == "__main__":
    unittest.main()
