from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from unidream.eval.action_execution import ActionExecutionContract
from unidream.experiments.chronological_oof import (
    ConditionalOOFArtifactError,
    ConditionalPathBlocked,
    build_conditional_oof_artifact,
    chronological_oof_predict,
    hash_conditional_oof_artifact,
    load_conditional_oof_artifact,
    require_conditional_oof_artifact,
    require_conditional_oof_inputs,
    validate_conditional_oof_artifact,
    write_conditional_oof_artifact,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class ConditionalOOFArtifactTest(unittest.TestCase):
    @staticmethod
    def _raw(*, n_rows: int = 10, horizon: int = 1, labels: np.ndarray | None = None) -> dict:
        features = np.arange(n_rows, dtype=np.float64).reshape(-1, 1)
        if labels is None:
            labels = features[:, 0] * 0.1
        target_end = np.arange(n_rows, dtype=np.int64) + horizon + 1

        def fit_predict(x_train, y_train, x_test):
            del x_train, x_test
            return {"prediction": [float(np.mean(y_train))]}

        return chronological_oof_predict(
            features,
            labels,
            fit_predict=fit_predict,
            horizon=horizon,
            purge=0,
            min_train_size=2,
            target_end=target_end,
        )

    @classmethod
    def _artifact(cls, *, coverage: list[dict] | None = None, labels=None) -> dict:
        return build_conditional_oof_artifact(
            cls._raw(labels=labels),
            horizon=1,
            action_execution_contract=ActionExecutionContract.canonical(),
            checkpoint_sha256=_sha("checkpoint"),
            normalizer_sha256=_sha("normalizer"),
            calibrator_sha256=_sha("calibrator"),
            teacher_weight_sha256=_sha("teacher"),
            coverage=coverage
            or [
                {
                    "head": "return",
                    "horizon": 1,
                    "target_count": 8,
                    "gradient_steps": 8,
                    "nonzero_gradient_steps": 8,
                    "target_coverage": 0.8,
                    "gradient_coverage": 0.8,
                    "status": "pass",
                }
            ],
        )

    def test_artifact_round_trips_typed_nan_arrays_and_hash(self) -> None:
        artifact = self._artifact()
        validate_conditional_oof_artifact(
            artifact,
            expected_action_execution_contract=ActionExecutionContract.canonical(),
            expected_heads_horizons=[("return", 1)],
        )
        self.assertEqual(artifact["artifact_sha256"], hash_conditional_oof_artifact(artifact))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditional_oof.json"
            self.assertEqual(write_conditional_oof_artifact(path, artifact), artifact["artifact_sha256"])
            loaded = load_conditional_oof_artifact(
                path,
                expected_action_execution_contract=ActionExecutionContract.canonical(),
            )
        loaded_predictions = loaded["predictions"]
        expected_predictions = artifact["predictions"]
        self.assertTrue(
            np.all(
                (loaded_predictions == expected_predictions)
                | (np.isnan(loaded_predictions) & np.isnan(expected_predictions))
            )
        )
        np.testing.assert_array_equal(loaded["prediction_mask"], artifact["prediction_mask"])
        self.assertEqual(loaded["artifact_sha256"], artifact["artifact_sha256"])

    def test_same_shape_prediction_tamper_is_rejected_by_content_hash(self) -> None:
        artifact = self._artifact()
        tampered = copy.deepcopy(artifact)
        row = int(np.flatnonzero(tampered["prediction_mask"])[0])
        tampered["predictions"][row, 0] += 123.0
        with self.assertRaises(ConditionalOOFArtifactError):
            validate_conditional_oof_artifact(tampered)

    def test_future_perturbation_cannot_change_same_origin_prediction(self) -> None:
        labels = np.arange(10, dtype=np.float64) * 0.1
        changed = labels.copy()
        changed[4] += 10_000.0
        first = self._raw(labels=labels)
        second = self._raw(labels=changed)
        np.testing.assert_array_equal(first["predictions"][4], second["predictions"][4])
        self.assertNotEqual(first["predictions"][6, 0], second["predictions"][6, 0])
        np.testing.assert_array_equal(
            first["target_end_exclusive"],
            np.arange(10, dtype=np.int64) + 2,
        )

    def test_training_overlap_and_bad_delay_are_rejected_after_digest_update(self) -> None:
        artifact = self._artifact()
        overlap = copy.deepcopy(artifact)
        origin = overlap["origins"][0]
        t = int(origin["prediction_index"])
        origin["train_indices"] = [t]
        origin["train_start"] = t
        origin["train_end_exclusive"] = t + 1
        origin["n_train"] = 1
        overlap["provenance"]["origin_sha256"] = _sha("tampered-origin")
        overlap["origin_sha256"] = overlap["provenance"]["origin_sha256"]
        overlap["artifact_sha256"] = hash_conditional_oof_artifact(overlap)
        overlap["artifact_hash"] = overlap["artifact_sha256"]
        with self.assertRaises(ConditionalOOFArtifactError):
            validate_conditional_oof_artifact(overlap)

        bad_delay = copy.deepcopy(artifact)
        bad_delay["provenance"]["execution_delay_bars"] = 0
        bad_delay["artifact_sha256"] = hash_conditional_oof_artifact(bad_delay)
        bad_delay["artifact_hash"] = bad_delay["artifact_sha256"]
        with self.assertRaises(ConditionalOOFArtifactError):
            validate_conditional_oof_artifact(bad_delay)

    def test_zero_h64_coverage_is_retained_but_strict_consumer_fails(self) -> None:
        artifact = self._artifact(
            coverage=[
                {
                    "head": "return",
                    "horizon": 64,
                    "target_count": 0,
                    "gradient_steps": 0,
                    "nonzero_gradient_steps": 0,
                    "target_coverage": 0.0,
                    "gradient_coverage": 0.0,
                    "status": "block",
                    "block_reason": "zero_valid_targets",
                }
            ]
        )
        validate_conditional_oof_artifact(artifact, require_nonzero_coverage=False)
        self.assertEqual(artifact["coverage"][0]["horizon"], 64)
        with self.assertRaises(ConditionalOOFArtifactError):
            validate_conditional_oof_artifact(artifact)

    def test_missing_artifact_and_contract_hash_mismatch_fail_closed(self) -> None:
        config = {
            "conditional_oracle_path": True,
            "require_conditional_oof_artifact": True,
        }
        with self.assertRaises(ConditionalPathBlocked):
            require_conditional_oof_inputs(
                config=config,
                oof_bundle={},
                caller="artifact-test",
            )
        artifact = self._artifact()
        with self.assertRaises(ConditionalPathBlocked):
            require_conditional_oof_artifact(
                config=config,
                artifact=artifact,
                caller="artifact-test",
                expected_action_execution_contract_hash=_sha("wrong-contract"),
            )
        require_conditional_oof_inputs(
            config=config,
            oof_bundle={"conditional_oof_artifact": artifact},
            caller="artifact-test",
        )


if __name__ == "__main__":
    unittest.main()
