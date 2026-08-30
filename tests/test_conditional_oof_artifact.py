from __future__ import annotations

import copy
import hashlib
import json
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
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle


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
                    "total_target_slots": 10,
                    "masked_target_slots": 8,
                    "valid_targets": 8,
                    "finite_targets": 10,
                    "finite_masked_targets": 8,
                    "finite_target_count": 10,
                    "finite_loss_steps": 8,
                    "gradient_steps": 8,
                    "nonzero_gradient_steps": 8,
                    "target_coverage": 0.8,
                    "gradient_coverage": 1.0,
                    "pass": True,
                    "status": "pass",
                    "block_reason": None,
                }
            ],
        )

    @staticmethod
    def _strict_config(*, heads=("return", 1)) -> dict:
        return {
            "conditional_oracle_path": True,
            "require_conditional_oof_artifact": True,
            "expected_heads_horizons": [heads],
            "expected_hashes": {
                "checkpoint_sha256": _sha("checkpoint"),
                "normalizer_sha256": _sha("normalizer"),
                "calibrator_sha256": _sha("calibrator"),
                "teacher_weight_sha256": _sha("teacher"),
            },
            "expected_action_execution_contract": ActionExecutionContract.canonical().to_dict(),
            "expected_action_execution_contract_hash": ActionExecutionContract.canonical().contract_hash,
        }

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

        missing_delay = copy.deepcopy(artifact)
        missing_delay["provenance"] = dict(missing_delay["provenance"])
        missing_delay["provenance"].pop("execution_delay_bars")
        missing_delay["artifact_sha256"] = hash_conditional_oof_artifact(missing_delay)
        missing_delay["artifact_hash"] = missing_delay["artifact_sha256"]
        with self.assertRaises(ConditionalPathBlocked):
            require_conditional_oof_inputs(
                config=self._strict_config(),
                oof_bundle={"conditional_oof_artifact": missing_delay},
                caller="artifact-test",
            )

        overflowing_horizon = copy.deepcopy(artifact)
        overflowing_horizon["provenance"] = dict(overflowing_horizon["provenance"])
        overflowing_horizon["provenance"]["horizon"] = 10**100
        overflowing_horizon["artifact_sha256"] = hash_conditional_oof_artifact(overflowing_horizon)
        overflowing_horizon["artifact_hash"] = overflowing_horizon["artifact_sha256"]
        with self.assertRaises(ConditionalPathBlocked):
            require_conditional_oof_inputs(
                config=self._strict_config(),
                oof_bundle={"conditional_oof_artifact": overflowing_horizon},
                caller="artifact-test",
            )

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
        config = self._strict_config()
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

    def test_strict_consumer_requires_external_expected_bindings_and_rejects_h64_drop(self) -> None:
        artifact = self._artifact()
        incomplete = {
            "conditional_oracle_path": True,
            "require_conditional_oof_artifact": True,
        }
        with self.assertRaises(ConditionalPathBlocked):
            require_conditional_oof_inputs(
                config=incomplete,
                oof_bundle={"conditional_oof_artifact": artifact},
                caller="artifact-test",
            )
        with self.assertRaises(ConditionalPathBlocked):
            require_conditional_oof_inputs(
                config=self._strict_config(heads=("return", 64)),
                oof_bundle={"conditional_oof_artifact": artifact},
                caller="artifact-test",
            )
        with self.assertRaises(ConditionalPathBlocked):
            require_conditional_oof_inputs(
                config={"require_conditional_oof_artifact": True},
                oof_bundle={"conditional_oof_artifact": artifact},
                caller="artifact-test",
            )

        hash_only = self._strict_config()
        hash_only.pop("expected_action_execution_contract")
        with self.assertRaises(ConditionalPathBlocked):
            require_conditional_oof_inputs(
                config=hash_only,
                oof_bundle={"conditional_oof_artifact": artifact},
                caller="artifact-test",
            )

        self_bound = copy.deepcopy(artifact)
        self_bound["expected_heads_horizons"] = [("return", 1)]
        self_bound["expected_hashes"] = self._strict_config()["expected_hashes"]
        self_bound["expected_action_execution_contract"] = ActionExecutionContract.canonical().to_dict()
        self_bound["expected_action_execution_contract_hash"] = ActionExecutionContract.canonical().contract_hash
        self_bound["artifact_sha256"] = hash_conditional_oof_artifact(self_bound)
        self_bound["artifact_hash"] = self_bound["artifact_sha256"]
        with self.assertRaises(ConditionalPathBlocked):
            require_conditional_oof_inputs(
                config={
                    "conditional_oracle_path": True,
                    "require_conditional_oof_artifact": True,
                },
                oof_bundle={"conditional_oof_artifact": self_bound},
                caller="artifact-test",
            )

    def test_strict_coverage_requires_pass_and_consistent_counts(self) -> None:
        base = self._artifact()
        for field, value in (
            ("gradient_steps", 0),
            ("nonzero_gradient_steps", 0),
            ("target_coverage", 0.0),
            ("gradient_coverage", 0.0),
            ("status", "block"),
        ):
            with self.subTest(field=field):
                artifact = copy.deepcopy(base)
                artifact["coverage"][0][field] = value
                artifact["artifact_sha256"] = hash_conditional_oof_artifact(artifact)
                artifact["artifact_hash"] = artifact["artifact_sha256"]
                with self.assertRaises(ConditionalOOFArtifactError):
                    validate_conditional_oof_artifact(artifact)

        for field in (
            "total_target_slots",
            "masked_target_slots",
            "valid_targets",
            "finite_targets",
            "finite_masked_targets",
            "finite_target_count",
            "finite_loss_steps",
            "pass",
            "block_reason",
        ):
            with self.subTest(missing_field=field):
                artifact = copy.deepcopy(base)
                artifact["coverage"][0].pop(field)
                artifact["artifact_sha256"] = hash_conditional_oof_artifact(artifact)
                artifact["artifact_hash"] = artifact["artifact_sha256"]
                with self.assertRaises(ConditionalOOFArtifactError):
                    validate_conditional_oof_artifact(artifact)

        for field, value in (
            ("total_target_slots", 0),
            ("masked_target_slots", 7),
            ("finite_masked_targets", 7),
            ("finite_targets", 7),
            ("finite_loss_steps", 0),
            ("pass", False),
            ("block_reason", "unexpected_block"),
        ):
            with self.subTest(invalid_field=field):
                artifact = copy.deepcopy(base)
                artifact["coverage"][0][field] = value
                artifact["artifact_sha256"] = hash_conditional_oof_artifact(artifact)
                artifact["artifact_hash"] = artifact["artifact_sha256"]
                with self.assertRaises(ConditionalOOFArtifactError):
                    validate_conditional_oof_artifact(artifact)

        inconsistent = {
            "head": "return",
            "horizon": 1,
            "target_count": 8,
            "total_target_slots": 10,
            "gradient_steps": 8,
            "nonzero_gradient_steps": 8,
            "target_coverage": 0.9,
            "gradient_coverage": 1.0,
            "status": "pass",
        }
        with self.assertRaises(ConditionalOOFArtifactError):
            self._artifact(coverage=[inconsistent])

    def test_false_prediction_rows_reject_inf_and_partial_finite_values(self) -> None:
        artifact = self._artifact()
        for replacement in (
            np.asarray([[np.inf]], dtype=np.float64),
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([[1.0, np.nan]], dtype=np.float64),
        ):
            with self.subTest(replacement=replacement.tolist()):
                tampered = copy.deepcopy(artifact)
                false_row = int(np.flatnonzero(~tampered["prediction_mask"])[0])
                if replacement.shape[1] == tampered["predictions"].shape[1]:
                    tampered["predictions"][false_row] = replacement[0]
                else:
                    tampered["predictions"][false_row, 0] = replacement[0, 0]
                    if tampered["predictions"].shape[1] > 1:
                        tampered["predictions"][false_row, 1] = replacement[0, 1]
                tampered["artifact_sha256"] = hash_conditional_oof_artifact(tampered)
                tampered["artifact_hash"] = tampered["artifact_sha256"]
                with self.assertRaises(ConditionalOOFArtifactError):
                    validate_conditional_oof_artifact(tampered)

        for shape in ((len(artifact["predictions"]), 0), (0, 1)):
            with self.subTest(shape=shape):
                tampered = copy.deepcopy(artifact)
                tampered["predictions"] = np.empty(shape, dtype=np.float64)
                tampered["artifact_sha256"] = hash_conditional_oof_artifact(tampered)
                tampered["artifact_hash"] = tampered["artifact_sha256"]
                with self.assertRaises(ConditionalOOFArtifactError):
                    validate_conditional_oof_artifact(tampered)

        empty_origins = copy.deepcopy(artifact)
        empty_origins["origins"] = []
        empty_origins["provenance"] = dict(empty_origins["provenance"])
        empty_origins["provenance"]["n_origins_called"] = 0
        empty_origins["provenance"]["origin_sha256"] = _sha("empty-origins")
        empty_origins["origin_sha256"] = empty_origins["provenance"]["origin_sha256"]
        empty_origins["artifact_sha256"] = hash_conditional_oof_artifact(empty_origins)
        empty_origins["artifact_hash"] = empty_origins["artifact_sha256"]
        with self.assertRaises(ConditionalOOFArtifactError):
            validate_conditional_oof_artifact(empty_origins)

    def test_action_contract_mapping_and_aliases_are_content_bound(self) -> None:
        artifact = self._artifact()
        tampered = copy.deepcopy(artifact)
        tampered["action_execution_contract"]["fee_rate"] = 0.0004
        tampered["provenance"]["action_execution_contract"]["fee_rate"] = 0.0004
        tampered["artifact_sha256"] = hash_conditional_oof_artifact(tampered)
        tampered["artifact_hash"] = tampered["artifact_sha256"]
        with self.assertRaises(ConditionalOOFArtifactError):
            validate_conditional_oof_artifact(tampered)

        alias_mismatch = copy.deepcopy(artifact)
        alias_mismatch["provenance"]["teacher_sha256"] = _sha("different-teacher")
        alias_mismatch["artifact_sha256"] = hash_conditional_oof_artifact(alias_mismatch)
        alias_mismatch["artifact_hash"] = alias_mismatch["artifact_sha256"]
        with self.assertRaises(ConditionalOOFArtifactError):
            validate_conditional_oof_artifact(alias_mismatch)

        root_origin_mismatch = copy.deepcopy(artifact)
        root_origin_mismatch["origin_sha256"] = _sha("different-origin")
        root_origin_mismatch["artifact_sha256"] = hash_conditional_oof_artifact(root_origin_mismatch)
        root_origin_mismatch["artifact_hash"] = root_origin_mismatch["artifact_sha256"]
        with self.assertRaises(ConditionalOOFArtifactError):
            validate_conditional_oof_artifact(root_origin_mismatch)

        with self.assertRaises(ConditionalOOFArtifactError):
            build_conditional_oof_artifact(
                self._raw(),
                horizon=1,
                action_execution_contract=_sha("contract-a"),
                action_execution_contract_hash=_sha("contract-b"),
                checkpoint_sha256=_sha("checkpoint"),
                normalizer_sha256=_sha("normalizer"),
                calibrator_sha256=_sha("calibrator"),
                teacher_weight_sha256=_sha("teacher"),
                coverage=[
                    {
                        "head": "return",
                        "horizon": 1,
                        "target_count": 8,
                        "gradient_steps": 8,
                        "nonzero_gradient_steps": 8,
                        "target_coverage": 0.8,
                        "gradient_coverage": 1.0,
                        "status": "pass",
                    }
                ],
            )

        for alias, value in (
            ("transition_cost_rate", 0.123),
            ("delta_grid", [0.0]),
            ("countdown_reset", 99),
            ("spread_side", "wrong_side"),
        ):
            with self.subTest(alias=alias):
                tampered = copy.deepcopy(artifact)
                tampered["action_execution_contract"][alias] = value
                tampered["provenance"]["action_execution_contract"][alias] = value
                tampered["artifact_sha256"] = hash_conditional_oof_artifact(tampered)
                tampered["artifact_hash"] = tampered["artifact_sha256"]
                with self.assertRaises(ConditionalOOFArtifactError):
                    validate_conditional_oof_artifact(tampered)

        for component, value in (
            ("normalizer", {"sha256": _sha("normalizer"), "in_sample": True}),
            ("calibrator", {"sha256": _sha("calibrator")}),
            ("teacher_weight", {"sha256": _sha("teacher"), "in_sample": "false"}),
            ("checkpoint", {"sha256": _sha("checkpoint"), "in_sample": True}),
        ):
            with self.subTest(component=component):
                tampered = copy.deepcopy(artifact)
                tampered["provenance"][component] = value
                tampered["artifact_sha256"] = hash_conditional_oof_artifact(tampered)
                tampered["artifact_hash"] = tampered["artifact_sha256"]
                with self.assertRaises(ConditionalOOFArtifactError):
                    validate_conditional_oof_artifact(tampered)

    def test_malformed_typed_json_is_normalized_and_bounded(self) -> None:
        malformed_payloads = (
            {
                "__ndarray__": True,
                "dtype": "O",
                "shape": [1],
                "data_b64": "AA==",
            },
            {
                "__ndarray__": True,
                "dtype": "float64",
                "shape": [2**63],
                "data_b64": "",
            },
            {
                "__ndarray__": True,
                "dtype": "float64",
                "shape": [1] * 9,
                "data_b64": "AAAAAAAAAAA=",
            },
            {
                "__ndarray__": True,
                "dtype": "float64",
                "shape": [1],
                "data_b64": "not-base64",
            },
            {
                "__ndarray__": True,
                "dtype": "uint8",
                "shape": [1],
                # Four base64 characters decode to three bytes; one was declared.
                "data_b64": "AAAA",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            for payload in malformed_payloads:
                with self.subTest(payload=payload):
                    path = Path(directory) / "malformed.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ConditionalOOFArtifactError):
                        load_conditional_oof_artifact(path)

        aggregate = {
            "arrays": [
                {
                    "__ndarray__": True,
                    "dtype": "uint8",
                    "shape": [10_000_000],
                    "data_b64": "",
                },
                {
                    "__ndarray__": True,
                    "dtype": "uint8",
                    "shape": [10_000_000],
                    "data_b64": "",
                },
                {
                    "__ndarray__": True,
                    "dtype": "uint8",
                    "shape": [1],
                    "data_b64": "",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            path.write_text(json.dumps(aggregate), encoding="utf-8")
            with self.assertRaises(ConditionalOOFArtifactError):
                load_conditional_oof_artifact(path)

    def test_atomic_write_uses_unique_same_directory_temporary_file(self) -> None:
        artifact = self._artifact()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "conditional_oof.json"
            write_conditional_oof_artifact(output, artifact)
            self.assertTrue(output.is_file())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_nested_envelope_cannot_shadow_artifact_core_even_with_equal_value(self) -> None:
        artifact = self._artifact()
        envelope = {"conditional_oof_artifact": artifact}
        for split in ("train", "val", "test"):
            envelope[split] = artifact["predictions"].copy()
            envelope[f"{split}_row_indices"] = np.arange(len(artifact["predictions"]), dtype=np.int64)
            envelope[f"{split}_mask"] = artifact["prediction_mask"].copy()
            envelope[f"{split}_prediction_eligibility_mask"] = artifact[
                "prediction_eligibility_mask"
            ].copy()
            envelope[f"{split}_training_label_eligibility_mask"] = artifact[
                "training_label_eligibility_mask"
            ].copy()
        # Equal values are still rejected: the outer envelope must not be able
        # to become a second source of truth for any artifact core key.
        envelope["predictions"] = artifact["predictions"].copy()
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((len(artifact["predictions"]), 1), dtype=np.float32),
                h_train=np.zeros((len(artifact["predictions"]), 1), dtype=np.float32),
                seq_len=1,
                ac_cfg={"conditional_oracle_path": True},
                log_ts=lambda: "00:00:00",
                oof_bundle=envelope,
            )


if __name__ == "__main__":
    unittest.main()
