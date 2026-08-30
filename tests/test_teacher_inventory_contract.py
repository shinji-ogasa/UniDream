from __future__ import annotations

import unittest

import numpy as np

from unidream.experiments.chronological_oof import (
    ConditionalPathBlocked,
    chronological_oof_predict,
)
from unidream.experiments.fold_inputs import (
    TeacherInventoryContractError,
    current_inventory_from_replay,
    prepare_fold_inputs,
    validate_current_inventory_source,
)
from unidream.experiments.predictive_state import build_wm_predictive_state_bundle


class TeacherInventoryContractTest(unittest.TestCase):
    @staticmethod
    def _complete_oof_bundle() -> dict:
        def fit_predict(x_train, y_train, x_test):
            del x_train, y_train, x_test
            return [0.0]

        raw = chronological_oof_predict(
            np.arange(3, dtype=np.float64).reshape(-1, 1),
            np.arange(3, dtype=np.float64),
            fit_predict=fit_predict,
            min_train_size=1,
        )
        bundle = dict(raw)
        for split in ("train", "val", "test"):
            bundle[split] = raw["predictions"].copy()
            bundle[f"{split}_row_indices"] = np.arange(3, dtype=np.int64)
            bundle[f"{split}_mask"] = raw["prediction_mask"].copy()
            bundle[f"{split}_prediction_eligibility_mask"] = raw[
                "prediction_eligibility_mask"
            ].copy()
            bundle[f"{split}_training_label_eligibility_mask"] = raw[
                "training_label_eligibility_mask"
            ].copy()
        return bundle

    def test_teacher_and_hindsight_inventory_sources_are_rejected(self) -> None:
        for source in ("hindsight_teacher", "teacher", "oracle", "signal_aim"):
            with self.subTest(source=source):
                with self.assertRaises(TeacherInventoryContractError):
                    validate_current_inventory_source(source)
        with self.assertRaises(TeacherInventoryContractError):
            validate_current_inventory_source("policy_replay", provenance={"producer": "hindsight_oracle"})

    def test_allowed_replay_sources_are_explicit_and_shifted(self) -> None:
        path = np.asarray([0.8, 1.0, 0.6], dtype=np.float32)
        current = current_inventory_from_replay(
            source="policy_replay",
            positions=path,
            benchmark_position=1.0,
            initial_position=1.0,
        )
        np.testing.assert_allclose(current, [1.0, 0.8, 1.0])
        benchmark = current_inventory_from_replay(
            source="benchmark_replay",
            length=3,
            benchmark_position=1.0,
        )
        np.testing.assert_array_equal(benchmark, [1.0, 1.0, 1.0])
        with self.assertRaises(TeacherInventoryContractError):
            current_inventory_from_replay(
                source="benchmark_replay",
                length=3,
                positions=path,
                benchmark_position=1.0,
            )
        with self.assertRaises(TeacherInventoryContractError):
            current_inventory_from_replay(
                source="policy_replay",
                positions=path,
                benchmark_position=1.0,
            )

    def test_conditional_predictive_state_does_not_fall_back_to_legacy_state(self) -> None:
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((2, 1), dtype=np.float32),
                h_train=np.zeros((2, 1), dtype=np.float32),
                seq_len=2,
                ac_cfg={
                    "conditional_oracle_path": True,
                    "use_wm_predictive_state": True,
                },
                log_ts=lambda: "00:00:00",
            )

    def test_conditional_bundle_rejects_partial_finite_state_outside_mask(self) -> None:
        partial = np.asarray([[1.0, np.nan]], dtype=np.float32)
        unavailable = np.full((1, 2), np.nan, dtype=np.float32)
        bundle = {
            "train": partial,
            "val": unavailable,
            "test": unavailable,
            "train_mask": np.asarray([False]),
            "val_mask": np.asarray([False]),
            "test_mask": np.asarray([False]),
            "provenance": {
                "fit_scheme": "chronological_oof",
                "in_sample": False,
            },
        }
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((1, 1), dtype=np.float32),
                h_train=np.zeros((1, 1), dtype=np.float32),
                seq_len=1,
                ac_cfg={"conditional_oracle_path": True},
                log_ts=lambda: "00:00:00",
                oof_bundle=bundle,
            )

    def test_conditional_bundle_masks_require_strict_bool_dtype(self) -> None:
        values = np.ones((1, 2), dtype=np.float32)
        invalid_masks = (
            np.asarray([1], dtype=np.int64),
            np.asarray([1.0], dtype=np.float32),
            np.asarray(["true"]),
            np.asarray([True, np.nan], dtype=object),
        )
        for invalid in invalid_masks:
            with self.subTest(dtype=invalid.dtype):
                bundle = {
                    "train": values,
                    "val": values,
                    "test": values,
                    "train_mask": invalid,
                    "val_mask": np.asarray([True]),
                    "test_mask": np.asarray([True]),
                    "provenance": {
                        "fit_scheme": "chronological_oof",
                        "in_sample": False,
                    },
                }
                with self.assertRaises(ConditionalPathBlocked):
                    build_wm_predictive_state_bundle(
                        wm_trainer=object(),
                        wfo_dataset=object(),
                        z_train=np.zeros((1, 1), dtype=np.float32),
                        h_train=np.zeros((1, 1), dtype=np.float32),
                        seq_len=1,
                        ac_cfg={"conditional_oracle_path": True},
                        log_ts=lambda: "00:00:00",
                        oof_bundle=bundle,
                    )

    def test_conditional_gate_rejects_missing_raw_eligibility_or_in_sample_provenance(self) -> None:
        for missing in (
            "prediction_eligibility_mask",
            "training_label_eligibility_mask",
        ):
            with self.subTest(missing=missing):
                bundle = self._complete_oof_bundle()
                bundle.pop(missing)
                with self.assertRaises(ConditionalPathBlocked):
                    build_wm_predictive_state_bundle(
                        wm_trainer=object(),
                        wfo_dataset=object(),
                        z_train=np.zeros((3, 1), dtype=np.float32),
                        h_train=np.zeros((3, 1), dtype=np.float32),
                        seq_len=1,
                        ac_cfg={"conditional_oracle_path": True},
                        log_ts=lambda: "00:00:00",
                        oof_bundle=bundle,
                    )

        bundle = self._complete_oof_bundle()
        bundle["provenance"] = dict(bundle["provenance"])
        bundle["provenance"].pop("in_sample")
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((3, 1), dtype=np.float32),
                h_train=np.zeros((3, 1), dtype=np.float32),
                seq_len=1,
                ac_cfg={"conditional_oracle_path": True},
                log_ts=lambda: "00:00:00",
                oof_bundle=bundle,
            )

    def test_conditional_gate_rejects_split_only_masks_without_raw_oof_result(self) -> None:
        raw = self._complete_oof_bundle()
        split_only = {
            key: raw[key]
            for key in (
                "train",
                "val",
                "test",
                "train_mask",
                "val_mask",
                "test_mask",
                "train_prediction_eligibility_mask",
                "val_prediction_eligibility_mask",
                "test_prediction_eligibility_mask",
                "train_training_label_eligibility_mask",
                "val_training_label_eligibility_mask",
                "test_training_label_eligibility_mask",
                "train_row_indices",
                "val_row_indices",
                "test_row_indices",
                "provenance",
            )
        }
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((3, 1), dtype=np.float32),
                h_train=np.zeros((3, 1), dtype=np.float32),
                seq_len=1,
                ac_cfg={"conditional_oracle_path": True},
                log_ts=lambda: "00:00:00",
                oof_bundle=split_only,
            )

    def test_conditional_split_eligibility_masks_must_cover_state_masks_only(self) -> None:
        for relation in ("prediction", "training_label"):
            with self.subTest(relation=relation):
                bundle = self._complete_oof_bundle()
                eligible = bundle["train_prediction_eligibility_mask"]
                if relation == "prediction":
                    state_index = int(np.flatnonzero(bundle["train_mask"])[0])
                    eligible[state_index] = False
                else:
                    training = bundle["train_training_label_eligibility_mask"]
                    eligible[0] = False
                    training[0] = True
                with self.assertRaises(ConditionalPathBlocked):
                    build_wm_predictive_state_bundle(
                        wm_trainer=object(),
                        wfo_dataset=object(),
                        z_train=np.zeros((3, 1), dtype=np.float32),
                        h_train=np.zeros((3, 1), dtype=np.float32),
                        seq_len=1,
                        ac_cfg={"conditional_oracle_path": True},
                        log_ts=lambda: "00:00:00",
                        oof_bundle=bundle,
                    )

    def test_conditional_split_views_must_match_indexed_raw_oof_rows(self) -> None:
        accepted = self._complete_oof_bundle()
        accepted_result = build_wm_predictive_state_bundle(
            wm_trainer=object(),
            wfo_dataset=object(),
            z_train=np.zeros((3, 1), dtype=np.float32),
            h_train=np.zeros((3, 1), dtype=np.float32),
            seq_len=1,
            ac_cfg={"conditional_oracle_path": True},
            log_ts=lambda: "00:00:00",
            oof_bundle=accepted,
        )
        np.testing.assert_array_equal(accepted_result["val_row_indices"], [0, 1, 2])

        transformed_metadata = self._complete_oof_bundle()
        transformed_metadata["mean"] = np.zeros((1, 1), dtype=np.float64)
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((3, 1), dtype=np.float32),
                h_train=np.zeros((3, 1), dtype=np.float32),
                seq_len=1,
                ac_cfg={"conditional_oracle_path": True},
                log_ts=lambda: "00:00:00",
                oof_bundle=transformed_metadata,
            )

        bundle = self._complete_oof_bundle()
        bundle["val"] = bundle["val"].copy()
        valid_row = int(np.flatnonzero(bundle["val_mask"])[0])
        bundle["val"][valid_row, 0] += 1.0
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((3, 1), dtype=np.float32),
                h_train=np.zeros((3, 1), dtype=np.float32),
                seq_len=1,
                ac_cfg={"conditional_oracle_path": True},
                log_ts=lambda: "00:00:00",
                oof_bundle=bundle,
            )

        bad_row_mapping = self._complete_oof_bundle()
        bad_row_mapping["val_row_indices"] = np.asarray([0, 2, 1], dtype=np.int64)
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((3, 1), dtype=np.float32),
                h_train=np.zeros((3, 1), dtype=np.float32),
                seq_len=1,
                ac_cfg={"conditional_oracle_path": True},
                log_ts=lambda: "00:00:00",
                oof_bundle=bad_row_mapping,
            )

        missing_indices = self._complete_oof_bundle()
        missing_indices.pop("test_row_indices")
        with self.assertRaises(ConditionalPathBlocked):
            build_wm_predictive_state_bundle(
                wm_trainer=object(),
                wfo_dataset=object(),
                z_train=np.zeros((3, 1), dtype=np.float32),
                h_train=np.zeros((3, 1), dtype=np.float32),
                seq_len=1,
                ac_cfg={"conditional_oracle_path": True},
                log_ts=lambda: "00:00:00",
                oof_bundle=missing_indices,
            )

    def test_conditional_fold_input_builder_blocks_before_hindsight_dp(self) -> None:
        with self.assertRaises(ConditionalPathBlocked):
            prepare_fold_inputs(
                wfo_dataset=None,
                cfg={"conditional_oracle_path": True},
                costs_cfg={},
                ac_cfg={},
                bc_cfg={},
                reward_cfg={},
                action_stats_fn=None,
                format_action_stats_fn=None,
                benchmark_position=1.0,
                forward_window_stats_fn=None,
                log_ts=lambda: "00:00:00",
            )


if __name__ == "__main__":
    unittest.main()
