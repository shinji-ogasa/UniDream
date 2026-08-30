"""Fixed-fixture tests for the preregistered P1 non-circular MBB."""
from __future__ import annotations

import tempfile
from pathlib import Path
import json
import stat
import unittest
import warnings
import zipfile

import numpy as np

from unidream.experiments.action_primitives import (
    ActionPrimitiveImplementationBlocked,
    run_action_primitive_mbb,
)
from unidream.experiments.p1_mbb import (
    P1_MBB_BLOCK_LENGTHS,
    P1_MBB_REPLICATES,
    P1MBBError,
    P1MBBImplementationBlocked,
    P1MBBIndexArtifact,
    build_p1_mbb_index_artifact,
    bootstrap_p1_metric as production_bootstrap_p1_metric,
    bootstrap_p1_metric_seed_aggregate as production_seed_aggregate,
    bootstrap_p1_metric_fixture as bootstrap_p1_metric,
    bootstrap_p1_metric_seed_aggregate_fixture as bootstrap_p1_metric_seed_aggregate,
    derive_p1_seed,
    draw_non_circular_mbb_starts,
    load_p1_mbb_index_artifact_fixture as load_p1_mbb_index_artifact,
    materialize_non_circular_mbb_indices,
    paired_bootstrap_mean_delta_fixture as paired_bootstrap_mean_delta,
    paired_bootstrap_mean_delta_sensitivity_fixture as paired_bootstrap_mean_delta_sensitivity,
    p1_mask_sha256,
    recompute_agreement_delta,
    recompute_agreement_mean,
    recompute_logloss_delta,
    recompute_logloss_mean,
    recompute_mse_delta,
    recompute_normalized_regret,
    recompute_policy_utility_delta,
    recompute_s2_level_contrast,
    recompute_s2_normalized_regret_contrast,
    recompute_s2_skill_contrast,
    recompute_s3_skill_did,
    recompute_s3_utility_did,
    recompute_skill,
    reject_unpaired_or_generic_mbb,
    save_p1_mbb_index_artifact,
    P1MBBResultArtifact,
    load_p1_mbb_result,
    load_p1_mbb_result_fixture,
    save_p1_mbb_result_artifact,
    save_p1_mbb_result_fixture,
)


class P1ExactMBBTests(unittest.TestCase):
    @staticmethod
    def _artifact(*, n: int = 19, block_length: int = 8):
        return build_p1_mbb_index_artifact(
            n,
            unit="synthetic_forecast",
            support_id="synthetic_validation",
            seed_ordinal=0,
            block_length=block_length,
        )

    def test_exact_start_draw_and_c_order_materialization(self) -> None:
        n = 19
        length = 8
        derived = derive_p1_seed("synthetic_forecast", length, 0)
        artifact = self._artifact(n=n, block_length=length)

        rng = np.random.default_rng(derived)
        expected_starts = rng.integers(
            low=0,
            high=n - length + 1,
            size=(n + length - 1) // length,
            endpoint=False,
            dtype=np.int64,
        )
        np.testing.assert_array_equal(artifact.starts[0], expected_starts)
        expected_indices = np.ascontiguousarray(
            expected_starts[:, None] + np.arange(length, dtype=np.int64)
        ).reshape(-1, order="C")[:n]
        np.testing.assert_array_equal(artifact.indices_for(0), expected_indices)
        self.assertTrue(artifact.starts.flags.c_contiguous)
        self.assertFalse(artifact.starts.flags.writeable)
        self.assertTrue(np.all(artifact.indices_for(0) >= 0))
        self.assertTrue(np.all(artifact.indices_for(0) < n))

    def test_l_boundaries_and_no_circular_wrap(self) -> None:
        for length in P1_MBB_BLOCK_LENGTHS:
            with self.subTest(length=length):
                boundary = build_p1_mbb_index_artifact(
                    length,
                    unit="synthetic_action",
                    support_id="synthetic_validation",
                    seed_ordinal=0,
                    block_length=length,
                )
                self.assertTrue(np.all(boundary.starts == 0))
                self.assertTrue(
                    np.array_equal(
                        boundary.indices_for(0), np.arange(length, dtype=np.int64)
                    )
                )
                with self.assertRaises(P1MBBError):
                    build_p1_mbb_index_artifact(
                        length - 1,
                        unit="synthetic_action",
                        support_id="synthetic_validation",
                        seed_ordinal=0,
                        block_length=length,
                    )
        for invalid_length in (1, 7, 9, 16 + 1, 64):
            with self.subTest(invalid_length=invalid_length):
                with self.assertRaises(P1MBBError):
                    self._artifact(block_length=invalid_length)

    def test_rng_lifecycle_is_deterministic_and_separated_by_unit_and_length(self) -> None:
        first = self._artifact()
        second = self._artifact()
        np.testing.assert_array_equal(first.starts, second.starts)
        self.assertEqual(first.artifact_sha256, second.artifact_sha256)
        action = build_p1_mbb_index_artifact(
            19,
            unit="synthetic_action",
            support_id="synthetic_validation",
            seed_ordinal=0,
            block_length=8,
        )
        different_length = self._artifact(block_length=16)
        self.assertFalse(np.array_equal(first.starts, action.starts))
        self.assertFalse(np.array_equal(first.starts, different_length.starts))
        self.assertEqual(first.derived_seed, 20260830 + 100000 + 8000)

    def test_paired_metric_reuses_full_grid_indices_and_rejects_mask_or_values(self) -> None:
        n = 19
        artifact = self._artifact(n=n)
        candidate = np.arange(n, dtype="<f8")
        baseline = np.ones(n, dtype="<f8")
        mask = np.ones(n, dtype=np.bool_)
        result = paired_bootstrap_mean_delta(
            candidate,
            baseline,
            candidate_mask=mask,
            baseline_mask=mask.copy(),
            artifact=artifact,
            metric="policy_utility_delta",
            direction="positive",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["replicates"], P1_MBB_REPLICATES)
        self.assertEqual(result["point_delta"], float(np.mean(candidate - baseline)))
        self.assertEqual(result["index_artifact_sha256"], artifact.artifact_sha256)
        self.assertEqual(result["bootstrap_values"].shape, (P1_MBB_REPLICATES,))

        mismatched = mask.copy()
        mismatched[-1] = False
        with self.assertRaises(P1MBBError):
            paired_bootstrap_mean_delta(
                candidate,
                baseline,
                candidate_mask=mask,
                baseline_mask=mismatched,
                artifact=artifact,
                metric="policy_utility_delta",
                direction="positive",
            )
        for bad_candidate in (
            candidate.astype(np.float32),
            candidate.copy(),
        ):
            if bad_candidate.dtype == np.dtype("<f8"):
                bad_candidate[0] = np.nan
            with self.subTest(dtype=bad_candidate.dtype, value=bad_candidate[0]):
                with self.assertRaises(P1MBBError):
                    paired_bootstrap_mean_delta(
                        bad_candidate,
                        baseline,
                        candidate_mask=mask,
                        baseline_mask=mask,
                        artifact=artifact,
                        metric="mse_delta",
                        direction="negative",
                    )
        with self.assertRaises(P1MBBError):
            paired_bootstrap_mean_delta(
                candidate[:-1],
                baseline,
                candidate_mask=mask,
                baseline_mask=mask,
                artifact=artifact,
                metric="mse_delta",
                direction="negative",
            )
        with self.assertRaises(P1MBBError):
            paired_bootstrap_mean_delta(
                candidate,
                baseline,
                candidate_mask=mask,
                baseline_mask=mask,
                artifact=artifact,
                metric="unregistered_metric",
                direction="positive",
            )

    def test_seed_alias_and_unpaired_generic_paths_fail_closed(self) -> None:
        with self.assertRaises(P1MBBError):
            derive_p1_seed(
                "synthetic_forecast",
                8,
                0,
                seed=20260830,
            )
        with self.assertRaises(P1MBBError):
            derive_p1_seed(
                "synthetic_forecast",
                8,
                0,
                unit_code=1,
            )
        with self.assertRaises(P1MBBError):
            derive_p1_seed("s3_forecast", 8, 1)
        with self.assertRaises(P1MBBImplementationBlocked):
            reject_unpaired_or_generic_mbb()
        with self.assertRaises(P1MBBImplementationBlocked):
            from unidream.experiments.p1_mbb import run_p1_mbb

            run_p1_mbb(candidate_values=np.ones(19, dtype="<f8"))
        # The old generic action primitive entrypoint remains deliberately
        # blocked; the new exact implementation is a separate P1 boundary.
        with self.assertRaises(ActionPrimitiveImplementationBlocked):
            run_action_primitive_mbb([], block_length=16)

    def test_fixed_mask_and_gap_rows_are_not_compressed(self) -> None:
        n = 19
        artifact = self._artifact(n=n)
        candidate = (np.arange(n) % 2).astype("<f8")
        baseline = np.zeros(n, dtype="<f8")
        mask = np.ones(n, dtype=np.bool_)
        mask[[2, 3, 17]] = False
        candidate[~mask] = np.nan
        baseline[~mask] = np.nan
        result = paired_bootstrap_mean_delta(
            candidate,
            baseline,
            candidate_mask=mask,
            baseline_mask=mask,
            artifact=artifact,
            metric="agreement",
            direction="positive",
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["point_delta"], float(np.mean(candidate[mask])))

    def test_index_artifact_round_trip_preserves_all_starts(self) -> None:
        artifact = self._artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.npz"
            digest = save_p1_mbb_index_artifact(path, artifact)
            self.assertEqual(digest, artifact.artifact_sha256)
            loaded = load_p1_mbb_index_artifact(path)
        np.testing.assert_array_equal(loaded.starts, artifact.starts)
        self.assertEqual(loaded.to_dict(include_starts=False), artifact.to_dict(include_starts=False))
        self.assertEqual(loaded.artifact_sha256, artifact.artifact_sha256)

    def test_production_index_binding_requires_external_and_fixed_stream(self) -> None:
        artifact = self._artifact()
        payload = artifact.to_dict()
        with self.assertRaises(P1MBBError):
            P1MBBIndexArtifact.from_dict(payload)
        loaded = P1MBBIndexArtifact.from_dict(
            payload,
            expected_artifact_sha256=artifact.artifact_sha256,
        )
        np.testing.assert_array_equal(loaded.starts, artifact.starts)
        forged = dict(payload)
        forged_starts = np.array(artifact.starts, dtype="<i8", copy=True)
        forged_starts[0, 0] += 1
        forged["starts"] = forged_starts
        forged["starts_sha256"] = __import__("hashlib").sha256(
            forged_starts.tobytes(order="C")
        ).hexdigest()
        forged_metadata = dict(forged)
        forged_metadata.pop("starts")
        forged_metadata.pop("artifact_sha256")
        forged["artifact_sha256"] = __import__("hashlib").sha256(
            json.dumps(
                forged_metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\0"
            + forged_starts.tobytes(order="C")
        ).hexdigest()
        with self.assertRaises(P1MBBError):
            P1MBBIndexArtifact.from_dict(
                forged,
                expected_artifact_sha256=forged["artifact_sha256"],
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "indices.npz"
            save_p1_mbb_index_artifact(path, artifact)
            with self.assertRaises(P1MBBError):
                from unidream.experiments.p1_mbb import load_p1_mbb_index_artifact

                load_p1_mbb_index_artifact(path)
            strict_loaded = load_p1_mbb_index_artifact(
                path,
                expected_artifact_sha256=artifact.artifact_sha256,
            )
            np.testing.assert_array_equal(strict_loaded.starts, artifact.starts)

    def test_index_artifact_layout_and_archive_budgets_fail_closed(self) -> None:
        artifact = self._artifact(n=19)
        payload = artifact.to_dict()
        payload["starts"] = np.asarray(artifact.starts, dtype="<i8")
        malformed = dict(payload)
        malformed["starts_dtype"] = "<f8"
        with self.assertRaises(P1MBBError):
            P1MBBIndexArtifact.from_dict(malformed)
        malformed = dict(payload)
        malformed["starts_shape"] = [P1_MBB_REPLICATES, 999]
        with self.assertRaises(P1MBBError):
            P1MBBIndexArtifact.from_dict(malformed)
        with self.assertRaises(P1MBBError):
            P1MBBIndexArtifact.from_dict(
                {**payload, "starts": artifact.starts.astype("<i4")}
            )
        with self.assertRaises(P1MBBError):
            P1MBBIndexArtifact.from_dict(
                {**payload, "starts": np.asfortranarray(artifact.starts)}
            )
        with self.assertRaises(P1MBBError):
            P1MBBIndexArtifact.from_dict(
                {**payload, "starts": artifact.starts.reshape(-1)}
            )
        huge = dict(payload)
        huge.update(
            {
                "n": 10**9,
                "starts_shape": [P1_MBB_REPLICATES, 125_000_000],
                "starts": np.empty((0, 0), dtype="<i8"),
            }
        )
        with self.assertRaises(P1MBBError):
            P1MBBIndexArtifact.from_dict(huge)

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            oversized_metadata = artifact.metadata()
            oversized_metadata.update(
                {
                    "n": 10**9,
                    "starts_shape": [P1_MBB_REPLICATES, 125_000_000],
                }
            )
            metadata = np.frombuffer(
                json.dumps(oversized_metadata, sort_keys=True).encode("utf-8"),
                dtype=np.uint8,
            )
            zip_bomb = directory_path / "declared-huge.npz"
            np.savez_compressed(
                zip_bomb,
                starts=np.zeros((1, 1), dtype="<i8"),
                metadata=metadata,
            )
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(zip_bomb)

            contradictory = directory_path / "contradictory-size.npz"
            np.savez_compressed(
                contradictory,
                starts=np.zeros((P1_MBB_REPLICATES, 100), dtype="<i8"),
                metadata=np.frombuffer(
                    json.dumps(artifact.metadata(), sort_keys=True).encode("utf-8"),
                    dtype=np.uint8,
                ),
            )
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(contradictory)

            metadata_object = np.frombuffer(
                json.dumps(artifact.metadata(), sort_keys=True).encode("utf-8"),
                dtype=np.uint8,
            )
            bad_metadata_dtype = directory_path / "bad-metadata-dtype.npz"
            np.savez_compressed(
                bad_metadata_dtype,
                starts=artifact.starts,
                metadata=metadata_object.astype("<f8"),
            )
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(bad_metadata_dtype)
            bad_metadata_shape = directory_path / "bad-metadata-shape.npz"
            np.savez_compressed(
                bad_metadata_shape,
                starts=artifact.starts,
                metadata=metadata_object.reshape(1, -1),
            )
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(bad_metadata_shape)
            bad_fortran = directory_path / "bad-fortran.npz"
            np.savez_compressed(
                bad_fortran,
                starts=np.asfortranarray(artifact.starts),
                metadata=metadata_object,
            )
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(bad_fortran)
            bad_members = directory_path / "bad-members.npz"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(bad_members, mode="w") as archive:
                    archive.writestr("starts.npy", b"duplicate-a")
                    archive.writestr("starts.npy", b"duplicate-b")
                    archive.writestr("metadata.npy", b"metadata")
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(bad_members)
            bad_path_member = directory_path / "bad-path-member.npz"
            with zipfile.ZipFile(bad_path_member, mode="w") as archive:
                archive.writestr("../starts.npy", b"starts")
                archive.writestr("metadata.npy", b"metadata")
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(bad_path_member)
            bad_archive_symlink = directory_path / "bad-archive-symlink.npz"
            with zipfile.ZipFile(bad_archive_symlink, mode="w") as archive:
                starts_link = zipfile.ZipInfo("starts.npy")
                starts_link.create_system = 3
                starts_link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(starts_link, b"not-a-regular-member")
                archive.writestr("metadata.npy", b"not-a-regular-member")
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(bad_archive_symlink)
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(directory_path)
            valid_path = directory_path / "valid.npz"
            save_p1_mbb_index_artifact(valid_path, artifact)
            symlink = directory_path / "indices-link.npz"
            symlink.symlink_to(valid_path)
            with self.assertRaises(P1MBBError):
                load_p1_mbb_index_artifact(symlink)

    def test_sensitivity_uses_all_fixed_lengths_and_conservative_raw_p(self) -> None:
        n = 35
        candidate = np.arange(n, dtype="<f8")
        baseline = np.zeros(n, dtype="<f8")
        mask = np.ones(n, dtype=np.bool_)
        result = paired_bootstrap_mean_delta_sensitivity(
            candidate,
            baseline,
            candidate_mask=mask,
            baseline_mask=mask,
            unit="synthetic_forecast",
            support_id="synthetic_validation",
            seed_ordinal=0,
            metric="mse_delta",
            direction="negative",
        )
        self.assertEqual(result["block_lengths"], [8, 16, 32])
        self.assertEqual(set(result["per_block_length"]), {8, 16, 32})
        self.assertEqual(
            result["raw_p"],
            max(row["p_value"] for row in result["per_block_length"].values()),
        )

    def test_preregistered_metric_recomputation_is_per_replicate(self) -> None:
        n = 35
        artifact = self._artifact(n=n, block_length=8)
        mask = np.ones(n, dtype=np.bool_)
        candidate_se = np.linspace(1.0, 2.0, n, dtype="<f8")
        baseline_se = np.linspace(2.0, 3.0, n, dtype="<f8")
        candidate_logloss = np.linspace(0.1, 0.5, n, dtype="<f8")
        baseline_logloss = np.linspace(0.2, 0.6, n, dtype="<f8")
        candidate_agreement = (np.arange(n) % 2).astype("<f8")
        baseline_agreement = ((np.arange(n) + 1) % 2).astype("<f8")
        candidate_utility = np.linspace(-0.2, 0.8, n, dtype="<f8")
        benchmark_hold = np.linspace(-0.1, 0.4, n, dtype="<f8")
        regret = np.linspace(0.1, 0.4, n, dtype="<f8")
        opportunity = np.linspace(0.5, 1.5, n, dtype="<f8")

        cases = {
            "mse_delta": {
                "candidate_se": candidate_se,
                "baseline_se": baseline_se,
            },
            "skill": {"model_se": candidate_se, "zero_se": baseline_se},
            "logloss": {
                "candidate_logloss": candidate_logloss,
                "baseline_logloss": baseline_logloss,
            },
            "agreement": {
                "candidate_agreement": candidate_agreement,
                "baseline_agreement": baseline_agreement,
            },
            "policy_utility_delta": {
                "candidate_utility": candidate_utility,
                "benchmark_hold_utility": benchmark_hold,
            },
            "s2_contrast": {
                "level_a_values": candidate_agreement,
                "level_b_values": baseline_agreement,
            },
            "normalized_regret": {"regret": regret, "opportunity": opportunity},
            "s3_skill_did": {
                "injected_model_se": candidate_se,
                "injected_zero_se": baseline_se,
                "control_model_se": baseline_se,
                "control_zero_se": candidate_se,
            },
            "s3_utility_did": {
                "injected_candidate_utility": candidate_utility,
                "injected_benchmark_hold_utility": benchmark_hold,
                "control_candidate_utility": benchmark_hold,
                "control_benchmark_hold_utility": np.zeros(n, dtype="<f8"),
            },
        }
        directions = {
            "mse_delta": "negative",
            "skill": "positive",
            "logloss": "negative",
            "agreement": "positive",
            "policy_utility_delta": "positive",
            "s2_contrast": "positive",
            "normalized_regret": "negative",
            "s3_skill_did": "positive",
            "s3_utility_did": "positive",
        }
        level_direction = {"s2_contrast": "high_ge_medium"}
        for metric, arrays in cases.items():
            with self.subTest(metric=metric):
                result = bootstrap_p1_metric(
                    metric,
                    artifact=artifact,
                    mask=mask,
                    direction=directions[metric],
                    level_direction=level_direction.get(metric),
                    **arrays,
                )
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["bootstrap_values"].shape, (2000,))
                if metric == "mse_delta":
                    expected = recompute_mse_delta(candidate_se, baseline_se, mask)
                    replicate = recompute_mse_delta(
                        candidate_se,
                        baseline_se,
                        mask,
                        indices=artifact.indices_for(0),
                    )
                elif metric == "skill":
                    expected = recompute_skill(candidate_se, baseline_se, mask)
                    replicate = recompute_skill(
                        candidate_se,
                        baseline_se,
                        mask,
                        indices=artifact.indices_for(0),
                    )
                elif metric == "logloss":
                    expected = recompute_logloss_delta(
                        candidate_logloss, baseline_logloss, mask
                    )
                    replicate = recompute_logloss_delta(
                        candidate_logloss,
                        baseline_logloss,
                        mask,
                        indices=artifact.indices_for(0),
                    )
                elif metric == "agreement":
                    expected = recompute_agreement_delta(
                        candidate_agreement, baseline_agreement, mask
                    )
                    replicate = recompute_agreement_delta(
                        candidate_agreement,
                        baseline_agreement,
                        mask,
                        indices=artifact.indices_for(0),
                    )
                elif metric == "policy_utility_delta":
                    expected = recompute_policy_utility_delta(
                        candidate_utility, benchmark_hold, mask
                    )
                    replicate = recompute_policy_utility_delta(
                        candidate_utility,
                        benchmark_hold,
                        mask,
                        indices=artifact.indices_for(0),
                    )
                elif metric == "s2_contrast":
                    expected = recompute_s2_level_contrast(
                        candidate_agreement,
                        baseline_agreement,
                        mask,
                        level_direction="high_ge_medium",
                    )
                    replicate = recompute_s2_level_contrast(
                        candidate_agreement,
                        baseline_agreement,
                        mask,
                        level_direction="high_ge_medium",
                        indices=artifact.indices_for(0),
                    )
                elif metric == "normalized_regret":
                    expected = recompute_normalized_regret(regret, opportunity, mask)
                    replicate = recompute_normalized_regret(
                        regret,
                        opportunity,
                        mask,
                        indices=artifact.indices_for(0),
                    )
                elif metric == "s3_skill_did":
                    expected = recompute_s3_skill_did(
                        candidate_se,
                        baseline_se,
                        baseline_se,
                        candidate_se,
                        mask,
                    )
                    replicate = recompute_s3_skill_did(
                        candidate_se,
                        baseline_se,
                        baseline_se,
                        candidate_se,
                        mask,
                        indices=artifact.indices_for(0),
                    )
                else:
                    expected = recompute_s3_utility_did(
                        candidate_utility,
                        benchmark_hold,
                        benchmark_hold,
                        np.zeros(n, dtype="<f8"),
                        mask,
                    )
                    replicate = recompute_s3_utility_did(
                        candidate_utility,
                        benchmark_hold,
                        benchmark_hold,
                        np.zeros(n, dtype="<f8"),
                        mask,
                        indices=artifact.indices_for(0),
                    )
                self.assertAlmostEqual(result["point_estimate"], expected)
                self.assertAlmostEqual(result["bootstrap_values"][0], replicate)

        self.assertAlmostEqual(
            recompute_logloss_mean(candidate_logloss, mask),
            float(np.mean(candidate_logloss)),
        )
        self.assertAlmostEqual(
            recompute_agreement_mean(candidate_agreement, mask),
            float(np.mean(candidate_agreement)),
        )

    def test_metric_contract_blocks_bad_denominators_masks_and_unregistered_fields(self) -> None:
        n = 35
        artifact = self._artifact(n=n)
        mask = np.ones(n, dtype=np.bool_)
        ones = np.ones(n, dtype="<f8")
        zeros = np.zeros(n, dtype="<f8")
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "skill",
                artifact=artifact,
                mask=mask,
                model_se=ones,
                zero_se=zeros,
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "normalized_regret",
                artifact=artifact,
                mask=mask,
                regret=ones,
                opportunity=zeros,
            )
        mismatched = mask.copy()
        mismatched[-1] = False
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "mse_delta",
                artifact=artifact,
                mask=mask,
                candidate_mask=mask,
                baseline_mask=mismatched,
                candidate_se=ones,
                baseline_se=ones,
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "mse_delta",
                artifact=artifact,
                mask=mask,
                candidate_se=ones,
                baseline_se=ones,
                reducer=lambda values: float(np.mean(values)),
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "mse_delta",
                artifact=artifact,
                mask=mask,
                direction="positive",
                candidate_se=ones,
                baseline_se=ones,
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "skill",
                artifact=artifact,
                mask=mask,
                direction="negative",
                model_se=ones,
                zero_se=ones,
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "normalized_regret",
                artifact=artifact,
                mask=mask,
                direction="positive",
                regret=ones,
                opportunity=ones,
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "s2_contrast",
                artifact=artifact,
                mask=mask,
                level_direction="high_le_medium",
                direction="positive",
                level_a_values=ones,
                level_b_values=zeros,
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "s2_contrast",
                artifact=artifact,
                mask=mask,
                level_direction="high_ge_medium",
                level_metric="normalized_regret",
                level_a_regret=ones,
                level_a_opportunity=ones,
                level_b_regret=ones,
                level_b_opportunity=ones,
            )

    def test_metric_domains_reject_negative_loss_bad_agreement_and_regret(self) -> None:
        n = 35
        artifact = self._artifact(n=n)
        mask = np.ones(n, dtype=np.bool_)
        ones = np.ones(n, dtype="<f8")
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "logloss",
                artifact=artifact,
                mask=mask,
                candidate_logloss=-ones,
                baseline_logloss=ones,
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "agreement",
                artifact=artifact,
                mask=mask,
                candidate_agreement=np.full(n, 0.5, dtype="<f8"),
                baseline_agreement=np.zeros(n, dtype="<f8"),
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "normalized_regret",
                artifact=artifact,
                mask=mask,
                regret=np.full(n, -1.0, dtype="<f8"),
                opportunity=ones,
            )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "normalized_regret",
                artifact=artifact,
                mask=mask,
                regret=np.zeros(n, dtype="<f8"),
                opportunity=np.full(n, -1.0, dtype="<f8"),
            )

    def test_production_bootstrap_requires_external_mask_and_source_binding(self) -> None:
        n = 19
        artifact = self._artifact(n=n)
        mask = np.ones(n, dtype=np.bool_)
        candidate = np.arange(n, dtype="<f8")
        baseline = np.zeros(n, dtype="<f8")
        payload_digest = "1" * 64
        schema_digest = "2" * 64
        content_digest = "3" * 64
        source_digest = "4" * 64
        common_digest = p1_mask_sha256(mask)
        action_provenance = {
            "kind": "action",
            "common_mask_sha256": common_digest,
            "common_mask_field": "common_mask",
            "action_primitive_payload_sha256": payload_digest,
            "action_primitive_schema_sha256": schema_digest,
            "action_primitive_content_sha256": content_digest,
            "source_result_sha256": source_digest,
        }
        result = production_bootstrap_p1_metric(
            "policy_utility_delta",
            artifact=artifact,
            mask=mask,
            candidate_mask=mask,
            baseline_mask=mask,
            candidate_utility=candidate,
            benchmark_hold_utility=baseline,
            provenance=action_provenance,
            expected_common_mask_sha256=common_digest,
            expected_common_mask_field="common_mask",
            expected_source_result_sha256=source_digest,
            expected_action_primitive_payload_sha256=payload_digest,
            expected_action_primitive_schema_sha256=schema_digest,
            expected_action_primitive_content_sha256=content_digest,
        )
        self.assertEqual(result["provenance"]["kind"], "action")
        with self.assertRaises(P1MBBError):
            production_bootstrap_p1_metric(
                "policy_utility_delta",
                artifact=artifact,
                mask=mask,
                candidate_utility=candidate,
                benchmark_hold_utility=baseline,
                provenance=action_provenance,
                expected_common_mask_sha256=common_digest,
                expected_common_mask_field="common_mask",
                expected_source_result_sha256=source_digest,
                expected_action_primitive_payload_sha256=payload_digest,
                expected_action_primitive_schema_sha256=schema_digest,
                expected_action_primitive_content_sha256=content_digest,
            )
        wrong_mask = mask.copy()
        wrong_mask[0] = False
        with self.assertRaises(P1MBBError):
            production_bootstrap_p1_metric(
                "policy_utility_delta",
                artifact=artifact,
                mask=wrong_mask,
                candidate_mask=wrong_mask,
                baseline_mask=wrong_mask,
                candidate_utility=candidate,
                benchmark_hold_utility=baseline,
                provenance=action_provenance,
                expected_common_mask_sha256=common_digest,
                expected_common_mask_field="common_mask",
                expected_source_result_sha256=source_digest,
                expected_action_primitive_payload_sha256=payload_digest,
                expected_action_primitive_schema_sha256=schema_digest,
                expected_action_primitive_content_sha256=content_digest,
            )

    def test_production_forecast_provenance_is_distinct_from_action(self) -> None:
        n = 19
        artifact = self._artifact(n=n)
        mask = np.ones(n, dtype=np.bool_)
        common_digest = p1_mask_sha256(mask)
        result = production_bootstrap_p1_metric(
            "mse_delta",
            artifact=artifact,
            mask=mask,
            candidate_mask=mask,
            baseline_mask=mask,
            candidate_se=np.ones(n, dtype="<f8"),
            baseline_se=np.full(n, 2.0, dtype="<f8"),
            provenance={
                "kind": "forecast",
                "common_mask_sha256": common_digest,
                "common_mask_field": "common_mask",
                "forecast_artifact_sha256": "a" * 64,
                "forecast_result_sha256": "b" * 64,
            },
            expected_common_mask_sha256=common_digest,
            expected_common_mask_field="common_mask",
            expected_forecast_artifact_sha256="a" * 64,
            expected_forecast_result_sha256="b" * 64,
        )
        self.assertEqual(result["provenance"]["kind"], "forecast")
        with self.assertRaises(P1MBBError):
            production_bootstrap_p1_metric(
                "mse_delta",
                artifact=artifact,
                mask=mask,
                candidate_mask=mask,
                baseline_mask=mask,
                candidate_se=np.ones(n, dtype="<f8"),
                baseline_se=np.full(n, 2.0, dtype="<f8"),
                provenance={
                    "kind": "action",
                    "common_mask_sha256": common_digest,
                    "common_mask_field": "common_mask",
                    "action_primitive_payload_sha256": "1" * 64,
                    "action_primitive_schema_sha256": "2" * 64,
                    "action_primitive_content_sha256": "3" * 64,
                    "source_result_sha256": "4" * 64,
                },
                expected_common_mask_sha256=common_digest,
                expected_common_mask_field="common_mask",
                expected_source_result_sha256="4" * 64,
                expected_action_primitive_payload_sha256="1" * 64,
                expected_action_primitive_schema_sha256="2" * 64,
                expected_action_primitive_content_sha256="3" * 64,
            )

    def test_result_artifact_is_typed_atomic_and_external_digest_bound(self) -> None:
        n = 19
        artifact = self._artifact(n=n)
        mask = np.ones(n, dtype=np.bool_)
        digest = p1_mask_sha256(mask)
        result = production_bootstrap_p1_metric(
            "policy_utility_delta",
            artifact=artifact,
            mask=mask,
            candidate_mask=mask,
            baseline_mask=mask,
            candidate_utility=np.arange(n, dtype="<f8"),
            benchmark_hold_utility=np.zeros(n, dtype="<f8"),
            provenance={
                "kind": "action",
                "common_mask_sha256": digest,
                "common_mask_field": "common_mask",
                "action_primitive_payload_sha256": "1" * 64,
                "action_primitive_schema_sha256": "2" * 64,
                "action_primitive_content_sha256": "3" * 64,
                "source_result_sha256": "4" * 64,
            },
            expected_common_mask_sha256=digest,
            expected_common_mask_field="common_mask",
            expected_source_result_sha256="4" * 64,
            expected_action_primitive_payload_sha256="1" * 64,
            expected_action_primitive_schema_sha256="2" * 64,
            expected_action_primitive_content_sha256="3" * 64,
        )
        typed = P1MBBResultArtifact.from_result_production(result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.npz"
            digest_result = save_p1_mbb_result_artifact(path, typed)
            self.assertEqual(digest_result, typed.result_sha256)
            with self.assertRaises(P1MBBError):
                load_p1_mbb_result(path)
            loaded = load_p1_mbb_result(
                path,
                expected_result_sha256=typed.result_sha256,
            )
            self.assertEqual(loaded.result_sha256, typed.result_sha256)
            np.testing.assert_array_equal(loaded.bootstrap_values, typed.bootstrap_values)
            with self.assertRaises(P1MBBError):
                load_p1_mbb_result(path, expected_result_sha256="f" * 64)

        fixture = self._fixture_policy_result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture-result.npz"
            save_p1_mbb_result_fixture(path, fixture)
            fixture_loaded = load_p1_mbb_result_fixture(path)
            self.assertEqual(fixture_loaded.result_sha256, P1MBBResultArtifact.from_result_fixture(fixture).result_sha256)

    def _fixture_policy_result(self) -> dict[str, object]:
        n = 19
        artifact = self._artifact(n=n)
        mask = np.ones(n, dtype=np.bool_)
        return bootstrap_p1_metric(
            "policy_utility_delta",
            artifact=artifact,
            mask=mask,
            candidate_mask=mask,
            baseline_mask=mask,
            candidate_utility=np.arange(n, dtype="<f8"),
            benchmark_hold_utility=np.zeros(n, dtype="<f8"),
        )

    def test_production_ten_seed_result_requires_and_persists_all_provenance(self) -> None:
        n = 19
        mask = np.ones(n, dtype=np.bool_)
        common_digest = p1_mask_sha256(mask)
        seed_inputs = {
            seed: {
                "mask": mask.copy(),
                "candidate_mask": mask.copy(),
                "baseline_mask": mask.copy(),
                "candidate_utility": np.full(n, float(seed + 1), dtype="<f8"),
                "benchmark_hold_utility": np.zeros(n, dtype="<f8"),
            }
            for seed in range(10)
        }
        provenance_by_seed = {
            seed: {
                "provenance": {
                    "kind": "action",
                    "common_mask_sha256": common_digest,
                    "common_mask_field": "common_mask",
                    "action_primitive_payload_sha256": f"{seed + 1:064x}",
                    "action_primitive_schema_sha256": f"{seed + 11:064x}",
                    "action_primitive_content_sha256": f"{seed + 21:064x}",
                    "source_result_sha256": f"{seed + 31:064x}",
                },
                "expected_common_mask_sha256": common_digest,
                "expected_common_mask_field": "common_mask",
                "expected_action_primitive_payload_sha256": f"{seed + 1:064x}",
                "expected_action_primitive_schema_sha256": f"{seed + 11:064x}",
                "expected_action_primitive_content_sha256": f"{seed + 21:064x}",
                "expected_source_result_sha256": f"{seed + 31:064x}",
            }
            for seed in range(10)
        }
        index_artifacts = {
            seed: build_p1_mbb_index_artifact(
                n,
                unit="synthetic_action",
                support_id="synthetic_validation",
                seed_ordinal=seed,
                block_length=8,
            )
            for seed in range(10)
        }
        with self.assertRaises(P1MBBError):
            production_seed_aggregate(
                "policy_utility_delta",
                unit="synthetic_action",
                support_id="synthetic_validation",
                block_length=8,
                seed_inputs=seed_inputs,
                direction="positive",
                provenance_by_seed=provenance_by_seed,
            )
        bad_index_digests = {
            seed: artifact.artifact_sha256
            for seed, artifact in index_artifacts.items()
        }
        bad_index_digests[3] = "f" * 64
        with self.assertRaises(P1MBBError):
            production_seed_aggregate(
                "policy_utility_delta",
                unit="synthetic_action",
                support_id="synthetic_validation",
                block_length=8,
                seed_inputs=seed_inputs,
                direction="positive",
                provenance_by_seed=provenance_by_seed,
                index_artifacts=index_artifacts,
                expected_index_artifact_sha256_by_seed=bad_index_digests,
            )
        result = production_seed_aggregate(
            "policy_utility_delta",
            unit="synthetic_action",
            support_id="synthetic_validation",
            block_length=8,
            seed_inputs=seed_inputs,
            direction="positive",
            provenance_by_seed=provenance_by_seed,
            index_artifacts=index_artifacts,
            expected_index_artifact_sha256_by_seed={
                seed: artifact.artifact_sha256
                for seed, artifact in index_artifacts.items()
            },
        )
        self.assertFalse(result["prereg_results_observed"])
        self.assertTrue(result["validation_results_observed"])
        self.assertFalse(result["outer_results_observed"])
        typed = P1MBBResultArtifact.from_result_production(result)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed-result.npz"
            save_p1_mbb_result_artifact(path, typed)
            loaded = load_p1_mbb_result(
                path,
                expected_result_sha256=typed.result_sha256,
            )
            self.assertEqual(loaded.metadata["seed_count"], 10)
            self.assertEqual(set(loaded.metadata["provenance_by_seed"]), {str(i) for i in range(10)})
            missing_declared = typed.to_dict()
            missing_declared.pop("result_sha256")
            with self.assertRaises(P1MBBError):
                P1MBBResultArtifact.from_dict(
                    missing_declared,
                    typed.bootstrap_values,
                    expected_result_sha256=typed.result_sha256,
                )
    def test_s2_skill_and_normalized_regret_recompute_before_level_contrast(self) -> None:
        n = 35
        artifact = self._artifact(n=n, block_length=8)
        mask = np.ones(n, dtype=np.bool_)
        zeros = np.zeros(n, dtype="<f8")
        level_a_model = np.full(n, 1.0, dtype="<f8")
        level_a_zero = np.full(n, 4.0, dtype="<f8")
        level_b_model = np.full(n, 2.0, dtype="<f8")
        level_b_zero = np.full(n, 4.0, dtype="<f8")
        skill_result = bootstrap_p1_metric(
            "s2_contrast",
            artifact=artifact,
            mask=mask,
            level_direction="high_ge_medium",
            level_metric="skill",
            level_a_model_se=level_a_model,
            level_a_zero_se=level_a_zero,
            level_b_model_se=level_b_model,
            level_b_zero_se=level_b_zero,
        )
        expected_skill = recompute_s2_skill_contrast(
            level_a_model,
            level_a_zero,
            level_b_model,
            level_b_zero,
            mask,
            level_direction="high_ge_medium",
        )
        expected_skill_replicate = recompute_s2_skill_contrast(
            level_a_model,
            level_a_zero,
            level_b_model,
            level_b_zero,
            mask,
            level_direction="high_ge_medium",
            indices=artifact.indices_for(0),
        )
        self.assertAlmostEqual(skill_result["point_estimate"], expected_skill)
        self.assertAlmostEqual(
            skill_result["bootstrap_values"][0], expected_skill_replicate
        )
        level_a_regret = np.full(n, 1.0, dtype="<f8")
        level_a_opportunity = np.full(n, 2.0, dtype="<f8")
        level_b_regret = np.full(n, 1.5, dtype="<f8")
        level_b_opportunity = np.full(n, 2.0, dtype="<f8")
        regret_result = bootstrap_p1_metric(
            "s2_contrast",
            artifact=artifact,
            mask=mask,
            level_direction="high_le_medium",
            level_metric="normalized_regret",
            level_a_regret=level_a_regret,
            level_a_opportunity=level_a_opportunity,
            level_b_regret=level_b_regret,
            level_b_opportunity=level_b_opportunity,
        )
        expected_regret = recompute_s2_normalized_regret_contrast(
            level_a_regret,
            level_a_opportunity,
            level_b_regret,
            level_b_opportunity,
            mask,
            level_direction="high_le_medium",
        )
        self.assertAlmostEqual(regret_result["point_estimate"], expected_regret)
        self.assertAlmostEqual(
            regret_result["bootstrap_values"][0],
            recompute_s2_normalized_regret_contrast(
                level_a_regret,
                level_a_opportunity,
                level_b_regret,
                level_b_opportunity,
                mask,
                level_direction="high_le_medium",
                indices=artifact.indices_for(0),
            ),
        )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric(
                "s2_contrast",
                artifact=artifact,
                mask=mask,
                level_direction="high_ge_medium",
                level_metric="normalized_regret",
                level_a_regret=level_a_regret,
                level_a_opportunity=zeros,
                level_b_regret=level_b_regret,
                level_b_opportunity=level_b_opportunity,
            )

    def test_synthetic_seed_aggregation_is_independent_and_equal_weighted(self) -> None:
        n = 35
        mask = np.ones(n, dtype=np.bool_)
        seed_inputs = {
            seed: {
                "mask": mask.copy(),
                "candidate_utility": np.full(n, float(seed + 1), dtype="<f8"),
                "benchmark_hold_utility": np.zeros(n, dtype="<f8"),
            }
            for seed in range(10)
        }
        result = bootstrap_p1_metric_seed_aggregate(
            "policy_utility_delta",
            unit="synthetic_action",
            support_id="synthetic_validation",
            block_length=8,
            seed_inputs=seed_inputs,
            direction="positive",
        )
        self.assertEqual(result["seed_count"], 10)
        self.assertEqual(result["seed_ordinals"], list(range(10)))
        self.assertAlmostEqual(result["point_estimate"], 5.5)
        expected = np.mean(
            np.stack(
                [result["per_seed"][seed]["bootstrap_values"] for seed in range(10)],
                axis=0,
            ),
            axis=0,
        )
        np.testing.assert_array_equal(result["bootstrap_values"], expected)
        self.assertEqual(len(result["index_artifact_sha256_by_seed"]), 10)
        self.assertEqual(
            len(set(result["index_artifact_sha256_by_seed"].values())),
            10,
        )
        with self.assertRaises(P1MBBError):
            bootstrap_p1_metric_seed_aggregate(
                "policy_utility_delta",
                unit="synthetic_action",
                support_id="synthetic_validation",
                block_length=8,
                seed_inputs={seed: payload for seed, payload in seed_inputs.items() if seed != 9},
                direction="positive",
            )

    def test_raw_start_helpers_require_exact_int64_and_non_circular_range(self) -> None:
        rng = np.random.default_rng(20260830)
        starts = draw_non_circular_mbb_starts(19, 8, rng)
        np.testing.assert_array_equal(
            materialize_non_circular_mbb_indices(starts, 8, 19).shape,
            (19,),
        )
        with self.assertRaises(P1MBBError):
            materialize_non_circular_mbb_indices(starts.astype(np.int32), 8, 19)
        with self.assertRaises(P1MBBError):
            materialize_non_circular_mbb_indices(
                np.array([12, 0, 0], dtype="<i8"),
                8,
                19,
            )


if __name__ == "__main__":
    unittest.main()
