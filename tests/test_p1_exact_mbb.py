"""Fixed-fixture tests for the preregistered P1 non-circular MBB."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

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
    build_p1_mbb_index_artifact,
    derive_p1_seed,
    draw_non_circular_mbb_starts,
    load_p1_mbb_index_artifact,
    materialize_non_circular_mbb_indices,
    paired_bootstrap_mean_delta,
    paired_bootstrap_mean_delta_sensitivity,
    reject_unpaired_or_generic_mbb,
    save_p1_mbb_index_artifact,
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
        candidate = np.arange(n, dtype="<f8")
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
            direction="positive",
        )
        self.assertEqual(result["block_lengths"], [8, 16, 32])
        self.assertEqual(set(result["per_block_length"]), {8, 16, 32})
        self.assertEqual(
            result["raw_p"],
            max(row["p_value"] for row in result["per_block_length"].values()),
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
