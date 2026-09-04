"""Integration tests that join the P0-A, P0-B, and P0-C boundaries."""
from __future__ import annotations

import tempfile
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from unidream.data.cache_v4 import MODEL_FEATURE_COLUMNS, cache_v4_paths, write_cache_v4
from unidream.data.oracle import (
    conditional_oracle_teacher_path,
    hindsight_upper_bound_path,
)
from unidream.eval.action_execution import ActionExecutionContract
from unidream.experiments.chronological_oof import (
    chronological_oof_predict,
    require_conditional_oof_inputs,
    validate_oof_result,
)
from unidream.experiments.fold_inputs import (
    TeacherInventoryContractError,
    current_inventory_from_replay,
    validate_current_inventory_source,
)
from unidream.experiments.p0_integration_audit import (
    audit_v4_cache,
    load_v4_contract_inputs,
)


def _write_fixture(root: Path, tag: str = "p0-integration") -> None:
    """Write a sparse body plus complete sidecar to exercise full-grid mapping."""
    grid = pd.date_range("2024-01-01", periods=13, freq="15min")
    body = grid.delete(6)  # A missing Spot row must not be compacted away.
    features = pd.DataFrame(
        np.arange(len(body) * len(MODEL_FEATURE_COLUMNS), dtype=np.float64).reshape(
            len(body), len(MODEL_FEATURE_COLUMNS)
        ),
        index=body,
        columns=MODEL_FEATURE_COLUMNS,
    )
    returns = pd.Series(
        np.linspace(-0.002, 0.003, len(body)),
        index=body,
        name="returns",
    )
    availability = pd.DataFrame(
        {
            "spot_bar_observed": [True] * 6 + [False] + [True] * 6,
            # Row 8 is an unavailable decision-time context feature; row 10
            # is deliberately unavailable funding inside an otherwise
            # observed Spot outcome block.  P0-C must still score that block.
            "funding_rate_available": [True] * 8 + [False] + [True] + [False] + [True] * 2,
            "mark_close_available": [True] * 13,
        },
        index=grid,
    )
    write_cache_v4(
        features,
        returns,
        availability,
        cache_dir=root,
        cache_tag=tag,
        source_provenance={"source": "p0_integration_fixture"},
        start="2024-01-01",
        end="2024-01-02",
    )


class P0IntegrationAuditTest(unittest.TestCase):
    def test_explicit_frozen_metadata_validates_same_body_but_tracks_provenance_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_fixture(root)
            paths = cache_v4_paths(root, "p0-integration")
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            metadata["source_provenance"] = {"source": "tracked_frozen_fixture"}
            metadata["source_provenance_digest"] = hashlib.sha256(
                json.dumps(
                    metadata["source_provenance"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            frozen_path = root / "tracked_frozen_metadata.json"
            frozen_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result = audit_v4_cache(
                root,
                "p0-integration",
                context_bars=4,
                run_u0=False,
                frozen_metadata_path=frozen_path,
            )

        frozen = result["frozen_metadata_validation"]
        self.assertEqual(frozen["status"], "v4_verified")
        self.assertTrue(frozen["content_digests_equal_to_cache_local"])
        self.assertTrue(frozen["body_and_sidecar_rows_equal_to_cache_local"])
        self.assertFalse(frozen["source_provenance_digest_equal_to_cache_local"])
        self.assertFalse(frozen["metadata_file_sha256_equal_to_cache_local"])

    def test_v4_masks_and_all_contract_paths_share_grid_mask_hash_and_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_fixture(root)
            inputs = load_v4_contract_inputs(root, "p0-integration", context_bars=4)
            result = audit_v4_cache(root, "p0-integration", context_bars=4, run_u0=True)

        # P0-A maps the sparse body to the complete sidecar grid.  Row 6 is
        # explicitly absent, while row 8 has an unavailable funding feature.
        self.assertEqual(result["feature_body_rows"], 12)
        self.assertEqual(result["grid_rows"], 13)
        self.assertEqual(result["decision_context_bars"], 4)
        self.assertEqual(result["body_row_eligible"], 10)
        self.assertEqual(result["decision_eligible_rows"], 3)
        # Funding is false at row 10, but Spot and its return are observed;
        # score eligibility intentionally does not inherit that feature mask.
        self.assertEqual(result["score_eligible_rows"], 12)
        self.assertTrue(inputs.score_eligible[10])
        self.assertFalse(inputs.decision_eligible[10])
        self.assertEqual(result["contract_path_counts"]["scheduled_decisions"], 3)
        self.assertEqual(result["contract_path_counts"]["scorable_blocks"], 2)
        self.assertEqual(result["contract_path_counts"]["eligible_decisions"], 1)
        # An outcome gap is scored out but does not erase a causally eligible
        # decision/fill block; the executed position is carried chronologically.
        self.assertEqual(result["contract_path_counts"]["eligible_blocks"], 1)
        self.assertEqual(result["contract_path_counts"]["execution_skipped_blocks"], 2)
        self.assertEqual(result["contract_path_counts"]["excluded_blocks"], 1)
        self.assertEqual(result["contract_path_counts"]["scored_bars"], 8)
        self.assertEqual(result["contract_path_counts"]["filled_blocks"], 0)

        self.assertTrue(result["contract_path_same_scored_mask"])
        self.assertTrue(result["contract_path_same_contract_hash"])
        self.assertTrue(result["contract_path_same_eligibility_mask_hash"])
        self.assertTrue(result["timeline_ok"])
        self.assertTrue(result["u0_run"])
        self.assertEqual(
            result["contract_hash"],
            ActionExecutionContract.canonical().contract_hash,
        )
        self.assertEqual(
            result["contract_path_eligibility_mask_hashes"]["strategy"],
            result["contract_path_eligibility_mask_hashes"]["u0"],
        )
        self.assertEqual(result["p0_status"], {"p0_a": "passed", "p0_b": "partial", "p0_c": "passed"})
        self.assertEqual(
            [(row["decision_t"], row["fill_t_plus_1"], row["returns_end_exclusive"])
             for row in result["timeline_sample"]],
            [(0, 1, 5), (4, 5, 9), (8, 9, 13)],
        )

    def test_oof_same_row_future_perturbation_and_hindsight_inventory_are_blocked_together(self) -> None:
        n_rows = 13
        features = np.arange(n_rows, dtype=np.float64).reshape(-1, 1)
        labels = (features[:, 0] * 0.001).astype(np.float64)
        row_mask = np.ones(n_rows, dtype=bool)

        def fit_predict(x_train, y_train, x_test):
            del x_train, x_test
            return {"prediction": [float(np.mean(y_train))]}

        base = chronological_oof_predict(
            features,
            labels,
            fit_predict=fit_predict,
            horizon=1,
            min_train_size=2,
            row_eligibility_mask=row_mask,
            row_eligibility_provenance={"source": "p0_a_integration_fixture"},
        )
        validate_oof_result(base)
        require_conditional_oof_inputs(
            config={"conditional_oracle_path": True},
            oof_bundle=base,
            caller="p0-integration-test",
        )

        perturbed_labels = labels.copy()
        perturbed_labels[4] += 100_000.0
        perturbed = chronological_oof_predict(
            features,
            perturbed_labels,
            fit_predict=fit_predict,
            horizon=1,
            min_train_size=2,
            row_eligibility_mask=row_mask,
            row_eligibility_provenance={"source": "p0_a_integration_fixture"},
        )
        # The label at t=4 cannot be in the prefix used to predict t=4.  It
        # may affect later origins, which is the intended chronological rule.
        np.testing.assert_array_equal(base["prediction_mask"], perturbed["prediction_mask"])
        np.testing.assert_array_equal(base["predictions"][4], perturbed["predictions"][4])
        self.assertNotEqual(base["predictions"][8, 0], perturbed["predictions"][8, 0])

        contract = ActionExecutionContract.canonical()
        score_eligible = np.ones(n_rows, dtype=bool)
        teacher_base = conditional_oracle_teacher_path(
            base["predictions"][:, 0],
            contract,
            decision_eligible=base["prediction_mask"],
            score_eligible=score_eligible,
        )
        teacher_perturbed = conditional_oracle_teacher_path(
            perturbed["predictions"][:, 0],
            contract,
            decision_eligible=perturbed["prediction_mask"],
            score_eligible=score_eligible,
        )
        np.testing.assert_array_equal(
            teacher_base.scored_mask,
            teacher_perturbed.scored_mask,
        )
        self.assertEqual(teacher_base.decision_deltas[4], teacher_perturbed.decision_deltas[4])
        self.assertEqual(teacher_base.contract_hash, contract.contract_hash)

        # A realized-future U0 path cannot become the current inventory for a
        # conditional transition target, even when hidden under policy_replay.
        u0 = hindsight_upper_bound_path(
            labels,
            contract,
            decision_eligible=np.ones(n_rows, dtype=bool),
            score_eligible=np.ones(n_rows, dtype=bool),
        )
        for source in ("hindsight_teacher", "teacher", "oracle"):
            with self.subTest(source=source):
                with self.assertRaises(TeacherInventoryContractError):
                    validate_current_inventory_source(source)
        with self.assertRaises(TeacherInventoryContractError):
            current_inventory_from_replay(
                source="policy_replay",
                positions=u0.effective_positions,
                benchmark_position=contract.p_start,
                initial_position=contract.p_start,
                provenance={"producer": "hindsight_oracle"},
            )


if __name__ == "__main__":
    unittest.main()
