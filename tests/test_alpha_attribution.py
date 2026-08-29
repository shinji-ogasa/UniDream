import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from unidream.eval.alpha_attribution import (
    FoldSeries,
    _classification_metrics,
    backtest_metrics,
    circular_shift_positions,
    diagnose_saved_artifact,
    evaluate_fold_attribution,
    feature_coverage_for_fold,
    load_timeseries_artifact,
    right_exclusive_mask,
    run_attribution,
    _position_utility_targets,
)
from unidream.eval.backtest import Backtest, compute_costs
from unidream.world_model.train_wm import WorldModelTrainer


def _cfg() -> dict:
    return {
        "data": {"symbol": "BTCUSDT", "interval": "1d"},
        "costs": {"spread_bps": 0.0, "fee_rate": 0.0, "slippage_bps": 0.0},
        "world_model": {
            "return_include_current": False,
            "position_utility_positions": [0.5, 1.0, 1.5],
            "position_utility_horizon": 2,
        },
    }


class AlphaAttributionMetricsTest(unittest.TestCase):
    def test_backtest_metrics_reuses_shared_metric_definitions(self) -> None:
        returns = np.asarray([0.02, -0.01, 0.03, -0.02], dtype=np.float64)
        positions = np.asarray([1.0, 0.5, 1.0, 0.75], dtype=np.float64)
        result = backtest_metrics(returns, positions, _cfg(), benchmark=1.0, execution_delay_bars=1)
        expected = Backtest(
            returns,
            positions,
            spread_bps=0.0,
            fee_rate=0.0,
            slippage_bps=0.0,
            interval="1d",
            benchmark_positions=np.ones_like(returns),
            execution_delay_bars=1,
        ).run()
        self.assertAlmostEqual(result["alpha_excess_pt"], 100.0 * expected.alpha_excess)
        self.assertAlmostEqual(result["maxdd_delta_pt"], 100.0 * expected.maxdd_delta)
        self.assertAlmostEqual(result["sharpe_delta"], expected.sharpe_delta)
        # Effective path is positions[:-1] = [1.0, 0.5, 1.0].  Compatibility
        # turnover omits the initial entry; cost_turnover includes it.
        self.assertAlmostEqual(result["turnover"], 1.0)
        self.assertAlmostEqual(result["cost_turnover"], 2.0)
        self.assertEqual(result["execution_delay_bars"], 1)

    def test_classification_metrics_definition_is_explicit(self) -> None:
        metrics, reasons = _classification_metrics(
            np.asarray([0.9, 0.1, 0.8, 0.2]),
            np.asarray([1, 0, 1, 0]),
        )
        self.assertEqual(reasons, [])
        self.assertAlmostEqual(metrics["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mcc"], 1.0)
        self.assertAlmostEqual(metrics["brier"], 0.025)
        self.assertGreaterEqual(metrics["ece"], 0.0)

    def test_circular_shift_null_is_deterministic(self) -> None:
        positions = np.asarray([0.5, 1.0, 1.5, 0.75], dtype=np.float64)
        np.testing.assert_array_equal(circular_shift_positions(positions, 3), np.roll(positions, 3))
        np.testing.assert_array_equal(
            circular_shift_positions(positions, 3), circular_shift_positions(positions, 3)
        )

    def test_cost_turnover_matches_effective_path_cost_basis(self) -> None:
        returns = np.asarray([0.01, 0.02, -0.01, 0.03], dtype=np.float64)
        positions = np.asarray([1.0, 0.5, 1.0, 0.25], dtype=np.float64)
        cfg = _cfg()
        cfg["costs"] = {"spread_bps": 3.0, "fee_rate": 0.0003, "slippage_bps": 1.0}

        result = backtest_metrics(returns, positions, cfg, execution_delay_bars=1)
        effective_positions = positions[:-1]
        expected_cost_turnover = float(np.abs(np.diff(effective_positions, prepend=0.0)).sum())
        self.assertAlmostEqual(result["cost_turnover"], expected_cost_turnover)

        expected_cost = float(
            compute_costs(
                effective_positions,
                spread_bps=3.0,
                fee_rate=0.0003,
                slippage_bps=1.0,
            ).sum()
        )
        per_unit_cost = 3.0 / 10000.0 / 2.0 + 0.0003 + 1.0 / 10000.0
        self.assertAlmostEqual(expected_cost, expected_cost_turnover * per_unit_cost)


class AlphaAttributionContractTest(unittest.TestCase):
    def test_holdout_cannot_be_used_for_selection(self) -> None:
        item = FoldSeries(
            15,
            np.arange(3, dtype=np.int64),
            np.asarray([0.01, 0.0, -0.01]),
            np.ones(3),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "timeseries.npz"
            np.savez(
                artifact,
                fold_15_time_ns=item.timestamps,
                fold_15_returns=item.returns,
                fold_15_positions=item.positions,
            )
            with self.assertRaisesRegex(ValueError, "reference-only"):
                run_attribution(
                    series=[item],
                    cfg=_cfg(),
                    config_path="synthetic.yaml",
                    artifact_path=artifact,
                    seed=7,
                    output_dir=Path(directory) / "out",
                )

    def test_right_exclusive_loader_drops_only_historical_boundary_endpoint(self) -> None:
        timestamps = pd.date_range("2020-01-01", periods=4, freq="15min").view("int64")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "timeseries.npz"
            np.savez(
                artifact,
                fold_00_time_ns=timestamps,
                fold_00_returns=np.asarray([0.01, 0.02, 0.03, 0.99]),
                fold_00_positions=np.ones(4),
            )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "fold": 0,
                                "test_start": "2020-01-01 00:00:00",
                                "test_end": "2020-01-01 00:45:00",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_timeseries_artifact(artifact)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(loaded[0].returns), 3)
        self.assertTrue(np.all(right_exclusive_mask(loaded[0].timestamps, "2020-01-01", "2020-01-01 00:45")))

    def test_selection_is_validation_only_when_test_path_changes(self) -> None:
        validation_returns = np.asarray([0.02, 0.01, 0.03], dtype=np.float64)
        validation_positions = np.asarray([0.9, 1.0, 1.1], dtype=np.float64)
        base = FoldSeries(
            0,
            np.arange(4, dtype=np.int64),
            np.asarray([0.01, 0.01, -0.01, 0.02]),
            np.asarray([0.8, 1.0, 0.9, 1.1]),
        )
        changed = FoldSeries(0, base.timestamps, np.asarray([9.0, -9.0, 4.0, -4.0]), base.positions)
        first = evaluate_fold_attribution(
            base,
            _cfg(),
            fixed_exposures=[1.0, 1.5],
            lags=[1],
            null_shifts=[1],
            validation_returns=validation_returns,
            validation_positions=validation_positions,
        )
        second = evaluate_fold_attribution(
            changed,
            _cfg(),
            fixed_exposures=[1.0, 1.5],
            lags=[1],
            null_shifts=[1],
            validation_returns=validation_returns,
            validation_positions=validation_positions,
        )
        self.assertEqual(
            first["selection"]["fixed_exposure"]["selected_candidate"],
            second["selection"]["fixed_exposure"]["selected_candidate"],
        )
        self.assertEqual(first["selection"]["actor_mean"]["status"], "selected_on_validation")
        lag_row = next(row for row in first["rows"] if row["method"] == "actor_lag")
        self.assertEqual(lag_row["metrics"]["execution_delay_bars"], 1)

    def test_feature_coverage_marks_zero_missing_ambiguity(self) -> None:
        index = pd.date_range("2020-01-01", periods=4, freq="15min")
        features = pd.DataFrame(
            {
                "funding_rate": [0.0, 0.1, 0.0, 0.1],
                "basis": [0.0, 0.0, 0.0, 0.0],
                "basis_mom": [1.0, 2.0, 3.0, 4.0],
            },
            index=index,
        )
        item = FoldSeries(
            0,
            index.view("int64"),
            np.zeros(4),
            np.ones(4),
            "2020-01-01 00:00:00",
            "2020-01-01 01:00:00",
        )
        coverage = feature_coverage_for_fold(features, item)
        self.assertEqual(coverage["rows"], 4)
        self.assertEqual(coverage["external"]["funding_rate"]["nonzero_count"], 2)
        self.assertEqual(
            coverage["external"]["funding_rate"]["quality_flag"],
            "N/A_zero_vs_missing_indistinguishable",
        )
        self.assertEqual(coverage["external"]["basis_abs"]["quality_flag"], "N/A_missing_column")
        self.assertEqual(coverage["status"], "N/A_missing_external_column")


class AlphaAttributionArtifactDiagnosticTest(unittest.TestCase):
    def test_unavailable_predictive_alignment_is_na_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample_input.npz"
            np.savez(sample, returns=np.asarray([0.01, -0.01, 0.02]), advantage=np.zeros((3, 1)))
            rows = diagnose_saved_artifact(sample, cfg=_cfg())
        self.assertEqual(rows[0]["status"], "N/A")
        self.assertIn("cannot be aligned", rows[0]["reason"])
        self.assertNotIn("metrics", rows[0])

    def test_one_sided_and_position_utility_heads_use_correct_metric_contracts(self) -> None:
        names = np.asarray(
            [
                "wm_pred_vol_h2",
                "wm_pred_position_utility_p0.5",
                "wm_pred_position_utility_p1",
            ],
            dtype=object,
        )
        returns = np.asarray([0.01, -0.02, 0.03, 0.01, -0.01, 0.02], dtype=np.float64)
        advantage = np.asarray(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.1],
                [0.1, 0.2, 0.4],
                [0.3, 0.1, 0.2],
                [0.2, 0.4, 0.1],
                [0.1, 0.2, 0.3],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample_input.npz"
            state = root / "predictive_state.npz"
            np.savez(sample, returns=returns, advantage=advantage)
            np.savez(
                state,
                mean=np.zeros((1, 3), dtype=np.float32),
                std=np.ones((1, 3), dtype=np.float32),
                names=names,
            )
            diagnostic_cfg = _cfg()
            diagnostic_cfg["world_model"]["position_utility_positions"] = [0.5, 1.0]
            rows = diagnose_saved_artifact(
                sample,
                predictive_state_path=state,
                cfg=diagnostic_cfg,
            )
        vol = next(row for row in rows if row["head"] == "wm_pred_vol_h2")
        self.assertIsNone(vol["metrics"]["sign_accuracy"])
        self.assertIn("one-sided", vol["reason"])
        utility = next(row for row in rows if row["head"] == "wm_pred_position_utility_p0.5")
        self.assertEqual(utility["target_type"], "position_utility_regression")
        self.assertNotIn("brier", utility["metrics"])
        self.assertNotIn("balanced_accuracy", utility["metrics"])
        argmax = next(row for row in rows if row["head"] == "position_utility_argmax")
        self.assertEqual(argmax["target_type"], "best_utility_action_from_regression_scores")
        self.assertEqual(argmax["position_name_alignment"], "exact_order_match")
        self.assertIn("target_distribution", argmax["class_summary"])
        self.assertIn("predicted_distribution", argmax["class_summary"])
        self.assertIn("majority_class_accuracy", argmax["metrics"])

    def test_position_utility_targets_match_world_model_generator(self) -> None:
        cfg = _cfg()
        returns = np.asarray([0.01, -0.02, 0.03, 0.01, -0.01, 0.02], dtype=np.float64)
        expected, expected_mask, _positions = _position_utility_targets(returns, cfg)

        # Avoid constructing a neural ensemble: the target generator only
        # depends on these explicit trainer hyperparameters.
        trainer = object.__new__(WorldModelTrainer)
        wm_cfg = cfg["world_model"]
        trainer.position_utility_horizon = int(wm_cfg["position_utility_horizon"])
        trainer.position_utility_positions = [float(x) for x in wm_cfg["position_utility_positions"]]
        trainer.benchmark_position = 1.0
        trainer.position_utility_dd_penalty = float(wm_cfg.get("position_utility_dd_penalty", 1.0))
        trainer.position_utility_dd_improve_reward = float(
            wm_cfg.get("position_utility_dd_improve_reward", 0.0)
        )
        trainer.position_utility_vol_penalty = float(wm_cfg.get("position_utility_vol_penalty", 0.25))
        trainer.position_utility_target_scale = float(wm_cfg.get("position_utility_target_scale", 1.0))
        trainer.cost_rate = 0.0
        actual, actual_mask = trainer._future_position_utility_targets(
            torch.as_tensor(returns, dtype=torch.float32).unsqueeze(0)
        )
        actual_values = actual.detach().cpu().numpy()[0]
        actual_valid = actual_mask.detach().cpu().numpy()[0]
        np.testing.assert_allclose(actual_values[expected_mask], expected[expected_mask], atol=1e-6)
        np.testing.assert_array_equal(
            actual_valid,
            np.repeat(expected_mask[:, None], len(_positions), axis=1),
        )

    def test_ledger_has_provenance_metrics_and_feature_quality(self) -> None:
        index = pd.date_range("2020-01-01", periods=4, freq="15min")
        features = pd.DataFrame(
            {
                "funding_rate": [0.0, 0.1, 0.0, 0.1],
                "basis": [0.0, 0.0, 0.0, 0.0],
                "basis_mom": [1.0, 2.0, 3.0, 4.0],
                "basis_abs": [1.0, 2.0, 3.0, 4.0],
            },
            index=index,
        )
        item = FoldSeries(0, index.view("int64"), np.asarray([0.01, 0.0, -0.01, 0.02]), np.ones(4))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "timeseries.npz"
            np.savez(
                artifact,
                fold_00_time_ns=item.timestamps,
                fold_00_returns=item.returns,
                fold_00_positions=item.positions,
            )
            payload = run_attribution(
                series=[item],
                cfg=_cfg(),
                config_path="synthetic.yaml",
                artifact_path=artifact,
                seed=7,
                fixed_exposures=[1.0, 1.005, 1.01, 1.015],
                # Every requested delay must leave at least one evaluated bar;
                # this four-bar fixture therefore uses only valid sensitivities.
                lags=[1],
                null_shifts=[1],
                output_dir=root / "out",
                feature_frame=features,
            )
            ledger_rows = [json.loads(line) for line in Path(payload["ledger_path"]).read_text().splitlines()]
        self.assertEqual(payload["feature_coverage"]["0"]["rows"], 4)
        self.assertEqual(payload["feature_cache_summary"]["rows"], 4)
        trial = next(row for row in ledger_rows if row["record_type"] == "alpha_attribution_trial")
        for key in (
            "commit_hash",
            "config_sha256",
            "data_contract",
            "data_contract_sha256",
            "data_sha256",
            "fold",
            "seed",
            "selection_status",
            "metrics",
            "artifact_paths",
            "status",
            "feature_coverage",
        ):
            self.assertIn(key, trial)
        self.assertEqual(trial["feature_coverage"]["fold"], 0)
        self.assertEqual(trial["selection_status"], "reference_benchmark")


if __name__ == "__main__":
    unittest.main()
