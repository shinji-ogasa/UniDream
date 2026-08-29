import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.data.dataset import WFOSplit
from unidream.eval.backtest import validate_execution_delay
from unidream.eval.forecast_context_tournament import (
    ALL_CANDIDATES,
    WAVE_FOLDS,
    _future_regression_targets,
    _strict_horizons,
    _strict_wave_folds,
    aggregate_context_gate,
    aligned_timing_economics,
    build_causal_context,
    classifier_metrics,
    context_eligibility_mask,
    context_prefix_is_causal,
    continuous_overlay_positions,
    future_downside_targets,
    future_target_eligibility_mask,
    run_context_tournament,
)
from unidream.eval.forecast_tournament import (
    EXPECTED_FEATURES,
    OHLCV_DERIVED_FEATURES,
    DevelopmentData,
)


def _features(index: pd.DatetimeIndex) -> pd.DataFrame:
    values = np.zeros((len(index), len(EXPECTED_FEATURES)), dtype=np.float64)
    values[:, 0] = np.arange(len(index), dtype=np.float64)
    values[:, OHLCV_DERIVED_FEATURES.index("close_ret")] = np.arange(
        len(index), dtype=np.float64
    )
    return pd.DataFrame(values, index=index, columns=EXPECTED_FEATURES)


def _split(fold: int, start: pd.Timestamp) -> WFOSplit:
    return WFOSplit(
        fold_idx=fold,
        train_start=start,
        train_end=start + pd.DateOffset(years=2),
        val_start=start + pd.DateOffset(years=2),
        val_end=start + pd.DateOffset(years=2, months=3),
        test_start=start + pd.DateOffset(years=2, months=3),
        test_end=start + pd.DateOffset(years=2, months=6),
    )


class ForecastContextContractTest(unittest.TestCase):
    def test_context_has_fixed_lag_rolling_schema_and_is_causal(self) -> None:
        index = pd.date_range("2020-01-01", periods=320, freq="15min")
        features = _features(index)
        context = build_causal_context(features.loc[:, list(OHLCV_DERIVED_FEATURES)])
        expected_columns = len(OHLCV_DERIVED_FEATURES) * (
            1 + len((1, 4, 16, 64)) + 3 * len((4, 16, 64, 256))
        )
        self.assertEqual(context.shape, (len(index), expected_columns))
        self.assertTrue(context_prefix_is_causal(features.loc[:, list(OHLCV_DERIVED_FEATURES)], cutoff=280))
        self.assertAlmostEqual(
            float(context.loc[index[280], "rolling_slope_4__close_ret"]),
            1.0,
            places=12,
        )

    def test_context_slope_uses_same_exact_window_as_rolling_mean(self) -> None:
        index = pd.date_range("2020-01-01", periods=300, freq="15min")
        features = _features(index).loc[:, ["close_ret"]]
        context = build_causal_context(features)
        row = index[270]
        historical_values = features["close_ret"].iloc[270 - 4 : 270].to_numpy()
        self.assertAlmostEqual(
            float(context.loc[row, "rolling_mean_4__close_ret"]),
            float(historical_values.mean()),
            places=12,
        )
        self.assertAlmostEqual(
            float(context.loc[row, "rolling_slope_4__close_ret"]),
            1.0,
            places=12,
        )

    def test_gap_invalidates_context_and_future_target_windows(self) -> None:
        index = pd.date_range("2020-01-01", periods=9, freq="15min").delete(5)
        context_mask = context_eligibility_mask(index, history=2)
        np.testing.assert_array_equal(
            context_mask,
            [False, False, True, True, True, False, False, True],
        )
        target_mask = future_target_eligibility_mask(index, [2])[:, 0]
        np.testing.assert_array_equal(
            target_mask,
            [True, True, True, False, False, True, False, False],
        )
        returns = np.arange(len(index), dtype=np.float64)
        _, downside_mask = future_downside_targets(returns, [2], timestamps=index)
        np.testing.assert_array_equal(downside_mask[:, 0], target_mask)
        targets, masks = _future_regression_targets(
            returns,
            index,
            (2,),
            target_kind="return",
        )
        self.assertEqual(float(targets[0, 0]), 3.0)
        np.testing.assert_array_equal(masks[:, 0], target_mask)

    def test_strict_horizon_and_fold_contract_rejects_coercion_or_incomplete_set(self) -> None:
        for invalid in ([0], [-1], [1.5], [True]):
            with self.assertRaises(ValueError):
                _strict_horizons(invalid)
        with self.assertRaises(ValueError):
            _strict_wave_folds([_split(0, pd.Timestamp("2018-01-01"))])
        with self.assertRaises(ValueError):
            _strict_wave_folds(
                [_split(0, pd.Timestamp("2018-01-01")), _split(0, pd.Timestamp("2018-01-01")), _split(8, pd.Timestamp("2018-01-01"))]
            )

    def test_execution_delay_and_timing_inputs_are_strict_and_common_window_is_used(self) -> None:
        cfg = {"data": {"interval": "15m"}, "costs": {"spread_bps": 0.0, "fee_rate": 0.0, "slippage_bps": 0.0}}
        returns = np.linspace(-0.001, 0.001, 40)
        positions = np.linspace(0.5, 1.1, 40)
        economics = aligned_timing_economics(returns, positions, 0.75, cfg)
        self.assertEqual(economics["common_evaluation_start_bars"], 17)
        self.assertEqual(economics["common_evaluation_rows"], 23)
        self.assertEqual(economics["dynamic"]["execution_delay_bars"], 0)
        self.assertEqual(economics["lags"]["16"]["execution_delay_bars"], 0)
        spike_returns = np.zeros(40)
        spike_returns[17] = 1.0e-6
        spike_positions = np.full(40, 0.75)
        spike_positions[16] = 1.12
        spike = aligned_timing_economics(spike_returns, spike_positions, 0.75, cfg)
        self.assertGreater(
            spike["dynamic"]["total_return_pt"],
            spike["constant"]["total_return_pt"],
        )
        steady = aligned_timing_economics(
            np.zeros(40),
            np.full(40, 0.75),
            0.75,
            {
                "data": {"interval": "15m"},
                "costs": {"spread_bps": 5.0, "fee_rate": 0.0004, "slippage_bps": 2.0},
            },
        )
        self.assertEqual(steady["dynamic"]["evaluation_initial_position"], 0.75)
        self.assertEqual(steady["dynamic"]["cost_turnover"], 0.0)
        self.assertEqual(steady["dynamic"]["n_trades"], 0)
        for invalid in (1.5, True):
            with self.assertRaises(ValueError):
                aligned_timing_economics(returns, positions, 0.75, cfg, execution_delay=invalid)
        with self.assertRaises(ValueError):
            aligned_timing_economics(returns, positions, 0.75, cfg, lags=(1.5,))
        with self.assertRaises(ValueError):
            aligned_timing_economics(returns, positions, 0.75, cfg, lags=(True,))
        with self.assertRaises(ValueError):
            aligned_timing_economics(returns, positions, 0.75, cfg, null_shifts=(1.5,))
        with self.assertRaises(ValueError):
            aligned_timing_economics(returns, positions, 0.75, cfg, null_shifts=(False,))
        with self.assertRaises(ValueError):
            aligned_timing_economics(np.ones(17), np.ones(17), 0.75, cfg)
        with self.assertRaises(ValueError):
            validate_execution_delay(1.5)

    def test_ineligible_overlay_row_resets_to_validation_constant(self) -> None:
        from unidream.eval.forecast_tournament import PolicyParams

        params = PolicyParams(
            threshold=0.0,
            overlay_magnitude=0.12,
            hysteresis=0.0,
            min_hold=0,
            execution_delay=1,
        )
        positions = continuous_overlay_positions(
            [1.0, 1.0, np.nan, -1.0],
            params,
            benchmark=0.75,
            eligible_mask=[True, True, False, True],
        )
        self.assertEqual(float(positions[2]), 0.75)
        self.assertGreaterEqual(float(positions.min()), 0.50)
        self.assertLessEqual(float(positions.max()), 1.12)

    def test_classifier_one_class_auc_is_na_with_reason_not_zero(self) -> None:
        result = classifier_metrics(
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [True, True, True],
            target_threshold=0.5,
            split="validation",
            horizon=4,
        )
        self.assertIsNone(result["metrics"]["auc"])
        self.assertIn("one class", result["reason"])
        self.assertIsNotNone(result["metrics"]["brier"])

    def test_gate_fails_closed_for_missing_fold(self) -> None:
        economics = {
            "dynamic": {"alpha_excess_pt": 1.0, "maxdd_delta_pt": 0.0, "turnover": 1.0},
            "constant": {"alpha_excess_pt": 0.5, "maxdd_delta_pt": 0.0, "turnover": 1.0},
            "lags": {"1": {"alpha_excess_pt": 0.4}, "16": {"alpha_excess_pt": 0.4}},
            "nulls": {"1": {"alpha_excess_pt": 0.4}, "16": {"alpha_excess_pt": 0.4}, "64": {"alpha_excess_pt": 0.4}},
        }
        rows = [
            {
                "feature_set": "ohlcv13",
                "candidate": ALL_CANDIDATES[0],
                "fold": fold,
                "selected_horizon": 4,
                "test_return_metrics": [{"horizon": 4, "metrics": {"spearman_ic": 0.2}}],
                "test_economics": economics,
            }
            for fold in (0, 2)
        ]
        gate = aggregate_context_gate(rows)
        self.assertEqual(len(gate), 1)
        self.assertFalse(gate[0]["criteria"]["complete_development_folds"])
        self.assertFalse(gate[0]["criteria"]["unique_development_folds"])
        self.assertEqual(gate[0]["status"], "fail")

    def test_configured_delay_rejects_float_before_any_screen_work(self) -> None:
        index = pd.date_range("2018-01-01", periods=4, freq="15min")
        features = _features(index)
        returns = pd.Series(np.zeros(len(index)), index=index, name="returns")
        splits = tuple(_split(fold, pd.Timestamp("2018-01-01")) for fold in WAVE_FOLDS)
        data = DevelopmentData(
            features,
            returns,
            splits,
            Path("features.parquet"),
            Path("returns.parquet"),
        )
        for invalid in (1.5, True):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "execution_delay_bars"):
                    run_context_tournament(
                        data=data,
                        cfg={"eval": {"forecast_execution_delay_bars": invalid}},
                        config_path="synthetic.yaml",
                        output_dir=Path(directory) / "out",
                    )


if __name__ == "__main__":
    unittest.main()
