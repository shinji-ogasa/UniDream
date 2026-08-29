import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.data.dataset import WFOSplit
from unidream.eval.backtest import Backtest
from unidream.eval.forecast_tournament import (
    DEV_FOLDS,
    EXPECTED_FEATURES,
    EXTERNAL_FEATURES,
    OHLCV_DERIVED_FEATURES,
    DevelopmentData,
    PolicyParams,
    aggregate_candidate_gate,
    feature_ablation_columns,
    future_targets,
    load_development_data,
    policy_positions,
    run_tournament,
    validate_requested_folds,
    _feature_quality,
    _assert_wfo_split_contract,
)


def _features(index: pd.DatetimeIndex) -> pd.DataFrame:
    names = [*OHLCV_DERIVED_FEATURES, *EXTERNAL_FEATURES]
    values = np.linspace(-1.0, 1.0, len(index) * len(names), dtype=np.float64).reshape(len(index), len(names))
    return pd.DataFrame(values, index=index, columns=names)


class ForecastTournamentContractTest(unittest.TestCase):
    def test_fold_allow_list_rejects_holdout_and_future_folds(self) -> None:
        self.assertEqual(validate_requested_folds(None), DEV_FOLDS)
        self.assertEqual(validate_requested_folds([8, 0, 2, 2]), (0, 2, 8))
        for forbidden in (15, 23, 24):
            with self.assertRaisesRegex(ValueError, "development-only"):
                validate_requested_folds([forbidden])

    def test_future_targets_are_t_plus_one_through_t_plus_h(self) -> None:
        returns = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        target, mask = future_targets(returns, [2], target_kind="return")
        np.testing.assert_allclose(target[:, 0], [5.0, 7.0, 9.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask[:, 0], [True, True, True, False, False])
        risk, risk_mask = future_targets(returns, [2], target_kind="risk")
        np.testing.assert_allclose(risk[:3, 0], [np.sqrt(6.5), np.sqrt(12.5), np.sqrt(20.5)])
        np.testing.assert_array_equal(risk_mask, mask)

    def test_future_target_offset_excludes_current_return(self) -> None:
        returns = np.asarray([100.0, 1.0, 2.0, 3.0, 4.0])
        target, mask = future_targets(returns, [2], target_kind="return")
        np.testing.assert_allclose(target[:, 0], [3.0, 5.0, 7.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask[:, 0], [True, True, True, False, False])
        mutated = returns.copy()
        mutated[0] = -999.0
        mutated_target, _ = future_targets(mutated, [2], target_kind="return")
        self.assertEqual(float(mutated_target[0, 0]), float(target[0, 0]))

    def test_fixed_delay_aligns_t_plus_one_target_to_next_return(self) -> None:
        # A decision from feature/forecast row t must be applied to returns[t+1]
        # when targets exclude returns[t]. Backtest delay=1 has that mapping.
        returns = np.asarray([0.0, 1.0e-8, 0.0, 0.0])
        positions = np.asarray([1.12, 0.50, 0.50, 0.50])
        result = Backtest(
            returns,
            positions,
            spread_bps=0.0,
            fee_rate=0.0,
            slippage_bps=0.0,
            benchmark_positions=np.ones(len(returns)),
            execution_delay_bars=1,
        ).run()
        self.assertEqual(len(result.pnl_series), 3)
        self.assertAlmostEqual(float(result.pnl_series[0]), 1.12e-8, places=16)

    def test_policy_is_causal_and_respects_bounds(self) -> None:
        params = PolicyParams(threshold=0.5, overlay_magnitude=0.12, hysteresis=0.25, min_hold=2, execution_delay=0)
        prefix = np.asarray([0.0, 1.0, 1.0, 0.0])
        original = policy_positions(prefix, params)
        changed = policy_positions(np.asarray([0.0, 1.0, 1.0, 99.0]), params)
        np.testing.assert_array_equal(original[:3], changed[:3])
        self.assertTrue(np.all(original >= 0.50))
        self.assertTrue(np.all(original <= 1.12))

    def test_feature_quality_keeps_zero_and_missing_distinct(self) -> None:
        index = pd.date_range("2020-01-01", periods=3, freq="h")
        frame = pd.DataFrame(
            {
                "funding_rate": [0.0, 0.1, np.nan],
                "basis": [0.0, 0.0, 0.2],
            },
            index=index,
        )
        quality = _feature_quality(frame, feature_set="full17", fold=0, split_name="validation")
        funding = quality["external"]["funding_rate"]
        self.assertEqual(funding["zero_count"], 1)
        self.assertEqual(funding["missing_count"], 1)
        self.assertEqual(funding["nonzero_count"], 1)
        self.assertEqual(funding["quality_flag"], "N/A_zero_vs_missing_indistinguishable")

    def test_load_development_data_rejects_data_at_cutoff(self) -> None:
        index = pd.date_range("2024-01-01", periods=4, freq="h")
        features = _features(index)
        returns = pd.DataFrame({"returns": np.ones(len(index))}, index=index)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "features.parquet"
            returns_path = root / "returns.parquet"
            features.to_parquet(feature_path)
            returns.to_parquet(returns_path)
            with self.assertRaisesRegex(ValueError, "development cutoff"):
                load_development_data(feature_path, returns_path, folds=[0])

    def test_feature_ablation_schema_is_fixed(self) -> None:
        frame = _features(pd.date_range("2020-01-01", periods=2, freq="h"))
        columns = feature_ablation_columns(frame)
        self.assertEqual(columns["ohlcv13"], list(OHLCV_DERIVED_FEATURES))
        self.assertEqual(columns["full17"], list(EXPECTED_FEATURES))
        with self.assertRaisesRegex(ValueError, "exactly match EXPECTED_FEATURES"):
            feature_ablation_columns(frame.assign(unexpected=0.0))
        with self.assertRaisesRegex(ValueError, "exactly match EXPECTED_FEATURES"):
            feature_ablation_columns(frame.loc[:, list(reversed(frame.columns))])

    def test_wfo_split_matches_explicit_configured_periods(self) -> None:
        start = pd.Timestamp("2018-01-01")
        split = WFOSplit(
            fold_idx=0,
            train_start=start,
            train_end=start + pd.DateOffset(years=2),
            val_start=start + pd.DateOffset(years=2),
            val_end=start + pd.DateOffset(years=2, months=3),
            test_start=start + pd.DateOffset(years=2, months=3),
            test_end=start + pd.DateOffset(years=2, months=6),
        )
        _assert_wfo_split_contract(split, train_years=2, val_months=3, test_months=3)
        bad = WFOSplit(
            **{**split.__dict__, "test_end": split.test_end + pd.Timedelta(hours=1)}
        )
        with self.assertRaisesRegex(ValueError, "configured WFO periods"):
            _assert_wfo_split_contract(bad, train_years=2, val_months=3, test_months=3)


class ForecastTournamentRunTest(unittest.TestCase):
    def test_run_records_train_validation_test_contract_and_gate(self) -> None:
        index = pd.date_range("2018-01-01", periods=120, freq="h")
        features = _features(index)
        returns = pd.Series(np.sin(np.arange(len(index)) / 7.0) * 0.002, index=index, name="returns")
        split = WFOSplit(
            fold_idx=0,
            train_start=index[0],
            train_end=index[60],
            val_start=index[60],
            val_end=index[90],
            test_start=index[90],
            test_end=index[-1] + pd.Timedelta(hours=1),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "features.parquet"
            returns_path = root / "returns.parquet"
            features.to_parquet(feature_path)
            returns.to_frame().to_parquet(returns_path)
            data = DevelopmentData(features, returns, (split,), feature_path, returns_path)
            cfg = {
                "data": {"symbol": "BTCUSDT", "interval": "1h"},
                "costs": {"spread_bps": 0.0, "fee_rate": 0.0, "slippage_bps": 0.0},
            }
            result = run_tournament(
                data=data,
                cfg=cfg,
                config_path="synthetic.yaml",
                horizons=[4, 16],
                policy_horizon=16,
                feature_sets=["ohlcv13"],
                candidates=["causal_trend_vol_rule"],
                execution_grid=[0, 1, 16],
                output_dir=root / "out",
            )
            self.assertEqual(result["folds"], [0])
            self.assertEqual(result["selection_contract"].split(";")[0], "fit train only")
            self.assertEqual(len(result["rows"]), 1)
            row = result["rows"][0]
            self.assertEqual(row["fit_split"], "train")
            self.assertEqual(row["selection_split"], "validation")
            self.assertEqual(row["report_split"], "development_test")
            self.assertTrue(row["test_is_report_only"])
            self.assertEqual(result["execution_delay_bars"], 1)
            self.assertEqual(row["policy"]["execution_delay"], 1)
            self.assertEqual(
                row["validation_policy_selection"]["execution_delay_selection"],
                "fixed_operational_contract",
            )
            self.assertEqual(set(row["test_economics"]["lags"]), {"1", "16"})
            self.assertEqual(set(row["test_economics"]["nulls"]), {"1", "16", "64"})
            self.assertTrue(result["gate"])
            ledger = [json.loads(line) for line in (root / "out" / "forecast_tournament_ledger.jsonl").read_text().splitlines()]
            self.assertTrue(all(item["fold"] is None or item["fold"] == 0 for item in ledger))
            selection_records = [item for item in ledger if item["record_type"] != "feature_coverage"]
            self.assertTrue(all(item["selection_split"] == "validation" for item in selection_records))

    def test_gate_requires_positive_timing_and_alpha(self) -> None:
        rows = []
        for fold, ic, alpha, timing in ((0, 0.1, 0.2, 0.1), (2, -0.2, -0.3, -0.2), (8, -0.1, -0.1, -0.2)):
            rows.append(
                {
                    "feature_set": "full17",
                    "candidate": "synthetic",
                    "test_return_metrics": [{"horizon": 16, "metrics": {"spearman_ic": ic}}],
                    "test_economics": {
                        "dynamic": {"alpha_excess_pt": alpha, "maxdd_delta_pt": 0.1, "turnover": 1.0},
                        "constant": {"maxdd_delta_pt": 0.1},
                        "timing_increment_alpha_excess_pt": timing,
                    },
                    "fold": fold,
                }
            )
        gate = aggregate_candidate_gate(rows)
        self.assertEqual(gate[0]["status"], "fail")
        self.assertIn("ic_sign_stable", gate[0]["failure_reasons"])
        self.assertIn("median_alpha_excess_positive", gate[0]["failure_reasons"])

    def test_gate_requires_temporal_null_superiority_with_fixed_margin(self) -> None:
        rows = []
        for fold in (0, 2, 8):
            rows.append(
                {
                    "feature_set": "ohlcv13",
                    "candidate": "synthetic",
                    "test_return_metrics": [{"horizon": 16, "metrics": {"spearman_ic": 0.1}}],
                    "test_economics": {
                        "dynamic": {"alpha_excess_pt": 0.30, "maxdd_delta_pt": 0.0, "turnover": 1.0},
                        "constant": {"alpha_excess_pt": 0.10, "maxdd_delta_pt": 0.0},
                        "lags": {
                            "1": {"alpha_excess_pt": 0.29},
                            "16": {"alpha_excess_pt": 0.10},
                        },
                        "nulls": {
                            "1": {"alpha_excess_pt": 0.29},
                            "16": {"alpha_excess_pt": 0.29},
                            "64": {"alpha_excess_pt": 0.10},
                        },
                    },
                    "fold": fold,
                }
            )
        gate = aggregate_candidate_gate(rows)
        self.assertEqual(gate[0]["status"], "pass")
        self.assertEqual(gate[0]["timing_superiority"]["null_shift64"]["win_folds"], 3)
        rows[0]["test_economics"]["nulls"]["64"]["alpha_excess_pt"] = 0.30
        rows[2]["test_economics"]["nulls"]["64"]["alpha_excess_pt"] = 0.30
        failed = aggregate_candidate_gate(rows)[0]
        self.assertEqual(failed["status"], "fail")
        self.assertIn("timing_beats_null_shift64", failed["failure_reasons"])
