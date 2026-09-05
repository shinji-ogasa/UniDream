import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.experiments.alpha_dd_search import (
    Candidate, FEATURE_NAMES, aggregate, candidate_universe, fit_predictions,
    make_features, metrics, rule_targets,
    select_development, file_digest, validate_data_artifact,
)


CONTRACT = {"one_way_cost": 0.00055, "borrow_annual": 0.1,
            "max_step": 0.08, "deadband": 0.01}


def bars(n=100, prices=None):
    idx = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    close = np.ones(n) * 100 if prices is None else np.asarray(prices, float)
    open_ = np.r_[100.0, close[:-1]]
    return pd.DataFrame({"open": open_, "close": close,
        "high": np.maximum(open_, close), "low": np.minimum(open_, close),
        "quote_volume": 100.0, "taker_buy_quote": 50.0, "bar_available": True}, index=idx)


class AlphaDDSearchTests(unittest.TestCase):
    def test_bh_identity_and_first_bar_drawdown(self):
        data = bars(8, [80, 81, 83, 85, 88, 84, 89, 90])
        result = metrics(data, np.ones(8), CONTRACT)
        self.assertAlmostEqual(result["total_return"], -0.1)
        self.assertAlmostEqual(result["maxdd"], 0.2)
        self.assertEqual(result["alpha_ex"], 0)
        self.assertEqual(result["maxdd_delta"], 0)
        self.assertEqual(result["trades"], 0)

    def test_next_open_execution_cost_and_missing_fill(self):
        data = bars(8)
        target = np.zeros(8)
        result = metrics(data, target, CONTRACT)
        self.assertEqual(result["trades"], 2)  # Decisions 0/4, fills 1/5.
        self.assertLess(result["total_return"], 0)
        gap = data.copy()
        gap.loc[gap.index[1], ["open", "high", "low", "close"]] = np.nan
        gap.loc[gap.index[1], "bar_available"] = False
        self.assertEqual(metrics(gap, target, CONTRACT)["trades"], 1)
        # Order at the last row never executes inside this window.
        last_only = np.full(8, np.nan)
        last_only[-1] = 0
        self.assertEqual(metrics(data, last_only, CONTRACT)["trades"], 0)

    def test_outcome_gap_keeps_prior_fill_and_gap_wealth(self):
        data = bars(8, [100, 100, 90, 90, 90, 90, 90, 90])
        target = np.full(8, np.nan)
        target[0] = 0.5
        reference = metrics(data, target, CONTRACT)
        data.loc[data.index[2], ["open", "high", "low", "close"]] = np.nan
        data.loc[data.index[2], "bar_available"] = False
        changed = metrics(data, target, CONTRACT)
        self.assertEqual(changed["trades"], reference["trades"])
        self.assertAlmostEqual(changed["total_return"], reference["total_return"])
        self.assertAlmostEqual(changed["fees_initial_equity_units"], reference["fees_initial_equity_units"])

    def test_missing_close_on_fill_bar_cannot_cancel_order(self):
        data = bars(8)
        target = np.zeros(8)
        reference = metrics(data, target, CONTRACT)
        data.loc[data.index[1], ["high", "low", "close"]] = np.nan
        data.loc[data.index[1], "bar_available"] = False
        changed = metrics(data, target, CONTRACT)
        self.assertEqual(changed["trades"], reference["trades"])
        self.assertAlmostEqual(changed["total_return"], reference["total_return"])
        self.assertAlmostEqual(changed["turnover"], reference["turnover"])

    def test_rebalance_is_not_free_and_borrow_is_charged(self):
        data = bars(48, np.exp(np.arange(48) * 0.003) * 100)
        leveraged = metrics(data, np.full(48, 1.12), CONTRACT)
        self.assertGreater(leveraged["borrow_initial_equity_units"], 0)
        self.assertGreater(leveraged["turnover"], 0.1)
        self.assertGreater(leveraged["fees_initial_equity_units"], 0)

    def test_features_and_intent_do_not_see_current_or_future(self):
        n = 96 * 100
        original = bars(n, 100 * np.exp(np.sin(np.arange(n) / 96) * 0.1 + np.arange(n) * 0.00001))
        boundary = 96 * 95
        mutated = original.copy()
        mutated.iloc[boundary:, mutated.columns.get_indexer(["open", "high", "low", "close"])] *= 5
        mutated.iloc[boundary + 4, mutated.columns.get_indexer(["open", "high", "low", "close"])] = np.nan
        before, after = make_features(original), make_features(mutated)
        np.testing.assert_allclose(before.iloc[:boundary + 1], after.iloc[:boundary + 1], equal_nan=True)
        for c in candidate_universe():
            if c.family not in ("ridge", "hgb", "logistic"):
                np.testing.assert_allclose(rule_targets(c, before)[:boundary + 1],
                    rule_targets(c, after)[:boundary + 1], equal_nan=True)

    def test_selection_requires_both_metrics_and_empty_is_not_pass(self):
        summary = {"a": aggregate([{"alpha_ex": .05, "maxdd_delta": .02}]),
                   "b": aggregate([{"alpha_ex": .02, "maxdd_delta": -.02}]),
                   "c": aggregate([{"alpha_ex": -.01, "maxdd_delta": -.05}])}
        self.assertEqual(select_development(summary), "b")
        self.assertTrue(summary["b"]["minimum_target_pass"])
        self.assertFalse(summary["b"]["preferred_target_pass"])
        self.assertFalse(summary["a"]["minimum_target_pass"])
        with self.assertRaises(ValueError):
            aggregate([])
        self.assertEqual(len(candidate_universe()), 83)

    def test_fit_does_not_use_validation_or_test_outcomes(self):
        n = 30000
        data = bars(n, 100 * np.exp(np.sin(np.arange(n) / 333) * .1))
        rng = np.random.default_rng(77)
        features = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))),
                                index=data.index, columns=FEATURE_NAMES)
        fold = {"fold": 0, "train_start": data.index[0], "train_end": data.index[24000],
                "val_start": data.index[24000], "test_end": data.index[-1]}
        modified = data.copy()
        modified.loc[modified.index >= fold["train_end"], "close"] *= 5
        with tempfile.TemporaryDirectory() as tmp:
            first, proof = fit_predictions(Candidate("ridge", 7), features, data, fold, Path(tmp) / "a")
            second, other = fit_predictions(Candidate("ridge", 7), features, modified, fold, Path(tmp) / "b")
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.isfinite(first[24000:-1]).all())
        self.assertEqual(proof["model_sha256"], other["model_sha256"])
        self.assertLess(pd.Timestamp(proof["fit_last_target_end_exclusive"]), fold["train_end"])

    def test_data_masks_and_hashes_are_bound_before_research(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = bars(8).drop(columns="bar_available")
            path, masks, ledger = root / "spot.parquet", root / "mask.parquet", root / "ledger.jsonl"
            data.to_parquet(path)
            availability = pd.DataFrame({"spot_bar_observed": True}, index=data.index)
            availability.to_parquet(masks)
            ledger.write_text('{}\n')
            sidecar = {"kind": "alpha_dd_spot_15m_artifact_sha256", "status": "complete",
                       "artifact_sha256": file_digest(path), "availability_path": str(masks),
                       "availability_sha256": file_digest(masks), "source_ledger_path": str(ledger),
                       "source_ledger_sha256": file_digest(ledger), "rows": len(data),
                       "columns": data.columns.tolist(), "availability_column": "spot_bar_observed"}
            sidecar_path = path.with_suffix(".sha256.json")
            sidecar_path.write_text(json.dumps(sidecar))
            self.assertEqual(validate_data_artifact(path)["status"], "complete")
            availability.iloc[2, 0] = False
            availability.to_parquet(masks)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                validate_data_artifact(path)
            sidecar["availability_sha256"] = file_digest(masks)
            sidecar_path.write_text(json.dumps(sidecar))
            with self.assertRaisesRegex(ValueError, "observed mask"):
                validate_data_artifact(path)


if __name__ == "__main__":
    unittest.main()
