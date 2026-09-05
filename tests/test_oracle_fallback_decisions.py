import copy
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unidream.experiments.oracle_fallback_decisions import (
    CONTROL_IDS, FOLDS, MEANS, NEW_IDS, action_masks, check_action_support,
    check_trace_support, paired_summaries, validate_config,
)


def fixture():
    rows = []
    for i, fold in enumerate(FOLDS):
        trend = "bull" if i < 2 else "bear" if i < 6 else "sideways"
        for cid in CONTROL_IDS + NEW_IDS:
            delta = (2. if i % 2 == 0 else -6.) if cid in NEW_IDS else 0.
            rows.append({"fold": fold, "candidate_id": cid, "regime": {"trend": trend},
                         "rows": 1 if i % 2 == 0 else 100,
                         **{cost: {metric: delta * factor * multiplier for metric, factor in
                            {"alpha_ex": 1., "maxdd_delta": -2., "turnover": 3., "trades": 4.,
                             "fees_initial_equity_units": .5, "borrow_initial_equity_units": .25}.items()}
                            for cost, multiplier in (("base", 1.), ("stress_2x", 10.))}})
    return rows


class OracleFallbackDecisionsTests(unittest.TestCase):
    def test_registration_does_not_allow_selecting_a_mean_or_weakening_support(self):
        cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/oracle_fallback_decisions_20260905.yaml").read_text())
        validate_config(cfg)
        for key, value in {"mean_sources": ["perp_delay0_scaled"], "variance_source": "perp_delay0_scaled",
                "fallback_target": .5, "policy": "point", "utility_cost_multiplier": 1,
                "minimum_quarters_per_regime": 2, "fallback_eligible_rows": 331,
                "score_rows": 2586, "missing_current_open_rows": 0,
                "adaptive_prior_policy_names_observed": 0}.items():
            with self.subTest(key=key):
                changed = copy.deepcopy(cfg); changed[key] = value
                with self.assertRaisesRegex(ValueError, "unregistered"):
                    validate_config(changed)

    def test_action_support_separates_missing_forecast_from_missing_current_open(self):
        index = pd.date_range("2021-01-01", periods=49, freq="15min", tz="UTC")
        opens = np.ones(49); opens[48] = np.nan
        inference = np.zeros(49, bool); inference[0] = True
        masks = action_masks(index, opens, inference)
        self.assertEqual(np.flatnonzero(masks["learned_eligible"]).tolist(), [0])
        self.assertEqual(np.flatnonzero(masks["fallback_eligible"]).tolist(), [24])
        self.assertEqual(np.flatnonzero(masks["missing_current_open"]).tolist(), [48])
        # Missing next-open must not cancel an intent that is causally admissible.
        opens[25] = np.nan
        changed = action_masks(index, opens, inference)
        np.testing.assert_array_equal(changed["fallback_eligible"], masks["fallback_eligible"])

    def test_action_guard_preserves_valid_hold_and_only_allows_registered_fallback(self):
        index = pd.date_range("2021-01-01", periods=49, freq="15min", tz="UTC")
        inference = np.zeros(49, bool); inference[0] = True
        opens = np.ones(49); opens[48] = np.nan
        masks = action_masks(index, opens, inference)
        targets = np.full(49, np.nan); targets[24] = 1.
        check_action_support(targets, masks)  # valid learned hold at zero remains NaN
        for slot, value in ((1, 1.), (48, 1.), (24, 1.12), (24, np.nan), (0, np.inf)):
            invalid = targets.copy(); invalid[slot] = value
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                check_action_support(invalid, masks)

    def test_masks_reject_misaligned_or_unscheduled_forecasts(self):
        index = pd.date_range("2021-01-01", periods=25, freq="15min", tz="UTC")
        for inference in (np.ones(24, bool), np.zeros(25, float), np.ones(25, bool)):
            with self.assertRaises(ValueError):
                action_masks(index, np.ones(25), inference)

    def test_trace_cannot_claim_a_forecast_or_score_for_fallback(self):
        index = pd.date_range("2021-01-01", periods=49, freq="15min", tz="UTC")
        inference = np.zeros(49, bool); inference[0] = True
        opens = np.ones(49); opens[48] = np.nan
        masks = action_masks(index, opens, inference)
        targets = np.full(49, np.nan); targets[24] = 1.
        d = {"learned_decision_count": 1, "fallback_decision_count": 1,
             "missing_open_decision_count": 1, "hold_decision_count": 1,
             "decision_masks": {"learned": masks["learned_eligible"].tolist(),
                 "fallback": masks["fallback_eligible"].tolist(),
                 "missing_open": masks["missing_current_open"].tolist(),
                 "hold": masks["learned_eligible"].tolist()},
             "decision_trace": {"bar_indices": [0, 24], "reasons": ["learned", "forecast_unavailable"],
                 "targets": [None, 1.], "known_open_nav": [1., 1.], "known_open_exposure": [1., 1.],
                 "estimated_utility_gain_over_hold": [0., None], "estimated_trade_turnover": [0., None]}}
        check_trace_support(targets, masks, d)
        for key, value in (("reasons", ["learned", "learned"]), ("targets", [None, None]),
                           ("estimated_utility_gain_over_hold", [0., .1]), ("bar_indices", [0, 23])):
            changed = copy.deepcopy(d); changed["decision_trace"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_trace_support(targets, masks, changed)

    def test_exact_seven_pairs_use_equal_quarter_not_row_weighted_difference(self):
        rows = fixture(); self.assertEqual(len(rows), 128)
        result = paired_summaries(rows); self.assertEqual(len(result), 7)
        for mean in MEANS:
            pair = result[mean + "_utility_risk1_fallback_bh_vs_" + mean + "_utility_risk1"]
            for regime, quarters in (("all", 8), ("bull", 2), ("bear", 4), ("sideways", 2)):
                r = pair["regimes"][regime]
                self.assertEqual(r["quarters"], quarters)
                self.assertEqual(r["base"]["alpha_ex"], -2.)
                self.assertEqual(r["base"]["maxdd_delta"], 4.)
                self.assertEqual(r["stress_2x"]["alpha_ex"], -20.)
                self.assertNotAlmostEqual(r["base"]["alpha_ex"], -598 / 101)

    def test_pairs_reject_missing_duplicate_unknown_and_mismatched_regimes(self):
        for fault in ("missing", "duplicate", "unknown", "regime"):
            rows = fixture()
            if fault == "missing": rows.pop()
            elif fault == "duplicate": rows.append(copy.deepcopy(rows[0]))
            elif fault == "unknown": rows[0]["candidate_id"] = "unregistered"
            else: rows[0]["regime"] = {"trend": "bear"}
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                paired_summaries(rows)


if __name__ == "__main__":
    unittest.main()
