import copy
import json
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.oracle_derivative_delay_features import make_delayed_perp_groups
from unidream.experiments.oracle_derivative_features import make_derivative_groups
from unidream.experiments.oracle_frontier_features import make_feature_groups
from unidream.experiments.oracle_risk_calibration import trailing_variances
from unidream.experiments.wm_rl_inputs import (
    MarketSequenceDataset, apply_normalizer, build_market_inputs, build_inference_inputs, fit_normalizer,
    mask_digest, sequence_masks, target_horizon_mask,
)


def index(n):
    return pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC", name="bar_open_ts")


def bars(n=9000):
    t = np.arange(n)
    close = 100 * np.exp(.00003 * t + .005 * np.sin(t / 13))
    quote = 1000 + .1 * t
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * .998,
        "close": close, "volume": 10., "quote_volume": quote,
        "taker_buy_quote": quote * (.5 + .1 * np.sin(t / 9)), "n_trades": 10.}, index=index(n))


class MarketInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spot = bars()
        cls.um = bars()
        cls.cutoff = cls.spot.index[-1] + pd.Timedelta(minutes=15)
        cls.built = build_market_inputs(cls.spot, cls.um, cutoff=cls.cutoff)

    def test_exact_inherited_groups_common_components_and_nominal_history(self):
        original = make_feature_groups(self.spot)
        derivative = make_derivative_groups(self.spot, self.um)
        delayed = make_delayed_perp_groups(self.spot, self.um)
        expected = [original["flow"], trailing_variances(self.spot, 24),
                    *derivative.values(), *delayed.values()]
        common = np.logical_and.reduce([np.isfinite(f.to_numpy()).all(axis=1) for f in expected])
        np.testing.assert_array_equal(self.built["full_feature_eligible"], common)
        for group, dimension in (("technical", 29), ("perp_delay0", 31)):
            pd.testing.assert_frame_equal(self.built["groups"][group], delayed[group])
            self.assertEqual(self.built["groups"][group].shape, (9000, dimension))
        self.assertFalse(common[:8641].any())
        self.assertTrue(common[8641:].all())
        self.assertEqual(self.built["source_contract"]["common_mask_sha256"],
                         mask_digest(self.spot.index, common))
        json.dumps(self.built["source_contract"], allow_nan=False)

    def test_causal_features_and_current_market_target_are_separate(self):
        changed = self.spot.copy()
        t = 8750
        changed.loc[changed.index[t]:, ["open", "high", "low", "close"]] *= 1.05
        changed_um = self.um.copy()
        changed_um.loc[changed_um.index[t]:, "taker_buy_quote"] *= .8
        result = build_market_inputs(changed, changed_um, cutoff=self.cutoff)
        for group in ("technical", "perp_delay0"):
            pd.testing.assert_frame_equal(result["groups"][group].iloc[:t + 1],
                                          self.built["groups"][group].iloc[:t + 1])
        self.assertNotEqual(result["returns"].iloc[t], self.built["returns"].iloc[t])
        self.assertEqual(self.built["returns"].iloc[t],
                         np.log(self.spot.close.iloc[t] / self.spot.close.iloc[t - 1]))
        np.testing.assert_array_equal(result["full_feature_eligible"][:t + 1],
                                      self.built["full_feature_eligible"][:t + 1])

    def test_um_gap_common_mask_does_not_change_physical_spot_truth(self):
        gap = 8750
        result = build_market_inputs(self.spot, self.um.drop(self.um.index[gap]), cutoff=self.cutoff)
        self.assertTrue(result["spot_observed"].all())
        self.assertFalse(result["availability"].um_bar_observed.iloc[gap])
        self.assertTrue(result["full_feature_eligible"][gap])  # same-row UM not known yet
        self.assertFalse(result["full_feature_eligible"][gap + 1:gap + 101].any())
        self.assertTrue(result["raw_target_validity"][gap:gap + 101].all())
        self.assertEqual(len(result["groups"]["perp_delay0"]), len(self.spot))
        self.assertTrue(result["availability"].spot_bar_observed.equals(
            self.built["availability"].spot_bar_observed))

    def test_spot_hole_breaks_two_returns_without_imputation(self):
        spot = self.spot.copy()
        gap = 8750
        spot.iloc[gap] = np.nan
        observed = np.ones(len(spot), dtype=bool)
        observed[gap] = False
        result = build_market_inputs(spot, self.um, cutoff=self.cutoff, spot_observed=observed)
        self.assertFalse(result["raw_target_validity"][gap:gap + 2].any())
        self.assertTrue(result["returns"].iloc[gap:gap + 2].isna().all())
        self.assertTrue(result["raw_target_validity"][gap + 2])
        self.assertTrue(result["full_feature_eligible"][gap])
        self.assertFalse(result["full_feature_eligible"][gap + 1])
        self.assertFalse(result["availability"].spot_bar_observed.iloc[gap])
        self.assertEqual(result["source_contract"]["physical_observation_source"], "explicit_sidecar")
        with self.assertRaisesRegex(ValueError, "unobserved Spot"):
            build_market_inputs(self.spot, self.um, cutoff=self.cutoff, spot_observed=observed)

    def test_trade_schema_cannot_silently_weaken_common_support(self):
        for spot, um in ((self.spot.drop(columns="n_trades"), self.um),
                         (self.spot, self.um.drop(columns="n_trades"))):
            with self.assertRaisesRegex(ValueError, "n_trades"):
                build_market_inputs(spot, um, cutoff=self.cutoff)
        contract = self.built["source_contract"]
        self.assertEqual(contract["required_trade_column"], "n_trades")
        self.assertEqual(len(contract["common_component_columns"]["original_flow"]), 24)
        self.assertIn("n_trades", contract["raw_schema"]["UM"])

    def test_raw_clock_cutoff_schema_and_input_nonmutation(self):
        before_s, before_u = self.spot.copy(deep=True), self.um.copy(deep=True)
        build_market_inputs(self.spot, self.um, cutoff=self.cutoff)
        pd.testing.assert_frame_equal(before_s, self.spot)
        pd.testing.assert_frame_equal(before_u, self.um)
        with self.assertRaisesRegex(ValueError, "post-cutoff"):
            build_market_inputs(self.spot, self.um, cutoff=self.spot.index[-1])
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            build_market_inputs(self.spot, self.um, cutoff="2026-01-01")
        with self.assertRaisesRegex(ValueError, "complete"):
            build_market_inputs(self.spot.drop(self.spot.index[100]), self.um, cutoff=self.cutoff)
        malformed = self.um.copy()
        malformed["decision_ts"] = malformed.index
        with self.assertRaisesRegex(ValueError, "timing contract"):
            build_market_inputs(self.spot, malformed, cutoff=self.cutoff)
        malformed = self.spot.copy()
        malformed["close"] = malformed.close.astype(complex)
        with self.assertRaisesRegex(ValueError, "complex"):
            build_market_inputs(malformed, self.um, cutoff=self.cutoff)


class InferenceInputTests(unittest.TestCase):
    def test_fixed64_at_origin_matches_training_row_and_normalizer(self):
        spot, um = bars(8800), bars(8800)
        origin = spot.index[8704]
        result = build_inference_inputs(spot.iloc[:8704], um.iloc[:8704], origin=origin)
        original = build_market_inputs(spot.iloc[:8705], um.iloc[:8705],
                                       cutoff=origin + pd.Timedelta(minutes=15))
        self.assertTrue(result["inference_available"])
        self.assertTrue(result["context_feature_eligible"].all())
        for group in ("technical", "perp_delay0"):
            pd.testing.assert_frame_equal(result["groups"][group], original["groups"][group])
            self.assertEqual(result["context_groups"][group].shape, (64, 29 if group == "technical" else 31))
            self.assertEqual(result["context_groups"][group].index[-1], origin)
        frame = result["groups"]["perp_delay0"]
        train = np.asarray(frame.index < origin - pd.Timedelta(minutes=15 * 10))
        normal = fit_normalizer(frame, train_mask=train, feature_eligible=result["full_feature_eligible"])
        scaled = apply_normalizer(frame, normal, feature_eligible=result["full_feature_eligible"])
        expected = apply_normalizer(original["groups"]["perp_delay0"], normal,
                                    feature_eligible=original["full_feature_eligible"])
        pd.testing.assert_frame_equal(scaled, expected)
        self.assertTrue(np.isfinite(scaled.iloc[-64:].to_numpy()).all())
        self.assertFalse(result["spot_observed"][-1])
        self.assertFalse(result["availability"].spot_bar_observed.iloc[-1])
        self.assertFalse(result["availability"].um_bar_observed.iloc[-1])
        self.assertNotIn("returns", result)
        self.assertNotIn("raw_target_validity", result)
        contract = result["source_contract"]
        self.assertEqual(contract["cutoff_exclusive"], origin.isoformat())
        self.assertEqual(contract["closed_raw_cutoff_exclusive"], origin.isoformat())
        self.assertEqual(contract["nominal_feature_grid_end_inclusive"], origin.isoformat())
        self.assertFalse(contract["placeholder_has_market_values"])
        self.assertFalse(contract["current_open_value_consumed"])
        self.assertFalse(contract["live_receipt_timeliness_established"])

    def test_current_future_poison_rejected_before_values_and_inputs_preserved(self):
        spot, um = bars(8800), bars(8800)
        origin = spot.index[8750]
        before_s, before_u = spot.copy(deep=True), um.copy(deep=True)
        baseline = build_inference_inputs(spot.iloc[:8750], um.iloc[:8750], origin=origin)
        for which in ("spot", "um"):
            s, u = spot.iloc[:8751].astype(object), um.iloc[:8751].astype(object)
            (s if which == "spot" else u).iloc[-1] = "future poison"
            if which == "spot":
                u = u.iloc[:-1]
            else:
                s = s.iloc[:-1]
            with self.assertRaisesRegex(ValueError, "current or future"):
                build_inference_inputs(s, u, origin=origin)
        future = spot.copy()
        future.loc[origin:, ["open", "high", "low", "close"]] *= 1.1
        changed = build_inference_inputs(future.loc[future.index < origin],
                                        um.loc[um.index < origin], origin=origin)
        for group in ("technical", "perp_delay0"):
            pd.testing.assert_frame_equal(changed["context_groups"][group], baseline["context_groups"][group])
        pd.testing.assert_frame_equal(spot, before_s)
        pd.testing.assert_frame_equal(um, before_u)

    def test_short_warmup_or_gap_fails_closed_without_time_compression(self):
        spot, um = bars(8704), bars(8704)
        short_origin = spot.index[-1]
        short = build_inference_inputs(spot.iloc[:-1], um.iloc[:-1], origin=short_origin)
        self.assertFalse(short["inference_available"])
        self.assertFalse(short["context_feature_eligible"][0])
        self.assertEqual(len(short["context_feature_eligible"]), 64)
        origin = spot.index[-1] + pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "end at origin"):
            build_inference_inputs(spot.iloc[:-1], um, origin=origin)
        gap = um.drop(um.index[-2])
        result = build_inference_inputs(spot, gap, origin=origin)
        self.assertFalse(result["inference_available"])
        self.assertFalse(result["context_feature_eligible"][-1])
        self.assertEqual(result["context_groups"]["perp_delay0"].index[-1], origin)
        self.assertEqual(len(result["groups"]["perp_delay0"]), len(spot) + 1)
        with self.assertRaisesRegex(ValueError, "complete"):
            build_inference_inputs(spot.drop(spot.index[-2]), um, origin=origin)
        with self.assertRaisesRegex(ValueError, "at least63"):
            build_inference_inputs(spot.iloc[:62], um.iloc[:62], origin=spot.index[62])


class NormalizerTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"varying": [1., 3., 5., 7., 100., 200.],
                             "constant": [2., 2., 2., 2., 3., 2.]}, index=index(6))

    def test_train_only_population_scale_floor_and_no_clipping(self):
        frame = self.frame()
        train = np.array([True] * 4 + [False] * 2)
        eligible = np.array([True, True, False, True, True, True])
        normal = fit_normalizer(frame, train_mask=train, feature_eligible=eligible)
        np.testing.assert_array_equal(normal["mean"], [11 / 3, 2.])
        self.assertAlmostEqual(normal["std"][0], np.std([1, 3, 7], ddof=0))
        self.assertEqual(normal["scale"][1], 1e-8)
        self.assertEqual(normal["n"], 3)
        output = apply_normalizer(frame, normal, feature_eligible=eligible)
        self.assertTrue(output.iloc[2].isna().all())
        self.assertEqual(output.constant.iloc[4], 1e8)  # deliberately not clipped
        self.assertGreater(output.varying.iloc[-1], 50.)
        self.assertTrue(output.index.equals(frame.index))
        json.dumps(normal, allow_nan=False)

    def test_future_and_ineligible_poison_do_not_fit_or_change_training_scale(self):
        frame = self.frame()
        original = frame.copy(deep=True)
        train = np.array([True, True, False, False, False, False])
        eligible = np.array([True, True, False, True, True, True])
        normal = fit_normalizer(frame, train_mask=train, feature_eligible=eligible)
        poison = frame.astype(object)
        poison.iloc[2:] = "unread future value"
        changed = fit_normalizer(poison, train_mask=train, feature_eligible=eligible)
        self.assertEqual(changed, normal)
        output = apply_normalizer(poison, normal, feature_eligible=train)
        self.assertTrue(output.iloc[2:].isna().all().all())
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(normal, copy.deepcopy(normal))

    def test_reordered_columns_changed_arithmetic_or_selected_bad_values_fail(self):
        frame = self.frame()
        mask = np.ones(6, dtype=bool)
        normal = fit_normalizer(frame, train_mask=mask, feature_eligible=mask)
        with self.assertRaisesRegex(ValueError, "order"):
            apply_normalizer(frame.iloc[:, ::-1], normal, feature_eligible=mask)
        for key, value in (("clip", 5), ("ddof", 1), ("scale_floor", 1e-6), ("scale", [1., 1.])):
            bad = copy.deepcopy(normal); bad[key] = value
            with self.assertRaises(ValueError):
                apply_normalizer(frame, bad, feature_eligible=mask)
        for value in (np.nan, np.inf, True, 1j, "1"):
            bad = frame.astype(object); bad.iloc[0, 0] = value
            with self.assertRaises(ValueError):
                fit_normalizer(bad, train_mask=mask, feature_eligible=mask)
        for bad_mask in (mask.astype(int), mask[:-1], np.ones((6, 1), dtype=bool)):
            with self.assertRaisesRegex(ValueError, "mask"):
                fit_normalizer(frame, train_mask=bad_mask, feature_eligible=mask)
        with self.assertRaisesRegex(ValueError, "no eligible"):
            fit_normalizer(frame, train_mask=~mask, feature_eligible=mask)


class MarketWindowTests(unittest.TestCase):
    def test_target_maturity_exact_boundary_and_no_current_label_in_inference(self):
        dates = index(160)
        returns_ok = np.ones(160, dtype=bool)
        cutoff = dates[100] + pd.Timedelta(minutes=15)
        result = target_horizon_mask(dates, returns_ok, cutoff=cutoff, horizon=64)
        np.testing.assert_array_equal(np.flatnonzero(result), np.arange(37))
        returns_ok[70] = False
        broken = target_horizon_mask(dates, returns_ok, cutoff=cutoff, horizon=64)
        self.assertTrue(broken[:6].all())
        self.assertFalse(broken[6:].any())
        features = np.ones(160, dtype=bool)
        inference = sequence_masks(dates, feature_eligible=features, seq_len=64)
        training = sequence_masks(dates, feature_eligible=features, target_eligible=broken, seq_len=64)
        self.assertEqual(len(inference["valid_starts"]), 97)
        self.assertEqual(len(training["valid_starts"]), 0)
        later = returns_ok.copy(); later[110:] = False
        np.testing.assert_array_equal(target_horizon_mask(dates, later, cutoff=cutoff), broken)

    def test_seq64_filters_windows_without_compressing_time_or_crossing_split(self):
        dates = index(160)
        features = np.ones(160, dtype=bool); features[70] = False
        row_mask = np.ones(160, dtype=bool); row_mask[:5] = False
        result = sequence_masks(dates, feature_eligible=features, row_mask=row_mask)
        np.testing.assert_array_equal(result["valid_starts"], np.r_[np.arange(5, 7), np.arange(71, 97)])
        np.testing.assert_array_equal(np.flatnonzero(result["endpoint_eligible"]),
                                      result["valid_starts"] + 63)
        self.assertFalse(result["row_eligible"][70])
        with self.assertRaisesRegex(ValueError, "complete"):
            sequence_masks(dates.delete(70), feature_eligible=np.ones(159, dtype=bool))

    def test_dataset_uses_original_rows_and_never_exposes_gap_nan(self):
        dates = index(160)
        features = pd.DataFrame({"clock": np.arange(160.), "other": 2.}, index=dates)
        eligible = np.ones(160, dtype=bool); eligible[70] = False
        features.iloc[70] = np.nan
        physical = np.ones(160, dtype=bool)  # raw was observed; derived feature absent
        returns = pd.Series(np.arange(160.) / 10000., index=dates)
        dataset = MarketSequenceDataset(features, feature_eligible=eligible,
            target_eligible=np.ones(160, dtype=bool), returns=returns, spot_observed=physical)
        self.assertEqual(len(dataset.features), 160)
        self.assertTrue(np.isnan(dataset.features[70].numpy()).all())
        self.assertEqual(dataset.valid_starts[7], 71)
        sample = dataset[7]
        self.assertEqual(sample["obs"][0, 0].item(), 71.)
        self.assertAlmostEqual(sample["returns"][0].item(), .0071, places=8)
        for item in dataset:
            self.assertTrue(np.isfinite(item["obs"].numpy()).all())
            self.assertTrue(np.isfinite(item["returns"].numpy()).all())
        self.assertTrue(dataset.availability.spot_bar_observed.all())
        self.assertTrue(dataset.source_timestamps.equals(dates))
        self.assertFalse(dataset.row_eligible[70])

    def test_dataset_fails_claimed_valid_nonfinite_and_ignores_excluded_poison(self):
        dates = index(8)
        frame = pd.DataFrame({"x": range(8)}, index=dates).astype(object)
        feature_mask = np.ones(8, dtype=bool)
        frame.iloc[-1, 0] = "poison"
        with self.assertRaises(ValueError):
            MarketSequenceDataset(frame, feature_eligible=feature_mask, seq_len=3)
        row_mask = feature_mask.copy(); row_mask[-1] = False
        returns = np.array([.01] * 7 + ["poison"], dtype=object)
        dataset = MarketSequenceDataset(frame, feature_eligible=feature_mask,
            row_mask=row_mask, returns=returns, seq_len=3)
        np.testing.assert_array_equal(dataset.valid_starts, np.arange(5))
        self.assertTrue(np.isnan(dataset.returns[-1].item()))
        with self.assertRaises(ValueError):
            MarketSequenceDataset(frame, feature_eligible=row_mask, returns=returns[:-1], seq_len=3)
        frame.iloc[0, 0] = 1e40
        with self.assertRaisesRegex(ValueError, "float32"):
            MarketSequenceDataset(frame, feature_eligible=row_mask, seq_len=3)

    def test_invalid_masks_horizons_and_clock_are_rejected(self):
        dates = index(8)
        mask = np.ones(8, dtype=bool)
        for horizon in (-1, True, 1., "1"):
            with self.assertRaises(ValueError):
                target_horizon_mask(dates, mask, cutoff=dates[-1], horizon=horizon)
        for length in (0, True, 1., "1"):
            with self.assertRaises(ValueError):
                sequence_masks(dates, feature_eligible=mask, seq_len=length)
        for malformed in (mask.astype(int), mask[:-1]):
            with self.assertRaises(ValueError):
                target_horizon_mask(dates, malformed, cutoff=dates[-1])
        with self.assertRaises(ValueError):
            sequence_masks(dates.tz_localize(None), feature_eligible=mask)
        shifted = dates + pd.Timedelta(minutes=15)
        self.assertNotEqual(mask_digest(dates, mask), mask_digest(shifted, mask))
        self.assertNotEqual(mask_digest(dates, mask), mask_digest(dates, ~mask))


if __name__ == "__main__":
    unittest.main()
