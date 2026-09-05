import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from unidream.experiments import oracle_confirmation_contract as contract
from unidream.experiments.oracle_derivative_delay import segment_masks as inherited_masks


def cohort():
    economic, predictions = [], []
    for fold in contract.FOLDS:
        regime = contract.STRATA[1 + (fold - 26) % 3]
        for cid in contract.CANDIDATES + contract.CONTROLS:
            economic.append({"fold": fold, "candidate_id": cid, "regime": regime,
                             **{cost: {"alpha_ex": .01, "maxdd_delta": -.02} for cost in contract.COSTS}})
        for mean in ("scale_mean", "technical_scaled", "perp_delay0_scaled") + contract.MEANS:
            predictions.append({"fold": fold, "mean_id": mean, "regime": regime,
                                "mse": .001 if mean in contract.MEANS else .002,
                                "scored_rows": 100 if fold % 2 else 200})
    return economic, predictions


class OracleConfirmationContractTests(unittest.TestCase):
    def test_future_calendar_anchors_test_start_and_retains_18_3_3_chronology(self):
        first, last = contract.calendar(26), contract.calendar(37)
        self.assertEqual(first["evaluation_start"], pd.Timestamp("2026-10-16T13:45Z"))
        self.assertEqual(last["evaluation_end"], pd.Timestamp("2029-10-16T13:45Z"))
        self.assertEqual(first["fit_start"], pd.Timestamp("2024-10-16T13:45Z"))
        self.assertEqual(first["fit_end"], pd.Timestamp("2026-04-16T13:45Z"))
        self.assertEqual(first["scale_end"], pd.Timestamp("2026-07-16T13:45Z"))
        for fold in contract.FOLDS[:-1]:
            self.assertEqual(contract.calendar(fold)["evaluation_end"], contract.calendar(fold + 1)["evaluation_start"])
        for bad in (-1, True, 26.5, "26"):
            with self.assertRaises(ValueError): contract.calendar(bad)

    def test_exact_parity_with_inherited_masks_at_new_evaluation_start(self):
        dates = contract.calendar(26)
        index = pd.date_range(dates["fit_start"], dates["evaluation_end"], freq="15min", inclusive="left")
        features = np.arange(len(index)) % 7 != 0
        labels = np.arange(len(index)) % 11 != 0
        masks = contract.segment_masks(index, features, labels, 26)
        old, _ = inherited_masks(index, features, labels, {"horizon_bars": 24},
                                  dates["evaluation_start"], dates["evaluation_end"])
        for name in old:
            np.testing.assert_array_equal(masks[name], old[name])
        np.testing.assert_array_equal(features, np.arange(len(index)) % 7 != 0)
        np.testing.assert_array_equal(labels, np.arange(len(index)) % 11 != 0)

    def test_future_labels_cannot_remove_inference_and_tail_scores_stay_purged(self):
        dates = contract.calendar(26)
        index = pd.date_range(dates["evaluation_start"], dates["evaluation_end"], freq="15min", inclusive="left")
        good = contract.segment_masks(index, np.ones(len(index), bool), np.ones(len(index), bool), 26)
        missing = contract.segment_masks(index, np.ones(len(index), bool), np.zeros(len(index), bool), 26)
        np.testing.assert_array_equal(good["inference"], missing["inference"])
        self.assertTrue(missing["inference"].any())
        self.assertFalse(missing["score"].any())
        self.assertTrue(np.any(good["inference"] & ~good["score"]))
        self.assertTrue(((index + contract.MATURITY)[good["score"]] <= dates["evaluation_end"]).all())

    def test_strict_calibration_and_inclusive_score_maturity_boundary(self):
        # Align a synthetic segment endpoint to a scheduled label maturity to
        # distinguish < from <=; production ANCHOR is never changed by config.
        with patch.object(contract, "ANCHOR", pd.Timestamp("2020-04-16T06:15Z")):
            dates = contract.calendar(26)
            for segment in ("fit", "scale", "interval", "evaluation"):
                end = dates[segment + "_end"]
                index = pd.date_range(end - pd.Timedelta(days=1), end, freq="15min", inclusive="left")
                masks = contract.segment_masks(index, np.ones(len(index), bool), np.ones(len(index), bool), 26)
                pos = index.get_loc(end - contract.MATURITY)
                self.assertEqual(bool(masks["score" if segment == "evaluation" else segment][pos]), segment == "evaluation")

    def test_calendar_and_support_metadata_errors_fail_instead_of_reindexing(self):
        index = pd.date_range("2026-10-16T13:45Z", periods=10, freq="15min")
        for bad in (index.tz_localize(None), index.delete(2), index[::-1], index.insert(2, index[1]),
                    index + pd.Timedelta(seconds=1), pd.DatetimeIndex([], tz="UTC")):
            with self.subTest(index=bad), self.assertRaises(ValueError):
                contract.segment_masks(bad, np.ones(len(bad), bool), np.ones(len(bad), bool), 26)
        for bad in (np.ones(10), np.ones((10, 1), bool), np.ones(9, bool)):
            with self.assertRaises(ValueError): contract.segment_masks(index, bad, np.ones(10, bool), 26)

    def test_eighty_unique_endpoints_and_ninety_six_candidate_references(self):
        endpoints, mapping = contract.endpoint_inventory()
        self.assertEqual(len(endpoints), 80)
        self.assertEqual(len({e["id"] for e in endpoints}), 80)
        self.assertEqual(sum(e["kind"] == "economic" for e in endpoints), 64)
        self.assertEqual(sum(e["kind"] == "predictive" for e in endpoints), 16)
        self.assertEqual(set(mapping), set(contract.CANDIDATES))
        self.assertTrue(all(len(ids) == 24 for ids in mapping.values()))
        for mean in contract.MEANS:
            left, right = [set(mapping[mean + "_" + rule]) for rule in contract.RULES]
            self.assertEqual(len(left & right), 8)

    def test_all_positive_synthetic_outcomes_never_unlock_inference_or_provenance(self):
        economic, predictions = cohort(); before = copy.deepcopy((economic, predictions))
        result = contract.describe_complete_family(economic, predictions)
        self.assertEqual(result["regime_counts"], {"bull": 4, "bear": 4, "sideways": 4})
        self.assertTrue(result["complete_quarter_inventory"])
        self.assertFalse(result["complete_bar_calendar_verified"])
        self.assertFalse(result["protocol_provenance_integrity_verified"])
        self.assertFalse(result["high_probability_generalization_established"])
        self.assertFalse(result["selection_performed"])
        self.assertEqual(len(result["endpoints"]), 80)
        for row in result["candidates"].values():
            self.assertTrue(row["observed_metric_and_coverage_conditions_met"])
            self.assertIsNone(row["candidate_primary_p"])
            self.assertIsNone(row["candidate_holm_p"])
        self.assertEqual((economic, predictions), before)

    def test_loss_means_are_equal_quarter_and_one_failed_component_blocks_candidate(self):
        economic, predictions = cohort()
        for row in predictions:
            if row["mean_id"] == "perp_delay0_half":
                row["mse"] = .002 if row["fold"] % 2 else .004
        result = contract.describe_complete_family(economic, predictions)
        eid = "predictive/perp_delay0_half/scale_mean/all/mse_reduction"
        self.assertAlmostEqual(result["endpoints"][eid]["favorable_mean"], -.001)
        for cid in contract.CANDIDATES:
            self.assertEqual(result["candidates"][cid]["observed_metric_and_coverage_conditions_met"], cid.startswith("technical"))
        economic, predictions = cohort()
        for row in economic:
            if row["candidate_id"] == contract.CANDIDATES[0] and row["regime"] == "bull":
                row["stress_2x"]["maxdd_delta"] = 0
        result = contract.describe_complete_family(economic, predictions)
        self.assertFalse(result["candidates"][contract.CANDIDATES[0]]["observed_economic_signs"])
        self.assertTrue(result["candidates"][contract.CANDIDATES[1]]["observed_economic_signs"])

    def test_missing_duplicate_unpaired_nonfinite_and_false_denominators_fail(self):
        for kind in ("missing", "duplicate", "regime", "metric", "fold", "score_count", "score_nan"):
            economic, predictions = cohort()
            if kind == "missing": economic.pop()
            elif kind == "duplicate": economic.append(copy.deepcopy(economic[0]))
            elif kind == "regime": economic[0]["regime"] = "bear"
            elif kind == "metric": economic[0]["base"]["alpha_ex"] = float("nan")
            elif kind == "fold": economic[0]["fold"] = 26.0
            elif kind == "score_count": predictions[0]["scored_rows"] += 1
            elif kind == "score_nan": predictions[0]["mse"] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                contract.describe_complete_family(economic, predictions)

    def test_nonreal_boolean_and_array_metrics_cannot_be_silently_cast(self):
        for value in (True, np.bool_(True), .01 + 100j, np.complex128(.01), np.array([.01]), "0.01"):
            for kind in ("economic", "predictive"):
                economic, predictions = cohort()
                if kind == "economic": economic[0]["base"]["alpha_ex"] = value
                else: predictions[0]["mse"] = value
                with self.subTest(value=value, kind=kind), self.assertRaises(ValueError):
                    contract.describe_complete_family(economic, predictions)

    def test_finite_large_metrics_do_not_overflow_quarter_mean(self):
        economic, predictions = cohort()
        for row in economic:
            for cost in contract.COSTS: row[cost]["alpha_ex"] = 1e308
        result = contract.describe_complete_family(economic, predictions)
        key = "economic/" + contract.CANDIDATES[0] + "/all/base/alpha_ex"
        self.assertTrue(np.isfinite(result["endpoints"][key]["favorable_mean"]))
        self.assertAlmostEqual(result["endpoints"][key]["favorable_mean"] / 1e308, 1.)

    def test_family_binding_cannot_be_overridden_through_duplicate_or_alias(self):
        cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/oracle_confirmation_contract_20260906.yaml").read_text())
        for path in (cfg["family_path"], str(Path(cfg["family_path"]).resolve()),
                     str(Path(cfg["family_path"]).parent / ".." / Path(cfg["family_path"]).parent.name / Path(cfg["family_path"]).name)):
            changed = copy.deepcopy(cfg); changed["bindings"][path] = "0" * 64
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "duplicate or aliased"):
                contract.validate_config(changed)
        changed = copy.deepcopy(cfg); changed["bindings"].pop(next(iter(changed["bindings"])))
        with self.assertRaisesRegex(ValueError, "dependency set"):
            contract.validate_config(changed)

    def test_absent_regime_is_retained_with_null_endpoint_and_failed_coverage(self):
        economic, predictions = cohort()
        for row in economic + predictions: row["regime"] = "bull"
        result = contract.describe_complete_family(economic, predictions)
        self.assertFalse(result["regime_coverage"])
        self.assertTrue(all(not r["observed_metric_and_coverage_conditions_met"] for r in result["candidates"].values()))
        self.assertTrue(all(r["favorable_mean"] is None for eid, r in result["endpoints"].items() if "/bear/" in eid))

    def test_config_forbids_selection_shorter_cohort_weaker_coverage_or_enabled_inference(self):
        cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/oracle_confirmation_contract_20260906.yaml").read_text())
        contract.validate_config(cfg)
        for key, value in {"evaluation_folds": list(range(26, 29)), "evaluation_split": "validation",
                           "mean_weight": .25, "minimum_quarters_per_regime": 2,
                           "selection_permitted": True, "marginal_engine_id": "bootstrap",
                           "decision_deadline_seconds": 900, "extra_unknown_override": 1}.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.validate_config({**cfg, key: value})


if __name__ == "__main__":
    unittest.main()
