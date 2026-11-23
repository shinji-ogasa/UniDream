"""Synthetic final-table report checks; no historical market-result reads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from unidream.experiments import wm_inference_robust_report as report
from unidream.experiments.wm_inference_robust_hac import power_scenarios


class RobustReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.final = self.root / "final"
        self.final.mkdir()
        self.cfg = {"references": [{"title": '<script>bad()</script>', "url": "https://example.com/paper?a=1&b=2"}]}
        self.registration = self.root / "posthoc_registration.json"
        self.registration.write_text(json.dumps({"config": self.cfg}))
        rows = []
        for benchmark in ("bh", "fixed50"):
            for condition in report.LABELS:
                rows.append({"benchmark": benchmark, "condition": condition, "primary_inference": True,
                    "method": "circular", "block_days": 7, "estimand": "market_fixed_seeds",
                    "estimate": .03, "ci_low": -.1, "ci_high": .15, "p_value": .1,
                    "p_holm_family": .8, "inference_available": True})
        self.primary = pd.DataFrame(rows)
        self.primary.loc[(self.primary.benchmark == "bh") & (self.primary.condition == "bc_A"), "p_holm_family"] = .02
        sensitivity = self.primary.copy()
        sensitivity["primary_inference"], sensitivity["method"], sensitivity["p_holm_family"] = False, "stationary", .00001
        self.write("bootstrap_inference", pd.concat([self.primary, sensitivity]))
        hac = self.primary.copy()
        hac["lag_days"], hac["standard_error"] = 6, .03
        self.write("hac_inference", hac)
        summary = self.primary.copy()
        summary["originallegacy_p_value"] = .002
        summary["p_holm_16_primary_controls"] = .3
        summary["cumulative_return_difference_descriptive"] = .04
        self.write("summary", summary)
        rc = self.primary.copy()
        rc["global_reality_check_p"], rc["max_adjusted_one_sided_p"] = .2, .3
        self.write("reality_check", rc)
        planning = power_scenarios({"inference_available": True, "market_variance": .002,
            "eligible_years": 1.4, "estimate": .03, "approximate_seed_variance_floor": None},
            years=(1, 2), alphas=(.05/8,), powers=(.8,), include_observed=False)
        for filename, key in (("power", "power"), ("mde", "mde"), ("required_years", "required_years")):
            data = [dict(row, benchmark=benchmark, condition=condition) for benchmark in ("bh", "fixed50")
                for condition in report.LABELS for row in planning[key]]
            self.write(filename, pd.DataFrame(data))
        self.write("calibration", pd.DataFrame([{"layout": "single", "scenario": "iid", "method": "hac_lag6",
            "outer_replicates": 300, "empirical_type1_error": .05, "type1_mc_ci_low": .03,
            "type1_mc_ci_high": .08, "empirical_coverage_95": .95}]))
        self.write("acf", pd.DataFrame([{"series": "diff", "lag_days": 1, "acf": .2}]))
        self.write("fold_diagnostics", pd.DataFrame([{"benchmark": "bh", "condition": "bc_A", "fold": 0,
            "n_market_bars": 96, "mean_daily_per_bar_log_excess": .00001, "daily_per_bar_std": .001}]))
        self.write("legacy_replay", pd.DataFrame([{"condition": "bc_A", "p_value": .002}]))
        (self.final / "audit.json").write_text(json.dumps({"panel": {"n_unique_market_rows": 100,
            "n_daily_units": 12, "n_segments": 2, "n_training_seeds": 3, "eligible_years": .002854},
            "legacy_p_replay": {"max_absolute_difference": 0.}}))

    def write(self, name, frame):
        frame.to_csv(self.final / (name+".csv"), index=False)

    def test_registration_guard_precedes_all_result_table_access(self):
        self.registration.unlink()
        with patch.object(report.pd, "read_csv") as read:
            with self.assertRaisesRegex(ValueError, "registration required"):
                report.write_report(self.root, self.cfg)
            read.assert_not_called()

    def test_config_must_match_registered_analysis(self):
        with self.assertRaisesRegex(ValueError, "config differs"):
            report.write_report(self.root, {"references": []})

    def test_exports_scientific_figures_and_reports_primary_only_without_completion(self):
        result = report.write_report(self.root, self.cfg)
        self.assertEqual(result["primary_bh_significant_improvements"], 1)
        self.assertEqual(result["primary_bh_conditions"], 8)
        self.assertFalse(result["completion_written"])
        for filename in ("index.html", "primary_forest.png", "primary_forest.svg", "power_curves.png", "power_curves.svg"):
            self.assertTrue((self.final / filename).is_file())
            self.assertGreater((self.final / filename).stat().st_size, 500)
        text = (self.final / "index.html").read_text()
        self.assertIn("中心化がありました", text)
        self.assertIn("点ごとの区間", text)
        self.assertIn("420", text)
        self.assertIn("観測後の検出力", text)
        self.assertIn("&lt;script&gt;bad()&lt;/script&gt;", text)
        self.assertNotIn("<script>bad()", text)
        self.assertIn("100本", text)
        self.assertNotIn("50038本", text)
        self.assertFalse((self.root / "completion.json").exists())
        self.assertEqual(result["report_sha256"], report._sha(self.final / "index.html"))

    def test_incomplete_primary_panel_fails_instead_of_claiming_zero_improvements(self):
        self.write("bootstrap_inference", self.primary.iloc[:-1])
        with patch.object(report, "_figures") as draw:
            with self.assertRaisesRegex(ValueError, "complete eight-condition"):
                report.write_report(self.root, self.cfg)
            draw.assert_not_called()

    def test_missing_hac_condition_fails_instead_of_silent_partial_comparison(self):
        hac = self.primary.iloc[:-1].copy()
        hac["lag_days"] = 6
        self.write("hac_inference", hac)
        with patch.object(report, "_figures") as draw:
            with self.assertRaisesRegex(ValueError, "complete unique lag6"):
                report.write_report(self.root, self.cfg)
            draw.assert_not_called()

    def test_changed_input_during_rendering_prevents_html_publish(self):
        def mutate(*args):
            (self.final / "audit.json").write_text('{"changed": true}')
            return []
        with patch.object(report, "_figures", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "result table changed"):
                report.write_report(self.root, self.cfg)
        self.assertFalse((self.final / "index.html").exists())
        self.assertFalse((self.root / "completion.json").exists())

    def test_changed_registration_during_rendering_prevents_html_publish(self):
        def mutate(*args):
            self.registration.write_text('{"changed": true}')
            return []
        with patch.object(report, "_figures", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "registration changed"):
                report.write_report(self.root, self.cfg)
        self.assertFalse((self.final / "index.html").exists())

    def test_string_false_is_not_a_primary_selection_flag(self):
        flags = report._boolean(pd.Series(["False", "True", "0", "1", None]))
        self.assertEqual(flags.tolist(), [False, True, False, True, False])

    def test_optional_twelve_contrasts_and_new_twenty_family_are_reported(self):
        self.write("ablation_primary", pd.DataFrame([{"condition": "bc_B_minus_A", "estimate": .02,
            "ci_low": -.02, "ci_high": .06, "p_value": .2, "p_holm_family": 1.}]))
        self.write("expanded_family20", pd.DataFrame([{"family": "new_ablation", "condition": "bc_B_minus_A",
            "p_value": .2, "p_holm_20": 1.}]))
        with patch.object(report, "_figures", return_value=[]):
            result = report.write_report(self.root, self.cfg)
        text = (self.final / "index.html").read_text()
        self.assertIn("元の12比較も同じ新手法", text)
        self.assertIn("旧非循環法の12個のp値をそのまま混ぜる補正ではありません", text)
        self.assertIn("ablation_primary.csv", text)
        self.assertTrue(any(name.endswith("expanded_family20.csv") for name in result["input_sha256"]))


if __name__ == "__main__":
    unittest.main()
