"""Synthetic causal-boundary and semantic tests; never inspect real Test rows."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unidream.experiments import wm_policy_ablation_data as data
from unidream.experiments.wm_seven_task_evaluation import TASKS


class PolicyDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {"output_dir": str(self.root), "splits": {"start": "2023-01-01", "end": "2026-10-01"},
            "policy": {"context": 4, "horizon": 2, "purge_bars": 4, "train_years": 2, "val_months": 3, "test_months": 3}}

    def row(self, *, group="A", model="wm", task="variance", form="joint", scalar_p=.001,
            class_p=.001, scalar_imp=.1, class_imp=.1):
        return {"group": group, "model": model, "task": task, "formulation": form,
            "scalar_primary_p_holm_global1320": scalar_p, "scalar_primary_improvement": scalar_imp,
            "class_BA_p_holm_global1320": class_p, "class_BA_improvement": class_imp}

    def test_native_only_excludes_scalar_proxy_ba(self):
        specs = data.native_head_specs(pd.DataFrame([self.row(form="regression")]))
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["columns"], ["scalar_variance"])
        self.assertEqual(specs[0]["head"], "regression")

    def test_classical_joint_aliases_deduplicate_both_heads(self):
        rows = [self.row(model="lightgbm", form=form) for form in ("regression", "classification", "joint")]
        specs = data.native_head_specs(pd.DataFrame(rows))
        self.assertEqual(len(specs), 2)
        self.assertEqual(sum(len(s["columns"]) for s in specs), 4)
        self.assertEqual({s["source_formulation"] for s in specs}, {"regression", "classification"})
        self.assertTrue(all("joint" in s["aggregate_aliases"] for s in specs))

    def test_neural_joint_heads_are_distinct_from_standalone(self):
        specs = data.native_head_specs(pd.DataFrame([self.row(form=form) for form in ("regression", "classification", "joint")]))
        self.assertEqual(len(specs), 4)
        self.assertEqual(sum(len(s["columns"]) for s in specs), 8)

    def test_crash_probability_has_only_positive_column(self):
        specs = data.native_head_specs(pd.DataFrame([self.row(task="crash", form="classification")]))
        self.assertEqual(specs[0]["columns"], ["p_crash_1"])

    def test_all_positive_heads_retained_without_taskwise_winner_selection(self):
        rows = [self.row(model=model, scalar_p=p) for model, p in (("wm", .001), ("direct", .049))]
        specs = data.native_head_specs(pd.DataFrame(rows))
        self.assertEqual(len(specs), 4)
        self.assertEqual({s["model"] for s in specs}, {"wm", "direct"})

    def test_strict_significance_and_positive_improvement(self):
        rows = [self.row(scalar_p=.05, class_imp=0), self.row(group="B", scalar_p=np.nan, class_imp=-.1),
            self.row(group="C", scalar_imp=-1, class_p=np.inf), self.row(group="D", form="regression")]
        specs = data.native_head_specs(pd.DataFrame(rows))
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["group"], "D")

    def test_feature_whitelist_rejects_actual_and_labels(self):
        spec = data.native_head_specs(pd.DataFrame([self.row(form="regression")]))[0]
        frame = pd.DataFrame({"scalar_variance": [.1, .2], "actual_variance": [1., 2.], "label_variance": [0, 2]})
        np.testing.assert_array_equal(data._extract_feature(frame, spec), [[.1], [.2]])
        for column in ("actual_variance", "label_variance"):
            altered = dict(spec, columns=[column])
            with self.assertRaisesRegex(ValueError, "whitelisted"):
                data._extract_feature(frame, altered)

    def test_invalid_native_probability_fails_instead_of_fill(self):
        spec = data.native_head_specs(pd.DataFrame([self.row(form="classification")]))[0]
        frame = pd.DataFrame({"p_variance_0": [.2], "p_variance_1": [.2], "p_variance_2": [.2]})
        with self.assertRaisesRegex(ValueError, "sum to one"):
            data._extract_feature(frame, spec)
        frame.iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            data._extract_feature(frame, spec)

    def test_six_walk_forward_dates_and_no_overlap(self):
        folds = data.policy_folds(self.cfg)
        self.assertEqual(len(folds), 6)
        self.assertEqual(folds[0]["test_start"], "2025-04-01T00:00:00+00:00")
        self.assertEqual(folds[-1]["test_end"], "2026-10-01T00:00:00+00:00")
        for left, right in zip(folds[:-1], folds[1:]):
            self.assertEqual(left["test_end"], right["test_start"])
        for fold in folds:
            self.assertEqual(fold["train_end"], fold["val_start"])
            self.assertEqual(fold["val_end"], fold["test_start"])

    def test_strict_purge_excludes_label_boundary_equality(self):
        times = pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC").asi8
        ids = data.purged_split_ids(times, np.ones(10, bool), pd.Timestamp(times[0], tz="UTC"),
            pd.Timestamp(times[8], tz="UTC"), purge_bars=2)
        np.testing.assert_array_equal(ids, np.arange(6))
        self.assertTrue(np.all(times[ids] + 2 * data.BAR_NS < times[8]))

    def test_context_valid_does_not_inspect_future(self):
        x = np.ones((12, 3))
        x[5] = np.nan
        valid = data.causal_context_valid(x, np.arange(12), context=3)
        np.testing.assert_array_equal(np.flatnonzero(valid), [2, 3, 4, 8, 9, 10, 11])
        x[11] = np.nan
        revised = data.causal_context_valid(x, np.arange(12), context=3)
        np.testing.assert_array_equal(revised[:11], valid[:11])

    def test_window_ends_at_current_token_and_uses_train_norm(self):
        x = np.arange(24, dtype=np.float32).reshape(12, 2)
        arrays = {"x_grid": x, "decision_id": np.array([3, 7]), "context_valid": np.ones(2, bool)}
        fold = {"x_mean": np.zeros(2), "x_std": np.ones(2)}
        windows = data.x_windows(arrays, [1], fold, context=4, clip=100)
        np.testing.assert_array_equal(windows[0], x[4:8])
        x[8:] = 999
        np.testing.assert_array_equal(data.x_windows(arrays, [1], fold, context=4, clip=100), windows)

    def test_missing_context_is_not_silently_filled(self):
        arrays = {"x_grid": np.ones((12, 2)), "decision_id": np.array([3]), "context_valid": np.array([False])}
        with self.assertRaisesRegex(ValueError, "unavailable"):
            data.x_windows(arrays, [0], {"x_mean": np.zeros(2), "x_std": np.ones(2)}, context=4)

    def test_future_path_is_cumulative_delayed_one_bar(self):
        close = 100 * np.exp(np.array([0., .01, .03, .06, .10, .15]))
        current, path, valid = data.future_labels(close, [2, 3], horizon=2)
        np.testing.assert_allclose(current, [.02, .03])
        np.testing.assert_allclose(path, [[.03, .07], [.04, .09]])
        np.testing.assert_array_equal(valid, [True, True])
        # First teacher increment is bar t+1, not the current reward bar t.
        self.assertNotAlmostEqual(path[0, 0], current[0])

    def test_future_unavailable_is_label_mask_only(self):
        close = np.arange(1., 8.)
        current, path, valid = data.future_labels(close, [4, 6], horizon=2)
        self.assertTrue(np.isfinite(current).all())
        np.testing.assert_array_equal(valid, [True, False])
        self.assertTrue(np.isnan(path[1]).all())

    def test_segment_gap_and_split_reset(self):
        times = np.array([0, 1, 2, 5, 6, 10]) * data.BAR_NS
        np.testing.assert_array_equal(data.contiguous_segments(times), [0, 0, 0, 1, 1, 2])
        np.testing.assert_array_equal(data.contiguous_segments(times[1:3]), [0, 0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            data.contiguous_segments([1, 1])

    def test_normalizers_are_unchanged_by_future_values(self):
        x = np.arange(20.).reshape(10, 2)
        first = data.fit_normalizer(x, [0, 1, 2])
        x[3:] = 1e9
        second = data.fit_normalizer(x, [0, 1, 2])
        np.testing.assert_array_equal(first, second)
        _, std = data.fit_normalizer(np.ones((4, 2)), [0, 1])
        np.testing.assert_array_equal(std, [1., 1.])

    def test_registration_guard_precedes_all_individual_data_access(self):
        with patch.object(data, "inspect_sources") as inspect, patch.object(data.np, "load") as load:
            with self.assertRaisesRegex(ValueError, "before individual"):
                data.build_cache(self.cfg, {})
            inspect.assert_not_called()
            load.assert_not_called()

    def test_inspection_binds_prepared_paths_without_reading_arrays(self):
        configs = {}
        for name in ("significance", "seven", "full", "joint", "lstm", "parent"):
            root = self.root / name
            root.mkdir()
            path = self.root / (name + ".yaml")
            path.write_text("metadata only")
            configs[name] = {"config_path": str(path), "output_dir": str(root)}
            data._json(root / "completion.json", {"complete": True})
        configs["seven"]["parent_config"] = configs["parent"]["config_path"]
        folds = []
        for i in range(15):
            start = pd.Timestamp("2023-01-01", tz="UTC") + pd.DateOffset(months=3*i)
            folds.append({"fold": i, "train_start": (start-pd.DateOffset(years=2, months=3)).isoformat(),
                "train_end": (start-pd.DateOffset(months=3)).isoformat(),
                "val_start": (start-pd.DateOffset(months=3)).isoformat(), "val_end": start.isoformat(),
                "test_start": start.isoformat(), "test_end": (start+pd.DateOffset(months=3)).isoformat(),
                "label_contract": {}, "scales": {}})
        for name in configs:
            data._json(Path(configs[name]["output_dir"]) / "preregistration.json",
                {"folds": folds, "prepared_sha256": {"target.npy": "seven-hash"} if name == "seven" else
                    {"prepared/arrays.npz": "parent-hash"} if name == "parent" else {}})
        final = Path(configs["significance"]["output_dir"]) / "final"
        final.mkdir()
        (final / "results_420_significance.csv").write_text("aggregate only")
        data._json(Path(configs["significance"]["output_dir"]) / "prepared/cache_manifest.json",
            {"baseline_path": "not-opened.npz", "baseline_sha256": "frozen-baseline"})
        cfg = copy.deepcopy(self.cfg)
        for name in ("significance", "seven", "full", "joint", "lstm"):
            cfg[name + "_config"] = configs[name]["config_path"]
        rows = [self.row(group=g, model=m, task=t, form=f) for g in "ABCD"
            for m in ("linear", "lightgbm", "direct", "wm", "lstm") for t in TASKS
            for f in ("regression", "classification", "joint")]
        with patch.object(data, "_load", side_effect=lambda p: configs[Path(p).stem]), \
                patch.object(data.pd, "read_csv", return_value=pd.DataFrame(rows)), \
                patch.object(data.np, "load") as load:
            inspected = data.inspect_sources(cfg)
        load.assert_not_called()
        expected = inspected["source_prepared_sha256"]
        self.assertEqual(expected[str(Path(configs["seven"]["output_dir"]) / "prepared/target.npy")], "seven-hash")
        self.assertEqual(expected[str(Path(configs["parent"]["output_dir"]) / "prepared/arrays.npz")], "parent-hash")

    def test_registration_must_match_disk(self):
        (self.root / "preregistration.json").write_text('{"study": 1}')
        with self.assertRaisesRegex(ValueError, "differs"):
            data.require_registration(self.cfg, {"study": 2})

    def test_cached_sources_and_arrays_are_hash_checked(self):
        source = self.root / "source.txt"
        source.write_text("frozen")
        array = self.root / "array.npy"
        data._npy(array, np.arange(5))
        manifest = {"complete": True, "registration_sha256": "abc", "source_artifact_sha256": {str(source): data._sha(source)},
            "arrays": {"x": {"path": str(array), "sha256": data._sha(array)}}, "fold_arrays": {}}
        path = self.root / "cache_manifest.json"
        data._json(path, manifest)
        self.assertTrue(data._prior_cache(path, "abc")["complete"])
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            data._prior_cache(path, "abc")

    def test_fold_builder_excludes_context_gaps_and_fits_train_only(self):
        times = pd.date_range("2023-01-01", "2025-07-01", freq="15min", inclusive="left", tz="UTC").asi8
        n = len(times)
        decisions = np.arange(n)
        x = np.column_stack((np.arange(n), np.ones(n))).astype(np.float32)
        target = np.ones((n, len(TASKS))) * .01
        target[:, 3] = 0
        arrays = {"index_ns": times, "context_valid": np.ones(n, bool), "f": np.ones((n, 4)),
            "bar_log_return": np.zeros(n), "teacher_valid": np.ones(n, bool), "decision_id": decisions,
            "x_grid": x, "wm_targets_grid": target, "wm_mask_grid": np.ones(n, bool),
            "wm_label_end_ns": times + 2 * data.BAR_NS}
        arrays["context_valid"][20] = False
        fold, audit = data._make_fold(arrays, data.policy_folds(self.cfg)[0], self.cfg)
        self.assertNotIn(20, fold["train"])
        self.assertEqual(audit["support"]["train"]["n_segments"], 2)
        self.assertTrue(set(fold["train"]).isdisjoint(fold["test"]))
        self.assertTrue(set(fold["earlystop"]).isdisjoint(fold["selection"]))
        means = fold["x_mean"].copy()
        arrays["x_grid"][fold["test"]] = 1e20
        revised, _ = data._make_fold(arrays, data.policy_folds(self.cfg)[0], self.cfg)
        np.testing.assert_array_equal(means, revised["x_mean"])
        self.assertEqual(fold["scale_crash_prior"][0], 0.)
        np.testing.assert_array_equal(fold["scale_variance"], [.01])

    def test_atomic_arrays_reject_pickle_and_leave_no_temporary_file(self):
        path = self.root / "test.npy"
        data._npy(path, np.arange(5))
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), np.arange(5))
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        with self.assertRaisesRegex(ValueError, "pickle"):
            data._npy(path, np.array([{}], dtype=object))


if __name__ == "__main__":
    unittest.main()
