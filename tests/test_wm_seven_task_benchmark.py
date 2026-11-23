"""Cross-module split, objective and Test-barrier contracts without training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_seven_task_benchmark as benchmark
from unidream.experiments.wm_seven_task_evaluation import TASKS, build_seven_targets
from unidream.experiments.wm_seven_task_models import (
    SUPERVISED_WEIGHTS, DirectSevenTaskTransformer, SevenTaskWM,
)
from unidream.experiments.wm_technical_ablation import _features


class SevenTaskBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_calendar_validation_parts_are_disjoint_with_label_end_purges(self):
        index = pd.date_range("2024-01-01", "2024-04-01", freq="15min", inclusive="left", tz="UTC")
        ids = np.arange(len(index))
        parts = benchmark.validation_parts(ids, index.asi8, "2024-01-01T00:00:00Z", 96)
        boundaries = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"], utc=True)
        selected = []
        for position, key in enumerate(("earlystop", "calibration", "selection")):
            rows = parts[key]
            self.assertGreater(len(rows), 300)
            self.assertTrue((index[rows] >= boundaries[position]).all())
            self.assertTrue((index[rows] + pd.Timedelta(hours=24) < boundaries[position + 1]).all())
            self.assertEqual(index[rows[-1]], boundaries[position + 1] - pd.Timedelta(hours=24, minutes=15))
            selected.extend(rows)
        self.assertEqual(len(selected), len(set(selected)))
        # Leap-year February is a calendar month, not a fixed 30-day window.
        self.assertEqual(len(parts["calibration"]), 29 * 96 - 96)

    def test_validation_partition_rejects_insufficient_selection_support(self):
        index = pd.date_range("2024-01-01", "2024-03-02", freq="15min", inclusive="left", tz="UTC")
        with self.assertRaisesRegex(ValueError, "selection validation support"):
            benchmark.validation_parts(np.arange(len(index)), index.asi8, "2024-01-01T00:00:00Z", 16)

    def test_tabular_arms_keep_full_common_raw_history_and_optional_ta(self):
        context = 256
        columns = {key: benchmark.tabular_columns(key, context) for key in "ABCD"}
        self.assertEqual(len(columns["A"]), 256 * 5)
        self.assertEqual(len(columns["B"]), 256 * 5 + 44)
        self.assertEqual(len(columns["C"]), 256 * 19)
        self.assertEqual(len(columns["D"]), 256 * 19 + 44)
        for key in "ABCD":
            self.assertEqual(len(columns[key]), len(np.unique(columns[key])))
            self.assertTrue(set(columns[key]).issubset(set(columns["D"])))
        np.testing.assert_array_equal(columns["B"][-44:], columns["D"][-44:])
        np.testing.assert_array_equal(columns["C"].reshape(context, 19)[:, :5].ravel(), columns["A"])

    def test_missing_early_train_cache_rows_are_computed_without_changing_old_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            old_cache = root / "old_cache"
            (parent / "prepared").mkdir(parents=True)
            old_cache.mkdir()
            fold_path = parent / "prepared/fold_00.npz"
            np.savez(fold_path, train=np.array([400, 300]), val=np.array([500]), test=np.array([400]))
            np.save(old_cache / "decisions.npy", np.array([400, 500]))
            old_design = np.arange(2 * 4908, dtype=np.float32).reshape(2, 4908) / 100.
            np.save(old_cache / "tabular_raw.npy", old_design)
            cfg = {"output_dir": str(root / "output"), "tabular_cache": str(old_cache),
                   "folds": [0], "model": {"context": 256}}
            config_path = root / "config.yaml"
            config_path.write_text(json.dumps(cfg))
            source_before = {path: path.read_bytes() for path in (fold_path, config_path, *old_cache.iterdir())}
            cfg_before = json.dumps(cfg, sort_keys=True)
            feature = np.arange(256 * 63, dtype=np.float32).reshape(1, 256, 63) / 1000.
            arrays = {"raw": np.zeros((600, 5)), "market": np.zeros((600, 14))}
            with patch.object(benchmark, "_features", return_value=feature) as compute:
                decisions = benchmark.prepare_tabular_cache(cfg, {"output_dir": str(parent)}, {}, arrays)
            np.testing.assert_array_equal(decisions, [300, 400, 500])
            compute.assert_called_once()
            np.testing.assert_array_equal(compute.call_args.args[2], [300])
            self.assertEqual(compute.call_args.args[3], 256)
            cache = benchmark.cache_directory(cfg)
            design = np.load(cache / "tabular_raw.npy")
            self.assertEqual(design.shape, (3, 4908))
            expected_new = np.concatenate((feature[..., :19].reshape(1, -1), feature[:, -1, 19:]), axis=1)
            np.testing.assert_array_equal(design[0:1], expected_new)
            np.testing.assert_array_equal(design[1:], old_design)
            np.testing.assert_array_equal(benchmark.tabular_x(cfg, np.array([500, 300]), "D"), design[[2, 0]])
            np.testing.assert_array_equal(benchmark.tabular_x(cfg, np.array([300]), "A"), design[np.ix_([0], benchmark.tabular_columns("A"))])
            self.assertEqual(json.dumps(cfg, sort_keys=True), cfg_before)
            for path, original_bytes in source_before.items():
                self.assertEqual(path.read_bytes(), original_bytes)
            construction = json.loads((cache / "construction.json").read_text())
            self.assertEqual(construction["reused_rows"], 2)
            self.assertEqual(construction["computed_missing_train_rows"], 1)
            self.assertFalse(construction["source_modified"])

    def test_input_ends_before_decision_and_label_begins_at_decision(self):
        context, decision, horizon = 256, 280, 16
        n = decision + horizon + 1
        close = 100 * np.exp(np.arange(n) * .001)
        raw = np.column_stack((close, close * 1.01, close * .99, close, np.ones(n)))
        market = np.arange(n * 14).reshape(n, 14)
        with patch("unidream.experiments.wm_technical_ablation.transform_windows", side_effect=lambda bars, *a, **kw: bars):
            received = _features(raw, market, np.array([decision]), context)
            changed = raw.copy()
            changed[decision:] *= 100
            altered = _features(changed, market, np.array([decision]), context)
        np.testing.assert_array_equal(received, raw[decision - context:decision][None])
        np.testing.assert_array_equal(received, altered)
        index = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        targets = build_seven_targets(index, close, horizons=[horizon])
        self.assertTrue(targets["mask"][decision, 0])
        self.assertAlmostEqual(targets["return"][decision, 0], np.log(close[decision + horizon - 1] / received[0, -1, 3]))

    def test_numpy_selection_objective_matches_both_models_supervised_losses(self):
        self.assertEqual(benchmark.LOSS_WEIGHTS, SUPERVISED_WEIGHTS)
        cfg = {"context": 8, "d_model": 16, "n_heads": 2, "n_layers": 1,
               "d_ff": 32, "encoder_hidden": 16, "n_categoricals": 3,
               "n_classes": 4, "dropout": 0.}
        scales = {key: [.002 if key == "variance" else .02] * 2 for key in TASKS if key != "crash"}
        torch.manual_seed(7)
        x = torch.randn(2, 8, 10)
        targets = {key: torch.randn(2, 8, 2) * .01 for key in TASKS}
        for key in ("variance", "downside", "dd_excess"):
            targets[key] = targets[key].abs() + .0001
        targets["crash"] = (targets["crash"] > 0).float()
        for cls in (DirectSevenTaskTransformer, SevenTaskWM):
            model = cls(10, cfg, scales).eval()
            prediction = model(x)
            _, parts = model.loss(x, x[..., :5], targets, torch.ones(2, 8, 2, dtype=torch.bool), torch.zeros(2, 8))
            result = benchmark.objective_numpy({k: v.numpy() for k, v in targets.items()},
                {k: v.detach().numpy() for k, v in prediction.items()}, scales)
            self.assertAlmostEqual(result["score"], float(parts["supervised"]), places=5)
            for key in TASKS:
                self.assertAlmostEqual(result["parts"][key], float(parts[key]), places=5)

    def test_selection_objective_rejects_missing_predictions(self):
        values = {key: np.ones((3, 1)) for key in TASKS}
        values["crash"] = np.zeros((3, 1))
        predicted = {key: value.copy() for key, value in values.items()}
        predicted["return"][1] = np.nan
        scales = {key: [1.] for key in TASKS}
        with self.assertRaisesRegex(ValueError, "fully finite aligned"):
            benchmark.objective_numpy(values, predicted, scales)

    def _prediction_frame(self, model="direct", split="selection"):
        actual = {key: np.array([.01, -.005, .02, -.01]) for key in TASKS}
        for key in ("variance", "downside", "dd_excess"):
            actual[key] = np.abs(actual[key])
        actual["crash"] = np.array([0., 0., 1., 1.])
        arrays = {**{key: value[:, None] for key, value in actual.items()},
                  "index_ns": pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC").asi8}
        predicted = {key: value.copy() for key, value in actual.items()}
        predicted["crash"] = np.array([.1, .2, .7, .8])
        if model == "past_variance":
            predicted = {key: value if key == "variance" else np.full(4, np.nan)
                         for key, value in predicted.items()}
        return benchmark.prediction_frame(arrays, np.arange(4), predicted,
            {"fold": 0, "group": "D", "seed": 7, "model": model}, split)

    def test_variance_only_baseline_scoring_does_not_drop_its_valid_forecasts(self):
        scores = benchmark.score_frame(self._prediction_frame("past_variance"))
        self.assertEqual(set(scores), {"variance"})
        self.assertAlmostEqual(scores["variance"]["qlike"], 0.)
        self.assertAlmostEqual(scores["variance"]["rmse"], 0.)
        learned = self._prediction_frame()
        learned.loc[0, "pred_return"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite forecasts"):
            benchmark.score_frame(learned)

    def test_seed_rank_scores_are_averaged_without_cross_seed_comparisons(self):
        first = self._prediction_frame()
        second = first.copy()
        second["seed"] = 17
        second["pred_return"] += 100.
        second["pred_crash"] = [.8, .85, .9, .95]
        frame = pd.concat((first, second), ignore_index=True)
        average = benchmark.score_frame(frame)
        first_score, second_score = benchmark.score_frame(first), benchmark.score_frame(second)
        for task, metric in (("return", "pearson_ic"), ("return", "spearman_ic"), ("crash", "pr_auc")):
            self.assertAlmostEqual(average[task][metric], (first_score[task][metric] + second_score[task][metric]) / 2)
        naive = benchmark.score_seven({key: frame["actual_" + key].to_numpy() for key in TASKS},
                                      {key: frame["pred_" + key].to_numpy() for key in TASKS})
        self.assertGreater(average["return"]["pearson_ic"], naive["return"]["pearson_ic"])
        self.assertGreater(average["return"]["spearman_ic"], naive["return"]["spearman_ic"])
        self.assertGreater(average["crash"]["pr_auc"], naive["crash"]["pr_auc"])

    def test_summary_serializes_seven_metrics_and_ignores_test_before_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "neural/fixture"
            directory.mkdir(parents=True)
            self._prediction_frame().to_parquet(directory / "validation_predictions.parquet", index=False)
            self._prediction_frame(split="test").to_parquet(directory / "test_predictions.parquet", index=False)
            with patch.object(benchmark, "verify", return_value={"neural_tasks": [{"job": "fixture"}]}):
                benchmark.summarize({"output_dir": str(root)})
            metrics = pd.read_csv(root / "fold_metrics.csv")
            self.assertEqual(set(metrics.task), set(TASKS))
            self.assertEqual(set(metrics.split), {"selection"})
            self.assertEqual(len(metrics), 7)
            predictions = pd.read_parquet(root / "all_predictions.parquet")
            self.assertEqual(len(predictions), 4)
            summary = json.loads((root / "summary.json").read_text())
            self.assertTrue(summary["test_report_only"])
            self.assertEqual(len(summary["metrics"]), 7)

    def _barrier_fixture(self, root):
        registration = root / "preregistration.json"
        registration.write_text('{"experiment": "unit fixture"}')
        cfg = {"output_dir": str(root)}
        tasks = {"neural_tasks": [{"job": "neural_fixture"}], "classical_tasks": [{"job": "classical_fixture"}]}
        reg_hash = benchmark._sha(registration)
        for kind, name in (("neural", "neural_tasks"), ("classical", "classical_tasks")):
            directory = root / kind / tasks[name][0]["job"]
            directory.mkdir(parents=True)
            model = directory / ("research_model.pt" if kind == "neural" else "models.pkl")
            model.write_bytes(b"frozen model fixture")
            validation = directory / "validation_predictions.parquet"
            validation.write_bytes(b"frozen validation fixture")
            fit = {"registration_sha256": reg_hash,
                   "checkpoint_sha256" if kind == "neural" else "model_sha256": benchmark._sha(model),
                   "validation_predictions_sha256": benchmark._sha(validation)}
            (directory / "fit.json").write_text(json.dumps(fit))
        return cfg, tasks

    def test_global_barrier_requires_every_neural_and_classical_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, reg = self._barrier_fixture(root)
            missing = root / "classical/classical_fixture/fit.json"
            missing.unlink()
            with patch.object(benchmark, "verify", return_value=reg):
                with self.assertRaisesRegex(ValueError, "Test closed: missing"):
                    benchmark.open_test_barrier(cfg)
            self.assertFalse((root / "test_barrier.json").exists())

    def test_global_barrier_freezes_fit_provenance_and_detects_later_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, reg = self._barrier_fixture(root)
            with patch.object(benchmark, "verify", return_value=reg):
                benchmark.open_test_barrier(cfg)
                original = json.loads((root / "test_barrier.json").read_text())
                self.assertFalse(original["test_selection_allowed"])
                self.assertEqual(len(original["frozen_fit_sha256"]), 2)
                benchmark.open_test_barrier(cfg)
                fit = root / "neural/neural_fixture/fit.json"
                value = json.loads(fit.read_text())
                value["changed_after_barrier"] = True
                fit.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "fits changed after Test barrier"):
                    benchmark.open_test_barrier(cfg)

    def test_global_barrier_rejects_fit_from_another_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, reg = self._barrier_fixture(root)
            fit = root / "neural/neural_fixture/fit.json"
            value = json.loads(fit.read_text())
            value["registration_sha256"] = "another registration"
            fit.write_text(json.dumps(value))
            with patch.object(benchmark, "verify", return_value=reg):
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    benchmark.open_test_barrier(cfg)
            self.assertFalse((root / "test_barrier.json").exists())

    def test_global_barrier_rejects_changed_checkpoint_or_validation_predictions(self):
        for relative_path, message in (
            ("neural/neural_fixture/research_model.pt", "selected model artifact changed"),
            ("classical/classical_fixture/validation_predictions.parquet", "frozen validation predictions changed"),
        ):
            with self.subTest(relative_path=relative_path), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, reg = self._barrier_fixture(root)
                (root / relative_path).write_bytes(b"changed after fitting")
                with patch.object(benchmark, "verify", return_value=reg):
                    with self.assertRaisesRegex(ValueError, message):
                        benchmark.open_test_barrier(cfg)
                self.assertFalse((root / "test_barrier.json").exists())

    def test_test_worker_rejects_changed_checkpoint_before_loading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, reg = self._barrier_fixture(root)
            with patch.object(benchmark, "verify", return_value=reg):
                benchmark.open_test_barrier(cfg)
                (root / "neural/neural_fixture/research_model.pt").write_bytes(b"changed after barrier")
                with patch.object(benchmark, "load_config", return_value=cfg), patch.object(benchmark, "arrays_for") as load_arrays:
                    with self.assertRaisesRegex(ValueError, "selected checkpoint changed"):
                        benchmark.evaluate_task("unused", "neural", {"job": "neural_fixture"})
                    load_arrays.assert_not_called()

    def test_test_worker_stops_before_data_or_model_access_without_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(benchmark, "load_config", return_value={"output_dir": tmp}), patch.object(benchmark, "verify", return_value={}), patch.object(benchmark, "arrays_for") as load_arrays:
                with self.assertRaisesRegex(ValueError, "Test barrier not opened"):
                    benchmark.evaluate_task("unused", "neural", {"job": "fixture"})
                load_arrays.assert_not_called()


if __name__ == "__main__":
    unittest.main()
