import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from unidream.experiments import wm_tail_transfer_training as training
from unidream.experiments import wm_tail_transfer_models as models


def fixture(root):
    n, context = 240, 8
    rng = np.random.default_rng(19)
    points = rng.normal(size=(n, 5)).astype(np.float32)
    positions = np.maximum(0, np.arange(n)[:, None]-context+1+np.arange(context)[None, :])
    windows = points[positions]
    close = 100*np.exp(np.cumsum(rng.normal(0, .001, n)))
    bars = np.column_stack((close, close*1.001, close*.999, close, np.ones(n)))
    ids = {"train": np.arange(8, 120), "earlystop": np.arange(120, 150),
           "calibration": np.arange(150, 180), "selection": np.arange(180, 210)}
    dates = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    ranges = {k: (dates[v[0]].isoformat(), dates[v[-1]+1].isoformat()) for k, v in ids.items()}
    cfg = {"output_dir": str(root), "device": "cpu", "model": {"context": context, "d_model": 8, "n_heads": 2,
            "n_layers": 1, "d_ff": 16, "encoder_hidden": 8, "n_categoricals": 2, "n_classes": 3, "dropout": 0.},
           "targets": {"horizons": [1, 4], "fixed_thresholds": [.01], "dynamic_multipliers": [1.],
                       "return_quantiles": [.05, .5, .95], "positive_quantiles": [.5, .95], "es_alphas": [.05],
                       "passive_leverages": [1., 2.], "passive_loss_thresholds": [.05], "hazard_thresholds": [.01],
                       "primary_horizon": 4, "barrier_ratios": [[1., 1.]]},
           "training": {"max_steps": 2, "min_steps": 1, "patience_steps": 3, "validation_every": 1,
                        "batch_size": 4, "evaluation_batch_size": 32, "selection_rows": 16, "learning_rate": .001},
           "classical": {"max_train_rows": 64, "feature_recipe": "window_summaries", "linear_steps": 2,
                         "linear_validation_every": 1, "linear_patience": 3, "linear_alpha": 1.},
           "runtime": {"cpu_threads": 1}}
    cfg["splits"] = {"first_train_start": "2021-01-01T00:00:00+00:00", "last_test_end_exclusive": "2024-01-01T00:00:00+00:00",
                     "train_months": 24, "validation_months": 3, "test_months": 3, "step_months": 3,
                     "producer_train_months": 10, "producer_validation_months": 2, "policy_train_months": 12, "folds": [0]}
    values = {"return": rng.normal(0, .01, (n, 2)).astype(np.float32),
              "rv": rng.uniform(.00001, .0002, (n, 2)).astype(np.float32),
              "reach_fixed": rng.binomial(1, .2, (n, 2, 2, 1)).astype(np.float32)}
    data = {"index_ns": dates.asi8, "ohlcspot": bars, "targets": values,
            "masks": {k: np.ones(v.shape, bool) for k, v in values.items()},
            "input_D03": points, "window_cache": windows, "valid_D03": np.ones(n, bool)}
    registration = Path(root)/"preregistration.json"; registration.write_text("{}")
    regsha = hashlib.sha256(registration.read_bytes()).hexdigest()
    return cfg, data, ids, ranges, regsha


class TailTransferTrainingTests(unittest.TestCase):
    def test_execution_device_is_used_for_fit_reload_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ids, ranges, regsha = fixture(tmp)
            cfg["device"] = "mps"
            task = {"job": "cpu_routed_wm", "role": "benchmark", "fold": 0, "seed": 7,
                    "group": "D03", "model": "wm", "objectives": ["return", "rv"]}
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch("unidream.experiments.wm_tail_transfer_benchmark.execution_device", return_value="cpu") as route, \
                 patch.object(training, "split_arrays", return_value=({"fold": 0}, ranges, ids)):
                fit = training.train(cfg, task, regsha, arrays=data)
                model, _, _, _ = training.load_fit(cfg, task, regsha)
                prediction = training.predict(cfg, task, regsha, np.arange(210, 220), split="producer", arrays=data)
            self.assertEqual(fit["execution_device"], "cpu")
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            self.assertEqual(cfg["device"], "mps")
            self.assertEqual(route.call_count, 3)
            self.assertTrue(all(call.args == (cfg, task) for call in route.call_args_list))
            self.assertTrue(np.isfinite(prediction["return"]).all())

    def test_latent_encoder_routes_by_source_fit_task_instead_of_native_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ids, ranges, regsha = fixture(tmp)
            source_task = {"job": "source_backend_wm", "role": "producer", "fold": 0, "seed": 7,
                           "group": "D03", "model": "wm", "objectives": ["return", "rv"]}
            probe_task = {**source_task, "job": "backend_latent_readout", "role": "ablation",
                          "architecture": "latent_probe", "source_job": source_task["job"], "model": "linear"}
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch.object(training, "split_arrays", return_value=({"fold": 0}, ranges, ids)):
                training.train(cfg, source_task, regsha, arrays=data)
            cfg["device"] = "mps"
            with patch("unidream.experiments.wm_tail_transfer_benchmark.execution_device", return_value="cpu") as route:
                manifest = training.prepare_latent_features(cfg, probe_task, regsha, arrays=data)
            route.assert_called_once_with(cfg, source_task)
            self.assertEqual(manifest["source_execution_device"], "cpu")
            self.assertFalse(manifest["encoder_updated"])
            self.assertTrue(np.isfinite(np.load(Path(manifest["directory"])/"features.npy")).all())

    def test_fresh_neural_and_classical_fit_reload_test_gate_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ids, ranges, regsha = fixture(tmp)
            for family in ("transformer", "wm", "linear"):
                task = {"job": family, "role": "benchmark", "fold": 0, "seed": 7, "group": "D03",
                        "model": family, "objectives": ["return", "rv", "reach_fixed"]}
                with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                     patch.object(training, "split_arrays", return_value=({"fold": 0}, ranges, ids)):
                    fit = training.train(cfg, task, regsha, arrays=data)
                self.assertTrue(fit["complete"])
                self.assertFalse(fit["test_read"])
                self.assertEqual(fit["actual_steps"], 0 if family == "linear" else 2)
                model, norm, saved, calibrators = training.load_fit(cfg, task, regsha)
                predicted = training.predict(cfg, task, regsha, np.arange(210, 220), split="producer", arrays=data)
                self.assertEqual(predicted["return"].shape, (10, 2))
                self.assertTrue(all(np.isfinite(v).all() for v in predicted.values()))
                self.assertEqual(saved["head_specs"][0]["name"], "return")
                self.assertTrue(np.allclose(norm["mean"], data["input_D03"][ids["train"]].mean(0), atol=1e-7))
                with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_test_barrier", side_effect=ValueError("closed Test")):
                    with self.assertRaisesRegex(ValueError, "closed Test"):
                        training.predict(cfg, task, regsha, np.arange(210, 220), split="test", arrays=data)

    def test_normalizers_do_not_change_when_future_targets_and_inputs_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ids, ranges, regsha = fixture(tmp)
            specs = models.active_specs(models.build_specs(cfg), ["return", "rv"])
            mean, std, scales = training.normalization(data, ids["train"], specs, "D03", cfg)
            data["input_D03"][120:] = 1e6
            data["targets"]["return"][120:] = 1e6
            mean2, std2, scales2 = training.normalization(data, ids["train"], specs, "D03", cfg)
            self.assertTrue(np.array_equal(mean, mean2)); self.assertTrue(np.array_equal(std, std2))
            self.assertEqual(scales, scales2)

    def test_tabular_recipe_preserves_eight_summaries_and_full_history_control(self):
        x = np.arange(2*8*3).reshape(2, 8, 3).astype(np.float32)
        summary = training.tabular_inputs(x)
        self.assertEqual(summary.shape, (2, 24))
        self.assertTrue(np.array_equal(summary[:, :3], x[:, -1]))
        self.assertTrue(np.array_equal(training.tabular_inputs(x, "full_flatten"), x.reshape(2, -1)))

    def test_auxiliary_return_alignment_and_current_summary_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ids, ranges, regsha = fixture(tmp)
            start, end = ranges["train"]
            reward, mask = training._auxiliary(data, np.array([40]), cfg, start, end, "D03")
            self.assertAlmostEqual(reward[0, -1], np.log(data["ohlcspot"][40, 3]/data["ohlcspot"][39, 3]), places=8)
            self.assertTrue(mask.all())
            _, sparse = training._auxiliary(data, np.array([40]), cfg, start, end, "D07")
            self.assertFalse(sparse[:, :-1].any()); self.assertTrue(sparse[:, -1].all())

    def test_calibration_fits_observed_month_only_and_native_shapes_remain(self):
        specs = [{"name": "prob", "kind": "binary", "shape": [1], "task": "T02"}]
        p = {"prob": np.linspace(.05, .95, 80)[:, None]}
        y = {"prob": (np.arange(80) > 50).astype(np.float32)[:, None]}
        m = {"prob": np.ones((80, 1), bool)}
        for method in ("none", "parametric", "isotonic"):
            fitted = training.fit_calibrators(p, y, m, specs, method)
            value = training.apply_calibrators(p, fitted)
            self.assertEqual(value["prob"].shape, (80, 1))
            self.assertTrue(np.all((value["prob"] >= 0) & (value["prob"] <= 1)))
        p = {"classes": np.tile([.2, .5, .3], (20, 1, 1))}
        y = {"classes": (np.arange(20) % 3).astype(np.float32)[:, None]}
        m = {"classes": np.ones((20, 1), bool)}
        specs = [{"name": "classes", "kind": "categorical", "shape": [1], "classes": 3, "task": "T03"}]
        calibrated = training.apply_calibrators(p, training.fit_calibrators(p, y, m, specs, "isotonic"))["classes"]
        self.assertTrue(np.allclose(calibrated.sum(-1), 1))
        self.assertTrue(np.isfinite(calibrated).all())

    def test_sparse_calendar_rows_and_future_oracle_respect_boundaries(self):
        values = np.array([[1., 2.], [3., 4.]], np.float32)
        rows = training.IndexedRows(np.array([2, 5]), values, 8)
        self.assertTrue(np.array_equal(rows[5], [3., 4.]))
        self.assertTrue(np.isnan(rows[[1, 2]])[0].all())
        dates = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
        data = {"index_ns": dates.asi8, "targets": {name: np.ones((20, 2), np.float32) for name in training.OracleRows.names},
                "masks": {name: np.ones((20, 2), bool) for name in training.OracleRows.names},
                "label_end_ns": dates.asi8[:, None]+np.array([1, 4])[None, :]*900_000_000_000,
                "passive_label_end_ns": dates.asi8[:, None]+np.array([2, 5])[None, :]*900_000_000_000}
        meta = {"producer_calibration_start": "2023-12-01T00:00:00+00:00", "producer_calibration_end": dates[0].isoformat(),
                "policy_train_start": dates[0].isoformat(), "policy_train_end": dates[10].isoformat(),
                "val_start": dates[10].isoformat(), "val_end": dates[15].isoformat(),
                "test_start": dates[15].isoformat(), "test_end": (dates[-1]+pd.Timedelta(minutes=15)).isoformat()}
        oracle = training.OracleRows(data, meta, [1, 4])
        self.assertEqual(oracle.shape, (20, 10))
        self.assertTrue(np.isfinite(oracle[3]).all())
        self.assertTrue(np.isnan(oracle[5]).all())

    def test_frozen_probe_inherits_exact_encoder_and_its_normalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ids, ranges, regsha = fixture(tmp)
            source_task = {"job": "source_wm", "role": "producer", "fold": 0, "seed": 7, "group": "D03",
                           "model": "wm", "objectives": ["return", "rv"]}
            probe_task = {**source_task, "job": "probe", "role": "ablation", "architecture": "frozen_probe",
                          "source_job": source_task["job"], "objectives": ["return", "rv", "reach_fixed"]}
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch.object(training, "split_arrays", return_value=({"fold": 0}, ranges, ids)):
                training.train(cfg, source_task, regsha, arrays=data)
                fit = training.train(cfg, probe_task, regsha, arrays=data)
            self.assertEqual(fit["normalizer_scope"], "source_frozen_Train_normalizer")
            probe, _, _, _ = training.load_fit(cfg, probe_task, regsha)
            self.assertTrue(probe.assert_backbone_unchanged())
            source, _, _, _ = training.load_fit(cfg, source_task, regsha)
            self.assertTrue(all(torch.equal(v, probe.wm.state_dict()[k]) for k, v in source.wm.state_dict().items()))

    def test_producer_cache_exports_sparse_causal_features_and_frozen_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ids, ranges, regsha = fixture(tmp)
            cfg["policy"] = {"source_input_group": "D03", "producer_calibration": "none"}
            dates = pd.to_datetime(data["index_ns"], utc=True)
            meta = {"fold": 0, "producer_calibration_start": dates[40].isoformat(), "producer_calibration_end": dates[80].isoformat(),
                    "policy_train_start": dates[80].isoformat(), "policy_train_end": dates[160].isoformat(),
                    "val_start": dates[160].isoformat(), "val_end": dates[200].isoformat(),
                    "test_start": dates[200].isoformat(), "test_end": dates[-1].isoformat()}
            data["label_end_ns"] = data["index_ns"][:, None]+np.array([1, 4])[None, :]*900_000_000_000
            data["passive_label_end_ns"] = data["label_end_ns"]+900_000_000_000
            tasks = [{"job": f"producer_{k}", "role": "producer", "group": "D03", "model": "wm", "seed": k, "fold": 0} for k in (7, 19, 42)]
            sources = {"old": tasks[0], "tail": tasks[1], "forecast": tasks[0], "ensemble": tasks}
            for task in tasks:
                directory = training.fit_directory(cfg, task); directory.mkdir(parents=True)
                (directory/"fit.json").write_text("{}")
            specs = [s for s in models.build_specs(cfg) if s["name"] in ("return", "rv", "reach_fixed", "mae")]
            fit = {"head_specs": specs, "scales": {"return": [.01, .02], "rv": [.0001, .0002], "mae": [.01, .02]}}
            def fake_predict(config, task, registration, raw_ids, **kwargs):
                count = len(raw_ids)
                p = {"return": np.full((count, 2), .0001*task["seed"], np.float32), "rv": np.full((count, 2), .0001, np.float32),
                     "reach_fixed": np.full((count, 2, 2, 1), .2, np.float32), "mae": np.full((count, 2), .01, np.float32)}
                return (p, np.full((count, 4), task["seed"], np.float32)) if kwargs.get("latents") else p
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch("unidream.experiments.wm_tail_transfer_benchmark.fold_definitions", return_value=[meta]), \
                 patch.object(training, "predict", side_effect=fake_predict), patch.object(training, "load_fit", return_value=(None, None, fit, None)):
                manifest = training.prepare_producer_cache(cfg, 0, 19, regsha, sources, arrays=data)
                policy_arrays = training.load_producer_cache(cfg, manifest, arrays=data)
                self.assertLess(manifest["producer_label_end_ns"], data["index_ns"][80])
                self.assertTrue(np.array_equal(policy_arrays["train_ids"], np.arange(80, 160)))
                self.assertTrue(np.array_equal(policy_arrays["val_ids"], np.arange(160, 200)))
                self.assertTrue(np.isfinite(policy_arrays["market"]["old_z"][[80, 120]]).all())
                self.assertTrue(np.isnan(policy_arrays["market"]["old_z"][[20]]).all())
                with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_test_barrier", side_effect=ValueError("closed Test")):
                    with self.assertRaisesRegex(ValueError, "closed Test"):
                        training.prepare_producer_cache(cfg, 0, 19, regsha, sources, split="test", arrays=data)

    def test_classical_latent_probe_uses_actual_frozen_encoder_not_raw_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, data, ids, ranges, regsha = fixture(tmp)
            source_task = {"job": "source_wm", "role": "producer", "fold": 0, "seed": 7, "group": "D03",
                           "model": "wm", "objectives": ["return", "rv"]}
            task = {**source_task, "job": "latent_readout", "role": "ablation", "architecture": "latent_probe",
                    "source_job": source_task["job"], "model": "linear"}
            with patch("unidream.experiments.wm_tail_transfer_benchmark.verify_registration", return_value={}), \
                 patch.object(training, "split_arrays", return_value=({"fold": 0}, ranges, ids)):
                training.train(cfg, source_task, regsha, arrays=data)
                fit = training.train(cfg, task, regsha, arrays=data)
            model, norm, saved, calibration = training.load_fit(cfg, task, regsha)
            self.assertEqual(fit["feature_recipe"], "latent_probe")
            self.assertEqual(model["input_dim"], 14)  # 2x3 categorical z + hidden8.
            self.assertEqual(len(fit["source_artifact_sha256"]), 3)
            predicted = training.predict(cfg, task, regsha, np.arange(210, 220), split="producer", arrays=data)
            self.assertEqual(predicted["return"].shape, (10, 2))
            self.assertTrue(np.isfinite(predicted["return"]).all())


if __name__ == "__main__":
    unittest.main()
