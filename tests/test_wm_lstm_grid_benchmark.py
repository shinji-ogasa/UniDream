"""Frozen LSTM comparison grid, active-head and all-fit Test barrier checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from unidream.experiments import wm_lstm_grid_benchmark as runner


def fixture(root, all_fits=False):
    raw = yaml.safe_load(runner.DEFAULT_CONFIG.read_text())
    raw["output_dir"] = str(root / "study")
    raw["point_table_path"] = str(root / "old336.csv")
    (root / "old336.csv").write_text("previously frozen336 results")
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = runner.load_config(path)
    output = Path(cfg["output_dir"])
    reg = {"neural_tasks": runner.jobs(cfg), "classical_tasks": [],
        "parent_results_336_sha256": runner._sha(root / "old336.csv")}
    runner.atomic_json(output / "preregistration.json", reg)
    if all_fits:
        for task in reg["neural_tasks"]:
            add_fit(cfg, task)
    return cfg, reg


def add_fit(cfg, task):
    directory = runner.directory(cfg, task)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "research_model.pt").write_bytes(b"selected synthetic research checkpoint")
    (directory / "validation_predictions.parquet").write_bytes(b"synthetic frozen Val predictions")
    fit = {**task, "registration_sha256": runner._sha(Path(cfg["output_dir"]) / "preregistration.json"),
        "best_step": 200, "target_counts": 128, "supervised_gradient_steps": dict.fromkeys(runner.active_outputs(task), 1),
        "checkpoint_sha256": runner._sha(directory / "research_model.pt"),
        "validation_predictions_sha256": runner._sha(directory / "validation_predictions.parquet")}
    runner.atomic_json(directory / "fit.json", fit)
    return fit


def add_choices(cfg, reg):
    output = Path(cfg["output_dir"])
    selection = output / "selection/results_84.csv"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text("frozen,Val,selection\n")
    choices = {"registration_sha256": runner._sha(output / "preregistration.json"),
        "frozen_fit_sha256": runner.frozen_fit_manifest(cfg, reg), "test_used": False,
        "selection_report_sha256": runner._sha(selection)}
    runner.atomic_json(output / "validation_choices.json", choices)
    return choices


class LSTMGridRunnerTests(unittest.TestCase):
    def test180_independent_fits_cover84_conditions_on_same15folds(self):
        cfg = runner.load_config()
        tasks = runner.jobs(cfg)
        self.assertEqual(len(tasks), 180)
        self.assertEqual(len({task["job"] for task in tasks}), 180)
        self.assertEqual({task["model"] for task in tasks}, {"lstm"})
        self.assertEqual({task["seed"] for task in tasks}, {7})
        self.assertEqual({task["fold"] for task in tasks}, set(range(15)))
        conditions = {(task["group"], task["model"], target, task["formulation"]) for task in tasks for target in runner.TASKS}
        self.assertEqual(len(conditions), 84)

    def test_causal_state_and_fixed_labels_cannot_be_changed_by_config(self):
        for key in ("bidirectional", "carry_state_across_windows"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg, _ = fixture(root)
                raw = yaml.safe_load(Path(cfg["config_path"]).read_text())
                raw["model"][key] = True
                Path(cfg["config_path"]).write_text(yaml.safe_dump(raw))
                with self.assertRaisesRegex(ValueError, "causal LSTM"):
                    runner.load_config(cfg["config_path"])

    def test_sufficient_gradient_coverage_is_required_for_every_active_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, reg = fixture(Path(tmp))
            for task in reg["neural_tasks"][:3]:
                fit = add_fit(cfg, task)
                self.assertEqual(runner.check_fit(cfg, task), fit)
                self.assertEqual(len(fit["supervised_gradient_steps"]), 14 if task["formulation"] == "joint" else 7)
                fit["supervised_gradient_steps"][next(iter(fit["supervised_gradient_steps"]))] = 0
                runner.atomic_json(runner.directory(cfg, task) / "fit.json", fit)
                with self.assertRaisesRegex(ValueError, "gradient coverage"):
                    runner.check_fit(cfg, task)

    def test_checkpoint_or_Val_mutation_prevents_resuming_the_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, reg = fixture(Path(tmp))
            task = reg["neural_tasks"][0]
            add_fit(cfg, task)
            (runner.directory(cfg, task) / "research_model.pt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "checkpoint or validation"):
                runner.check_fit(cfg, task)

    def test_Test_cannot_open_before_all180_fits_are_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, reg = fixture(Path(tmp))
            add_fit(cfg, reg["neural_tasks"][0])
            with patch.object(runner, "verify", return_value=(reg, {}, {})):
                with self.assertRaises((FileNotFoundError, ValueError)):
                    runner.open_test_barrier(cfg)
            self.assertFalse((Path(cfg["output_dir"]) / "test_barrier.json").exists())

    def test_Test_barrier_binds_all_fits_Val_choices_and_frozen336_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, reg = fixture(Path(tmp), all_fits=True)
            add_choices(cfg, reg)
            with patch.object(runner, "verify", return_value=(reg, {}, {})):
                barrier = runner.open_test_barrier(cfg)
                self.assertEqual(len(barrier["frozen_fit_sha256"]), 180)
                self.assertEqual(runner.open_test_barrier(cfg), barrier)
                self.assertEqual(runner.test_access(cfg, reg["neural_tasks"][0]), barrier)
                (Path(cfg["output_dir"]) / "validation_choices.json").write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, "barrier changed"):
                    runner.test_access(cfg, reg["neural_tasks"][0])

    def test_Val_choices_from_another_registration_do_not_open_Test(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, reg = fixture(Path(tmp), all_fits=True)
            choices = add_choices(cfg, reg)
            choices["registration_sha256"] = "different"
            runner.atomic_json(Path(cfg["output_dir"]) / "validation_choices.json", choices)
            with patch.object(runner, "verify", return_value=(reg, {}, {})):
                with self.assertRaisesRegex(ValueError, "all180"):
                    runner.open_test_barrier(cfg)

    def test_missing_fit_cannot_be_trained_after_Test_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, reg = fixture(Path(tmp))
            runner.atomic_json(Path(cfg["output_dir"]) / "test_barrier.json", {})
            with patch.object(runner, "verify", return_value=(reg, {}, {})), \
                 patch.object(runner.parent, "arrays_for", side_effect=AssertionError("Train data opened after Test")):
                with self.assertRaisesRegex(ValueError, "cannot be trained after"):
                    runner.train_neural(cfg["config_path"], reg["neural_tasks"][0])

    def test_new_runner_preserves_frozen_prediction_frame_contract(self):
        from unidream.experiments.wm_full_grid_evaluation import fit_label_contract
        n = 3000
        rng = np.random.default_rng(17)
        arrays = {target: rng.normal(size=(n, 1)) for target in runner.TASKS}
        arrays["variance"] = np.exp(arrays["variance"])
        arrays["downside"] = np.exp(arrays["downside"]) * .01
        arrays["crash"] = (arrays["downside"] >= .012).astype(float)
        arrays["dd_excess"] = np.maximum(arrays["downside"] - .012, 0)
        arrays["recovery"] = np.tanh(arrays["recovery"])
        arrays["index_ns"] = np.arange(n) * 15 * 60 * 10**9
        contract = fit_label_contract(arrays, np.arange(n))
        ids = np.arange(100, 120)
        task = dict(fold=0, group="A", seed=7, model="lstm", formulation="joint")
        pred = {"regression": {t: arrays[t][ids] for t in runner.TASKS},
            "classification": {t: np.full((len(ids), 1, 2 if t == "crash" else 3), 1 / (2 if t == "crash" else 3)) for t in runner.TASKS}}
        frame = runner.prediction_frame(arrays, ids, pred, task, "selection", contract)
        self.assertEqual(set(frame.model), {"lstm"})
        self.assertEqual(set(frame.formulation), {"joint"})
        self.assertTrue(frame.p_crash_2.isna().all())
        self.assertTrue(np.isfinite(frame.scalar_return).all())


if __name__ == "__main__":
    unittest.main()
