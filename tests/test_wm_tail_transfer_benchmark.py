"""Planner and chronology contracts without fitting or reading Test results."""
import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from unidream.experiments import wm_tail_transfer_benchmark as campaign
from unidream.experiments import wm_tail_transfer_runner as runner


class TailTransferPlannerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = campaign.load_config()

    def test_seven_folds_have_disjoint_producer_policy_and_evaluation_periods(self):
        folds = campaign.fold_definitions(self.cfg)
        self.assertEqual(len(folds), 7)
        for meta in folds:
            time = {k: pd.Timestamp(v) for k, v in meta.items() if k.endswith(("_start", "_end"))}
            self.assertEqual(time["producer_train_start"], time["train_start"])
            self.assertEqual(time["producer_train_end"], time["train_start"]+pd.DateOffset(months=10))
            self.assertEqual(time["producer_train_end"], time["producer_earlystop_start"])
            self.assertEqual(time["producer_earlystop_end"], time["producer_calibration_start"])
            self.assertEqual(time["producer_calibration_end"], time["policy_train_start"])
            self.assertEqual(time["policy_train_start"], time["train_start"]+pd.DateOffset(months=12))
            self.assertEqual(time["policy_train_end"], time["train_end"])
            self.assertEqual(time["train_end"], time["val_start"])
            self.assertEqual(time["earlystop_end"], time["calibration_start"])
            self.assertEqual(time["calibration_end"], time["selection_start"])
            self.assertEqual(time["selection_end"], time["test_start"])
            self.assertEqual(time["test_start"], time["val_end"])
        self.assertEqual(pd.Timestamp(folds[0]["test_start"]), pd.Timestamp("2025-04-01", tz="UTC"))
        self.assertEqual(pd.Timestamp(folds[-1]["test_start"]), pd.Timestamp("2026-10-01", tz="UTC"))
        self.assertEqual(pd.Timestamp(folds[-1]["test_end"]), pd.Timestamp("2026-11-01", tz="UTC"))
        for first, second in zip(folds[:-1], folds[1:]):
            self.assertEqual(first["test_end"], second["test_start"])

    def test_purge_uses_max_actual_97_bar_label_end_and_strict_upper_boundary(self):
        dates = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
        t = dates.asi8
        data = {"index_ns": t, "valid_D03": np.ones(300, bool),
                "label_end_ns": t[:, None]+np.array([1, 4, 16, 48, 96])[None, :]*900_000_000_000,
                "passive_label_end_ns": t[:, None]+np.array([2, 5, 17, 49, 97])[None, :]*900_000_000_000}
        ids = campaign.split_ids(data, "D03", dates[50], dates[250])
        self.assertTrue(np.array_equal(ids, np.arange(50, 153)))
        self.assertNotIn(153, ids)  # Its longest label ends exactly at the split boundary.
        self.assertTrue((data["passive_label_end_ns"][ids].max(1) < t[250]).all())
        unpurged = campaign.split_ids(data, "D03", dates[50], dates[250], require_labels=False)
        self.assertTrue(np.array_equal(unpurged, np.arange(50, 250)))

    def test_dense_and_sparse_input_comparisons_use_identical_timestamps(self):
        dates = pd.date_range("2024-01-01", periods=30, freq="15min", tz="UTC")
        dense = np.zeros(30, bool); dense[[4, 6, 8, 10, 12]] = True
        sparse = np.zeros(30, bool); sparse[[9, 15]] = True
        liquidation_only = sparse.copy(); liquidation_only[20] = True
        data = {"index_ns": dates.asi8, "valid_D03": np.ones(30, bool), "valid_D04": np.ones(30, bool),
                "valid_D06": liquidation_only, "valid_D07_dense": dense, "valid_D07": sparse,
                "label_end_ns": dates.asi8[:, None]}
        for group in ("D03", "D04"):
            actual = campaign.split_ids(data, group, dates[0], dates[-1], purge=False, cohort="dense_common")
            self.assertTrue(np.array_equal(actual, np.flatnonzero(dense)))
        for group in ("D03", "D06"):
            actual = campaign.split_ids(data, group, dates[0], dates[-1], purge=False, cohort="liquidation_common")
            self.assertTrue(np.array_equal(actual, np.flatnonzero(sparse)))
        for group in ("D03", "D06"):
            actual = campaign.split_ids(data, group, dates[0], dates[-1], purge=False, cohort="liquidation_only_common")
            self.assertTrue(np.array_equal(actual, np.flatnonzero(liquidation_only)))
        with self.assertRaisesRegex(ValueError, "cohort"):
            campaign.split_ids(data, "D03", dates[0], dates[-1], cohort="invented")

    def test_job_ids_include_actual_objective_and_cohort_and_collisions_raise(self):
        first = campaign.make_job("ablation", "D03", "wm", 0, 7, loss={"a": 1, "b": 2})
        reordered = campaign.make_job("ablation", "D03", "wm", 0, 7, loss={"b": 2, "a": 1})
        self.assertEqual(first["job"], reordered["job"])
        changed = campaign.make_job("ablation", "D03", "wm", 0, 7, objectives="classification")
        self.assertNotEqual(first["job"], changed["job"])
        dense = campaign.make_job("benchmark", "D03", "wm", 0, 7, cohort="dense_common")
        sparse = campaign.make_job("benchmark", "D03", "wm", 0, 7, cohort="liquidation_common")
        self.assertNotEqual(dense["job"], sparse["job"])
        self.assertEqual(len(runner._deduplicate([first, copy.deepcopy(first)])), 1)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            runner._deduplicate([first, {**first, "seed": 19}])

    def test_backend_assignment_is_paired_and_independent_of_outcomes(self):
        for model in ('lstm','transformer','wm'):
            assigned=set()
            for fold in self.cfg['splits']['folds']:
                for seed in self.cfg['seeds']:
                    task=campaign.make_job('benchmark','D03',model,fold,seed)
                    device=campaign.execution_device(self.cfg,task);assigned.add(device)
                    for group,objectives in [('D00','classification'),('D07','regression')]:
                        changed={**task,'group':group,'objectives':objectives,'validation_score':-1e6}
                        self.assertEqual(campaign.execution_device(self.cfg,changed),device)
            self.assertEqual(assigned,{'cpu','mps'})
        self.assertEqual(campaign.execution_device(self.cfg,dict(model='lightgbm',fold=0,seed=7)),'cpu')
        legacy=copy.deepcopy(self.cfg);legacy['runtime'].pop('forecast_cpu_fractions')
        self.assertEqual(campaign.execution_device(legacy,dict(model='wm',fold=0,seed=7)),'mps')
        legacy['runtime']['forecast_cpu_fractions']={'wm':1.1}
        with self.assertRaisesRegex(ValueError,'fraction'):
            campaign.execution_device(legacy,dict(model='wm',fold=0,seed=7))

    def test_declared_training_jobs_are_complete_unique_and_not_seed_inflated(self):
        screen, producers, ablations = campaign.screen_jobs(self.cfg), campaign.producer_jobs(self.cfg), campaign.ablation_jobs(self.cfg)
        self.assertEqual((len(screen), len(producers), len(ablations)), (396, 224, 231))
        for jobs in (screen, producers, ablations):
            self.assertEqual(len({x["job"] for x in jobs}), len(jobs))
        self.assertEqual({x["seed"] for x in screen if x["model"] == "linear"}, {7})
        self.assertEqual({x["seed"] for x in screen if x["model"] == "wm"}, {7, 19, 42})
        self.assertTrue(any(x["group"] == "D03" and x["cohort"] == "liquidation_common" for x in screen))
        self.assertTrue(any(x["group"] == "D07_dense" and x["cohort"] == "dense_common" for x in screen))
        single = [x for x in ablations if x.get("item") == "L01_single_task"]
        self.assertTrue(all(x['cohort']=='dense_common' for x in ablations))
        self.assertEqual({x["objectives"][0] for x in single}, {f"T{i:02d}" for i in range(1, 13)})

    def test_policy_2520_fits_and_1040_continuous_paths_have_separate_bc_and_sensitivities(self):
        tasks = campaign.policy_tasks(self.cfg)
        self.assertEqual(len(tasks), 36*7*10)
        self.assertEqual(len({x["job"] for x in tasks}), 2520)
        sources = {(f, s): {} for f in self.cfg["splits"]["folds"] for s in self.cfg["seeds"]}
        paths = runner.path_specs(self.cfg, tasks, sources)
        self.assertEqual(len(paths), 1040)
        self.assertEqual(sum(x["stage"] == "BC" for x in paths), 360)
        self.assertEqual(sum(x["stage"] == "RL" and x["sensitivity"] is None for x in paths), 360)
        self.assertEqual(sum(x["sensitivity"] is not None for x in paths), 320)
        self.assertEqual(len({x["job"] for x in paths}), len(paths))
        for path in paths:
            self.assertEqual([x["fold"] for x in path["tasks"]], list(range(7)))
            self.assertEqual(len({x["seed"] for x in path["tasks"]}), 1)
            if path["sensitivity"] is not None:
                self.assertEqual(path["condition"]["control"]["account_mode"], "linear_perp")
                self.assertEqual(path["stage"], "RL")

    def test_continuous_path_requires_each_fold_once(self):
        tasks = [x for x in campaign.policy_tasks(self.cfg) if x["policy_variant"] == "R00" and x["seed"] == 7
                 and x["control"]["account_mode"] == "cash_borrow" and x["control"]["cap"] == 1.12]
        sources = {(f, 7): {} for f in range(7)}
        with self.assertRaisesRegex(ValueError, "fold coverage"):
            runner.path_specs(self.cfg, tasks[:-1], sources)
        broken = copy.deepcopy(tasks); broken[1]["fold"] = broken[0]["fold"]
        with self.assertRaisesRegex(ValueError, "fold coverage"):
            runner.path_specs(self.cfg, broken, sources)

    def test_input_selection_never_uses_validation_after_each_test_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg); cfg["output_dir"] = tmp
            tasks = []
            for fold in (0, 3, 5):
                for group, score in {"D04": 1. if fold == 0 else 30., "D05": 2. if fold == 0 else .01,
                                     "D07_dense": 3. if fold == 0 else 20.}.items():
                    task = campaign.make_job("benchmark", group, "lightgbm", fold, 7, cohort="dense_common")
                    folder = Path(tmp)/"fits"/task["job"]; folder.mkdir(parents=True)
                    (folder/"fit.json").write_text(json.dumps({"validation_score": score, "test_score": -999999.}))
                    tasks.append(task)
            selection = runner.select_inputs(cfg, tasks)
            for fold in (0, 1, 2):
                self.assertEqual(selection[str(fold)]["selected"], "D04")
                self.assertEqual(selection[str(fold)]["eligible_screen_folds"], [0])
            for row in selection.values():
                self.assertLessEqual(pd.Timestamp(row["latest_used_validation_end"]), pd.Timestamp(row["test_start"]))
            # A future screen quarter's score changes cannot change an earlier
            # test's chosen input, even when future scores become extreme.
            for task in tasks:
                if task["fold"] == 5:
                    (Path(tmp)/"fits"/task["job"]/"fit.json").write_text(json.dumps({"validation_score": -1e12 if task["group"] == "D07_dense" else 1e12}))
            changed = runner.select_inputs(cfg, tasks)
            self.assertEqual(selection["0"], changed["0"])

    def test_producer_model_selection_uses_first_month_and_not_its_calibration_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg); cfg["output_dir"] = tmp
            tasks = campaign.producer_jobs(cfg)
            for task in tasks:
                if task["model"] not in ("linear", "lightgbm"):
                    continue
                directory = Path(tmp)/"fits"/task["job"]; directory.mkdir(parents=True)
                fit = {"earlystop_score": .4 if task["model"] == "linear" else .5,
                       "validation_score": 100. if task["model"] == "linear" else -100.}
                (directory/"fit.json").write_text(json.dumps(fit))
            sources = runner.producer_sources(cfg, tasks)
            self.assertEqual(len(sources), 70)
            for (fold, seed), value in sources.items():
                self.assertEqual(value["forecast"]["model"], "linear")
                self.assertEqual(value["old"]["seed"], seed)
                self.assertEqual(value["tail"]["fold"], fold)
                self.assertEqual([x["seed"] for x in value["ensemble"]], [7, 19, 42])

    def test_configuration_rejects_test_selection_overlap_and_free_resets(self):
        for field, value in (("test_selection_allowed", True), ("independently_unused_holdout", True)):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = copy.deepcopy(self.cfg); cfg[field] = value
                path = Path(tmp)/"config.yaml"; path.write_text(yaml.safe_dump(cfg))
                with self.assertRaises(ValueError):
                    campaign.load_config(path)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg); cfg["accounting"]["reset_positions_at_segment_boundary"] = True
            path = Path(tmp)/"config.yaml"; path.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "continuous account"):
                campaign.load_config(path)


if __name__ == "__main__":
    unittest.main()
