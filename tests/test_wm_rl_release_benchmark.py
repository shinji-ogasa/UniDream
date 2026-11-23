"""Synthetic coordinator contracts; never load actual market arrays/models."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

from unidream.experiments import wm_rl_release_benchmark as benchmark


class ReleaseCoordinatorTests(unittest.TestCase):
    def fixture(self, root, *, fits=True):
        cfg = benchmark.load_config()
        cfg["output_dir"] = str(root)
        sources = {"source_root": str(root / "immutable_parent"),
                   "source_sha256": {"parent.py": "frozen-parent"},
                   "artifact_sha256": {"rl.pt": "source-rl", "bc.pt": "source-bc"},
                   "source_fits": 90}
        with patch.object(benchmark, "source_hashes", return_value={"new.py": "frozen-new"}), \
                patch.object(benchmark, "source_bindings", return_value=sources):
            registration = benchmark.register(cfg)
        if fits:
            self.write_fits(cfg)
        return cfg, sources, registration

    def write_fits(self, cfg):
        root = Path(cfg["output_dir"])
        registration_sha = benchmark.sha(root / "preregistration.json")
        for task in benchmark.jobs(cfg):
            path = root / task["fit_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            artifacts = {}
            for name in ("final.pt", "training_history.json", "validation_predictions.parquet"):
                artifact = path.parent / name
                artifact.write_bytes((task["job"] + name).encode())
                artifacts[name] = benchmark.sha(artifact)
            fit = {**task, "complete": True, "registration_sha256": registration_sha,
                   "updates": 200, "selected_update": 125,
                   "test_used": False, "initial_state_sha256": f"initial-{task['fold']}-{task['seed']}-{task['arm']}",
                   "market_start_sequence_sha256": f"market-{task['fold']}-{task['seed']}-{task['arm']}",
                   "immutable_bc_prior_state_sha256": f"bc-prior-{task['fold']}-{task['seed']}-{task['arm']}",
                   "optimizer_moments_inherited": False, "optimizer_reset_same_for_all_branches": True,
                   "optimizer_reset_at_stage_boundary": False, "bc_prior_immutable": True,
                   "wm_frozen": True,
                   "artifact_sha256": artifacts}
            benchmark.write_json(path, fit)

    def bindings(self, sources):
        return (patch.object(benchmark, "source_hashes", return_value={"new.py": "frozen-new"}),
                patch.object(benchmark, "source_bindings", return_value=sources))

    def freeze(self, cfg, sources):
        hashes, bindings = self.bindings(sources)
        with hashes, bindings:
            return benchmark.freeze_test_barrier(cfg)

    def verify_barrier(self, cfg, sources):
        hashes, bindings = self.bindings(sources)
        with hashes, bindings:
            return benchmark.verify_test_barrier(cfg)

    def mutate_fit(self, cfg, **changes):
        task = benchmark.jobs(cfg)[0]
        path = Path(cfg["output_dir"]) / task["fit_path"]
        value = json.loads(path.read_text())
        value.update(changes)
        benchmark.write_json(path, value)
        return path

    def test_exact180_grid_has_five_paired_branches_per_source_policy(self):
        cfg = benchmark.load_config()
        tasks = benchmark.jobs(cfg)
        self.assertEqual(len(tasks), 180)
        self.assertEqual(len({task["job"] for task in tasks}), 180)
        self.assertEqual(len({task["fit_path"] for task in tasks}), 180)
        expected = {(fold, seed, arm, condition) for fold in range(6) for seed in [7, 19, 42]
                    for arm in ["A", "C"] for condition in benchmark.CONDITIONS}
        self.assertEqual({(t["fold"], t["seed"], t["arm"], t["condition"]) for t in tasks}, expected)
        for task in tasks:
            self.assertEqual(task["fit_path"], f"neural/{task['job']}/fit.json")

    def test_schedule_budget_Val_window_Test_scope_and_costs_are_fixed(self):
        original = benchmark.load_config()
        mutations = [
            lambda c: c["continuation"].update(max_updates=199),
            lambda c: c["continuation"].update(selection_updates=[100, 125, 150, 175, 200]),
            lambda c: c["continuation"].update(early_stopping=True),
            lambda c: c["continuation"].update(wm_update_during_policy_training=True),
            lambda c: c["continuation"].update(rl_rollout_source="imagined"),
            lambda c: c.update(test_selection_allowed=True),
            lambda c: c.update(independently_unused_holdout=True),
            lambda c: c.update(historical_seen_test=False),
            lambda c: c.update(arms=["A", "B", "C"]),
            lambda c: c["schedules"]["floor_gradual"][1].update(position_min=.2),
            lambda c: c["schedules"]["prior_gradual"][0].update(through_update=60),
            lambda c: c["accounting"].update(cost_rate=.0001),
            lambda c: c["accounting"].update(position_max=2.),
            lambda c: c["accounting"].update(max_position_step=.16),
            lambda c: c["accounting"].update(drawdown_penalty=0.),
            lambda c: c["runtime"].update(cpu_threads=2),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.yaml"
            for mutate in mutations:
                cfg = copy.deepcopy(original)
                mutate(cfg)
                path.write_text(yaml.safe_dump(cfg))
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    benchmark.load_config(path)

    def test_registration_binds_sources_grid_and_known_Test_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, sources, registration = self.fixture(Path(directory), fits=False)
            self.assertEqual(len(registration["jobs"]), 180)
            self.assertEqual(registration["sources"], sources)
            self.assertFalse(registration["test_selection_allowed"])
            self.assertTrue(registration["historical_seen_test"])
            self.assertIn("fresh optimizer", registration["initialization"])
            self.assertIn("never reset", registration["optimizer_state"])
            self.assertIn("original immutable BC", registration["BC_reference"])
            self.assertFalse((Path(directory) / "selection_barrier.json").exists())
            hashes, bindings = self.bindings(sources)
            with hashes, bindings as source_check:
                self.assertEqual(benchmark.verify(cfg, check_sources=True), registration)
                source_check.assert_called_once()

    def test_registered_source_config_and_job_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, sources, _ = self.fixture(Path(directory), fits=False)
            hashes, bindings = self.bindings(sources)
            with hashes, bindings:
                changed = copy.deepcopy(cfg)
                changed["continuation"]["max_updates"] = 201
                with self.assertRaises(ValueError):
                    benchmark.verify(changed)
            with patch.object(benchmark, "source_hashes", return_value={"new.py": "changed"}):
                with self.assertRaises(ValueError):
                    benchmark.verify(cfg)
            with patch.object(benchmark, "source_hashes", return_value={"new.py": "frozen-new"}), \
                    patch.object(benchmark, "source_bindings", return_value={**sources, "source_fits": 89}):
                with self.assertRaises(ValueError):
                    benchmark.verify(cfg, check_sources=True)

    def test_all180_fits_freeze_before_Test_and_barrier_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, sources, _ = self.fixture(Path(directory))
            first = self.freeze(cfg, sources)
            self.assertEqual(len(first["frozen_fit_sha256"]), 180)
            self.assertEqual(set(first["frozen_fit_sha256"]), {t["fit_path"] for t in benchmark.jobs(cfg)})
            self.assertFalse(first["test_selection_allowed"])
            choices = json.loads((Path(directory) / "validation_choices.json").read_text())
            self.assertEqual(len(choices["choices"]), 180)
            self.assertFalse(choices["test_used"])
            self.assertEqual(first, self.freeze(cfg, sources))
            self.assertEqual(first, self.verify_barrier(cfg, sources))

    def test_missing_fit_prevents_any_global_Test_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, sources, _ = self.fixture(Path(directory))
            (Path(directory) / benchmark.jobs(cfg)[-1]["fit_path"]).unlink()
            with self.assertRaises((FileNotFoundError, ValueError)):
                self.freeze(cfg, sources)
            self.assertFalse((Path(directory) / "selection_barrier.json").exists())

    def test_paired_initial_weights_budget_and_final_Val_selection_are_required(self):
        for mutation in ({"initial_state_sha256": "different"},
                         {"market_start_sequence_sha256": "different-market-stream"},
                         {"immutable_bc_prior_state_sha256": "different-BC-prior"},
                         {"updates": 199}, {"selected_update": 100}):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                cfg, sources, _ = self.fixture(Path(directory))
                self.mutate_fit(cfg, **mutation)
                with self.assertRaises(ValueError):
                    self.freeze(cfg, sources)
                self.assertFalse((Path(directory) / "selection_barrier.json").exists())

    def test_fit_identity_and_Test_tainted_fit_are_rejected(self):
        for mutation in ({"fold": 5}, {"seed": 42}, {"arm": "C"},
                         {"condition": "floor_abrupt"}, {"test_used": True}):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                cfg, sources, _ = self.fixture(Path(directory))
                self.mutate_fit(cfg, **mutation)
                with self.assertRaises(ValueError):
                    self.freeze(cfg, sources)

    def test_checkpoint_mutation_prevents_freezing(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, sources, _ = self.fixture(Path(directory))
            checkpoint = Path(directory) / Path(benchmark.jobs(cfg)[0]["fit_path"]).parent / "final.pt"
            checkpoint.write_bytes(b"changed selected weights")
            with self.assertRaises(ValueError):
                self.freeze(cfg, sources)

    def test_late_checkpoint_and_fit_metadata_mutation_revoke_Test_barrier(self):
        for name in ("final.pt", "fit.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                cfg, sources, _ = self.fixture(Path(directory))
                self.freeze(cfg, sources)
                path = Path(directory) / Path(benchmark.jobs(cfg)[0]["fit_path"]).parent / name
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaises(ValueError):
                    self.verify_barrier(cfg, sources)

    def test_partial_barrier_or_Val_selection_mutation_is_rejected(self):
        for target in ("barrier", "Val"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                cfg, sources, _ = self.fixture(Path(directory))
                self.freeze(cfg, sources)
                path = Path(directory) / ("selection_barrier.json" if target == "barrier" else "validation_choices.json")
                value = json.loads(path.read_text())
                if target == "barrier":
                    value["frozen_fit_sha256"].pop(next(iter(value["frozen_fit_sha256"])))
                else:
                    value["test_used"] = True
                benchmark.write_json(path, value)
                with self.assertRaises(ValueError):
                    self.verify_barrier(cfg, sources)

    def test_run_never_starts_Test_pool_when_training_grid_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, sources, _ = self.fixture(Path(directory), fits=False)
            phases = []
            hashes, bindings = self.bindings(sources)
            with hashes, bindings, patch.object(benchmark, "run_pool", side_effect=lambda _cfg, evaluation_phase=False: phases.append(evaluation_phase)):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    benchmark.run(cfg)
            self.assertEqual(phases, [False])

    def test_run_opens_Test_pool_only_after_global_barrier_and_frozen_Val(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, sources, _ = self.fixture(root, fits=False)
            phases = []

            def pool(_cfg, evaluation_phase=False):
                phases.append(evaluation_phase)
                if not evaluation_phase:
                    self.assertFalse((root / "selection_barrier.json").exists())
                    self.write_fits(_cfg)
                else:
                    self.assertEqual(len(benchmark.verify_test_barrier(_cfg)["frozen_fit_sha256"]), 180)

            def aggregate(_cfg, registration_sha):
                self.assertEqual(phases, [False, True])
                (root / "final").mkdir()
                (root / "final/results.csv").write_text("synthetic,only\n1,1\n")
                return {"complete": True, "registration_sha256": registration_sha}

            module = types.SimpleNamespace(aggregate=aggregate)
            hashes, bindings = self.bindings(sources)
            with hashes, bindings, patch.object(benchmark, "run_pool", side_effect=pool), \
                    patch.object(benchmark, "progress"), \
                    patch.dict("sys.modules", {"unidream.experiments.wm_rl_release_evaluation": module}):
                completed = benchmark.run(cfg)
            self.assertTrue(completed["complete"])
            self.assertEqual(completed["fits"], 180)
            self.assertEqual(phases, [False, True])


if __name__ == "__main__":
    unittest.main()
