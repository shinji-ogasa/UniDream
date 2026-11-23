import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from unidream.experiments import wm_control_release_benchmark as benchmark


class ControlReleaseBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(benchmark.DEFAULT_CONFIG.read_text())

    def test_registered_grid_contains_every_seed_arm_condition_and_frozen_control(self):
        benchmark.validate_contract(self.cfg)
        fits, paths = benchmark.fit_jobs(self.cfg), benchmark.evaluation_jobs(self.cfg)
        self.assertEqual(len(fits), 600)
        self.assertEqual(len({x['job'] for x in fits}), 600)
        self.assertEqual(len(benchmark.context_jobs(self.cfg)), 120)
        self.assertEqual(len(paths), 200)
        self.assertEqual(sum(x['kind'] == 'trained' for x in paths), 100)
        self.assertEqual(sum(x['kind'] == 'frozen' for x in paths), 100)

    def test_test_selection_and_reset_cannot_be_enabled(self):
        for key in ('test_selection_allowed', 'independently_unused_holdout'):
            cfg = copy.deepcopy(self.cfg); cfg[key] = True
            with self.assertRaises(ValueError): benchmark.validate_contract(cfg)
        cfg = copy.deepcopy(self.cfg); cfg['accounting']['reset_positions_at_segment_boundary'] = True
        with self.assertRaises(ValueError): benchmark.validate_contract(cfg)

    def test_risk_objective_and_costs_are_not_relaxed_with_position_limits(self):
        for name in ('drawdown_penalty', 'cost_rate', 'borrow_apr', 'execution_delay_bars'):
            cfg = copy.deepcopy(self.cfg); cfg['accounting'][name] = 0
            with self.assertRaises(ValueError): benchmark.validate_contract(cfg)

    def test_shorter_budget_or_post_release_selection_is_not_silently_changed(self):
        for field, value in [('max_updates', 100), ('selection_updates', [50, 100, 150, 200]),
                             ('validation_scope', 'first16days'), ('trainable', ['policy_head'])]:
            cfg = copy.deepcopy(self.cfg); cfg['training'][field] = value
            with self.assertRaises(ValueError): benchmark.validate_contract(cfg)

    def test_no_registered_branch_or_seed_can_be_removed_after_results(self):
        for field in ('conditions', 'seeds', 'arms', 'folds'):
            cfg = copy.deepcopy(self.cfg); cfg[field] = cfg[field][:-1]
            with self.assertRaises(ValueError): benchmark.validate_contract(cfg)

    def test_all_branches_have_the_same_first50_updates(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['schedules']['free'][0]['bc_prior_coefficient'] = 0
        with self.assertRaises(ValueError): benchmark.validate_contract(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg['schedules']['speed'][1]['through_update'] = 150
        with self.assertRaises(ValueError): benchmark.validate_contract(cfg)

    def test_missing_family_comparisons_and_price_rows_are_rejected(self):
        for field, value in [('primary_family_size', 12), ('secondary_family_size', 39),
                             ('combined_family_size', 52), ('expected_market_rows', 50038)]:
            cfg = copy.deepcopy(self.cfg); cfg['statistics'][field] = value
            with self.assertRaises(ValueError): benchmark.validate_contract(cfg)

    def _fits(self, root):
        cfg = copy.deepcopy(self.cfg); cfg['output_dir'] = str(root)
        cfg['conditions'] = ['control']; cfg['source_pairs'] = {}
        tasks = [dict(job='fold_00_A_s7_control', fold=0, arm='A', seed=7, condition='control'),
                 dict(job='fold_00_C_s7_control', fold=0, arm='C', seed=7, condition='control')]
        for task in tasks:
            directory = root/'neural'/task['job']; directory.mkdir(parents=True)
            (directory/'final.pt').write_bytes(b'selected-head-state')
            value = dict(complete=True, registration_sha256='registered', **task, test_used=False,
                         selected_update=125, encoder_unchanged=True, head_changed=True,
                         updates=200, historical_transitions=204800, market_encoders_trainable=False,
                         wm_frozen=True, bc_prior_immutable=True, optimizer_moments_inherited=False,
                         optimizer_reset_at_stage_boundary=False, initial_state_sha256='initial',
                         initial_encoder_sha256='encoder', immutable_bc_prior_state_sha256='prior',
                         continuation_random_seed=7, market_start_sequence_sha256='stream',
                         context_manifest_sha256='context', source_initial_checkpoint_sha256='initial-source',
                         source_bc_checkpoint_sha256='bc-source',
                         artifact_sha256={'final.pt': benchmark.sha(directory/'final.pt')})
            benchmark.write_json(directory/'fit.json', value)
            cfg['source_pairs'][f"fold_00_{task['arm']}_s7"] = dict(initial_checkpoint_path='source.pt',
                bc_checkpoint_path='bc.pt', source_artifact_sha256={'source.pt':'initial-source','bc.pt':'bc-source'})
        return cfg, tasks

    def test_barrier_requires_all_completed_fits_before_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, tasks = self._fits(root)
            (root/'neural'/tasks[1]['job']/'fit.json').unlink()
            with patch.object(benchmark, 'verify', return_value='registered'), patch.object(benchmark, 'fit_jobs', return_value=tasks):
                with self.assertRaises(FileNotFoundError): benchmark.freeze_test_barrier(cfg)
            self.assertFalse((root/'test_barrier.json').exists())

    def test_consumer_rejects_empty_or_missing_barrier_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, tasks = self._fits(root)
            with patch.object(benchmark, 'verify', return_value='registered'), patch.object(benchmark, 'fit_jobs', return_value=tasks):
                benchmark.freeze_test_barrier(cfg)
                benchmark.verify_test_barrier(cfg, full=True)
                value = json.loads((root/'test_barrier.json').read_text())
                value['frozen_fit_sha256'] = {}
                benchmark.write_json(root/'test_barrier.json', value)
                with self.assertRaisesRegex(ValueError, 'exact complete set'): benchmark.verify_test_barrier(cfg)

    def test_selected_weights_tampering_is_detected_by_full_barrier_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, tasks = self._fits(root)
            with patch.object(benchmark, 'verify', return_value='registered'), patch.object(benchmark, 'fit_jobs', return_value=tasks):
                benchmark.freeze_test_barrier(cfg)
                (root/'neural'/tasks[0]['job']/'final.pt').write_bytes(b'changed')
                with self.assertRaises(ValueError): benchmark.verify_test_barrier(cfg, full=True)

    def test_fit_with_test_selection_or_encoder_update_cannot_enter_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, tasks = self._fits(root)
            path = root/'neural'/tasks[0]['job']/'fit.json'
            value = json.loads(path.read_text()); value['test_used'] = True
            benchmark.write_json(path, value)
            with self.assertRaises(ValueError): benchmark.checked_fit(cfg, tasks[0], 'registered')
            value['test_used'] = False; value['encoder_unchanged'] = False
            benchmark.write_json(path, value)
            with self.assertRaises(ValueError): benchmark.checked_fit(cfg, tasks[0], 'registered')


if __name__ == '__main__':
    unittest.main()
