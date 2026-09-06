"""Synthetic guards only; no financial data, model fitting or outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from unidream.experiments import wm31_production_release as production


class ProductionConfigGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base_path = self.root / 'registered_screen.yaml'
        self.base = {key: {'unchanged': key} for key in production.TRAINING_SECTIONS}
        self.base.update(release={'execution': {'oneway_cost': .00055}, 'device': 'cpu'},
                         run={'deterministic_algorithms': False})
        self.base_path.write_text(yaml.safe_dump(self.base))
        self.selection = {'schema': 'wm-rl-paper-selection-v1', 'qualified': True,
            'selected_arm': 'ac_decay_dd25',
            'source_config_sha256': production.file_digest(self.base_path)}
        self.selection_path = self.root / 'selection.json'
        self.selection_path.write_text(json.dumps(self.selection))
        artifacts = {}
        for name in ('spot_15m.parquet', 'spot_15m_availability.parquet',
                     'um_15m.parquet', 'um_15m_availability.parquet'):
            p = self.root / name
            p.write_bytes(('synthetic bytes: ' + name).encode())
            artifacts[name] = {'path': str(p), 'sha256': production.file_digest(p)}
        self.input_path = self.root / 'inputs.json'
        self.input_path.write_text(json.dumps({'status': 'production_input_snapshot_complete',
            'training_start': production.START.isoformat(),
            'cutoff_exclusive': production.CUTOFF.isoformat(), 'artifacts': artifacts}))
        self.cfg = copy.deepcopy(self.base)
        self.cfg['release'] = {'schema': production.SCHEMA, 'group': 'perp_delay0',
            'evaluation_split': 'none', 'economic_scoring': False, 'test_scoring': False,
            'screen_only': False, 'seed': 7, 'wm_selection': 'fixed_endpoint700_train_only',
            'fit_market_models': 1, 'learned_rl_arms_per_model': 1,
            'folds': [{'fold': 25, 'train_start': production.START.isoformat(),
                       'train_end': production.CUTOFF.isoformat()}],
            'source_screen_config': str(self.base_path),
            'source_screen_config_sha256': production.file_digest(self.base_path),
            'execution': self.base['release']['execution'], 'device': 'cpu',
            'production_input_manifest': str(self.input_path),
            'production_input_manifest_sha256': production.file_digest(self.input_path),
            'selected_arm': 'ac_decay_dd25', 'ac_arm_names': ['ac_decay_dd25'],
            'selection_manifest_path': str(self.selection_path),
            'selection_manifest_sha256': production.file_digest(self.selection_path)}
        for market in ('spot', 'um'):
            for suffix, name in [('_path', market + '_15m.parquet'),
                    ('_availability_path', market + '_15m_availability.parquet')]:
                self.cfg['release'][market + suffix] = artifacts[name]['path']
                self.cfg['release'][market + suffix.replace('_path', '_sha256')] = artifacts[name]['sha256']
        self.cfg['run'].update(folds=[25], start=production.START.isoformat(),
                              end=production.CUTOFF.isoformat(), clean_checkpoint_dir=False)
        self.path = self.root / 'production.yaml'

    def check(self, *, fitting=True):
        self.path.write_text(yaml.safe_dump(self.cfg))
        return production.validate_config(self.path, fitting=fitting)

    def update_selection(self):
        self.selection_path.write_text(json.dumps(self.selection))
        self.cfg['release']['selection_manifest_sha256'] = production.file_digest(self.selection_path)

    def test_valid_qualified_contract_binds_without_training(self):
        with patch('unidream.experiments.wm_rl_training.train_wm_bc_ac') as train:
            cfg, manifest = self.check()
            self.assertEqual(cfg['release']['ac_arm_names'], ['ac_decay_dd25'])
            self.assertEqual(len(manifest['artifacts']), 4)
            train.assert_not_called()

    def test_preflight_allows_unbound_selection_but_fit_rejects(self):
        self.cfg['release']['selection_manifest_sha256'] = None
        self.check(fitting=False)
        with self.assertRaises(ValueError):
            self.check()

    def test_selection_requires_explicit_qualification(self):
        for bad in (False, None, 1, 'true'):
            with self.subTest(bad=bad):
                self.selection['qualified'] = bad
                self.update_selection()
                with self.assertRaisesRegex(ValueError, 'qualified selection'):
                    self.check()

    def test_selection_must_match_schema_arm_and_source(self):
        for key, bad in [('schema', 'other'), ('selected_arm', 'ac_anchor_dd25'),
                         ('source_config_sha256', '0' * 64)]:
            with self.subTest(key=key):
                prior = self.selection[key]
                self.selection[key] = bad
                self.update_selection()
                with self.assertRaisesRegex(ValueError, 'qualified selection'):
                    self.check()
                self.selection[key] = prior

    def test_single_selected_arm_and_trainer_agree(self):
        self.cfg['release']['ac_arm_names'] = ['ac_decay_dd25', 'ac_anchor_dd25']
        with self.assertRaisesRegex(ValueError, 'exactly'):
            self.check()
        self.cfg['release']['ac_arm_names'] = ['ac_decay_dd25']
        with patch('unidream.experiments.wm_rl_training.selected_ac_arms', return_value={}):
            with self.assertRaisesRegex(ValueError, 'resolve exactly'):
                self.check()

    def test_unregistered_learning_change_is_rejected(self):
        self.cfg['world_model']['extra_scale'] = 100
        with self.assertRaisesRegex(ValueError, 'training section'):
            self.check()

    def test_validation_test_or_shifted_period_is_rejected(self):
        for key, value in [('evaluation_split', 'validation'), ('test_scoring', True),
                           ('economic_scoring', True)]:
            with self.subTest(key=key):
                before = self.cfg['release'][key]
                self.cfg['release'][key] = value
                with self.assertRaisesRegex(ValueError, 'production-only'):
                    self.check()
                self.cfg['release'][key] = before
        self.cfg['release']['folds'][0]['train_end'] = '2026-09-02T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'interval'):
            self.check()

    def test_changed_bound_data_or_screen_fails(self):
        Path(self.cfg['release']['um_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bound artifact'):
            self.check()
        self.base_path.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'bound artifact'):
            self.check()

    def test_no_checkpoint_deletion(self):
        self.cfg['run']['clean_checkpoint_dir'] = True
        with self.assertRaisesRegex(ValueError, 'delete/reuse'):
            self.check()


if __name__ == '__main__':
    unittest.main()
