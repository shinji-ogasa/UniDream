import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from unidream.experiments import wm_classification_optimizer_amendment as stage


class OptimizerAmendmentContractTest(unittest.TestCase):
    def config(self,directory):
        out=Path(directory); config=out/'config.yaml';config.write_text('fixed config')
        (out/'preregistration.json').write_text('{}')
        return {'output_dir':str(out),'config_path':str(config),'tabular':
            {'logistic_Cs':[.001,.01,.1,1.],'logistic_tol':1e-4,'logistic_max_iter':1000}}

    def test_amendment_keeps_original_registration_and_freezes_its_source(self):
        with TemporaryDirectory() as folder,patch.object(stage,'verify'),patch.object(stage,'source_hashes',return_value={'amended.py':'fixed'}):
            cfg=self.config(folder); original=(Path(folder)/'preregistration.json').read_bytes()
            meta=stage.register(cfg); verified,_=stage.verify_amendment(cfg)
            self.assertEqual(meta,verified)
            self.assertEqual((Path(folder)/'preregistration.json').read_bytes(),original)
            self.assertEqual(meta['unchanged']['C_candidates'],[.001,.01,.1,1.])
            with patch.object(stage,'source_hashes',return_value={'amended.py':'changed'}):
                with self.assertRaises(ValueError):stage.verify_amendment(cfg)

    def test_new_registration_is_forbidden_after_test_or_completed_original_fits(self):
        for case in ('test','fit'):
            with TemporaryDirectory() as folder,patch.object(stage,'verify'):
                cfg=self.config(folder); out=Path(folder)
                if case=='test':(out/'test_access_barrier.json').write_text('{}')
                else:
                    fit=out/'classical/fold_00_lgbm';fit.mkdir(parents=True);(fit/'fit.json').write_text('{}')
                with self.assertRaises(ValueError):stage.register(cfg)

    def test_amendment_must_precede_test_barrier(self):
        with TemporaryDirectory() as folder,patch.object(stage,'verify'),patch.object(stage,'source_hashes',return_value={}):
            cfg=self.config(folder);stage.register(cfg)
            (Path(folder)/'test_access_barrier.json').write_text(json.dumps({'time_utc':'2020-01-01T00:00:00+00:00'}))
            with self.assertRaises(ValueError):stage.verify_amendment(cfg)
