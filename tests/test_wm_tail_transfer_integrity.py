import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from unidream.experiments import wm_tail_transfer_benchmark as campaign


class TailTransferIntegrityTests(unittest.TestCase):
    def test_barrier_binds_artifacts_and_each_selection_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            names = ['preregistration.json', 'comparison_plan.json', 'execution_manifest.json',
                     'input_selection.json', 'producer_selection.json', 'fits/one/fit.json']
            for name in names:
                campaign.write_json(out/name, {'name': name})
            artifact = 'fits/one/fit.json'
            barrier = dict(registration_sha256=campaign.sha(out/'preregistration.json'),
                           fit_sha256={artifact: campaign.sha(out/artifact)},
                           fit_file_identity={artifact: campaign.file_identity(out/artifact)})
            for name in names[1:5]:
                barrier[name.removesuffix('.json')+'_sha256'] = campaign.sha(out/name)
            campaign.write_json(out/'test_barrier.json', barrier)
            cfg = {'output_dir': directory}
            with patch.object(campaign, 'verify_registration', return_value={}):
                self.assertEqual(campaign.verify_test_barrier(cfg), barrier)
                self.assertEqual(campaign.verify_test_barrier(cfg, full=True), barrier)
                for name in names[1:5]:
                    original = (out/name).read_bytes()
                    (out/name).write_bytes(original+b' ')
                    with self.assertRaisesRegex(ValueError, 'selection manifest'):
                        campaign.verify_test_barrier(cfg)
                    (out/name).write_bytes(original)
                (out/artifact).write_text('{}')
                with self.assertRaisesRegex(ValueError, 'artifact metadata changed'):
                    campaign.verify_test_barrier(cfg)

    def test_legacy_barrier_always_checks_artifact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            files = ['preregistration', 'comparison_plan', 'execution_manifest', 'input_selection',
                     'producer_selection', 'model']
            for name in files:
                campaign.write_json(out/(name+'.json'), {'name': name})
            barrier = dict(registration_sha256=campaign.sha(out/'preregistration.json'),
                           fit_sha256={'model.json': campaign.sha(out/'model.json')})
            barrier.update({name+'_sha256': campaign.sha(out/(name+'.json')) for name in files[1:5]})
            campaign.write_json(out/'test_barrier.json', barrier)
            with patch.object(campaign, 'verify_registration', return_value={}):
                campaign.verify_test_barrier({'output_dir': directory})
                (out/'model.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'selection changed'):
                    campaign.verify_test_barrier({'output_dir': directory})

    def test_source_content_cache_invalidates_on_filesystem_change(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.py'
            source.write_text('first')
            first = campaign._verification_sha(source)
            self.assertEqual(first, campaign.sha(source))
            source.write_text('other')
            self.assertNotEqual(first, campaign._verification_sha(source))
            self.assertEqual(campaign._verification_sha(source, full=True), campaign.sha(source))


if __name__ == '__main__':
    unittest.main()
