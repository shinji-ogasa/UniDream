import copy
import unittest

from unidream.experiments import wm_tail_transfer_report as report


class StatisticalOutputCompletenessTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'splits': {'folds': list(range(7))}}
        self.manifest = {'statistical_jobs': [dict(job=f'custom_f{f}', fold=f) for f in range(7)]}
        self.records = [dict(task=dict(role='statistical', job=job['job']+'__'+model,
                                      fold=job['fold'], model=model))
                        for job in self.manifest['statistical_jobs'] for model in report.STATISTICAL_MODELS]

    def test_exact_fifty_six_outputs_and_unavailable_cells_are_complete(self):
        self.assertEqual(len(self.records), 56)
        self.records[0]['available'] = False
        self.records.append(dict(task=dict(role='benchmark', job='other', fold=0, model='wm')))
        report._validate_statistical_output_keyset(self.cfg, self.records, self.manifest)

    def test_missing_output_identifies_job_fold_and_model(self):
        with self.assertRaisesRegex(ValueError, 'missing.*custom_f0__zero_return'):
            report._validate_statistical_output_keyset(self.cfg, self.records[1:], self.manifest)

    def test_unexpected_output_and_metadata_mismatch_are_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]['task']['model'] = 'unexpected_model'
        with self.assertRaisesRegex(ValueError, 'unexpected_model'):
            report._validate_statistical_output_keyset(self.cfg, records, self.manifest)

    def test_duplicate_output_is_rejected_even_when_set_is_complete(self):
        with self.assertRaisesRegex(ValueError, 'duplicates.*custom_f0__zero_return'):
            report._validate_statistical_output_keyset(self.cfg, self.records+[self.records[0]], self.manifest)

    def test_manifest_cannot_silently_omit_a_registered_fold(self):
        manifest = {'statistical_jobs': self.manifest['statistical_jobs'][:-1]}
        with self.assertRaisesRegex(ValueError, 'exact fold coverage'):
            report._validate_statistical_output_keyset(self.cfg, self.records, manifest)

    def test_standalone_expected_names_come_from_configured_folds(self):
        records = [dict(task=dict(role='statistical', job=f'statistical_f{f:02d}__{model}',
                                  fold=f, model=model))
                   for f in range(7) for model in report.STATISTICAL_MODELS]
        report._validate_statistical_output_keyset(self.cfg, records)


if __name__ == '__main__':
    unittest.main()
