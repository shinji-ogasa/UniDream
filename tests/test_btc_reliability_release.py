"""Synthetic contract checks; no market fit or production artifact is created."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from unidream.experiments import btc_reliability_release as release


def fixture():
    rng = np.random.default_rng(20260906)
    n = 800
    index = pd.date_range('2020-01-01', periods=n, freq='15min', tz='UTC')
    technical = rng.normal(size=(n, 29))
    perp = np.column_stack((technical, rng.normal(size=(n, 2))))
    y = .002 * technical[:, 0] + .001 * perp[:, -1] + .0005 * rng.normal(size=n)
    actual = np.column_stack((y, np.abs(y) + .001, np.sqrt(.0001 + .00005 * (technical[:, 1] > 0))))
    masks = {k: np.zeros(n, dtype=bool) for k in ('fit', 'scale', 'interval', 'predict', 'inference', 'score')}
    for k, start, end in [('fit', 0, 520), ('scale', 528, 600), ('interval', 608, 680), ('predict', 528, 780)]:
        masks[k][start:end] = True
    actual[~(masks['fit'] | masks['scale'] | masks['interval'])] = np.nan
    groups = {'technical': pd.DataFrame(technical, index=index), 'perp_delay0': pd.DataFrame(perp, index=index)}
    return groups, actual, masks


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.actual, cls.masks = fixture()
        cls.result = release.fit_production_inputs(cls.groups, cls.actual, cls.masks)

    def test_fixed_calendar_and_no_eval(self):
        dates = release.calendar(25)
        self.assertEqual(dates['fit_start'], pd.Timestamp('2024-07-16T13:45Z'))
        self.assertEqual(dates['fit_end'], pd.Timestamp('2026-01-16T13:45Z'))
        self.assertEqual(dates['scale_end'], pd.Timestamp('2026-04-16T13:45Z'))
        self.assertEqual(dates['evaluation_start'], pd.Timestamp('2026-07-16T13:45Z'))
        ix = pd.date_range(dates['fit_start'] - pd.Timedelta(days=1), dates['evaluation_start'], freq='15min', inclusive='left')
        masks = release.release_masks(ix, np.ones(len(ix), bool), np.ones(len(ix), bool))
        for key in ('inference', 'score', 'scheduled'):
            self.assertFalse(masks[key].any())
        for key in ('fit', 'scale', 'interval'):
            self.assertTrue((ix[masks[key]] + pd.Timedelta(minutes=375) < dates[key + '_end']).all())
        extended = ix.append(pd.DatetimeIndex([dates['evaluation_start']]))
        with self.assertRaisesRegex(ValueError, 'post-cutoff'):
            release.release_masks(extended, np.ones(len(extended), bool), np.ones(len(extended), bool))

    def test_evaluation_support_rejected_before_fit(self):
        masks = {k: v.copy() for k, v in self.masks.items()}
        masks['inference'][-1] = True
        with patch.object(release, 'fit_frozen_forecasts') as spy:
            with self.assertRaisesRegex(ValueError, 'evaluation support'):
                release.fit_production_inputs(self.groups, self.actual, masks)
            spy.assert_not_called()

    def test_empty_eval_still_fits_exact_three_models(self):
        r = self.result
        self.assertEqual(set(r['fitted']['models']), set(release.MODEL_NAMES))
        self.assertTrue(all(np.isnan(x).all() for x in r['fitted']['means'].values()))
        self.assertGreaterEqual(r['reliability']['weight'], 0.)
        self.assertLessEqual(r['reliability']['weight'], 1.)
        self.assertEqual(r['reliability']['n'], 72)
        self.assertTrue(all(v <= 1e-12 for v in r['scalar_prediction_max_abs_difference'].values()))
        self.assertTrue(np.isnan(r['mu'][~self.masks['predict']]).all())

    def test_interval_outcomes_cannot_fit_reliability(self):
        actual = self.actual.copy()
        actual[self.masks['interval'], 0] += 1.
        changed = release.fit_production_inputs(self.groups, actual, self.masks)
        self.assertEqual(changed['reliability'], self.result['reliability'])
        assert_array_equal(changed['mu'], self.result['mu'])
        assert_array_equal(changed['variance'], self.result['variance'])

    def test_constant_mix_and_variance_match_frozen_arithmetic(self):
        r = self.result; fit = r['fitted']; cal = fit['calibration']; pp = self.masks['predict']
        full = fit['raw_predictions']['perp_delay0']['mu'] + cal['return_bias']['perp_delay0']
        w = r['reliability']['weight']
        expected = full[pp] if w == 1. else np.full(int(pp.sum()), cal['scale_mean']) if w == 0. else w * full[pp] + (1 - w) * cal['scale_mean']
        assert_array_equal(r['mu'][pp], expected)
        assert_array_equal(r['variance'][pp], np.maximum(fit['raw_predictions']['technical']['variance'][pp] * cal['variance_multiplier'], 1e-12))

    def test_config_rejects_different_candidate_and_calendar(self):
        cfg = {**copy.deepcopy(release.FIXED), **{k: None for k in release.EXTRA}}
        cfg['source_bindings'] = {p: '0' * 64 for p in release.SOURCES}
        release.validate_config(cfg)
        for k, value in [('candidate_id', 'perp_delay0_half_utility_risk1'), ('calendar_fold', 24), ('scores_permitted', True), ('model_retry_permitted', True)]:
            other = {**cfg, k: value}
            with self.assertRaises(ValueError):
                release.validate_config(other)

    def test_file_writer_exclusive_and_alias_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'test.json'
            release._new_json(p, {'a': 1})
            with self.assertRaises(FileExistsError):
                release._new_json(p, {'a': 1})
            with self.assertRaisesRegex(ValueError, 'aliased'):
                release._bindings([{str(p): '0' * 64, str(Path(td) / 'x' / '..' / 'test.json'): '0' * 64}])

    def test_common_mask_covers_every_dependency(self):
        # Remove finite support in a dependency not consumed by the selected
        # return model; it must still remove common availability.
        ix = pd.date_range('2020-01-01', periods=4, freq='15min', tz='UTC')
        frame = lambda n: pd.DataFrame(np.ones((4, n)), index=ix)
        original = {'flow': frame(2)}
        derivative = {'base16': frame(16), 'technical': frame(29), 'perp_flow': frame(31), 'derivative': frame(37)}
        delayed = {'technical': frame(29), 'perp_delay0': frame(31), 'perp_delay1': frame(31), 'perp_delay4': frame(31)}
        derivative['derivative'].iloc[1, -1] = np.nan
        delayed['perp_delay4'].iloc[2, -1] = np.nan
        trailing = frame(4); trailing.iloc[3, -1] = np.nan
        with patch.object(release, 'make_feature_groups', return_value=original), patch.object(release, 'make_derivative_groups', return_value=derivative), patch.object(release, 'make_delayed_perp_groups', return_value=delayed), patch.object(release, 'trailing_variances', return_value=trailing):
            groups, common, _ = release.common_feature_inputs(frame(1), frame(1))
        self.assertEqual(set(groups), {'technical', 'perp_delay0'})
        assert_array_equal(common, [True, False, False, False])


if __name__ == '__main__':
    unittest.main()
