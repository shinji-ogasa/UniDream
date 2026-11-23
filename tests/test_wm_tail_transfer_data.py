import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from unidream.experiments.wm_tail_transfer_data import (
    BAR_NS, MIN_NS, DEPTH_COLUMNS, LIQ_COLUMNS, MINUTE_COLUMNS,
    _liquidation, build_targets, group_columns, window_inputs,
)


class TailTransferDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                                  'configs/experiments/wm_tail_transfer_20261112.yaml').read_text())

    def fixture(self, n=220):
        ix = pd.date_range('2023-01-01', periods=n, freq='15min', tz='UTC').asi8
        p = 100 * np.exp(np.arange(n) * .0001)
        raw = np.column_stack([p, p * 1.002, p * .998, p, np.ones(n)])
        return ix, raw

    def test_realized_semivariances_sum_to_variance(self):
        ix, raw = self.fixture()
        raw[40:, :4] *= .99
        target, mask, _ = build_targets(ix, raw, self.cfg)
        valid = mask['rv']
        np.testing.assert_allclose((target['up_semivariance'] + target['down_semivariance'])[valid],
                                   target['rv'][valid], rtol=1e-6)

    def test_horizon_masks_are_individual(self):
        ix, raw = self.fixture()
        target, mask, meta = build_targets(ix, raw, self.cfg)
        self.assertTrue(mask['return'][-1, 0])
        self.assertFalse(mask['return'][-1, 1])
        self.assertFalse(mask['passive_account_return'][-1].any())
        np.testing.assert_array_equal(meta['passive_label_end_ns'] - meta['predictive_label_end_ns'], BAR_NS)

    def test_mae_low_target_differs_from_legacy_close_crash(self):
        ix, raw = self.fixture()
        raw[50, 2] = 90
        target, mask, _ = build_targets(ix, raw, self.cfg)
        self.assertGreater(target['mae'][50, 0], .09)
        self.assertEqual(target['crash'][50, 0], 0)

    def test_same_bar_double_hit_is_unresolved_without_minutes(self):
        ix, raw = self.fixture()
        raw[50, 1:3] = [103, 97]
        target, mask, meta = build_targets(ix, raw, self.cfg)
        self.assertFalse(mask['barrier'][50, 0, 0, 0])
        self.assertFalse(mask['time_hazard'][50, 0, 0])
        self.assertEqual(meta['barrier_bounds'][50, 0, 0, 0, 0], 0)
        self.assertEqual(meta['barrier_bounds'][50, 0, 0, 0, 1], 2)

    def test_one_minute_order_resolves_first_passage(self):
        ix, raw = self.fixture()
        raw[50, 1:3] = [103, 97]
        minute = np.repeat(raw, 15, axis=0)
        minute[50 * 15:51 * 15, 1] = raw[50, 3]
        minute[50 * 15:51 * 15, 2] = raw[50, 3]
        minute[50 * 15 + 2, 1] = 103
        minute[50 * 15 + 10, 2] = 97
        target, mask, _ = build_targets(ix, raw, self.cfg, minute=minute, minute_start=ix[0])
        self.assertTrue(mask['barrier'][50, 0, 0, 0])
        self.assertEqual(target['barrier'][50, 0, 0, 0], 2)
        self.assertEqual(target['time_hazard'][50, 0, 0], 2)
        self.assertFalse(mask['time_hazard'][50, 1, 0])

    def test_one_minute_close_drawdown_keeps_observed_order(self):
        ix, raw = self.fixture()
        minute = np.repeat(raw, 15, axis=0)
        minute[50 * 15 + 2, 3] = 105
        minute[50 * 15 + 10, 3] = 95
        target, mask, _ = build_targets(ix, raw, self.cfg, minute=minute, minute_start=ix[0])
        self.assertTrue(mask['peak_dd'][50, 0])
        self.assertAlmostEqual(target['peak_dd'][50, 0], 1 - 95 / 105, places=6)

    def test_passive_label_fills_next_open_and_charges_actual_entry(self):
        ix, raw = self.fixture()
        raw[:, :4] = 100
        raw[51, 0] = 110
        cfg = copy.deepcopy(self.cfg)
        cfg['accounting']['borrow_apr'] = 0
        target, mask, _ = build_targets(ix, raw, cfg)
        lev = np.asarray(cfg['targets']['passive_leverages'])
        c = cfg['accounting']['cost_rate']
        expected = 1 - lev / (1 + c * lev) * (1 + c) + lev / (1 + c * lev) * 100 / 110 - 1
        np.testing.assert_allclose(target['passive_account_return'][50, 0], expected, atol=1e-7)

    def test_passive_low_ruin_is_absorbing(self):
        ix, raw = self.fixture()
        raw[:, :4] = 100
        raw[51, 2] = 50
        target, mask, _ = build_targets(ix, raw, self.cfg)
        self.assertEqual(target['ruin'][50, 0, -1], 1)
        self.assertEqual(target['passive_account_return'][50, 0, -1], -1)
        self.assertEqual(target['passive_account_return'][50, 1, -1], -1)

    def test_passive_loss_event_includes_intrabar_loss_then_recovery(self):
        ix, raw = self.fixture()
        raw[:, :4] = 100
        raw[51, 2] = 90
        target, mask, _ = build_targets(ix, raw, self.cfg)
        self.assertGreater(target['passive_account_return'][50, 0, 1], -.01)
        self.assertEqual(target['loss_probability'][50, 0, 1, 0], 1)

    def test_window_uses_no_outside_or_future_observation(self):
        ix, raw = self.fixture()
        data = {'index_ns': ix, 'ohlcspot': raw.copy(), 'market': np.zeros((len(ix), 14), np.float32)}
        x = window_inputs(data, [150], 'D03')
        data['ohlcspot'][:22] *= 3
        data['ohlcspot'][150:] *= 5
        data['market'][:22] = 2
        data['market'][150:] = 3
        np.testing.assert_array_equal(window_inputs(data, [150], 'D03'), x)
        self.assertEqual(x[0, 0, 3], 0)
        self.assertEqual(x.shape[-1], len(group_columns('D03')))

    def test_sparse_liquidation_current_summary_does_not_impute_history(self):
        ix, raw = self.fixture()
        liq = np.full((len(ix), len(LIQ_COLUMNS)), np.nan, np.float32)
        liq[149] = [1, 2, 3, 4]
        data = {'index_ns': ix, 'ohlcspot': raw, 'market': np.zeros((len(ix), 14), np.float32),
                'liquidation_features': liq}
        x = window_inputs(data, [150], 'D06')
        np.testing.assert_array_equal(x[0, :, -4:], np.tile(liq[149], (128, 1)))
        with self.assertRaises(ValueError):
            window_inputs(data, [151], 'D06')

    def test_liquidation_availability_uses_collector_clock(self):
        ix, raw = self.fixture()
        event = pd.DataFrame({'timestamp': [ix[0] // 1000 + 1],
                              'local_timestamp': [ix[1] // 1000 + 1],
                              'side': ['sell'], 'price': [100.], 'amount': [2.]})
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'archives/liquidation_sample'; folder.mkdir(parents=True)
            event.to_csv(folder / '2023-01-01.csv.gz', index=False, compression='gzip')
            values, audit = _liquidation(Path(d), ix)
        self.assertEqual(values[0, 0], 0)
        self.assertGreater(values[1, 0], 0)
        self.assertTrue(np.isnan(values[100]).all())


if __name__ == '__main__':
    unittest.main()
