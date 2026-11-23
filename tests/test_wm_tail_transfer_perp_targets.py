import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from unidream.experiments import wm_tail_transfer_data as data_module
from unidream.experiments import wm_tail_transfer_perp_targets as labels


def config():
    return {"targets": {"horizons": [1, 2], "passive_leverages": [.5, 1., 2., 5.]},
        "accounting": {"cost_rate": 0., "borrow_apr": .1, "annual_bars": 35040},
        "perpetual": {"fee_rate": 0., "liquidation_fee_rate": .005,
            "primary_maintenance_margin_rate": .005, "primary_slippage_rate": 0.}}


def arrays(n=8):
    candles = np.broadcast_to([100., 100., 100., 100., 0.], (n, 5)).copy()
    return {"index_ns": np.arange(n, dtype=np.int64) * data_module.BAR_NS,
        "ohlcperp": candles.copy(), "ohlcmark": candles.copy(),
        "funding_settlement_rate": np.zeros(n), "funding_event_count": np.zeros(n, np.int16)}


class PerpetualTargetTests(unittest.TestCase):
    def test_decision_enters_next_open_and_then_holds_fixed_units(self):
        cfg, data = config(), arrays()
        # Decision0's own bar is unscored; entry is100 at row1, then110 close2.
        data["ohlcperp"][0, :4] = [1000., 1000., 1000., 1000.]
        data["ohlcmark"][0, :4] = [1000., 1000., 1000., 1000.]
        data["ohlcperp"][2, :4] = [100., 110., 100., 110.]
        data["ohlcmark"][2, :4] = [100., 110., 100., 110.]
        targets, masks = labels.build_perpetual_targets(data, cfg, chunk_rows=3)
        np.testing.assert_allclose(targets["perp_passive_return"][0, 0], 0., atol=1e-7)
        np.testing.assert_allclose(targets["perp_passive_return"][0, 1], [.05, .1, .2, .5], atol=1e-7)
        self.assertTrue(masks["perp_passive_return"][0].all())
        self.assertFalse(masks["perp_passive_return"][-1].any())
        self.assertTrue(masks["perp_passive_return"][-2, 0].all())
        self.assertFalse(masks["perp_passive_return"][-2, 1].any())

    def test_fee_slippage_and_mark_trade_basis_independent_equation(self):
        cfg, data = config(), arrays()
        cfg["perpetual"].update(fee_rate=.00055, primary_slippage_rate=.0005)
        data["ohlcmark"][:, :4] = 101.
        target, _ = labels.build_perpetual_targets(data, cfg)
        lev = np.asarray(cfg["targets"]["passive_leverages"])
        fill = 100. * 1.0005
        units = lev / (101. + lev * (fill - 101.) + lev * .00055 * fill)
        expected = 1. - units * fill * 1.00055 + units * 101. - 1.
        np.testing.assert_allclose(target["perp_passive_return"][0, 0], expected, atol=1e-7)
        np.testing.assert_allclose(target["perp_passive_return"][0, 1], expected, atol=1e-7)

    def test_actual_funding_once_with_no_rebalancing(self):
        cfg, data = config(), arrays()
        data["funding_settlement_rate"][1] = .01
        data["funding_event_count"][1] = 1
        target, _ = labels.build_perpetual_targets(data, cfg)
        expected = -np.asarray(cfg["targets"]["passive_leverages"]) * .01
        np.testing.assert_allclose(target["perp_passive_return"][0, 0], expected, atol=1e-7)
        np.testing.assert_allclose(target["perp_passive_return"][0, 1], expected, atol=1e-7)
        # The event was before decision1's entry and must not be charged again.
        np.testing.assert_allclose(target["perp_passive_return"][1], 0., atol=1e-7)

    def test_liquidation_keeps_positive_residual_distinct_from_bankruptcy(self):
        cfg, data = config(), arrays()
        data["ohlcperp"][1, 2] = 82.
        data["ohlcmark"][1, 2] = 80.3
        target, _ = labels.build_perpetual_targets(data, cfg)
        self.assertEqual(target["perp_liquidation"][0, 0, -1], 1.)
        self.assertEqual(target["perp_liquidation"][0, 1, -1], 1.)
        expected_residual = 1 - 5 + .05 * 82. * (1 - .005)
        self.assertGreater(expected_residual, 0.)
        self.assertAlmostEqual(target["perp_passive_return"][0, 0, -1], expected_residual - 1, places=6)
        self.assertAlmostEqual(target["perp_passive_return"][0, 1, -1], expected_residual - 1, places=6)
        self.assertTrue((target["perp_liquidation"][0, :, :-1] == 0).all())

    def test_missing_future_after_liquidation_remains_unobserved(self):
        cfg, data = config(), arrays()
        data["ohlcmark"][1, 2] = 80.3
        data["ohlcperp"][1, 2] = 82.
        data["ohlcmark"][2, 3] = np.nan
        target, masks = labels.build_perpetual_targets(data, cfg)
        self.assertTrue(masks["perp_liquidation"][0, 0].all())
        self.assertFalse(masks["perp_liquidation"][0, 1].any())
        self.assertTrue(np.isnan(target["perp_passive_return"][0, 1]).all())

    def test_absent_funding_event_with_nonzero_rate_masks_label(self):
        cfg, data = config(), arrays()
        data["funding_settlement_rate"][1] = .01
        _, masks = labels.build_perpetual_targets(data, cfg)
        self.assertFalse(masks["perp_liquidation"][0].any())

    def test_chunking_is_exactly_invariant(self):
        cfg, data = config(), arrays(13)
        a, ma = labels.build_perpetual_targets(data, cfg, chunk_rows=2)
        b, mb = labels.build_perpetual_targets(data, cfg, chunk_rows=20)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
            np.testing.assert_array_equal(ma[name], mb[name])

    def test_manifest_snapshot_source_hashes_and_idempotence(self):
        cfg, data = config(), arrays()
        with tempfile.TemporaryDirectory() as tmp:
            cfg["output_dir"] = tmp
            prepared = Path(tmp) / "data" / "prepared"
            prepared.mkdir(parents=True)
            data["passive_label_end_ns"] = data["index_ns"][:, None] + (1 + np.asarray(cfg["targets"]["horizons"])[None]) * data_module.BAR_NS
            data["predictive_label_end_ns"] = data["index_ns"][:, None] + np.asarray(cfg["targets"]["horizons"])[None] * data_module.BAR_NS
            for name, value in data.items():
                np.save(prepared / (name + ".npy"), value)
            manifest = {"complete": True, "rows": 8, "targets": {}, "label_definitions": {}, "groups": {},
                "artifacts": {p.name: data_module._sha(p) for p in prepared.glob("*.npy")}}
            data_module._json(prepared / "manifest.json", manifest)
            before = (prepared / "manifest.json").read_bytes()
            result = labels.prepare_perpetual_targets(cfg)
            self.assertEqual((Path(tmp) / "data/validation/manifest_before_perpetual_targets.json").read_bytes(), before)
            self.assertEqual(result["targets"]["perp_liquidation"]["shape"], [2, 4])
            loaded = data_module.load_data(cfg)
            self.assertIn("perp_liquidation", loaded["targets"])
            self.assertEqual(result, labels.prepare_perpetual_targets(cfg))
            changed = copy.deepcopy(cfg)
            changed["perpetual"]["primary_maintenance_margin_rate"] = .01
            with self.assertRaisesRegex(ValueError, "settings changed"):
                labels.prepare_perpetual_targets(changed)


if __name__ == "__main__":
    unittest.main()
