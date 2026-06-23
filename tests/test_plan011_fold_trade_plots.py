from __future__ import annotations

import unittest

import numpy as np

from unidream.cli.plot_plan011_fold_trades import _parse_folds, active_blocks, trade_indices


class Plan011FoldTradePlotTest(unittest.TestCase):
    def test_parse_fold_ranges_preserves_order_and_uniqueness(self) -> None:
        self.assertEqual(_parse_folds("0-2,2,4", (9,)), (0, 1, 2, 4))

    def test_trade_indices_ignore_initial_position_and_small_noise(self) -> None:
        positions = np.asarray([1.0, 1.00001, 1.02, 1.02, 0.98], dtype=np.float64)
        np.testing.assert_array_equal(trade_indices(positions, trade_eps=1e-4), np.asarray([2, 4]))

    def test_active_blocks_use_benchmark_overlay(self) -> None:
        positions = np.asarray([1.0, 1.06, 1.07, 1.0, 0.94, 0.93, 1.0], dtype=np.float64)
        self.assertEqual(active_blocks(positions, benchmark=1.0, active_eps=0.05), [(1, 2), (4, 5)])


if __name__ == "__main__":
    unittest.main()
