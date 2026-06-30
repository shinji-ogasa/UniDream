import unittest

import numpy as np

from unidream.eval.backtest import Backtest


class BacktestFinalExcessTest(unittest.TestCase):
    def test_alpha_excess_uses_final_equity_difference(self) -> None:
        returns = np.asarray([0.10, -0.02, 0.03], dtype=np.float64)
        positions = np.asarray([0.5, 0.5, 0.5], dtype=np.float64)
        benchmark_positions = np.ones_like(positions)

        metrics = Backtest(
            returns,
            positions,
            spread_bps=0.0,
            fee_rate=0.0,
            slippage_bps=0.0,
            interval="1d",
            benchmark_positions=benchmark_positions,
        ).run()

        expected_final_excess = float(metrics.total_return - metrics.benchmark_total_return)
        expected_annual_excess = float(metrics.annual_return - metrics.benchmark_annual_return)

        self.assertAlmostEqual(metrics.final_excess, expected_final_excess)
        self.assertAlmostEqual(metrics.alpha_excess, expected_final_excess)
        self.assertAlmostEqual(metrics.annual_alpha_excess, expected_annual_excess)
        self.assertNotAlmostEqual(metrics.alpha_excess, metrics.annual_alpha_excess)

    def test_to_dict_exposes_final_and_annual_alpha(self) -> None:
        returns = np.asarray([0.01, 0.02], dtype=np.float64)
        metrics = Backtest(
            returns,
            np.ones_like(returns),
            spread_bps=0.0,
            fee_rate=0.0,
            slippage_bps=0.0,
            interval="1d",
            benchmark_positions=np.zeros_like(returns),
        ).run()
        payload = metrics.to_dict()

        self.assertIn("final_excess", payload)
        self.assertIn("annual_alpha_excess", payload)
        self.assertEqual(payload["alpha_excess"], payload["final_excess"])


if __name__ == "__main__":
    unittest.main()
