"""History support, gaps, analytic recursions, and units for classical models."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from unidream.experiments.wm_comparison_classical import (
    GARCHFit, _objective_and_gradient, ewma_variance_forecast,
    fit_gap_reset_garch, garch_conditional_variance,
    garch_variance_forecast, log_returns_15m, score_forecasts,
)


def _fit(omega=1e-8, alpha=.08, beta=.9):
    return GARCHFit(omega=omega, alpha=alpha, beta=beta,
                    initial_variance=1e-6, return_scale=100., success=True,
                    optimizer_success=True, status=0, message="fixture", nobs=100,
                    n_segments=1, iterations=1, function_evaluations=1,
                    optimizer_objective_scaled=0., mean_negative_log_likelihood=0.,
                    gradient_norm=0., constraints_valid=True)


class TestWMComparisonClassical(unittest.TestCase):
    def test_returns_keep_gaps_and_do_not_bridge_them(self):
        close = np.array([100., 110., np.nan, 130., 143., 0., 160.])
        expected = [np.nan, np.log(1.1), np.nan, np.nan, np.log(1.1), np.nan, np.nan]
        np.testing.assert_allclose(log_returns_15m(close), expected, atol=1e-15, equal_nan=True)

    def test_forecasts_ignore_future_and_history_before_window(self):
        close = np.exp(np.r_[0., np.cumsum(np.random.default_rng(11).normal(0, .001, 399))])
        decisions = np.array([256, 300])
        altered_future = close.copy()
        altered_future[300:] *= 10
        for forecaster in [
            lambda values, indices: ewma_variance_forecast(values, indices, initial_variance=1e-6, decay=.97),
            lambda values, indices: garch_variance_forecast(values, indices, _fit(), initial_variance=1e-6),
        ]:
            np.testing.assert_array_equal(forecaster(close, decisions), forecaster(altered_future, decisions))
            altered_prefix = close.copy()
            altered_prefix[:44] *= 10
            np.testing.assert_array_equal(forecaster(close, [300]), forecaster(altered_prefix, [300]))

    def test_ewma_uses_exactly_context_minus_one_returns(self):
        returns = np.array([.1, -.2, .3])
        close = np.exp(np.r_[0., np.cumsum(returns)])
        state = .04
        for value in returns:
            state = .97 * state + .03 * value ** 2
        actual = ewma_variance_forecast(close, [4], initial_variance=.04, decay=.97, context=4, horizon=16)
        np.testing.assert_allclose(actual, [16 * state], rtol=1e-14)
        close[1] = np.nan
        self.assertTrue(np.isnan(ewma_variance_forecast(close, [4], initial_variance=.04, decay=.97, context=4)[0]))

    def test_garch_gap_resets_likelihood_state(self):
        returns = np.array([.1, .2, np.nan, -.3, .4])
        actual = garch_conditional_variance(returns, omega=.01, alpha=.1, beta=.8, initial_variance=.1)
        np.testing.assert_allclose(actual, [.1, .091, np.nan, .1, .099], atol=1e-15, equal_nan=True)

    def test_garch_multistep_forecast_matches_analytic_recursion(self):
        returns = np.array([.1, -.2, .3])
        close = np.exp(np.r_[0., np.cumsum(returns)])
        fit = _fit(omega=.01, alpha=.1, beta=.8)
        state = .04
        for value in returns:
            state = fit.omega + fit.alpha * value ** 2 + fit.beta * state
        unconditional = fit.omega / (1 - fit.alpha - fit.beta)
        expected = 16 * unconditional + (state - unconditional) * (1 - (fit.alpha + fit.beta) ** 16) / (1 - fit.alpha - fit.beta)
        actual = garch_variance_forecast(close, [4], fit, initial_variance=.04, context=4, horizon=16)
        np.testing.assert_allclose(actual, [expected], rtol=1e-14)
        with self.assertRaisesRegex(ValueError, "unsuccessful"):
            garch_variance_forecast(close, [4], replace(fit, success=False), initial_variance=.04, context=4)

    def test_gap_likelihood_analytic_gradient(self):
        rng = np.random.default_rng(6)
        segments = [rng.normal(0, .2, 80), rng.normal(0, .2, 50)]
        theta = np.array([np.log(.003), -.2, 2.])
        _, analytic = _objective_and_gradient(theta, segments, .04)
        numerical = approx_derivative(lambda point: _objective_and_gradient(point, segments, .04)[0], theta).ravel()
        np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)

    def test_fitted_units_are_original_log_return_variances(self):
        rng = np.random.default_rng(13)
        observations = np.empty(800)
        state = 1e-6
        for i in range(len(observations)):
            observations[i] = np.sqrt(state) * rng.normal()
            state = 1e-8 + .08 * observations[i] ** 2 + .90 * state
        observations[300:304] = np.nan
        initial = float(np.nanmean(observations ** 2))
        percent = fit_gap_reset_garch(observations, initial_variance=initial, return_scale=100.)
        ordinary = fit_gap_reset_garch(observations, initial_variance=initial, return_scale=1.)
        self.assertTrue(percent.success, percent.message)
        self.assertTrue(ordinary.success, ordinary.message)
        self.assertEqual(percent.nobs, 796)
        self.assertEqual(percent.n_segments, 2)
        np.testing.assert_allclose([percent.omega, percent.alpha, percent.beta], [ordinary.omega, ordinary.alpha, ordinary.beta], rtol=3e-5, atol=1e-10)
        self.assertAlmostEqual(percent.mean_negative_log_likelihood, ordinary.mean_negative_log_likelihood, places=9)

    def test_optional_score_heads_and_zero_return_reference(self):
        returns = score_forecasts(actual_return=[.1, -.2], pred_return=[0., 0.])
        self.assertEqual(returns["return_oos_r2"], 0)
        self.assertIsNone(returns["return_spearman_ic"])
        self.assertIsNone(returns["variance_qlike"])
        variance = score_forecasts(actual_variance=[1., 4.], pred_variance=[1., 4.])
        self.assertEqual(variance["variance_qlike"], 0)
        self.assertEqual(variance["n_return"], 0)


if __name__ == "__main__":
    unittest.main()
