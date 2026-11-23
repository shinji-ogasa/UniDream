import pickle
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from unidream.experiments import wm_classification_logistic_optimizer as optimizer


def correlated_data(*, rank_deficient=False, large_mean=False):
    rng = np.random.default_rng(123)
    base = rng.normal(size=(400, 4))
    x = np.column_stack((base, base[:, :2] * 2., base[:, 2:] * .1))
    if not rank_deficient:
        x += rng.normal(0, .002, x.shape)
    if large_mean:
        x += np.array([1000., -300., 50., 80., 150., -70., 40., -40.])
    probability = softmax(np.column_stack((base[:, 0] - .4 * base[:, 1], base[:, 1], -base[:, 0])), axis=1)
    y = np.array([rng.choice(3, p=row) for row in probability])
    return x, y


class PreconditionedLogisticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        cls.limits.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.limits.__exit__(None, None, None)

    def test_full_dimension_with_rank_deficiency_and_no_objective_ridge(self):
        x, y = correlated_data(rank_deficient=True)
        x = np.column_stack((x, np.full(len(x), 17.)))
        prepared = optimizer.prepare_logistic_preconditioner(x)
        self.assertEqual(prepared.eigenvectors.shape, (9, 9))
        self.assertEqual(prepared.rotated.shape, x.shape)
        self.assertEqual(prepared.metadata["retained_dimension"], 9)
        self.assertEqual(prepared.metadata["added_statistical_ridge"], 0.)
        np.testing.assert_allclose(prepared.eigenvectors @ prepared.eigenvectors.T, np.eye(9), atol=1e-13)
        model, result = optimizer.fit_preconditioned_logistic(prepared, y, .1)
        self.assertEqual(model.coef_.shape, (3, 9))
        self.assertLessEqual(result["original_gradient_linf"], 1e-4)
        self.assertTrue(result["original_gradient_all_finite"])

    def test_logits_objective_and_gradient_chain_identity_with_large_means(self):
        x, y = correlated_data(rank_deficient=True, large_mean=True)
        prepared = optimizer.prepare_logistic_preconditioner(x)
        c = .1
        regularization = 1. / (c * len(x))
        scale = np.sqrt(prepared.eigenvalues + 5. * regularization)
        z = prepared.rotated / scale
        packed = np.random.default_rng(7).normal(0, .1, (3, x.shape[1] + 1))
        transformed_loss, transformed_gradient = optimizer.transformed_objective_gradient(packed.ravel(), z, y, scale, regularization)
        w = (packed[:, :-1] / scale) @ prepared.eigenvectors.T
        b = packed[:, -1] - w @ prepared.mean
        loss, gw, gb = optimizer.original_objective_gradient(x, y, w, b, c)
        np.testing.assert_allclose(x @ w.T + b, z @ packed[:, :-1].T + packed[:, -1], rtol=1e-11, atol=1e-11)
        self.assertAlmostEqual(loss, transformed_loss, places=11)
        mapped = (gw - gb[:, None] * prepared.mean) @ prepared.eigenvectors / scale
        transformed_gradient = transformed_gradient.reshape(packed.shape)
        np.testing.assert_allclose(transformed_gradient[:, :-1], mapped, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(transformed_gradient[:, -1], gb, atol=1e-12)
        # Omitting mean*g_bias would materially break the original gradient.
        self.assertGreater(np.abs(gb[:, None] * prepared.mean).max(), 1.)

    def test_same_sklearn_objective_and_probabilities_correlated_and_large_mean(self):
        for deficient, shifted in ((False, False), (True, False), (True, True)):
            with self.subTest(rank_deficient=deficient, large_mean=shifted):
                x, y = correlated_data(rank_deficient=deficient, large_mean=shifted)
                prepared = optimizer.prepare_logistic_preconditioner(x)
                model, result = optimizer.fit_preconditioned_logistic(prepared, y, .3, tol=1e-7)
                # Centering the reference is an exact unpenalized-intercept
                # reparameterization and avoids an ill-conditioned reference.
                reference = LogisticRegression(C=.3, solver="lbfgs", max_iter=10000, tol=1e-11).fit(x - x.mean(0), y)
                original_intercept = reference.intercept_ - reference.coef_ @ x.mean(0)
                reference_loss, _, _ = optimizer.original_objective_gradient(x, y, reference.coef_, original_intercept, .3)
                self.assertLess(abs(result["objective"] - reference_loss), 1e-9)
                self.assertLessEqual(result["original_gradient_linf"], 1e-7)
                np.testing.assert_allclose(model.predict_proba(x), reference.predict_proba(x - x.mean(0)), rtol=2e-5, atol=2e-6)

    def test_candidate_grid_reuses_train_basis_and_same_validation_selector(self):
        x, y = correlated_data(rank_deficient=True)
        priors = np.bincount(y[:300], minlength=3) / 300.
        real_prepare = optimizer.prepare_logistic_preconditioner
        with patch.object(optimizer, "prepare_logistic_preconditioner", wraps=real_prepare) as observed:
            model, history = optimizer.fit_logistic_candidates_preconditioned(
                x[:300], y[:300], x[300:], y[300:], priors,
                {"logistic_Cs": [.01, .1, 1.], "logistic_tol": 1e-4},
            )
        self.assertEqual(observed.call_count, 1)
        self.assertEqual([row["C"] for row in history["candidates"]], [.01, .1, 1.])
        selected = min(history["candidates"], key=lambda row: (-row["balanced_accuracy"], row["log_loss"]))
        self.assertEqual(history["selected"], selected)
        self.assertEqual(model.C, selected["C"])
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.classes_, [0, 1, 2])
        np.testing.assert_allclose(restored.predict_proba(x), model.predict_proba(x), atol=0, rtol=0)

    def test_original_gradient_includes_nonzero_mean_intercept_contribution(self):
        x, y = correlated_data(large_mean=True)
        model, result = optimizer.fit_preconditioned_logistic(optimizer.prepare_logistic_preconditioner(x), y, .1)
        _, gw, gb = optimizer.original_objective_gradient(x, y, model.coef_, model.intercept_, .1)
        self.assertEqual(result["original_gradient_linf"], max(float(np.abs(gw).max()), float(np.abs(gb).max())))
        self.assertLessEqual(result["original_gradient_linf"], 1e-4)

    def test_rejects_nonfinite_bad_regularization_and_material_negative_eigenvalues(self):
        with self.assertRaisesRegex(ValueError, "finite matrix"):
            optimizer.prepare_logistic_preconditioner(np.array([[1., np.nan]]))
        x, y = correlated_data()
        prepared = optimizer.prepare_logistic_preconditioner(x)
        with self.assertRaisesRegex(ValueError, "C must"):
            optimizer.fit_preconditioned_logistic(prepared, y, 0.)
        with patch.object(optimizer, "eigh", return_value=(np.array([-1., 2.]), np.eye(2))):
            with self.assertRaisesRegex(FloatingPointError, "materially negative"):
                optimizer.prepare_logistic_preconditioner(np.arange(12.).reshape(6, 2))
        with patch.object(optimizer, "eigh", return_value=(np.array([np.nan, 2.]), np.eye(2))):
            with self.assertRaisesRegex(FloatingPointError, "nonfinite covariance"):
                optimizer.prepare_logistic_preconditioner(np.arange(12.).reshape(6, 2))


if __name__ == "__main__":
    unittest.main()
