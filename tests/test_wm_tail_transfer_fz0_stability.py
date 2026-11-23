"""Strict FZ0 domains survive saturated affine/neural logits and unit changes."""
import unittest

import numpy as np
import torch

from unidream.experiments import wm_tail_transfer_classical as classical
from unidream.experiments import wm_tail_transfer_evaluation as evaluation
from unidream.experiments import wm_tail_transfer_models as models


SPEC = {"name": "var_es", "kind": "var_es", "shape": [1, 1], "task": "T10", "levels": [.05]}
RAW = np.array([[-1000., -1000.], [-1000., 0.], [-1000., 1000.],
                [0., -1000.], [1000., -1000.], [1000., 1000.], [-15., -1000.], [-3., 2.]])


class FZ0StabilityTests(unittest.TestCase):
    def test_saturated_float32_decode_stays_strictly_negative_in_physical_units(self):
        for physical_scale in (1e-6, .001, .02, 2.5):
            with self.subTest(scale=physical_scale):
                raw = torch.tensor(RAW[:, None, None], dtype=torch.float32, requires_grad=True)
                scale = torch.tensor([[physical_scale]], dtype=torch.float32)
                prediction = models.decode_head(raw, SPEC, scale)
                q, e = prediction[..., 0], prediction[..., 1]
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertTrue((q < 0).all())
                self.assertTrue((e <= q).all())
                self.assertAlmostEqual(float(q[0, 0, 0].detach()) / physical_scale, -1e-5, places=11)
                # Costs and targets use physical returns. The floor belongs in
                # standardized units and scales with the target normalizer.
                target = torch.tensor([-.02, .01, 0., -.1, .2, -.002, -.001, -.08], dtype=torch.float32)[:, None, None]
                value, observed = models.head_loss(raw, prediction, target, torch.ones_like(target, dtype=torch.bool), SPEC, scale)
                self.assertTrue(observed.all())
                self.assertTrue(torch.isfinite(value).all())
                value.mean().backward()
                self.assertTrue(torch.isfinite(raw.grad).all())

    def test_numpy_and_torch_links_and_standardized_scores_match(self):
        actual = np.array([-.02, .01, 0., -.1, .2, -.002, -.001, -.08])
        score, q, es = classical.fz0_numpy(RAW, actual, .05)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue((q < 0).all())
        self.assertTrue((es <= q).all())
        self.assertEqual(q[0], -1e-5)
        for physical_scale in (1e-6, .02, 2.5):
            raw = torch.tensor(RAW[:, None, None], dtype=torch.float64)
            scale = torch.tensor([[physical_scale]], dtype=torch.float64)
            prediction = models.decode_head(raw, SPEC, scale)
            np.testing.assert_allclose(prediction[..., 0].numpy().ravel(), q * physical_scale, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(prediction[..., 1].numpy().ravel(), es * physical_scale, rtol=1e-12, atol=1e-12)
            target = torch.tensor(actual * physical_scale, dtype=torch.float64)[:, None, None]
            loss, _ = models.head_loss(raw, prediction, target, torch.ones_like(target, dtype=torch.bool), SPEC, scale)
            np.testing.assert_allclose(loss.numpy().ravel(), score, rtol=1e-11, atol=1e-9)

    def test_exact_gradient_remains_finite_and_matches_differences_at_extremes(self):
        actual = np.array([-.02, .01, 0., -.1, .2, -.002, -.001, -.08])
        gradient, surrogate = classical.fz0_gradient_hessian(RAW, actual, .05)
        self.assertTrue(np.isfinite(gradient).all())
        self.assertTrue(np.isfinite(surrogate).all())
        self.assertTrue((surrogate > 0).all())
        for axis in range(2):
            plus, minus = RAW.copy(), RAW.copy()
            plus[:, axis] += 1e-5
            minus[:, axis] -= 1e-5
            difference = (classical.fz0_numpy(plus, actual, .05)[0] - classical.fz0_numpy(minus, actual, .05)[0]) / 2e-5
            np.testing.assert_allclose(gradient[:, axis], difference, rtol=3e-5, atol=1e-7)
        # A smooth positive floor has zero derivative in the saturated region.
        np.testing.assert_array_equal(gradient[0], [0., 0.])

    def test_evaluator_accepts_extremes_but_still_rejects_negative_zero(self):
        actual = np.array([-.02, .01, 0., -.1, .2, -.002, -.001, -.08], dtype=np.float32)
        raw = torch.tensor(RAW[:, None, None], dtype=torch.float32)
        prediction = models.decode_head(raw, SPEC, torch.tensor([[.02]])).numpy()
        target = actual[:, None, None]
        data = {"targets": {"var_es": target}, "masks": {"var_es": np.ones_like(target, bool)},
                "index_ns": np.arange(len(target), dtype=np.int64)}
        cfg = {"targets": {"horizons": [16], "es_alphas": [.05]}}
        metrics = evaluation.evaluate_predictions(data, np.arange(len(target)), {"var_es": prediction}, [SPEC], cfg,
                                                   train_ids=np.arange(4))
        self.assertTrue(np.isfinite(metrics.native_loss).all())
        bad = prediction.copy()
        bad[0, 0, 0, 0] = -0.
        with self.assertRaisesRegex(ValueError, "negative ordered"):
            evaluation.evaluate_predictions(data, np.arange(len(target)), {"var_es": bad}, [SPEC], cfg,
                                              train_ids=np.arange(4))


if __name__ == "__main__":
    unittest.main()
