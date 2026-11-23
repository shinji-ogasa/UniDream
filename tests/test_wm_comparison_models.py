import unittest
from unittest.mock import patch

import torch

from unidream.experiments.wm_comparison_models import DirectForecastTransformer


SMALL_CONFIG = {
    "context": 12, "d_model": 16, "n_heads": 2, "n_layers": 2,
    "d_ff": 32, "encoder_hidden": 16, "dropout": .2,
}
SCALES = {"return": [.01, .02, .04], "variance": [.0001, .0002, .0004]}


class DirectForecastTransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_model(self, scales=None):
        torch.manual_seed(42)
        return DirectForecastTransformer(7, SMALL_CONFIG, scales or SCALES)

    def batch(self):
        generator = torch.Generator().manual_seed(123)
        x = torch.randn((2, 12, 7), generator=generator)
        targets = {
            "return": torch.randn((2, 12, 3), generator=generator) * .01,
            "variance": torch.rand((2, 12, 3), generator=generator) * .0004,
        }
        valid = torch.ones((2, 12, 3), dtype=torch.bool)
        return x, targets, valid

    def test_shapes_positive_variance_and_explicit_scale(self):
        model = self.make_model().eval()
        changed_scales = {"return": [.02, .08, .24], "variance": [.0003, .001, .0028]}
        scaled_model = self.make_model(scales=changed_scales).eval()
        x, _, _ = self.batch()
        prediction, scaled = model(x), scaled_model(x)
        self.assertEqual(set(prediction), {"return", "variance"})
        self.assertEqual(prediction["return"].shape, (2, 12, 3))
        self.assertEqual(prediction["variance"].shape, (2, 12, 3))
        self.assertTrue((prediction["variance"] > 0).all())
        torch.testing.assert_close(scaled["return"], prediction["return"] * torch.tensor([2., 4., 6.]))
        torch.testing.assert_close(scaled["variance"], prediction["variance"] * torch.tensor([3., 5., 7.]))
        for key, value in model.predict_last(x).items():
            self.assertEqual(value.shape, (2, 3))
            torch.testing.assert_close(value, prediction[key][:, -1])

    def test_future_token_mutations_leave_earlier_predictions_unchanged(self):
        model = self.make_model().eval()
        x, _, _ = self.batch()
        changed = x.clone()
        changed[:, 7:] = changed[:, 7:] * 100.0 + 30.0
        before, after = model(x), model(changed)
        for key in before:
            torch.testing.assert_close(before[key][:, :7], after[key][:, :7], atol=0, rtol=0)

    def test_masked_nan_targets_have_finite_loss_and_gradients(self):
        model = self.make_model().train()
        x, targets, valid = self.batch()
        valid[:, -4:, :] = False
        valid[0, 2, 2] = False
        for value in targets.values():
            value[~valid] = float("nan")
        loss, parts = model.loss(x, None, targets, valid, None)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"return", "variance"})
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue((model.head[-1].weight.grad.abs().sum(1) > 0).all())
        self.assertGreater(float(model.position_embedding.grad.abs().sum()), 0.0)

    def test_loss_exactly_matches_valid_normalized_mse_and_variance_qlike(self):
        model = self.make_model().eval()
        x, targets, valid = self.batch()
        valid[:, -2:, :] = False
        predictions = model(x)
        expected_return = (((predictions["return"] - targets["return"]) / model.return_scale).square())[valid].mean()
        ratio = targets["variance"].clamp_min(model.variance_floor) / predictions["variance"].clamp_min(model.variance_floor)
        expected_variance = (ratio - torch.log(ratio) - 1.)[valid].mean()
        loss, parts = model.loss(x, torch.full((1,), float("nan")), targets, valid, torch.full((1,), float("nan")))
        torch.testing.assert_close(parts["return"], expected_return)
        torch.testing.assert_close(parts["variance"], expected_variance)
        torch.testing.assert_close(loss, expected_return + .2 * expected_variance)

    def test_eval_is_deterministic_and_training_uses_dropout(self):
        model = self.make_model()
        x, _, _ = self.batch()
        model.eval()
        first = model.predict_last(x)
        torch.manual_seed(999)
        second = model.predict_last(x)
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
        model.train()
        first = model(x)["return"]
        second = model(x)["return"]
        self.assertFalse(torch.equal(first, second))

    def test_no_raw_feature_shortcut_after_continuous_encoder(self):
        model = self.make_model().eval()
        x, _, _ = self.batch()
        encoded = model.encoder(x)
        with patch.object(model.encoder, "forward", return_value=encoded):
            first, second = model(x), model(x * 100.0 + 31.)
        for key in first:
            torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_all_invalid_labels_produce_zero_finite_loss(self):
        model = self.make_model().train()
        x, targets, valid = self.batch()
        valid[:] = False
        for value in targets.values():
            value[:] = float("nan")
        loss, _ = model.loss(x, None, targets, valid, None)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None))

    def test_rejects_bad_scale_and_wrong_context(self):
        with self.assertRaisesRegex(ValueError, "three finite positive"):
            self.make_model(scales={"return": [1., 0., 1.], "variance": [1., 1., 1.]})
        with self.assertRaisesRegex(ValueError, "context"):
            self.make_model()(torch.zeros(2, 13, 7))


if __name__ == "__main__":
    unittest.main()
