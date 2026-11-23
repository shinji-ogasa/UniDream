"""Scientific contracts for the isolated technical-feature WM comparison."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from unidream.experiments.wm_ablation_features import (
    BASE_COLUMNS,
    TA_COLUMNS,
    feature_columns,
    transform_windows,
)
from unidream.experiments.wm_ablation_model import FeatureAblationWM


SMALL_MODEL = {
    "n_categoricals": 3,
    "n_classes": 4,
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 1,
    "d_ff": 32,
    "encoder_hidden": 16,
    "context": 12,
    "dropout": 0.2,
}
SCALES = {"return": [.01, .02, .04], "variance": [.001, .002, .004], "downside": [.01, .02, .04]}


def example_inputs(input_dim=10):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn((2, 12, input_dim), generator=generator)
    targets = {
        "return": torch.randn((2, 12, 3), generator=generator) * .01,
        "variance": torch.rand((2, 12, 3), generator=generator) * .005 + .0001,
        "downside": torch.rand((2, 12, 3), generator=generator) * .03,
        "crash": (torch.rand((2, 12, 3), generator=generator) > .7).float(),
    }
    valid = torch.ones((2, 12, 3), dtype=torch.bool)
    next_return = torch.randn((2, 12), generator=generator) * .005
    return x, targets, valid, next_return


class WMAblationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def make_model(self, input_dim=10, seed=42):
        torch.manual_seed(seed)
        return FeatureAblationWM(input_dim, SMALL_MODEL, SCALES)

    def test_eval_is_deterministic_independent_of_rng(self):
        model = self.make_model().eval()
        x, _, _, _ = example_inputs()
        first = model.predict_last(x)
        torch.manual_seed(987654)
        second = model.predict_last(x)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_future_tokens_cannot_change_past_forecasts(self):
        model = self.make_model().eval()
        x, _, _, _ = example_inputs()
        changed = x.clone()
        changed[:, 7:] = changed[:, 7:] * 100.0 + 100.0
        first, _ = model(x)
        second, _ = model(changed)
        for name in first:
            torch.testing.assert_close(first[name][:, :7], second[name][:, :7], rtol=0, atol=0)

    def test_forecast_heads_have_no_raw_feature_shortcut(self):
        model = self.make_model().eval()
        x, _, _, _ = example_inputs()
        fixed_z, fixed_posterior = model.wm.encode(x)
        with patch.object(model.wm, "encode", return_value=(fixed_z, fixed_posterior)):
            first, _ = model(x)
            second, _ = model(x * 100.0 + 73.0)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_all_arms_share_architecture_and_seeded_initialization(self):
        columns = feature_columns("D", ("funding", "oi"))
        models = [self.make_model(input_dim=len(columns)) for _ in "ABCD"]
        reference = models[0].state_dict()
        for model in models[1:]:
            self.assertEqual(set(reference), set(model.state_dict()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(reference[name], value, rtol=0, atol=0)
        self.assertEqual(models[0].head[0].in_features, models[0].wm.z_dim + models[0].wm.d_model)
        self.assertEqual(models[0].wm.decoder.net[-1].out_features, len(BASE_COLUMNS))

    def test_ohlcv_encoding_retains_the_same_raw_information_as_ta(self):
        rng = np.random.default_rng(123)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, .003, 12)))
        open_ = np.r_[100.0, close[:-1]]
        raw = np.stack((open_, np.maximum(open_, close) * 1.01,
                        np.minimum(open_, close) * .99, close,
                        rng.uniform(1., 20., 12)), axis=-1)[None]
        base = transform_windows(raw, "A")
        # A contains the entire candle path up to an irrelevant price scale;
        # B's technical columns are deterministic functions of that path.
        reconstructed = np.empty_like(raw)
        previous_close = np.ones((1, 1)) * raw[0, 0, 0]
        for step in range(12):
            reconstructed[:, step, :4] = np.exp(base[:, step, :4]) * previous_close
            reconstructed[:, step, 4] = np.expm1(base[:, step, 4])
            previous_close = reconstructed[:, step, 3:4]
        np.testing.assert_allclose(reconstructed, raw, rtol=1e-6)
        original_ta = transform_windows(raw, "B")[:, :, 5:]
        reconstructed_ta = transform_windows(reconstructed, "B")[:, :, 5:]
        np.testing.assert_allclose(original_ta, reconstructed_ta, rtol=1e-4, atol=1e-5)
        self.assertEqual(original_ta.shape[-1], len(TA_COLUMNS))

    def test_qlike_uses_variance_predictions_and_target_mask(self):
        model = self.make_model().eval()
        x, targets, valid, next_return = example_inputs()
        valid[:, -2:, :] = False
        predictions, _ = model(x)
        _, parts = model.loss(x, x[:, :, :5], targets, valid, next_return)
        ratio = targets["variance"] / predictions["variance"]
        expected = (ratio - torch.log(ratio) - 1)[valid].mean()
        torch.testing.assert_close(parts["variance"], expected)
        self.assertTrue(torch.all(predictions["variance"] > 0))
        self.assertTrue(torch.all(predictions["downside"] >= 0))
        self.assertTrue(torch.all((predictions["crash"] >= 0) & (predictions["crash"] <= 1)))

    def test_each_horizon_of_each_forecast_head_receives_finite_gradient(self):
        model = self.make_model().train()
        x, targets, valid, next_return = example_inputs()
        loss, _ = model.loss(x, x[:, :, :5], targets, valid, next_return)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        output_gradient = model.head[-1].weight.grad
        self.assertTrue(torch.isfinite(output_gradient).all())
        # Output rows 0:3 return, 3:6 variance, 6:9 downside, 9:12 crash.
        self.assertTrue(torch.all(output_gradient.abs().sum(dim=1) > 0))
        for component in (model.wm.encoder.net[0], model.wm.blocks[0].attn.out_proj,
                          model.wm.decoder.net[-1], model.wm.reward_head):
            gradient = component.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_masked_finite_placeholders_do_not_affect_forecast_losses(self):
        model = self.make_model().eval()
        x, targets, valid, next_return = example_inputs()
        valid[:, -3:, :] = False
        changed = {name: value.clone() for name, value in targets.items()}
        for name, value in changed.items():
            value[~valid] = 1e5 if name != "crash" else 1.0
        first, first_parts = model.loss(x, x[:, :, :5], targets, valid, next_return)
        second, second_parts = model.loss(x, x[:, :, :5], changed, valid, next_return)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name in first_parts:
            torch.testing.assert_close(first_parts[name], second_parts[name], rtol=0, atol=0)

    def test_prior_dynamics_gradient_when_kl_exceeds_free_bits(self):
        model = self.make_model().train()
        with torch.no_grad():
            # Force nontrivial KL so the intentional free-bits floor cannot
            # hide a disconnected prior-head training path.
            model.wm.prior_head.bias.fill_(-10.0)
            model.wm.prior_head.bias[::SMALL_MODEL["n_classes"]] = 10.0
        x, targets, valid, next_return = example_inputs()
        loss, parts = model.loss(x, x[:, :, :5], targets, valid, next_return)
        loss.backward()
        self.assertGreater(float(parts["kl_dyn"]), 1.0)
        gradient = model.wm.prior_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
