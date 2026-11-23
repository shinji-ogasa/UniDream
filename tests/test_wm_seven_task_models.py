"""Scientific contracts of the isolated seven-task continuous/WM models."""
import unittest

import torch
from torch.nn import functional as F

from unidream.experiments.wm_seven_task_models import (
    TASKS, SUPERVISED_WEIGHTS, DirectSevenTaskTransformer, SevenTaskWM,
)


MODEL = {"n_categoricals": 3, "n_classes": 4, "d_model": 16,
         "n_heads": 2, "n_layers": 1, "d_ff": 32,
         "encoder_hidden": 16, "context": 12, "dropout": .2}


def scales(horizons=1):
    return {task: [.001 if task == "variance" else .02] * horizons
            for task in TASKS if task != "crash"}


def inputs(horizons=1, length=12):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn((2, length, 10), generator=generator)
    shape = (2, length, horizons)
    targets = {task: torch.randn(shape, generator=generator) * .02 for task in TASKS}
    for task in ("variance", "downside", "dd_excess"):
        targets[task] = torch.rand(shape, generator=generator) * (.002 if task == "variance" else .03)
    targets["crash"] = (torch.rand(shape, generator=generator) > .8).float()
    valid = torch.ones(shape, dtype=torch.bool)
    rewards = torch.randn((2, length), generator=generator) * .005
    return x, targets, valid, rewards


class SevenTaskModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def models(self, horizons=1):
        for kind in (DirectSevenTaskTransformer, SevenTaskWM):
            torch.manual_seed(42)
            yield kind(10, MODEL, scales(horizons))

    def test_one_and_four_horizon_output_shapes_and_constraints(self):
        for horizon_count in (1, 4):
            x, _, _, _ = inputs(horizon_count)
            for model in self.models(horizon_count):
                model.eval()
                predictions = model(x)
                self.assertEqual(tuple(predictions), TASKS)
                for task in TASKS:
                    self.assertEqual(predictions[task].shape, (2, 12, horizon_count))
                    self.assertEqual(model.predict_last(x)[task].shape, (2, horizon_count))
                for task in ("variance", "downside", "dd_excess"):
                    self.assertTrue((predictions[task] > 0).all())
                self.assertTrue(((predictions["crash"] >= 0) & (predictions["crash"] <= 1)).all())
                self.assertEqual(model.latent_last(x).shape[0], 2)

    def test_eval_is_deterministic_and_future_tokens_do_not_change_past(self):
        x, _, _, _ = inputs(2)
        changed = x.clone()
        changed[:, 7:] = changed[:, 7:] * 100 + 100
        for model in self.models(2):
            model.eval()
            first = model(x)
            torch.manual_seed(98765)
            repeated = model(x)
            perturbed = model(changed)
            for task in TASKS:
                torch.testing.assert_close(first[task], repeated[task], rtol=0, atol=0)
                torch.testing.assert_close(first[task][:, :7], perturbed[task][:, :7], rtol=0, atol=0)

    def test_each_task_and_horizon_gets_nonzero_finite_gradient(self):
        x, targets, valid, rewards = inputs(4)
        for model in self.models(4):
            model.train()
            self.assertFalse(any(model.supervised_gradient_coverage().values()))
            loss, parts = model.loss(x, x[..., :5], targets, valid, rewards)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(set(model.supervised_gradient_coverage()), set(TASKS))
            self.assertTrue(all(model.supervised_gradient_coverage().values()))
            self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_masked_nan_labels_do_not_affect_loss_or_gradients(self):
        x, targets, valid, rewards = inputs(2)
        valid[:, -3:] = False
        changed = {task: value.clone() for task, value in targets.items()}
        for value in changed.values():
            value[~valid] = float("nan")
        for model in self.models(2):
            model.eval()
            reference, _ = model.loss(x, x[..., :5], targets, valid, rewards)
            reference.backward()
            gradients = {key: value.grad.clone() for key, value in model.named_parameters()
                         if value.grad is not None}
            model.zero_grad(set_to_none=True)
            actual, _ = model.loss(x, x[..., :5], changed, valid, rewards)
            actual.backward()
            torch.testing.assert_close(reference, actual, rtol=0, atol=0)
            for key, value in model.named_parameters():
                if key in gradients:
                    torch.testing.assert_close(value.grad, gradients[key], rtol=0, atol=0)

    def test_supervised_losses_match_independently_computed_objectives(self):
        x, targets, valid, rewards = inputs(2)
        valid[:, -2:] = False
        for model in self.models(2):
            model.eval()
            predictions = model(x)
            total, parts = model.loss(x, x[..., :5], targets, valid, rewards)
            expected = {}
            for task in TASKS:
                if task == "variance":
                    ratio = targets[task][valid].clamp_min(1e-12) / predictions[task][valid]
                    expected[task] = (ratio - ratio.log() - 1).mean()
                elif task == "crash":
                    expected[task] = F.binary_cross_entropy(predictions[task][valid], targets[task][valid])
                else:
                    normalized = (predictions[task] - targets[task]) / getattr(model, task + "_scale")
                    expected[task] = normalized[valid].square().mean()
                torch.testing.assert_close(parts[task], expected[task])
            expected_supervised = sum(SUPERVISED_WEIGHTS[key] * value for key, value in expected.items())
            torch.testing.assert_close(parts["supervised"], expected_supervised)
            if isinstance(model, DirectSevenTaskTransformer):
                torch.testing.assert_close(total, expected_supervised)
            else:
                expected_aux = (.3 * parts["reconstruction"] + .2 * parts["kl_dyn"]
                                + .05 * parts["kl_rep"] + .3 * parts["next_return"])
                torch.testing.assert_close(total, expected_supervised + expected_aux)

    def test_wholly_invalid_forecast_horizon_contributes_no_gradient(self):
        x, targets, valid, rewards = inputs(2)
        valid[..., 1] = False
        for task in TASKS:
            targets[task][..., 1] = float("nan")
        for model in self.models(2):
            model.eval()
            loss, _ = model.loss(x, x[..., :5], targets, valid, rewards)
            loss.backward()
            gradient = model.supervised_output_layer.weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient[0::2].abs().sum(dim=1) > 0).all())
            self.assertEqual(float(gradient[1::2].abs().sum()), 0.)

    def test_empty_supervision_and_missing_reward_are_finite(self):
        x, targets, valid, rewards = inputs(1, length=1)
        valid[:] = False
        rewards[:] = float("nan")
        for value in targets.values():
            value[:] = float("nan")
        for model in self.models():
            model.eval()
            loss, parts = model.loss(x, x[..., :5], targets, valid, rewards)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(float(parts["supervised"]), 0.)
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            if isinstance(model, SevenTaskWM):
                self.assertEqual(float(parts["kl_dyn"]), 0.)
                self.assertEqual(float(parts["kl_rep"]), 0.)
                self.assertEqual(float(parts["next_return"]), 0.)

    def test_auxiliary_components_have_gradients_when_free_bits_exceeded(self):
        model = SevenTaskWM(10, MODEL, scales()).train()
        with torch.no_grad():
            model.wm.prior_head.bias.fill_(-10.)
            model.wm.prior_head.bias[::MODEL["n_classes"]] = 10.
        x, targets, valid, rewards = inputs()
        loss, parts = model.loss(x, x[..., :5], targets, valid, rewards)
        loss.backward()
        self.assertGreater(float(parts["kl_dyn"]), 1.)
        for component in (model.wm.encoder.net[0], model.wm.blocks[0].attn.out_proj,
                          model.wm.prior_head, model.wm.decoder.net[-1], model.wm.reward_head):
            gradient = component.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_crash_loss_uses_stable_logits_for_extreme_predictions(self):
        x, targets, valid, rewards = inputs()
        for model in self.models():
            with torch.no_grad():
                model.supervised_output_layer.weight[TASKS.index("crash")].zero_()
                model.supervised_output_layer.bias[TASKS.index("crash")] = 1000.
            loss, parts = model.loss(x, x[..., :5], targets, valid, rewards)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(parts["crash"]))
            self.assertGreater(float(parts["crash"]), 100.)

    def test_recovery_domain_projection_does_not_clip_signed_return_or_advantage(self):
        x, _, _, _ = inputs(2)
        for model in self.models(2):
            model.eval()
            output = model.supervised_output_layer
            with torch.no_grad():
                output.weight.zero_()
                output.bias[TASKS.index("recovery") * 2:TASKS.index("recovery") * 2 + 2] = torch.tensor([-1000., 1000.])
                output.bias[TASKS.index("return") * 2:TASKS.index("return") * 2 + 2] = 1000.
                output.bias[TASKS.index("advantage") * 2:TASKS.index("advantage") * 2 + 2] = -1000.
            prediction = model(x)
            torch.testing.assert_close(prediction["recovery"][..., 0], -torch.ones(2, 12))
            torch.testing.assert_close(prediction["recovery"][..., 1], torch.ones(2, 12))
            self.assertTrue((prediction["return"] > 1.).all())
            self.assertTrue((prediction["advantage"] < -1.).all())

    def test_invalid_scale_shapes_and_masks_fail_before_training(self):
        for kind in (DirectSevenTaskTransformer, SevenTaskWM):
            wrong_scales = scales()
            wrong_scales["advantage"] = [.02, .03]
            with self.assertRaisesRegex(ValueError, "advantage scales"):
                kind(10, MODEL, wrong_scales)
            wrong_scales = scales()
            wrong_scales["return"] = [float("nan")]
            with self.assertRaisesRegex(ValueError, "return scales"):
                kind(10, MODEL, wrong_scales)
            model = kind(10, MODEL, scales())
            x, targets, valid, rewards = inputs()
            with self.assertRaisesRegex(ValueError, "valid must"):
                model.loss(x, x[..., :5], targets, valid.float(), rewards)
            with self.assertRaisesRegex(ValueError, "sequence length"):
                model(x[:, :0])


if __name__ == "__main__":
    unittest.main()
