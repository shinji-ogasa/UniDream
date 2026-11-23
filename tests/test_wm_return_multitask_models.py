"""Matched architecture and supervision contracts of Return-only/joint arms."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from unidream.experiments.wm_return_multitask_models import (
    ARMS, ARM_WEIGHTS, DirectReturnMultitask, WMReturnMultitask,
)


MODEL = {"context": 10, "d_model": 16, "n_heads": 2, "n_layers": 1, "d_ff": 32,
         "encoder_hidden": 16, "n_categoricals": 3, "n_classes": 4, "dropout": .1}


def example(horizons=1):
    generator = torch.Generator().manual_seed(601)
    x = torch.randn(2, 10, 10, generator=generator)
    returns = torch.randn(2, 10, horizons, generator=generator) * .02
    labels = torch.randint(0, 3, returns.shape, generator=generator)
    valid = torch.ones_like(labels, dtype=torch.bool)
    return x, returns, labels, valid


def models(cls, horizons=1):
    result = []
    for arm in ARMS:
        torch.manual_seed(7)
        result.append(cls(10, MODEL, {"return": [.02] * horizons}, arm).eval())
    return result


class ReturnMultitaskModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_all_three_arms_have_exactly_identical_initial_tensors_and_capacity(self):
        for cls in (DirectReturnMultitask, WMReturnMultitask):
            group = models(cls, 2)
            reference = group[0].state_dict()
            for model in group:
                self.assertEqual(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in group[0].parameters()))
                self.assertEqual(set(reference), set(model.state_dict()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(reference[name], value, rtol=0, atol=0)
                self.assertEqual(model.supervised_output_layer.out_features, 8)
                self.assertEqual(set(dict(model.named_buffers())).intersection({"variance_scale", "downside_scale", "advantage_scale", "recovery_scale", "dd_excess_scale"}), set())
                self.assertAlmostEqual(sum(model.loss_weights.values()), 1.)

    def test_public_shapes_expose_only_trained_heads_and_metadata_marks_inactive(self):
        x, _, _, _ = example(2)
        for cls in (DirectReturnMultitask, WMReturnMultitask):
            for model in models(cls, 2):
                prediction = model(x)
                expected = ({"return"} if model.arm == "reg_only" else {"class_probabilities"} if model.arm == "class_only"
                            else {"return", "class_probabilities"})
                self.assertEqual(set(prediction), expected)
                self.assertEqual(set(model.predict_last(x)), expected)
                for name, active in model.prediction_availability.items():
                    self.assertEqual(name in prediction, active)
                    self.assertEqual(model.head_training_status[name], "trained" if active else "inactive_untrained")
                if "return" in prediction:
                    self.assertEqual(prediction["return"].shape, (2, 10, 2))
                    self.assertEqual(model.predict_last(x)["return"].shape, (2, 2))
                if "class_probabilities" in prediction:
                    self.assertEqual(prediction["class_probabilities"].shape, (2, 10, 2, 3))
                    torch.testing.assert_close(prediction["class_probabilities"].sum(-1), torch.ones(2, 10, 2))

    def test_active_head_gradients_are_nonzero_and_single_arm_inactive_rows_are_zero(self):
        x, returns, labels, valid = example(2)
        for cls in (DirectReturnMultitask, WMReturnMultitask):
            for model in models(cls, 2):
                loss, _ = model.loss(x, x[..., :5], {"return": returns}, valid, torch.zeros(2, 10), labels)
                loss.backward()
                gradient = model.supervised_output_layer.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                for name, output_slice in model.task_output_slices.items():
                    selected = gradient[output_slice]
                    if name in model.trained_prediction_tasks:
                        self.assertTrue((selected.abs().sum(1) > 0).all())
                    else:
                        self.assertEqual(int(torch.count_nonzero(selected)), 0)
                self.assertEqual(set(model.supervised_gradient_coverage()), set(model.trained_prediction_tasks))
                self.assertTrue(all(model.supervised_gradient_coverage().values()))

    def test_loss_is_normalized_mse_ce_or_fixed_half_each(self):
        x, returns, labels, valid = example()
        valid[:, -2:] = False
        for cls in (DirectReturnMultitask, WMReturnMultitask):
            group = models(cls)
            predictions, _, _ = group[0]._predict_with_state(x)
            mse = ((predictions["return"][valid] - returns[valid]) / .02).square().mean()
            probability = predictions["class_probabilities"][valid]
            ce = -probability[torch.arange(len(probability)), labels[valid]].log().mean()
            for model in group:
                _, parts = model.loss(x, x[..., :5], {"return": returns}, valid, torch.zeros(2, 10), labels)
                expected = ARM_WEIGHTS[model.arm]["return_mse"] * mse + ARM_WEIGHTS[model.arm]["return_ce"] * ce
                torch.testing.assert_close(parts["supervised"], expected)
                if model.arm != "class_only":
                    torch.testing.assert_close(parts["return_mse"], mse)
                if model.arm != "reg_only":
                    torch.testing.assert_close(parts["return_ce"], ce)

    def test_wm_uses_one_stochastic_encoder_pass_and_identical_auxiliary_objective(self):
        x, returns, labels, valid = example()
        auxiliary = []
        for model in models(WMReturnMultitask):
            with patch.object(model.wm, "encode", wraps=model.wm.encode) as encode:
                total, parts = model.loss(x, x[..., :5], {"return": returns}, valid, torch.zeros(2, 10), labels)
            self.assertEqual(encode.call_count, 1)
            torch.testing.assert_close(total, parts["supervised"] + parts["auxiliary"])
            expected = .3 * parts["reconstruction"] + .2 * parts["kl_dyn"] + .05 * parts["kl_rep"] + .3 * parts["next_return"]
            torch.testing.assert_close(parts["auxiliary"], expected)
            auxiliary.append(parts["auxiliary"])
        for value in auxiliary[1:]:
            torch.testing.assert_close(auxiliary[0], value, rtol=0, atol=0)

    def test_masked_placeholders_and_unused_supervision_do_not_enter_losses_or_gradients(self):
        x, returns, labels, valid = example()
        valid[:, -3:] = False
        changed_return = returns.clone(); changed_return[~valid] = float("nan")
        changed_labels = labels.clone(); changed_labels[~valid] = -999
        for cls in (DirectReturnMultitask, WMReturnMultitask):
            for model in models(cls):
                clean, _ = model.loss(x, x[..., :5], {"return": returns}, valid, torch.zeros(2, 10), labels)
                dirty, _ = model.loss(x, x[..., :5], {"return": changed_return}, valid, torch.zeros(2, 10), changed_labels)
                torch.testing.assert_close(clean, dirty, rtol=0, atol=0)
                dirty.backward()
                self.assertTrue(torch.isfinite(model.supervised_output_layer.weight.grad).all())
                if model.arm == "reg_only":
                    no_labels, _ = model.loss(x, x[..., :5], {"return": returns}, valid, torch.zeros(2, 10))
                    torch.testing.assert_close(clean, no_labels, rtol=0, atol=0)
                if model.arm == "class_only":
                    no_return, _ = model.loss(x, x[..., :5], {}, valid, torch.zeros(2, 10), labels)
                    torch.testing.assert_close(clean, no_return, rtol=0, atol=0)

    def test_future_causality_and_no_six_task_supervision_interface(self):
        x, returns, labels, valid = example()
        altered = x.clone(); altered[:, 6:] = altered[:, 6:] * 100 + 200
        for cls in (DirectReturnMultitask, WMReturnMultitask):
            for model in models(cls):
                before, after = model(x), model(altered)
                for name in before:
                    torch.testing.assert_close(before[name][:, :6], after[name][:, :6], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "only the return task"):
                    model.loss(x, x[..., :5], {"return": returns, "variance": returns}, valid, torch.zeros(2, 10), labels)


if __name__ == "__main__":
    unittest.main()
