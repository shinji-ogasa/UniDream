"""Return-CE replacement preserves six losses and the WM forward contract."""
import unittest
from unittest.mock import patch

import torch

from unidream.experiments.wm_return_formulation_models import (
    OTHER_TASKS, DirectReturnClassifier, WMReturnClassifier,
)
from unidream.experiments.wm_seven_task_models import (
    TASKS, DirectSevenTaskTransformer, SevenTaskWM,
)


MODEL = {"n_categoricals": 3, "n_classes": 4, "d_model": 16,
         "n_heads": 2, "n_layers": 1, "d_ff": 32,
         "encoder_hidden": 16, "context": 12, "dropout": .2}


def scales(horizons=1):
    return {task: [.001 if task == "variance" else .02] * horizons
            for task in TASKS if task != "crash"}


def inputs(horizons=1):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2, 12, 10, generator=generator)
    shape = (2, 12, horizons)
    targets = {task: torch.randn(shape, generator=generator) * .02 for task in TASKS}
    for task in ("variance", "downside", "dd_excess"):
        targets[task] = targets[task].abs() + .0001
    targets["crash"] = (targets["crash"] > 0).float()
    valid = torch.ones(shape, dtype=torch.bool)
    reward = torch.randn(2, 12, generator=generator) * .005
    labels = torch.randint(0, 3, shape, generator=generator)
    return x, targets, valid, reward, labels


class ReturnFormulationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def models(self, horizons=1):
        for cls in (DirectReturnClassifier, WMReturnClassifier):
            torch.manual_seed(7)
            yield cls(10, MODEL, scales(horizons))

    def test_output_shapes_domains_and_untrained_return_is_not_exposed(self):
        for horizon_count in (1, 2):
            x, _, _, _, _ = inputs(horizon_count)
            for model in self.models(horizon_count):
                model.eval()
                output = model(x)
                self.assertEqual(set(output), {*OTHER_TASKS, "class_probabilities"})
                self.assertEqual(output["class_probabilities"].shape, (2, 12, horizon_count, 3))
                shape = (2, 3) if horizon_count == 1 else (2, horizon_count, 3)
                self.assertEqual(model.predict_proba(x).shape, shape)
                torch.testing.assert_close(output["class_probabilities"].sum(-1), torch.ones(2, 12, horizon_count))
                self.assertTrue((output["class_probabilities"] >= 0).all())
                self.assertEqual(set(model.predict_last(x)), set(OTHER_TASKS))
                self.assertEqual(model.unused_prediction_tasks, ("return",))

    def test_eval_determinism_and_future_causality_apply_to_every_output(self):
        x, _, _, _, _ = inputs()
        changed = x.clone()
        changed[:, 7:] = changed[:, 7:] * 100 + 100
        for model in self.models():
            model.eval()
            first = model(x)
            torch.manual_seed(9999)
            repeated = model(x)
            perturbed = model(changed)
            for task in first:
                torch.testing.assert_close(first[task], repeated[task], rtol=0, atol=0)
                torch.testing.assert_close(first[task][:, :7], perturbed[task][:, :7], rtol=0, atol=0)

    def test_ce_replaces_return_mse_with_identical_six_losses_and_wm_auxiliary(self):
        x, targets, valid, reward, labels = inputs(2)
        for classifier in self.models(2):
            classifier.eval()
            base_cls = SevenTaskWM if isinstance(classifier, WMReturnClassifier) else DirectSevenTaskTransformer
            base = base_cls(10, MODEL, scales(2)).eval()
            base.load_state_dict({key: value for key, value in classifier.state_dict().items()
                                  if not key.startswith("classification_head.")})
            base_total, base_parts = base.loss(x, x[..., :5], targets, valid, reward)
            total, parts = classifier.loss(x, x[..., :5], targets, valid, reward, labels)
            for task in OTHER_TASKS:
                torch.testing.assert_close(parts[task], base_parts[task], rtol=0, atol=0)
            for task in ("reconstruction", "kl_dyn", "kl_rep", "next_return", "auxiliary"):
                if task in base_parts:
                    torch.testing.assert_close(parts[task], base_parts[task], rtol=0, atol=0)
            torch.testing.assert_close(total, base_total - base_parts["return"] + parts["classification"])
            self.assertNotIn("return", parts)

    def test_return_targets_are_unused_and_all_trained_heads_have_gradients(self):
        x, targets, valid, reward, labels = inputs(2)
        changed = {**targets, "return": torch.full_like(targets["return"], float("nan"))}
        for model in self.models(2):
            model.eval()
            first, _ = model.loss(x, x[..., :5], targets, valid, reward, labels)
            second, parts = model.loss(x, x[..., :5], changed, valid, reward, labels)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            second.backward()
            coverage = model.supervised_gradient_coverage()
            self.assertEqual(set(coverage), {*OTHER_TASKS, "classification"})
            self.assertTrue(all(coverage.values()))
            scalar_return_gradient = model.supervised_output_layer.weight.grad[:2]
            self.assertEqual(float(scalar_return_gradient.abs().sum()), 0.)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertTrue(torch.isfinite(parts["classification"]))

    def test_masked_nan_targets_and_invalid_class_placeholders_have_no_effect(self):
        x, targets, valid, reward, labels = inputs(2)
        valid[:, -3:] = False
        changed_targets = {key: value.clone() for key, value in targets.items()}
        changed_labels = labels.clone()
        changed_labels[~valid] = -9876
        for value in changed_targets.values():
            value[~valid] = float("nan")
        for model in self.models(2):
            model.eval()
            first, _ = model.loss(x, x[..., :5], targets, valid, reward, labels)
            first.backward()
            gradients = {key: value.grad.clone() for key, value in model.named_parameters() if value.grad is not None}
            model.zero_grad(set_to_none=True)
            second, _ = model.loss(x, x[..., :5], changed_targets, valid, reward, changed_labels)
            second.backward()
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            for key, parameter in model.named_parameters():
                if key in gradients:
                    torch.testing.assert_close(gradients[key], parameter.grad, rtol=0, atol=0)

    def test_wm_classifier_uses_exactly_one_latent_forward_during_training(self):
        model = WMReturnClassifier(10, MODEL, scales()).train()
        x, targets, valid, reward, labels = inputs()
        with patch.object(model.wm, "encode", wraps=model.wm.encode) as encode:
            loss, _ = model.loss(x, x[..., :5], targets, valid, reward, labels)
            loss.backward()
        self.assertEqual(encode.call_count, 1)
        self.assertGreater(float(model.wm.encoder.net[0].weight.grad.abs().sum()), 0.)

    def test_probabilities_are_raw_softmax_without_prior_adjustment(self):
        x, _, _, _, _ = inputs()
        for model in self.models():
            model.eval()
            with torch.no_grad():
                model.classification_output_layer.weight.zero_()
                model.classification_output_layer.bias.copy_(torch.tensor([.2, .3, .5]).log())
            torch.testing.assert_close(model.predict_proba(x), torch.tensor([[.2, .3, .5], [.2, .3, .5]]))

    def test_extreme_class_logits_have_finite_cross_entropy_and_gradient(self):
        x, targets, valid, reward, labels = inputs()
        labels[:] = 2
        for model in self.models():
            with torch.no_grad():
                model.classification_output_layer.weight.zero_()
                model.classification_output_layer.bias.copy_(torch.tensor([1000., -1000., 0.]))
            total, parts = model.loss(x, x[..., :5], targets, valid, reward, labels)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertAlmostEqual(float(parts["classification"]), 1000.)
            self.assertTrue(torch.isfinite(model.classification_output_layer.bias.grad).all())

    def test_classification_uses_all_valid_tokens_including_earlier_labels(self):
        x, targets, valid, reward, labels = inputs()
        labels[:] = 0
        changed = labels.clone()
        changed[:, :-1] = 2
        for model in self.models():
            model.eval()
            with torch.no_grad():
                model.classification_output_layer.weight.zero_()
                model.classification_output_layer.bias.copy_(torch.tensor([2., 0., -2.]))
            _, first = model.loss(x, x[..., :5], targets, valid, reward, labels)
            _, second = model.loss(x, x[..., :5], targets, valid, reward, changed)
            self.assertGreater(float(second["classification"]), float(first["classification"]) + 3.)

    def test_empty_valid_mask_has_zero_supervised_loss(self):
        x, targets, valid, reward, labels = inputs()
        valid[:] = False
        labels[:] = -999
        for value in targets.values(): value[:] = float("nan")
        for model in self.models():
            total, parts = model.loss(x, x[..., :5], targets, valid, reward, labels)
            self.assertEqual(float(parts["supervised"]), 0.)
            total.backward()
            self.assertTrue(torch.isfinite(total))

    def test_class_labels_require_explicit_int64_token_horizon_shape(self):
        x, targets, valid, reward, labels = inputs()
        for model in self.models():
            with self.assertRaisesRegex(ValueError, "labels must be int64"):
                model.loss(x, x[..., :5], targets, valid, reward, labels.float())
            with self.assertRaisesRegex(ValueError, "labels must be int64"):
                model.loss(x, x[..., :5], targets, valid, reward, labels[:, -1])


if __name__ == "__main__":
    unittest.main()
