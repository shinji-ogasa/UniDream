import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from unidream.experiments.wm_classification_models import ForecastClassifier


CONFIG = {"context": 12, "d_model": 16, "n_heads": 2, "n_layers": 2,
          "d_ff": 32, "encoder_hidden": 16, "n_categoricals": 3,
          "n_classes": 4, "dropout": .2}
SCALES = {"return": [.01, .02, .04], "variance": [.001, .002, .004], "downside": [.01, .02, .04]}


def batch():
    generator = torch.Generator().manual_seed(123)
    x = torch.randn((3, 12, 7), generator=generator)
    targets = {
        "return": torch.randn((3, 12, 3), generator=generator) * .01,
        "variance": torch.rand((3, 12, 3), generator=generator) * .004 + .0001,
        "downside": torch.rand((3, 12, 3), generator=generator) * .03,
        "crash": (torch.rand((3, 12, 3), generator=generator) > .7).float(),
    }
    valid = torch.ones((3, 12, 3), dtype=torch.bool)
    next_return = torch.randn((3, 12), generator=generator) * .005
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    side = torch.tensor([[-2.], [-1.], [-.5]])
    return x, targets, valid, next_return, labels, side


class ForecastClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def model(self, kind):
        torch.manual_seed(42)
        return ForecastClassifier(kind, 7, CONFIG, SCALES)

    def test_probability_shape_normalization_and_deterministic_eval(self):
        x, _, _, _, _, side = batch()
        for kind in ("direct", "wm"):
            with self.subTest(kind=kind):
                model = self.model(kind).eval()
                probabilities = model.predict_proba(x, side)
                self.assertEqual(probabilities.shape, (3, 3))
                self.assertTrue(torch.isfinite(probabilities).all())
                self.assertTrue((probabilities > 0).all())
                torch.testing.assert_close(probabilities.sum(1), torch.ones(3))
                torch.manual_seed(999)
                torch.testing.assert_close(probabilities, model.predict_proba(x, side), atol=0, rtol=0)

    def test_future_tokens_do_not_change_earlier_representations(self):
        x, _, _, _, _, _ = batch()
        changed = x.clone()
        changed[:, 7:] = changed[:, 7:] * 100. + 50.
        for kind in ("direct", "wm"):
            with self.subTest(kind=kind):
                model = self.model(kind).eval()
                before, after = model._representations(x), model._representations(changed)
                torch.testing.assert_close(before[:, :7], after[:, :7], atol=0, rtol=0)

    def test_threshold_is_appended_only_to_final_state(self):
        x, _, _, _, _, side = batch()
        for kind in ("direct", "wm"):
            with self.subTest(kind=kind):
                model = self.model(kind).eval()
                representation = model._representations(x)
                head_inputs = []
                handle = model.classification_head.register_forward_pre_hook(
                    lambda _module, inputs: head_inputs.append(inputs[0].detach().clone())
                )
                try:
                    first = model(x, side)
                    second = model(x, side + 1.)
                finally:
                    handle.remove()
                self.assertEqual(head_inputs[0].shape, (3, representation.shape[-1] + 1))
                torch.testing.assert_close(head_inputs[0][:, :-1], representation[:, -1], atol=0, rtol=0)
                torch.testing.assert_close(head_inputs[0][:, :-1], head_inputs[1][:, :-1], atol=0, rtol=0)
                torch.testing.assert_close(head_inputs[0][:, -1:], side)
                self.assertFalse(torch.equal(first, second))

    def test_direct_loss_is_exactly_unweighted_last_token_ce(self):
        x, _, _, _, labels, side = batch()
        model = self.model("direct").eval()
        expected = F.cross_entropy(model(x, side), labels)
        loss, parts = model.loss(x, None, None, None, None, labels, side)
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(parts["ce"], expected)
        self.assertEqual(float(parts["auxiliary"]), 0.)

    def test_ce_gradient_reaches_core_encoder_without_raw_shortcut(self):
        x, _, _, _, labels, side = batch()
        for kind in ("direct", "wm"):
            with self.subTest(kind=kind):
                model = self.model(kind).train()
                ce = F.cross_entropy(model(x, side), labels)
                ce.backward()
                encoder = model.direct_core.encoder[0] if kind == "direct" else model.wm.wm.encoder.net[0]
                self.assertTrue(torch.isfinite(encoder.weight.grad).all())
                self.assertGreater(float(encoder.weight.grad.abs().sum()), 0.)
                model.eval()
                representation = model._representations(x)
                with patch.object(model, "_representations", return_value=representation):
                    first, second = model(x, side), model(x * 100. + 7., side)
                torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_wm_loss_is_exact_full_auxiliary_plus_ce_on_same_forward(self):
        x, targets, valid, next_return, labels, side = batch()
        model = self.model("wm").eval()
        expected_auxiliary, _ = model.wm.loss(x, x[:, :, :5], targets, valid, next_return)
        expected_ce = F.cross_entropy(model(x, side), labels)
        with patch.object(model.wm, "forward", wraps=model.wm.forward) as observed:
            loss, parts = model.loss(x, x[:, :, :5], targets, valid, next_return, labels, side)
        self.assertEqual(observed.call_count, 1)
        self.assertEqual(len(model.wm._forward_hooks), 0)
        torch.testing.assert_close(parts["auxiliary"], expected_auxiliary)
        torch.testing.assert_close(parts["ce"], expected_ce)
        torch.testing.assert_close(loss, expected_auxiliary + expected_ce)

    def test_masked_future_auxiliary_nan_has_finite_gradients(self):
        x, targets, valid, next_return, labels, side = batch()
        valid[:, -4:, 2] = False
        valid[:, -2:, :] = False
        for value in targets.values():
            value[~valid] = float("nan")
        next_return[:, -2:] = float("nan")
        model = self.model("wm").train()
        loss, _ = model.loss(x, x[:, :, :5], targets, valid, next_return, labels, side)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        gradient = model.classification_head[-1].weight.grad
        self.assertTrue((gradient.abs().sum(1) > 0).all())

    def test_wm_auxiliary_is_independent_of_threshold_side(self):
        x, targets, valid, next_return, labels, side = batch()
        model = self.model("wm").eval()
        _, first = model.loss(x, x[:, :, :5], targets, valid, next_return, labels, side)
        _, second = model.loss(x, x[:, :, :5], targets, valid, next_return, labels, side + 10.)
        for key in first:
            if key != "ce":
                torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)

    def test_rejects_invalid_shape_labels_and_nonfinite_inputs(self):
        x, targets, valid, next_return, labels, side = batch()
        model = self.model("direct")
        with self.assertRaisesRegex(ValueError, "shape.*batch, 1"):
            model(x, side.expand(3, 12))
        with self.assertRaisesRegex(ValueError, "finite"):
            model(x, torch.full_like(side, float("nan")))
        bad_x = x.clone()
        bad_x[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model(bad_x, side)
        with self.assertRaisesRegex(ValueError, "labels"):
            model.loss(x, None, None, None, None, torch.tensor([0, 1, 3]), side)
        wm = self.model("wm")
        targets["return"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "return auxiliary target"):
            wm.loss(x, x[:, :, :5], targets, valid, next_return, labels, side)


if __name__ == "__main__":
    unittest.main()
