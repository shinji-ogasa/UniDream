import copy
import unittest

import numpy as np
import torch

from unidream.experiments.wm_tail_transfer_models import (
    TailModel, build_specs, decode_head, head_loss, output_shape,
)


def fixture():
    cfg = {"targets": {"horizons": [1, 4], "fixed_thresholds": [.01, .03],
        "dynamic_multipliers": [1., 2.], "hazard_thresholds": [.01, .03],
        "return_quantiles": [.05, .5, .95], "positive_quantiles": [.5, .95],
        "es_alphas": [.05], "passive_leverages": [1., 2.],
        "passive_loss_thresholds": [.1]}, "model": {"context": 8, "d_model": 12,
        "n_heads": 3, "n_layers": 1, "encoder_hidden": 12, "d_ff": 24,
        "n_categoricals": 3, "n_classes": 4, "dropout": 0.,
        "lstm_hidden": 12, "lstm_layers": 1}}
    specs = build_specs(cfg)
    targets, masks = {}, {}
    for spec in specs:
        shape = (3, *spec["shape"])
        if spec["kind"] in ("categorical", "hazard"):
            target = torch.randint(0, 3, shape).float()
        elif spec["kind"] == "binary":
            target = torch.randint(0, 2, shape).float()
        elif spec["kind"] in ("positive", "positive_quantile", "variance"):
            target = torch.rand(shape) * .02 + .001
        else:
            target = torch.randn(shape) * .01
        targets[spec["name"]] = target
        masks[spec["name"]] = torch.ones(shape, dtype=torch.bool)
    return cfg, specs, targets, masks


class TailModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_native_shapes_and_every_head_learns(self):
        cfg, specs, targets, masks = fixture()
        scales = {s["name"]: .01 for s in specs}
        scales["next_return"] = .005
        for variant in ("transformer", "lstm", "wm"):
            with self.subTest(variant=variant):
                torch.manual_seed(8)
                model = TailModel(5, cfg, specs, scales, variant)
                x = torch.randn(3, 8, 5)
                loss, parts = model.loss(x, targets, masks, next_return=torch.randn(3, 8) * .005)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(model.supervised_gradient_coverage().values()))
                self.assertIn("supervised", parts)
                prediction = model.predict_last(x)
                self.assertEqual(set(prediction), set(targets))
                for spec in specs:
                    self.assertEqual(prediction[spec["name"]].shape, (3, *output_shape(spec)))
                self.assertTrue(torch.isfinite(model.latent_last(x)).all())

    def test_future_inputs_do_not_change_earlier_outputs(self):
        cfg, specs, _, _ = fixture()
        x = torch.randn(2, 8, 5)
        changed = x.clone()
        changed[:, 5:] += 10
        for variant in ("transformer", "lstm", "wm"):
            model = TailModel(5, cfg, specs[:1], {"return": .01}, variant).eval()
            before, after = model(x)["return"], model(changed)["return"]
            torch.testing.assert_close(before[:, :5], after[:, :5])
            torch.testing.assert_close(before, model(x)["return"])

    def test_masked_missing_targets_do_not_poison_gradients(self):
        cfg, specs, targets, masks = fixture()
        for name in targets:
            targets[name][0] = float("nan")
            masks[name][0] = False
        model = TailModel(5, cfg, specs, {s["name"]: .01 for s in specs})
        loss, _ = model.loss(torch.randn(3, 8, 5), targets, masks)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        targets["return"][1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite observed"):
            model.loss(torch.randn(3, 8, 5), targets, masks)

    def test_quantile_order_and_var_es_domain(self):
        cfg, specs, _, _ = fixture()
        for spec in specs:
            raw = torch.randn(4, *output_shape(spec))
            scale = torch.full(spec["shape"], .01)
            value = decode_head(raw, spec, scale)
            if spec["kind"] in ("quantile", "positive_quantile"):
                self.assertTrue((value.diff(dim=-1) >= 0).all())
            if spec["kind"] == "positive_quantile":
                self.assertTrue((value >= 0).all())
            if spec["kind"] == "var_es":
                self.assertTrue((value[..., 1] < value[..., 0]).all())
                self.assertTrue((value[..., 0] < 0).all())

    def test_right_censored_likelihood_does_not_use_future_steps(self):
        spec = {"name": "hazard", "kind": "hazard", "shape": [4, 1], "task": "T11", "classes": 3}
        raw = torch.zeros(2, 4, 1, 3, requires_grad=True)
        actual = torch.tensor([[[0.], [1.], [float("nan")], [float("nan")]], [[0.], [0.], [0.], [0.]]])
        mask = torch.isfinite(actual)
        value, valid = head_loss(raw, raw.softmax(-1), actual, mask, spec, torch.ones(4, 1))
        np.testing.assert_allclose(value.detach().numpy()[:, 0], np.log(3) * np.array([2, 4]) / 4)
        self.assertTrue(valid.all())
        value.mean().backward()
        self.assertTrue((raw.grad[0, 2:] == 0).all())
        self.assertTrue((raw.grad[1].abs().sum(-1) > 0).all())

    def test_fz0_matches_formula(self):
        spec = {"name": "ve", "kind": "var_es", "shape": [1], "task": "T10", "levels": [.05]}
        prediction = torch.tensor([[[-.02, -.03]], [[-.02, -.03]]])
        actual = torch.tensor([[-.04], [.01]])
        value, _ = head_loss(torch.zeros_like(prediction), prediction, actual, torch.ones_like(actual, dtype=torch.bool), spec, torch.ones(1))
        expected = -np.array([.02, 0]) / (.05 * -.03) + (-.02 / -.03) + np.log(.03) - 1
        np.testing.assert_allclose(value.numpy().ravel(), expected, rtol=1e-6)

    def test_family_average_is_not_affected_by_number_of_heads(self):
        spec = {"name": "a", "kind": "real", "shape": [1], "task": "T01"}
        specs = [spec, {**spec, "name": "b"}, {**spec, "name": "c", "task": "T02"}]
        cfg = {"model": {"context": 2, "d_model": 4, "n_heads": 1, "n_layers": 1, "dropout": 0.}}
        model = TailModel(5, cfg, specs)
        for layer in model.heads.values():
            torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        targets = {"a": torch.tensor([[1.]]), "b": torch.tensor([[3.]]), "c": torch.tensor([[2.]])}
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in targets.items()}
        loss, _ = model.loss(torch.zeros(1, 2, 5), targets, masks)
        self.assertAlmostEqual(loss.item(), ((1 + 9) / 2 + 4) / 2)

    def test_actual_shared_joint_and_stochastic_world_rollout(self):
        cfg, specs, targets, masks = fixture()
        model = TailModel(5, cfg, specs, {s["name"]: .01 for s in specs}, "wm").eval()
        x = torch.randn(2, 8, 5)
        paths = model.sample_rollout(x, 3, 5)
        self.assertEqual(paths.shape, (2, 5, 3))
        self.assertTrue(torch.isfinite(paths).all())
        self.assertGreater(paths.std().item(), 0)
        self.assertFalse(model.training)
        original = copy.deepcopy(model.state_dict())
        restored = TailModel(5, cfg, specs, {s["name"]: .01 for s in specs}, "wm").eval()
        restored.load_state_dict(original)
        torch.testing.assert_close(model.predict_last(x)["return"], restored.predict_last(x)["return"])
        reg = TailModel(5, cfg, specs, objectives="regression")
        clas = TailModel(5, cfg, specs, objectives="classification")
        self.assertNotIn("return_class", reg.heads)
        self.assertNotIn("return", clas.heads)


if __name__ == "__main__":
    unittest.main()
