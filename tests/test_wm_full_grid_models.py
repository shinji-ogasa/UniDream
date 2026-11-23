"""Scientific contracts of the fresh 7-target × 2-formulation neural arms."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from unidream.experiments.wm_full_grid_models import DirectFullGrid, WMFullGrid
from unidream.experiments.wm_seven_task_models import (
    TASKS, SUPERVISED_WEIGHTS, DirectSevenTaskTransformer, SevenTaskWM,
)


MODEL = {"context": 12, "d_model": 16, "n_heads": 2, "n_layers": 1,
         "d_ff": 32, "encoder_hidden": 16, "n_categoricals": 3,
         "n_classes": 4, "dropout": .2}


def scales(horizons=1):
    return {task: [.001 if task == "variance" else .02] * horizons
            for task in TASKS if task != "crash"}


def examples(formulation, horizons=1):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2, 12, 10, generator=generator)
    shape = (2, 12, horizons)
    if formulation == "classification":
        targets = {task: torch.randint(0, 2 if task == "crash" else 3, shape, generator=generator)
                   for task in TASKS}
    else:
        targets = {task: torch.randn(shape, generator=generator) * .02 for task in TASKS}
        for task in ("variance", "downside", "dd_excess"):
            targets[task] = targets[task].abs() + .0001
        targets["crash"] = (targets["crash"] > 0).float()
    valid = torch.ones(shape, dtype=torch.bool)
    reward = torch.randn(2, 12, generator=generator) * .005
    return x, targets, valid, reward


class FullGridModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def models(self, formulation, horizons=1):
        for cls in (DirectFullGrid, WMFullGrid):
            torch.manual_seed(7)
            yield cls(10, MODEL, scales(horizons), formulation)

    def test_shapes_domains_and_single_crash_logit_in_both_formulations(self):
        for formulation in ("regression", "classification"):
            for count in (1, 2):
                x, _, _, _ = examples(formulation, count)
                for model in self.models(formulation, count):
                    model.eval()
                    output = model(x)
                    self.assertEqual(tuple(output), TASKS)
                    self.assertEqual(model.supervised_output_layer.out_features, count * (7 if formulation == "regression" else 19))
                    crash_slice = model.task_output_slices["crash"]
                    self.assertEqual(crash_slice.stop - crash_slice.start, count)
                    for task in TASKS:
                        tail = () if formulation == "regression" else (2 if task == "crash" else 3,)
                        self.assertEqual(output[task].shape, (2, 12, count, *tail))
                        self.assertEqual(model.predict_last(x)[task].shape, (2, count, *tail))
                        if formulation == "classification":
                            torch.testing.assert_close(output[task].sum(-1), torch.ones(2, 12, count))
                            self.assertTrue((output[task] >= 0).all())
                    if formulation == "regression":
                        for task in ("variance", "downside", "dd_excess"):
                            self.assertTrue((output[task] > 0).all())
                        self.assertTrue(((output["crash"] >= 0) & (output["crash"] <= 1)).all())
                        self.assertTrue((output["recovery"].abs() <= 1).all())

    def test_every_exposed_output_row_receives_finite_nonzero_gradient(self):
        for formulation in ("regression", "classification"):
            x, targets, valid, reward = examples(formulation, 2)
            for model in self.models(formulation, 2):
                model.train()
                loss, parts = model.loss(x, x[..., :5], targets, valid, reward)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(set(model.supervised_gradient_coverage()), set(TASKS))
                self.assertTrue(all(model.supervised_gradient_coverage().values()))
                gradient = model.supervised_output_layer.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue((gradient.abs().sum(1) > 0).all())
                self.assertTrue(all(torch.isfinite(value) for value in parts.values()))

    def test_regression_changes_only_crash_bce_into_brier_and_keeps_wm_aux(self):
        x, targets, valid, reward = examples("regression")
        for model in self.models("regression"):
            model.eval()
            base_cls = SevenTaskWM if isinstance(model, WMFullGrid) else DirectSevenTaskTransformer
            base = base_cls(10, MODEL, scales()).eval()
            base.load_state_dict(model.state_dict())
            original_loss, original_parts = base.loss(x, x[..., :5], targets, valid, reward)
            modified_loss, modified_parts = model.loss(x, x[..., :5], targets, valid, reward)
            expected_brier = (model(x)["crash"] - targets["crash"]).square().mean()
            torch.testing.assert_close(modified_parts["crash"], expected_brier)
            for task in TASKS:
                if task != "crash":
                    torch.testing.assert_close(modified_parts[task], original_parts[task], rtol=0, atol=0)
            for key in ("reconstruction", "kl_dyn", "kl_rep", "next_return", "auxiliary"):
                if key in original_parts:
                    torch.testing.assert_close(modified_parts[key], original_parts[key], rtol=0, atol=0)
            expected_total = original_loss - SUPERVISED_WEIGHTS["crash"] * original_parts["crash"] + SUPERVISED_WEIGHTS["crash"] * expected_brier
            torch.testing.assert_close(modified_loss, expected_total)

    def test_classification_is_independent_weighted_ce_and_binary_bce(self):
        x, targets, valid, reward = examples("classification", 2)
        valid[:, -2:] = False
        for model in self.models("classification", 2):
            model.eval()
            output = model(x)
            _, parts = model.loss(x, x[..., :5], targets, valid, reward)
            expected = {}
            for task in TASKS:
                if task == "crash":
                    expected[task] = F.binary_cross_entropy(output[task][..., 1][valid], targets[task][valid].float())
                else:
                    probability = output[task][valid]
                    labels = targets[task][valid]
                    expected[task] = -probability[torch.arange(len(labels)), labels].log().mean()
                torch.testing.assert_close(parts[task], expected[task])
            total = sum(SUPERVISED_WEIGHTS[task] * expected[task] for task in TASKS)
            torch.testing.assert_close(parts["supervised"], total)

    def test_seeded_backbone_and_shared_hidden_head_match_between_formulations(self):
        x, _, _, _ = examples("regression")
        for cls in (DirectFullGrid, WMFullGrid):
            torch.manual_seed(42)
            regression = cls(10, MODEL, scales(), "regression").eval()
            torch.manual_seed(42)
            classification = cls(10, MODEL, scales(), "classification").eval()
            torch.testing.assert_close(regression.latent_last(x), classification.latent_last(x), rtol=0, atol=0)
            torch.testing.assert_close(regression.head[0].weight, classification.head[0].weight, rtol=0, atol=0)

    def test_future_causality_and_deterministic_eval(self):
        for formulation in ("regression", "classification"):
            x, _, _, _ = examples(formulation)
            changed = x.clone()
            changed[:, 7:] = changed[:, 7:] * 100 + 100
            for model in self.models(formulation):
                model.eval()
                first = model(x)
                torch.manual_seed(999)
                repeat = model(x)
                altered = model(changed)
                for task in TASKS:
                    torch.testing.assert_close(first[task], repeat[task], rtol=0, atol=0)
                    torch.testing.assert_close(first[task][:, :7], altered[task][:, :7], rtol=0, atol=0)

    def test_masked_nan_regression_and_invalid_class_labels_do_not_affect_gradients(self):
        for formulation in ("regression", "classification"):
            x, targets, valid, reward = examples(formulation)
            valid[:, -3:] = False
            changed = {task: value.clone() for task, value in targets.items()}
            for value in changed.values():
                value[~valid] = float("nan") if formulation == "regression" else -999
            for model in self.models(formulation):
                model.eval()
                first, _ = model.loss(x, x[..., :5], targets, valid, reward)
                first.backward()
                gradients = {key: parameter.grad.clone() for key, parameter in model.named_parameters() if parameter.grad is not None}
                model.zero_grad(set_to_none=True)
                second, _ = model.loss(x, x[..., :5], changed, valid, reward)
                second.backward()
                torch.testing.assert_close(first, second, rtol=0, atol=0)
                for key, parameter in model.named_parameters():
                    if key in gradients:
                        torch.testing.assert_close(parameter.grad, gradients[key], rtol=0, atol=0)

    def test_classification_wm_shares_one_latent_forward_and_preserves_aux_values(self):
        model = WMFullGrid(10, MODEL, scales(), "classification").eval()
        x, labels, valid, reward = examples("classification")
        with patch.object(model.wm, "encode", wraps=model.wm.encode) as encode:
            total, parts = model.loss(x, x[..., :5], labels, valid, reward)
        self.assertEqual(encode.call_count, 1)
        _, _, state = model._predict_with_state(x)
        from unidream.world_model.encoder import symlog
        expected_reconstruction = F.mse_loss(model.wm.decoder(state["z"], state["h"]), symlog(x[..., :5]))
        torch.testing.assert_close(parts["reconstruction"], expected_reconstruction)
        torch.testing.assert_close(total, parts["supervised"] + parts["auxiliary"])

    def test_extreme_binary_logit_uses_stable_bce_in_classification(self):
        x, targets, valid, reward = examples("classification")
        targets["crash"][:] = 0
        for model in self.models("classification"):
            with torch.no_grad():
                task_slice = model.task_output_slices["crash"]
                model.supervised_output_layer.weight[task_slice].zero_()
                model.supervised_output_layer.bias[task_slice] = 1000.
            total, parts = model.loss(x, x[..., :5], targets, valid, reward)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertAlmostEqual(float(parts["crash"]), 1000.)

    def test_wholly_invalid_targets_contribute_zero_supervised_loss(self):
        for formulation in ("regression", "classification"):
            x, targets, valid, reward = examples(formulation)
            valid[:] = False
            for value in targets.values(): value[:] = float("nan") if formulation == "regression" else -999
            for model in self.models(formulation):
                total, parts = model.loss(x, x[..., :5], targets, valid, reward)
                total.backward()
                self.assertEqual(float(parts["supervised"]), 0.)
                self.assertTrue(torch.isfinite(total))
                self.assertFalse(any(model.supervised_gradient_coverage().values()))

    def test_invalid_modes_and_binary_label_domain_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "formulation"):
            DirectFullGrid(10, MODEL, scales(), "unexpected")
        x, targets, valid, reward = examples("classification")
        targets["crash"][:] = 2
        model = DirectFullGrid(10, MODEL, scales(), "classification")
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            model.loss(x, x[..., :5], targets, valid, reward)


if __name__ == "__main__":
    unittest.main()
