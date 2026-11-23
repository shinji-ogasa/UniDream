"""Scientific contracts of the shared seven-target joint formulation."""
import unittest
from unittest.mock import patch

import torch

from unidream.experiments.wm_full_grid_models import DirectFullGrid, WMFullGrid
from unidream.experiments.wm_joint_grid_models import DirectJointGrid, WMJointGrid
from unidream.experiments.wm_seven_task_models import TASKS


MODEL = {"context": 12, "d_model": 16, "n_heads": 2, "n_layers": 1,
         "d_ff": 32, "encoder_hidden": 16, "n_categoricals": 3,
         "n_classes": 4, "dropout": .2}


def scales(horizons=1):
    return {task: [.001 if task == "variance" else .02] * horizons
            for task in TASKS if task != "crash"}


def example(horizons=1):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2, 12, 10, generator=generator)
    shape = (2, 12, horizons)
    targets = {task: torch.randn(shape, generator=generator) * .02 for task in TASKS}
    for task in ("variance", "downside", "dd_excess"):
        targets[task] = targets[task].abs() + .0001
    targets["crash"] = (targets["crash"] > 0).float()
    labels = {task: torch.randint(0, 2 if task == "crash" else 3, shape, generator=generator)
              for task in TASKS}
    labels["crash"] = targets["crash"].long()
    valid = torch.ones(shape, dtype=torch.bool)
    reward = torch.randn(2, 12, generator=generator) * .005
    return x, targets, labels, valid, reward


class JointGridModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def models(self, horizons=1):
        for cls in (DirectJointGrid, WMJointGrid):
            torch.manual_seed(7)
            yield cls(10, MODEL, scales(horizons))

    def test_both_native_predictions_have_registered_shapes_and_domains(self):
        for horizons in (1, 2):
            x, _, _, _, _ = example(horizons)
            for model in self.models(horizons):
                model.eval()
                output = model(x)
                last = model.predict_last(x)
                self.assertEqual(set(output), {"regression", "classification"})
                self.assertEqual(model.supervised_output_layer.out_features, 26 * horizons)
                for task in TASKS:
                    classes = 2 if task == "crash" else 3
                    self.assertEqual(output["regression"][task].shape, (2, 12, horizons))
                    self.assertEqual(output["classification"][task].shape, (2, 12, horizons, classes))
                    self.assertEqual(last["regression"][task].shape, (2, horizons))
                    self.assertEqual(last["classification"][task].shape, (2, horizons, classes))
                    torch.testing.assert_close(output["classification"][task].sum(-1), torch.ones(2, 12, horizons))
                    self.assertTrue((output["classification"][task] >= 0).all())
                for task in ("variance", "downside", "dd_excess"):
                    self.assertTrue((output["regression"][task] > 0).all())
                self.assertTrue(((output["regression"]["crash"] >= 0) & (output["regression"]["crash"] <= 1)).all())
                self.assertTrue((output["regression"]["recovery"].abs() <= 1).all())

    def test_seeded_backbone_and_each_native_output_match_frozen_controls_exactly(self):
        x, _, _, _, _ = example(2)
        for joint_cls, control_cls in ((DirectJointGrid, DirectFullGrid), (WMJointGrid, WMFullGrid)):
            torch.manual_seed(41)
            joint = joint_cls(10, MODEL, scales(2)).eval()
            for formulation in ("regression", "classification"):
                torch.manual_seed(41)
                control = control_cls(10, MODEL, scales(2), formulation).eval()
                torch.testing.assert_close(joint.latent_last(x), control.latent_last(x), rtol=0, atol=0)
                torch.testing.assert_close(joint.head[0].weight, control.head[0].weight, rtol=0, atol=0)
                native_slice = slice(0, 14) if formulation == "regression" else slice(14, 52)
                torch.testing.assert_close(joint.head[-1].weight[native_slice], control.head[-1].weight, rtol=0, atol=0)
                torch.testing.assert_close(joint.head[-1].bias[native_slice], control.head[-1].bias, rtol=0, atol=0)
                for task in TASKS:
                    # Wider GEMM kernels may round the same dot products
                    # differently even though the parameters match exactly.
                    torch.testing.assert_close(joint(x)[formulation][task], control(x)[task], rtol=2e-6, atol=1e-8)

    def test_objective_is_half_of_each_control_plus_one_unchanged_wm_auxiliary(self):
        x, targets, labels, valid, reward = example(2)
        valid[:, -2:] = False
        for joint_cls, control_cls in ((DirectJointGrid, DirectFullGrid), (WMJointGrid, WMFullGrid)):
            torch.manual_seed(13)
            joint = joint_cls(10, MODEL, scales(2)).eval()
            joint_loss, parts = joint.loss(x, x[..., :5], targets, valid, reward, labels=labels)
            expected_supervised = 0.
            control_aux = None
            for formulation, native_targets in (("regression", targets), ("classification", labels)):
                torch.manual_seed(13)
                control = control_cls(10, MODEL, scales(2), formulation).eval()
                _, control_parts = control.loss(x, x[..., :5], native_targets, valid, reward)
                expected_supervised = expected_supervised + .5 * control_parts["supervised"]
                for task in TASKS:
                    torch.testing.assert_close(parts[f"{formulation}_{task}"], control_parts[task], rtol=2e-6, atol=1e-7)
                if "auxiliary" in control_parts:
                    control_aux = control_parts["auxiliary"]
                    for name in ("reconstruction", "kl_dyn", "kl_rep", "next_return", "auxiliary"):
                        torch.testing.assert_close(parts[name], control_parts[name], rtol=0, atol=0)
            torch.testing.assert_close(parts["supervised"], expected_supervised, rtol=2e-6, atol=1e-7)
            torch.testing.assert_close(joint_loss, expected_supervised + (0. if control_aux is None else control_aux), rtol=2e-6, atol=1e-7)

    def test_every_native_head_and_horizon_receives_a_finite_nonzero_gradient(self):
        x, targets, labels, valid, reward = example(2)
        for model in self.models(2):
            loss, parts = model.loss(x, x[..., :5], targets, valid, reward, labels=labels)
            loss.backward()
            coverage = model.supervised_gradient_coverage()
            self.assertEqual(len(coverage), 14)
            self.assertTrue(all(coverage.values()))
            gradient = model.supervised_output_layer.weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient.abs().sum(1) > 0).all())
            self.assertTrue(all(torch.isfinite(value) for value in parts.values()))

    def test_single_shared_representation_pass_per_joint_training_batch(self):
        x, targets, labels, valid, reward = example()
        for model in self.models():
            representation = model._representations
            with patch.object(model, "_representations", wraps=representation) as calls:
                model.loss(x, x[..., :5], targets, valid, reward, labels=labels)
            self.assertEqual(calls.call_count, 1)
            if isinstance(model, WMJointGrid):
                with patch.object(model.wm, "encode", wraps=model.wm.encode) as encode:
                    model.loss(x, x[..., :5], targets, valid, reward, labels=labels)
                self.assertEqual(encode.call_count, 1)

    def test_masked_nan_and_invalid_class_placeholders_preserve_values_and_gradients(self):
        x, targets, labels, valid, reward = example()
        valid[:, -3:] = False
        changed_targets = {task: value.clone() for task, value in targets.items()}
        changed_labels = {task: value.clone() for task, value in labels.items()}
        for task in TASKS:
            changed_targets[task][~valid] = float("nan")
            changed_labels[task][~valid] = -999
        for model in self.models():
            model.eval()
            first, _ = model.loss(x, x[..., :5], targets, valid, reward, labels=labels)
            first.backward()
            gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
            model.zero_grad(set_to_none=True)
            second, _ = model.loss(x, x[..., :5], changed_targets, valid, reward, labels=changed_labels)
            second.backward()
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            for name, parameter in model.named_parameters():
                if name in gradients:
                    torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)

    def test_all_invalid_supervised_positions_contribute_zero_and_zero_head_gradient(self):
        x, targets, labels, valid, reward = example()
        valid[:] = False
        for task in TASKS:
            targets[task][:] = float("nan")
            labels[task][:] = -999
        for model in self.models():
            total, parts = model.loss(x, x[..., :5], targets, valid, reward, labels=labels)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertEqual(float(parts["supervised"]), 0.)
            self.assertFalse(any(model.supervised_gradient_coverage().values()))

    def test_nested_target_convenience_api_has_identical_objective(self):
        x, targets, labels, valid, reward = example()
        for model in self.models():
            model.eval()
            separate, _ = model.loss(x, x[..., :5], targets, valid, reward, labels=labels)
            nested, _ = model.loss(x, x[..., :5], {"regression": targets, "classification": labels}, valid, reward)
            torch.testing.assert_close(separate, nested, rtol=0, atol=0)

    def test_future_input_cannot_change_past_forecasts_and_eval_is_deterministic(self):
        x, _, _, _, _ = example()
        changed = x.clone()
        changed[:, 7:] = changed[:, 7:] * 100 + 100
        for model in self.models():
            model.eval()
            first, repeat, altered = model(x), model(x), model(changed)
            for formulation in first:
                for task in TASKS:
                    torch.testing.assert_close(first[formulation][task], repeat[formulation][task], rtol=0, atol=0)
                    torch.testing.assert_close(first[formulation][task][:, :7], altered[formulation][task][:, :7], rtol=0, atol=0)

    def test_metadata_discloses_capacity_difference_and_shared_model_contract(self):
        for model in self.models(2):
            metadata = model.checkpoint_metadata()
            self.assertTrue(metadata["shared_parameters"])
            self.assertEqual(metadata["supervised_formulation_weights"], {"regression": .5, "classification": .5})
            self.assertEqual(metadata["joint_output_rows"], 52)
            self.assertEqual(metadata["additional_parameters_vs_regression"], 38 * 17)
            self.assertEqual(metadata["additional_parameters_vs_classification"], 14 * 17)
            self.assertEqual(metadata["representation_passes_per_batch"], 1)
            self.assertEqual(metadata["wm_auxiliary_evaluations_per_batch"], int(isinstance(model, WMJointGrid)))

    def test_invalid_target_contracts_fail_clearly_and_binary_ce_remains_stable(self):
        x, targets, labels, valid, reward = example()
        for model in self.models():
            with self.assertRaisesRegex(ValueError, "requires regression and classification"):
                model.loss(x, x[..., :5], targets, valid, reward)
            wrong_labels = {**labels, "return": labels["return"].float()}
            with self.assertRaisesRegex(ValueError, "int64"):
                model.loss(x, x[..., :5], targets, valid, reward, labels=wrong_labels)
            wrong_labels = {**labels, "crash": torch.full_like(labels["crash"], 2)}
            with self.assertRaisesRegex(ValueError, "binary 0/1"):
                model.loss(x, x[..., :5], targets, valid, reward, labels=wrong_labels)
            labels["crash"][:] = 0
            with torch.no_grad():
                output_slice = model.classification_task_output_slices["crash"]
                model.supervised_output_layer.weight[output_slice].zero_()
                model.supervised_output_layer.bias[output_slice] = 1000.
            total, parts = model.loss(x, x[..., :5], targets, valid, reward, labels=labels)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            self.assertAlmostEqual(float(parts["classification_crash"]), 1000.)


if __name__ == "__main__":
    unittest.main()
