from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from unidream.experiments.wm_lstm_grid_models import LSTMFullGrid, LSTMJointGrid, model_for
from unidream.experiments.wm_seven_task_models import TASKS


CFG = {"context": 8, "d_model": 16, "n_heads": 4, "n_layers": 2, "d_ff": 32,
       "encoder_hidden": 16, "dropout": 0., "lstm_hidden_size": 20, "lstm_layers": 2}
SCALES = {task: [1e-4 if task == "variance" else .1] for task in TASKS if task != "crash"}


def make(formulation, cfg=None, scales=None):
    return (LSTMJointGrid(5, cfg or CFG, scales or SCALES) if formulation == "joint"
            else LSTMFullGrid(5, cfg or CFG, scales or SCALES, formulation))


def fixture():
    torch.manual_seed(49)
    x = torch.randn(2, 6, 5)
    y = {task: torch.randn(2, 6, 1) * .1 for task in TASKS}
    y["variance"] = y["variance"].abs() * .001 + 1e-6
    y["downside"] = y["downside"].abs()
    y["dd_excess"] = y["dd_excess"].abs()
    y["crash"] = torch.randint(0, 2, (2, 6, 1)).float()
    c = {task: torch.randint(0, 2 if task == "crash" else 3, (2, 6, 1)) for task in TASKS}
    return x, y, c, torch.ones(2, 6, 1, dtype=torch.bool), torch.zeros(2, 6, 5), torch.zeros(2, 6)


def flatten(prediction, formulation):
    return ({mode + "_" + task: value for mode, targets in prediction.items() for task, value in targets.items()}
            if formulation == "joint" else prediction)


class LSTMGridModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_every_formulation_is_strictly_causal(self):
        x, *_ = fixture()
        changed = x.clone()
        changed[:, 4:] += 100
        for mode in ("regression", "classification", "joint"):
            with self.subTest(mode=mode), torch.no_grad():
                model = make(mode).eval()
                before, after = flatten(model(x), mode), flatten(model(changed), mode)
                for key in before:
                    torch.testing.assert_close(before[key][:, :4], after[key][:, :4], rtol=0, atol=1e-7)

    def test_recurrent_state_never_carries_between_windows(self):
        x, *_ = fixture()
        model = make("joint").eval()
        with torch.no_grad():
            before = flatten(model(x), "joint")
            model(torch.randn_like(x) * 100)
            after = flatten(model(x), "joint")
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_seeded_backbone_hidden_head_and_native_rows_match_all_controls(self):
        models = {}
        for mode in ("regression", "classification", "joint"):
            torch.manual_seed(17)
            models[mode] = make(mode).eval()
        joint = models["joint"]
        for mode, model in models.items():
            if mode == "joint":
                continue
            for name, value in model.state_dict().items():
                if name not in ("head.2.weight", "head.2.bias"):
                    torch.testing.assert_close(value, joint.state_dict()[name], rtol=0, atol=0)
            part = slice(0, 7) if mode == "regression" else slice(7, 26)
            torch.testing.assert_close(model.head[-1].weight, joint.head[-1].weight[part], rtol=0, atol=0)
            torch.testing.assert_close(model.head[-1].bias, joint.head[-1].bias[part], rtol=0, atol=0)

    def test_joint_loss_is_exact_half_of_each_native_control_objective(self):
        x, y, c, mask, base, nxt = fixture()
        models = {}
        for mode in ("regression", "classification", "joint"):
            torch.manual_seed(17)
            models[mode] = make(mode).eval()
        reg, _ = models["regression"].loss(x, base, y, mask, nxt)
        classification, _ = models["classification"].loss(x, base, c, mask, nxt)
        joint, parts = models["joint"].loss(x, base, {"regression": y, "classification": c}, mask, nxt)
        torch.testing.assert_close(joint, .5 * (reg + classification), rtol=1e-6, atol=1e-7)
        self.assertNotIn("auxiliary", parts)
        self.assertNotIn("next_return", parts)

    def test_both_head_sets_use_one_recurrent_forward_pass(self):
        x, y, c, mask, base, nxt = fixture()
        model = make("joint")
        calls = []
        handle = model.recurrent.register_forward_hook(lambda *args: calls.append(1))
        try:
            model.loss(x, base, {"regression": y, "classification": c}, mask, nxt)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 1)

    def test_all_seven_or_fourteen_native_heads_receive_finite_nonzero_gradients(self):
        x, y, c, mask, base, nxt = fixture()
        for mode in ("regression", "classification", "joint"):
            with self.subTest(mode=mode):
                model = make(mode)
                targets = {"regression": y, "classification": c} if mode == "joint" else c if mode == "classification" else y
                loss, _ = model.loss(x, base, targets, mask, nxt)
                loss.backward()
                coverage = model.supervised_gradient_coverage()
                self.assertEqual(len(coverage), 14 if mode == "joint" else 7)
                self.assertTrue(all(coverage.values()))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_masked_nan_and_invalid_class_placeholders_never_poison_gradients(self):
        x, y, c, mask, base, nxt = fixture()
        mask[:, 2:4] = False
        for values in y.values():
            values[:, 2:4] = float("nan")
        for values in c.values():
            values[:, 2:4] = 999
        model = make("joint")
        loss, _ = model.loss(x, base, {"regression": y, "classification": c}, mask, nxt)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_classification_probabilities_and_scalar_domains_match_frozen_contract(self):
        x, *_ = fixture()
        prediction = make("joint").eval().predict_last(x)
        for task, value in prediction["classification"].items():
            self.assertEqual(value.shape, (2, 1, 2 if task == "crash" else 3))
            torch.testing.assert_close(value.sum(-1), torch.ones(2, 1))
        point = prediction["regression"]
        for task in ("variance", "downside", "dd_excess"):
            self.assertTrue((point[task] >= 0).all())
        self.assertTrue(((point["crash"] >= 0) & (point["crash"] <= 1)).all())
        self.assertTrue((point["recovery"].abs() <= 1).all())

    def test_checkpoint_roundtrip_reproduces_joint_predictions(self):
        x, y, c, mask, base, nxt = fixture()
        model = make("joint")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss, _ = model.loss(x, base, {"regression": y, "classification": c}, mask, nxt)
        loss.backward()
        optimizer.step()
        model.eval()
        before = flatten(model.predict_last(x), "joint")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "research_model.pt"
            torch.save({"state_dict": model.state_dict(), "metadata": model.checkpoint_metadata()}, path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            restored = make("joint").eval()
            restored.load_state_dict(checkpoint["state_dict"])
        for name, value in flatten(restored.predict_last(x), "joint").items():
            torch.testing.assert_close(before[name], value, rtol=0, atol=0)

    def test_metadata_reports_capacity_and_lstm_control_scope(self):
        reg, classification, joint = (make(mode) for mode in ("regression", "classification", "joint"))
        metadata = joint.checkpoint_metadata()
        self.assertFalse(metadata["bidirectional"])
        self.assertFalse(metadata["persistent_recurrent_state"])
        self.assertEqual(metadata["wm_auxiliary_evaluations_per_batch"], 0)
        self.assertEqual(metadata["total_parameters"], sum(p.numel() for p in joint.parameters()))
        self.assertEqual(metadata["additional_parameters_vs_regression"], sum(p.numel() for p in joint.parameters()) - sum(p.numel() for p in reg.parameters()))
        self.assertEqual(metadata["additional_parameters_vs_classification"], sum(p.numel() for p in joint.parameters()) - sum(p.numel() for p in classification.parameters()))
        self.assertFalse(metadata["initial_backbone_and_hidden_head_match_frozen_controls"])
        self.assertTrue(metadata["same_seed_backbone_and_hidden_head_match_lstm_controls"])

    def test_invalid_bidirectionality_dimensions_input_and_mode_are_rejected(self):
        for cfg in ({**CFG, "lstm_bidirectional": True}, {**CFG, "lstm_hidden_size": 0}, {**CFG, "lstm_layers": 0}):
            with self.assertRaises(ValueError):
                make("regression", cfg)
        with self.assertRaises(ValueError):
            make("unregistered")
        with self.assertRaises(ValueError):
            make("regression")(torch.zeros(1, 9, 5))
        with self.assertRaises(ValueError):
            model_for({"model": "wm", "formulation": "regression"}, {"model": CFG}, SCALES)

    def test_factory_requires_explicit_mode_and_supports_all_three_arms(self):
        for mode in ("regression", "classification", "joint"):
            model = model_for({"model": "lstm", "formulation": mode}, {"model": CFG}, SCALES)
            self.assertEqual(model.input_dim, 63)
            self.assertEqual(model.formulation, mode)


if __name__ == "__main__":
    unittest.main()
