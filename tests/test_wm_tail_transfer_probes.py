import unittest

import torch

from unidream.experiments.wm_tail_transfer_models import TailModel
from unidream.experiments.wm_tail_transfer_probes import FrozenTailProbe, HurdleTailModel


class TailProbesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        cfg = {"model": {"context": 4, "d_model": 8, "n_heads": 2,
            "n_layers": 1, "encoder_hidden": 8, "d_ff": 16,
            "n_categoricals": 2, "n_classes": 4, "dropout": .2,
            "lstm_hidden": 8, "lstm_layers": 1}}
        specs = [dict(name="return", kind="real", shape=[1], task="T01", levels=None),
                 dict(name="mae", kind="positive", shape=[1], task="T05", levels=None)]
        x = torch.randn(4, 4, 5)
        targets = {"return": torch.randn(4, 1) * .02,
                   "mae": torch.tensor([[0.], [.01], [0.], [.03]])}
        masks = {k: torch.ones_like(v, dtype=torch.bool) for k, v in targets.items()}
        return cfg, specs, x, targets, masks

    def test_probe_keeps_market_encoder_exactly_fixed_and_trains_fresh_readout(self):
        cfg, specs, x, targets, masks = self.fixture()
        for variant in ("transformer", "lstm", "wm"):
            with self.subTest(variant=variant):
                base = TailModel(5, cfg, specs, {"return": .02, "mae": .02}, variant).eval()
                reference = base.latent_last(x).clone()
                probe = FrozenTailProbe(base, cfg, specs, {"return": .02, "mae": .02})
                before = probe.heads["mae"].weight.detach().clone()
                optimizer = torch.optim.Adam([p for p in probe.parameters() if p.requires_grad], lr=.01)
                loss, _ = probe.loss(x, targets, masks)
                loss.backward()
                optimizer.step()
                self.assertTrue(probe.assert_backbone_unchanged())
                torch.testing.assert_close(reference, probe.latent_last(x))
                torch.testing.assert_close(reference, base.latent_last(x))
                self.assertFalse(torch.equal(before, probe.heads["mae"].weight))
                self.assertTrue(probe.metadata()["frozen_market_encoder"])

    def test_hurdle_uses_conditional_targets_and_correct_unconditional_mean(self):
        cfg, specs, x, targets, masks = self.fixture()
        model = HurdleTailModel(5, cfg, specs, {"return": .02, "mae": .02}).eval()
        transformed, available = model._conditional_targets(targets, masks)
        self.assertEqual(available["mae"].ravel().tolist(), [False, True, False, True])
        self.assertEqual(transformed["mae__occurs"].ravel().tolist(), [0., 1., 0., 1.])
        self.assertTrue(available["mae__occurs"].all())
        native = model.model.predict_last(x)
        combined = model.predict_last(x)
        torch.testing.assert_close(combined["mae"], native["mae"] * native["mae__occurs"])
        self.assertEqual(set(combined), {"return", "mae"})
        loss, _ = model.loss(x, targets, masks)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(model.supervised_gradient_coverage().values()))

    def test_zero_event_cases_are_censored_for_magnitude_but_train_occurrence(self):
        cfg, specs, x, targets, masks = self.fixture()
        targets["mae"][:] = 0
        model = HurdleTailModel(5, cfg, specs, {"return": .02, "mae": .02})
        loss, _ = model.loss(x, targets, masks)
        loss.backward()
        self.assertIsNone(model.model.heads["mae"].weight.grad)
        self.assertGreater(model.model.heads["mae__occurs"].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.predict_last(x)["mae"]).all())


if __name__ == "__main__":
    unittest.main()
