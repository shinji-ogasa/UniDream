"""Exact coverage semantics while synchronizing accelerator results once."""
import unittest
from unittest.mock import patch

import torch

from unidream.experiments.wm_tail_transfer_models import TailModel
from unidream.experiments.wm_tail_transfer_probes import FrozenTailProbe, HurdleTailModel


def fixture(kind, device):
    cfg = {"model": {"context": 4, "d_model": 8, "n_heads": 2, "n_layers": 1,
                      "d_ff": 16, "encoder_hidden": 8, "dropout": 0.}}
    specs = [dict(name=name, kind="positive" if name == "mae" else "real", shape=[2],
                  task="T05" if name == "mae" else "T01", levels=None)
             for name in ("finite", "zero_row", "nan", "inf", "negative_inf", "abs_sum", "none", "mae", "noncontiguous")]
    scales = {s["name"]: .02 for s in specs}
    if kind == "hurdle":
        model = HurdleTailModel(5, cfg, specs, scales)
        native = model.model
    else:
        base = TailModel(5, cfg, specs, scales)
        model = FrozenTailProbe(base, cfg, specs, scales) if kind == "frozen" else base
        native = model
    model.to(device)
    for name, head in native.heads.items():
        head.weight.grad = torch.ones_like(head.weight)
        if name == "zero_row":
            head.weight.grad[0].zero_()
        elif name == "nan":
            head.weight.grad[0, 0] = float("nan")
        elif name == "inf":
            head.weight.grad[0, 0] = float("inf")
        elif name == "negative_inf":
            head.weight.grad[0, 0] = -float("inf")
        elif name == "abs_sum":
            head.weight.grad[:, ::2] = -1.
        elif name == "none":
            head.weight.grad = None
        elif name == "mae":
            head.weight.grad.zero_()
        elif name == "noncontiguous":
            head.weight.grad = torch.ones((head.weight.shape[1], head.weight.shape[0]), device=device).t()
            assert not head.weight.grad.is_contiguous()
    return model, native


def previous_semantics(native):
    result = {}
    for spec in native.specs:
        gradient = native.heads[spec["name"]].weight.grad
        result[spec["name"]] = bool(gradient is not None and torch.isfinite(gradient).all()
                                   and (gradient.abs().sum(1) > 0).all())
    return result


class TailTransferGradientCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def exercise(self, device):
        results = []
        for kind in ("standard", "frozen", "hurdle"):
            with self.subTest(device=device, kind=kind):
                model, native = fixture(kind, device)
                expected = previous_semantics(native)
                original_cpu = torch.Tensor.cpu
                transfers = []
                def counted_cpu(tensor, *args, **kwargs):
                    transfers.append((tensor.device.type, tuple(tensor.shape), tensor.dtype))
                    return original_cpu(tensor, *args, **kwargs)
                cpu_rng = torch.get_rng_state().clone()
                with patch.object(torch.Tensor, "cpu", new=counted_cpu):
                    actual = model.supervised_gradient_coverage()
                self.assertEqual(actual, expected)
                self.assertEqual(list(actual), [s["name"] for s in native.specs])
                self.assertTrue(all(type(v) is bool for v in actual.values()))
                self.assertEqual(transfers, [(device, (len(native.specs)-1,), torch.bool)])
                self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                self.assertTrue(actual["finite"])
                self.assertTrue(actual["abs_sum"])
                self.assertTrue(actual["noncontiguous"])
                self.assertFalse(actual["zero_row"])
                self.assertFalse(actual["nan"])
                self.assertFalse(actual["inf"])
                self.assertFalse(actual["negative_inf"])
                self.assertFalse(actual["none"])
                self.assertFalse(actual["mae"])
                if kind == "hurdle":
                    self.assertTrue(actual["mae__occurs"])
                results.append(actual)
        return results

    def test_cpu_matches_previous_none_nonfinite_and_every_row_nonzero_semantics(self):
        self.exercise("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_matches_previous_semantics_with_one_cpu_transfer(self):
        self.assertEqual(self.exercise("cpu"), self.exercise("mps"))

    def test_all_missing_gradients_return_false_without_any_transfer(self):
        model, native = fixture("standard", "cpu")
        for head in native.heads.values():
            head.weight.grad = None
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no transfer expected")):
            result = model.supervised_gradient_coverage()
        self.assertEqual(result, {s["name"]: False for s in native.specs})


if __name__ == "__main__":
    unittest.main()
