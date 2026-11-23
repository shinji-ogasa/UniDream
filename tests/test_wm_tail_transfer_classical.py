import inspect
import pickle
import unittest

import numpy as np
import torch

from unidream.experiments.wm_tail_transfer_classical import (
    fit_bundle, predict_bundle, fz0_numpy, fz0_gradient_hessian,
)


class TailClassicalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(80, 3)).astype(np.float32)
        xv = rng.normal(size=(24, 3)).astype(np.float32)
        specs = [
            dict(name="r", kind="real", shape=[1], task="T01", levels=None),
            dict(name="event", kind="binary", shape=[1], task="T02", levels=None),
            dict(name="rare", kind="binary", shape=[1], task="T02", levels=None),
            dict(name="class", kind="categorical", shape=[1], task="T03", levels=None, classes=3),
            dict(name="rv", kind="variance", shape=[1], task="T07", levels=None),
            dict(name="quantile", kind="quantile", shape=[1, 3], task="T09", levels=[.05, .5, .95]),
            dict(name="ve", kind="var_es", shape=[1, 1], task="T10", levels=[.05]),
            dict(name="hazard", kind="hazard", shape=[4, 2], task="T11", levels=None, classes=3),
        ]

        def labels(features):
            r = (.01 * features[:, 0] - .005 * features[:, 1])[:, None]
            values = dict(r=r, event=(features[:, :1] > 0).astype(float), rare=np.zeros((len(features), 1)),
                          rv=r ** 2 + 1e-5, quantile=np.repeat(r[:, :, None], 3, axis=-1), ve=r[:, :, None])
            values["class"] = np.digitize(features[:, :1], [-.5, .5]).astype(float)
            values["hazard"] = np.zeros((len(features), 4, 2))
            values["hazard"][features[:, 0] > .5, 2, 0] = 2
            values["hazard"][features[:, 0] < -.5, 1, 1] = 1
            masks = {name: np.ones_like(value, dtype=bool) for name, value in values.items()}
            masks["hazard"][features[:, 0] > .5, 3:, 0] = False
            masks["hazard"][features[:, 0] < -.5, 2:, 1] = False
            for name in values:
                values[name][~masks[name]] = np.nan
            return values, masks

        y, mask = labels(x)
        yv, mv = labels(xv)
        cfg = {"classical": {"max_train_rows": 64, "linear_steps": 16,
            "linear_validation_every": 4, "linear_patience": 12, "linear_batch_size": 32,
            "linear_alpha": .1, "lightgbm_rounds": 12, "lightgbm_patience": 4,
            "lightgbm_min_data_leaf": 4, "lightgbm_leaves": 5, "lightgbm_l2": 1.,
            "hazard_max_val_risksets": 96, "threads": 1}}
        scales = {spec["name"]: (1e-4 if spec["kind"] == "variance" else .01) for spec in specs}
        return x, y, mask, xv, yv, mv, cfg, specs, scales

    def test_fz0_exact_gradient_against_finite_differences(self):
        rng = np.random.default_rng(6)
        raw = rng.normal(size=(20, 2))
        actual = rng.normal(size=20)
        gradient, hessian = fz0_gradient_hessian(raw, actual, .05)
        for axis in range(2):
            plus, minus = raw.copy(), raw.copy()
            plus[:, axis] += 1e-5
            minus[:, axis] -= 1e-5
            finite = (fz0_numpy(plus, actual, .05)[0] - fz0_numpy(minus, actual, .05)[0]) / 2e-5
            np.testing.assert_allclose(gradient[:, axis], finite, rtol=2e-6, atol=2e-6)
        self.assertTrue(np.isfinite(hessian).all())
        self.assertTrue((hessian > 0).all())

    def test_native_affine_fit_shapes_calibration_and_serialization(self):
        args = self.fixture()
        bundle = fit_bundle(*args[:7], specs=args[7], scales=args[8], family="linear")
        prediction = predict_bundle(pickle.loads(pickle.dumps(bundle)), args[3])
        self.assertFalse(bundle["metadata"]["true_joint_representation"])
        self.assertTrue(bundle["metadata"]["no_test_used"])
        self.assertEqual(bundle["metadata"]["train_rows"], 64)
        self.assertGreater(bundle["model"]["metadata"]["selected_step"], 0)
        self.check_prediction(prediction, len(args[3]))
        # At least this ordinary affine regression is learned, not a placeholder.
        self.assertGreater(np.corrcoef(prediction["r"].ravel(), args[4]["r"].ravel())[0, 1], .5)

    def test_native_lightgbm_fz0_and_competing_risk_fit(self):
        args = self.fixture()
        bundle = fit_bundle(*args[:7], specs=args[7], scales=args[8], family="lightgbm")
        prediction = predict_bundle(pickle.loads(pickle.dumps(bundle)), args[3])
        self.check_prediction(prediction, len(args[3]))
        self.assertTrue((prediction["rare"] == 0).all())
        self.assertEqual(bundle["model"]["fitted"]["hazard"]["kind"], "hazard_pool")
        self.assertIn("proper FZ0", bundle["model"]["metadata"]["var_es"])
        fzhead = bundle["model"]["fitted"]["ve"]["heads"][0]["head"]
        self.assertEqual(fzhead["native_kind"], "var_es")
        self.assertGreater(fzhead["selected_iteration"], 0)

    def test_focal_and_log_variance_use_their_native_links(self):
        args = list(self.fixture())
        args[6]["loss"] = {"classification_loss": "focal", "variance_loss": "log_mse"}
        specs = [s for s in args[7] if s["name"] in ("event", "rv")]
        bundle = fit_bundle(*args[:7], specs=specs, scales=args[8], family="lightgbm")
        prediction = predict_bundle(bundle, args[3])
        self.assertTrue(((prediction["event"] > 0) & (prediction["event"] < 1)).all())
        self.assertTrue((prediction["rv"] > 0).all())
        self.assertTrue(bundle["model"]["fitted"]["event"]["heads"][0]["head"]["custom_binary"])
        self.assertTrue(bundle["model"]["fitted"]["rv"]["heads"][0]["head"]["log_variance"])

    def test_unobserved_output_is_explicitly_unavailable_not_fabricated_fit(self):
        args = list(self.fixture())
        args[2]["rare"][:] = False
        args[5]["rare"][:] = False
        specs = [s for s in args[7] if s["name"] in ("r", "rare")]
        bundle = fit_bundle(*args[:7], specs=specs, scales=args[8], family="lightgbm")
        metadata = bundle["model"]["metadata"]["native_heads"]
        unavailable = [row for row in metadata if row["name"] == "rare"]
        self.assertTrue(unavailable[0]["unavailable_fit"])
        self.assertEqual(unavailable[0]["train_observed"], 0)
        self.assertTrue(np.isfinite(predict_bundle(bundle, args[3])["rare"]).all())

    def check_prediction(self, p, n):
        self.assertEqual(p["r"].shape, (n, 1))
        self.assertEqual(p["class"].shape, (n, 1, 3))
        self.assertEqual(p["ve"].shape, (n, 1, 1, 2))
        self.assertEqual(p["hazard"].shape, (n, 4, 2, 3))
        for values in p.values():
            self.assertTrue(np.isfinite(values).all())
        np.testing.assert_allclose(p["class"].sum(-1), 1., atol=1e-6)
        np.testing.assert_allclose(p["hazard"].sum(-1), 1., atol=1e-6)
        self.assertTrue((p["ve"][..., 1] < p["ve"][..., 0]).all())
        self.assertTrue((p["ve"][..., 0] < 0).all())
        self.assertTrue((np.diff(p["quantile"], axis=-1) >= 0).all())

    def test_observed_nonfinite_is_error_masked_nonfinite_is_allowed(self):
        args = list(self.fixture())
        args[1]["r"] = args[1]["r"].copy()
        args[1]["r"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "observed targets"):
            fit_bundle(*args[:7], specs=args[7], scales=args[8])
        args[2]["r"][0, 0] = False
        result = fit_bundle(*args[:7], specs=args[7], scales=args[8])
        self.assertTrue(result["metadata"]["no_test_used"])
        self.assertFalse(any("test" in p for p in inspect.signature(fit_bundle).parameters))


if __name__ == "__main__":
    unittest.main()
