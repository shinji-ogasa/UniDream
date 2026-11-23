import unittest

import numpy as np

from unidream.experiments.wm_model_comparison import (
    diagnostic_scores, prediction_frame, tabular_design, tabular_scaler, update_patience,
)


class ComparisonProtocolTests(unittest.TestCase):
    def test_tabular_scaling_matches_window_scaling_without_extra_history(self):
        rng = np.random.default_rng(8)
        raw = rng.normal(size=(3, 256, 63)).astype(np.float32)
        mean = rng.normal(size=63).astype(np.float32)
        std = rng.uniform(.1, 3, size=63).astype(np.float32)
        positions = [0, 192, 240, 252]
        expanded_mean, expanded_std = tabular_scaler({'mean':mean,'std':std},256,positions)
        direct = tabular_design((raw-mean)/std,positions)
        cached = (tabular_design(raw,positions)-expanded_mean)/expanded_std
        self.assertEqual(direct.shape,(3,1394))
        np.testing.assert_array_equal(direct,cached)

    def test_patience_accumulates_small_improvements_from_anchor(self):
        anchor, step = update_patience(1.,100,float('inf'),0,.0005)
        self.assertEqual((anchor,step),(1.,100))
        anchor, step = update_patience(.9997,200,anchor,step,.0005)
        self.assertEqual((anchor,step),(1.,100))
        anchor, step = update_patience(.9994,300,anchor,step,.0005)
        self.assertEqual((anchor,step),(.9994,300))

    def test_patience_rejects_nonfinite_validation_score(self):
        with self.assertRaises(ValueError): update_patience(float('nan'),100,1.,0,.0005)

    def test_selection_uses_four_hour_scaled_return_and_variance_only(self):
        arrays = {'returns':np.array([[99.,.02,99.]]),'variance':np.array([[99.,.03,99.]])}
        prediction = {'return':np.array([[-99.,.01,-99.]]),'variance':np.array([[.1,.03,.1]])}
        result = diagnostic_scores(arrays,np.array([0]),prediction,.01)
        self.assertAlmostEqual(result['score'],1.)
        self.assertAlmostEqual(result['variance_qlike'],0.)

    def test_prediction_frame_does_not_accept_invalid_scored_predictions(self):
        arrays = {'index_ns':np.array([0]),'returns':np.zeros((1,3)),'variance':np.ones((1,3))}
        for value in [0.,-1.,float('nan')]:
            with self.assertRaises(ValueError):
                prediction_frame(arrays,np.array([0]),0,'bad',0,'test',None,np.array([value]))


if __name__ == '__main__': unittest.main()
