import unittest
import numpy as np
import pandas as pd
from unidream.experiments.oracle_risk_calibration import corrected_quantile, scale_and_bias, trailing_variances, interval_targets


class RiskCalibrationTests(unittest.TestCase):
    def test_finite_sample_rank_is_not_interpolated(self):
        self.assertEqual(corrected_quantile(np.arange(10.),.9),9.)
        with self.assertRaises(ValueError):corrected_quantile(np.arange(3.),.9)

    def test_scale_minimizes_qlike_and_bias_is_disjoint_input(self):
        y=np.array([1.,3.,2.]);p=np.array([.5,1.,1.])
        b,c=scale_and_bias([.1,.3,.2],y,[0.,.1,.1],p)
        self.assertAlmostEqual(b, .4/3)
        f=lambda z:np.mean(np.log(z*p)+y/(z*p))
        self.assertLess(f(c),f(c*.9));self.assertLess(f(c),f(c*1.1))

    def test_risk_features_shifted_full_history_and_future_invariant(self):
        index=pd.date_range('2020-01-01',periods=3000,freq='15min',tz='UTC')
        close=np.exp(np.arange(3000)*.001)
        bars=pd.DataFrame({'close':close},index=index)
        before=trailing_variances(bars)
        self.assertTrue(before['24'].iloc[:25].isna().all())
        self.assertAlmostEqual(before['24'].iloc[25],24e-6)
        bars.iloc[2900:,0]*=2
        after=trailing_variances(bars)
        np.testing.assert_allclose(before.iloc[:2901],after.iloc[:2901],equal_nan=True)

    def test_interval_gate_respects_ambiguity_and_missing(self):
        contract={'one_way_cost':.00055,'borrow_annual':.1}
        got=interval_targets(np.array([.05,-.05,0,np.nan]),np.full(4,.0001),2.,contract)
        np.testing.assert_allclose(got[:3],[1.12,.5,1.]);self.assertTrue(np.isnan(got[-1]))


if __name__=='__main__':unittest.main()
