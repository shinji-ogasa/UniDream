import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

from unidream.experiments.wm_volatility_classification import (
    DEFAULT_CONFIG, audit_oof_inputs, load_config, selection_key,
    tabular_design, tabular_scaler,
)


class ClassificationContractTest(unittest.TestCase):
    def test_tabular_keeps_all_raw_and_market_history_and_train_scaler(self):
        x=np.arange(2*256*63,dtype=np.float32).reshape(2,256,63)
        result=tabular_design(x)
        self.assertEqual(result.shape,(2,4908))
        np.testing.assert_array_equal(result[:,:4864],x[...,:19].reshape(2,-1))
        np.testing.assert_array_equal(result[:,4864:],x[:,-1,19:])
        mean=np.arange(63,dtype=np.float32);std=np.arange(63,dtype=np.float32)+1
        expanded_mean,expanded_std=tabular_scaler({'mean':mean,'std':std})
        expected=tabular_design((x-mean)/std)
        np.testing.assert_array_equal((result-expanded_mean)/expanded_std,expected)

    def test_selection_prefers_balanced_accuracy_then_probability_loss(self):
        self.assertGreater(selection_key({'balanced_accuracy':.4,'log_loss':2}),
                           selection_key({'balanced_accuracy':.39,'log_loss':.1}))
        self.assertGreater(selection_key({'balanced_accuracy':.4,'log_loss':1}),
                           selection_key({'balanced_accuracy':.4,'log_loss':2}))

    def test_test_selection_cannot_be_enabled_by_config(self):
        cfg=yaml.safe_load(DEFAULT_CONFIG.read_text())
        cfg['test_selection_allowed']=True
        with TemporaryDirectory() as folder:
            path=Path(folder)/'config.yaml'; path.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError): load_config(path)

    def test_oof_requires_exact_support_timestamps_and_past_selection(self):
        index=np.arange(6,dtype=np.int64)*900_000_000_000
        frame=pd.DataFrame({'decision_id':[2,3,4], 'timestamp':pd.to_datetime(index[[2,3,4]],utc=True),'quarter_id':[0,0,0]})
        info={'quarter_id':0,'n':3,'forecast_decision_strictly_after_model_selection_end':True,
              'forecast_quarter_labels_read':False,'model_selection_end_ns':int(index[1])}
        audit_oof_inputs(frame,index,np.array([2,3,4]),[info])
        for changed in ({'model_selection_end_ns':int(index[2])}, {'forecast_quarter_labels_read':True}):
            with self.assertRaises(ValueError): audit_oof_inputs(frame,index,np.array([2,3,4]),[info|changed])
        with self.assertRaises(ValueError): audit_oof_inputs(frame,index,np.array([2,3,5]),[info])
        altered=frame.copy(); altered.loc[0,'timestamp']=pd.Timestamp(index[1],tz='UTC')
        with self.assertRaises(ValueError): audit_oof_inputs(altered,index,np.array([2,3,4]),[info])
