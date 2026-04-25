# TransformerWM Probe Report

Config: `configs/wm_probe_multitask_regime_aux_s007.yaml`
Checkpoint dir: `checkpoints/wm_probe_multitask_regime_aux_s007`
Folds: `4`

## 判定基準

- return は `rank_ic`, `direction_acc`, `decile_spread` を重視する。
- risk/vol は `event_auc` と decile bucket の単調性を重視する。
- teacher/action は AUC / balanced accuracy / action advantage top-k を見る。

## Results

### Fold 4

#### val

##### raw
- `return_h1`: rank_ic=+0.0208, direction_acc=+0.5010, decile_spread=+0.0002
- `vol_h1`: rank_ic=+0.0474, direction_acc=+1.0000, decile_spread=+0.0000, event_auc=+0.5374
- `drawdown_risk_h1`: rank_ic=+0.0720, direction_acc=+1.0000, decile_spread=+0.0011, event_auc=+0.6246
- `return_h4`: rank_ic=-0.0057, direction_acc=+0.4903, decile_spread=+0.0005
- `vol_h4`: rank_ic=+0.4771, direction_acc=+1.0000, decile_spread=+0.0022, event_auc=+0.7320
- `drawdown_risk_h4`: rank_ic=+0.2974, direction_acc=+1.0000, decile_spread=+0.0034, event_auc=+0.6825
- `return_h8`: rank_ic=-0.0074, direction_acc=+0.4856, decile_spread=+0.0009
- `vol_h8`: rank_ic=+0.4902, direction_acc=+1.0000, decile_spread=+0.0022, event_auc=+0.7055
- `drawdown_risk_h8`: rank_ic=+0.3464, direction_acc=+1.0000, decile_spread=+0.0049, event_auc=+0.6693
- `return_h16`: rank_ic=+0.0331, direction_acc=+0.5097, decile_spread=+0.0036
- `vol_h16`: rank_ic=+0.4565, direction_acc=+1.0000, decile_spread=+0.0020, event_auc=+0.6947
- `drawdown_risk_h16`: rank_ic=+0.3356, direction_acc=+1.0000, decile_spread=+0.0066, event_auc=+0.6384
- `return_h32`: rank_ic=+0.0667, direction_acc=+0.5211, decile_spread=+0.0083
- `vol_h32`: rank_ic=+0.3961, direction_acc=+1.0000, decile_spread=+0.0018, event_auc=+0.6735
- `drawdown_risk_h32`: rank_ic=+0.3011, direction_acc=+1.0000, decile_spread=+0.0083, event_auc=+0.6259
- `teacher_underweight`: accuracy=+0.5592, balanced_accuracy=+0.5209, auc=+0.5427
- `teacher_class`: accuracy=+0.5592, balanced_accuracy=+0.5209, auc=+0.5427
- `recovery_h16`: accuracy=+0.6098, balanced_accuracy=+0.5114, auc=+0.5489
- `action_advantage_h16`: top1_accuracy=+0.5207, top2_accuracy=+0.5207, chosen_minus_benchmark_adv=-0.0003, pred_long_rate=+0.0000, pred_underweight_rate=+0.2886

##### latent_zh
- `return_h1`: rank_ic=+0.0010, direction_acc=+0.4992, decile_spread=-0.0000
- `vol_h1`: rank_ic=-0.0087, direction_acc=+0.9948, decile_spread=+0.0000, event_auc=+0.4943
- `drawdown_risk_h1`: rank_ic=+0.0448, direction_acc=+0.9860, decile_spread=+0.0006, event_auc=+0.5764
- `return_h4`: rank_ic=-0.0016, direction_acc=+0.5003, decile_spread=-0.0002
- `vol_h4`: rank_ic=+0.3739, direction_acc=+0.9999, decile_spread=+0.0017, event_auc=+0.6746
- `drawdown_risk_h4`: rank_ic=+0.2292, direction_acc=+0.9968, decile_spread=+0.0022, event_auc=+0.6337
- `return_h8`: rank_ic=-0.0158, direction_acc=+0.4835, decile_spread=-0.0002
- `vol_h8`: rank_ic=+0.3887, direction_acc=+0.9999, decile_spread=+0.0017, event_auc=+0.6500
- `drawdown_risk_h8`: rank_ic=+0.2599, direction_acc=+0.9991, decile_spread=+0.0037, event_auc=+0.6068
- `return_h16`: rank_ic=-0.0382, direction_acc=+0.4857, decile_spread=-0.0010
- `vol_h16`: rank_ic=+0.3538, direction_acc=+1.0000, decile_spread=+0.0015, event_auc=+0.6334
- `drawdown_risk_h16`: rank_ic=+0.2272, direction_acc=+0.9993, decile_spread=+0.0047, event_auc=+0.5773
- `return_h32`: rank_ic=-0.0656, direction_acc=+0.4802, decile_spread=-0.0037
- `vol_h32`: rank_ic=+0.3158, direction_acc=+1.0000, decile_spread=+0.0014, event_auc=+0.6298
- `drawdown_risk_h32`: rank_ic=+0.2043, direction_acc=+0.9995, decile_spread=+0.0053, event_auc=+0.5855
- `teacher_underweight`: accuracy=+0.5054, balanced_accuracy=+0.4922, auc=+0.4866
- `teacher_class`: accuracy=+0.5061, balanced_accuracy=+0.4934, auc=+0.4861
- `recovery_h16`: accuracy=+0.5619, balanced_accuracy=+0.5194, auc=+0.5285
- `action_advantage_h16`: top1_accuracy=+0.4929, top2_accuracy=+0.4929, chosen_minus_benchmark_adv=-0.0007, pred_long_rate=+0.0000, pred_underweight_rate=+0.4605

##### raw_plus_latent
- `return_h1`: rank_ic=+0.0068, direction_acc=+0.5027, decile_spread=-0.0001
- `vol_h1`: rank_ic=-0.0003, direction_acc=+0.9924, decile_spread=+0.0000, event_auc=+0.4957
- `drawdown_risk_h1`: rank_ic=+0.0417, direction_acc=+0.9865, decile_spread=+0.0007, event_auc=+0.5816
- `return_h4`: rank_ic=+0.0080, direction_acc=+0.5034, decile_spread=-0.0000
- `vol_h4`: rank_ic=+0.4022, direction_acc=+1.0000, decile_spread=+0.0018, event_auc=+0.6919
- `drawdown_risk_h4`: rank_ic=+0.2503, direction_acc=+0.9979, decile_spread=+0.0028, event_auc=+0.6529
- `return_h8`: rank_ic=-0.0432, direction_acc=+0.4745, decile_spread=-0.0006
- `vol_h8`: rank_ic=+0.4222, direction_acc=+1.0000, decile_spread=+0.0018, event_auc=+0.6662
- `drawdown_risk_h8`: rank_ic=+0.2783, direction_acc=+0.9995, decile_spread=+0.0035, event_auc=+0.6192
- `return_h16`: rank_ic=-0.0336, direction_acc=+0.4924, decile_spread=-0.0017
- `vol_h16`: rank_ic=+0.3967, direction_acc=+1.0000, decile_spread=+0.0018, event_auc=+0.6567
- `drawdown_risk_h16`: rank_ic=+0.2694, direction_acc=+0.9999, decile_spread=+0.0053, event_auc=+0.5984
- `return_h32`: rank_ic=-0.0615, direction_acc=+0.4796, decile_spread=-0.0019
- `vol_h32`: rank_ic=+0.3352, direction_acc=+1.0000, decile_spread=+0.0014, event_auc=+0.6362
- `drawdown_risk_h32`: rank_ic=+0.2215, direction_acc=+0.9995, decile_spread=+0.0051, event_auc=+0.5895
- `teacher_underweight`: accuracy=+0.5115, balanced_accuracy=+0.4980, auc=+0.4942
- `teacher_class`: accuracy=+0.5127, balanced_accuracy=+0.4989, auc=+0.4945
- `recovery_h16`: accuracy=+0.5732, balanced_accuracy=+0.5184, auc=+0.5333
- `action_advantage_h16`: top1_accuracy=+0.4931, top2_accuracy=+0.4931, chosen_minus_benchmark_adv=-0.0007, pred_long_rate=+0.0000, pred_underweight_rate=+0.4588

#### test

##### raw
- `return_h1`: rank_ic=+0.0263, direction_acc=+0.5073, decile_spread=+0.0002
- `vol_h1`: rank_ic=-0.0010, direction_acc=+1.0000, decile_spread=-0.0000, event_auc=+0.5098
- `drawdown_risk_h1`: rank_ic=+0.0672, direction_acc=+1.0000, decile_spread=+0.0007, event_auc=+0.6135
- `return_h4`: rank_ic=+0.0073, direction_acc=+0.4984, decile_spread=+0.0003
- `vol_h4`: rank_ic=+0.3976, direction_acc=+1.0000, decile_spread=+0.0015, event_auc=+0.6850
- `drawdown_risk_h4`: rank_ic=+0.2742, direction_acc=+1.0000, decile_spread=+0.0022, event_auc=+0.6663
- `return_h8`: rank_ic=+0.0227, direction_acc=+0.4923, decile_spread=+0.0017
- `vol_h8`: rank_ic=+0.4149, direction_acc=+1.0000, decile_spread=+0.0015, event_auc=+0.6751
- `drawdown_risk_h8`: rank_ic=+0.3113, direction_acc=+1.0000, decile_spread=+0.0034, event_auc=+0.6467
- `return_h16`: rank_ic=+0.0430, direction_acc=+0.4989, decile_spread=+0.0033
- `vol_h16`: rank_ic=+0.3597, direction_acc=+1.0000, decile_spread=+0.0014, event_auc=+0.6535
- `drawdown_risk_h16`: rank_ic=+0.2899, direction_acc=+1.0000, decile_spread=+0.0043, event_auc=+0.6316
- `return_h32`: rank_ic=+0.0867, direction_acc=+0.5122, decile_spread=+0.0069
- `vol_h32`: rank_ic=+0.2819, direction_acc=+1.0000, decile_spread=+0.0010, event_auc=+0.6471
- `drawdown_risk_h32`: rank_ic=+0.2233, direction_acc=+1.0000, decile_spread=+0.0038, event_auc=+0.5745
- `teacher_underweight`: accuracy=+0.5098, balanced_accuracy=+0.5259, auc=+0.5444
- `teacher_class`: accuracy=+0.5098, balanced_accuracy=+0.5259, auc=+0.5444
- `recovery_h16`: accuracy=+0.5121, balanced_accuracy=+0.5343, auc=+0.5428
- `action_advantage_h16`: top1_accuracy=+0.5146, top2_accuracy=+0.5146, chosen_minus_benchmark_adv=+0.0000, pred_long_rate=+0.0000, pred_underweight_rate=+0.3450

##### latent_zh
- `return_h1`: rank_ic=-0.0013, direction_acc=+0.4967, decile_spread=+0.0000
- `vol_h1`: rank_ic=+0.0081, direction_acc=+0.9974, decile_spread=-0.0000, event_auc=+0.5126
- `drawdown_risk_h1`: rank_ic=+0.0186, direction_acc=+0.9671, decile_spread=+0.0004, event_auc=+0.5459
- `return_h4`: rank_ic=+0.0030, direction_acc=+0.4999, decile_spread=+0.0001
- `vol_h4`: rank_ic=+0.2923, direction_acc=+0.9971, decile_spread=+0.0011, event_auc=+0.6314
- `drawdown_risk_h4`: rank_ic=+0.1633, direction_acc=+0.9927, decile_spread=+0.0014, event_auc=+0.5966
- `return_h8`: rank_ic=+0.0027, direction_acc=+0.4991, decile_spread=-0.0000
- `vol_h8`: rank_ic=+0.3116, direction_acc=+0.9974, decile_spread=+0.0011, event_auc=+0.6303
- `drawdown_risk_h8`: rank_ic=+0.1862, direction_acc=+0.9956, decile_spread=+0.0020, event_auc=+0.5907
- `return_h16`: rank_ic=+0.0129, direction_acc=+0.5102, decile_spread=+0.0004
- `vol_h16`: rank_ic=+0.2494, direction_acc=+0.9977, decile_spread=+0.0008, event_auc=+0.6059
- `drawdown_risk_h16`: rank_ic=+0.1628, direction_acc=+0.9962, decile_spread=+0.0022, event_auc=+0.5669
- `return_h32`: rank_ic=+0.0005, direction_acc=+0.4903, decile_spread=+0.0003
- `vol_h32`: rank_ic=+0.1728, direction_acc=+0.9976, decile_spread=+0.0005, event_auc=+0.5827
- `drawdown_risk_h32`: rank_ic=+0.0856, direction_acc=+0.9968, decile_spread=+0.0010, event_auc=+0.4982
- `teacher_underweight`: accuracy=+0.5227, balanced_accuracy=+0.5271, auc=+0.5376
- `teacher_class`: accuracy=+0.5228, balanced_accuracy=+0.5275, auc=+0.5376
- `recovery_h16`: accuracy=+0.4864, balanced_accuracy=+0.4973, auc=+0.5085
- `action_advantage_h16`: top1_accuracy=+0.5065, top2_accuracy=+0.5065, chosen_minus_benchmark_adv=-0.0002, pred_long_rate=+0.0000, pred_underweight_rate=+0.4816

##### raw_plus_latent
- `return_h1`: rank_ic=+0.0103, direction_acc=+0.5049, decile_spread=+0.0001
- `vol_h1`: rank_ic=+0.0056, direction_acc=+0.9972, decile_spread=+0.0000, event_auc=+0.5046
- `drawdown_risk_h1`: rank_ic=+0.0279, direction_acc=+0.9694, decile_spread=+0.0003, event_auc=+0.5548
- `return_h4`: rank_ic=+0.0273, direction_acc=+0.5105, decile_spread=+0.0004
- `vol_h4`: rank_ic=+0.3431, direction_acc=+0.9968, decile_spread=+0.0012, event_auc=+0.6580
- `drawdown_risk_h4`: rank_ic=+0.1989, direction_acc=+0.9913, decile_spread=+0.0019, event_auc=+0.6279
- `return_h8`: rank_ic=+0.0072, direction_acc=+0.4997, decile_spread=+0.0006
- `vol_h8`: rank_ic=+0.3411, direction_acc=+0.9981, decile_spread=+0.0012, event_auc=+0.6567
- `drawdown_risk_h8`: rank_ic=+0.2079, direction_acc=+0.9966, decile_spread=+0.0022, event_auc=+0.6102
- `return_h16`: rank_ic=+0.0188, direction_acc=+0.5063, decile_spread=+0.0014
- `vol_h16`: rank_ic=+0.2892, direction_acc=+0.9986, decile_spread=+0.0010, event_auc=+0.6260
- `drawdown_risk_h16`: rank_ic=+0.1891, direction_acc=+0.9969, decile_spread=+0.0030, event_auc=+0.5858
- `return_h32`: rank_ic=+0.0086, direction_acc=+0.4959, decile_spread=+0.0012
- `vol_h32`: rank_ic=+0.1807, direction_acc=+0.9978, decile_spread=+0.0006, event_auc=+0.5838
- `drawdown_risk_h32`: rank_ic=+0.0860, direction_acc=+0.9968, decile_spread=+0.0013, event_auc=+0.5048
- `teacher_underweight`: accuracy=+0.5151, balanced_accuracy=+0.5230, auc=+0.5308
- `teacher_class`: accuracy=+0.5159, balanced_accuracy=+0.5234, auc=+0.5305
- `recovery_h16`: accuracy=+0.4956, balanced_accuracy=+0.5068, auc=+0.5176
- `action_advantage_h16`: top1_accuracy=+0.4997, top2_accuracy=+0.4997, chosen_minus_benchmark_adv=-0.0002, pred_long_rate=+0.0000, pred_underweight_rate=+0.4973
