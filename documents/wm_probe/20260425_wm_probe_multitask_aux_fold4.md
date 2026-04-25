# TransformerWM Probe Report

Config: `configs/wm_probe_multitask_aux_s007.yaml`
Checkpoint dir: `checkpoints/wm_probe_multitask_aux_s007`
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
- `return_h1`: rank_ic=+0.0046, direction_acc=+0.5042, decile_spread=+0.0001
- `vol_h1`: rank_ic=-0.0103, direction_acc=+0.9972, decile_spread=-0.0000, event_auc=+0.4890
- `drawdown_risk_h1`: rank_ic=+0.0364, direction_acc=+0.9888, decile_spread=+0.0007, event_auc=+0.5703
- `return_h4`: rank_ic=+0.0012, direction_acc=+0.5009, decile_spread=+0.0000
- `vol_h4`: rank_ic=+0.3720, direction_acc=+1.0000, decile_spread=+0.0016, event_auc=+0.6840
- `drawdown_risk_h4`: rank_ic=+0.2125, direction_acc=+0.9992, decile_spread=+0.0021, event_auc=+0.6291
- `return_h8`: rank_ic=+0.0121, direction_acc=+0.5072, decile_spread=-0.0000
- `vol_h8`: rank_ic=+0.3884, direction_acc=+1.0000, decile_spread=+0.0016, event_auc=+0.6516
- `drawdown_risk_h8`: rank_ic=+0.2628, direction_acc=+0.9995, decile_spread=+0.0031, event_auc=+0.6097
- `return_h16`: rank_ic=+0.0053, direction_acc=+0.5017, decile_spread=+0.0006
- `vol_h16`: rank_ic=+0.3528, direction_acc=+0.9998, decile_spread=+0.0015, event_auc=+0.6443
- `drawdown_risk_h16`: rank_ic=+0.2384, direction_acc=+0.9997, decile_spread=+0.0042, event_auc=+0.5729
- `return_h32`: rank_ic=-0.0342, direction_acc=+0.4920, decile_spread=-0.0012
- `vol_h32`: rank_ic=+0.3145, direction_acc=+1.0000, decile_spread=+0.0014, event_auc=+0.6443
- `drawdown_risk_h32`: rank_ic=+0.1838, direction_acc=+0.9997, decile_spread=+0.0047, event_auc=+0.5815
- `teacher_underweight`: accuracy=+0.5186, balanced_accuracy=+0.4977, auc=+0.4995
- `teacher_class`: accuracy=+0.5185, balanced_accuracy=+0.4975, auc=+0.4998
- `recovery_h16`: accuracy=+0.5873, balanced_accuracy=+0.5086, auc=+0.5241
- `action_advantage_h16`: top1_accuracy=+0.5095, top2_accuracy=+0.5095, chosen_minus_benchmark_adv=-0.0005, pred_long_rate=+0.0000, pred_underweight_rate=+0.4190

##### raw_plus_latent
- `return_h1`: rank_ic=+0.0095, direction_acc=+0.5001, decile_spread=+0.0001
- `vol_h1`: rank_ic=-0.0289, direction_acc=+0.9934, decile_spread=-0.0000, event_auc=+0.4675
- `drawdown_risk_h1`: rank_ic=+0.0454, direction_acc=+0.9881, decile_spread=+0.0008, event_auc=+0.5783
- `return_h4`: rank_ic=+0.0063, direction_acc=+0.5012, decile_spread=+0.0004
- `vol_h4`: rank_ic=+0.4146, direction_acc=+1.0000, decile_spread=+0.0018, event_auc=+0.7095
- `drawdown_risk_h4`: rank_ic=+0.2382, direction_acc=+0.9994, decile_spread=+0.0027, event_auc=+0.6465
- `return_h8`: rank_ic=-0.0132, direction_acc=+0.4996, decile_spread=-0.0003
- `vol_h8`: rank_ic=+0.4268, direction_acc=+1.0000, decile_spread=+0.0019, event_auc=+0.6755
- `drawdown_risk_h8`: rank_ic=+0.2828, direction_acc=+0.9999, decile_spread=+0.0037, event_auc=+0.6208
- `return_h16`: rank_ic=+0.0168, direction_acc=+0.5046, decile_spread=+0.0016
- `vol_h16`: rank_ic=+0.3910, direction_acc=+1.0000, decile_spread=+0.0018, event_auc=+0.6649
- `drawdown_risk_h16`: rank_ic=+0.2813, direction_acc=+0.9999, decile_spread=+0.0056, event_auc=+0.5991
- `return_h32`: rank_ic=-0.0324, direction_acc=+0.4912, decile_spread=+0.0004
- `vol_h32`: rank_ic=+0.3311, direction_acc=+1.0000, decile_spread=+0.0014, event_auc=+0.6460
- `drawdown_risk_h32`: rank_ic=+0.2006, direction_acc=+1.0000, decile_spread=+0.0052, event_auc=+0.5834
- `teacher_underweight`: accuracy=+0.5202, balanced_accuracy=+0.4943, auc=+0.4964
- `teacher_class`: accuracy=+0.5202, balanced_accuracy=+0.4945, auc=+0.4965
- `recovery_h16`: accuracy=+0.6035, balanced_accuracy=+0.5189, auc=+0.5328
- `action_advantage_h16`: top1_accuracy=+0.5153, top2_accuracy=+0.5153, chosen_minus_benchmark_adv=-0.0004, pred_long_rate=+0.0000, pred_underweight_rate=+0.4278

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
- `return_h1`: rank_ic=-0.0084, direction_acc=+0.4959, decile_spread=-0.0001
- `vol_h1`: rank_ic=+0.0337, direction_acc=+0.9984, decile_spread=+0.0000, event_auc=+0.5330
- `drawdown_risk_h1`: rank_ic=+0.0229, direction_acc=+0.9689, decile_spread=+0.0004, event_auc=+0.5525
- `return_h4`: rank_ic=+0.0147, direction_acc=+0.5070, decile_spread=+0.0004
- `vol_h4`: rank_ic=+0.2885, direction_acc=+0.9994, decile_spread=+0.0011, event_auc=+0.6365
- `drawdown_risk_h4`: rank_ic=+0.1557, direction_acc=+0.9961, decile_spread=+0.0013, event_auc=+0.5911
- `return_h8`: rank_ic=+0.0093, direction_acc=+0.5014, decile_spread=+0.0005
- `vol_h8`: rank_ic=+0.2934, direction_acc=+0.9990, decile_spread=+0.0010, event_auc=+0.6203
- `drawdown_risk_h8`: rank_ic=+0.1803, direction_acc=+0.9994, decile_spread=+0.0018, event_auc=+0.5969
- `return_h16`: rank_ic=+0.0355, direction_acc=+0.5123, decile_spread=+0.0007
- `vol_h16`: rank_ic=+0.2502, direction_acc=+0.9998, decile_spread=+0.0009, event_auc=+0.6083
- `drawdown_risk_h16`: rank_ic=+0.1497, direction_acc=+0.9989, decile_spread=+0.0027, event_auc=+0.5703
- `return_h32`: rank_ic=+0.0393, direction_acc=+0.5135, decile_spread=+0.0010
- `vol_h32`: rank_ic=+0.1997, direction_acc=+0.9995, decile_spread=+0.0007, event_auc=+0.6185
- `drawdown_risk_h32`: rank_ic=+0.0768, direction_acc=+0.9987, decile_spread=+0.0008, event_auc=+0.5040
- `teacher_underweight`: accuracy=+0.5235, balanced_accuracy=+0.5233, auc=+0.5300
- `teacher_class`: accuracy=+0.5225, balanced_accuracy=+0.5227, auc=+0.5293
- `recovery_h16`: accuracy=+0.5293, balanced_accuracy=+0.5282, auc=+0.5329
- `action_advantage_h16`: top1_accuracy=+0.5213, top2_accuracy=+0.5213, chosen_minus_benchmark_adv=-0.0001, pred_long_rate=+0.0000, pred_underweight_rate=+0.4546

##### raw_plus_latent
- `return_h1`: rank_ic=+0.0143, direction_acc=+0.5016, decile_spread=-0.0000
- `vol_h1`: rank_ic=+0.0371, direction_acc=+0.9977, decile_spread=+0.0000, event_auc=+0.5370
- `drawdown_risk_h1`: rank_ic=+0.0434, direction_acc=+0.9721, decile_spread=+0.0005, event_auc=+0.5673
- `return_h4`: rank_ic=+0.0097, direction_acc=+0.4954, decile_spread=+0.0006
- `vol_h4`: rank_ic=+0.3334, direction_acc=+0.9994, decile_spread=+0.0013, event_auc=+0.6612
- `drawdown_risk_h4`: rank_ic=+0.1922, direction_acc=+0.9961, decile_spread=+0.0017, event_auc=+0.6174
- `return_h8`: rank_ic=+0.0301, direction_acc=+0.5050, decile_spread=+0.0012
- `vol_h8`: rank_ic=+0.3476, direction_acc=+0.9994, decile_spread=+0.0013, event_auc=+0.6510
- `drawdown_risk_h8`: rank_ic=+0.2194, direction_acc=+0.9989, decile_spread=+0.0026, event_auc=+0.6277
- `return_h16`: rank_ic=+0.0275, direction_acc=+0.4997, decile_spread=+0.0015
- `vol_h16`: rank_ic=+0.2943, direction_acc=+1.0000, decile_spread=+0.0011, event_auc=+0.6269
- `drawdown_risk_h16`: rank_ic=+0.1853, direction_acc=+0.9991, decile_spread=+0.0031, event_auc=+0.5904
- `return_h32`: rank_ic=+0.0520, direction_acc=+0.5169, decile_spread=+0.0031
- `vol_h32`: rank_ic=+0.2072, direction_acc=+1.0000, decile_spread=+0.0008, event_auc=+0.6151
- `drawdown_risk_h32`: rank_ic=+0.0930, direction_acc=+0.9994, decile_spread=+0.0009, event_auc=+0.5205
- `teacher_underweight`: accuracy=+0.5290, balanced_accuracy=+0.5286, auc=+0.5317
- `teacher_class`: accuracy=+0.5303, balanced_accuracy=+0.5295, auc=+0.5323
- `recovery_h16`: accuracy=+0.5234, balanced_accuracy=+0.5248, auc=+0.5371
- `action_advantage_h16`: top1_accuracy=+0.5099, top2_accuracy=+0.5099, chosen_minus_benchmark_adv=-0.0001, pred_long_rate=+0.0000, pred_underweight_rate=+0.4460
