# TransformerWM Probe Report

Config: `configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007.yaml`
Checkpoint dir: `checkpoints/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007`
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
- `return_h1`: rank_ic=+0.0108, direction_acc=+0.5052, decile_spread=-0.0000
- `vol_h1`: rank_ic=+0.0149, direction_acc=+0.9960, decile_spread=+0.0000, event_auc=+0.5105
- `drawdown_risk_h1`: rank_ic=+0.0443, direction_acc=+0.9898, decile_spread=+0.0004, event_auc=+0.5553
- `return_h4`: rank_ic=+0.0042, direction_acc=+0.5032, decile_spread=+0.0001
- `vol_h4`: rank_ic=+0.2497, direction_acc=+0.9998, decile_spread=+0.0008, event_auc=+0.6077
- `drawdown_risk_h4`: rank_ic=+0.1879, direction_acc=+0.9977, decile_spread=+0.0013, event_auc=+0.5960
- `return_h8`: rank_ic=+0.0270, direction_acc=+0.5042, decile_spread=+0.0004
- `vol_h8`: rank_ic=+0.2698, direction_acc=+0.9999, decile_spread=+0.0010, event_auc=+0.5957
- `drawdown_risk_h8`: rank_ic=+0.2054, direction_acc=+0.9992, decile_spread=+0.0022, event_auc=+0.5888
- `return_h16`: rank_ic=-0.0026, direction_acc=+0.4950, decile_spread=+0.0007
- `vol_h16`: rank_ic=+0.2741, direction_acc=+1.0000, decile_spread=+0.0009, event_auc=+0.6007
- `drawdown_risk_h16`: rank_ic=+0.2078, direction_acc=+0.9995, decile_spread=+0.0026, event_auc=+0.5769
- `return_h32`: rank_ic=+0.0124, direction_acc=+0.5055, decile_spread=+0.0011
- `vol_h32`: rank_ic=+0.2491, direction_acc=+1.0000, decile_spread=+0.0009, event_auc=+0.5726
- `drawdown_risk_h32`: rank_ic=+0.1750, direction_acc=+0.9995, decile_spread=+0.0034, event_auc=+0.5745
- `teacher_underweight`: accuracy=+0.5261, balanced_accuracy=+0.5154, auc=+0.5233
- `teacher_class`: accuracy=+0.5255, balanced_accuracy=+0.5143, auc=+0.5234
- `recovery_h16`: accuracy=+0.5905, balanced_accuracy=+0.5323, auc=+0.5314
- `action_advantage_h16`: top1_accuracy=+0.4989, top2_accuracy=+0.4989, chosen_minus_benchmark_adv=-0.0006, pred_long_rate=+0.0000, pred_underweight_rate=+0.4354

##### raw_plus_latent
- `return_h1`: rank_ic=+0.0198, direction_acc=+0.5034, decile_spread=+0.0003
- `vol_h1`: rank_ic=+0.0036, direction_acc=+0.9950, decile_spread=+0.0000, event_auc=+0.5103
- `drawdown_risk_h1`: rank_ic=+0.0610, direction_acc=+0.9902, decile_spread=+0.0009, event_auc=+0.5865
- `return_h4`: rank_ic=-0.0069, direction_acc=+0.4962, decile_spread=+0.0005
- `vol_h4`: rank_ic=+0.4092, direction_acc=+1.0000, decile_spread=+0.0019, event_auc=+0.7005
- `drawdown_risk_h4`: rank_ic=+0.2464, direction_acc=+0.9995, decile_spread=+0.0028, event_auc=+0.6525
- `return_h8`: rank_ic=-0.0071, direction_acc=+0.4948, decile_spread=+0.0000
- `vol_h8`: rank_ic=+0.4327, direction_acc=+1.0000, decile_spread=+0.0018, event_auc=+0.6798
- `drawdown_risk_h8`: rank_ic=+0.2976, direction_acc=+0.9999, decile_spread=+0.0044, event_auc=+0.6478
- `return_h16`: rank_ic=+0.0159, direction_acc=+0.5002, decile_spread=+0.0004
- `vol_h16`: rank_ic=+0.4069, direction_acc=+1.0000, decile_spread=+0.0016, event_auc=+0.6714
- `drawdown_risk_h16`: rank_ic=+0.2893, direction_acc=+0.9999, decile_spread=+0.0055, event_auc=+0.6187
- `return_h32`: rank_ic=+0.0321, direction_acc=+0.5077, decile_spread=+0.0025
- `vol_h32`: rank_ic=+0.3411, direction_acc=+1.0000, decile_spread=+0.0012, event_auc=+0.6323
- `drawdown_risk_h32`: rank_ic=+0.2411, direction_acc=+1.0000, decile_spread=+0.0050, event_auc=+0.6028
- `teacher_underweight`: accuracy=+0.5361, balanced_accuracy=+0.5234, auc=+0.5332
- `teacher_class`: accuracy=+0.5342, balanced_accuracy=+0.5220, auc=+0.5321
- `recovery_h16`: accuracy=+0.6013, balanced_accuracy=+0.5288, auc=+0.5439
- `action_advantage_h16`: top1_accuracy=+0.5063, top2_accuracy=+0.5063, chosen_minus_benchmark_adv=-0.0004, pred_long_rate=+0.0000, pred_underweight_rate=+0.4435

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
- `return_h1`: rank_ic=-0.0041, direction_acc=+0.4948, decile_spread=-0.0000
- `vol_h1`: rank_ic=-0.0011, direction_acc=+0.9977, decile_spread=+0.0000, event_auc=+0.5120
- `drawdown_risk_h1`: rank_ic=+0.0173, direction_acc=+0.9742, decile_spread=+0.0003, event_auc=+0.5329
- `return_h4`: rank_ic=+0.0024, direction_acc=+0.5015, decile_spread=-0.0001
- `vol_h4`: rank_ic=+0.2019, direction_acc=+0.9987, decile_spread=+0.0007, event_auc=+0.5858
- `drawdown_risk_h4`: rank_ic=+0.1236, direction_acc=+0.9939, decile_spread=+0.0008, event_auc=+0.5641
- `return_h8`: rank_ic=-0.0136, direction_acc=+0.4939, decile_spread=-0.0002
- `vol_h8`: rank_ic=+0.2188, direction_acc=+0.9995, decile_spread=+0.0008, event_auc=+0.5770
- `drawdown_risk_h8`: rank_ic=+0.1406, direction_acc=+0.9979, decile_spread=+0.0015, event_auc=+0.5573
- `return_h16`: rank_ic=+0.0144, direction_acc=+0.5053, decile_spread=+0.0005
- `vol_h16`: rank_ic=+0.1949, direction_acc=+0.9991, decile_spread=+0.0007, event_auc=+0.5849
- `drawdown_risk_h16`: rank_ic=+0.1402, direction_acc=+0.9985, decile_spread=+0.0017, event_auc=+0.5758
- `return_h32`: rank_ic=+0.0092, direction_acc=+0.5075, decile_spread=+0.0005
- `vol_h32`: rank_ic=+0.1782, direction_acc=+0.9991, decile_spread=+0.0007, event_auc=+0.6018
- `drawdown_risk_h32`: rank_ic=+0.1207, direction_acc=+0.9980, decile_spread=+0.0025, event_auc=+0.5336
- `teacher_underweight`: accuracy=+0.4943, balanced_accuracy=+0.4874, auc=+0.4858
- `teacher_class`: accuracy=+0.4949, balanced_accuracy=+0.4874, auc=+0.4852
- `recovery_h16`: accuracy=+0.5234, balanced_accuracy=+0.5050, auc=+0.5035
- `action_advantage_h16`: top1_accuracy=+0.5114, top2_accuracy=+0.5114, chosen_minus_benchmark_adv=-0.0002, pred_long_rate=+0.0000, pred_underweight_rate=+0.4118

##### raw_plus_latent
- `return_h1`: rank_ic=+0.0067, direction_acc=+0.5038, decile_spread=+0.0000
- `vol_h1`: rank_ic=+0.0297, direction_acc=+0.9942, decile_spread=+0.0000, event_auc=+0.5267
- `drawdown_risk_h1`: rank_ic=+0.0376, direction_acc=+0.9696, decile_spread=+0.0005, event_auc=+0.5644
- `return_h4`: rank_ic=+0.0187, direction_acc=+0.5108, decile_spread=+0.0001
- `vol_h4`: rank_ic=+0.3266, direction_acc=+0.9994, decile_spread=+0.0013, event_auc=+0.6552
- `drawdown_risk_h4`: rank_ic=+0.1993, direction_acc=+0.9964, decile_spread=+0.0017, event_auc=+0.6182
- `return_h8`: rank_ic=+0.0217, direction_acc=+0.5020, decile_spread=+0.0007
- `vol_h8`: rank_ic=+0.3452, direction_acc=+0.9997, decile_spread=+0.0013, event_auc=+0.6522
- `drawdown_risk_h8`: rank_ic=+0.2317, direction_acc=+0.9987, decile_spread=+0.0027, event_auc=+0.6137
- `return_h16`: rank_ic=+0.0327, direction_acc=+0.5084, decile_spread=+0.0010
- `vol_h16`: rank_ic=+0.3079, direction_acc=+0.9999, decile_spread=+0.0012, event_auc=+0.6364
- `drawdown_risk_h16`: rank_ic=+0.2186, direction_acc=+0.9995, decile_spread=+0.0036, event_auc=+0.6000
- `return_h32`: rank_ic=+0.0490, direction_acc=+0.5210, decile_spread=+0.0025
- `vol_h32`: rank_ic=+0.2226, direction_acc=+0.9998, decile_spread=+0.0009, event_auc=+0.6170
- `drawdown_risk_h32`: rank_ic=+0.1542, direction_acc=+0.9994, decile_spread=+0.0030, event_auc=+0.5398
- `teacher_underweight`: accuracy=+0.4934, balanced_accuracy=+0.4896, auc=+0.4910
- `teacher_class`: accuracy=+0.4930, balanced_accuracy=+0.4897, auc=+0.4931
- `recovery_h16`: accuracy=+0.5228, balanced_accuracy=+0.5086, auc=+0.5180
- `action_advantage_h16`: top1_accuracy=+0.5150, top2_accuracy=+0.5150, chosen_minus_benchmark_adv=-0.0001, pred_long_rate=+0.0000, pred_underweight_rate=+0.4350
