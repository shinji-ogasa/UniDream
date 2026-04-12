# Optimization Loop: Issue 6 External Source Evaluation

## 判定
- `issue6` は現時点で false
- source を足しても current learner family では改善しない
- 本丸はまだ learner / head family 側

## 初回比較

期間:
- `2021-01-01 -> 2023-06-01`

fold:
- `4`

### basis-only
config:
- `medium_basis_signal_aim_regimebias_l0`

BC-only val:
- `teacher_to_bc_mean_abs_gap = 0.1399`
- `bc_short_ratio = 0.9895`
- `bc_flat_ratio = 0.0105`

test:
- `alpha_excess -84.01 pt/yr`
- `sharpe_delta -0.317`
- `win_rate_vs_bh 48.2%`
- `short 100%`

### orderflow-added
config:
- `medium_ext_signal_aim_regimebias_l0`

added sources:
- `signed_order_flow`
- `taker_imbalance`

BC-only val:
- `teacher_to_bc_mean_abs_gap = 0.1347`
- `bc_short_ratio = 0.9985`
- `bc_flat_ratio = 0.0015`

test:
- `alpha_excess -91.93 pt/yr`
- `sharpe_delta -0.353`
- `win_rate_vs_bh 48.3%`
- `short 99% / flat 1%`

結論:
- orderflow を足すと BC gap は少し改善する
- ただし test alpha は悪化

## current keep family での再評価

config:
- `medium_ext_signal_aim_regimebias_shift15_blend625_l0`

teacher:
- `signal_aim`
- `signal_scale = 1.5`

learner:
- `regimebias`
- `residual_shift = 0.15`

inference:
- `infer_logits_target_blend = 0.625`

added sources:
- `signed_order_flow`
- `taker_imbalance`
- `active_address_growth`

BC-only val:
- `teacher_to_bc_mean_abs_gap = 0.138356`
- `bc_short_ratio = 0.998291`
- `bc_flat_ratio = 0.001709`

test:
- `alpha_excess -126.84 pt/yr`
- `sharpe_delta -0.774`
- `maxdd_delta -0.78 pt`
- `win_rate_vs_bh 47.3%`
- `short 100%`

## 結論
- old ext branch でも negative
- current keep family でもさらに negative
- issue6 は closed

## 次
1. source family への期待は下げる
2. learner / head family の別枝に戻る
3. source は learner が更新された後で必要なら再評価する
