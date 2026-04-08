# Optimization Loop: Issue 10 Action-Head Bottleneck / Logits Blend

## 概要
- current learner family は target mass をほぼ benchmark に寄せているのに、final action が `short 100%` に collapse する
- inference-only の `logits blend` は効く
- なので主因は action-head / execution-side の bottleneck とみなす

## baseline

### `medium_l1_bc_continuous_exec_shortmass`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1287`
- `bc_short_ratio = 0.9966`
- `bc_flat_ratio = 0.0034`
- test `alpha -1.19 pt/yr`
- `sharpe delta -0.030`
- `test dist: short 100%`

## inference-only

### `direct target track`
- test `alpha -33.12 pt/yr`
- `sharpe delta -1.061`
- `test dist: short 97% / flat 3%`
- reject

### `logits blend 0.25`
- test `alpha -0.65 pt/yr`
- `sharpe delta -0.015`
- `test dist: short 100%`

### `logits blend 0.375`
- test `alpha -0.48 pt/yr`
- `sharpe delta -0.011`
- `test dist: short 40% / flat 60%`

### `logits blend 0.50`
- test `alpha -0.34 pt/yr`
- `sharpe delta -0.008`
- `test dist: flat 100%`

判断:
- inference-only では `blend 0.50` が最良
- action-head bottleneck 仮説は強い

## training-side follow-up

### `medium_l1_bc_continuous_exec_shortmass_align`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.3380`
- `bc_short_ratio = 0.9980`
- `bc_flat_ratio = 0.0020`
- reject

### `medium_l0_bc_continuous_execaux`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1432`
- `bc_short_ratio = 0.9988`
- `bc_flat_ratio = 0.0012`
- reject

### `medium_l1_bc_continuous_exec_shortmass_balanced`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1265`
- `bc_short_ratio = 0.9973`
- `bc_flat_ratio = 0.0027`
- test `alpha -2.14 pt/yr`
- `sharpe delta -0.055`
- `test dist: short 100%`
- reject

### `medium_l1_bc_continuous_exec_shortmass_quality`
### `medium_l1_bc_continuous_exec_shortmass_quality_balanced`
- `balanced` と同一挙動
- reject

### `medium_l1_bc_continuous_exec_shortmass_regimebias`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1070`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha -0.26 pt/yr`
- `sharpe delta -0.006`
- `test dist: flat 100%`

判断:
- current keep を更新
- training-side winner
- collapse を `short 100%` から `flat 100%` へ戻せた
- ただし still 過補正

### `medium_l1_bc_continuous_exec_shortmass_regimebias25`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1085`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha -0.38 pt/yr`
- `sharpe delta -0.009`
- `test dist: flat 100%`
- `regimebias 0.50` より悪いので reject

### `medium_l1_bc_continuous_exec_shortmass_regimeshift`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1307`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha -0.29 pt/yr`
- `sharpe delta -0.007`
- `test dist: flat 100%`
- `regimebias 0.50` を更新できないので reject

### `medium_l1_bc_continuous_exec_shortmass_regimebias_floor`
- test `alpha -0.26 pt/yr`
- `sharpe delta -0.006`
- `test dist: flat 100%`

判断:
- `min_trade_floor` では過補正は戻らない
- `regimebias 0.50` と同一挙動
- reject

## inference sweep on regimebias winner

### `medium_l1_bc_continuous_exec_shortmass_regimebias_blend375`
- test `alpha -0.35 pt/yr`
- `sharpe delta -0.008`
- `test dist: flat 100%`
- reject

### `medium_l1_bc_continuous_exec_shortmass_regimebias_blend25`
- test `alpha -0.46 pt/yr`
- `sharpe delta -0.011`
- `test dist: flat 100%`
- reject

## 判断
- issue10 は true
- inference-only では `logits blend 0.50` が有効
- training-side では `regimebias 0.50` が最良
- current keep は
  - learner: `medium_l1_bc_continuous_exec_shortmass_regimebias`
  - inference: `infer_logits_target_blend = 0.50`

## 次
- `flat 100%` の過補正を戻せる軽量 head family に進む
- 既存の `align / execaux / balanced / quality / regimebias25 / regimebias_floor / blend375 / blend25 / regimeshift` は打ち切り
