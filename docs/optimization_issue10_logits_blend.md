# Optimization Loop: issue10 action-head bottleneck

## 背景
- current learner family は BC 監査では teacher に近づいても、最終 action が `short 100%` か `flat 100%` に潰れやすい
- inference-only の `logits blend` は効くので、主因は action-head / execution-side の bottleneck とみなす

## baseline

### `medium_l1_bc_continuous_exec_shortmass`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1287`
- `bc_short_ratio = 0.9966`
- `bc_flat_ratio = 0.0034`
- test `alpha_excess -1.19 pt/yr`
- `sharpe_delta -0.030`
- `test dist: short 100%`

## inference-only

### `direct target track`
- test `alpha_excess -33.12 pt/yr`
- `sharpe_delta -1.061`
- `test dist: short 97% / flat 3%`
- reject

### `logits blend 0.25`
- test `alpha_excess -0.65 pt/yr`
- `sharpe_delta -0.015`
- reject

### `logits blend 0.375`
- test `alpha_excess -0.48 pt/yr`
- `sharpe_delta -0.011`
- reject

### `logits blend 0.50`
- test `alpha_excess -0.34 pt/yr`
- `sharpe_delta -0.008`
- `test dist: flat 100%`
- inference-only winner

## training-side follow-up

### reject 済み
- `medium_l1_bc_continuous_exec_shortmass_align`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.3380`
- `medium_l0_bc_continuous_execaux`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1432`
- `medium_l1_bc_continuous_exec_shortmass_balanced`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1265`
  - test `alpha_excess -2.14 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_quality`
  - `balanced` と同一挙動
- `medium_l1_bc_continuous_exec_shortmass_quality_balanced`
  - `balanced` と同一挙動
- `medium_l1_bc_continuous_exec_shortmass_regimebias25`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1085`
  - test `alpha_excess -0.38 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimebias_floor`
  - test `alpha_excess -0.26 pt/yr`
  - `regimebias 0.50` と同一挙動
- `medium_l1_bc_continuous_exec_shortmass_regimebias_blend375`
  - test `alpha_excess -0.35 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimebias_blend25`
  - test `alpha_excess -0.46 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimeshift`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1076`
  - test `alpha_excess -0.29 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimebias_tradebias`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1083`
  - `trade_prob_mean = 0.0738`
  - test `alpha_excess -0.40 pt/yr`

### current training-side winner

#### `medium_l1_bc_continuous_exec_shortmass_regimebias`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1070`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha_excess -0.26 pt/yr`
- `sharpe_delta -0.006`
- `test dist: flat 100%`

## 結論
- issue10 は true
- inference-only では `logits blend 0.50` が有効
- training-side では `regimebias 0.50` が current winner
- ただし現状の改善は `short 100%` を `flat 100%` に戻した段階で、alpha はまだ作れていない
- `trade bias` を足しても execution は十分戻らなかった

## current keep
- teacher: `signal_aim`
- learner: `medium_l1_bc_continuous_exec_shortmass_regimebias`
- inference: `infer_logits_target_blend = 0.50`

## 次
- `flat 100%` の過補正を戻せる別 head family に進む
- 次枝は `trade` ではなく `band / execution gate` 側を直接触る
