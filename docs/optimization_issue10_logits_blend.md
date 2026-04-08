# Optimization Loop: Issue 10 Action-Head Bottleneck / Logits Blend

## 課題
- current learner family は target 側の mass がほぼ benchmark でも final action が `short 100%` に collapse する
- inference では `logits blend` が benchmark 近傍まで戻せるので、action-head 側の bottleneck を疑う

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

判定:
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

判定:
- inference 側では `blend 0.50` が最良
- action-head bottleneck 自体はかなり濃い

## training-side follow-up

### `medium_l1_bc_continuous_exec_shortmass_align`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.3380`
- `bc_short_ratio = 0.9980`
- `bc_flat_ratio = 0.0020`

判定:
- benchmark 側へ寄りすぎ
- reject

### `medium_l0_bc_continuous_execaux`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1432`
- `bc_short_ratio = 0.9988`
- `bc_flat_ratio = 0.0012`

判定:
- current keep を更新できず
- reject

### `medium_l1_bc_continuous_exec_shortmass_balanced`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1265`
- `bc_short_ratio = 0.9973`
- `bc_flat_ratio = 0.0027`
- test `alpha -2.14 pt/yr`
- `sharpe delta -0.055`
- `test dist: short 100%`

判定:
- collapse 指標はわずかに改善
- ただし test は悪化
- reject

## 結論
- inference-only では `logits blend 0.50` が最良
- ただし training-side で同等の改善を出せる軽量 branch はまだ無い
- current keep は
  - learner: `medium_l1_bc_continuous_exec_shortmass`
  - inference: `logits blend 0.50`

## 次
- issue10 は別 family の action-head / learner branch で再開する
- 既存の mass-match / balanced / execaux 系は一旦打ち切る
