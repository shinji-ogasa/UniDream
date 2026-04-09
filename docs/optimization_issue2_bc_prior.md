# Optimization Loop: Issue 2 BC Prior の再現性診断

## 概要
- `signal_aim` teacher は baseline より改善した
- それでも BC prior が teacher を再現できず、`short 100%` または `flat 100%` に collapse しやすい
- なので issue2 は true

## baseline 診断

### `medium_l1_bc_continuous_regimegate_exec`
- `teacher_short 0.353`
- `teacher_flat 0.647`
- `bc_short 0.999`
- `bc_flat 0.001`
- `teacher_to_bc_mean_abs_gap 0.145`

判定:
- issue2 は true
- 主因は BC collapse

## learner family の比較

### `medium_l1_bc_continuous_exec_shortmass`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1287`
- `bc_short_ratio = 0.9966`
- `bc_flat_ratio = 0.0034`

### `medium_l0_bc_continuous_signalaim_regimegate_exec`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1424`
- `bc_short_ratio = 0.9980`
- `bc_flat_ratio = 0.0020`
- reject

### `medium_l1_bc_continuous_exec_shortmass_regimebias`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1070`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha_excess -0.26 pt/yr`
- `sharpe_delta -0.006`

判定:
- BC prior mismatch 自体はさらに縮んだ
- ただし `flat 100%` への過補正が残る
- issue2 の current keep は `medium_l1_bc_continuous_exec_shortmass_regimebias`

## weighting branch

### `medium_l0_bc_weighted_regimebias`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1092`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha_excess -0.48 pt/yr`
- `sharpe_delta -0.011`

判定:
- current keep (`gap 0.1070`, `alpha -0.26`) を更新できない
- weighting 枝は current keep 上では negative
- reject

## 判断
- issue2 は主因のまま
- `signal_aim teacher` を使っても BC は still collapse しやすい
- current best learner family は `regimebias 0.50`

## 次
- issue2 の weighting 枝は一段閉じる
- 次は sequence / multimodal のような別 learner family か、issue10 に近い head family の別枝を探す
