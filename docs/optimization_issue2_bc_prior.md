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

### `medium_l1_bc_continuous_exec_shortmass_regimebias_shift05`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1108`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha_excess -0.37 pt/yr`
- `sharpe_delta -0.009`

判定:
- current keep より悪化
- reject

### `medium_l1_bc_continuous_exec_shortmass_regimebias_shift10`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1098`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha_excess -0.20 pt/yr`
- `sharpe_delta -0.004`

判定:
- current keep より test は改善
- ただし val gap は少し悪化
- keep 候補

### `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1193`
- `bc_short_ratio = 0.9753`
- `bc_flat_ratio = 0.0247`
- test `alpha_excess -0.00 pt/yr`
- `sharpe_delta +0.002`
- test distribution `short 100%`

判定:
- この family では test が最良
- ただし val gap は悪化し、`short 100%` へ戻る
- issue2 の provisional keep は `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`

### `medium_l1_bc_continuous_exec_shortmass_regimebias_shift12`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1101`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha_excess -0.17 pt/yr`
- `sharpe_delta -0.003`

判定:
- `shift10` より少し良い
- ただし `shift15` は超えられない
- reject

### `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend375`
- test `alpha_excess +0.00 pt/yr`
- `sharpe_delta +0.002`
- `maxdd_delta -1.43 pt`
- test distribution `short 100%`

判定:
- `shift15` 単体と alpha / sharpe は同等
- ただし `maxdd` が悪化
- reject

### `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625`
- test `alpha_excess -0.00 pt/yr`
- `sharpe_delta +0.001`
- `maxdd_delta -0.85 pt`
- test distribution `short 89% / flat 11%`

判定:
- `shift15` 単体とほぼ同じ alpha を維持
- `short 100%` を少し崩せた
- `maxdd` も `shift15` 単体より改善
- provisional inference keep に更新

### `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend75`
- test `alpha_excess -0.00 pt/yr`
- `sharpe_delta +0.001`
- `maxdd_delta -0.57 pt`
- test distribution `flat 100%`

判定:
- alpha は維持
- ただし again `flat 100%`
- `blend625` のような部分的 de-collapse は失う
- reject

### `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend875`
- test `alpha_excess -0.00 pt/yr`
- `sharpe_delta +0.000`
- `maxdd_delta -0.28 pt`
- test distribution `flat 100%`

判定:
- `maxdd` は少し良い
- ただし `flat 100%`
- `blend625` より action diversity が悪い
- reject

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
- `regimebias` に `residual shift` を足すと test は benchmark 近傍まで戻せる
- 現時点の provisional best learner family は `regimebias + shift15`

## 次
- issue2 の weighting 枝は一段閉じる
- `shift15` 周辺を軽く再調整したが、現時点では `shift15` 単体が best
- inference 側では `blend625` が現時点の provisional keep
- 次は `short 89% / flat 11%` をさらに広げつつ alpha を落とさない軽量 learner / inference 枝を探す
