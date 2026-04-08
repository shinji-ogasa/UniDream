# Optimization Loop: Issue 5 Conservative AC

## 課題
- `issue2` で BC collapse が主因だと確認した
- それでも current learner family 上で AC を保守化すると改善が出るかを tiny 実測で確認する

## baseline
- 旧 baseline family では
  - `teacher_short 0.4998`
  - `bc_short 1.0`
  - `ac_short 1.0`
  - `bc_to_ac_short_mismatch = 0.0`
- baseline では AC drift は主因ではなかった

## current learner family 上の tiny 実測

### `medium_l0_ac_conservative_signal_aim`

val 4096 bars support audit:
- `teacher_short_ratio 0.357`
- `bc_short_ratio 0.998`
- `ac_short_ratio 0.002`
- `ac_flat_ratio 0.998`
- `teacher_to_ac_mean_abs_gap 0.113`

test:
- `alpha_excess -0.56 pt/yr`
- `sharpe_delta -0.013`
- `maxdd_delta -0.67 pt`
- `win_rate_vs_bh 49.9%`
- distribution: `flat 100%`

判定:
- BC の short collapse は解消する
- ただし flat へ過補正する
- benchmark 近傍までは戻るが M2 には届かない

### `medium_l0_ac_supportbudget_signal_aim`

val 4096 bars support audit:
- `teacher_short_ratio 0.357`
- `bc_short_ratio 0.998`
- `ac_short_ratio 0.905`
- `ac_flat_ratio 0.095`
- `teacher_to_ac_mean_abs_gap 0.116`

test:
- `alpha_excess -0.77 pt/yr`
- `sharpe_delta -0.018`
- `maxdd_delta -0.83 pt`
- `win_rate_vs_bh 50.0%`
- distribution: `short 97% / flat 3%`

判定:
- short collapse は少ししか緩まない
- `conservative AC` より悪い
- reject

## 結論
- issue5 は baseline family では false だったが、current learner family 上では部分的に true
- AC 保守化は `BC short collapse` を benchmark 近傍まで戻す効果がある
- ただし現状は `flat 100%` への過補正が強く、alpha を作れていない

## current keep
- issue5 current keep は `medium_l0_ac_conservative_signal_aim`
- ただし用途は `candidate rescue / benchmark recovery` であって、まだ winner ではない

## 次
- issue5 は `KL budget` か `lighter conservative AC` を 1 本だけ追加で見る余地がある
- ただし優先度としては issue2/issue10 の learner 側改善の方が高い
