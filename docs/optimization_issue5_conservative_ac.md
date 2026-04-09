# Optimization Loop: Issue 5 Conservative AC

## 課題
- `issue2` で BC collapse が主因だと確認した
- そのうえで、current keep の learner family 上で AC を保守化すると rescue が効くかを軽量 run で確認する

## baseline
- 旧 baseline family では
  - `teacher_short 0.4998`
  - `bc_short 1.0`
  - `ac_short 1.0`
  - `bc_to_ac_short_mismatch = 0.0`
- baseline では AC drift は主因ではなかった

## current keep 上の軽量実測

current keep:
- teacher: `signal_aim`
- learner keep: `medium_l1_bc_continuous_exec_shortmass_regimebias`

### `medium_l0_ac_conservative_regimebias`

val 4096 bars support audit:
- `teacher_short_ratio 0.357`
- `bc_short_ratio 0.0029`
- `ac_short_ratio 0.0000`
- `bc_flat_ratio 0.9971`
- `ac_flat_ratio 1.0000`
- `teacher_to_ac_mean_abs_gap 0.1060`

test:
- `alpha_excess -0.15 pt/yr`
- `sharpe_delta -0.003`
- `maxdd_delta -0.26 pt`
- `win_rate_vs_bh 49.9%`
- distribution: `flat 100%`

判定:
- rescue としては効く
- ただし `flat 100%` の過補正が強い

### `medium_l0_ac_conservative_regimebias_soft`

val 4096 bars support audit:
- `teacher_short_ratio 0.357`
- `bc_short_ratio 0.0000`
- `ac_short_ratio 0.0000`
- `bc_flat_ratio 1.0000`
- `ac_flat_ratio 1.0000`
- `teacher_to_ac_mean_abs_gap 0.1053`

test:
- `alpha_excess -0.11 pt/yr`
- `sharpe_delta -0.002`
- `maxdd_delta -0.20 pt`
- `win_rate_vs_bh 50.0%`
- distribution: `flat 100%`

判定:
- issue5 の current winner
- strict より少し良い
- ただし alpha を作る本命ではなく benchmark recovery 寄り

### `medium_l0_ac_supportbudget_regimebias`

val 4096 bars support audit:
- `teacher_short_ratio 0.357`
- `bc_short_ratio 0.0029`
- `ac_short_ratio 0.0000`
- `bc_flat_ratio 0.9971`
- `ac_flat_ratio 1.0000`
- `teacher_to_ac_mean_abs_gap 0.1065`

test:
- `alpha_excess -0.21 pt/yr`
- `sharpe_delta -0.005`
- `maxdd_delta -0.28 pt`
- `win_rate_vs_bh 50.0%`
- distribution: `flat 100%`

判定:
- strict / soft より悪い
- reject

## 結論
- issue5 は current keep 上でも部分的に true
- `conservative AC` は BC collapse の後始末として効く
- ただし現状の役割は `short 100%` を `flat 100%` へ戻す rescue
- alpha を作る本命にはまだなっていない

## current keep
- issue5 current keep は `medium_l0_ac_conservative_regimebias_soft`
- 用途は `candidate rescue / benchmark recovery`

## 次
- issue5 は一段閉じる
- 次の本命は issue2 / issue10 側、つまり learner head / target family の改善
