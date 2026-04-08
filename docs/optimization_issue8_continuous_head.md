# Optimization Loop: Issue 8 Continuous Target Head

## 目的

issue7 で既存の 1-step CE actor family はほぼ全面的に `BC short ~ 1.0` に collapse した。
そのため issue8 では、teacher marginal を少しでも保持できる learner family として
`continuous target head + regime gate + execution_aux` を検証した。

## 比較条件

- 期間: `2020-01-01 -> 2024-01-01`
- fold: `4`
- 実行: `--stop-after bc`
- 比較軸:
  - `bc_short_ratio`
  - `bc_flat_ratio`
  - `short_target_mass_mean`
  - `baseline_target_mass_mean`
  - `teacher_to_bc_mean_abs_gap`

## 基準

### `medium_l0_bc_continuous_regimegate_exec`

- `teacher_short 0.3752`
- `bc_short 0.9968`
- `bc_flat 0.0032`
- `short_target_mass_mean 0.0082`
- `baseline_target_mass_mean 0.9918`
- `teacher_to_bc_mean_abs_gap 0.1293`

結論:
- issue8 の current best はまだこの枝
- ただし final action は依然として short collapse が強い

## inference / target-mass branch

### `medium_l0_bc_continuous_regimegate_exec_bootstrap`

- `bc_short 0.0000`
- `bc_flat 1.0000`
- `gap 0.1137`

判定:
- over-flat
- reject

### `medium_l0_bc_continuous_regimegate_exec_damped`

- `bc_short 0.9968`
- `bc_flat 0.0032`
- `gap 0.1293`

判定:
- head 指標は基準と同じ
- inference 側で効いた形跡が弱い
- reject

### `medium_l0_bc_continuous_regimegate_execsplit`

- `bc_short 0.9980`
- `bc_flat 0.0020`
- `short_target_mass_mean 0.0038`
- `gap 0.1302`

判定:
- separate execution head でも collapse 改善なし
- reject

### `medium_l0_bc_continuous_regimegate_exec_regimedist`

- `bc_short 0.9990`
- `bc_flat 0.0010`
- `short_target_mass_mean 0.1022`
- `gap 0.1456`

判定:
- regime-aware dist match はむしろ悪化
- reject

### `medium_l0_bc_continuous_regimegate_exec_distcombo`

- `bc_short 0.9988`
- `bc_flat 0.0012`
- `short_target_mass_mean 0.1211`
- `gap 0.1458`

判定:
- dist 系の複合も悪化
- reject

### `medium_l0_bc_continuous_regimegate_exec_shortmass`

L0:
- `bc_short 0.9971`
- `bc_flat 0.0029`
- `short_target_mass_mean 0.0541`
- `baseline_target_mass_mean 0.9459`
- `gap 0.1226`

判定:
- L0 では head 側の marginal が少し改善
- この枝だけ keep して L1 へ

L1:
### `medium_l1_bc_continuous_regimegate_exec_shortmass`

- teacher: `signal_aim`
- `teacher_short 0.3572`
- `bc_short 0.9983`
- `bc_flat 0.0017`
- `short_target_mass_mean 0.0000002`
- `baseline_target_mass_mean 0.9999999`
- `gap 0.1431`

判定:
- `signal_aim` teacher にすると完全に劣化
- robustness が無い
- reject

## L1 follow-up: execution head vs direct target track

上の結果を受けて、`signal_aim` + 軽量 continuous family の中で
実際に残っていた checkpoint に対して test-only 比較を追加した。

### `medium_l1_bc_continuous_exec_shortmass`

collapse audit:
- `teacher_short 0.3572`
- `bc_short 0.9966`
- `bc_flat 0.0034`
- `gap 0.1287`

test-only:
- `alpha -1.19 pt/yr`
- `sharpe delta -0.030`
- `maxdd delta -1.48 pt`
- `win 49.9%`
- `test dist: short 100%`

判定:
- 行動分布は still `short 100%`
- ただし test 指標は benchmark 近傍まで戻る
- issue8 の中では「まだ使える」枝として keep

### `medium_l1_bc_continuous_exec_shortmass_directtrack`

test-only:
- `alpha -33.12 pt/yr`
- `sharpe delta -1.061`
- `maxdd delta +3.81 pt`
- `win 45.9%`
- `test dist: short 97% / flat 3%`

判定:
- `direct target track` はむしろ大幅悪化
- target head をそのまま execution に流す方向は reject

## 結論

- issue8 の current best はまだ
  `medium_l0_bc_continuous_regimegate_exec`
- ただし current best でも short collapse は強い
- inference / target-mass の小手先調整では
  - flat collapse
  - 変化なし
  - L1 非再現
  のどれかに落ちる
- 追確認すると、`medium_l1_bc_continuous_exec_shortmass` は
  collapse を解けていないが test は benchmark 近傍まで戻る
- 一方で `direct target track` は悪化したため、
  次枝は `orderflow/rawonly + shortmass 系` を優先する

## 次

- issue8 はここで一段閉じる
- 次は `execution_aux` を維持したまま別 learner family を見る
- 併行して core loop の issue2 `BC prior audit` を実行し、
  teacher と learner の mismatch を正式に切る
