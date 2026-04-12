# Optimization Loop: Issue 5 Conservative AC

## 判定
- `issue5` は global keep としては false
- ただし fold によっては rescue として効く
- 現時点の結論は `fold-conditional rescue`

## baseline
- baseline family では
  - `teacher_short 0.4998`
  - `bc_short 1.0`
  - `ac_short 1.0`
  - `bc_to_ac_short_mismatch = 0.0`
- baseline では AC drift は主因ではない

## old keep 上の結果

### `medium_l0_ac_conservative_regimebias`
- test `alpha -0.15 pt/yr`
- `sharpeΔ -0.003`
- `flat 100%`

### `medium_l0_ac_conservative_regimebias_soft`
- test `alpha -0.11 pt/yr`
- `sharpeΔ -0.002`
- `flat 100%`
- old keep 上の winner

### `medium_l0_ac_supportbudget_regimebias`
- test `alpha -0.21 pt/yr`
- `sharpeΔ -0.005`
- reject

解釈:
- old keep では `conservative AC` は benchmark recovery としては少し効く
- ただし alpha winner ではない

## current keep 上の再評価

current keep:
- teacher: `signal_aim`
- teacher tuning: `signal_scale=1.5`
- learner: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
- inference: `infer_logits_target_blend = 0.625`

### `medium_l0_ac_conservative_regimebias_shift15_blend625_foldprobe_sig15_soft`

#### fold 1
- BC-only: `alpha +0.95 pt/yr`
- rescue AC: `alpha -7.89 pt/yr`
- 判定: 悪化

#### fold 2
- BC-only: `alpha -1.84 pt/yr`
- rescue AC: `alpha -1910.08 pt/yr`
- `flat 100%`
- 判定: 大幅悪化

#### fold 3
- BC-only: `alpha -23.85 pt/yr`
- rescue AC: `alpha -53.10 pt/yr`
- `flat 100%`
- 判定: 悪化

#### fold 4
- BC-only: `alpha +0.03 pt/yr`
- rescue AC: `alpha +0.70 pt/yr`
- `sharpeΔ +0.017`
- `flat 100%`
- 判定: 改善

#### fold 5
- BC-only: `alpha -1.24 pt/yr`
- rescue AC: `alpha +0.33 pt/yr`
- `sharpeΔ -0.021`
- `flat 99%`
- 判定: 改善

## 結論
- issue5 は global keep にはならない
- `conservative AC` は
  - fold 4 / 5 では rescue として効く
  - fold 1 / 2 / 3 では悪化する
- 特に fold 2 は `flat 100%` へ逃げて `alpha -1910 pt/yr` まで壊れるので、そのまま全 fold 適用は不可

## 現在の扱い
- `medium_l0_ac_conservative_regimebias_shift15_blend625_foldprobe_sig15_soft`
  は `fold-conditional rescue candidate`
- global rescue keep は置かない
- 本命は引き続き learner / head family 側

## 次
1. issue5 は一段閉じる
2. learner head の別 family に戻る
3. 必要なら後で `rescue gate` を fold-conditional selector として別 issue で再検討する
