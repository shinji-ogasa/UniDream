# Optimization Status

## 現在の主課題

1. teacher は弱い
2. BC prior が teacher marginal を保持できない
3. AC / WM は主因というより、その後段で悪化している可能性が高い
4. source family は補助要因で、主因ではない

## issue 別の状態

### issue1 teacher audit by regime

- 完了
- teacher はほぼ `long 0% / short 50% / flat 50%`
- `min_hold` を振っても行動分布はほぼ不変
- 次候補 teacher は `signal_aim`

### issue2 BC prior の再現性診断

- 完了
- `signal_aim teacher + current best learner family` でも
  - `teacher_short 0.353`
  - `bc_short 0.999`
  - `teacher_to_bc_mean_abs_gap 0.145`
- issue2 は主因として確定
- 単純な `weighted / sequence / residual / balanced` では改善なし

### issue3 AC の support 逸脱診断

- 診断コードと runner は実装済み
- 次の本番対象

### issue4 WM に regime 補助目的を追加

- 診断コードと runner は実装済み
- issue3 の後に評価

### issue5 AWR/AWAC or IQL/CQL 系へ寄せる

- 候補 config と runner は実装済み
- issue3 / issue4 の結果を見て入る

### issue6 external source を追加評価

- source rollout は整備済み
- `orderflow > basis`
- ただし主因ではないので後順位

### issue7 learner / output collapse

- 完了
- 既存 1-step CE actor family はほぼ全面的に `BC short ~ 1.0`
- 主因は旧 actor family 側

### issue8 continuous target head

- 一段完了
- current best:
  - `medium_l0_bc_continuous_regimegate_exec`
  - `bc_short 0.9968`
  - `teacher_to_bc_mean_abs_gap 0.1293`
- inference / target-mass の派生:
  - `bootstrap`: flat collapse
  - `damped`: 変化なし
  - `execsplit`: 悪化
  - `regimedist`: 悪化
  - `distcombo`: 悪化
  - `shortmass`: L0 は少し改善する
  - `medium_l1_bc_continuous_exec_shortmass`: test `alpha -1.19 pt/yr` で benchmark 近傍
  - `direct target track`: `alpha -33.12 pt/yr` で悪化
  - `rawonly/orderflow + shortmass`: `gap 0.1515` で悪化

結論:
- issue8 の current best は維持
- ただし issue8 family だけでは M2 は遠い

## 現在の best

- teacher:
  - `signal_aim`
- learner family:
  - `continuous target head + execution_aux + shortmass`
- inference keep:
  - `logits blend 0.50` (`alpha -0.34 pt/yr`)

## 次の本命

1. `issue10 logits blend` の keep 候補を learner 側へ戻せるか切る
2. `direct target track` と `rawonly` では解けなかったので、別 learner family を優先
3. heavy な sequence family は runtime 制約で後回し

## いま避けるもの

- 重い all-fold run
- inference の小手先調整を増やすこと
- source family を主因扱いすること
