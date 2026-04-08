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

- baseline 監査まで完了
- `medium_v2 / checkpoints/fold_4` では
  - `teacher_short 0.4998`
  - `bc_short 1.0`
  - `ac_short 1.0`
  - `bc_to_ac_short_mismatch = 0.0`
  - `teacher_to_ac_mean_abs_gap = 0.4992`
- baseline では AC drift は主因として薄い
- ただし古い actor head を含む checkpoint は互換性がないので、系統ごとの監査は今後も必要

### issue4 WM に regime 補助目的を追加

- baseline と current family の監査まで完了
- `medium_v2`: val balanced accuracy `0.353 / 0.340`
- `medium_l1_bc_continuous_exec_shortmass_align`: val balanced accuracy `0.331 / 0.364`
- issue4 は `主因候補に戻す`
- L0 first pass:
  - `medium_l0_wm_idmreturn`: `0.366 / 0.325` で mixed
  - `medium_l0_wm_capacity`: `0.338 / 0.337` で reject
  - `medium_l0_wm_idmreturn_capacity`: `0.342 / 0.371` で mixed
- 3 本とも弱いので、次は Web で絞った表現学習系の枝へ進む

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
  - `logits blend 0.375` (`alpha -0.48 pt/yr`)
- training-side follow-up:
  - `medium_l1_bc_continuous_exec_shortmass_align`: `gap 0.3380` で reject
  - `medium_l0_bc_continuous_execaux`: `gap 0.1432` で reject

## 次の本命

1. `issue4`: CPC / explicit regime auxiliary など Web で絞った表現学習系の枝へ進む
2. `issue5`: conservative AC tiny 系を issue3/4 の結果と照らして再比較する
3. action-head 側は新 family が必要になるまで保留

## いま避けるもの

- 重い all-fold run
- inference の小手先調整を増やすこと
- source family を主因扱いすること
