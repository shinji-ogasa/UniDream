# Optimization Loop: issue6 external source evaluation

## 背景
- source family suite では `orderflow > basis` の傾向があった
- ただし full learner family に入れた時に本当に効くかは未確認だった
- ここでは current keep に近い軽量 learner を、同一期間・同一 WFO 条件で `basis-only` と `orderflow-added` で比較する

## 比較条件
- period: `2021-01-01 -> 2023-06-01`
- fold: `4`
- config family:
  - `medium_basis_signal_aim_regimebias_l0`
  - `medium_ext_signal_aim_regimebias_l0`
- teacher: `signal_aim`
- learner core: `regimebias`
- world model: `250 steps`
- BC-only / test-only まで

## basis-only

### `medium_basis_signal_aim_regimebias_l0`
- feature cache: `obs_dim = 17`
- BC-only val:
  - `teacher_to_bc_mean_abs_gap = 0.1399`
  - `bc_short_ratio = 0.9895`
  - `bc_flat_ratio = 0.0105`
- test:
  - `alpha_excess -84.01 pt/yr`
  - `sharpe_delta -0.317`
  - `win_rate_vs_bh 48.2%`
  - `test dist: short 100%`

## orderflow-added

### `medium_ext_signal_aim_regimebias_l0`
- feature cache: `obs_dim = 47`
- added sources:
  - `signed_order_flow`
  - `taker_imbalance`
- BC-only val:
  - `teacher_to_bc_mean_abs_gap = 0.1347`
  - `bc_short_ratio = 0.9985`
  - `bc_flat_ratio = 0.0015`
- test:
  - `alpha_excess -91.93 pt/yr`
  - `sharpe_delta -0.353`
  - `win_rate_vs_bh 48.3%`
  - `test dist: short 99% / flat 1%`

## 結論
- orderflow を足すと BC-only の gap は `0.1399 -> 0.1347` でわずかに改善する
- ただし test は `alpha_excess -84.01 -> -91.93 pt/yr` で悪化する
- current learner family では、orderflow は BC collapse を救えていない
- したがって issue6 の first light comparison は **negative**

## 次
- source family 単独の期待は下げる
- current keep は external source なしのまま維持
- 次に source を触るなら、learner family 側を更新した後に再評価する
