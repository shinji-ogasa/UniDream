# Optimization Status

## 全体判断
- issue1 は true
- issue2 は true
- issue3 は baseline では薄い
- issue4 は true だが winner なし
- issue5 は rescue として部分的に有効
- issue6 の first light comparison は negative
- issue10 は true だが一段閉じた

## issueごとの現状

### issue1 teacher audit by regime
- teacher はほぼ `long 0% / short 50% / flat 50%`
- `min_hold` を振っても行動分布はほぼ不変
- 採用 teacher は `signal_aim`

### issue2 BC prior の再現性診断
- baseline:
  - `teacher_short 0.353 -> bc_short 0.999`
  - `teacher_to_bc_mean_abs_gap 0.145`
- current best learner:
  - `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
  - BC-only val `teacher_to_bc_mean_abs_gap 0.1193`
  - test `alpha_excess -0.00 pt/yr`, `sharpe_delta +0.002`
  - caveat: `short 100%`
- weighting follow-up:
  - `medium_l0_bc_weighted_regimebias`
  - BC-only val `teacher_to_bc_mean_abs_gap 0.1092`
  - test `alpha_excess -0.48 pt/yr`, `sharpe_delta -0.011`
  - reject
- residual-shift follow-up:
  - `shift05`: `gap 0.1108`, `alpha -0.37`
  - `shift10`: `gap 0.1098`, `alpha -0.20`
  - `shift12`: `gap 0.1101`, `alpha -0.17`
  - `shift15`: `gap 0.1193`, `alpha -0.00`
  - `shift15_blend375`: `alpha +0.00`, `sharpeΔ +0.002`, `maxddΔ -1.43`
  - `shift15_blend625`: `alpha -0.00`, `sharpeΔ +0.001`, `maxddΔ -0.85`, `short 89% / flat 11%`
  - `shift15_blend5625`: `alpha -0.00`, `sharpeΔ +0.002`, `maxddΔ -1.00`, `short 99% / flat 1%`
  - `shift15_blend6875`: `alpha -0.00`, `sharpeΔ +0.001`, `maxddΔ -0.71`, `flat 100%`
  - `shift15_blend75`: `alpha -0.00`, `sharpeΔ +0.001`, `maxddΔ -0.57`, `flat 100%`
  - `shift15_blend875`: `alpha -0.00`, `sharpeΔ +0.000`, `maxddΔ -0.28`, `flat 100%`
- 主因は still BC collapse

### issue3 AC support drift
- baseline:
  - `teacher_short 0.4998 -> bc_short 1.0 -> ac_short 1.0`
  - `bc_to_ac_short_mismatch = 0.0`
- baseline では AC drift は主因ではない

### issue4 WM に regime 補助目的を追加
- baseline:
  - `medium_v2`: val balanced accuracy `0.353 / 0.340`
- branches:
  - `medium_l0_wm_idmreturn`: `0.366 / 0.325`
  - `medium_l0_wm_capacity`: `0.338 / 0.337`
  - `medium_l0_wm_idmreturn_capacity`: `0.342 / 0.371`
  - `medium_l0_wm_regimeaux`: `0.343 / 0.355`
  - `medium_l0_wm_idmreturn_regimeaux`: `0.331 / 0.359`
- `current / next` を同時改善する winner は出ず、`mixed / no winner`

### issue5 conservative AC
- old current-family winner:
  - `medium_l0_ac_conservative_signal_aim`
  - test `alpha_excess -0.56 pt/yr`
- updated current-keep winner:
  - `medium_l0_ac_conservative_regimebias_soft`
  - val audit `teacher_to_ac_mean_abs_gap 0.1053`
  - test `alpha_excess -0.11 pt/yr`, `sharpe_delta -0.002`, `flat 100%`
- strict:
  - `medium_l0_ac_conservative_regimebias`
  - test `alpha_excess -0.15 pt/yr`
- supportbudget:
  - `medium_l0_ac_supportbudget_regimebias`
  - test `alpha_excess -0.21 pt/yr`
- on top of `shift15`:
  - `medium_l0_ac_conservative_regimebias_shift15_soft`
  - test `alpha_excess -0.43 pt/yr`, `sharpe_delta -0.010`
  - reject
- 結論:
  - rescue としては有効
  - ただし alpha を作る本命ではなく `flat 100%` への benchmark recovery
  - `shift15` learner 上では rescue 効果は出ず悪化

### issue6 external source
- source family suite では `orderflow > basis` の傾向がある
- ただし current learner family 上の first light comparison では negative
  - `basis-only`: `alpha_excess -84.01 pt/yr`, `gap 0.1399`
  - `orderflow-added`: `alpha_excess -91.93 pt/yr`, `gap 0.1347`
- source 単独では BC collapse を救えていない

### issue10 action-head bottleneck
- inference-only winner:
  - `logits blend 0.50`
  - test `alpha_excess -0.34 pt/yr`
- training-side winner:
  - `medium_l1_bc_continuous_exec_shortmass_regimebias`
  - BC-only val `teacher_to_bc_mean_abs_gap 0.1070`
  - test `alpha_excess -0.26 pt/yr`
- post-winner rejects:
  - `tradebias`
  - `bandbias`
  - `execbias`
  - `dualbias`
  - `dualbias_execbias`
- 現状は `short 100%` を `flat 100%` に戻した段階

## current keep
- teacher: `signal_aim`
- learner keep: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15` (provisional)
- inference keep: `infer_logits_target_blend = 0.625` on top of `shift15` (provisional)
- issue5 rescue keep: `medium_l0_ac_conservative_regimebias_soft`

## 次
1. source family 単独の期待は下げて learner family 側へ戻る
2. `shift15` は provisional keep に維持
3. `short 100%` を崩しつつ `shift15` の alpha 近傍を維持できる別 learner / inference 枝を探す
4. source は learner update 後に再評価する
