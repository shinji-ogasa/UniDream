# Optimization Status

## 全体判断
- issue1 は true
- issue2 は true
- issue3 は baseline では薄いが current learner family 上では部分的に true
- issue4 は true だが winner なし
- issue5 は rescue として部分的に有効
- issue6 は次の本命
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
  - `medium_l1_bc_continuous_exec_shortmass_regimebias`
  - BC-only val `teacher_to_bc_mean_abs_gap 0.1070`
  - test `alpha_excess -0.26 pt/yr`, `sharpe_delta -0.006`
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
- `medium_l0_ac_conservative_signal_aim`
  - val: `bc_short 0.998 -> ac_short 0.002`, `ac_flat 0.998`
  - test: `alpha_excess -0.56 pt/yr`, `sharpe_delta -0.013`, `flat 100%`
- `medium_l0_ac_supportbudget_signal_aim`
  - test: `alpha_excess -0.77 pt/yr`, `sharpe_delta -0.018`, `short 97%`
- rescue としては有効だが alpha を作る本命ではない

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
- current keep は維持して issue10 は一段閉じる

## current keep
- teacher: `signal_aim`
- learner keep: `medium_l1_bc_continuous_exec_shortmass_regimebias`
- inference keep: `infer_logits_target_blend = 0.50`

## 次
1. source family 単独の期待は下げて、learner family 側へ戻る
2. current keep を維持したまま次の learner branch を探す
3. source は learner update 後に再評価する
