# Optimization Status

## 現在の主課題
1. teacher は弱い
2. BC prior が teacher を再現できていない
3. AC support drift は baseline では主因ではない
4. WM regime 表現は依然として弱い
5. source family は補助要因で、主因ではない

## issue ごとの状態

### issue1 teacher audit by regime
- 完了
- 結論:
  - teacher はほぼ `long 0% / short 50% / flat 50%`
  - `min_hold` を振っても行動分布はほぼ不変
  - 採用候補は `signal_aim`

### issue2 BC prior の再現性診断
- 完了
- 結論:
  - `signal_aim teacher + current best learner family`
  - `teacher_short 0.353 -> bc_short 0.999`
  - `teacher_to_bc_mean_abs_gap 0.145`
  - 主因は BC collapse
- current keep 比較:
  - `medium_l1_bc_continuous_exec_shortmass_align`: `gap 0.1287`
  - `medium_l0_bc_continuous_signalaim_regimegate_exec`: `gap 0.1424`
  - keep は `medium_l1_bc_continuous_exec_shortmass_align`

### issue3 AC の support 逸脱診断
- baseline 実測完了
- 結論:
  - `teacher_short 0.4998 -> bc_short 1.0 -> ac_short 1.0`
  - `bc_to_ac_short_mismatch = 0.0`
  - baseline では AC drift は主因ではない

### issue4 WM に regime 補助目的を追加
- 軽量 branch 実測完了
- baseline:
  - `medium_v2`: val balanced accuracy `0.353 / 0.340`
- first pass:
  - `medium_l0_wm_idmreturn`: `0.366 / 0.325`
  - `medium_l0_wm_capacity`: `0.338 / 0.337`
  - `medium_l0_wm_idmreturn_capacity`: `0.342 / 0.371`
- post-Web:
  - `medium_l0_wm_regimeaux`: `0.343 / 0.355`
  - `medium_l0_wm_idmreturn_regimeaux`: `0.331 / 0.359`
- 結論:
  - true のまま
  - ただし current / next の両方を改善する winner はまだ無い
  - `mixed / no winner`

### issue5 conservative AC
- baseline family では false
- current learner family 上では部分的に true
- `medium_l0_ac_conservative_signal_aim`
  - val 4096 bars: `bc_short 0.998 -> ac_short 0.002`, `ac_flat 0.998`
  - test: `alpha -0.56 pt/yr`, `sharpe_delta -0.013`, `flat 100%`
- `medium_l0_ac_supportbudget_signal_aim`
  - val 4096 bars: `ac_short 0.905`, `ac_flat 0.095`
  - test: `alpha -0.77 pt/yr`, `sharpe_delta -0.018`, `short 97%`
- 結論:
  - conservative AC は candidate を benchmark 近傍へ戻す効果はある
  - ただし flat 過補正で alpha は作れていない
  - current keep は `medium_l0_ac_conservative_signal_aim`

### issue6 external source
- rollout 導線は揃っている
- `orderflow > basis` の兆候はある
- ただし主因ではない

## 現在の keep
- teacher: `signal_aim`
- learner keep:
  - `medium_l1_bc_continuous_exec_shortmass_align`
- inference keep:
  - `logits blend 0.50`

## issue10 update
- `medium_l1_bc_continuous_exec_shortmass_balanced`
  - BC-only val `gap 0.1265`
  - test `alpha -2.14 pt/yr`
  - collapse 指標はわずかに改善
  - ただし test は current keep より悪い
  - reject

## 次
1. issue10 の learner 側を別 family で再開
2. issue5 は必要なら `lighter conservative AC` を 1 本だけ追加
3. その後に issue6 へ戻る
