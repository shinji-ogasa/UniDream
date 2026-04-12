# Optimization Loop: issue10 action-head bottleneck

## 背景
- current learner family は BC 監査では teacher に近づいても、最終 action が `short 100%` か `flat 100%` に潰れやすい
- inference-only の `logits blend` は効くので、主因は action-head / execution-side の bottleneck とみなす

## baseline

### `medium_l1_bc_continuous_exec_shortmass`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1287`
- `bc_short_ratio = 0.9966`
- `bc_flat_ratio = 0.0034`
- test `alpha_excess -1.19 pt/yr`
- `sharpe_delta -0.030`
- `test dist: short 100%`

## inference-only

### `direct target track`
- test `alpha_excess -33.12 pt/yr`
- `sharpe_delta -1.061`
- `test dist: short 97% / flat 3%`
- reject

### `logits blend 0.25`
- test `alpha_excess -0.65 pt/yr`
- `sharpe_delta -0.015`
- reject

### `logits blend 0.375`
- test `alpha_excess -0.48 pt/yr`
- `sharpe_delta -0.011`
- reject

### `logits blend 0.50`
- test `alpha_excess -0.34 pt/yr`
- `sharpe_delta -0.008`
- `test dist: flat 100%`
- inference-only winner

## training-side follow-up

### current training-side winner

#### `medium_l1_bc_continuous_exec_shortmass_regimebias`
- BC-only val `teacher_to_bc_mean_abs_gap = 0.1070`
- `bc_short_ratio = 0.0000`
- `bc_flat_ratio = 1.0000`
- test `alpha_excess -0.26 pt/yr`
- `sharpe_delta -0.006`
- `test dist: flat 100%`

### reject 済み
- `medium_l1_bc_continuous_exec_shortmass_align`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.3380`
- `medium_l0_bc_continuous_execaux`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1432`
- `medium_l1_bc_continuous_exec_shortmass_balanced`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1265`
  - test `alpha_excess -2.14 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_quality`
  - `balanced` と同一挙動
- `medium_l1_bc_continuous_exec_shortmass_quality_balanced`
  - `balanced` と同一挙動
- `medium_l1_bc_continuous_exec_shortmass_regimebias25`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1085`
  - test `alpha_excess -0.38 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimebias_floor`
  - test `alpha_excess -0.26 pt/yr`
  - `regimebias 0.50` と同一挙動
- `medium_l1_bc_continuous_exec_shortmass_regimebias_blend375`
  - test `alpha_excess -0.35 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimebias_blend25`
  - test `alpha_excess -0.46 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimeshift`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1076`
  - test `alpha_excess -0.29 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimebias_tradebias`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1083`
  - `trade_prob_mean = 0.0738`
  - test `alpha_excess -0.40 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimebias_bandbias`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1090`
  - `trade_prob_mean = 0.0703`
  - test `alpha_excess -0.51 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_regimebias_execbias`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1119`
  - `bc_short_ratio = 0.0129`
  - test `alpha_excess -1.62 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_dualbias`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1128`
  - `bc_short_ratio = 0.0137`
  - `target_entropy_mean = 0.2665`
  - test `alpha_excess -0.72 pt/yr`
- `medium_l1_bc_continuous_exec_shortmass_dualbias_execbias`
  - BC-only val `teacher_to_bc_mean_abs_gap = 0.1079`
  - `trade_prob_mean = 0.1674`
  - test `alpha_excess -0.80 pt/yr`

## 結論
- issue10 は true
- inference-only では `logits blend 0.50` が有効
- training-side では `regimebias 0.50` が current winner
- ただし現状の改善は `short 100%` を `flat 100%` に戻した段階で、alpha はまだ作れていない
- `trade / band / execution` の軽量 head 3 本は keep を更新できなかった
- Web 後の `multimodal target-bias` 2 本も keep 更新なし

## current keep
- teacher: `signal_aim`
- learner: `medium_l1_bc_continuous_exec_shortmass_regimebias`
- inference: `infer_logits_target_blend = 0.50`

## 次
- issue10 は current keep を保持したまま一段閉じる
- 次は issue6 の external source 比較へ進む

## 2026-04-13 latest update
- `tradebias` family was tested on top of the current learner family.
  - `tradebias(0.50) + signal_scale=1.5`: val gap `0.1083`, test `alpha +0.70`, `sharpeΔ -0.009`, `flat 100%`
  - `tradebias(0.25) + signal_scale=1.5`: val gap `0.0916`, test `alpha +0.70`, `sharpeΔ -0.009`, `flat 100%`
  - `tradebias(0.25)`: val gap `0.1071`, test `alpha +0.82`, `sharpeΔ -0.009`, `flat 100%`
- conclusion: training-side trade bias improves collapse metrics but still collapses to `flat 100%` at test time, so the family is closed.
- inference-only threshold/gap retune is better than training-side trade bias:
  - `infer_trade_threshold=0.60 / 0.65 / 0.675`
  - `infer_gap_boost=0.05`
  - all converge to `alpha +0.91`, `sharpeΔ +0.027`, `maxddΔ -1.47`, `short 15% / flat 85%`
- updated keep
  - teacher: `signal_aim`
  - learner: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
  - inference: `infer_logits_target_blend = 0.625`, `infer_trade_threshold = 0.65`

## 2026-04-13 follow-up
- `infer_trade_threshold=0.65` is only a fold-4 local winner.
  - fold 4: `alpha +0.91`, `sharpeΔ +0.027`, `short 15% / flat 85%`
  - fold 0: `alpha -11.34`, `sharpeΔ -0.017`, `flat 100%`
  - fold 5: `alpha -225.84`, `sharpeΔ -0.044`, `short 49% / flat 51%`
- conclusion
  - do not promote threshold retune to global keep
  - global keep remains `infer_logits_target_blend = 0.625`

## 2026-04-13 regime-gate probe
- `infer_regime_active_state=0`, `infer_regime_active_threshold=0.50`
  - fold 4 test: `alpha +0.14`, `sharpeΔ -0.032`, `maxddΔ -0.42`, `flat 100%`
- `infer_regime_active_state=1`, `infer_regime_active_threshold=0.50`
  - fold 4 test: `alpha +0.25`, `sharpeΔ +0.015`, `maxddΔ -0.42`, `flat 100%`
- conclusion
  - regime-gating is not a winner on top of the current global keep
  - it improves neither alpha/sharpe jointly nor action diversity relative to the threshold retune

## 2026-04-13 bootstrap probe
- `confidence bootstrap`
  - fold 4 test: `alpha +0.00`, `sharpeΔ +0.000`, `maxddΔ +0.00`, `flat 100%`
  - reject
- `support bootstrap`
  - fold 4 test: `alpha +0.91`, `sharpeΔ +0.027`, `maxddΔ -1.47`, `short 15% / flat 85%`
  - same landing as threshold retune
  - not a new winner

## 2026-04-13 dual-target probe
- `dual regime target bias`
  - fold 4 val gap `0.1071`
  - fold 4 test: `alpha +0.82`, `sharpeΔ -0.009`, `maxddΔ -1.49`, `flat 100%`
- conclusion
  - no better than the trade-bias side branch
  - reject
