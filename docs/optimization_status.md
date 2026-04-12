# Optimization Status

## 現在地
- `issue1 teacher audit`: true
- `issue2 BC prior`: true
- `issue3 AC support drift`: baseline では薄い
- `issue4 WM regime representation`: mixed / no winner
- `issue5 conservative AC`: fold-conditional rescue
- `issue6 external source`: false / closed
- `issue10 action-head bottleneck`: true

## 現 keep
- teacher: `signal_aim`
- teacher keep: `signal_scale=1.5`
- learner keep: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
- inference keep: `infer_logits_target_blend = 0.625`
- rescue keep: global keep なし

## issue2
- `signal_scale=1.5 + shift15 + blend625` が current keep
- multi-fold ではこれが一番安定
  - fold 0: `alpha +0.57`, `sharpeΔ +0.016`
  - fold 1: `alpha +0.95`, `sharpeΔ +0.000`
  - fold 2: `alpha -1.84`, `sharpeΔ +0.003`
  - fold 3: `alpha -23.85`, `sharpeΔ -0.002`
  - fold 4: `alpha +0.03`, `sharpeΔ +0.002`
  - fold 5: `alpha -1.24`, `sharpeΔ -0.108`
- `signal_scale=1.35 / 1.65` の周辺 sweep は reject
  - alpha は少し上がる
  - ただし current keep の `sharpeΔ +0.002` を壊す

## issue3
- baseline:
  - `teacher_short 0.4998 -> bc_short 1.0 -> ac_short 1.0`
  - `bc_to_ac_short_mismatch = 0.0`
- baseline では AC drift は主因ではない

## issue4
- `idmreturn / capacity / regimeaux / idmreturn_regimeaux`
  は全部 `mixed / no winner`
- current / next balanced accuracy を同時改善する winner はまだない

## issue5
- `conservative AC` は global keep にできない
- current keep 上の `conservative_regimebias_shift15_blend625_sig15_soft`
  - fold 1: `+0.95 -> -7.89`
  - fold 2: `-1.84 -> -1910.08`
  - fold 3: `-23.85 -> -53.10`
  - fold 4: `+0.03 -> +0.70`
  - fold 5: `-1.24 -> +0.33`
- 結論:
  - fold 4 / 5 では効く
  - fold 1 / 2 / 3 では悪化
  - fold-conditional rescue として扱う

## issue6
- old ext branch:
  - `alpha_excess -91.93 pt/yr`
  - `sharpe_delta -0.353`
- current keep family ext branch:
  - `teacher_to_bc_mean_abs_gap = 0.138356`
  - `alpha_excess -126.84 pt/yr`
  - `sharpe_delta -0.774`
  - `short 100%`
- source 追加は現時点で negative

## issue10
- true
- `regimebias` 系は `short 100%` を benchmark 近傍へ戻すところまでは有効
- ただし training-side だけで alpha winner はまだ出ていない
- teacher 強化と inference 調整を含めた組み合わせが現実的

## 次
1. learner head の別 family に戻る
2. 必要なら後で `rescue gate` を別 issue で再検討する
3. source は learner 更新後に再評価する

## 2026-04-13 latest update
- issue2 keep updated
  - teacher: `signal_aim`
  - teacher keep: `signal_scale=1.5`
  - learner keep: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
  - inference keep: `infer_logits_target_blend = 0.625`, `infer_trade_threshold = 0.65`
- `tradebias` family is closed
  - `tradebias(0.50) + signal_scale=1.5`: val gap `0.1083`, test `alpha +0.70`, `sharpeΔ -0.009`, `flat 100%`
  - `tradebias(0.25) + signal_scale=1.5`: val gap `0.0916`, test `alpha +0.70`, `sharpeΔ -0.009`, `flat 100%`
  - `tradebias(0.25)`: val gap `0.1071`, test `alpha +0.82`, `sharpeΔ -0.009`, `flat 100%`
- inference-only retune on current keep is the new local winner
  - `infer_trade_threshold=0.60 / 0.65 / 0.675`
  - `infer_gap_boost=0.05`
  - all converge to `alpha +0.91`, `sharpeΔ +0.027`, `maxddΔ -1.47`, `short 15% / flat 85%`

## 2026-04-13 follow-up
- local winner only:
  - `infer_trade_threshold=0.65`
  - fold 4: `alpha +0.91`, `sharpeΔ +0.027`
  - fold 0: `alpha -11.34`, `sharpeΔ -0.017`
  - fold 5: `alpha -225.84`, `sharpeΔ -0.044`
- conclusion:
  - threshold retune is not a global keep
  - global keep stays:
    - teacher: `signal_aim`
    - teacher keep: `signal_scale=1.5`
    - learner keep: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
    - inference keep: `infer_logits_target_blend = 0.625`

## 2026-04-13 regime-gate probe
- `regimegate0`
  - fold 4: `alpha +0.14`, `sharpeΔ -0.032`, `flat 100%`
- `regimegate1`
  - fold 4: `alpha +0.25`, `sharpeΔ +0.015`, `flat 100%`
- conclusion
  - rescue gate via `infer_regime_active_*` is not a winner on top of the current global keep
  - current global keep is unchanged

## 2026-04-13 bootstrap probe
- `confidence bootstrap`
  - fold 4: `alpha +0.00`, `sharpeΔ +0.000`, `flat 100%`
  - reject
- `support bootstrap`
  - fold 4: `alpha +0.91`, `sharpeΔ +0.027`, `short 15% / flat 85%`
  - same landing as the local threshold retune
  - no new global keep

## 2026-04-13 dual-target probe
- `dual regime target bias`
  - fold 4 val gap `0.1071`
  - fold 4 test: `alpha +0.82`, `sharpeΔ -0.009`, `flat 100%`
  - reject
