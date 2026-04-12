# Optimization Status

## 現在地
- `issue1 teacher audit`: true
- `issue2 BC prior`: true
- `issue3 AC support drift`: baseline では薄い
- `issue4 WM regime representation`: mixed / no winner
- `issue5 conservative AC`: fold-conditional rescue
- `issue6 external source`: 現 learner family では negative
- `issue10 action-head bottleneck`: true

## 現 keep
- teacher: `signal_aim`
- teacher keep: `signal_scale=1.5`
- learner keep: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
- inference keep: `infer_logits_target_blend = 0.625`
- rescue keep: global keep なし

## 直近の重要更新

### issue2 teacher / learner / inference
- `signal_scale=1.5` は fold 0 の `teacher_to_bc_mean_abs_gap` を `0.1167 -> 0.1007` まで改善
- `signal_scale=1.5 + deadzone=0.10` は `gap 0.1092` で負け
- `shift15 + blend625` は multi-fold で一番バランスが良い
  - fold 0: `alpha +0.57 pt/yr`, `sharpeΔ +0.016`, `maxddΔ -1.03`, `flat 100%`
  - fold 1: `alpha +0.95 pt/yr`, `sharpeΔ +0.000`, `maxddΔ -0.95`, `flat 100%`
  - fold 2: `alpha -1.84 pt/yr`, `sharpeΔ +0.003`, `maxddΔ -0.99`, `flat 100%`
  - fold 3: `alpha -23.85 pt/yr`, `sharpeΔ -0.002`, `maxddΔ -0.68`, `flat 100%`
  - fold 4: `alpha +0.03 pt/yr`, `sharpeΔ +0.002`, `maxddΔ -0.74`, `flat 100%`
  - fold 5: `alpha -1.24 pt/yr`, `sharpeΔ -0.108`, `maxddΔ -0.35`, `short 41% / flat 59%`
- `blend500` は fold 0 では強いが、fold 2 / 3 を含めると `blend625` より不安定
- 現時点では `signal_scale=1.5 + shift15 + blend625` が provisional keep
- `signal_scale=1.35 / 1.65` の current-keep 周辺 sweep は
  - fold 4 alpha は少し上がる
  - ただし `sharpeΔ` が `-0.009` まで落ちる
  - current keep を更新できず reject

### issue3 AC support drift
- baseline:
  - `teacher_short 0.4998 -> bc_short 1.0 -> ac_short 1.0`
  - `bc_to_ac_short_mismatch = 0.0`
- baseline family では AC drift は主因ではない

### issue4 WM regime representation
- `idmreturn` / `capacity` / `regimeaux` / `idmreturn_regimeaux`
  は全部 `mixed / no winner`
- `current / next` balanced accuracy を同時改善する winner は出ていない

### issue5 conservative AC
- old keep の `medium_l0_ac_conservative_regimebias_soft`
  - test `alpha -0.11 pt/yr`, `sharpeΔ -0.002`
- current keep 上の `medium_l0_ac_conservative_regimebias_shift15_blend625_foldprobe_sig15_soft`
  - fold 1: `alpha +0.95 -> -7.89`
  - fold 2: `alpha -1.84 -> -1910.08`
  - fold 3: `alpha -23.85 -> -53.10`
  - fold 4: `alpha +0.03 -> +0.70`
  - fold 5: `alpha -1.24 -> +0.33`
- 結論:
  - fold 4 / 5 では効く
  - fold 1 / 2 / 3 では悪化
  - global keep にはならず、fold-conditional rescue 扱い

### issue6 external source
- orderflow を足すと BC gap は少し改善する
- ただし現 learner family では test alpha は悪化
- external source 単独では collapse を救えていない

### issue10 action-head bottleneck
- true
- `regimebias` 系は `short 100%` を benchmark 近傍へ戻すところまでは有効
- ただし training-side だけで alpha winner はまだ出ていない
- teacher 強化と inference 調整を含めた組み合わせが現実的

## 次
1. issue5 は fold-conditional rescue として固定
2. learner head の別 family に戻る
3. 必要なら後で `rescue gate` を別 issue で再検討する
