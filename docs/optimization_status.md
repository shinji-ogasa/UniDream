# Optimization Status

## 現在地
- `issue1 teacher audit`: true
- `issue2 BC prior`: true
- `issue3 AC support drift`: baseline では薄い
- `issue4 WM regime representation`: mixed / no winner
- `issue5 conservative AC`: rescue としては有効
- `issue6 external source`: 現 learner family では negative
- `issue10 action-head bottleneck`: true

## 現 keep
- teacher: `signal_aim`
- teacher keep: `signal_scale=1.5`
- learner keep: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
- inference keep: `infer_logits_target_blend = 0.50` on top of `signal_scale=1.5`
- rescue keep: `medium_l0_ac_conservative_regimebias_soft`

## 重要な更新

### issue1 teacher / signal_aim 再調整
- `signal_scale=1.5` は fold 0 の `teacher_to_bc_mean_abs_gap` を `0.1167 -> 0.1007` まで改善
- `signal_scale=1.5 + deadzone=0.10` は `gap 0.1092`
- teacher を少し強める枝は fold 0 の learner collapse 緩和に効く

### issue2 BC prior / inference
- 旧 provisional keep:
  - `shift15 + blend625`
  - fold 4: `alpha -0.00`, `sharpeΔ +0.001`, `maxddΔ -0.85`
  - fold 0: `alpha +0.79`, `sharpeΔ -0.009`, `maxddΔ -1.58`
- `signal_scale=1.5 + blend500`
  - fold 4: `alpha +0.04`, `sharpeΔ +0.003`, `maxddΔ -0.99`, `short 98% / flat 2%`
  - fold 0: `alpha +0.76`, `sharpeΔ +0.021`, `maxddΔ -1.38`, `short 28% / flat 72%`
  - fold 1: `alpha +1.27`, `sharpeΔ +0.000`, `maxddΔ -1.27`, `short 100%`
- `signal_scale=1.5 + blend4375`
  - fold 4: `alpha +0.04`, `sharpeΔ +0.003`, `maxddΔ -1.12`, `short 99% / flat 1%`
  - fold 0: `alpha +0.86`, `sharpeΔ +0.024`, `maxddΔ -1.55`, `short 70% / flat 30%`
- 現時点では `blend500` を keep
  - 理由: fold 0 / 1 / 4 で alpha がすべてプラス、かつ `blend4375` より DD が少しマシ

### issue3 AC support drift
- baseline:
  - `teacher_short 0.4998 -> bc_short 1.0 -> ac_short 1.0`
  - `bc_to_ac_short_mismatch = 0.0`
- baseline では AC drift は主因ではない

### issue4 WM regime representation
- winner なし
- `idmreturn` / `capacity` / `regimeaux` / `idmreturn_regimeaux` を切ったが、`current / next` balanced accuracy を同時改善する枝は出ていない

### issue5 conservative AC
- `medium_l0_ac_conservative_regimebias_soft`
  - test `alpha -0.11`, `sharpeΔ -0.002`
- `std10 + conservative soft` on fold 0
  - test `alpha +0.70`, `sharpeΔ +0.013`, `maxddΔ -1.30`
- rescue としては効くが、本命の alpha 枝ではない

### issue6 external source
- orderflow を足すと BC gap は少し良くなるが、現 learner family では test alpha は改善しない
- source 単独では collapse を救えていない

### issue10 action-head bottleneck
- true
- `regimebias` 系は `short 100%` を benchmark 近傍へ戻すところまでは効いた
- ただし training-side 同系統だけでは alpha winner が出ず、teacher 強化との組み合わせへ移行した

## 次
1. `signal_scale=1.5 + blend500` を基準に fold 2 / fold 3 を軽量で追加確認する
2. 2〜4 fold で方向が揃うなら provisional keep を更新して issue5 を再評価する
3. そこで止まるなら、次は teacher 側ではなく learner head の別 family へ戻る
