# Optimization Loop: Issue 2 BC Prior

## 判定
- `issue2` は true
- baseline では `teacher -> BC` の collapse が強い
- 現在は `teacher 強化 + learner shift + inference 調整` の組み合わせが一番効いている

## baseline
- `medium_l1_bc_continuous_regimegate_exec`
  - `teacher_short 0.353`
  - `teacher_flat 0.647`
  - `bc_short 0.999`
  - `bc_flat 0.001`
  - `teacher_to_bc_mean_abs_gap 0.145`

## 既存 learner branch
- `medium_l1_bc_continuous_exec_shortmass`
  - val gap `0.1287`
- `medium_l1_bc_continuous_exec_shortmass_regimebias`
  - val gap `0.1070`
  - test `alpha -0.26 pt/yr`
  - `flat 100%`
- `shift15`
  - test `alpha -0.00 pt/yr`
  - `sharpeΔ +0.002`
  - ただし `short 100%`

## learner-only softening の結果
- `entropy05`: 変化なし
- `std10`: fold 0 は少し改善、fold 4 を壊す
- `shortmass1`: 変化なし
- `softlabel05`: 変化なし
- `softlabel05 + std10`: `std10` と同じ

結論:
- learner-only の softening では cross-fold の改善が出ない
- teacher 側の強化が必要

## teacher 強化

### `signal_scale=1.5`
- fold 0 val gap `0.1007`
- fold 0 test: `alpha +0.57`, `sharpeΔ +0.016`, `maxddΔ -1.03`
- fold 4 test: `alpha +0.03`, `sharpeΔ +0.002`, `maxddΔ -0.74`

### `signal_scale=1.5 + deadzone=0.10`
- fold 0 val gap `0.1092`
- `signal_scale=1.5` 単体に負け

結論:
- teacher 側は `signal_scale=1.5` が keep

## inference retune on top of `signal_scale=1.5`

### `shift15 + blend625`
- fold 0: `alpha +0.57`, `sharpeΔ +0.016`, `maxddΔ -1.03`, `flat 100%`
- fold 1: `alpha +0.95`, `sharpeΔ +0.000`, `maxddΔ -0.95`, `flat 100%`
- fold 2: `alpha -1.84`, `sharpeΔ +0.003`, `maxddΔ -0.99`, `flat 100%`
- fold 3: `alpha -23.85`, `sharpeΔ -0.002`, `maxddΔ -0.68`, `flat 100%`
- fold 4: `alpha +0.03`, `sharpeΔ +0.002`, `maxddΔ -0.74`, `flat 100%`
- fold 5: `alpha -1.24`, `sharpeΔ -0.108`, `maxddΔ -0.35`, `short 41% / flat 59%`

### `shift15 + blend500`
- fold 0: `alpha +0.76`, `sharpeΔ +0.021`, `maxddΔ -1.38`, `short 28% / flat 72%`
- fold 1: `alpha +1.27`, `sharpeΔ +0.000`, `maxddΔ -1.27`, `short 100%`
- fold 2: `alpha -2.45`, `sharpeΔ +0.004`, `maxddΔ -1.32`, `short 99% / flat 1%`
- fold 3: `alpha -31.58`, `sharpeΔ -0.002`, `maxddΔ -0.91`, `flat 98%`
- fold 4: `alpha +0.04`, `sharpeΔ +0.003`, `maxddΔ -0.99`, `short 98% / flat 2%`

### finer blend window around `625`
- `blend5625`
  - fold 4: `alpha -0.00`, `sharpeΔ +0.002`, `maxddΔ -1.00`, `short 99% / flat 1%`
- `blend6875`
  - fold 4: `alpha -0.00`, `sharpeΔ +0.001`, `maxddΔ -0.71`, `flat 100%`

結論:
- fold 0 だけ見ると `blend500` が良い
- fold 2 / 3 を含めると `blend625` の方が安定
- `625` 周辺の finer sweep でも keep は変わらない

## teacher-scale 微調整 around current keep

### `signal_scale=1.35`
- fold 4 test: `alpha +0.76 pt/yr`
- `sharpeΔ -0.009`
- `maxddΔ -1.40 pt`
- `flat 100%`

### `signal_scale=1.65`
- fold 4 test: `alpha +0.64 pt/yr`
- `sharpeΔ -0.009`
- `maxddΔ -1.18 pt`
- `flat 100%`

結論:
- `1.35 / 1.65` はどちらも alpha は少し上がる
- ただし current keep の `sharpeΔ +0.002` を壊す
- `signal_scale=1.5` の keep は維持

## 現 keep
- teacher: `signal_aim`
- teacher tuning: `signal_scale=1.5`
- learner: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
- inference: `infer_logits_target_blend = 0.625`

## 解釈
- learner 単独では fold 0 を崩しやすい
- teacher を強めると fold 0 / 4 の両方で改善する
- いまのボトルネックは `BC head` 単独ではなく、teacher signal strength との組み合わせも含む
- multi-fold で見ると `blend625` が一番守備範囲が広い

## 次
1. `shift15 + blend625 + conservative soft` を fold 2 / 3 に当てて rescue の汎化を確認
2. rescue が局所改善止まりなら issue5 は fold-conditional rescue 扱いにする
3. その後、learner head の別 family に戻る

## 2026-04-13 latest update
- `tradebias` family
  - `tradebias(0.50) + signal_scale=1.5`: val gap `0.1083`, test `alpha +0.70`, `sharpeΔ -0.009`, `flat 100%`
  - `tradebias(0.25) + signal_scale=1.5`: val gap `0.0916`, test `alpha +0.70`, `sharpeΔ -0.009`, `flat 100%`
  - `tradebias(0.25)`: val gap `0.1071`, test `alpha +0.82`, `sharpeΔ -0.009`, `flat 100%`
  - conclusion: val gap は縮むが test では `flat 100%` に落ちるので reject
- inference-only retune on top of current learner
  - `infer_trade_threshold=0.60 / 0.65 / 0.675`
  - `infer_gap_boost=0.05`
  - all converge to the same fold-4 result: `alpha +0.91`, `sharpeΔ +0.027`, `maxddΔ -1.47`, `short 15% / flat 85%`
- updated keep
  - teacher: `signal_aim`
  - teacher tuning: `signal_scale=1.5`
  - learner: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
  - inference: `infer_logits_target_blend = 0.625`, `infer_trade_threshold = 0.65`

## 2026-04-13 follow-up
- `infer_trade_threshold=0.65` was checked out-of-fold using the existing `foldprobe_sig15` checkpoints.
  - fold 0: `alpha -11.34`, `sharpeΔ -0.017`, `flat 100%`
  - fold 5: `alpha -225.84`, `sharpeΔ -0.044`, `short 49% / flat 51%`
- conclusion
  - `threshold=0.65` is only a fold-4 local winner
  - it is not a global keep
  - global issue2 keep stays at `infer_logits_target_blend = 0.625`
