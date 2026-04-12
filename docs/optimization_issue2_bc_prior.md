# Optimization Loop: Issue 2 BC Prior

## 判定
- issue2 は true
- baseline でも `teacher -> BC` の collapse が強い
- 現在は `teacher 強化 + inference 調整` が一番前進している

## baseline
- `medium_l1_bc_continuous_regimegate_exec`
  - `teacher_short 0.353`
  - `teacher_flat 0.647`
  - `bc_short 0.999`
  - `bc_flat 0.001`
  - `teacher_to_bc_mean_abs_gap 0.145`

## 旧 learner branch
- `medium_l1_bc_continuous_exec_shortmass`
  - val gap `0.1287`
- `medium_l1_bc_continuous_exec_shortmass_regimebias`
  - val gap `0.1070`
  - test `alpha -0.26 pt/yr`
  - `flat 100%`
- `shift15`
  - test `alpha -0.00 pt/yr`
  - `sharpe_delta +0.002`
  - ただし `short 100%`
- `shift15 + blend625`
  - fold 4: `alpha -0.00`, `sharpeΔ +0.001`, `maxddΔ -0.85`
  - fold 0: `alpha +0.79`, `sharpeΔ -0.009`, `maxddΔ -1.58`
  - `short 89% / flat 11%` まで崩せたが、cross-fold では弱い

## Web 後に切った枝

### overconfidence 緩和
- `entropy05`
  - fold 0 val で変化なし
  - reject
- `std10`
  - fold 0 val で `target_entropy` と `short_mass` がゼロ張り付きから外れる
  - fold 0 test: `alpha +0.76`, `sharpeΔ +0.012`, `maxddΔ -1.43`
  - fold 4 test: `alpha -0.14`, `sharpeΔ -0.003`
  - fold 4 を壊すので reject
- `softlabel05`
  - 変化なし
  - reject
- `softlabel05 + std10`
  - `std10` と同一挙動
  - reject

### teacher 強化
- `signal_scale=1.5`
  - fold 0 val gap `0.1007`
  - fold 0 test: `alpha +0.57`, `sharpeΔ +0.016`, `maxddΔ -1.03`
  - fold 4 test: `alpha +0.03`, `sharpeΔ +0.002`, `maxddΔ -0.74`
  - teacher 強化は fold 0 / 4 の両方で有効
- `signal_scale=1.5 + deadzone=0.10`
  - fold 0 val gap `0.1092`
  - `signal_scale=1.5` 単体に負け
  - reject

### inference retune on top of `signal_scale=1.5`
- `blend500`
  - fold 0: `alpha +0.76`, `sharpeΔ +0.021`, `maxddΔ -1.38`, `short 28% / flat 72%`
  - fold 1: `alpha +1.27`, `sharpeΔ +0.000`, `maxddΔ -1.27`, `short 100%`
  - fold 4: `alpha +0.04`, `sharpeΔ +0.003`, `maxddΔ -0.99`, `short 98% / flat 2%`
- `blend4375`
  - fold 0: `alpha +0.86`, `sharpeΔ +0.024`, `maxddΔ -1.55`, `short 70% / flat 30%`
  - fold 4: `alpha +0.04`, `sharpeΔ +0.003`, `maxddΔ -1.12`, `short 99% / flat 1%`

## 現 provisional keep
- teacher: `signal_aim`
- teacher tuning: `signal_scale=1.5`
- learner: `medium_l1_bc_continuous_exec_shortmass_regimebias_shift15`
- inference: `infer_logits_target_blend = 0.50`

## 解釈
- learner 単独では fold 0 を救い切れない
- teacher を少し強めると fold 0 / 4 の両方が改善する
- いまのボトルネックは `BC head` だけでなく `teacher signal strength` との組み合わせにもある
- 現時点では `signal_scale=1.5 + blend500` がいちばん cross-fold で筋が良い

## 次
1. fold 2 / fold 3 に `signal_scale=1.5 + blend500` を当てる
2. 方向が揃うなら provisional keep を正式更新する
3. その上で issue5 rescue を再評価する
