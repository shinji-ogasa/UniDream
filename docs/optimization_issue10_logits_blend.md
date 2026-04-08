# Optimization Loop: Issue 10 Action-Head Bottleneck / Logits Blend

## 目的

issue8 の追確認で、`medium_l1_bc_continuous_exec_shortmass` は

- `bc_short_ratio ≈ 0.9966`
- `short_target_mass_mean ≈ 0.0039`
- `baseline_target_mass_mean ≈ 0.9961`

だった。  
つまり、target head は benchmark 近傍なのに final action は `short 100%` に潰れていた。

このため issue10 では、

- target head をそのまま使う `direct target track`
ではなく、
- target logits を少しだけ混ぜる `infer_target_from_logits`

で action-head bottleneck を軽く緩められるかを test-only で確認した。

## baseline

### `medium_l1_bc_continuous_exec_shortmass`

- test `alpha -1.19 pt/yr`
- `sharpe delta -0.030`
- `maxdd delta -1.48 pt`
- `win 49.9%`
- `test dist: short 100%`

## 失敗した枝

### `direct target track`

- config: `medium_l1_bc_continuous_exec_shortmass_directtrack`
- test `alpha -33.12 pt/yr`
- `sharpe delta -1.061`
- `maxdd delta +3.81 pt`
- `win 45.9%`
- `test dist: short 97% / flat 3%`

判定:
- target mean をそのまま execution に流すと悪化
- full bypass は reject

## post-issue10: logits blend

### `medium_l1_bc_continuous_exec_shortmass_logitsblend25`

- test `alpha -0.65 pt/yr`
- `sharpe delta -0.015`
- `maxdd delta -0.99 pt`
- `win 50.0%`
- `test dist: short 100%`

判定:
- baseline より改善
- ただし action collapse は残る

### `medium_l1_bc_continuous_exec_shortmass_logitsblend50`

- test `alpha -0.34 pt/yr`
- `sharpe delta -0.008`
- `maxdd delta -0.60 pt`
- `win 50.0%`
- `test dist: flat 100%`

判定:
- baseline より改善
- `blend 0.25` よりも少し良い
- ただし benchmark/flat 側へ寄りすぎる

### `medium_l1_bc_continuous_exec_shortmass_logitsblend375`

- test `alpha -0.48 pt/yr`
- `sharpe delta -0.011`
- `maxdd delta -0.78 pt`
- `win 50.0%`
- `test dist: short 40% / flat 60%`

判定:
- `blend 0.25` より良い
- ただし `blend 0.50` には届かない
- 中間 blend でも action-head bottleneck は少しだけ緩む

## 結論

- `logits blend` は `direct target track` より明確に良い
- この checkpoint では `blend 0.50` が test alpha を最も benchmark 近傍まで戻した
- ただし learner collapse 自体を解いたわけではなく、
  action-head の崩壊を inference で少し緩めただけ

## training-side follow-up

### `medium_l1_bc_continuous_exec_shortmass_align`

- BC-only val `teacher_to_bc_mean_abs_gap = 0.3380`
- `bc_short_ratio = 0.9980`
- `bc_flat_ratio = 0.0020`
- `target_entropy_mean = 0.6086`
- `short_target_mass_mean = 0.5146`
- `baseline_target_mass_mean = 0.4854`

判定:
- mass-match と entropy を強めると benchmark 側へ飛びすぎる
- teacher 再現は current best (`gap 0.1287`) より大幅に悪化
- reject

### `medium_l0_bc_continuous_execaux`

- BC-only val `teacher_to_bc_mean_abs_gap = 0.1432`
- `bc_short_ratio = 0.9988`
- `bc_flat_ratio = 0.0012`
- `target_entropy_mean = 0.2484`
- `short_target_mass_mean = 0.0861`
- `baseline_target_mass_mean = 0.9139`

判定:
- execution / target 補助だけ少し強めても collapse は残る
- current best (`gap 0.1287`) を更新できない
- reject

## 次

- issue10 の training-side 軽量枝は 2 本とも reject
- current keep は `medium_l1_bc_continuous_exec_shortmass + logits blend 0.50`
- 次は issue4 / issue5 の実測へ進みつつ、action-head 側は必要なら別 family で再開する
