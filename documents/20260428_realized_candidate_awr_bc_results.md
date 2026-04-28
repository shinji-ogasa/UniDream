# realized candidate advantage AWR-style residual BC 検証結果 2026-04-28

## 目的

Q argmax 型 AC ではなく、実現リターンで評価した candidate advantage を使って residual BC / AWR-style policy extraction を行う。

Phase 8 比で以下を満たせるか確認した。

- AlphaEx: Phase 8 比 +0.2 pt/yr 以上
- SharpeDelta: >= 0
- MaxDDDelta: <= -1.0 pt
- turnover: <= 3.5
- short: <= 25%
- long: <= 5%

基準にした Phase 8 / AC safe baseline は概ね以下。

| metric | Phase 8 baseline |
|---|---:|
| AlphaEx | +0.90 to +0.91 pt/yr |
| SharpeDelta | -0.010 to -0.011 |
| MaxDDDelta | -1.59 to -1.61 pt |
| turnover | 2.60 to 2.62 |
| short | 16 to 17% |
| flat | 83 to 84% |
| long | 0% |
| recovery gate active | 0.7 to 0.9% |

## 実装した計算

Phase 8 actor を anchor policy として、各時点で候補 position を作った。

```text
anchor = Phase 8 actor の rollout position
current = 直前 position
candidate = [anchor, anchor - step, anchor + step, hold/current, benchmark=1.0, overweight candidates]
```

各 candidate の value は realized future return から計算した。

```text
value(candidate)
= (candidate - benchmark) * future_return_sum
- one_way_trade_cost * abs(candidate - current)
- volatility_penalty
- drawdown_penalty
- leverage_penalty
```

使った horizon は Phase 8 の transition advantage と同じ。

```text
horizons = [4, 8, 16, 32]
weights  = [0.15, 0.25, 0.35, 0.25]
```

candidate advantage は以下。

```text
candidate_advantage = value(candidate) - value(anchor)
valid = candidate_advantage > margin
margin = 0.0005
```

AWR-style target は以下。

```text
p(candidate) ∝ exp((candidate_advantage - margin) / tau)
target = anchor_mix * anchor + (1 - anchor_mix) * Σ p(candidate) * candidate
```

BC loss には補助損失として追加した。

```text
loss += realized_candidate_bc_coef
      * normalized_aw_weight
      * Huber(actor_target_position, realized_candidate_target)

loss += realized_candidate_execute_coef
      * Huber(executed_position, realized_candidate_target)
```

## 実験結果

### R1: conservative route AWR

設定:

```text
step = 0.05
overweight candidates = [1.05, 1.10]
anchor_mix = 0.70
max_delta = 0.05
realized_candidate_bc_coef = 0.10
trainable = route_head, route_delta_head, route_advantage_gate
epochs = 3
```

AWR target 生成:

| item | value |
|---|---:|
| active_rate | 30.7% |
| mean_best_improvement | 0.00164 |
| target_short | 99.57% |
| target_flat | 0.001% |
| target_long | 0.43% |
| best minus_0.05 | 5.91% |
| best benchmark | 1.25% |
| best ow_1.10 | 23.54% |

Test:

| metric | result |
|---|---:|
| AlphaEx | +0.91 pt/yr |
| SharpeDelta | -0.010 |
| MaxDDDelta | -1.60 pt |
| turnover | 2.38 |
| short | 15% |
| flat | 85% |
| long | 0% |
| recovery gate active | 1.0% |

判定: 不採用。Phase 8 と実質同じで、AlphaEx +0.2 pt 改善なし、SharpeDelta >= 0 未達。

### R2: low-mix AWR

設定:

```text
step = 0.10
overweight candidates = [1.05, 1.10, 1.25]
anchor_mix = 0.25
max_delta = 0.15
realized_candidate_bc_coef = 0.25
route_delta_scale = 0.015
trainable = route_head, route_delta_head, route_advantage_gate
epochs = 4
```

AWR target 生成:

| item | value |
|---|---:|
| active_rate | 40.0% |
| mean_best_improvement | 0.00240 |
| target_short | 72.62% |
| target_flat | 0.0% |
| target_long | 27.38% |
| best minus_0.10 | 11.94% |
| best benchmark | 1.25% |
| best ow_1.25 | 26.80% |

Test:

| metric | result |
|---|---:|
| AlphaEx | +0.91 pt/yr |
| SharpeDelta | -0.010 |
| MaxDDDelta | -1.62 pt |
| turnover | 2.84 |
| short | 19% |
| flat | 81% |
| long | 0% |
| recovery gate active | 0.9% |

判定: 不採用。target 上は overweight が増えたが、実行 policy では long 0%。改善なし。

### R3: hard target / stronger route

設定:

```text
mode = hard
step = 0.10
overweight candidates = [1.05, 1.10, 1.25]
anchor_mix = 0.0
max_delta = 0.20
realized_candidate_bc_coef = 0.40
realized_candidate_execute_coef = 0.05
route_delta_scale = 0.02
trainable = route_head, route_delta_head, route_advantage_gate, inventory_recovery_head
epochs = 4
```

Test:

| metric | result |
|---|---:|
| AlphaEx | +0.90 pt/yr |
| SharpeDelta | -0.010 |
| MaxDDDelta | -1.61 pt |
| turnover | 2.70 |
| short | 17% |
| flat | 83% |
| long | 0% |
| recovery gate active | 3.3% |

判定: 不採用。hard target でも long は出ず、Phase 8 から改善なし。

### R4: realized-only capacity stress

目的: 旧 teacher loss を弱め、route/trade 実行側まで開放したら AWR target が position に出るか確認。

設定:

```text
target_aux_coef = 0.0
trade_aux_coef = 0.0
execution_aux_coef = 0.0
route_target_coef = 0.0
realized_candidate_bc_coef = 1.0
realized_candidate_execute_coef = 0.30
route_delta_scale = 0.05
route_max_step = 0.10
overweight route step = 0.20
trainable = route_head, route_delta_head, route_advantage_gate, trade_head, execution_head, band_head
epochs = 5
batch_size = 256
```

注意: この条件は route capacity を先に広げたため、target生成時の anchor も Phase 8 と同一ではない。これは採用候補ではなく、capacity stress test として扱う。

AWR target 生成:

| item | value |
|---|---:|
| active_rate | 40.7% |
| mean_best_improvement | 0.00188 |
| anchor_long | 99.997% |
| target_short | 15.46% |
| target_long | 84.54% |
| best ow_1.25 | 20.23% |

Test:

| metric | result |
|---|---:|
| AlphaEx | -0.53 pt/yr |
| SharpeDelta | +0.022 |
| MaxDDDelta | +0.87 pt |
| turnover | 13.49 |
| short | 0% |
| flat | 65% |
| long | 35% |
| recovery gate active | 0.0% |

判定: 不採用。long は出せたが、turnover 13.49、long 35%、AlphaEx 悪化で完全に制約違反。

### R5: capped AWR

目的: Phase 8 anchor を固定し、long target を上位5%に制限して小さな overweight だけ出るか確認。

設定:

```text
realized_candidate_anchor_overrides:
  route_max_step = 0.05
  route_delta_scale = 0.005
  route_max_step_by_route = Phase 8相当
realized_candidate_long_rate_max = 0.05
step = 0.10
overweight candidates = [1.05, 1.10, 1.25]
anchor_mix = 0.30
max_delta = 0.15
realized_candidate_bc_coef = 0.45
realized_candidate_execute_coef = 0.15
route_delta_scale = 0.02
route_max_step = 0.075
overweight route step = 0.10
batch_size = 1024
trainable = route_head, route_delta_head, route_advantage_gate, trade_head, execution_head, band_head
epochs = 4
```

AWR target 生成:

| item | value |
|---|---:|
| active_rate | 29.82% |
| mean_best_improvement | 0.00220 |
| anchor_short | 100.0% |
| target_short | 95.00% |
| target_flat | 0.0% |
| target_long | 5.00% |
| best minus_0.10 | 11.93% |
| best benchmark | 1.24% |
| best ow_1.25 | 16.65% |

Test:

| metric | result |
|---|---:|
| AlphaEx | +0.87 pt/yr |
| SharpeDelta | -0.011 |
| MaxDDDelta | -1.54 pt |
| turnover | 3.66 |
| short | 10% |
| flat | 90% |
| long | 0% |
| recovery gate active | 1.0% |

判定: 不採用。long target を5%に制限しても実行 long は0%。AlphaEx は Phase 8 より悪化し、turnover も 3.5 を少し超過。

## まとめ

realized candidate advantage AWR-style residual BC は、今回の Phase 8 本流には採用しない。

理由:

- R1/R2/R3 は safety 制約を保ったままでは policy がほぼ動かず、AlphaEx +0.2 pt 改善なし。
- R4 は long を出せるが、turnover 13.49 / long 35% / AlphaEx -0.53 で崩壊。
- R5 は long target を5%に制限しても実行 long は0%、AlphaExも悪化。

一番重要な発見:

```text
AWR target は作れているが、Phase 8 の route/state-machine/execution controller が最終positionに反映しない。
```

特に route controller は overweight route でも current inventory から route_max_step だけ動く設計なので、underweight から `1.25` target へ直接は届かない。実行側を広げると long は出るが、turnover と long 比率が崩れる。

## 判断

AC 移行条件は満たしていない。

```text
AlphaEx +0.2pt以上: 未達
SharpeDelta >= 0: R4のみ達成だが他制約違反
MaxDDDelta <= -1.0: R1/R2/R3/R5は達成、R4は違反
turnover <= 3.5: R1/R2/R3は達成、R4/R5は違反
short <= 25%: 全て達成
long <= 5%: R1/R2/R3/R5は達成、R4は違反
```

結論:

```text
realized candidate advantage residual BC / AWR-style extraction は不採用。
制限付きACへ戻る条件は満たさない。
Phase 8 safe baseline を維持する。
```

## 次にやるなら

AWR target をさらに強くするより、position controller の責務を分けるべき。

- recovery to benchmark と overweight entry を同じ route に押し込まない
- underweight recovery は inventory controller に残す
- overweight は benchmark 到達後だけ許可する state machine にする
- AC をやる場合も actor full update ではなく、benchmark到達後の small overweight adapter だけに限定する

今回の AWR 実装は本流から削除する。
