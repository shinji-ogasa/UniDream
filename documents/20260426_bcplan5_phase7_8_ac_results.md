# BC Plan 5 Phase 7/8 + AC Results

作成日: 2026-04-26  
対象: user directive `Phase 7 -> Phase 8 -> AC`  
実行範囲: BTCUSDT 15m / seed 7 / fold 4 / 2018-01-01 to 2024-01-01  
文字コード: UTF-8

## 結論

Phase 8 の state machine route が現時点の最良。AC は採用しない。

Phase 8 は Phase 7/6C の churn を止め、以下まで改善した。

```text
short 17%
flat 83%
turnover 2.62
AlphaEx +0.91 pt/yr
SharpeΔ -0.010
MaxDDΔ -1.61 pt
recovery gate active 0.7%
```

ただし SharpeΔ は境界で、recovery gate はまだ弱い。AC を2本試したが、どちらも test では Phase 8 BC を上回らなかった。

## Web確認に基づくAC方針

AC前に offline RL / actor-critic の方針を確認した。

- TD3+BC は offline RL で actor 更新に behavior cloning loss を足し、分布外 action へ逸脱しにくくする設計。UniDream AC でも `td3bc_alpha` と prior loss を強める方針にした。Source: https://github.com/sfujim/TD3_BC
- 2024 ICML の offline actor-critic scaling 論文では、offline actor-critic が強い BC baseline を上回り得るとされるが、これは大規模・多様なデータでの話。UniDream の fold 単位金融データでは、まずBC priorから大きく外れない設定が妥当。Source: https://openreview.net/forum?id=tl2qmO5kpD

そのため、ACは以下の保守設定で実行した。

- actor lr を小さくする。
- `td3bc_alpha` を大きくする。
- `alpha_final` を高くして BC 混合を残す。
- `prior_kl_coef`, `prior_flow_coef`, `turnover_coef` を強める。
- まず 500 step、次に 100 step micro AC を確認する。

## 実装内容

### Phase 7: Recoveryをroute分類から分離

`route_dim=3` を追加した。

```text
exposure_route:
  neutral
  de_risk
  overweight
```

4値 route label は BC 学習時に以下へ写像する。

```text
neutral  -> neutral
recovery -> neutral
 de_risk -> de_risk
overweight -> overweight
```

recovery は `inventory_recovery_head` に分離した。

```text
if current_inventory < benchmark - gap
and underweight_duration >= D:
    recovery_prob = sigmoid(inventory_recovery_head)
    target = mix(target, recover_to_benchmark, recovery_prob)
```

追加した主な設定:

- `use_inventory_recovery_controller`
- `inventory_recovery_gap`
- `inventory_recovery_min_duration`
- `inventory_recovery_max_step`
- `inventory_recovery_scale`
- `inventory_recovery_coef`
- `inventory_recovery_pos_weight`

### Phase 8: Position-State-Conditioned State Machine Route

underweight 状態で追加 de-risk を抑制する state machine gate を追加した。

```text
if current_inventory < -gap
and underweight_duration >= D:
    de_risk_logit -= penalty
```

4値 route の場合は recovery logit boost も可能だが、今回の Phase 8 は `route_dim=3` なので recovery は inventory controller 側に残した。

追加した主な設定:

- `use_state_machine_route`
- `state_machine_underweight_gap`
- `state_machine_underweight_min_duration`
- `state_machine_derisk_logit_down`

## 実行 Config

| ID | Config | 内容 |
|---|---|---|
| Phase 7 | `configs/bcplan5_phase7_inventory_recovery_s007.yaml` | exposure route + inventory recovery controller |
| Phase 8 | `configs/bcplan5_phase8_state_machine_s007.yaml` | Phase 7 + underweight state machine gate |
| AC tune | `configs/bcplan5_phase8_ac_tune_s007.yaml` | Phase 8から500 step conservative AC |
| AC micro | `configs/bcplan5_phase8_ac_micro_s007.yaml` | Phase 8から100 step stronger-BC AC |

## Backtest Results

| ID | Route dist | Recovery gate | Position dist | AlphaEx | SharpeΔ | MaxDDΔ | Turnover | 判定 |
|---|---|---|---|---:|---:|---:|---:|---|
| Phase 7 | neutral 47 / de_risk 53 / overweight 0 | mean 0.230 / active 0.0% | long 0 / short 35 / flat 65 | +0.25 | -0.055 | -0.38 | 12.11 | churnで不可 |
| Phase 8 | neutral 35 / de_risk 65 / overweight 0 | mean 0.293 / active 0.7% | long 0 / short 17 / flat 83 | +0.91 | -0.010 | -1.61 | 2.62 | 現時点ベスト、AC候補境界 |
| AC tune 500 | neutral 35 / de_risk 64 / overweight 1 | mean 0.150 / active 0.0% | long 0 / short 35 / flat 65 | +0.89 | -0.013 | -1.60 | 4.91 | Phase 8より悪化 |
| AC micro 100 | neutral 34 / de_risk 0 / overweight 66 | mean 0.151 / active 0.0% | long 28 / short 0 / flat 72 | -0.86 | -0.010 | +1.67 | 4.01 | overweight反転で不可 |

## Route Probe Results

`route_dim=3` の exposure route 診断。recovery は route probe ではなく inventory recovery gate で見る。

| ID | CE | Acc | Macro-F1 | Active Recall | False Active | ECE | Top Active Adv |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 7 | 0.7995 | 0.646 | 0.531 | 0.515 | 0.181 | 0.055 | 0.011723 |
| Phase 8 | 0.7896 | 0.647 | 0.548 | 0.537 | 0.204 | 0.047 | 0.011624 |

## Per-Route Test Recall

| ID | Neutral | De-risk | Overweight |
|---|---:|---:|---:|
| Phase 7 | 0.819 | 0.725 | 0.096 |
| Phase 8 | 0.796 | 0.733 | 0.137 |

## 読み取り

### 1. Recovery分離だけでは足りない

Phase 7 は recovery を別headにしたが、recovery gate active は 0.0%。short 35%、flat 65% までは行くが、turnover 12.11 で churn が残った。

### 2. State machine gate は効いた

Phase 8 は underweight中の追加 de-risk を抑えたことで、turnover が 12.11 から 2.62 まで落ちた。shortも 35% から 17% に下がった。

これは、前に予想した通り「de_riskをどれだけ出すか」より「underweight状態でde_riskを連発しない」制約が重要だったという結果。

### 3. ACはまだ早い

AC tune 500 は test で short 35%、turnover 4.91 まで悪化した。AC micro 100 は long 28%、short 0% に反転し、AlphaEx -0.86、MaxDDΔ +1.67 で悪化した。

valではACが良く見える局面があったが、testでは汎化しない。これは critic / imagined reward が fold-specific に寄って、BCで作った安全な state machine policy を壊している可能性が高い。

## AC採用判定

ACは採用しない。

採用候補は Phase 8 BC のみ。

Phase 8 BC は compromise 条件にかなり近いが、以下が未達。

- SharpeΔ が -0.010 で境界。
- recovery gate active が 0.7% と低い。
- val 側のBC scoreが弱く、ACのval改善がtestへ移らなかった。

## 次にやるべきこと

次はACではなく、Phase 8 BCを少しだけ改善する。

優先順位:

1. recovery gateを 1〜3% へ上げる。
2. state machineに hysteresis を入れる。
3. route_dim=3 の overweight recallを改善する。
4. ACは Phase 8 BC が SharpeΔ >= 0、recovery gate >= 1%、val/testの方向が一致してから再開する。

具体案:

```yaml
inventory_recovery_scale: 1.25
inventory_recovery_logit_boost: 0.75
inventory_recovery_hard_threshold: 0.45
state_machine_underweight_min_duration: 16
state_machine_derisk_logit_down: 12.0
route_max_step_by_route:
  neutral: 0.0
  de_risk: 0.04
  overweight: 0.04
```

ただし recovery を上げすぎるとまた churn するので、同時に hysteresis/cooldown が必要。

## 生成物

- `configs/bcplan5_phase7_inventory_recovery_s007.yaml`
- `configs/bcplan5_phase8_state_machine_s007.yaml`
- `configs/bcplan5_phase8_ac_tune_s007.yaml`
- `configs/bcplan5_phase8_ac_micro_s007.yaml`
- `documents/logs/20260426_bcplan5_phase7_inventory_recovery_fold4.log`
- `documents/logs/20260426_bcplan5_phase8_state_machine_fold4.log`
- `documents/logs/20260426_bcplan5_phase8_ac_tune_fold4.log`
- `documents/logs/20260426_bcplan5_phase8_ac_micro_fold4.log`
- `documents/route_probe/20260426_phase7_route_probe_fold4.md`
- `documents/route_probe/20260426_phase8_route_probe_fold4.md`
- `documents/20260426_bcplan5_phase7_8_ac_results.md`