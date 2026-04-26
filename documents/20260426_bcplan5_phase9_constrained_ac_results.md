# BC Plan 5 Phase 9 + Constrained AC Results

作成日: 2026-04-26
対象: Phase 9 / AC-0 to AC-3 constrained actor-critic curriculum
実行範囲: BTCUSDT 15m / seed 7 / fold 4 / 2018-01-01 to 2024-01-01
文字コード: UTF-8

## 結論

ACへ進む判断自体は妥当。ただし、現状のUniDreamでは「制限付きACは崩壊しないが、まだBCを改善できていない」。

現時点の採用ベースは引き続き Phase 8 state machine BC。

```text
Phase 8 / AC safe baseline:
short 16〜17%
flat 83〜84%
turnover 2.60〜2.62
AlphaEx +0.90〜+0.91 pt/yr
SharpeDelta -0.010〜-0.011
MaxDDDelta -1.59〜-1.61 pt
recovery gate active 0.7〜0.9%
```

今回のAC curriculumでは、AC-0/1/2は安全だが改善なし。AC-3でroute側を少し解放するとvalidationが悪化し、best checkpoint restoreでPhase 8相当に戻った。よって、AC-4 full actorへ進む条件は満たしていない。

## Web確認に基づく方針

実装前に offline actor-critic / TD3+BC 系の方向を再確認した。

- TD3+BC はoffline RLでactor更新にbehavior cloning lossを足し、policyがdataset/action priorから大きく外れないようにする設計。UniDreamでは`td3bc_alpha`、prior loss、段階的なactor解凍で同じ方向に寄せた。Source: https://github.com/sfujim/TD3_BC
- ICML 2024のoffline actor-critic scaling論文では、offline actor-criticは強いBC baselineから段階的に離れる選択肢として有望とされる。ただし大規模・多様なデータでの話なので、UniDreamではいきなりfull actorを動かさず、critic-only -> residual/adapter -> recovery -> routeの順に制限解除する方針にした。Source: https://openreview.net/forum?id=tl2qmO5kpD

## 実装した本流機能

### 1. AC critic-only mode

`ac.critic_only: true` を追加。actorを更新せずcritic pretrainだけ実行できる。

目的:

```text
actorを動かす前にcritic学習が破綻しないか確認する
BC policyの分布を壊さずACステージを検証する
```

対象:

- `unidream/actor_critic/imagination_ac.py`
- `unidream/experiments/ac_stage.py`

### 2. AC trainable actor prefixes

`ac.trainable_actor_prefixes` を追加。ACで更新するactor submoduleを限定できる。

今回の段階:

```yaml
AC-1:
  - route_delta_head

AC-2:
  - route_delta_head
  - inventory_recovery_head

AC-3:
  - route_delta_head
  - inventory_recovery_head
  - route_head
  - route_advantage_gate
```

目的:

```text
full actor更新を避ける
route/recovery/controllerの責務ごとに安全に解凍する
片側collapseが出た時に責任範囲を特定しやすくする
```

対象:

- `unidream/actor_critic/imagination_ac.py`

### 3. Phase 9 cooldown/hysteresis hook

`state_machine_recovery_cooldown_bars` と `cooldown_state_scale` を追加し、underweightから回復した直後にde_riskを抑制できる状態量を追加した。

ただし今回の実測では flat 100% に寄ったため、推奨本流設定には採用しない。コードはフラグ無効時に挙動を変えない安全hookとして残すが、現時点では実験用扱い。

対象:

- `unidream/actor_critic/actor.py`
- `unidream/experiments/bc_setup.py`

## 実行 Config

| ID | Config | 内容 |
|---|---|---|
| Phase 9 | `configs/bcplan5_phase9_cooldown_s007.yaml` | Phase 8 + strict recovery cooldown/hysteresis |
| Phase 9B | `configs/bcplan5_phase9b_cooldown_mild_s007.yaml` | Phase 8 + mild recovery cooldown |
| AC-0 | `configs/bcplan5_ac0_critic_only_s007.yaml` | actor freeze / critic only 250 steps |
| AC-1 | `configs/bcplan5_ac1_delta_only_s007.yaml` | route delta adapter only + critic |
| AC-2 | `configs/bcplan5_ac2_recovery_only_s007.yaml` | route delta + inventory recovery head + critic |
| AC-3 | `configs/bcplan5_ac3_route_lite_s007.yaml` | route/recovery lite unlock + critic |

## Backtest Results

| ID | Route dist | Recovery gate | Position dist | AlphaEx | SharpeDelta | MaxDDDelta | Turnover | 判定 |
|---|---|---|---|---:|---:|---:|---:|---|
| Phase 8 reference | neutral 35 / de_risk 65 / overweight 0 | mean 0.293 / active 0.7% | long 0 / short 17 / flat 83 | +0.91 | -0.010 | -1.61 | 2.62 | 現時点ベスト |
| Phase 9 strict | neutral 97 / de_risk 2 / overweight 1 | mean 0.313 / active 3.4% | long 0 / short 0 / flat 100 | -0.01 | +0.006 | +0.03 | 0.43 | flat 100で不可 |
| Phase 9B mild | neutral 98 / de_risk 1 / overweight 1 | mean 0.269 / active 0.6% | long 0 / short 0 / flat 100 | -0.03 | +0.003 | +0.03 | 0.43 | flat 100で不可 |
| AC-0 critic-only | neutral 35 / de_risk 65 / overweight 0 | mean 0.294 / active 0.9% | long 0 / short 16 / flat 84 | +0.90 | -0.011 | -1.59 | 2.62 | 安全、改善なし |
| AC-1 delta only | neutral 35 / de_risk 65 / overweight 0 | mean 0.293 / active 0.9% | long 0 / short 16 / flat 84 | +0.90 | -0.011 | -1.59 | 2.60 | 安全、改善なし |
| AC-2 recovery only | neutral 35 / de_risk 65 / overweight 0 | mean 0.293 / active 0.9% | long 0 / short 16 / flat 84 | +0.90 | -0.011 | -1.59 | 2.60 | 安全、改善なし |
| AC-3 route lite | neutral 35 / de_risk 65 / overweight 0 | mean 0.293 / active 0.9% | long 0 / short 16 / flat 84 | +0.90 | -0.011 | -1.59 | 2.60 | 解凍中に悪化、restoreで安全化 |

## Phase 9の読み取り

Phase 9の狙いは以下だった。

```text
recovery gate mild boost
hysteresis/cooldown
de_risk連発抑制
route_max_step縮小
```

結果は、strict/mildともに flat 100%。turnoverは0.43まで落ちたが、alphaが消えた。

```text
Phase 9 strict: AlphaEx -0.01 / flat 100 / recovery active 3.4%
Phase 9B mild:  AlphaEx -0.03 / flat 100 / recovery active 0.6%
```

つまり「de_risk連発抑制」は効きすぎると、risk controlではなく無取引policyになる。Phase 9は非採用。

## AC curriculumの読み取り

### AC-0: critic-only validation

critic pretrain lossは下がった。

```text
50/250  loss 0.8229
100/250 loss 0.5568
150/250 loss 0.5725
200/250 loss 0.1919
250/250 loss 0.1619
```

actorを動かさないためtest結果はPhase 8相当。critic学習自体は破綻していない。

ただし重要な制約がある。現在のCriticは`Critic(Value function)`で、入力は`z + h`のみ。厳密な`Q(s,a)`ではないため、「action別Qランキング」はまだ測れていない。今回のAC-0は、Q ranking検証ではなく「critic-onlyで分布を壊さないこと」の検証に留まる。

### AC-1: route delta adapter only

崩壊なし。ただし改善なし。

```text
AlphaEx +0.90
SharpeDelta -0.011
turnover 2.60
short 16 / flat 84
```

`route_delta_head`だけではtest時のpolicyを動かす力が弱い。

### AC-2: recovery controller only

崩壊なし。ただし改善なし。

recovery gateは active 0.9% のまま。Phase 6で見えた「recoveryは効くがchurnする」問題に対して、ACのgradientだけではrecovery使用率を1〜3%へ押し上げられていない。

### AC-3: route lite unlock

更新中はvalidationが悪化した。

```text
checkpoint 1: alpha -106.16pt / SharpeDelta -0.073 / short 18 / flat 82
checkpoint 2: alpha -102.57pt / SharpeDelta -0.066 / short 19 / flat 81
```

best checkpoint restoreによりtestはPhase 8相当に戻った。これは安全装置としては有効だが、AC-3の学習方向はまだ良くない。

## AC-4へ進まなかった理由

AC-4 full actorは実行しない判断にした。

理由:

- AC-1/2で改善が出ていない。
- AC-3でroute側を少し解放しただけでvalidationが悪化した。
- 以前のAC tune/microでは、short churn悪化またはoverweight反転が出ている。
- 現criticはstate-valueであり、action別の過大評価を直接検出できない。

この状態でfull actorを解放すると、過去と同じ片側解に落ちる確率が高い。

## 採用・非採用

### 採用

- Phase 8 state machine BCを現在の本流baselineにする。
- ACの`critic_only`を本流に入れる。
- ACの`trainable_actor_prefixes`を本流に入れる。
- ACはAC-0 -> AC-1 -> AC-2 -> AC-3の順でのみ解放する。
- full actor ACはguard付きでも、AC-3改善を確認するまで禁止。

### 非採用

- Phase 9 strict/mild cooldown設定は非採用。flat 100%に寄りすぎる。
- recovery hard thresholdを強くする方向は非採用。alphaが消える。
- route head full unlockは非採用。AC-3でvalidationが悪化。
- AC-4 full actorは非採用。現段階ではcollapse riskが高い。
- low-confidence neutral fallbackを強くする方向は非採用。これもflat化しやすい。

## 次の改善案

ACを続けるなら、次はfull actorではなくcritic/actorの責務を直すべき。

優先順位:

1. State-action criticを追加する。

```text
Q(s, action_candidate)
```

を明示的に評価できるようにする。現在は`V(s)`なので、action別にshort/flat/overweightのどれが良いかをcriticが直接比較できない。

2. True residual action adapterを追加する。

今のAC-1は`route_delta_head`だけで、最終positionに小さい残差を足す構造ではない。レビュー案の

```text
a_final = clip(a_BC + epsilon * delta_a_AC)
```

に近い専用moduleを作る方が筋が良い。

3. AC lossをaction distribution guardと直結する。

```text
short <= 25%
turnover <= 3.5
long <= 5%
recovery active 1〜3%
```

をvalidation rejectだけでなく、actor lossのsoft constraintへ入れる。

4. AC再開条件。

```text
AC-3 route liteでvalidation alpha/sharpeが悪化しない
または state-action criticでaction rankingがrealized advantageと正相関
```

このどちらかを満たしてからAC-4に進む。

## 生成物

- `configs/bcplan5_phase9_cooldown_s007.yaml`
- `configs/bcplan5_phase9b_cooldown_mild_s007.yaml`
- `configs/bcplan5_ac0_critic_only_s007.yaml`
- `configs/bcplan5_ac1_delta_only_s007.yaml`
- `configs/bcplan5_ac2_recovery_only_s007.yaml`
- `configs/bcplan5_ac3_route_lite_s007.yaml`
- `documents/logs/20260426_bcplan5_phase9_cooldown_fold4.log`
- `documents/logs/20260426_bcplan5_phase9b_cooldown_mild_fold4.log`
- `documents/logs/20260426_bcplan5_ac0_critic_only_fold4.log`
- `documents/logs/20260426_bcplan5_ac1_delta_only_fold4.log`
- `documents/logs/20260426_bcplan5_ac2_recovery_only_fold4.log`
- `documents/logs/20260426_bcplan5_ac3_route_lite_fold4.log`
- `documents/20260426_bcplan5_phase9_constrained_ac_results.md`
