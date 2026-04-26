結論：**次の本丸は「ACの制限解除」じゃなくて、その前に `Q(s, action_candidate)` を持つ state-action critic を作って、行動別に本当に価値比較できるかを検証すること**。
今のACは安全にはなったけど、改善しない。理由はかなり明確で、現criticが `V(s)` 寄りだから、`neutral / de_risk / overweight / residual action` のどれが良いかを直接比較できてない。報告書でも、現criticは `z+h` 入力のValue functionで、厳密な `Q(s,a)` ではなく、action別Q rankingが測れていないと整理されてる。

Web確認ベースでも、TD3+BCはoffline RLでpolicyがデータ分布から外れすぎないようBC項をactor updateに足す方向、IQLは未観測actionを直接評価しすぎずadvantage-weighted BCでpolicy抽出する方向、CQL/BCQ/BEAR系はoffline RLの分布外action・価値過大評価・bootstrapping errorを抑える方向。つまりUniDreamでも、いきなりactor自由度を上げるより、まず**行動候補の価値を保守的に比較できるcritic**が必要。([Emergent Mind][1])

## 現状の整理

今の採用baselineは **Phase 8 state machine BC**。数値は `short 16〜17% / flat 83〜84% / turnover 2.60〜2.62 / AlphaEx +0.90〜+0.91 / MaxDDΔ -1.59〜-1.61` でかなり安定してる。一方、AC-0/1/2は崩壊しないが改善なし、AC-3はroute側を少し解放しただけでvalidationが悪化し、restoreでPhase 8相当に戻った。

だから現状はこう。

```text
BC:
  かなり安定した安全policyを作れた

AC:
  制限付きなら壊れない
  でも改善方向の勾配が弱い/間違っている

次:
  actor解放ではなく、criticを行動比較可能にする
```

## AC最適化ロードマップ

### Phase AC-A: State-action critic probe

最初はactorを一切更新しない。
`Q(s, action_candidate)` を作って、行動候補のランキングが実現advantageと合うかを見る。

候補actionは最初は離散でいい。

```text
route candidates:
  neutral
  de_risk
  overweight
  recovery_controller_on/off

position candidates:
  current
  current - 0.05
  current + 0.05
  benchmark 1.0
  mild overweight 1.05 / 1.10
```

検証項目：

```text
1. Q rank IC
   Q順位とrealized cost-adjusted advantageの相関

2. top-decile Q advantage
   Q上位10%の実現advantageが正か

3. action別Q calibration
   de_risk/neutral/overweightを過大評価してないか

4. BC action vs candidate action
   Q(s, a_BC)より高い候補が本当に後で勝つか

5. one-sided bias check
   Qがde_riskやoverweightだけを常に高くしてないか
```

合格条件：

```text
Q rank IC > 0
top-decile realized advantage > 0
de_risk/overweight一極集中なし
Q上位候補のturnover/costが許容内
```

ここが通らないならAC actor更新はまだ禁止。まったく、criticが行動差を見えてないのにactorを動かすなんてバカよ。

---

### Phase AC-B: Conservative state-action critic training

AC-Aでrankingが少しでも正なら、criticを保守化する。
CQLはoffline RLで分布外actionの価値過大評価を抑えるために保守的なQを学ぶ発想で、BCQ/BEARも固定データ上の分布外action・bootstrapping errorを避けるためにaction制約を入れる。UniDreamではこれを「候補action critic + conservative penalty」として軽く入れるのが筋。([ScienceStack][2])

候補：

```text
B1. discrete route critic
  Q(s, route)

B2. position candidate critic
  Q(s, target_position)

B3. residual critic
  Q(s, delta_position)

B4. ensemble/min-Q critic
  Q = min(Q1, Q2)
  過大評価を抑える

B5. CQL-lite penalty
  dataset/BC action以外のQを上げすぎない
```

検証項目：

```text
1. Q overestimation gap
   Q高評価actionの実現advantageが本当に高いか

2. conservative gap
   Q(BC action) と Q(non-BC candidate) の差が暴れてないか

3. route別Q分布
   de_riskだけ高い / overweightだけ高いを検出

4. fold4 val/test一致
   valでQが良いactionがtestでも効くか

5. critic ensemble disagreement
   disagreementが高いactionをactorが選びすぎないか
```

合格条件：

```text
Q rankingがval/testで同方向
ensemble disagreement高いactionを避けられる
de_risk/overweight一極Qなし
```

---

### Phase AC-C: True residual action adapter

今のAC-1は `route_delta_head` だけで、最終positionへ小さい残差を足す構造ではない。報告書でも、次は `a_final = clip(a_BC + epsilon * delta_a_AC)` に近い専用moduleが筋と書かれてる。

実装案：

```text
a_BC = Phase 8 state machine policy output
delta_AC = residual_adapter(z, h, inventory_state)

a_final = clip(
  a_BC + ε * delta_AC,
  lower_bound,
  upper_bound
)
```

探索：

```text
ε:
  0.01
  0.02
  0.05

trainable:
  residual_adapter only
  critic
```

固定するもの：

```text
route_head
state_machine gate
inventory_recovery_controller
neutral fallback
position cap
turnover guard
```

検証項目：

```text
1. residual magnitude
   deltaが大きくなりすぎてないか

2. residual direction attribution
   deltaがどの局面で+/-に出ているか

3. BCからの乖離
   |a_final - a_BC| の分布

4. realized advantage by residual bucket
   delta上位/下位で実現advantageが改善するか

5. policy distribution
   short <= 25%, long <= 5%, turnover <= 3.5を維持するか
```

合格条件：

```text
AlphaEx >= Phase8 + 0.2pt
SharpeΔ >= 0
MaxDDΔ <= -1.0
turnover <= 3.5
short <= 25%
long <= 5%
```

---

### Phase AC-D: Distribution-constrained actor loss

validation rejectだけだと遅い。actor lossに制約を直接入れる。報告書でも、`short <= 25% / turnover <= 3.5 / long <= 5% / recovery active 1〜3%` をvalidation rejectだけでなくsoft constraintへ入れるべきと整理されてる。

loss案：

```text
actor_loss =
  - Q(s, a_final)
  + λ_bc * ||a_final - a_BC||²
  + λ_turnover * turnover
  + λ_short * max(0, short_rate - 0.25)²
  + λ_long * max(0, long_rate - 0.05)²
  + λ_recovery * recovery_target_loss
  + λ_kl * KL(route || route_BC)
```

検証項目：

```text
1. 制約項別の寄与
   Q改善で勝ってるのか、制約で抑えてるだけか

2. short/long/turnoverのtrain-val-test乖離

3. recovery gate active
   1〜3%に近づくか

4. Q改善と実現PnLの一致

5. constraint violation rate
   batch内だけ守ってtestで破るパターンを検出
```

合格条件：

```text
constraint violationがtestでも低い
Phase8よりAlpha/Sharpe改善
MaxDD優位を維持
```

---

### Phase AC-E: Limited route unlock

ここで初めてroute側を少し触る。
ただし、AC-3でroute lite unlockはvalidation悪化しているので、いきなりroute_head全体を解放しない。報告書では、AC-3でroute側を少し解放しただけでvalidationが悪化し、full actorはまだ禁止と判断されてる。

解放順：

```text
E1:
  route_advantage_gate adapter only

E2:
  overweight logit bias only

E3:
  de_risk logit temperature only

E4:
  route_head last layer only

E5:
  route_head full unlock
  ※E1-E4で改善した場合のみ
```

検証項目：

```text
1. route dist
   de_risk 65%固定から変化するか

2. overweight usage
   0% → 1〜5%程度に出せるか

3. de_risk過多の再発
   short 35%超えないか

4. route別realized advantage
   使い始めたrouteが実際に儲かっているか

5. val/test一致
   valだけ改善してtest悪化しないか
```

合格条件：

```text
overweight 1〜5%
short <= 25%
turnover <= 4
AlphaEx +1.5pt以上
SharpeΔ >= +0.02
MaxDDΔ <= -1.0
```

---

### Phase AC-F: Fold expansion before full actor

fold4で良くなっても、full actorにはまだ行かない。
offline actor-criticが強いBC baselineを上回り得るという報告はあるけど、それは大規模・多様なデータでの話。UniDreamのようなfold単位の金融系列では、fold4だけで自由度を上げるとfold固有のcriticに寄りやすい。([OpenReview][3])

検証順：

```text
1. fold4 AC-C/D/E
2. fold0 test-only
3. fold5 test-only
4. fold0/4/5 full run
5. 3fold平均でAC採用判定
```

検証項目：

```text
1. fold別AlphaEx
2. fold別MaxDDΔ
3. fold別turnover
4. fold別position dist
5. fold別route dist
6. fold別Q rank IC
```

採用ライン：

```text
3fold平均:
  AlphaEx > Phase8平均
  SharpeΔ >= 0
  MaxDDΔ <= 0
  turnover <= 4
  one-sided collapseなし

最低:
  3fold中2foldでPhase8を上回る
```

---

## 具体的な実験セット

### Experiment AC-A1: candidate Q probe

```text
目的:
  Q(s,a)が行動別advantageを順位付けできるか確認

実装:
  actor更新なし
  candidate actionごとにQを出す
  realized transition advantageと比較
```

出す結果：

```text
Q rank IC
top-decile realized advantage
route別Q平均
Q-selected action distribution
```

---

### Experiment AC-B1: CQL-lite candidate critic

```text
目的:
  Q過大評価を抑えたstate-action criticを作る

候補:
  min-Q ensemble
  candidate action CQL penalty
  BC action anchor
```

比較：

```text
V-only critic
Q(s, route)
Q(s, position candidate)
Q ensemble min
```

---

### Experiment AC-C1: residual adapter epsilon sweep

```text
ε = 0.01 / 0.02 / 0.05
trainable = residual_adapter only
state_machine = fixed
```

見るもの：

```text
Phase8から改善したか
short/long/turnoverが制約内か
deltaが特定方向に偏ってないか
```

---

### Experiment AC-D1: residual + distribution guard

```text
AC-C1 best
+ short_rate loss
+ long_rate loss
+ turnover loss
+ KL to BC route
```

見るもの：

```text
制約を守りながらAlpha/Sharpeが伸びるか
```

---

### Experiment AC-E1: overweight-lite unlock

```text
目的:
  Phase8のDD回避型policyに少量のupsideを足す

解放:
  overweight logit bias or adapter only

制約:
  long <= 5%
  turnover <= 4
  MaxDDΔ <= -1.0
```

---

## AC移行・停止条件

### 継続条件

```text
Phase8比で:
  AlphaEx +0.2pt以上
  SharpeΔ改善
  MaxDDΔ <= -1.0維持
  turnover <= 3.5〜4
  short <= 25%
  long <= 5%
  recovery active 0.7%以上維持
```

### 即停止条件

```text
short > 35%
long > 10%
turnover > 5
MaxDDΔ > -0.5
SharpeΔがPhase8より悪化
overweight route > 20%
Q-selected actionが一極集中
val改善 / test悪化が再発
```

### full actor解放条件

```text
AC-C/D/E のどれかで:
  fold4でPhase8を明確に上回る
  かつ fold0/fold5でも崩れない
  かつ Q rank ICが正

この3つを満たすまでfull actor禁止。
```

## 優先順位

```text
最優先:
  AC-A state-action critic probe
  AC-B conservative candidate critic

次:
  AC-C true residual action adapter
  AC-D distribution-constrained actor loss

その次:
  AC-E limited route/overweight unlock
  fold0/fold5展開

禁止:
  AC-4 full actor
  route head full unlock
  Phase9 cooldown継続
  V-only criticのままAC拡大
```

要するに、次は**ACを続けるけど、actorを解放するんじゃなくて、criticを `V(s)` から `Q(s,a)` に直して、行動候補の価値比較ができるか検証する**。
ここを通さずにACを強くしても、また `short churn` か `overweight反転` に戻るだけ。

[1]: https://www.emergentmind.com/papers/2106.06860?utm_source=chatgpt.com "Minimalist Offline RL Approach"
[2]: https://www.sciencestack.ai/paper/2006.04779v3?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning (arXiv:2006.04779v3) - ScienceStack"
[3]: https://openreview.net/forum?id=tl2qmO5kpD&utm_source=chatgpt.com "Offline Actor-Critic Reinforcement Learning Scales to Large Models | OpenReview"
