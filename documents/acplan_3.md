今回の結果で一番大事なのはこれ。

```text
AWR target自体は作れている
でも Phase 8 の route / state-machine / execution controller が
そのtargetを最終positionに反映していない
```

R1〜R3は安全だけどPhase 8とほぼ同じ、R4はlong 35%まで出せたけど `turnover 13.49 / AlphaEx -0.53` で崩壊、R5はlong targetを5%に絞っても実行longは0%で改善なし。つまり **「AWRを強くすると壊れる、弱くすると動かない」** が確定。

## 今の状況

現本流はまだこれ。

```text
Phase 8 safe baseline:
  AlphaEx +0.90〜+0.91
  SharpeΔ -0.010〜-0.011
  MaxDDΔ -1.59〜-1.61
  turnover 2.60〜2.62
  short 16〜17%
  flat 83〜84%
  long 0%
```

弱点は明確。

```text
DD回避はできる
turnoverも許容
でも upside / overweight が出ない
SharpeΔがまだ微マイナス
```

だから次の問いは、

```text
Phase 8の安全性を壊さずに、
benchmark付近へ戻った後だけ、少し上方向に張れるか？
```

これ。
まったく、やっと本丸が見えたわね。

---

# 次の本線：Benchmark-gated Small Overweight AC

## AC-OW0: overweight可能局面の事前probe

まずactorを動かさない。

目的：

```text
benchmark到達後だけ overweight する価値が本当にあるか確認
```

対象状態を絞る。

```text
current position >= 0.95
underweight_duration == 0 or small
recovery gate inactive
de_risk confidence low
turnover budget余裕あり
```

候補：

```text
anchor
anchor + 0.025
anchor + 0.05
min(anchor + 0.075, 1.10)
```

見る指標：

```text
candidate +0.025 / +0.05 の realized advantage
top-decile advantage
採用可能サンプル率
採用候補の平均holding length
MaxDD悪化寄与
```

合格条件：

```text
benchmark付近サンプルの top-decile overweight advantage > 0
採用可能サンプル率 3〜15%
MaxDD寄与が大きく悪化しない
```

ここがダメなら、overweight ACはまだ無理。
ここが良いなら進む。

---

## AC-OW1: small overweight adapter only

ここで初めて actor を少し動かす。

固定するもの：

```text
Phase 8 route_head
state_machine
inventory_recovery_controller
de_risk制御
neutral fallback
```

学習するもの：

```text
overweight_adapter only
state-action critic or filtered critic
```

形はこれ。

```text
a_final = a_Phase8 + gate_benchmark * ε * positive_delta
```

制約：

```text
ε = 0.01〜0.03
long <= 3%
turnover <= 3.2
short <= 20%
MaxDDΔ <= -1.2
```

重要なのは、**underweight状態からいきなりoverweightへ飛ばさない**こと。

```text
禁止:
  underweight -> overweight
  recovery中 -> overweight
  de_risk直後 -> overweight
```

許可：

```text
benchmark付近 -> mild overweight
```

---

## AC-OW2: TD3+BC-lite overweight residual

OW1で壊れなければ、少しだけACらしくする。

loss案：

```text
actor_loss =
  - Q(s, a_final)
  + λ_bc * ||a_final - a_Phase8||²
  + λ_turnover * turnover
  + λ_long * max(0, long_rate - cap)²
  + λ_dd * drawdown_penalty
```

初期値の思想：

```text
λ_bc 高め
λ_long 高め
ε 小さめ
critic更新多め
actor更新少なめ
```

見るもの：

```text
longが1〜3%だけ出るか
AlphaExが+0.2pt以上伸びるか
SharpeΔが0以上に戻るか
MaxDDΔを維持できるか
```

失敗パターン：

```text
long 0%:
  adapter弱すぎ / gate厳しすぎ

long > 5%:
  adapter強すぎ / cap不足

turnover増加:
  entry/exit hysteresis不足

MaxDD悪化:
  overweight条件が広すぎ
```

---

## AC-OW3: overweight bias unlock

OW2で良ければ、overweight routeの bias だけ少し解放。

解除するもの：

```text
overweight logit bias
or overweight gate adapter
```

まだ触らないもの：

```text
de_risk route
neutral route
recovery controller
state machine
full route_head
```

目的：

```text
long 0% -> 1〜5%へ
でも MaxDD を壊さない
```

条件：

```text
long <= 5%
turnover <= 3.5
short <= 20〜25%
MaxDDΔ <= -1.0
AlphaEx >= +1.1
SharpeΔ >= 0
```

---

## AC-OW4: benchmark-state sizing adapter

次はrouteではなく、position sizingだけ調整。

```text
if benchmark_state:
    size_multiplier = 1.00〜1.08
else:
    Phase8のまま
```

これはかなり安全寄り。
overweight routeを直接増やすより、benchmark状態で微妙に強く張るだけ。

見る指標：

```text
benchmark状態の平均position
long率
turnover
MaxDD
AlphaEx
```

---

# 段階的な制限解除

順番はこれ。

```text
1. overweight_adapter only
2. benchmark-state sizing adapter
3. overweight logit bias
4. overweight gate adapter
5. route_head last layer
6. full actor
```

ただし、**5以降はまだ禁止寄り**。
AC-3でroute側を少し解放しただけでvalidation悪化してるから、route本体を触るのはかなり後。

解除してはいけないもの：

```text
state machine
underweight de-risk suppression
turnover cap
long cap
short cap
neutral fallback
```

このへんは最後まで残す。
「制限解除」と言っても、安全柵を全部外すわけじゃない。

---

# 採用条件

AC-OW系の合格ラインはこれ。

```text
Phase 8比:
  AlphaEx +0.2pt以上
  SharpeΔ >= 0
  MaxDDΔ <= -1.0
  turnover <= 3.5
  short <= 25%
  long 1〜5%
  flat 70〜88%
```

理想はこれ。

```text
AlphaEx +1.2〜+1.5
SharpeΔ +0.02以上
MaxDDΔ -1.0以下
turnover 3.0前後
long 2〜4%
```

停止条件：

```text
long > 8%
turnover > 5
MaxDDΔ > -0.5
AlphaEx < Phase8
short > 35%
flat 100%
overweight反転
val改善 / test悪化
```

---

# 今回のAWR結果から得た教訓

AWRがダメだった理由は単純じゃない。

```text
targetはある
でも実行controllerが反映しない
```

R2ではtarget上はlong 27.38%まで出てるのに実行policyはlong 0%。R4では実行側を広げるとlong 35%まで出るが、turnover 13.49で壊れる。つまり、**overweight能力は存在するが、現controllerでは「安全に少量だけ出す」経路がない**。

だから次はAWRじゃなくて、

```text
benchmark付近だけ
小さいadapterだけ
hard cap付き
ACで少しだけ動かす
```

が筋。

---

# 最短指示文

Codex/GLMに投げるならこれ。

```text
次は realized candidate AWR BC ではなく、Phase 8をanchorにした benchmark-gated small overweight AC を実装する。

目的:
Phase 8のDD改善・turnover安定を維持したまま、benchmark到達後だけ small overweight を許可し、AlphaEx/Sharpeを改善できるか検証する。

実装:
1. Phase 8 actor/state-machine/recovery/de-risk制御はfreeze。
2. current position >= 0.95、underweight_duration small、de_risk confidence low の時だけ overweight gate を開く。
3. a_final = a_phase8 + gate * epsilon * positive_delta_adapter とする。
4. epsilon は 0.01/0.02/0.03 で探索。
5. long cap 3%、turnover cap 3.5、short cap 25%、MaxDD guardを入れる。
6. まず actor更新なしの overweight candidate probe を行う。
7. probeで top-decile overweight advantage が正なら、overweight_adapter only の constrained AC を実行。
8. 成功した場合だけ benchmark-state sizing adapter / overweight bias unlock に進む。
9. route_head full unlock と full actor ACは禁止。
```

結論：**AWRは切りで正解。次は限定ACへ戻る。ただしACの対象は「全actor」じゃなくて、benchmark到達後の small overweight adapter だけ。**
これならPhase 8の安全性を壊さず、long 0%問題を直接潰しにいける。
