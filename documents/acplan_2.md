その通り。前の返答は粒度が足りない。
今必要なのは **「次はresidual BC」だけじゃなく、BC側の改善候補、ACへの移行条件、制限付きACの複数案、制限解除の実験計画** まで切ったロードマップ。

現状の事実はこう。

```text id="hsoow1"
Phase 8 BC:
  安定baseline

candidate Q probe:
  row Spearmanは一部で正
  でも Q-selected realized advantage は正にならない

MSE/CQL系:
  安全だが flat/anchor維持

rank系:
  short/long collapse再発

dynamic candidate:
  best possible advantage は存在する
  でも Q が選べていない
```

つまり、**候補行動には改善余地がある。でもQ argmaxで選ぶにはまだ危険**。だから次は「candidate advantageを使ったresidual BC」を本線にしつつ、効かなかった場合の分岐と、ACへ戻る道筋を用意する。

---

# 0. 現状の採用baseline

まず固定。

```text id="ae7o6d"
採用baseline:
  Phase 8 state machine BC

基準:
  AlphaEx +0.89〜+0.91 pt/yr
  SharpeΔ -0.010〜-0.011
  MaxDDΔ -1.58〜-1.61 pt
  short 16〜17%
  flat 83〜84%
  turnover 2.60〜2.65
```

以後の全実験は、このPhase 8を超えるかで判定。

やってはいけない比較：

```text id="wx6xg2"
AlphaExだけで勝ち判定
Q-selected actionをそのまま採用
rank CE argmax policy
full actor unlock
```

理由は、rank系が `short 100%` や `long 37%` に崩れてるから。

---

# 1. 現状の改善策：candidate advantage residual BC

## Phase R1: realized advantage residual BC

一番本線。

目的：

```text id="tl3ul5"
Qで選ばず、実現advantageで「明確に良かった残差候補」だけをBCで吸収する
```

候補：

```text id="2jws4f"
dynamic candidates:
  bc
  bc_minus_0.05
  bc_plus_0.05
  hold_current
  benchmark
  ow_1.05
  ow_1.10
```

教師：

```text id="7x45cj"
best_candidate = argmax(realized_cost_adjusted_advantage)

ただし:
  best_advantage - anchor_advantage > margin
の時だけ学習
```

loss：

```text id="7x1pke"
loss =
  base_BC_loss
  + λ_residual * w_adv * residual_BC_loss
  + λ_turnover * turnover_penalty
  + λ_dist * distribution_guard
```

重み：

```text id="z7z9la"
w_adv = clip(exp((adv_best - adv_anchor) / tau), 0, w_max)
```

検証項目：

```text id="4qq7f8"
selected residual dist
residual bucket別 realized advantage
short/flat/long dist
turnover
MaxDDΔ
AlphaEx
SharpeΔ
```

成功条件：

```text id="qjney3"
AlphaEx >= Phase8 +0.2pt
SharpeΔ >= 0
MaxDDΔ <= -1.0
turnover <= 3.5
short <= 25%
long <= 5%
```

失敗時の読み：

```text id="vs5jm5"
flat 100%:
  margin/tauが保守的すぎる

short増加:
  bc_minus候補が多すぎる / short cap不足

turnover増加:
  residual更新頻度が高すぎる

Alpha改善なし:
  candidate advantageが特徴から学習できてない
```

---

## Phase R2: support-filtered residual BC

R1で崩れる場合の保守版。

目的：

```text id="hkj1yu"
全サンプルでresidualを学ばず、信頼できる条件の時だけ動く
```

フィルタ案：

```text id="sxm9za"
1. realized advantage margin > m
2. route_probe active advantage top quantile
3. Q margin > 0 ただしQ argmaxは使わない
4. volatility regimeが極端でない
5. current positionがbenchmark近辺
6. turnover budget内
```

学習対象を絞る。

```text id="0tbnz5"
train residual only if:
  realized_adv > margin
  and candidate not one-sided collapse
  and turnover_budget_ok
```

検証項目：

```text id="jjh5k3"
採用サンプル率
採用サンプルの平均advantage
非採用サンプルとの差
testでのactive率
```

成功条件：

```text id="vqkf8x"
採用サンプル率 5〜20%
AlphaEx改善
turnover維持
short/long collapseなし
```

---

## Phase R3: AWR/IQL-style residual extraction

R1/R2の発展版。

目的：

```text id="lyq876"
best labelをhardに真似ず、advantageの大きい候補だけsoftに寄せる
```

候補確率：

```text id="ie6vx6"
p(candidate) ∝ exp(adv(candidate) / tau)
```

ただし、neutral/BC anchorを常に混ぜる。

```text id="x123j5"
p_final = (1 - β) * p_adv + β * onehot(anchor)
```

検証項目：

```text id="r9fmn5"
candidate entropy
anchor維持率
short/long率
top candidateのrealized advantage
```

狙い：

```text id="2uq5ny"
rank CEのような一極化を防ぎつつ、
MSE/CQLのflat維持より少し動かす
```

---

## Phase R4: residual BC + distribution guard

R1〜R3で少しでも改善が出たら入れる。

lossに直接入れる：

```text id="kdxcw1"
short_rate <= 25%
long_rate <= 5%
turnover <= 3.5
flat 70〜90%
```

soft loss：

```text id="nywdyl"
loss_dist =
  max(0, short_rate - 0.25)^2
  + max(0, long_rate - 0.05)^2
  + max(0, turnover - 3.5)^2
```

これはvalidation rejectだけじゃなくtrainにも入れる。

---

# 2. R系が効かない場合の選択肢

## Option B1: candidate setを変える

今のdynamic候補では `bc_minus_0.05` がbestに出やすい。
これがshort方向に寄りすぎるなら候補を変える。

候補案：

```text id="1r0wb2"
bc
bc_minus_0.025
bc_plus_0.025
benchmark
recover_to_benchmark
hold_current
```

一旦 `ow_1.25` や大きいminusは外す。
目的は「大きく勝つ」より「Phase8を少し改善する」。

---

## Option B2: residualではなく recovery controller を教師あり強化

candidate advantageが効かないなら、alpha改善より先にSharpe改善狙い。

```text id="nnznhz"
inventory_recovery_headだけを教師ありで再学習
```

教師：

```text id="rxc5yd"
underweight状態で
benchmarkへ戻した方がhorizon後に良い場合:
  recovery = 1
そうでなければ:
  recovery = 0
```

成功条件：

```text id="s4qoai"
recovery active 1〜3%
turnover <= 3.5
SharpeΔ >= 0
MaxDDΔ維持
```

---

## Option B3: routeを固定して position sizingだけ学習

route選択はPhase8で固定。
AC/BCが触るのは position size だけ。

```text id="v7gykx"
route = Phase8 output
size_multiplier ∈ {0.5, 0.75, 1.0}
```

狙い：

```text id="m9gbfb"
de_riskするかどうかではなく、
de_riskの強さを調整する
```

これはcollapseしにくい。

---

## Option B4: fold0/fold5へcandidate probeを広げる

fold4だけでcandidate advantageが見えている可能性もある。
R系の前に確認してもいい。

見るもの：

```text id="qa2u54"
best possible adv vs anchor
Q row Spearman
Q-selected realized advantage
best candidate distribution
```

判定：

```text id="pp76kz"
fold0/5でも best possible adv > 0:
  residual BCに進む価値あり

fold4だけ:
  fold4過適合疑い
```

---

# 3. 次段階のAC移行条件

ACに戻る条件は2段階に分ける。

## AC probe移行条件

actor更新なしのAC診断へ進んでよい条件。

```text id="6vwpdt"
R系またはcandidate probeで:
  best possible adv > 0
  selected/weighted candidate adv > 0
  selected distが一極集中しない
  short <= 30%
  long <= 10%
  turnover <= 4
```

## AC actor更新移行条件

実際にactorを動かす条件。

```text id="ff3yic"
Phase8比で:
  AlphaEx +0.2pt以上
  SharpeΔ改善
  MaxDDΔ <= -1.0
  turnover <= 3.5
  short <= 25%
  long <= 5%
  flat 70〜90%

かつ:
  candidate selected adv > 0
  Q rank / candidate rankが正
  val/testの方向が一致
```

妥協条件：

```text id="gr8oyw"
AlphaEx改善なしでも:
  SharpeΔ >= 0
  MaxDDΔ維持
  turnover維持
  recovery active 1〜3%
ならAC-0/AC-1のみ許可
```

禁止条件：

```text id="2g2ud1"
short > 35%
long > 10%
flat 100%
turnover > 5
Q-selected adv <= 0
rank系collapse再発
val改善/test悪化
```

---

# 4. 制限付きAC：複数手法

ここからがAC側。

## AC Method 1: Q-filtered residual actor

Q argmaxはしない。
Qは「使っていいサンプルのフィルタ」にだけ使う。

```text id="qkz3t5"
if Q(candidate) - Q(anchor) > q_margin
and realized_support_filter passed:
    train residual adapter
else:
    keep anchor
```

実装：

```text id="k9mcq8"
a_final = a_BC + ε * residual_adapter(s)
ε = 0.01〜0.03
```

検証：

```text id="r3mt9h"
Q-filter通過率
通過サンプルのrealized advantage
short/long dist
turnover
Phase8比改善
```

狙い：

```text id="jg9ky8"
Qを直接信用せず、補助的に使う
```

---

## AC Method 2: TD3+BC-lite residual

Phase8から小さい残差だけ学習。

```text id="ar330h"
actor_loss =
  - Q(s, a)
  + λ_bc * ||a - a_BC||²
  + λ_dist * distribution_guard
```

固定：

```text id="qtd9k2"
route_head
state_machine
inventory_recovery
neutral fallback
```

学習：

```text id="ktjlfv"
residual_adapter only
critic
```

制約：

```text id="t2vtsz"
ε <= 0.03
short <= 25%
long <= 5%
turnover <= 3.5
```

検証：

```text id="k2dnt1"
ε別改善
BC乖離分布
Q改善とrealized PnL一致
```

---

## AC Method 3: IQL/AWR-style actor extraction

criticからargmaxしない。
advantage-weighted BCとしてpolicy抽出する。

```text id="7su14o"
w = exp(clip(A(s,a) / tau))
loss = w * BC_to_candidate
```

候補actionはdataset/Phase8周辺だけ。

使うaction：

```text id="vs3tbu"
a_BC
bc_minus_0.025
bc_plus_0.025
recover_to_benchmark
```

狙い：

```text id="gxcl1j"
未観測actionを直接最大化しない
明確に良いcandidateだけ重く学ぶ
```

これは今の結果に一番合う。
Q argmaxが危険で、advantageが明確なサンプルだけ学ぶ方が安全という流れだから。

---

## AC Method 4: Conservative candidate policy improvement

候補集合内でだけpolicy改善。

```text id="0b7o2z"
π_new(a|s) ∝ π_BC(a|s) * exp(Q_conservative(s,a)/tau)
```

ただし cap を入れる。

```text id="1b5t0u"
max_prob(non_anchor) <= 0.20
max_short_candidate_rate <= 0.25
max_long_candidate_rate <= 0.05
```

検証：

```text id="wl6lj5"
candidate policy entropy
anchor維持率
non-anchor採用率
selected realized advantage
```

Qを使うけど、argmaxではなく soft improvement。

---

## AC Method 5: Critic-only model selection

actor更新ではなく、複数BC候補をcriticで選ぶ。

候補：

```text id="0idbxe"
Phase8
R1
R2
R3
recovery controller variant
position sizing variant
```

criticはpolicy内のaction選択ではなく、**config selection** に使う。

```text id="ibp5ms"
fold4 valでcritic scoreが高いconfig
かつ test safety条件を満たすものだけ採用
```

これは地味だけど安全。

---

# 5. ACの制御を段階的に外す実験

ここは「何を外すと何が壊れるか」を調べる実験として設計する。

## Stage 0: critic / Q probe only

actor更新なし。

目的：

```text id="ujz1qu"
criticが行動差を見られるか確認
```

解除なし。

失敗なら：

```text id="u5g6ta"
AC actor更新禁止
R系BCへ戻る
```

---

## Stage 1: residual adapter only

解除：

```text id="ej0fmr"
residual_adapter only
```

固定：

```text id="qt7k87"
route_head
state_machine
recovery_controller
position cap
turnover cap
```

見ること：

```text id="cf0bxc"
小さいposition補正だけでPhase8を超えるか
```

失敗パターン：

```text id="96knvp"
何も変わらない:
  residualが弱い / Q勾配弱い

short増加:
  residual方向が負に偏る

turnover増加:
  residualが頻繁に反転
```

---

## Stage 2: recovery controller unlock

解除：

```text id="8hktws"
inventory_recovery_head
```

固定：

```text id="x7234g"
route_head
state_machine
exposure route
```

目的：

```text id="vqil2m"
recovery active 0.7% → 1〜3%
```

見ること：

```text id="ozw5tg"
recovery使用率
recovery後のturnover
underweight duration
SharpeΔ
```

失敗なら：

```text id="yvcfv5"
churn増加:
  cooldown/hysteresisを再設計

flat化:
  recovery threshold強すぎ
```

---

## Stage 3: position sizing unlock

解除：

```text id="eijr4a"
route_delta / sizing adapter
```

固定：

```text id="ncqxj5"
route選択
state machine
```

目的：

```text id="dd8o36"
routeはそのまま、張り具合だけ改善
```

見ること：

```text id="4o1gt0"
de_risk時のposition深さ
benchmark復帰速度
turnover
MaxDD維持
```

---

## Stage 4: overweight bias unlock

解除：

```text id="xwqdxi"
overweight logit bias only
or overweight adapter only
```

目的：

```text id="n59r43"
DD回避型Phase8に少量のupsideを足す
```

制約：

```text id="i8vsak"
long <= 5%
overweight route <= 5%
MaxDDΔ <= -1.0
```

失敗なら：

```text id="4bh5jy"
overweight反転再発:
  overweight unlock禁止
```

---

## Stage 5: de_risk temperature unlock

解除：

```text id="7t7w2p"
de_risk logit temperature
```

目的：

```text id="h5grw3"
de_risk 65%を少し下げる
short 16%を維持または低下
```

制約：

```text id="ad3svd"
short <= 25%
MaxDDΔ <= -1.0
```

失敗なら：

```text id="sygydz"
MaxDD悪化:
  de_riskは固定に戻す
```

---

## Stage 6: route_head last layer unlock

解除：

```text id="s0bf5l"
route_head last layer only
```

条件：

```text id="lqvyhs"
Stage 1〜5のどれかでPhase8を明確に上回った場合のみ
```

見ること：

```text id="4n534i"
route distが壊れないか
de_risk/overweight一極化しないか
val/test一致するか
```

---

## Stage 7: full actor unlock

今はまだ禁止。
解除条件はかなり厳しくする。

```text id="2cbgx2"
fold4でPhase8超え
fold0/fold5でも崩れない
Q/candidate selected advantageが正
Stage 6でcollapseなし
```

それまではやらない。

---

# 6. 実験順ロードマップ

## Step 1: R系BCで改善余地を吸収

```text id="nwx3so"
R1 realized residual BC
R2 support-filtered residual BC
R3 AWR residual extraction
R4 distribution guard付きresidual BC
```

目標：

```text id="uud0nb"
Phase8 +0.2pt以上
SharpeΔ >= 0
turnover <= 3.5
```

---

## Step 2: R系が微改善したら制限付きAC

```text id="xw7m0j"
AC Method 1:
  Q-filtered residual actor

AC Method 2:
  TD3+BC-lite residual

AC Method 3:
  IQL/AWR extraction
```

目標：

```text id="fr3y0y"
Phase8を壊さず微改善
```

---

## Step 3: 制限を1つずつ外す

順番：

```text id="df4yol"
1. residual_adapter
2. recovery_controller
3. sizing_adapter
4. overweight_bias
5. de_risk_temperature
6. route_head last layer
7. full actor
```

各段階で、

```text id="q3k6zt"
何が改善したか
何が壊れたか
次にどの制限を戻すか
```

を記録する。

---

## Step 4: fold拡張

fold4だけで採用しない。

```text id="c3gtls"
fold4:
  設計探索

fold0/fold5:
  collapse検証

3fold:
  採用判定
```

採用条件：

```text id="70tez3"
3fold平均でPhase8超え
3fold中2foldでAlpha/Sharpe改善
全foldでcollapseなし
```

---

# 7. 次にCodex/GLMへ投げるならこれ

```text id="wrzwll"
次はAC actor更新ではなく、candidate advantage residual BCを複数案で検証する。

1. Phase8をanchorにして dynamic residual candidates を作る。
2. realized cost-adjusted advantageでbest candidateを作る。
3. marginを満たすサンプルだけ residual BC 補助lossを入れる。
4. AWR-style weighted residual BCも試す。
5. support filterとして Q margin / realized advantage margin / regime filter / turnover budget を試す。
6. distribution guardとして short<=25%, long<=5%, turnover<=3.5 をlossに入れる。
7. R系でPhase8を超えた場合のみ、制限付きACへ進む。

ACは以下の順に試す。
AC-1: Q-filtered residual actor
AC-2: TD3+BC-lite residual adapter
AC-3: IQL/AWR-style policy extraction
AC-4: conservative candidate policy improvement

制限解除は residual_adapter -> recovery_controller -> sizing_adapter -> overweight_bias -> de_risk_temperature -> route_head last layer の順。
full actor unlockは禁止。fold0/fold5で崩れないことを確認するまで実行しない。
```

## 最終判断

しんじの言う通り、必要なのは「residual BCだけ」じゃない。
正しい整理はこれ。

```text id="1gxmrt"
短期:
  candidate advantage residual BCでPhase8を微改善できるか確認

中期:
  Qをargmaxではなくfilter/weightとして使う制限付きAC

長期:
  residual -> recovery -> sizing -> route の順に制限解除

禁止:
  Q argmax
  rank CE extraction
  full actor unlock
```

つまり、次は **BC改善とAC移行を分ける**。
まず実現advantageで安全に候補改善を吸収し、それが少しでも効いたら、ACは「自由に学習」じゃなくて **どの制限を外すと壊れるかを見る実験装置** として進める。
