うん、**BC設計としてはかなり良い方向**。
ただし今は「設計骨格は正しいが、route分類・recovery・active化の3点が未完成」って段階。

現状、true routing BCは direct relabel の high-turnover collapse をかなり抑えた。一方で、test時のrouteが neutral に寄り、recovery がほぼ0%、activeを強めるとDD/Alphaが悪化する。つまり次の最適化は **BCの工程ごとに分解して潰す** のが正解。

## BC最適化ロードマップ

### Phase 0: 診断基盤を固める

目的は、BC lossだけ見て満足しないこと。
今は「学習できてるように見えて neutral collapse」になりやすい。

最適化案：

```text id="08tq1k"
route classifier probe:
  route CE
  accuracy
  macro-F1
  active recall
  de_risk recall
  recovery recall
  overweight recall

route別評価:
  predicted route別 forward advantage
  route別 PnL
  route別 turnover
  route別 avg_hold
  route score分位別 realized advantage

遷移評価:
  transition matrix
  recovery latency
  recovery success rate
  underweight duration別の復帰率
```

優先度は最高。
ここがないと、どの改善が効いたか分からない。

合格目安：

```text id="mspccl"
macro-F1 >= 0.30
active recall >= 0.25
de_risk recall >= 0.20
overweight recall >= 0.20
recovery recall > 0
predicted active route上位decileの realized advantage > 0
```

---

### Phase 1: transition advantage label の再設計

今の transition advantage 方向は当たり。
ただし `best transition` をそのまま使うと壊れるので、label生成をもっと保守的にする。

工程：

```text id="kpaihj"
raw future path
↓
candidate action評価
↓
cost-adjusted advantage計算
↓
route label化
↓
soft target化
```

最適化案：

```text id="etwwwy"
1. horizon ensemble
   h=4,8,16,32 の重みを固定ではなく検証
   short horizonを強くしすぎるとturnover増加
   long horizonを強くしすぎるとflat化

2. neutral baseline強化
   adv(route) - adv(neutral) が小さいなら neutral
   marginを volatility regime別に変える

3. route別 margin
   de_risk margin: 小さめ
   recovery margin: 小さめ
   overweight margin: 大きめ
   これでlong collapseを防ぐ

4. route別 penalty
   overweight: drawdown/leverage penalty強め
   de_risk: opportunity cost penalty
   recovery: turnover penalty弱め
   neutral: no-trade prior
```

次に試すならこれ。

```yaml id="zax0k7"
transition_advantage_margin:
  neutral: 0.0005
  de_risk: 0.0003
  recovery: 0.0002
  overweight: 0.0010

transition_horizon_weights:
  h4: 0.15
  h8: 0.25
  h16: 0.35
  h32: 0.25
```

狙いは、**recoveryを出しやすく、overweightを出しにくくする**こと。

---

### Phase 2: route classifier を単体で強くする

今の最大問題はここ。
route labelには active があるのに、test予測が neutral に潰れる。

工程：

```text id="2r2zup"
latent z/h + state features
↓
route_head
↓
neutral / de_risk / recovery / overweight
```

最適化案：

```text id="o3papu"
1. class weight
   neutralを弱める
   recoveryを強める
   overweightは中程度

2. focal loss
   easy neutralを軽くする
   rare active/recoveryを学ばせる

3. balanced sampler
   batch内に de_risk / recovery / overweight を一定数入れる

4. two-stage route head
   まず active vs neutral
   次に active内で de_risk / recovery / overweight

5. route calibration
   temperature scaling
   neutral閾値調整
   active probabilityの校正
```

特におすすめは **two-stage route head**。

```text id="gfzxet"
stage1:
  neutral vs active

stage2:
  activeなら
    de_risk / recovery / overweight
```

理由は、4クラス一発分類だと neutral が強すぎる。
まず「動くべきか」を分けた方が安定する。

合格目安：

```text id="3c8xl6"
active recall >= 0.30
false active rate が暴れない
recovery recall > 0.05
overweight recall > 0.15
neutral precision >= 0.70
```

---

### Phase 3: predictive gate routing

predictive state直結はダメだった。
でも gate限定ならまだ本命。

工程：

```text id="irxd2h"
WM predictive heads
↓
route別 gate input
↓
route_head
```

最適化案：

```text id="lhw2cq"
risk preds:
  de_risk gateへ

return preds:
  overweight gateへ

drawdown / vol preds:
  de_risk抑制 or de_risk促進へ

current position:
  recovery gateへ

underweight duration:
  recovery gateへ

distance from benchmark:
  recovery / neutral gateへ
```

絶対にやらない方がいいのはこれ。

```text id="hv1tua"
predictive_state → final position head 直結
```

これはまた risk-off shortcut になる。

検証セット：

```text id="5ka9et"
M baseline
M + risk gate
M + return gate
M + recovery inventory gate
M + all gated predictive state
```

合格目安：

```text id="vv6xnp"
active recall 上昇
recovery route > 0%
turnover <= 4〜6
MaxDDΔ <= 0〜+0.2
flat <= 95%
```

---

### Phase 4: recovery専用最適化

ここが一番弱い。
今は train recovery label が少なく、test predicted recovery がほぼ0%。このままだと de-risk しても戻れない。

工程：

```text id="410ugf"
underweight状態
↓
recovery route判定
↓
benchmark方向へ small delta
```

最適化案：

```text id="2mfo84"
1. recovery oversampling
   recovery sampleをbatch内で増やす

2. recovery class weight
   5〜20倍から探索

3. synthetic inventory rollout
   teacherではなくmodelがunderweightに落ちた状態を作る
   そこから戻すlabelを付ける

4. underweight duration feature
   何本benchmarkから外れているかを入れる

5. recovery margin緩和
   recoveryだけadvantage marginを小さくする

6. recovery delta制御
   一気に戻さず 0.05〜0.10 ずつ戻す
```

最初にやるならこれ。

```yaml id="s076lu"
recovery_class_weight: [5, 10, 20]
recovery_oversample_rate: [0.10, 0.20, 0.30]
recovery_margin: [0.0001, 0.0002, 0.0003]
use_underweight_duration: true
```

合格目安：

```text id="b8dc6b"
predicted recovery route >= 1〜3%
recovery latency 改善
underweight後のbenchmark復帰率 上昇
turnover急増なし
MaxDD悪化なし
```

---

### Phase 5: route_delta_head の最適化

routeが決まった後、「どれだけ動かすか」の部分。
ここが強すぎるとturnover/DD悪化、弱すぎるとflat化。

工程：

```text id="mmykq7"
route選択
↓
route別 delta 出力
↓
current position + small delta
```

最適化案：

```text id="3aq76z"
1. route別 max_step
   neutral: 0
   de_risk: 0.05〜0.10
   recovery: 0.05〜0.10
   overweight: 0.03〜0.08

2. asymmetric delta
   overweightのstepは小さめ
   recovery/de_riskはやや大きめ

3. delta smooth loss
   急なposition変化を抑える

4. route-consistent delta loss
   de_riskならdelta <= 0
   recoveryならbenchmark方向
   overweightならdelta >= 0

5. min-hold制約
   route変更後、数barは頻繁に反転しない
```

候補：

```yaml id="cv3hb4"
route_max_step:
  de_risk: 0.075
  recovery: 0.075
  overweight: 0.050

delta_smooth_coef: [0.05, 0.10, 0.20]
route_consistency_coef: [0.25, 0.50]
min_hold_bars: [4, 8, 16]
```

合格目安：

```text id="c0cmlq"
turnover <= 4
avg_hold 上昇
flat <= 95
MaxDDΔ <= 0〜+0.2
```

---

### Phase 6: no-trade / turnover制約の最適化

no-tradeは効く。効きすぎるとflat 100%。
だから一律marginではなく、regime別・route別にするべき。

最適化案：

```text id="mfc76b"
1. volatility regime別 no-trade margin
   high vol: margin低めでde_riskしやすく
   low vol: margin高めで無駄取引しない

2. route別 turnover budget
   recoveryは少し許す
   overweightは厳しく
   neutralは動かさない

3. validation selector
   Alphaだけで選ばない
   Alpha - λ*turnover - μ*MaxDD悪化 で選ぶ

4. trade cooldown
   route反転にペナルティ
```

selectorはこれがいい。

```text id="3zcyio"
score =
  alpha_excess
  + 0.5 * sharpe_delta
  - 0.5 * max(0, maxdd_delta)
  - 0.2 * max(0, turnover - 4)
  - collapse_penalty
```

合格目安：

```text id="9py6tz"
turnover <= 4 推奨
turnover <= 6 妥協
flat 80〜95
active 5〜20
```

---

### Phase 7: BC最終判定とAC移行

ACはBCを救うものじゃない。
**まともなBC priorを軽く改善するもの**として使うべき。

BC推奨ライン：

```text id="3k4uj6"
fold4 BC:

AlphaEx >= +1.0 pt/yr
SharpeΔ >= +0.02
MaxDDΔ <= 0
turnover <= 4
flat 80〜92%
active 8〜20%
recovery route >= 1%
collapse_guard pass
route top-decile advantage > 0
```

妥協ライン：

```text id="qn9b2s"
AlphaEx >= 0
SharpeΔ >= -0.01
MaxDDΔ <= +0.2
turnover <= 6
flat <= 95%
active >= 3%
recovery route > 0
collapse_guard pass
```

禁止ライン：

```text id="iufk0j"
flat 100%
recovery 0%
turnover > 8
MaxDDΔ > +0.5
active routeの realized advantage <= 0
long/short/neutral 95%以上の一極集中
```

## 実行順ロードマップ

### Step 1: route classifier probe

まず学習可能性の診断。

```text id="3txtml"
目的:
  routeが本当に予測可能か確認

出力:
  macro-F1
  active recall
  recovery recall
  route別advantage
  calibration
```

ここで recovery が完全に無理なら、次はlabel/feature問題。

---

### Step 2: recovery補強

今一番欠けてるrouteなので優先。

```text id="fgs2gj"
recovery class weight
recovery oversampling
underweight duration feature
synthetic inventory rollout
```

目標：

```text id="h4yj17"
recovery route 1〜3%
flat 95%以下
MaxDD悪化なし
```

---

### Step 3: predictive gate routing

position直結ではなくgate限定。

```text id="ckhp0w"
risk → de_risk
return → overweight
inventory age → recovery
```

目標：

```text id="ntbcmy"
active recall増加
turnover <= 6
MaxDDΔ <= +0.2
```

---

### Step 4: two-stage route head

neutralが強すぎるなら構造で分ける。

```text id="5ejz84"
neutral vs active
↓
de_risk / recovery / overweight
```

目標：

```text id="5szv7e"
active recall >= 0.30
neutral collapse回避
```

---

### Step 5: route別 delta / no-trade 調整

activeが出たら、次は取引品質。

```text id="y2dx97"
route別max_step
min_hold
delta smooth
route別turnover penalty
```

目標：

```text id="xjgjc7"
turnover <= 4
MaxDDΔ <= 0
AlphaEx >= +1
```

---

### Step 6: fold4で合格後、fold0/fold5

fold4だけで過適合する可能性があるから、合格後に広げる。

```text id="gbt2ls"
fold4:
  設計探索

fold0/fold5:
  再現性確認

3fold平均:
  AC移行判断
```

---

## 最短でCodex/GLMに投げる指示

```text id="cjcw57"
次はACではなくBC最適化を継続する。

Phase 1:
route classifier probeを実装し、train/val/testで route CE, accuracy, macro-F1, active recall, de_risk recall, recovery recall, overweight recall, route別 realized advantage, calibration を出す。

Phase 2:
recovery routeが1%程度しかないため、recovery class weight / oversampling / underweight duration feature / synthetic inventory rollout を追加する。

Phase 3:
predictive stateをposition headへ直結せず、risk predsをde_risk gate、return predsをoverweight gate、inventory/current position/durationをrecovery gateに限定して入れる。

Phase 4:
neutral vs active のtwo-stage route headを試す。

Phase 5:
route別max_step、delta smooth、min_hold、route別turnover penaltyで turnover <= 4 を目標にする。

AC移行条件:
AlphaEx >= +1pt, SharpeΔ >= +0.02, MaxDDΔ <= 0, turnover <= 4, flat 80-92%, active 8-20%, recovery route >= 1%, collapse_guard pass。
妥協条件:
AlphaEx >= 0, MaxDDΔ <= +0.2, turnover <= 6, flat <= 95%, active >= 3%, recovery route > 0。
```

## 優先順位まとめ

```text id="ocbpb0"
最優先:
  route classifier probe
  recovery補強
  predictive gate routing

次点:
  two-stage route head
  route別delta制御
  no-trade/turnover selector

後回し:
  ACチューニング
  PatchTST大型化
  full short
  leverage拡張
```

今のBC設計は方向としてはかなりいい。
でも、まったく…ここで係数ガチャだけ続けるのはバカよ。
**次は「routeを見分けられるか」「recoveryを出せるか」「動いた時にturnover/DDを壊さないか」を工程別に潰す段階。**
