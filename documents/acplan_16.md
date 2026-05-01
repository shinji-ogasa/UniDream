いいわよ。
次のロードマップは **「ラベル追加」だけじゃなくて、fireを種類別に分けて、AlphaとDDが両立するfireだけを許可する方向** が本線。

今の状況はこう。

```text
分かったこと:
  fire_advantage_h32 は使える
  でもDD安全性の判定が不安定
  combined scoreでは全foldでMDDリスクを下げられない

次の本質:
  fireを全部同じものとして扱わない
  fire type / regime ごとに分ける
```

Plan15では `fire_advantage_h32` のtop10/top20がfold4/5/6で正だった一方、low-harm rankingやcombined scoreはguard化できるほど安定していなかった。つまり「儲かるfire」は読めるが、「DDを壊さないfire」はまだ読めていない。

---

## Roadmap 0: 本流は固定

まず本流は触らない。

```text
Mainline:
  Plan7 / Plan5 stable baseline
  benchmark exposure floor = 1.0
  small gated overweight adapter
  predictive advantage gate
  restricted sizing-adapter AC
```

まだ禁止。

```text
禁止:
  full actor AC
  route head unlock
  advantage gate緩和
  floor > 1.0 一律
  Q argmax actor update
  WM head v2統合
```

今は攻めるより、**fire判定器の設計フェーズ**。

---

# Plan15-C: fire type / regime split

## 目的

今は全fireをまとめて判定してる。
でも実際にはfireには種類がある。

```text
1. recovery fire
   DD後の回復に乗るfire

2. trend continuation fire
   上昇継続に乗るfire

3. pre-DD dangerous fire
   DD直前に踏む危険fire

4. MDD-inside profitable fire
   最大DD区間中だがAlpha主エンジンでもあるfire

5. noise fire
   advantageもDD改善も弱いfire
```

これを混ぜると、
「Alphaを取るにはfireしたい」
「DDを守るにはfireしたくない」
が衝突する。

## 実装する診断

fire barごとに特徴を出す。

```text
features:
  current drawdown depth
  underwater duration
  trailing return slope 16/32
  trailing vol 64
  equity slope
  benchmark-relative equity slope
  position / adapter delta
  fire_advantage_h32
  post_fire_dd_contribution
  future_mdd_overlap
  MDD interval proximity
```

やること。

```text
1. fire barsをクラスタリング
2. clusterごとに advantage / DD / MDD overlap を集計
3. fold4/5/6で似たclusterが存在するか確認
4. AlphaとDDが両立するclusterだけ残す
```

合格条件。

```text
selected cluster:
  fire_advantage_h32 > 0
  post_fire_dd_contribution が低い
  MDD overlap が低い
  fold4/5/6で同方向
```

ここで「使えるfire type」が見つからなければ、Plan16 guardに進まない。

---

# Plan15-D: fire type別 scoring

Plan15-Cでclusterが見えたら、type別にscoreを作る。

例。

```text
recovery_fire_score =
  fire_advantage_h32
  + recovery_slope
  - drawdown_worsening

trend_fire_score =
  fire_advantage_h32
  + trailing_momentum
  - volatility_risk

danger_fire_score =
  future_mdd_overlap
  + post_fire_dd_contribution
  + pre_dd_state
```

最終的には、

```text
allow_fire =
  fire_advantage high
  and danger_score low
  and fire_type in {recovery, trend_continuation}
```

を見る。

この段階ではまだ本流に入れない。
probeだけ。

---

# Plan16: inference-only fire guard

Plan15-C/Dで使えるtypeが見つかったら、初めてguardを刺す。

比較対象。

```text
Plan7 current
E current
Hbest current
E + fire_type_guard
Hbest + fire_type_guard
```

guard候補。

```text
Guard A:
  fire_advantage_h32 top quantileのみ許可

Guard B:
  fire_type in recovery/trend のみ許可

Guard C:
  danger_score low かつ advantage high のみ許可

Guard D:
  MDD-risk cluster は deltaを0.25〜0.5倍に縮小
```

採用条件。

```text
fold4/5/6 mean:
  AlphaEx >= Plan7
  SharpeΔ >= Plan7
  MaxDDΔ <= 0
  turnover <= 3.5
  long <= 3%
  short = 0%
```

fold5だけ良くても不採用。
fold4/6を壊さないこと。

---

# Plan17: WM control head v2

Plan16の inference guard が効いたら、WM head化。

追加headはこれ。

```text
fire_advantage_h32_head
fire_type_head
danger_fire_head
future_mdd_overlap_head
post_fire_dd_contribution_head
```

重要なのは、**full WM retrainは禁止**。
既存WM backboneは固定。

```text
freeze:
  WM backbone
  standard predictive heads

train:
  control heads only
```

合格条件。

```text
fold4/5/6:
  inference guardと同じ方向に改善
  MaxDDΔ <= 0
  AlphaEx >= Plan7
```

ここで初めて「WMにfire制御境界を読ませる」段階。

---

# Plan18: restricted AC 再開

WM head v2 が通ったら、ACを再開してよい。
ただし触るのはここだけ。

```text
trainable:
  benchmark_overweight_sizing_adapter only
```

固定するもの。

```text
route head
full actor
benchmark floor
predictive advantage gate
fire_type_guard
WM control heads
```

ACの目的は、

```text
fireを増やすことではなく、
許可されたfireのサイズだけ微調整すること
```

採用条件。

```text
AlphaEx >= guard baseline +0.2
SharpeΔ >= baseline
MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
```

---

# Plan19: multi-seed / multi-fold

Plan18まで通ったら、seedとfoldを広げる。

順番。

```text
1. fold4/5/6 seed11
2. fold0/4/5 seed7
3. fold4/5/6 seed7/11
4. 全fold
```

見る指標。

```text
mean AlphaEx
median AlphaEx
fold win rate
worst fold AlphaEx
worst fold MaxDDΔ
mean MaxDDΔ
turnover
long / short
fire type distribution
fire_pnl
fire DD contribution
```

平均だけは禁止。
fold5みたいな大勝ちで盛れるから、medianとfold win rateを見る。

---

# Plan20: PoC向け整理

ここまで通ったら、PoCとしてかなり見せやすい。

ストーリー。

```text
1. B&H floorでupside missを防ぐ
2. gated overweightで限定的に上乗せ
3. sizing adapter ACで微改善
4. fire type guardでDD悪化fireを抑制
5. WM control headでfire判断を安定化
```

売り方は、

```text
絶対収益AIトレーダー
```

ではなく、

```text
B&H長期保有を改善するrisk-adjusted overlay
```

が一番筋いい。

---

## 最短実行順

```text
Plan15-C:
  fire type / regime clustering probe

Plan15-D:
  fire type別 score probe

Plan16:
  inference-only fire guard

Plan17:
  WM control head v2

Plan18:
  sizing-adapter-only AC

Plan19:
  multi-fold / multi-seed

Plan20:
  PoC整理
```

---

## 今すぐ投げる指示文

```text
次はPlan15-Cとして、fireを一括で扱わず、fire type / regime別に分解する診断を実装する。

やること:
1. fire barごとに current drawdown, underwater duration, trailing return slope, vol, equity slope, adapter delta, fire_advantage_h32, post_fire_dd_contribution, future_mdd_overlap を出す。
2. fire barsを clustering または rule-based binning で分類する。
3. 各fire typeごとに fire_advantage_h32, post_fire_dd_contribution, MDD overlap, fire_pnl をfold別に集計する。
4. fold4/5/6で、Alphaが正かつMDD寄与が低いfire typeが再現するか確認する。
5. まだinference guard / WM head / AC unlockは実装しない。
6. configs/trading.yamlは変更しない。
```

一言でまとめると、

```text
次は「fireの良し悪し」を直接当てるより先に、
fireの種類を分けろ。
```

儲かるfireとDDを壊すfireが同じラベル空間に混ざってるから、今のままだとguardがAlphaを削りすぎる。
**fire typeを分けて、AlphaとDDが両立するtypeだけ拾う**。これが次の最短ルートよ。
