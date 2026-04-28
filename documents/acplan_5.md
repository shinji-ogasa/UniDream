うん、これは**かなり重要な冷や水**。
結論から言うと、**fold4ではイケてるけど、まだ汎化してない**。

今の状態はこう更新。

```text id="d6lbh9"
fold4:
  Phase 8 + benchmark-gated overweight adapter は成功

fold0/fold5:
  ほぼflat/benchmark寄りになってB&Hの上昇を取り逃がす

結論:
  adapterは維持していい
  でもAC制限解除・trainable化・full actor unlockはまだダメ
```

fold4だけなら current adapter はかなり良い。

```text id="fr24v4"
fold4 current adapter:
  AlphaEx +1.15 pt/yr
  SharpeΔ +0.063
  MaxDDΔ -1.97 pt
  turnover 3.05
  long 1%
  short 16%
  flat 83%
```

Phase8 adapter off が `AlphaEx +0.90 / SharpeΔ -0.011 / MaxDDΔ -1.59` なので、fold4ではちゃんと上振れを足せてる。しかも advantage gate を外すと turnover `104.60` で即崩壊してるから、**predictive advantage gate は必須部品**。

ただし、fold展開が致命的。

```text id="h8wdh4"
fold0:
  AlphaEx -5.13
  flat 100%

fold4:
  AlphaEx +1.15
  pass

fold5:
  AlphaEx -51.36
  flat 98%

3fold avg:
  AlphaEx -18.45
  flat 93.7%
```

これはもう「fold4で勝った」ではなく、**fold4局所解の疑いが濃い**。特にfold0/fold5はturnoverやDDは悪くないけど、それは単に動いてないからで、B&Hのupsideを取り逃してる。レポートにも、fold0/fold5ではbase Phase8 policy自体が受動的になりすぎていて、adapterが主因ではないと整理されてる。

## 何が起きてるか

今の失敗は、前の short collapse じゃない。

```text id="6pdy2o"
昔:
  underweight / short に寄りすぎる

今:
  fold0/fold5で benchmark/flat に寄りすぎる
  upsideを取り逃がす
```

つまり、問題が変わった。

```text id="z78h5h"
リスクを避ける能力:
  ある

リスクを取るべき局面で張る能力:
  fold4では少しある
  fold0/fold5では弱い
```

これはモデルとしてはかなり自然な進化。
最初は暴走を止めるのが課題だった。今は止まりすぎて、相場によってはB&Hに負ける。

## AC Plan 4の判断

本流configを追加変更しなかった判断は正しい。

採用維持：

```text id="6z3j6y"
Phase 8 state machine BC
benchmark-gated overweight adapter
predictive advantage gate
hard safety
state machine
neutral fallback
```

未採用で正しい：

```text id="tuxthz"
advantage gate off
adv_min=0.5
epsilon=0.25
long_rate_max=0.01
full actor unlock
route head unlock
trainable adapter
```

`epsilon=0.25` や `long_rate_max=0.01` はfold4では良く見えるけど、fold外でのB&H劣後を説明できないから採用しない、という判断は妥当。

## いま一番大事な論点

**adapterの問題ではなく、fold0/fold5でPhase8本体がpassiveすぎる問題**。

だから次にやるべきことは、AC制限解除じゃない。
まずこれ。

```text id="qcimsy"
fold0/fold5で、
なぜ active / overweight / de-risk が出ないのかを分解する
```

見るべき診断はこれ。

```text id="d7x5hh"
1. Phase8 base vs adapter のfold0/fold5比較
2. adapter fire count / fire rate
3. predictive advantage[0] の分布
4. advantage_min=1.0 を超えたサンプル率
5. benchmark付近にいる時間率
6. benchmark付近なのにadapterが発火しない理由
7. fold別B&H return / trend regime
8. route dist / position dist / recovery gate active
```

特に重要なのは、

```text id="9iasmh"
fold5で flat 98% なのに AlphaEx -51.36
```

ここ。
これは、強い上昇局面でほぼ何もしなかった可能性がある。B&H比較なら、flat/低exposureは上昇相場で致命傷になる。

## 次のロードマップ

### Step 1: fold0/fold5のpassive原因診断

まず学習ではなく診断。

```text id="ll9bp5"
fold0/fold5:
  Phase8 adapter off
  current adapter on
  adapter fire attribution
  predictive advantage distribution
  route/position distribution
```

目的：

```text id="938833"
adapterが発火しないのか
Phase8がbenchmark付近まで戻らないのか
predictive advantage gateが厳しすぎるのか
そもそもWM predictionがfold外で死んでいるのか
```

### Step 2: fold別gate calibration

今の `advantage_min=1.0` がfold4にだけ合ってる可能性がある。

ただし単純に `adv_min=0.5` にするとfold4で turnover `20.97` なので危険。
だからやるなら、固定閾値じゃなくて分位点・regime補正。

```text id="qokf86"
advantage_min:
  fixed 1.0
  fold/train quantile 80/90/95%
  volatility-regime normalized threshold
```

### Step 3: benchmark exposure floor

fold0/fold5でflat/低exposureがB&Hに負けるなら、adapter以前に exposure floor が必要かもしれない。

```text id="85ybmb"
bull/trend regimeなら:
  minimum benchmark exposure を維持

ただし:
  MaxDD悪化チェック必須
```

これは危険だけど、B&H比較をするなら避けられない論点。

### Step 4: fold展開後にAC再検討

3fold平均AlphaExがマイナスのうちは、AC制限解除はダメ。
レポートでも、3fold AlphaExが正になるまでは restricted AC around benchmark-overweight sizing を再開しない方針になってる。

## 現在の正しいステータス

```text id="0yrxgu"
fold4:
  AC Plan 3 adapter は成功

3fold:
  失敗

本流:
  Phase8 + current adapter は維持

次:
  fold0/fold5のpassive化診断

やらない:
  full actor unlock
  trainable adapter
  fold4用のepsilon/long_rate最適化
```

まったく…ここで「fold4で+1.15だから勝ち！」って言ったら危ないわよ。
でも逆に、これは後退じゃない。**短期の課題が “collapse防止” から “fold外でupsideを取り逃がさない” に変わった**ってこと。ここまで来たのは普通に前進。
