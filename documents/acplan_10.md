うん、Eは楽しそう。
でも採用しなかった判断は正しい。**Eは“攻めれば勝てる可能性”を示したけど、“安全に採用できる再現レシピ”ではまだない**。

今の状況はこう。

```text
A: seed7 WM/BC + seed7 AC
  AlphaEx +41.31 / SharpeΔ +0.026 / MaxDDΔ -0.25
  → 現Plan7 fold5 baseline

E: seed11 WM/BC + seed11 AC
  AlphaEx +62.04 / SharpeΔ +0.050 / MaxDDΔ +0.04
  → 攻め性能は強いがDD条件外

B/D/G:
  ACを乗せると悪化する組み合わせあり
```

だから、**BC/WM seed単体が主犯ではない**。seed11 WM/BCのACなしは `AlphaEx -0.08` でほぼB&H近似、崩壊はしてない。問題は、AC seedとWM/BC checkpointの組み合わせで **overweight fire timing** が変わり、良いfireにも悪いfireにも振れること。レポートでも、AC seedだけでなくWM/BC checkpointとの相性でfire timingが変わる、と整理されてる。

## Eの意味

Eは捨てるには惜しいけど、採用には早い。

```text
良い:
  AlphaEx +62.04
  SharpeΔ +0.050
  long 2%
  turnover 1.18
  fire_pnl +0.1080

悪い:
  MaxDDΔ +0.04
  fold5単体
  Plan7とは別のfire集合
  seed/selector overfit疑い
```

fire probeでも、Eは `fwd16 +0.00297` で一番良い一方、Aとのfire overlapは Jaccard `0.204` しかない。つまり、EはPlan7と同じことをして強くなったのではなく、**別のタイミング集合を拾って強くなってる**。これはチャンスでもあり、過学習リスクでもある。

しんじの言う、

> DD+になってからalpha+にするほうが賢い

これはかなり正しい。
今はAlphaを伸ばすより、**MaxDDΔ <= 0 の制約内でEみたいなfireを拾えるか**が次の勝負。

## 次にやるべきこと

次はAC拡張じゃなくて、**fire_pnl-aware checkpoint selector**。
性能が出るseed/checkpointはある。だから、actorを広げるより先に「良いfireを拾ったcheckpointだけ採用する選別器」を作る。

### Plan9-B: fire_pnl-aware checkpoint selector

AC中に複数checkpointを保存する。

```text
steps:
  250 / 300 / 350 / 400 / 450
```

各checkpointで出す。

```text
AlphaEx
SharpeΔ
MaxDDΔ
turnover
long/short/flat
fire_pnl
fire_forward_return
incr16
fire_rate
mean_delta
adapter fire overlap
```

採用条件はこれ。

```text
必須:
  MaxDDΔ <= 0
  AlphaEx > baseline
  SharpeΔ >= baseline
  fire_pnl > 0
  fire_forward_return > 0
  long <= 3%
  turnover <= 3.5
  short = 0%
```

Eは `MaxDDΔ +0.04` だけで落ちてる。
だから、E系列の途中checkpointに **Alphaは少し落ちるが MaxDDΔ <= 0 を満たす点** があるか探すのがかなり有望。

### Plan9-C: seed × checkpoint grid

fold5だけでまずやる。

```text
WM/BC seed:
  7, 11

AC seed:
  7, 11, 21

checkpoint steps:
  250, 300, 350, 400, 450
```

ただし採用はfold5単体でしない。
目的は、**良いcheckpointの条件を見つけること**。

見るべきはこれ。

```text
良いcheckpoint:
  fire_pnl > 0
  fwd16 > 0
  incr16 > 0
  MaxDDΔ <= 0

悪いcheckpoint:
  fire_pnlが正でもMaxDDが悪化
  fwd16は良いがdeltaが弱い
  fire timingがズレてDD寄与が悪い
```

### Plan9-D: Eを安全側へ寄せる

Eをそのまま採用じゃなくて、Eのfire集合を少し安全側にする。

候補：

```text
1. E checkpointの途中restore
2. Eのadapter deltaを少し縮小
3. Eのlong capを2%に固定
4. fire後のDD contribution guard
5. MaxDD悪化局面のfireを抑制
```

ただし、前の guarded sizing variant は悪化してる。単純に弱めるだけだとダメ。
なので、弱めるより **fire checkpoint selection** が先。

## 今のロードマップ更新

```text
Step 1:
  Plan7は維持

Step 2:
  Eは採用しないが、良いfire候補として研究対象に残す

Step 3:
  fire_pnl-aware checkpoint selectorを作る

Step 4:
  AC中checkpointを複数保存して、fire metricsで選ぶ

Step 5:
  fold5でMaxDD<=0かつAlpha改善のcheckpointを探す

Step 6:
  見つかったselectorをfold0/4/5に適用

Step 7:
  その後に全fold展開
```

まだ禁止は変わらない。

```text
禁止:
  route head unlock
  full actor AC
  advantage gate緩和
  floor > 1.0
  scale grid selector
  Q argmax actor update
```

## 判定

Plan9-Aはかなり有益。
「BC/WMが全部悪い」ではなく、**AC fire timing / checkpoint selectionが主犯** と切れた。

```text
分かったこと:
  seed11 WM/BC単体は崩壊しない
  ACを乗せると良くも悪くもfire timingが変わる
  Eのように強いfire集合も存在する
  でもDD制約を満たさないので採用不可

次:
  fire_pnl-aware checkpoint selector
```

Eは捨てるんじゃなくて、**“DD制約内に押し込めば伸びる可能性”の証拠**として使う。
まったく、こういうのが一番おもしろいのよ。Alphaを追う前にDDを通す、そこからAlphaを伸ばす。これが次の勝ち筋。
