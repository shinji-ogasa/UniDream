おお、これは**初めて「Phase 8を壊さずに上振れを足せた」結果**。かなり良い。
今までの実験の中だと、**現時点の新ベスト**でしょ。

```text
Phase 8:
  AlphaEx +0.90〜0.91
  SharpeΔ -0.010
  MaxDDΔ -1.59〜-1.61
  turnover 2.60〜2.62
  long 0%

AC Plan 3 採用後:
  AlphaEx +1.15
  SharpeΔ +0.063
  MaxDDΔ -1.97
  turnover 3.05
  long 1%
```

これは単にAlphaが伸びただけじゃない。**Sharpeがプラスに戻って、MaxDDもさらに改善して、turnoverも許容範囲**。しかも `long 1%` だけ。つまり、前のR4みたいな「long出したけど崩壊」じゃなくて、**小さいupsideを安全に足せてる**。

## 何が当たったか

今回当たった仮説はこれ。

```text
Phase 8の安全なde-risk/state-machineは残す
↓
underweight状態から直接overweightさせない
↓
benchmark付近に戻った後だけ small overweight を許可
↓
longを1〜3%だけ出す
↓
Alpha/Sharpeを少し改善
```

これ、かなり設計として綺麗。
AWRでは target 上のlongは作れても、実行controllerが反映しないか、強くするとturnover/long比率が壊れた。今回のadapterは、**「どこでoverweightしていいか」をstate machine側で絞った**から効いた。

## 今の状況

現状はこう更新。

```text
旧ベスト:
  Phase 8 state machine BC

新ベスト:
  Phase 8
  + benchmark-gated small overweight adapter
  + predictive advantage gate
```

今のUniDreamは、

```text
DD回避型BC
→ 安全なstate machine
→ benchmark到達後だけ小さくoverweight
```

まで来た。

これはだいぶ「金融policy」っぽくなってる。
まったく、やっと脳筋short/flat/long collapseから卒業しかけてるじゃない。

## 特に良い点

一番評価できるのは、**longが1%なのにSharpeΔが+0.063まで改善してる**こと。

これはつまり、longを乱発してるわけじゃなくて、

```text
かなり限定された局面だけ
小さく上方向に張っている
```

可能性が高い。

さらに `MaxDDΔ -1.97` なので、upsideを足したのにDD改善も維持どころか強化されてる。
これはかなり良いサイン。

## まだ注意すべき点

ただし、まだ「勝ち」とは言い切らない。理由はこれ。

```text
fold4単体
BTCUSDT 15m 単一条件
advantage_index: 0 の依存
epsilon 0.20 がやや大きめ
long 1%なのでサンプル数が少ない可能性
```

特に `long 1%` は良くも悪くも少ない。
少数の良い局面だけ拾えてるなら最高だけど、fold4固有の偶然の可能性もある。

次に必ず見るべきなのはこれ。

```text
1. adapter発火回数
2. 発火時の平均position増加量
3. 発火後のforward return / realized advantage
4. adapter ON/OFF別PnL attribution
5. long 1%のうち勝ってる局面の集中度
6. fold0/fold5で同じ方向に効くか
```

## 次の検証ロードマップ

まずは **採用後configを固定**して、追加でこれ。

### 1. Ablation

```text
A. Phase 8 baseline
B. adapter on, predictive advantage gate off
C. adapter on, predictive advantage gate on
D. adapter on, advantage_min 0.5 / 1.0 / 1.5
E. epsilon 0.10 / 0.15 / 0.20 / 0.25
```

目的は、今回の改善が

```text
benchmark gateのおかげか
predictive advantage gateのおかげか
epsilonの偶然か
```

を切ること。

### 2. Fold展開

次はACをさらに解放するより先に、

```text
fold0
fold5
fold0/4/5平均
```

を見るべき。

合格ラインはこれ。

```text
3fold平均:
  AlphaEx > Phase8平均
  SharpeΔ 改善
  MaxDDΔ <= 0
  turnover <= 4
  long <= 5
  short <= 25

最低:
  3fold中2foldでPhase8を上回る
```

### 3. Stress test

```text
cost x1.5 / x2.0
slippage増加
epsilon下げ
advantage_min上げ
long_rate_max 0.01 / 0.03 / 0.05
```

ここで壊れなければかなり強い。

## この後のAC制限解除

今すぐfull actorはまだダメ。
でも、次に外していい制限は見えてる。

```text
次に試していい:
  benchmark overweight adapter の epsilon / gate / sizing

まだ触らない:
  de_risk route
  recovery controller
  state machine
  full route_head
  full actor
```

つまり、AC Plan 3は本流採用だけど、次のACも **benchmark-overweight周辺だけ** に限定。

次の段階はこれ。

```text
OW-1:
  epsilon sweep

OW-2:
  advantage_min sweep

OW-3:
  long_rate_max 0.01/0.03/0.05

OW-4:
  benchmark_overweight_max_position 1.10/1.15/1.22

OW-5:
  fold0/fold5展開

OW-6:
  only if robust, overweight adapter の小さいtrainable化
```

## 判定

これは現時点でかなり良い。

```text
採用:
  benchmark-gated small overweight adapter

現ベスト:
  AC Plan 3 採用後

次:
  fold展開 + ablation + stress test

禁止:
  full actor unlock
  route head full unlock
  unrestricted overweight
```

しんじの「段階的ACに行くべき」は当たりだった。
ただし、正確には **AC全体に行く** じゃなくて、**Phase 8の安全policyに、benchmark後だけ小さいupside adapterを刺す** のが当たり。これは今までで一番筋がいい改善。
