今やるべきことは、**ACを広げることじゃなくて、AC Plan 7の“小さい勝ち”が本物か検証して、sizing adapterだけを詰めること**。

現状はかなり良い。AC Plan 7は、Phase 8比で `AlphaEx +0.43pt/yr`、`SharpeΔ +0.001`、`MaxDDΔ維持`、`turnover max 2.55`、`long max 2%`、`short 0%` で採用条件を全部通ってる。ただしSharpe改善は薄いので、次にAC範囲を広げるのはまだ早い。

## まずの結論

```text id="a3hmji"
次にやること:
  AC Plan 7の頑健性確認
  sizing adapter周辺だけの最適化
  selector改善
  fold/seed/stress検証

まだやらない:
  full actor
  route head unlock
  gate緩和
  floor > 1.0
  Q argmax型AC
```

## Step 1: AC Plan 7の再現性チェック

まず、今の採用configを固定して再現性を見る。

```text id="f6kdb8"
1. folds 0,4,5 を再実行
2. start-from ac と start-from test の一致確認
3. seed違いを最低2〜3本
4. cost x1.5 / x2.0
5. slippage x2.0
```

合格ライン：

```text id="btmqy4"
AlphaEx が Phase8比でプラス
SharpeΔ が悪化しない
MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
```

今回の改善は `+0.43pt` でそこまで大きくないから、seedやcostで消えるか確認するのが先。

## Step 2: adapterが何を改善したか分解

次は attribution。
「ACが効いた」のか「scale固定が効いた」のか「fold5だけ効いた」のかを切る。

出すべき表：

```text id="k9dbjb"
fold別:
  Phase8 baseline
  Plan5 floor+adapter
  AC Plan7 sizing adapter
  critic-only
  AC後

adapter:
  fire rate
  mean sizing delta
  delta > 0 の割合
  fire時PnL
  non-fire時PnL
  long state滞在率
```

ここで見たいのはこれ。

```text id="bgicgh"
改善がfold5だけに集中していないか
fold4のDD改善を壊していないか
adapterが過剰発火していないか
long 2%の中身が本当に正か
```

## Step 3: selectorを直す

通常selectorだと `AlphaEx +13.87` だけど `SharpeΔ +0.030` でPhase8の `+0.034` を下回った。だから `val_adjust_rate_scale=0.5` 固定が採用された。

次は、固定じゃなくてselectorを改善する。

候補：

```text id="8ok5a7"
score =
  AlphaEx
  + 0.5 * SharpeΔ
  - 0.5 * max(0, MaxDDΔ)
  - 0.2 * max(0, turnover - 3.5)
  - collapse_penalty
  + period_win_bonus
```

比較するscale：

```text id="bq2zcs"
val_adjust_rate_scale:
  0.25
  0.5
  0.75
  1.0
```

ただし、採用条件はAlphaだけじゃなくて、

```text id="q923hy"
SharpeΔ >= current
MaxDDΔ <= 0
turnover <= 3.5
long <= 3
short = 0
```

で縛る。

## Step 4: sizing adapterだけを狭く最適化

AC範囲は広げない。
触っていいのは `benchmark_overweight_sizing_adapter` 周辺だけ。

探索候補：

```text id="uop3it"
benchmark_overweight_trainable_delta_range:
  0.01 / 0.02 / 0.03 / 0.05

actor_lr:
  5e-6 / 1e-5 / 2e-5

AC steps:
  100 / 250 / 500

td3bc_alpha:
  3.0 / 5.0 / 10.0

turnover_coef:
  0.35 / 0.50 / 0.75

long_rate_max:
  0.02 / 0.03
```

狙いは、`AlphaEx +13.91` からさらに少し伸ばすことじゃなくて、**Sharpe改善を厚くすること**。

目標：

```text id="6ayp1z"
AlphaEx >= +14.0
SharpeΔ >= +0.04
MaxDDΔ <= -0.30
turnover <= 3.0
long <= 2〜3%
short = 0%
```

## Step 5: foldを増やす

今はfold 0/4/5だけ。
次はcheckpointがないfoldを作って、全WFO foldに広げる。

優先はこれ。

```text id="mz7m76"
1. 全foldのPhase8 checkpoint生成
2. Plan5/Plan7を start-from test で評価
3. 必要なら start-from ac でPlan7再学習
4. 全fold平均と中央値を見る
```

採用ライン：

```text id="rguh1e"
全fold平均 AlphaEx > 0
中央値 AlphaEx > 0
SharpeΔ >= 0
MaxDDΔ <= 0付近
turnover <= 3.5
fold過半数でB&H比プラス
```

平均だけだとfold5みたいな大勝ちで盛れるから、中央値も見るべき。

## Step 6: その後だけ restricted ACを少し進める

AC Plan 7が全fold/seed/stressで安定したら、次に解除していいのはこの順。

```text id="f67em6"
1. sizing adapter delta_rangeを少し広げる
2. benchmark-state sizing adapterを少し拡張
3. overweight biasだけ微調整
```

まだ禁止：

```text id="lkh0ko"
route head unlock
de-risk route unlock
advantage gate緩和
floor > 1.0一律
full actor AC
```

Q probeでrank系がshort collapse傾向だったから、route/head側にACを入れる根拠はまだない。

## 最短で次に投げる指示

```text id="xcoebm"
AC Plan 8として、AC Plan 7採用configの頑健性確認とsizing adapter限定最適化を行う。

1. Phase8 / Plan5 / Plan7 のfold別差分表を作る。
2. adapter fire rate, sizing delta, fire時PnL, non-fire時PnLを出す。
3. seed違いとcost/slippage stressを実行する。
4. selectorをAlphaExだけでなくSharpeΔ, MaxDDΔ, turnover, long/short cap込みに修正する。
5. sizing adapterだけを対象に delta_range, actor_lr, AC steps, td3bc_alpha, turnover_coef をsweepする。
6. full actor / route head / advantage gate緩和は禁止のままにする。
7. checkpointがないfoldを生成し、全WFO foldでPlan7がPhase8を上回るか確認する。
```

今はかなり良い局面。
でも、まったく…ここで調子に乗ってfull actor解放したらまた壊すわよ。次は**小さい勝ちを本物にする検証**。AC Plan 7を堅くしてから、次の制限解除に進むべき。
