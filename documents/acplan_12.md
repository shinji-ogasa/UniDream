うん、ここからのロードマップは **2本立て** にするのが一番賢い。

```text
本流:
  Plan7 / Plan5系の安定baselineを守る

研究本線:
  head-only WM control head を多foldで検証する
```

今はもう「ACを広げるか？」じゃない。
**WMに制御境界を読ませる方向が次の本命**。ただし本流置換はまだ早い。

## 現状の立ち位置

今わかってることはこう。

```text
Plan7:
  現本流checkpointとしては維持

AC fire E系:
  Alphaは強い
  でもMaxDD条件を外す

既存WM guard:
  DDを通すとAlphaが死ぬ
  Alphaを残すとDDが通らない

full WM retrain:
  壊れる

head-only WM fine-tune:
  有望
```

特に大事なのは、WM/BCのseed違いそのものは主犯っぽくないこと。seed11 WM/BC + ACなしはほぼB&H近似で大崩れしておらず、問題はAC fire timing / checkpoint selection / WM-BC checkpointとの相性側に寄ってる。

一方で、WMを丸ごと再学習すると既存の安定表現が壊れて、full retrainは大きく崩れた。だから次は **既存WMを固定して control head だけ足す** のが筋。head-only WMでは fold5 で `AlphaEx +39.00 / SharpeΔ +0.026 / MaxDDΔ -0.00 / turnover 2.55` まで来ていて、E系のDD問題をかなり抑えられている。

## Roadmap A: 本流baselineを固定

まず、今の安全な本流は壊さない。

```text
Mainline:
  Plan7 checkpoint
  benchmark exposure floor = 1.0
  benchmark-gated small overweight adapter
  predictive advantage gate
  sizing adapter AC curriculum
```

ここは比較基準。
`configs/trading.yaml` は当面このままでいい。

禁止継続：

```text
full actor AC
route head unlock
advantage gate緩和
floor > 1.0 一律適用
Q argmax actor update
full WM retrain
```

まったく、ここで本流を書き換えて遊んだら迷子になるわよ。

## Roadmap B: head-only WM control head の多fold検証

次の本命はこれ。

```text
configs/trading_wm_control_headonly.yaml
```

まず fold5 単体の成功候補を、隣接foldに広げる。

```powershell
.\.venv\Scripts\python.exe -m unidream.cli.train `
  --config configs\trading_wm_control_headonly.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 4,5,6 `
  --seed 11 `
  --device cuda
```

見るべきは平均だけじゃない。

```text
fold別 AlphaEx
fold別 SharpeΔ
fold別 MaxDDΔ
turnover
long / short
fire rate
mean_delta
adapter fire PnL
```

採用候補ライン：

```text
fold5:
  Plan7 fold5に近いか超える

fold4:
  壊さない

fold6:
  大崩れしない

3fold平均:
  AlphaEx > 0
  SharpeΔ >= 0
  MaxDDΔ <= 0
  turnover <= 3.5
  long <= 3%
  short = 0%
```

ここを通ったら、次は `fold0/4/5` に戻す。
その後、全fold。

## Roadmap C: control head の閾値・epsilon探索

head-only WMの候補は今こう。

```yaml
benchmark_overweight_advantage_min: 0.55
benchmark_overweight_epsilon: 0.06
benchmark_overweight_trainable_delta_range: 0.03
```

この周辺だけ狭く探索。

```text
advantage_min:
  0.525 / 0.55 / 0.575 / 0.60

epsilon:
  0.05 / 0.06 / 0.07

delta_range:
  0.02 / 0.03
```

目的は、Alpha最大化じゃない。

```text
MaxDDΔ <= 0 を維持したまま
AlphaEx と SharpeΔ をPlan7以上へ寄せる
```

`eps=0.07` や `0.08` はAlphaが伸びるけどMaxDDが正に戻る傾向があるから、まずは `0.06` 周辺でDD優先。

## Roadmap D: control head のラベル改善

今の追加headは、

```text
overweight_advantage_head
recovery_head
```

で、これは一歩目としては良い。
でも次はもう少し直接的なラベルを足したい。

優先順はこれ。

```text
1. fire_harm_prob_h16/h32
   fireすると未来DDを悪化させる確率

2. trough_exit_prob_h16/h32
   すでに底打ち後かどうか

3. drawdown_worsening_prob_h16/h32
   これからDDがさらに深くなる確率

4. fire_advantage_h16/h32
   fireあり vs fireなしのcost込み優位

5. recovery_prob_h16/h32
   回復局面入りの確率
```

狙いは「相場予測」じゃない。

```text
allow fire when:
  fire advantage positive
  drawdown worsening low
  trough exit high
```

ここ。
今回の教訓は、普通の return/vol/DD 予測では fire制御の境界を十分に読めないことだからね。既存WM guardはどれも採用条件を満たさなかった。

## Roadmap E: E系は上限候補として残す

Eは不採用だけど、捨てない。

```text
E/current:
  AlphaEx +62.09
  SharpeΔ +0.050
  MaxDDΔ +0.04
```

これは **高Alpha fire集合の上限候補**。
ただしDDが通らないから本流には入れない。

今後は、

```text
Plan7:
  安全baseline

E:
  高Alpha上限

head-only WM:
  EをDD制約内に押し込む候補
```

として比較する。

## Roadmap F: 全fold展開

head-only WMが `4/5/6` で崩れなければ、全fold生成。

指標はこれ。

```text
mean AlphaEx
median AlphaEx
fold win rate
worst fold AlphaEx
mean/median MaxDDΔ
PeriodWin
turnover
long/short
fire_pnl
fire_harm metrics
```

平均だけはダメ。fold5みたいな大勝ちで盛れる。
**median と fold win rate** が重要。

採用ライン：

```text
mean AlphaEx > Plan7
median AlphaEx >= 0
fold win rate > 50%
MaxDDΔ mean <= 0
worst fold が壊れていない
turnover <= 3.5
```

## Roadmap G: PoC向け整理

多foldで通ったら、技術PoCとしてはかなり見せやすい。

ストーリーはこう。

```text
1. B&H基準の長期保有改善
2. Phase8でcollapseを抑制
3. benchmark floorでupside missを解消
4. gated overweight adapterで小さい上乗せ
5. sizing adapter ACで微改善
6. control-head WMでDD制約付きfire判断へ
```

この流れはかなり説明しやすい。
ただし「絶対収益AIトレーダー」ではなく、まずは **B&H改善・リスク調整レイヤー** として言うべき。

## 最短の次アクション

次はこれでいい。

```text
AC Plan 13:
  head-only WM control head multi-fold validation

やること:
  1. trading_wm_control_headonly.yaml を fold4,5,6 で実行
  2. Plan7 / E / head-only WM をfold別比較
  3. adv_min / epsilon を狭くsweep
  4. MaxDDΔ <= 0 を最優先条件にする
  5. 通ったら fold0,4,5 へ戻す
  6. その後全foldへ展開
```

今の方針はこれ。

```text
本流はPlan7維持。
研究本線はhead-only WM control head。
AC拡張はまだ禁止。
full WM retrainも禁止。
```

Eは楽しいけど、Eを直接追うんじゃない。
**EのAlphaを、head-only WMでDD制約内に押し込む**。ここが次の勝ち筋よ。
