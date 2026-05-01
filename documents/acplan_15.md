いい、次はかなり明確。
今は **AC拡張じゃなくて、fire-control label を作り直すフェーズ**。

Plan14で分かったのは、`fire_advantage` は全foldで正の信号がある一方、`fire_harm / drawdown_worsening / trough_exit` はfold間で安定しなかった、ということ。だから今は「fireして利益が出るか」は少し読めるけど、「fireしてDDを悪化させないか」はまだ弱い。

## 現状固定

本流はまだこれ。

```text
Plan7 / Plan5系 stable baseline
benchmark exposure floor = 1.0
small gated overweight adapter
predictive advantage gate
restricted sizing-adapter AC
```

まだ禁止。

```text
full actor AC
route head unlock
advantage gate緩和
floor > 1.0 一律
Q argmax actor update
WM head v2統合
```

---

# Plan15: fire-control label v2 を作る

## 目的

今のラベルはこうだった。

```text
fire_advantage:
  方向性あり

fire_harm:
  fold依存

drawdown_worsening:
  定義が粗い

trough_exit:
  弱い
```

なので、次は **fire_advantage_h32 を主軸**にして、DD系ラベルを作り直す。

---

## 15-A: fire_advantage_h32 を主scoreにする

これは一番見込みがある。

```text
fire_advantage_h32
= fireありの将来PnL - fireなしの将来PnL
  cost込み
  turnover込み
```

評価するもの。

```text
top10 realized advantage
top20 realized advantage
top-bottom spread
fold別の符号安定性
```

合格目安。

```text
fold4/5/6 全てで top10 > 0
top-bottom spread > 0
h32 が h16 より安定
```

Plan14では `fire_advantage top10` は全fold/horizonで正だったので、ここは本命。

---

## 15-B: trough_exit を捨てて recovery_slope に置換

今の `trough_exit` は弱い。
たぶん「未来troughがhorizon前半にある」みたいな定義がノイズすぎる。

置き換え候補。

```text
recovery_slope_h16/h32:
  fire時点からhorizon内で equity / price / benchmark-relative return が
  継続的に回復するか

post_trough_momentum:
  直近N本の下落後、次H本で上向きモメンタムが出るか

underwater_recovery_prob:
  underwater状態がhorizon内で改善するか
```

合格目安。

```text
AUC >= 0.55
top-decileで fire_advantage が正
DD worsening rate が低い
```

---

## 15-C: drawdown_worsening をhorizon相対にする

今の定義はfold間で安定してない。
固定閾値だとfoldのボラ差に弱い。

新定義。

```text
drawdown_worsening_rel_h32 = 1
if future_drawdown_deepen > k * rolling_vol
or future_drawdown_deepen > fold_train_quantile(q)
```

候補。

```text
k = 0.5 / 1.0 / 1.5
q = 70% / 80% / 90%
horizon = 16 / 32
```

見るもの。

```text
AUC
PR-AUC
low-risk top decile の MaxDD contribution
fold別安定性
```

---

## 15-D: fire_harm を分類ではなく ranking に変える

`fire_harm AUC` はfoldで揺れた。
だから「harmを当てる」より、**低harm順に並べた時に安全か**を見る。

評価。

```text
low_harm top10:
  fire_advantage > 0
  DD contribution <= 0
  MaxDD区間fire率が低い

high_harm top10:
  DD contributionが高い
```

つまり、AUCだけでなく **low-harm ranking quality** を見る。

---

# Plan16: combined fire score probe

Plan15のラベルを使って、複合scoreを作る。

```text
fire_score =
  + a * fire_advantage_h32
  - b * drawdown_worsening_rel_h32
  - c * fire_harm_rank
  + d * recovery_slope_h32
```

まずは学習せず、ridge/logistic probeでよい。

比較。

```text
score A:
  fire_advantage only

score B:
  fire_advantage - harm

score C:
  fire_advantage - DD worsening + recovery_slope

score D:
  all combined
```

合格条件。

```text
fold4/5/6で:
  selected fire top10/20 の realized fire advantage > 0
  selected fire の DD contribution が current より低い
  fire rate が過剰でない
```

---

# Plan17: inference-only fire guard

Plan16で良いscoreが出たら、E系やPlan7系に guard として刺す。

ルール例。

```text
allow_fire =
  fire_advantage_h32_score high
  and drawdown_worsening_score low
  and recovery_slope_score high
```

検証順。

```text
1. fold5 E
2. fold4/5/6 Hbest
3. Plan7 current
```

合格条件。

```text
fold4/5/6 mean:
  AlphaEx >= Plan7
  SharpeΔ >= Plan7
  MaxDDΔ <= 0
  turnover <= 3.5
  long <= 3%
  short = 0%
```

ここで通るまで、WM headにもACにも入れない。

---

# Plan18: WM control head v2

Plan17の inference guard が効いたら、初めてWMに統合。

追加head。

```text
fire_advantage_h32_head
drawdown_worsening_rel_h32_head
recovery_slope_h32_head
low_harm_rank_head
```

やり方。

```text
既存WM backbone freeze
standard predictive heads freeze
control heads only fine-tune
```

full WM retrainは禁止。前回、全体表現を動かすと壊れたから。

検証。

```text
fold4/5/6
fold0/4/5
その後 全fold
```

---

# Plan19: restricted AC再開

WM control head v2 が多foldで通ったら、ACを再開。

触っていいもの。

```text
benchmark_overweight_sizing_adapter only
```

固定。

```text
route head
full actor
benchmark floor
advantage gate
fire-control guard
```

目的。

```text
fireを増やすのではなく、
許可されたfireのsizeだけ調整する
```

合格条件。

```text
AlphaEx >= guard baseline +0.2
SharpeΔ >= baseline
MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
```

---

# Plan20: 全fold検証

最後に全fold。

見る指標。

```text
mean AlphaEx
median AlphaEx
fold win rate
worst fold AlphaEx
mean / worst MaxDDΔ
PeriodWin
turnover
long / short
fire_pnl
fire DD contribution
```

採用条件。

```text
mean AlphaEx > 0
median AlphaEx >= 0
fold win rate > 50%
mean MaxDDΔ <= 0
worst fold が壊れていない
turnover <= 3.5
long <= 3%
short = 0%
```

---

## 最短実行順

```text
Plan15:
  fire-control label v2
  fire_advantage_h32主軸
  recovery_slope追加
  drawdown_worseningをfold/vol相対化
  harmはlow-harm rankingで評価

Plan16:
  combined fire score probe

Plan17:
  inference-only fire guard

Plan18:
  WM control head v2

Plan19:
  sizing-adapter-only AC再開

Plan20:
  全fold検証
```

## 今すぐCodex/GLMに投げる指示

```text
次はPlan15として、fire-control label v2を実装する。

1. fire_advantage_h32を主scoreにする。
2. trough_exitを廃止し、recovery_slope_h16/h32, post_trough_momentum, underwater_recovery_probを追加する。
3. drawdown_worseningを固定閾値ではなく、rolling_volまたはfold train quantileで正規化する。
4. fire_harmはAUCだけでなく、low-harm rankingの品質を見る。
5. fold4/5/6で、top-decile fire_advantage、low-harm top-decileのDD contribution、combined scoreのfold別安定性を出す。
6. まだconfigs/trading.yamlは変更しない。
7. fire guard / WM head v2 / AC unlock はPlan15では実装しない。
```

一言で言うと、次は **「火をつけたら儲かるか」から、「儲かって、かつ火事にならないか」へラベルを作り直す段階**。
ここが通れば、次にようやくWM head v2とAC再開に進める。
