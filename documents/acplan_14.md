いいわよ。
今後のロードマップは、**「本流を守りながら、fireしていい局面を直接学習する」**に絞るべき。

今の結論はこれ。

```text
本流:
  Plan7 / Plan5系を維持

研究本線:
  fire_harm / trough_exit / drawdown_worsening / fire_advantage の
  直接control labelを作る

まだやらない:
  full actor AC
  route head unlock
  full WM retrain
  advantage gate緩和
```

head-only WM control headはfold5単体では期待があったけど、fold4/5/6平均で `AlphaEx -5.02`, `MaxDDΔ +0.32`, `turnover 3.65` になって不採用。つまり、今の `overweight_advantage / recovery` だけでは fire 境界を安定して読めてない。

---

## Phase 0: 本流固定

まず、現時点の安全baselineを固定。

```text
Mainline:
  benchmark exposure floor = 1.0
  small gated overweight adapter
  predictive advantage gate
  restricted sizing-adapter AC
```

これは **B&H改善レイヤー** として残す。

禁止継続：

```text
full actor AC
route head unlock
floor > 1.0 一律適用
advantage gate緩和
Q argmax actor update
full WM retrain
```

理由は、ここを触るとまた collapse / churn / fold過適合に戻りやすいから。

---

## Phase 1: 直接control labelを作る

次の本丸はこれ。

```text
fire_harm_prob_h16 / h32
trough_exit_prob_h16 / h32
drawdown_worsening_prob_h16 / h32
fire_advantage_h16 / h32
```

### 1. fire_harm_prob

目的：

```text
fireすると未来DDが悪化するか
```

例：

```text
fire_harm = 1
if adapter_on の future worst drawdown が
adapter_off より一定以上悪化する
```

これは一番重要。
今の問題は「fireのPnLが正でも、MaxDDを悪化させる」ことだから。

### 2. trough_exit_prob

目的：

```text
今がDDの底打ち後かどうか
```

fireは、DDの入口では危険。
でもDDの出口なら利益源になりやすい。

### 3. drawdown_worsening_prob

目的：

```text
これからDDがさらに深くなるか
```

これが高いならfire禁止。

### 4. fire_advantage

目的：

```text
fireあり vs fireなしのcost込み優位
```

単なるreturn予測じゃなくて、**adapterを使う意味があるか**を見る。

---

## Phase 2: label quality probe

いきなりWMに入れない。
まずラベルが予測可能かを見る。

入力：

```text
WM latent z/h
既存 predictive state
current drawdown
underwater duration
trailing return slope
vol regime
position / floor state
adapter gate features
```

出力：

```text
fire_harm_prob
trough_exit_prob
drawdown_worsening_prob
fire_advantage
```

見る指標：

```text
AUC
PR-AUC
top-decile realized fire advantage
top-decile fire harm rate
calibration
fold別再現性
```

合格ラインの目安：

```text
fire_harm AUC >= 0.58
drawdown_worsening AUC >= 0.58
trough_exit AUC >= 0.55
fire_advantage top-decile > 0
fold4/5/6で同方向
```

ここでAUCが出ないなら、WMに入れても無駄。

---

## Phase 3: inference-only fire guard

次に、学習済みモデルは変えず、guardだけ試す。

ルール：

```text
allow_fire =
  fire_advantage high
  and fire_harm low
  and drawdown_worsening low
  and trough_exit high
```

比較対象：

```text
Plan7 current
E current
E + fire_harm guard
E + trough_exit guard
E + combined control guard
```

採用条件：

```text
fold5:
  AlphaEx > Plan7
  SharpeΔ >= Plan7
  MaxDDΔ <= 0
  turnover <= 3.5
  long <= 3%
  short = 0%
```

ただし、fold5だけでは採用しない。
ここは「候補発見」まで。

---

## Phase 4: fold4/5/6 multi-fold check

Phase 3で良いguardが見つかったら、fold4/5/6へ。

見る指標：

```text
mean AlphaEx
mean SharpeΔ
mean MaxDDΔ
worst fold MaxDDΔ
turnover
long/short
fire rate
fire_pnl
fire_harm rate
```

採用条件：

```text
mean AlphaEx > Plan7
mean SharpeΔ >= Plan7
mean MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
fold5だけ勝ちではない
```

ここで落ちたら、またfold5局所解。

---

## Phase 5: WM head-only v2

inference guardで効いたら、初めてWM側に入れる。

やること：

```text
既存WM backbone freeze
standard predictive heads freeze
control headsだけ学習
```

追加head：

```text
fire_harm_head
trough_exit_head
drawdown_worsening_head
fire_advantage_head
```

ここでも full WM retrain は禁止。
前回、full WM retrain は既存の安定表現を壊したから。

探索範囲：

```text
head loss weight:
  0.5 / 1.0 / 2.0

horizon:
  h16
  h32
  h16+h32

guard threshold:
  conservative / balanced / aggressive
```

目的：

```text
fireの許可/禁止境界を直接読ませる
```

---

## Phase 6: sizing adapter ACに戻す

control head がmulti-foldで効いたら、restricted ACに戻す。

触っていいもの：

```text
benchmark_overweight_sizing_adapter only
```

固定：

```text
benchmark floor
advantage gate
fire_harm guard
state machine
route head
full actor
```

ACの目的：

```text
fireを増やすことではなく
許可されたfire内で sizing を微調整すること
```

採用条件：

```text
AlphaEx >= control-head guard baseline +0.2
SharpeΔ >= baseline
MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
```

---

## Phase 7: 全fold検証

ここまで通ったら全fold。

見るもの：

```text
mean AlphaEx
median AlphaEx
fold win rate
worst fold AlphaEx
worst fold MaxDDΔ
mean MaxDDΔ
turnover
PeriodWin
upside/downside capture
fire_harm attribution
```

平均だけ見るのは禁止。
fold5みたいな大勝ちで盛れるから。

採用ライン：

```text
mean AlphaEx > 0
median AlphaEx >= 0
fold win rate > 50%
mean MaxDDΔ <= 0
worst foldが壊れていない
turnover <= 3.5
long <= 3%
short = 0%
```

---

## 最短実行順

実際にはこれで進めるのがいい。

```text
Plan14:
  fire_harm / trough_exit / drawdown_worsening / fire_advantage label生成

Plan15:
  lightweight classifier probe
  AUC / top-decile / fold別再現性を見る

Plan16:
  E checkpointに inference-only combined guard を刺す

Plan17:
  fold4/5/6で multi-fold guard 検証

Plan18:
  head-only WM v2 に control heads を統合

Plan19:
  sizing-adapter-only AC を再開

Plan20:
  全fold検証
```

## 今すぐやるなら

次の一手はこれ。

```text
Plan14:
  direct fire-control label builder を実装する

出力:
  fire_harm_prob_h16/h32
  trough_exit_prob_h16/h32
  drawdown_worsening_prob_h16/h32
  fire_advantage_h16/h32

検証:
  fold5 E
  fold4/5/6
  AUC
  top-decile realized advantage
  MaxDD contribution
```

## 一言でまとめると

今のロードマップは、

```text
ACを広げる
```

じゃなくて、

```text
fireしていいかを直接判定する教師信号を作る
```

に切り替える。

Plan7で守りの本流はできた。
次は、**E系の高Alpha fireを、DD制約内で許可できる判定器を作る段階**。

まったく、ここまで来たら係数ガチャじゃないわよ。
次はラベル設計。ここが通れば一段上に行ける。
