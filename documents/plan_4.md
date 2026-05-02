ごめん。これは完全にあたしが悪い。
「次のロードマップ」じゃなくて、しんじが求めてるのは **現状の課題を分解して、複数の実験候補を並べて、どれから潰すか決める探索ボード** だった。さっきの返しはまた単線に寄ってた。反省する。

じゃあ、ちゃんとやり直す。

---

## 現状の課題分析

今のPlan3 Round1で分かったことはこれ。

```text
Ridge overlay:
  fold4 / fold6 に薄く効く
  安全寄り
  ただし火力は小さい

WM overlay:
  fold5 に鋭く効く
  1トレードで +8.431pt
  ただし fold4/6 は完全neutral

共通:
  MaxDD悪化なし
  worst AlphaEx >= 0
  active率が低い
```

WM版は f456 で `AlphaEx mean +2.810`, `worst 0.000`, `MaxDDΔ mean -0.040`, `turnover max 0.2`。fold5だけ `+8.431` で、fold4/6は neutral。つまり、**WMは強いスポット検出器っぽいが、汎用overlayとしてはまだ薄い**。

今の本質課題は3つ。

```text
課題1:
  WM予測値のスケールが信用できるか不明

課題2:
  ridgeとWMが相補的なのか、たまたま別foldで当たっただけなのか不明

課題3:
  安全性を守るguardが強すぎて、fold4/6や他foldでneutralになりすぎている可能性
```

だから次は、**WM校正だけに決め打ちしない**。
複数仮説を並べて、低コスト検証で潰す。

---

# 実験候補A: WM予測値スケール校正

## 仮説

WMの `return/vol/drawdown` head は方向性を持っているが、utilityに入れる数値スケールがズレている。
特に `risk_target_scale=100` で訓練されているなら、DD/volを実績値と同じ単位として扱うと penalty が過大/過小になる可能性がある。

## 検証

```text
fold4/5/6:
  pred_return_h32 vs realized_return_h32
  pred_vol_h32 vs realized_vol_h32
  pred_dd_h32 vs realized_dd_h32

見る:
  Pearson / Spearman
  rank IC
  mean/std scale ratio
  quantile calibration
  top-decile realized outcome
  sign accuracy
```

## 判断

```text
returnだけICあり:
  return主導utilityへ寄せる

DD/volもICありだがscaleズレ:
  fold内z-score / percentile化してutilityへ

DD/volが読めない:
  DD/vol headをpenaltyに直接使わない
```

## 優先度

**最優先候補。**
WM overlayのfold5成功が本物か、ただのscale偶然かを切れる。

---

# 実験候補B: Ridge vs WM の相補性検証

## 仮説

ridgeとWMは同じ現象を別の表現で見ているのではなく、**別タイプのイベントを拾っている**。
ridgeは fold4/6 型、WMは fold5 型。なら ensemble で active fold を増やせる可能性がある。

## 検証

まずは重い学習なしで、既存scoreを組み合わせる。

```text
B1. OR:
  ridge_signal or WM_signal

B2. AND:
  ridge_signal and WM_signal

B3. WM primary + ridge veto:
  WMが発火。ただしridgeが危険判定なら止める。

B4. ridge primary + WM boost:
  ridgeが発火。WMが強い時だけ通す/size上げる。

B5. max utility:
  max(ridge_utility, wm_utility)

B6. percentile ensemble:
  ridge_score_pctile + wm_score_pctile - risk_pctile
```

## 見る指標

```text
f456:
  active folds
  AlphaEx mean / median / worst
  MaxDDΔ worst
  turnover max
  selected event count
  fold5の+8.431を保持できるか
  fold4/6のridge反応を保持できるか

all14:
  fold3依存
  fold5依存
  median
  fold win rate
```

## 判断

```text
ORで壊れる:
  信号は相補的だがguardが必要

ANDでneutral:
  2信号は別イベントを見ている

WM primary + ridge vetoが良い:
  WMは攻め、ridgeは安全確認

ridge primary + WM boostが良い:
  ridgeを本流、WMはスポット補強
```

## 優先度

**高い。**
今の発見を一番活かせる。

---

# 実験候補C: Blocked Event Attribution

## 仮説

fold4/6がneutralなのは、WMが読めていないからではなく、`danger_blocked` / `pullback_blocked` が止めすぎているだけかもしれない。

WM版では fold4で `danger_blocked 2078`、fold5で `pullback_blocked 201`, `danger_blocked 635` が出ている。
これが正しいブロックなのか、過剰抑制なのかを見ないといけない。

## 検証

blocked eventについて、

```text
counterfactual:
  blockしなかった時の AlphaEx寄与
  MaxDD寄与
  turnover寄与

event stats:
  future_return_h32
  future_dd_h32
  ridge utility
  WM utility
  triple-barrier label
  pullback label
```

分類する。

```text
correct block:
  blockしないとAlpha悪化 or MaxDD悪化

over-block:
  blockしない方がAlpha改善し、MaxDDも悪化しない

risky block:
  blockしないとAlphaは増えるがMaxDD悪化
```

## 判断

```text
fold4/6でover-blockが多い:
  guard緩和/階層化

fold5でcorrect blockが多い:
  現guard維持

risky blockが多い:
  blockではなくdelta縮小を試す
```

## 優先度

**高い。**
活性化不足の原因を切れる。

---

# 実験候補D: Guardを「消す」ではなく「縮小」に変える

## 仮説

今の guard は0/1で止めるから、active率が低くなる。
Hbestの時にも、alpha源とDDリスクが同じfire runに混在していた。だから、止めるより **deltaを縮小する** 方が良い可能性がある。

## 検証

```text
D1. danger_blocked を scale 0 にする現行

D2. danger_blocked を scale 0.25

D3. danger_blocked を scale 0.5

D4. pullback_blocked は scale 0固定、dangerだけ縮小

D5. dangerは縮小、pullbackは完全停止
```

## 見る指標

```text
active fold数
AlphaEx mean/median/worst
MaxDDΔ worst
turnover
fold4/6の活性化
fold5の+8.431保持
```

## 判断

```text
scale 0.25/0.5でfold4/6が動き、DD悪化なし:
  guardはhard blockではなくsoft throttleへ

DD悪化:
  hard guard維持
```

## 優先度

**中〜高。**
active率改善に効きそう。

---

# 実験候補E: WM Utility Sensitivity Grid

## 仮説

WMはfold5の良いイベントを拾えているが、penalty/thresholdがfold4/6に合っていない。

## 検証

小さく回す。

```text
dd_penalty:
  0.25 / 0.5 / 1.0 / 1.5 / 2.0

vol_penalty:
  0 / 0.25 / 0.5 / 1.0

threshold_floor:
  0 / 0.0005 / 0.001 / 0.002

cooldown:
  0 / 16 / 32
```

ただしガチャ化しないため、採用指標を固定。

```text
primary:
  worst AlphaEx >= 0
  worst MaxDDΔ <= 0
  turnover <= 3.5

secondary:
  active folds >= 2
  median AlphaEx > 0
  fold5を壊さない
```

## 優先度

**中。**
A/Cの診断後にやる方が良い。

---

# 実験候補F: Triple-Barrier / Meta-Labeling側を強化

## 仮説

WM utilityより、**ラベル側のイベント抽出** がまだ粗い。
金融MLでは、固定時間ラベルより triple-barrier / meta-labeling でイベントベースにする発想が一般的に使われる。De Pradoの *Advances in Financial Machine Learning* でも、Chapter 3で dynamic thresholds, triple-barrier, meta-labeling が扱われている。([O'Reilly Media][1])

## 検証

```text
F1. volatility-normalized triple-barrier
F2. asymmetric downside barrier
F3. CUSUM / DD-entry event sampling
F4. meta-labeling:
    primary = WM/ridge candidate signal
    secondary = fireしてよいか
```

## 見る指標

```text
false-active worst
selected event utility
MaxDD contribution
fold4/5/6再現性
```

## 優先度

**中。**
長期的には重要。今すぐoverlay改善にも効く可能性あり。

---

# 実験候補G: Safe Policy Improvement風のbaseline fallback

## 仮説

Plan7/Plan5が安全baselineとして機能しているなら、signal不確実な場所ではbaselineに戻すべき。
SPIBBは、batch RLで不確実な状態-actionではbaselineへ戻すことで安全改善を狙う考え方。([Microsoft][2])

## 検証

```text
G1. Plan7 fallback:
  overlay確信が低いならPlan7 position

G2. support-aware gate:
  過去に似たイベントが少ないならno-active

G3. confidence score:
  ridgeとWMが両方弱いならno-active
  片方だけ強い時は低size
  両方強い時だけ通常size
```

## 見る指標

```text
baseline deviation
fold worst
MaxDDΔ
active rate
turnover
```

## 優先度

**中〜高。**
安全性を維持するにはかなり筋がいい。

---

# 実験候補H: Conservative/OOD action penalty

## 仮説

F_listwiseやHbestが壊れたのは、offline RLでありがちな分布外action過大評価に近い。
CQLはOOD actionのQを保守的にすることでoffline RLの過大評価を抑える方向。([ScienceStack][3])

## 検証

まだACではなく、selector診断として。

```text
H1. action support penalty:
  過去類似状態で少ないposition変更をpenalty

H2. fire rarity penalty:
  rare fireは高utilityでも採用しにくくする

H3. candidate utility lower confidence bound:
  utility_mean - k * utility_uncertainty
```

## 見る指標

```text
selected event数
danger_fire_rate
turnover
worst AlphaEx
fold3/fold5依存
```

## 優先度

**中。**
AC再開前の安全selectorとして有望。

---

# 実験候補I: all14 WM overlay拡張

## 仮説

今のWM結果は f456だけなので、fold5偶然かもしれない。
all14で見ると別foldにもWMがスポット検出する可能性がある。

## 検証

```text
I1. WM overlay all14
I2. fold別 threshold/cooldown/blocked stats
I3. fold別 active count
I4. fold3/fold5依存チェック
```

## 判断

```text
WMが他foldにもスポット検出:
  ensemble本命

fold5だけ:
  WMは補助信号として扱う

壊れるfoldがある:
  WM単体採用禁止
```

## 優先度

**高い。**
WMを語るならall14が必要。

---

# 実験候補J: Stress / Robustness

## 仮説

overlayは低turnoverだからcostには強いはず。
ただし1トレード依存ならslippageやタイミングに弱い可能性がある。

## 検証

```text
J1. cost x1.5 / x2.0
J2. slippage x2.0
J3. threshold jitter
J4. execution delay 1 bar
J5. seed 11 / 21
```

## 優先度

**後段。**
候補が絞れてからでいい。

---

## 優先順位

しんじ、今ならこう進めるのが一番筋いい。

```text
Round 1: 診断
  A. WMスケール校正
  C. blocked event attribution
  I. WM overlay all14

Round 2: 統合候補
  B. ridge + WM ensemble
  D. hard block vs soft throttle
  G. Plan7 fallback

Round 3: 改善候補
  E. WM utility grid
  F. meta-labeling / triple-barrier改善
  H. conservative/OOD penalty

Round 4: 採用前
  fold3/fold5依存チェック
  stress test
  standalone overlay化
```

## 直近で投げるなら

```text
Plan3 Round 2として、以下を並列に検証する。

1. WM prediction calibration
   pred_return/vol/dd と realized h32 の correlation, rank IC, scale ratio, top-decile outcomeをfold4/5/6とall14で出す。

2. WM overlay all14
   fold別Alpha, MaxDD, active count, threshold, cooldown, pullback_blocked, danger_blockedを出す。

3. Blocked event attribution
   blockされたeventのcounterfactual PnL/MaxDDを出し、guardが正しいのか過剰抑制なのか分類する。

4. Ridge+WM ensemble
   OR, AND, WM primary + ridge veto, ridge primary + WM boost, max utility, percentile ensembleを比較する。

5. Soft throttle guard
   danger_blockedを0/0.25/0.5に縮小する設定を試す。

6. Plan7 fallback / support-aware gate
   overlay確信が低い場合はPlan7に戻す方式を試す。

まだAC/route unlock/config本流変更はしない。
```

これで、**WMが本当に使えるのか、ridgeと混ぜるべきなのか、guardが止めすぎなのか** が切れる。

さっきのは雑だった。
次からは、こうやって **課題分析→複数仮説→検証項目→優先順位** で出す。

[1]: https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml?utm_source=chatgpt.com "Chapter 3 Labeling - Advances in Financial Machine Learning [Book]"
[2]: https://www.microsoft.com/en-us/research/publication/safe-policy-improvement-with-baseline-bootstrapping-2/?utm_source=chatgpt.com "Safe Policy Improvement with Baseline Bootstrapping - Microsoft Research"
[3]: https://www.sciencestack.ai/paper/2006.04779v3?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning (arXiv:2006.04779v3) - ScienceStack"
