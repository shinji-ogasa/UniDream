いい。ここからは **「ACを強くするロードマップ」じゃなくて、「再現性ある教師・BC・ACへ戻すロードマップ」** に切り替えるべき。

今の根本結論はこれ。

```text
今のroute/fire系の失敗原因:
  モデル構造より、教師ラベルと選択基準が不安定

特に:
  current route label は teacher inventory shortcut に寄っている
  benchmark label にすると市場状態だけではほぼ読めない
  Hbest/fire系は post-hoc guard で救えない
```

route separability では `context` や `wm_context` が一見強いけど、改善の主因は current position / inventory state で、`wm_advantage` や `wm_regime` 単体はほぼランダム寄り。つまり、今のroute教師は「市場状態」ではなく「teacherのposition状態」をかなり読んでいる疑いが強い。
さらに benchmark-state label にすると mean AUC がだいたい `0.52` 前後まで落ちるので、単純に benchmark position でラベルを作り直しても clean な market-state label にはならない。

だから長期ロードマップはこれ。

```text
Phase 1: route teacher を市場イベント単位に再設計
Phase 2: label separability probe
Phase 3: two-stage route gate を再設計
Phase 4: BC-only 安定化
Phase 5: fire-aware checkpoint selector
Phase 6: risk-sensitive / safe improvement 学習
Phase 7: restricted AC 再開
Phase 8: multi-fold / multi-seed / stress 評価
Phase 9: WM control head v2 は後段
Phase 10: PoC化
```

---

# Phase 1: route teacher 再設計

次の本丸。
今の route label は teacher inventory path に依存しすぎている。

だから route をいきなり `neutral / de-risk / recovery / overweight` の4classで作らない。
まず **市場イベントごとのclean label** に分ける。

## 1-A. risk-off / de-risk event label

目的：

```text
未来のDD窓に入る前に、
de-riskするとB&HよりMaxDDを一定以上減らせる局面を検出する
```

候補ラベル：

```text
risk_off_event = 1
if future drawdown window occurs
and reducing exposure would reduce future max drawdown by threshold
and avoided drawdown > cost + margin
```

検証項目：

```text
AUC
PR-AUC
top-decile avoided drawdown
false-active rate
fold別 worst AUC
event density
```

合格目安：

```text
fold4/5/6 worst AUC >= 0.60
top-decile avoided DD > 0
false-active worst <= 0.15
event density が極端に低すぎない
```

## 1-B. recovery / re-entry event label

目的：

```text
DD後に benchmark へ戻す価値がある局面を検出する
```

候補ラベル：

```text
recovery_event = 1
if current drawdown is non-trivial
and future return/recovery slope is positive
and returning to benchmark improves future PnL without worsening DD
```

検証項目：

```text
recovery AUC
underwater duration別の性能
future recovery slope top-decile
re-entry後のturnover
false recovery rate
```

合格目安：

```text
AUC >= 0.58
top-decile future recovery > 0
DD再悪化率が低い
fold間で同方向
```

## 1-C. trend / overweight event label

目的：

```text
B&H以上に少し張ってよい局面を検出する
```

候補ラベル：

```text
overweight_event = 1
if fire_advantage_h32 > margin
and future DD worsening is low
and volatility / drawdown state passes filter
```

ここは Plan15 の結果を使う。`fire_advantage_h32` は全foldで top10/top20 が正だったので、攻め側の主信号としては有望。ただし harm/DD/recovery はまだ guard 化できるほど安定していない。

検証項目：

```text
fire_advantage_h32 top10/top20
top-bottom spread
top-decile MDD overlap
post-fire DD contribution
fold6のMDD区間率
```

合格目安：

```text
top10/top20 advantage > 0 in all folds
MDD overlap がHbestより低い
post-fire DD contribution が悪化しない
fold6で破綻しない
```

## 1-D. neutral label

neutral は「その他」ではなく、**activeすべき根拠が足りない状態**として定義する。

```text
neutral = not risk_off
          and not recovery
          and not overweight
          and uncertainty high or margin insufficient
```

検証項目：

```text
neutral precision
active false positive
active density
turnover
neutral collapseしていないか
```

---

# Phase 2: label separability probe

ラベルを作ったら、すぐBCに入れない。
まず separability probe。

入力は分けて評価。

```text
raw
WM latent
predictive state
context without current position
context with current position
raw + WM
raw + WM + context
```

重要なのは、**current position を主特徴にしないこと**。
position は controller state として後段で使う。ラベルを当てる主因にしてはいけない。

検証項目：

```text
1. active/no-active AUC
2. active AP
3. recall under false-active cap
4. false-active worst
5. predicted-active rate
6. one-vs-rest AUC
   - risk_off
   - recovery
   - overweight
7. calibration / ECE
8. fold間 worst
```

合格条件：

```text
active/no-active worst AUC >= 0.65
false-active worst <= 0.15
recall worst >= 0.25〜0.35
overweight one-vs-rest AUC >= 0.65
de-risk one-vs-rest AUC >= 0.70
fold4/5/6 と fold0/4/5 で同方向
```

ここを通らないなら、BCへ進まない。

今回の two-stage route v1/v2 は active recall がほぼゼロで、benchmark floor に逃げただけだった。だから、head構造だけ変えてもダメ。まず教師ラベルの separability を通す。

---

# Phase 3: route model 再設計

label separability が通ったら、route model を作る。

ただし4class一発ではなく、2段階。

```text
Stage 1:
  active / no-active gate

Stage 2:
  activeなら type 判定
    risk_off
    recovery
    overweight
```

## 3-A. active gate

検証項目：

```text
active AUC
active AP
recall under false-active cap
pred-active rate
val-selected thresholdのtest安定性
fold worst false-active
```

合格条件：

```text
worst AUC >= 0.65
worst false-active <= 0.15
pred-active が 0% に潰れない
active recall が全foldで非ゼロ
```

## 3-B. route type head

検証項目：

```text
one-vs-rest AUC
macro-F1
type別 recall
type別 false positive
type別 realized advantage
```

合格条件：

```text
de-risk AUC安定
overweight AUC安定
recoveryは弱くてもよいが、誤爆しない
macro-F1だけで採用しない
```

## 3-C. anti-shortcut test

positionを入れた場合・外した場合の差を見る。

```text
positionありでAUCだけ高い:
  teacher shortcut疑い

positionなしでもAUCが残る:
  market-state labelとして有望
```

---

# Phase 4: BC-only 安定化

route model が probe を通ったら、BC-only。

ここでもACはまだ禁止。

BC設計：

```text
benchmark floor = 1.0
state machine維持
active gate threshold固定
route type head
small overweight adapterはhard cap
turnover cap
```

検証項目：

```text
AlphaEx
SharpeΔ
MaxDDΔ
turnover
long / short / flat
active recall
false-active
route type distribution
danger_fire_rate
pre_dd_danger_rate
fire_pnl
```

合格条件：

```text
3fold worst AlphaEx >= 0.0
3fold worst MaxDDΔ <= +0.25pt
mean MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
active recall 非ゼロ
false-active worst <= 0.15〜0.20
```

ここで neutral collapse したら、active gate が厳しすぎる。
false-active が増えたら、label/threshold が甘い。

---

# Phase 5: fire-aware checkpoint selector v2

BCが通ったら、restricted ACの前に selector を作る。

今後は mean Alpha だけで checkpoint を選ばない。

selector の考え方：

```text
score =
  AlphaEx
  + SharpeΔ
  - MaxDD penalty
  - turnover penalty
  - pre_dd_danger_fire penalty
  - danger_fire_rate penalty
  + safe_fire_pnl bonus
```

検証項目：

```text
checkpoint別:
  AlphaEx
  SharpeΔ
  MaxDDΔ
  turnover
  fire_pnl
  danger_fire_rate
  pre_dd_danger_rate
  MDD overlap
  fold worst
```

合格条件：

```text
MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
fire_pnl >= 0
pre_dd_danger_rate がHbestより明確に低い
fold worstが壊れない
```

ここでは Hbest を救わない。
Hbestは「fireが多すぎる失敗例」として、danger fire の診断基準にする。

---

# Phase 6: train-time fire budget / conservative update

checkpoint selectorだけでなく、学習時に危険fireを出しにくくする。

入れる候補：

```text
fire_rate_penalty
pre_dd_danger_penalty
fire_run_length_penalty
turnover_fire_penalty
Plan7 position deviation penalty
```

方針：

```text
Plan7から大きく外れない
fire総量を制限
危険fire率を制限
許可されたfireのsizeだけ微調整
```

検証項目：

```text
fire rate
fire run length
pre_dd_danger_rate
Plan7 position deviation
delta distribution
turnover
MaxDDΔ
```

合格条件：

```text
Plan7比で AlphaEx +0.2〜+1.0 pt/yr
SharpeΔ >= Plan7
MaxDDΔ <= 0
turnover <= 3.5
danger_fire_rate not worse
```

ここは大勝ち狙いじゃない。
**再現性ある微改善**が目的。

---

# Phase 7: restricted AC 再開

ここでようやくAC。

触っていいもの：

```text
benchmark_overweight_sizing_adapter only
```

固定：

```text
route head
full actor
benchmark floor
active gate
danger guard
advantage gate
```

AC検証項目：

```text
AC step別 checkpoint
actor_lr sweep
critic pretrain steps
td3bc_alpha
delta_range
fire_pnl
pre_dd_danger_rate
MaxDDΔ
turnover
```

合格条件：

```text
AC前BC baseline +0.2pt以上
SharpeΔ >= baseline
MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
pre_dd_danger_rate not worse
```

失敗時の分岐：

```text
Alpha改善なし:
  ACを止めてBC/selectorへ戻る

MaxDD悪化:
  fire budget / danger penalty強化

turnover悪化:
  run-level hysteresis / cooldown再設計

fold5だけ改善:
  採用しない
```

---

# Phase 8: multi-fold / multi-seed 評価基盤

次は評価の厳格化。

評価セット：

```text
dev:
  fold4/5/6

legacy comparison:
  fold0/4/5

adoption:
  全fold

seeds:
  7 / 11 / 21
```

検証項目：

```text
mean AlphaEx
median AlphaEx
fold win rate
worst fold AlphaEx
mean MaxDDΔ
worst MaxDDΔ
turnover
long/short
fire_pnl
danger_fire_rate
pre_dd_danger_rate
monthly relative win rate
upside capture
downside capture
```

採用条件：

```text
mean AlphaEx > 0
median AlphaEx >= 0
fold win rate > 50%
mean MaxDDΔ <= 0
worst foldが壊れていない
turnover <= 3.5
long <= 3%
short = 0%
seed間で方向一致
```

平均だけで採用しない。
fold5みたいな大勝ちで盛れるから。

---

# Phase 9: WM control head v2

これは後段。
route teacher と BC が安定してから。

目的は「fireを増やす」ではなく、**危険fireを出さない状態認識**。

追加head候補：

```text
risk_off_event_head
pre_dd_danger_head
future_mdd_overlap_head
safe_fire_advantage_head
fire_run_risk_head
```

やり方：

```text
freeze:
  WM backbone
  standard predictive heads

train:
  control heads only
```

full WM retrain は禁止。
以前、full WM retrain は既存の安定表現を壊したから。

検証項目：

```text
label AUC
top-decile precision
selected fire advantage
selected fire MDD overlap
BC-only性能
AC前後性能
fold worst
```

合格条件：

```text
control headなしよりfalse-active低下
MaxDDΔ改善
Alphaを殺しすぎない
fold間で同方向
```

---

# 最短の実行順

```text
1. 本流Plan7/Plan5を固定
2. Hbest post-hoc guardを正式停止
3. market-state route teacher を作る
4. route separability probe
5. two-stage active gate / route type probe
6. BC-only backtest
7. fire-aware checkpoint selector v2
8. train-time fire budget
9. restricted sizing-adapter AC
10. multi-fold / multi-seed
11. WM control head v2
```

