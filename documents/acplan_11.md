うん、ロードマップはかなり明確になった。
今は **「E系の高Alphaを、MaxDD条件内に押し込む」フェーズ**。

Plan10では `AlphaEx +60.78`, `SharpeΔ +0.049`, `turnover 1.17`, `long 2%`, `short 0%`, `fire_pnl +0.1271` まで出た。でも `MaxDDΔ +0.04pt` で必須条件 `MaxDDΔ <= 0` を外したから不採用。つまり、fire自体は良いが、DD悪化を消せてない。

## 現状固定

```text
本流維持:
  Plan7 checkpoint
  benchmark floor = 1.0
  predictive advantage gate
  benchmark-gated small overweight adapter
  sizing adapter AC curriculum

Plan10:
  fire selector基盤は採用
  seed11 fire selector checkpointは不採用
  configs/trading.yamlでは fire_checkpoint_selector off
```

禁止継続：

```text
full actor AC
route head unlock
advantage gate緩和
floor > 1.0 一律適用
scale grid selector
Q argmax actor update
```

## Phase 11: fire-time drawdown attribution

まず、**どのfireがDD悪化に寄与してるか**を出す。

見るもの：

```text
fire bar
fire後 4/8/16/32 bar return
fire後 4/8/16/32 bar min equity
fireが最大DD区間に含まれるか
fireあり/なしのcounterfactual equity差
fire時のpred_adv
fire時のposition increment
```

出す表：

```text
good_fire:
  fwd return > 0
  drawdown contribution <= 0

bad_fire:
  fwd return > 0 でも一時DDを悪化
  または最大DD区間に重なるfire

neutral_fire:
  PnLもDD寄与も小さい
```

ここで、E系の `MaxDDΔ +0.04` が **少数のbad fire** 由来なのか、**全体的なposition floor/long state** 由来なのかを切る。

## Phase 12: inference-only DD guard

次は学習せず、fireを少しだけ削る。

候補：

```text
Guard A:
  fire後の想定min pathが悪い局面はfire禁止

Guard B:
  current equityが直近peakから一定以上沈んでいる時はfire禁止

Guard C:
  predicted advantageは高いがDD-risk予測も高い時はfire禁止

Guard D:
  fire後cooldownを入れて連続fireを抑える

Guard E:
  max drawdown区間に入りやすいregimeではepsilonを半減
```

狙い：

```text
AlphaEx +60級を全部維持する必要はない
AlphaEx +30〜50でもいいから
MaxDDΔ <= 0 に戻す
```

採用条件：

```text
fold5:
  AlphaEx > Plan7
  SharpeΔ > Plan7
  MaxDDΔ <= 0
  turnover <= 3.5
  long <= 3%
  short = 0%
```

ここで通ったらかなり強い。

## Phase 13: DD-aware fire checkpoint selector

今のfire selectorは、

```text
fire_pnl > 0
fwd16 > 0
incr16 > 0
```

を見るだけでは足りなかった。
次は selector のscoreにDD寄与を入れる。

例：

```text
score =
  AlphaEx
  + 0.5 * SharpeΔ
  + fire_pnl_bonus
  + fwd16_bonus
  - maxdd_violation_penalty
  - fire_drawdown_contribution_penalty
  - turnover_penalty
```

hard guard：

```text
MaxDDΔ <= 0
fire_drawdown_contribution <= threshold
long <= 3%
short = 0%
turnover <= 3.5
```

AC中に保存するcheckpoint：

```text
step 250 / 300 / 350 / 400 / 450
```

狙いは、E系の途中にあるかもしれない

```text
AlphaはEより少し低い
でもMaxDDΔ <= 0を満たす
```

checkpointを拾うこと。

## Phase 14: fold5で通ったら 0/4/5 に戻す

fold5だけで採用しない。
次は fold0/4/5 で比較。

比較対象：

```text
Plan7 current
E系 final
E + DD guard
E + DD-aware selector
```

合格条件：

```text
3fold avg:
  AlphaEx >= Plan7
  SharpeΔ >= Plan7
  MaxDDΔ <= Plan7付近、最低でも <= 0
  turnover <= 3.5
  long <= 3%
  short = 0%

fold単位:
  fold5だけ勝ちではなく、fold4を壊さない
```

ここで fold4 のDD改善が壊れるなら、E系はまだfold5過適合。

## Phase 15: seed robustness

次は seed を見る。

最低セット：

```text
WM/BC seed:
  7, 11

AC seed:
  7, 11, 21
```

見るもの：

```text
fire_pnl
fire_drawdown_contribution
fire overlap
AlphaEx
SharpeΔ
MaxDDΔ
long/short/turnover
```

採用条件：

```text
複数seedで:
  MaxDDΔ <= 0 を満たすcheckpointが出る
  fire_pnl > 0
  long <= 3%
  short = 0%
```

seed7だけ、seed11だけ、みたいな単発はまだ採用しない。

## Phase 16: 全fold展開

ここまで通ってから全fold。

```text
全14fold:
  Phase8/Plan7 checkpoint生成
  Plan7評価
  DD-aware fire selector評価
```

見る指標：

```text
mean AlphaEx
median AlphaEx
fold win rate
worst fold AlphaEx
mean/median MaxDDΔ
PeriodWin
turnover
fire_pnl
fire DD contribution
```

平均だけはダメ。fold5みたいな大勝ちで盛れるから、中央値とfold勝率を見る。

## Phase 17: その後の性能改善

DD guard付きで安定したら、初めて少し拡張。

触っていい順：

```text
1. sizing adapter delta range
2. fire DD guard threshold
3. epsilon schedule
4. critic pretrain steps
5. AC steps
```

まだ触らない：

```text
route head
full actor
advantage gate threshold
floor > 1.0
de-risk route
Q argmax
```

## 最短の次アクション

次はこれでいい。

```text
AC Plan 11:
  fire-time drawdown contribution guard

実装:
  1. fireごとのforward returnとpost-fire max drawdown contributionを計算
  2. 最大DD区間に含まれるfireを抽出
  3. fire_pnlは正だがDDを悪化させるbad fireを分類
  4. inference-only guardでbad fireだけ抑制
  5. fold5で MaxDDΔ <= 0 かつ AlphaEx/Sharpeを維持できるか確認
  6. 通ったらfold0/4/5へ展開
```

## 今の判断

Eはめちゃくちゃ惜しい。
でも今採用すべきはEそのものじゃなくて、**Eの良いfireを残して、DDを悪化させるfireだけ削る仕組み**。

```text
今の勝ち筋:
  Plan7の安全性
  + E系の高Alpha fire
  - fire-time DD悪化
```

ここを通せたら、かなり本物に近づく。DDを通してからAlphaを伸ばす。これが次の最短ルートよ。
