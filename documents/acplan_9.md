うん、ロードマップはこう。
今は **性能拡張フェーズじゃなくて、Plan7を“再現可能な手法”にするフェーズ**。採用checkpointは良いけど、seed 11でAC再学習するとfold5が崩れているから、次は原因分解が最優先。

## Roadmap 0: 現状固定

まず、比較基準を固定。

```text
Current baseline:
  AC Plan 7 checkpoint
  benchmark exposure floor = 1.0
  benchmark-gated small overweight adapter
  predictive advantage gate
  scale = 0.5 fixed
```

これは維持。
やらないものも固定。

```text
禁止:
  full actor AC
  route head unlock
  advantage gate緩和
  floor > 1.0 一律適用
  Q argmax型AC
  rank loss critic actor update
  scale grid selector
```

理由は単純。Plan8で、scale gridはPlan7固定より悪化、seed11 AC retrainはfold5崩壊、guarded sizing variantも失敗してる。

---

## Roadmap 1: seed variance切り分け

最初にやるのはこれ。
**ACが悪いのか、BC/WM seedも悪いのかを分ける。**

対象はまず fold5。ここが seed 11 で崩れたから。

```text
Fold5 seed matrix:

A. seed7 WM/BC + seed7 AC
   → 現Plan7基準

B. seed7 WM/BC + seed11 AC
   → 既に失敗確認済み

C. seed11 WM/BC + ACなし
   → BC/WMだけで安全か確認

D. seed11 WM/BC + seed7 AC
   → ACが別BCに乗るか確認

E. seed11 WM/BC + seed11 AC
   → 完全別seedで再現するか確認
```

出す指標。

```text
AlphaEx
SharpeΔ
MaxDDΔ
turnover
long / short / flat
adapter fire rate
fire_pnl
nonfire_pnl
mean_delta
positive_delta_rate
long_state rate
floor発動率
predictive advantage分布
```

判定。

```text
Cが悪い:
  BC/WM seedが主犯。AC以前に戻る。

Cは良いがD/Eが悪い:
  ACがBC checkpoint差分に敏感。

C/Dは良いがBだけ悪い:
  AC seed/selector/checkpoint selectionが主犯。

全部良い:
  seed11問題は偶然寄り。seed21へ拡張。
```

---

## Roadmap 2: fold5 overweight timing診断

seed 11で崩れた原因は、short collapseじゃなくて **overweight timing悪化**。
だから、fire単位で見る。

比較対象。

```text
seed7 AC good checkpoint
seed11 AC bad checkpoint
guarded variant bad checkpoint
```

見るもの。

```text
1. adapter fireしたbarの一致率
2. seed7ではfireしてseed11ではfireしないbar
3. seed11だけfireしたbar
4. fire時 forward return
5. fire時 realized advantage
6. fire時 predictive advantage
7. fire後 4/8/16/32 bar PnL
8. fireの連続回数
9. fireからlong_stateへ残る割合
```

ここで原因を分類。

```text
seed11だけ悪いbarでfire:
  gate/critic/adapter timing問題

fire timingは同じだがdeltaが悪い:
  sizing adapter出力問題

fireもdeltaも同じだがPnL悪い:
  評価経路/seed以外の差を疑う

fireが少なすぎる:
  adapterが萎縮してB&H upsideを逃す
```

---

## Roadmap 3: AC recipe安定化

原因が見えたら、AC recipeを直す。
ただし、いきなり強い正則化はダメ。guarded variantはfold5 `AlphaEx -23.36` で失敗してるから、単純に弱めるだけでは直らない。

候補はこれ。

```text
A. checkpoint selection改善
  fire_pnl > 0
  fold5 period win低下なし
  long <= 3%
  turnover <= 3.5
  MaxDDΔ <= 0
  を満たすcheckpointだけ採用

B. AC checkpoint ensembleではなく seed selection
  seed7/11/21でACだけ回し、
  guardを通るcheckpointだけ採用

C. adapter fire penalty
  fire頻度ではなく「悪いfire」を抑える
  bad_fire_loss = max(0, -fire_forward_advantage)

D. critic-only長め
  critic_pretrain 250 → 500
  ただしactor更新はsizing adapter onlyのまま

E. actor update early stop
  val Alphaだけでなく fire_pnl / Sharpe / MaxDD / long capで止める
```

採用条件。

```text
fold5で:
  AlphaEx >= seed7 baseline - 許容差
  fire_pnl > 0
  MaxDDΔ <= 0
  long <= 3%
  turnover <= 3.5

3foldで:
  AlphaEx >= +13.91付近
  SharpeΔ >= +0.035付近
  MaxDDΔ <= -0.30付近
```

---

## Roadmap 4: BC/WM seed robustness

ACだけじゃなく、BC/WM seedも見る。
ただし、全fold多seedは重いから段階的に。

```text
Stage 1:
  fold5 only, seed11 WM/BC, ACなし

Stage 2:
  fold5 only, seed11 WM/BC + Plan7 AC

Stage 3:
  folds 0/4/5, seed11 WM/BC + ACなし

Stage 4:
  folds 0/4/5, seed11 WM/BC + Plan7 AC

Stage 5:
  seed21を追加
```

ここで知りたいのは、

```text
Phase8/Plan5相当がseed違いで出るか
benchmark floorが毎回効くか
adapterの発火が毎回正寄与か
BC自体がflat/underweight/long collapseしないか
```

BCがseedで崩れるなら、ACどころじゃない。
BCが安定していてACだけ崩れるなら、Plan7 recipeの問題。

---

## Roadmap 5: 全fold検証

seed問題を最低限見たら、別ジョブで全14fold。

```text
全14fold:
  WM
  BC
  Plan5
  Plan7
  test
```

必須指標。

```text
mean AlphaEx
median AlphaEx
fold win rate
worst fold AlphaEx
SharpeΔ mean/median
MaxDDΔ mean/median
turnover
long/short/flat
PeriodWin
adapter fire_pnl
floor effect
```

平均だけはダメ。fold5みたいな大勝ちで盛れる。

合格条件。

```text
mean AlphaEx > 0
median AlphaEx >= 0
fold win rate > 50%
MaxDDΔ mean <= 0付近
turnover <= 3.5
short collapseなし
long <= 3%
```

---

## Roadmap 6: その後の性能改善

ここまで通ってから、ようやく性能改善。

触っていい順。

```text
1. sizing adapter checkpoint selection
2. sizing adapter critic_pretrain / AC steps
3. sizing adapter delta_range
4. overweight sizing adapterだけのseed selection
5. regime-gated adapter
```

まだ触らない。

```text
route head
full actor
advantage gate threshold緩和
floor > 1.0一律
de-risk route
Q argmax
```

次の実験候補。

```text
Plan9-A:
  sizing adapter seed selection
  seed7/11/21/42 のACだけ回して guard通過bestを選ぶ

Plan9-B:
  fire_pnl-aware checkpoint selector

Plan9-C:
  critic_pretrain 250/500/1000
  AC steps 100/250/500
  actor_lr 5e-6/1e-5

Plan9-D:
  fold5 timing fix
  bad fire抑制loss

Plan9-E:
  all-fold generation
```

---

## 最短でやるなら

次の1本目はこれ。

```text
AC Plan 9:
  fold5 seed variance diagnosis

比較:
  seed7 WM/BC + seed7 AC
  seed7 WM/BC + seed11 AC
  seed11 WM/BC + ACなし
  seed11 WM/BC + seed7 AC
  seed11 WM/BC + seed11 AC

出力:
  performance table
  adapter fire attribution
  fire timing overlap
  fire forward PnL
  floor/adaptor/gate attribution
```

これで、**ACだけが不安定なのか、BC/WMもseedでズレるのか**が分かる。

まったく、ここでまた「性能上げたい！」って暴れると全部崩れるわよ。
今の最優先は **Plan7を再学習しても再現できる手法にすること**。そこが通ったら、初めて全fold展開と性能改善に進む。
