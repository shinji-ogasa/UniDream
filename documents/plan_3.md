うん、次は **ACに戻る前に「安全overlay候補を本物にする」フェーズ**。
今の一番マシな候補は `D_risk_sensitive + h32 triple-barrier downside guard + pullback false-de-risk blocker`。全14foldで `AlphaEx worst 0.000`, `MaxDDΔ worst 0.000`, `turnover max 3.0` まで守れてるのはかなり良い。でも `pass rate 0.286`、neutral foldが多い、平均Alphaはfold3の `+41.381` に強く引っ張られている。つまり **安全だけど薄い**。

だから次は一本化しない。複数レーンで検証して、勝ち筋だけ残す。

---

## 現状の課題

```text
良い:
  all14でworst AlphaEx >= 0
  all14でworst MaxDDΔ <= 0
  turnover max 3.0
  long 0でoverweight事故なし

悪い:
  neutral foldが多い
  pass rate 0.286
  fold3依存が強い
  false-active / false de-risk の制御がまだ雑
  overlayがeval-only guardなので本流実装前に独立再現が必要
```

今の候補は「勝つモデル」じゃなくて、**負けにくいde-risk overlay**。
だから次の目標はこれ。

```text
安全性を維持したまま、
active foldを増やす。
ただしfold10/12/13を壊さない。
```

---

# 実験候補A: overlay独立実装・再現性確認

## 仮説

今の `eval-only pullback guard` は効いているけど、probe内の評価ロジックに依存している可能性がある。
まず独立overlayとして切り出して、同じ結果が再現するか確認する。

## やること

```text
A1. D+A+pullback を standalone overlay module 化
A2. configs/trading.yaml は触らず、post-process overlay として実行
A3. all14で probe結果と一致するか確認
A4. start-from test / seed固定 / cost条件で再現性を見る
```

## 合格ライン

```text
AlphaEx worst >= 0
MaxDDΔ worst <= 0
turnover <= 3.5
fold10/12が再崩壊しない
probe結果との差分が小さい
```

これは最優先。
ここで再現しないなら、後続の実験全部が砂上の楼閣よ。

---

# 実験候補B: attribution / ablation

## 仮説

今の改善が何で出ているか、まだ分解が足りない。
`D utility`, `triple-barrier guard`, `cooldown`, `pullback guard`, `threshold floor` のどれが本当に効いたか切る。

## やること

```text
B1. D risk-sensitive utility only
B2. D + triple-barrier downside
B3. D + cooldown
B4. D + pullback
B5. D + triple-barrier + pullback
B6. full current candidate
```

各foldで見る。

```text
selected event count
false de-risk count
pullbackで止めた回数
止めたeventのcounterfactual PnL
turnover寄与
MaxDD寄与
```

## 合格ライン

```text
pullback guard が fold10/12 で明確に損失回避している
triple-barrier guard が false active を減らしている
cooldown が turnover を下げている
```

これで「どれを伸ばすべきか」が見える。

---

# 実験候補C: fold3依存を剥がす

## 仮説

平均Alpha `+3.747` はfold3の `+41.381` に引っ張られてる。
fold3抜きでも価値があるか確認しないと、また局所解。

## やること

```text
C1. fold3抜き mean / median / win rate
C2. positive fold数
C3. active fold数
C4. neutral foldを除いたconditional Alpha
C5. worst 3 folds の挙動
```

## 見るべき指標

```text
mean AlphaEx without fold3
median AlphaEx
fold win rate
active fold rate
non-neutral fold average
```

## 判断

```text
fold3抜きで mean > 0:
  研究候補としてかなり良い

fold3抜きで mean ≒ 0:
  安全filterではあるが、収益源は限定的

fold3抜きで negative:
  fold3局所寄り。採用不可
```

これは必須。
平均だけで喜ぶのは禁止。

---

# 実験候補D: false-de-risk / pullback recovery label

## 仮説

今回効いたのは、たぶん「de-riskすべき局面を当てた」より、**false de-riskを止めたこと**。
なら、pullback / recovery / false-de-risk を直接ラベル化した方がいい。

## 作るラベル候補

```text
false_derisk_label:
  de-riskすると、その後のreboundを取り逃がしてB&H劣後する局面

pullback_recovery_label:
  下落後に短期回復が起き、de-riskを入れると損する局面

no_fire_reentry_label:
  current DD中でも、benchmark維持の方が良い局面
```

## 検証

```text
AUC / PR-AUC
top-decile avoided false-de-risk PnL
false-active reduction
fold10/12/13で効くか
fold3のalphaを削りすぎないか
```

## 合格ライン

```text
false-de-risk を減らす
MaxDD worst <= 0維持
Alpha meanを削りすぎない
active foldを増やしてもfold10/12/13が壊れない
```

これは次の本命。
いま効いてるpullback guardを、ちゃんと学習可能な診断に落とす。

---

# 実験候補E: active foldを増やす安全探索

## 仮説

今は安全すぎてneutralが多い。
安全制約を維持しながら、少しだけactiveを増やせるかを見る。

## やること

```text
E1. threshold floor 0.001周辺 sweep
  0.0005 / 0.001 / 0.0015 / 0.002

E2. cooldown auto候補拡張
  0 / 16 / 32 / 64

E3. active budget制御
  foldごとに最大turnover <= 3.5
  selected event max rate cap

E4. no-trade margin緩和
  ただし fold10/12/13 hard guard
```

## 合格ライン

```text
active fold数が増える
median AlphaEx > 0
worst AlphaEx >= 0
worst MaxDDΔ <= 0
turnover <= 3.5
```

ただし、これはA〜Dが済んでから。
いきなり閾値ガチャに戻るとまた迷子。

---

# 実験候補F: triple-barrier label改善

## 仮説

downside barrierのAUCは `0.57〜0.63` 程度で強くはないけど、guardとしては効いた。
なら、barrier設計を少し洗練するとfalse-activeが下がるかも。

## 候補

```text
F1. volatility-normalized barrier
  rolling volで上下バリアを動的調整

F2. asymmetric barrier
  downsideは厳しく、upside/recoveryは緩く

F3. event sampling
  毎barではなく、CUSUM / vol spike / drawdown entryだけ評価

F4. h32/h64 ensemble
  h32で短期DD、h64で長めの危険を確認
```

## 見る指標

```text
AUC worst
false-active worst
selected event PnL
fold10/12/13の改善
neutral foldのactive化
```

## 期待

中程度。
単体AUCは強くないけど、risk-sensitive utilityと組ませると効く可能性がある。

---

# 実験候補G: Plan7 overlayとの比較

## 仮説

今のD+A+pullbackは「B&H/benchmark基準のde-risk overlay」っぽい。
Plan7/Plan5本流に後段で重ねた時に価値があるかを確認する必要がある。

## 比較

```text
G1. B&H + overlay
G2. Plan7 + overlay
G3. Plan5 + overlay
G4. overlay only
```

## 見る指標

```text
AlphaEx
MaxDDΔ
turnover
overlay発火時のPlan7 position
Plan7とoverlayの衝突率
```

## 合格ライン

```text
Plan7を壊さない
MaxDD改善が増える
turnoverが3.5以内
fold worstが非負
```

これはかなり重要。
今の候補が単独では安全でも、Plan7と重ねたら邪魔になる可能性がある。

---

# 実験候補H: seed / cost / slippage stress

## 仮説

今の結果はseed7・コスト条件固定。
安全overlay候補ならstressに耐えてほしい。

## やること

```text
H1. seed 11 / 21
H2. cost x1.5 / x2.0
H3. slippage x2.0
H4. spread悪化
H5. fold3抜き stress
```

## 合格ライン

```text
worst AlphaEx >= 0付近
worst MaxDDΔ <= 0付近
turnover <= 3.5
大幅なfold崩壊なし
```

これは本流採用前の必須チェック。

---

# 優先順位

あたしならこう進める。

```text
Round 1:
  A. overlay独立実装
  B. ablation
  C. fold3依存チェック

Round 2:
  D. false-de-risk / pullback label
  F. triple-barrier改善
  G. Plan7 overlay比較

Round 3:
  E. active fold増加探索
  H. seed/cost stress

Round 4:
  条件を満たしたらBC/ACに戻る
```

---

## 直近で投げる指示文

```text
次はPlan2候補を本流に混ぜず、inference-only overlay候補として厳密に検証する。

1. D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly を standalone overlay として実装する。
2. all14で exploration_board_probe と同じ結果が再現するか確認する。
3. D utility / triple-barrier guard / cooldown / pullback guard / threshold floor のablationを行う。
4. fold3抜き mean, median, active fold rate, fold win rate を出す。
5. false_derisk / pullback_recovery label を作り、fold10/12/13の崩れを止める信号として使えるかprobeする。
6. Plan7 + overlay / B&H + overlay / overlay only を比較する。
7. まだAC、route unlock、configs/trading.yaml変更は行わない。
```

---

## 今の結論

今やるべきは、

```text
AC再開
```

じゃなくて、

```text
安全overlay候補の独立化・分解・fold3依存排除・false de-risk削減
```

よ。

これはかなり良い局面。
今までの「fold単体で強いけど壊れる」から、ようやく **全14foldで最悪負けない安全候補** が出た。
次はこれを **薄い安全filter** から **安定した収益overlay** に育てられるか。ここが勝負ね。
