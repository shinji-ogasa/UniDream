今後のロードマップは、**「B&H + small gated overweight」を本流にして、まず多foldで壊れないか確認。その後、overweight sizing周辺だけ制限付きACで伸ばす**、が正解。

いまの採用構成はこれ。

```text
Phase 8 BC
+ predictive state input
+ benchmark-gated small overweight adapter
+ predictive advantage gate
+ benchmark exposure floor = 1.0
+ per-fold deterministic seed reset
```

Plan 5で、3fold平均は `AlphaEx -18.45 → +13.48 pt/yr`、SharpeΔも `+0.020 → +0.034`、turnoverも `1.72 → 1.45` に改善した。一方でMaxDD改善幅は `-1.03 → -0.30` に弱まっていて、policyは「B&H-relative de-risk」より **B&Hを最低維持して、少しだけoverweightする** 性格に変わってる。

## Phase A: 多fold検証を最優先

まず、今の3foldだけで喜ばない。
次は全WFO fold、または最低でも利用可能なfold全部で `start-from test` を回す。

見るものはこれ。

```text
fold別:
  AlphaEx
  SharpeΔ
  MaxDDΔ
  turnover
  long / short / flat
  adapter fire rate
  floor発動率
  B&H return regime
```

合格ラインは、

```text
全fold平均 AlphaEx > 0
SharpeΔ >= 0
turnover <= 3〜4
short/underweight collapseなし
fold過半数でPhase 8またはB&H比改善
```

特に、fold5の大敗がfloorで直ったのは良いけど、`floor=1.02/1.05` はDDやfold4悪化で不採用だった。だから今は `floor=1.0` を固定して、fold数を増やすのが先。

## Phase B: 何で勝っているか分解

次に attribution。
今の改善は「賢い売買」というより、**B&H未満に落ちないことでupside missを消した**影響が大きい。

分解すべきはこれ。

```text
1. floorだけ
2. overweight adapterだけ
3. floor + adapter
4. predictive advantage gate on/off
5. adapter fire時だけのPnL
6. floor発動時だけのPnL
```

これで、

```text
floorが主因なのか
adapterが本当に上乗せしているのか
predictive advantage gateが効いているのか
```

を切り分ける。

## Phase C: selectorをM2向けに直す

今のM2がMISSなのは、低いbar-level win rateに引っ張られている可能性がある。
でも今のpolicyは、barごとの勝率で勝つというより、**大きなupside missを消して年率Alphaを改善する**タイプ。

だからselectorはAlphaだけでもWinRateだけでもダメ。

候補：

```text
score =
  AlphaEx
  + 0.5 * SharpeΔ
  - 0.5 * max(0, MaxDDΔ)
  - 0.2 * max(0, turnover - 3.5)
  - collapse_penalty
  + benchmark_winrate_bonus
```

追加で見るべき指標：

```text
monthly B&H-relative win rate
fold-level win rate
upside capture ratio
downside capture ratio
max underperformance streak
```

bar-level勝率より、月次やfold単位のB&H相対勝率のほうが今のpolicyには合う。

## Phase D: restricted ACは overweight sizing 周辺だけ

多foldでAlphaExが正のままなら、ACを再開していい。
ただし触るのは **overweight sizingだけ**。

まだ触らない：

```text
route head full unlock
de-risk route
state machine
benchmark floor
full actor
advantage threshold緩和
```

触っていい順番：

```text
1. benchmark-state sizing adapter
2. overweight epsilon adapter
3. overweight max_position
4. overweight logit bias only
5. overweight adapter trainable化
```

目的は、

```text
long 1% → 1〜3%
turnover <= 3.5
MaxDDΔ <= 0
AlphaEx改善
```

くらい。
`advantage gate off` はturnover `104.60` で即死しているから、gate緩和は当面禁止。

## Phase E: regime別 floor / upside capture

次の改善余地は、たぶんここ。

今はfloorが一律 `1.0`。
でも相場によっては、

```text
bull/uptrend:
  floor 1.0 を維持、または small overweightを許可

sideways/high-vol:
  floor 1.0固定で十分

crash/high-DD-risk:
  de-riskを少し復活させたい
```

という分岐があり得る。

ただし、いきなり `floor=1.02` 以上は危険。Plan 5でも高いfloorはfold5では強いけど、fold4のDD/turnoverを悪化させて不採用だった。

なのでやるなら、

```text
regime-gated floor:
  default floor = 1.0
  only strong-uptrend and low-risk → floor 1.02
  high-risk → floor 1.0維持
```

みたいに限定する。

## Phase F: 実運用/PoC向けの形に整える

ここまでで多foldが正なら、研究ログからPoC資料にできる。

必要なのは、

```text
B&H比較の3fold/全fold表
cost込みであること
DD改善/turnover低下
fold5 upside missがfloorで解消された説明
採用/不採用実験の履歴
collapse guard
再現コマンド
```

今のストーリーは悪くない。

```text
最初はRL/BCがcollapseした
→ transition advantage / routingで安定化
→ state machineでchurn抑制
→ B&H floorでmulti-foldのupside missを解消
→ small overweight adapterで少し上乗せ
```

これはかなり説明しやすい。

## 優先順位

最短ロードマップはこれ。

```text
1. 全WFO foldで採用configを評価
2. floor / adapter / gate のattribution
3. selectorを月次・fold相対勝率込みに修正
4. 多foldが正なら overweight sizing限定AC
5. regime-gated floorを検証
6. PoC資料化
```

今は **ACを広げる段階ではなく、採用configが本当に多foldで強いか確認する段階**。
まったく…fold4だけの局所解から一歩抜けたんだから、ここでまたfull actorに突っ込んで壊すのはナンセンスよ。まず多foldで地盤固め。
