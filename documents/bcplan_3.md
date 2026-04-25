結論：**次の一手は true routing BC + no-trade/turnover制約**。
いきなりACチューニングに行くのはまだ早い。今のbestは `AlphaEx +3.57pt / SharpeΔ +0.082 / MaxDDΔ -0.28pt` まで来てるけど、`turnover=8.42` と `flat=94%` がまだ弱点。つまり「当たる局面はあるが、行動が細かく揺れすぎる」状態。

## Webサーチ後の主仮説

### 仮説1：position直接BCをやめて、route分類にする

IQLは、未観測actionを直接評価せず advantage-weighted behavioral cloning でpolicy抽出する方向。これは今のUniDreamの「事後価値のあるtransitionだけ強く真似る」設計とかなり整合する。([Hugging Face][1])
AWACも advantage によって actor update を重み付けする設計なので、`transition/action単位のadvantageでBCする` 方針を支持してる。([ScienceStack][2])

だから次はこれ。

```text
今:
  best transition → position labelへ変換 → BC

次:
  best transition → route labelへ変換 → routing BC
```

routeは最低4つ。

```text
neutral
de_risk
recovery
overweight
```

ここで重要なのは、**positionを直接当てにいかないこと**。
positionは route の後に小さな delta として出す。

```text
route_head:
  neutral / de_risk / recovery / overweight

delta_head:
  routeごとの position adjustment
  max_step: 0.05〜0.10
```

検証名を付けるなら **Experiment K: true_routing_bc**。

---

### 仮説2：turnoverはpenaltyだけでなく no-trade band で抑える

ポートフォリオ最適化では、取引コストがあると「何もしない方が最適な領域」、つまり no-trade region が出るという古典的な結果がある。比例コストなら、領域外に出た時だけ境界へ戻すのが基本になる。([Taylor & Francis Online][3])

今のUniDreamの `turnover=8.42` は、これと逆。
ちょっとでもadvantageがあると細かく動いてそう。

次は、

```text
if best_advantage - neutral_advantage < margin:
    route = neutral
```

を明示するべき。

margin候補：

```yaml
transition_advantage_margin: [0.0005, 0.0010, 0.0015]
transition_relabel_max_step: [0.05, 0.10]
min_hold_bars: [8, 16, 32]
turnover_budget: [2.0, 4.0, 6.0]
```

検証名は **Experiment L: routing_bc_notrade_band**。

これで見るべきは、`AlphaEx` を保ったまま `turnover 8.42 → 4以下` に落とせるか。

---

### 仮説3：hard labelではなく soft advantage label にする

H1 direct relabel は `short 100% / turnover 665.46 / AlphaEx -79.19pt` まで壊れてる。これは「best actionを1個に決め打ちする」と金融ノイズで壊れる証拠。

なので route target は one-hot じゃなくて、

```text
p(route) = softmax(adv_route / tau)
```

にする方がいい。

候補：

```yaml
route_label_mode: softmax_adv
adv_tau: [0.0005, 0.0010, 0.0020]
adv_weight_clip: [2.0, 4.0]
label_smoothing: [0.05, 0.10]
```

狙いは、

```text
direct relabelのhigh-turnover collapseを避ける
でもsmoothしすぎのflat 100%も避ける
```

検証名は **Experiment M: soft_advantage_routing_bc**。

---

### 仮説4：predictive stateはpositionではなくgateにだけ渡す

前の実験で、predictive stateをActorに直結すると underweight 側に寄った。adapterでも `scale=0.25` で `short=68%`、`scale=1.0` で `short=99%` だから、予測特徴はrisk-off shortcutになってる。

だから次はこう。

```text
risk preds:
  de_risk gateへ

return preds:
  overweight gateへ

current position / inventory age:
  recovery gateへ

latent_zh:
  shared trunkへ
```

ダメなのはこれ。

```text
[predictive_state] → position head直結
```

検証名は **Experiment N: predictive_gate_routing_bc**。

比較は必ず、

```text
K: routing without predictive state
N: routing with predictive gate
```

でやる。Nだけ良くても、predictive stateが効いたのかroutingが効いたのか分からないから。

---

### 仮説5：ACはBCが安定してから、TD3+BC/CQL-liteで軽く回す

TD3+BCは、offline RLでpolicyがdataset actionから外れすぎないようにBC termを足すシンプルな方法。つまり、今のような「BCを土台にACで少し改善する」用途に合う。([Emergent Mind][4])
CQLは、offline RLで分布外actionのQ過大評価を抑えるために conservative Q を学習する方法なので、overweight/de-riskのQが過大評価されるのを抑える用途に合う。([ScienceStack][5])

ただし、BCがcollapseしてる状態でACに行くのはダメ。
前の報告でも「崩れたBCをACで回すとcollapseを増幅するリスクが高い」と整理されてる。

ACに進むなら、

```yaml
max_steps: 500 -> 1000 -> 2000
actor_lr: 1e-5
td3bc_alpha: [5.0, 10.0]
cql_alpha: [0.05, 0.10, 0.20]
turnover_coef: [0.5, 1.0]
prior_kl_coef: [0.2, 0.5]
```

くらいの保守的チューニングから。

## 検証順序

まずこれでいい。

```text
1. Experiment K
   true routing BC
   predictive stateなし

2. Experiment L
   no-trade band / min-hold / turnover budget追加

3. Experiment M
   soft advantage routing label

4. Experiment N
   predictive stateをgate限定で追加

5. 条件達成後だけ AC
   TD3+BC + CQL-lite
```

まったく、ここでPatchTSTとかモデル大型化に逃げるのは違うわよ。
今の問題は「情報がない」じゃなくて、**情報をどの行動に割り当てるか**だから。

## ACチューニングへ移行する前のBC推奨ライン

「BC精度」は単純な accuracy で見ちゃダメ。金融だと class label 自体がノイズだから、見るべきは **route精度 + 取引品質 + collapse耐性**。

### 推奨ライン

```text
BC stage fold4:

AlphaEx:
  >= +1.0 pt/yr

SharpeΔ:
  >= +0.02

MaxDDΔ:
  <= 0

Turnover:
  <= 4.0

Flat比率:
  80〜92%

Active比率:
  8〜20%

単一路線collapse:
  neutral/flat以外のrouteが 90%超えない
  long/underweight/overweightが一極集中しない

Collapse guard:
  pass

Route diagnostics:
  predicted route上位decileのcost-adjusted advantageが正
  route score分位と実現advantageが単調に近い
  recovery routeが0%ではない
```

このラインを満たしたら、AC 500〜1000 stepsに進んでいい。

### 妥協ライン

```text
BC stage fold4:

AlphaEx:
  >= 0 近辺

SharpeΔ:
  >= -0.01

MaxDDΔ:
  <= +0.2 pt まで

Turnover:
  <= 6.0

Flat比率:
  <= 95%

Active比率:
  >= 3%

Collapse guard:
  pass

追加条件:
  top-decile route advantageが明確に正
  direct relabel型のhigh-turnover collapseがない
  ACなしでもcostで死んでいない
```

この妥協ラインなら、**AC 500 stepsだけ試す価値はある**。
ただし fold0/fold5へ広げる前に、fold4で `turnover <= 4` へ落とす努力を優先。

### AC移行禁止ライン

```text
AlphaEx < -1.0 pt/yr
SharpeΔ < -0.03
MaxDDΔ > +0.5 pt
Turnover > 8
flat 100%
short/long/overweight 95%以上
route上位decileの実現advantageが正でない
collapse_guard fail
```

この状態でACを回すのは時間の無駄。
ACは「まともなBC priorを微調整する」ためのもの。壊れたBCを救済する道具じゃない。

## 最短指示文

次にCodex/GLMへ投げるならこれ。

```text
次はACチューニングではなく、true routing BCを実装する。

目的:
transition advantageをposition labelへ直接変換せず、
neutral / de_risk / recovery / overweight のroute labelとして学習する。

実装:
1. route_headを追加する。
2. delta_headはrouteごとの小さいposition adjustmentだけ出す。
3. route targetはcost_adjusted_advantageから作る。
4. best routeとneutral routeのadvantage差がmargin未満ならneutralにする。
5. hard labelではなくsoftmax(adv/tau)のsoft route targetも試す。
6. predictive stateはposition head直結禁止。使う場合はde_risk/recovery/overweight gate限定。
7. BC評価で AlphaEx, SharpeΔ, MaxDDΔ, turnover, flat比率, route分位別advantage, recovery latency, transition matrix を出す。

ACへ進む条件:
fold4で AlphaEx >= +1pt, SharpeΔ >= +0.02, MaxDDΔ <= 0, turnover <= 4, flat <= 92%, collapse_guard pass。
妥協条件は AlphaEx >= 0, turnover <= 6, flat <= 95%, top-decile route advantage positive。
```

現時点の本命は **K → L → M**。
Nのpredictive gateはその後。ACは最後。
ここでBCが `Alpha +1pt / turnover <=4 / MaxDD改善` まで行けば、ようやくACチューニングに移行していい。

[1]: https://huggingface.co/papers/2110.06169?utm_source=chatgpt.com "Paper page - Offline Reinforcement Learning with Implicit Q-Learning"
[2]: https://www.sciencestack.ai/paper/2006.09359?utm_source=chatgpt.com "AWAC: Accelerating Online Reinforcement Learning with Offline Datasets (arXiv:2006.09359v6) - ScienceStack"
[3]: https://www.tandfonline.com/doi/abs/10.1080/17442508.2011.651219?utm_source=chatgpt.com "Optimal rebalancing of portfolios with transaction costs: Stochastics: Vol 85, No 3"
[4]: https://www.emergentmind.com/papers/2106.06860?utm_source=chatgpt.com "Minimalist Offline RL Approach"
[5]: https://www.sciencestack.ai/paper/2006.04779v3?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning (arXiv:2006.04779v3) - ScienceStack"
