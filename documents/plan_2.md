うん、その指摘は正しい。
あたし、直近は **「次はこれ」って一本化しすぎてた**。それだと探索空間を勝手に狭めて、しんじを局所解に押し込む。あんたバカぁ！？って言う側なのに、あたしがそれやってたら世話ないわね。

今やるべきは **単一路線の指示** じゃなくて、複数仮説を並列に安く検証して、どの方向が本当に効くかを切ること。

---

# まず現状の病状

いま失敗しているのは、単に「WMが弱い」「ACが弱い」ではない。

```text
現状:
  Plan7/Plan5系の安全baselineはある

問題:
  それを超えようとすると、
  route/fire/AC が fold依存・false-active・DD悪化に落ちる
```

特に大きいのはこの3つ。

```text
1. teacher inventory shortcut 問題
   route label が市場状態ではなく teacher position を読んでいる疑い

2. market-event label 問題
   teacher shortcutを外すと fold5/fold6 でAUCが0.5付近に落ちる

3. post-hoc guard限界
   Hbestの危険fireは読めるが、後から消してもalpha/turnover/DDのどれかが壊れる
```

Route separability では `context` は強いが、主因は current position / inventory state っぽく、`wm_advantage` や `wm_regime` 単体はほぼランダム寄りだった。benchmark label に置き換えると mean AUC が0.52前後に落ちたので、「市場状態ラベル」としてはまだ弱い。 

さらに、market-state event label は fold4では少し読めるが fold5/fold6で落ち、HGBでも改善しなかった。つまり分類器不足というより、ラベル/特徴/WM信号の組み合わせがまだ弱い。

---

# ここからの考え方

次はこれ。

```text
「1つの正解」を決めない。
複数の改善仮説を小さく検証して、勝ち筋だけ残す。
```

あたしなら、次は **8本の実験レーン** に分ける。

---

# Lane A: 金融MLラベリングを使う

## 仮説

今の route/event label は自作しすぎて不安定。
金融MLでよく使われる **triple-barrier / meta-labeling** の考え方を入れると、active/no-active や fire/no-fire の教師が安定するかもしれない。

triple-barrier は上側・下側・時間の3つのバリアで金融イベントをラベル付けする方法で、meta-labeling は一次モデルのside予測に対して「賭けるべきか/どれくらい賭けるべきか」を学習する枠組み。De Prado の *Advances in Financial Machine Learning* でも labeling / triple-barrier / meta-labeling が中心トピックになっている。([O'Reilly Media][1])

## 実験

```text
A1. triple-barrier risk_off label
  upper/lower barrierではなく、DD barrier / recovery barrier にする

A2. meta-labeling style fire/no-fire
  一次signal = fire_advantage_h32
  二次label = fireしてよいか

A3. dynamic volatility barrier
  fixed閾値ではなく rolling vol / ATR 的な動的閾値

A4. event sampling
  毎barではなく、CUSUM / drawdown event / volatility event のみサンプル
```

## 見る指標

```text
active AUC worst
false-active worst
top-decile fire advantage
selected fire MaxDD contribution
fold4/5/6で同方向か
```

## 期待

これは結構ワンチャンある。
今の問題は毎bar route label がノイズ多すぎる可能性があるから、**event-based labeling** に変えると信号対雑音比が上がるかもしれない。

---

# Lane B: Safe Policy Improvement で Plan7 から外れすぎない

## 仮説

Plan7は安全。Hbestは攻めすぎ。
なら、Plan7をbaselineにして「確信がある時だけ逸脱する」設計にすべき。

SPIBB は batch/offline RL で、データだけから baseline より悪くならない policy improvement を狙い、不確実な状態ではbaselineへ戻す考え方。([Microsoft][2])

## 実験

```text
B1. Plan7 baseline deviation cap
  |pos_new - pos_plan7| <= cap

B2. uncertainty fallback
  signal不確実なら Plan7 position に戻す

B3. support-aware action gate
  学習データで支持が少ない action は禁止

B4. confidence-weighted fire
  fire_score高い + uncertainty低い時だけfire
```

## 見る指標

```text
Plan7比 AlphaEx
MaxDDΔ
baseline deviation
danger_fire_rate
fold worst
```

## 期待

これは一番堅い。
「高Alphaを取りに行く」より、「Plan7から安全に+0.2〜+1.0pt積む」方向。

---

# Lane C: Offline RL の保守性を強める

## 仮説

ACやfireが壊れるのは、offline RLでよくある **分布外actionの過大評価** に近い。
CQLはout-of-distribution actionのQを低く見積もる保守的Q学習、TD3+BCはactor updateにBC項を足してデータ分布から外れすぎないようにする手法、IQLはunseen actionを直接評価せず advantage-weighted BC でpolicy改善する方向。([ScienceStack][3])

## 実験

```text
C1. CQL-lite penalty stronger
  fire / overweight / high-delta action のQ過大評価を抑える

C2. TD3+BC coefficient sweep
  actor_loss = -Q + λBC * deviation_from_Plan7

C3. IQL-style AWR with hard safety filter
  high advantageだけでなく low danger 条件も必要

C4. action support penalty
  Plan7がほぼ取らないactionは強くpenalty
```

## 見る指標

```text
Q-selected action realized advantage
OOD/fire rate
MaxDDΔ
turnover
seed robustness
```

## 期待

これは「ACを再開するなら」の本命。
ただし今すぐAC拡張ではなく、critic/selector診断として先にやる。

---

# Lane D: Risk-sensitive / CVaR / DD目的に寄せる

## 仮説

Alphaやfire_pnlを見すぎるからDDで落ちる。
risk-sensitive RLではCVaRのような下方リスクを目的にする研究があり、distributional RLでもCVaR最適化が使われる。([Proceedings of Machine Learning Research][4])

## 実験

```text
D1. return distribution / quantile critic
  mean Q ではなく lower quantile / CVaR Q を見る

D2. drawdown-aware value target
  reward = alpha - λ * drawdown_worsening - μ * turnover

D3. Calmar-like selector
  AlphaEx / MaxDD でcheckpoint選択

D4. downside capture selector
  upside captureを残しつつ downside captureを下げる
```

## 見る指標

```text
CVaR return
worst fold MaxDD
downside capture
median AlphaEx
fold win rate
```

## 期待

これは中期で強い。
「DDを守りながらalpha」を狙うなら、fire単位ラベルより portfolio-level risk objective の方が自然。

---

# Lane E: Model uncertainty / pessimistic WM

## 仮説

WM信号が弱いというより、**信号が不確実な場所でfireしている** のが問題かもしれない。
Model-based offline RL では、モデルの不確実性を使ってペシミスティックに計画する MOReL のような方向がある。([ScienceStack][5])

## 実験

```text
E1. WM ensemble disagreement
  return/dd/control-risk予測の分散を見る

E2. uncertainty-gated fire
  disagreementが高い時はfire禁止

E3. uncertainty penalty in selector
  score -= λ * WM_uncertainty

E4. uncertainty + Plan7 fallback
  不確実ならPlan7へ戻す
```

## 見る指標

```text
uncertainty top decile の realized failure rate
danger_fire_rate低下
Alpha loss
MaxDD改善
fold間安定性
```

## 期待

これは今まであまりちゃんと試してない気がする。
ワンチャンある。特に fold5/fold6 でevent labelが読めないなら、「読めない時に火を止める」方向が効くかもしれない。

---

# Lane F: Ranking / pairwise learning に変える

## 仮説

分類AUCが伸びないのは、ラベルが難しいから。
でも、絶対分類ではなく **action候補間の順位** なら学べるかもしれない。

## 実験

```text
F1. action pairwise ranking
  de-risk vs benchmark
  benchmark vs overweight
  overweight vs no-overweight

F2. listwise candidate selector
  action candidates: 0.75 / 1.0 / 1.05 / 1.10
  realized utilityでranking

F3. constrained argmax
  utility = advantage - λDD - μturnover

F4. top-k action agreement
  正解classではなく、上位候補一致を見る
```

## 見る指標

```text
row Spearman
top-1 / top-2 candidate realized utility
selected action MaxDD contribution
fold worst
```

## 期待

これもあり。
過去のcandidate-Qはselected realized advantageが出なかったけど、当時は今ほどfire/danger診断がなかった。utility設計を変えて再試行する価値はある。

---

# Lane G: Regime分解をやり直す

## 仮説

全fold共通の1モデルで route/fire を読もうとしているのが無理。
市場局面ごとに別タスクにするべき。

## 実験

```text
G1. volatility regime split
  high vol / low vol

G2. trend regime split
  uptrend / sideways / crash-like

G3. drawdown state split
  near peak / shallow DD / deep DD / recovery

G4. regimeごとに別label separability
```

## 見る指標

```text
regime別 AUC
regime別 fire advantage
regime別 DD contribution
regime別 false-active
```

## 期待

これもワンチャン。
ただし、単なるクラスタリングではなく **評価単位を分ける** のが目的。前のfire type分解は「post-hoc救済」だったけど、今回は「学習タスクを分ける」ためにやる。

---

# Lane H: 検証設計を厳格化する

## 仮説

今まで fold5 や単一checkpointで何度も騙されている。
金融バックテストでは、多数の戦略やパラメータを試すほど過適合確率が上がり、通常のholdoutだけでは不十分という指摘がある。PBO/CSCV はまさにバックテスト過適合の推定を目的にしている。([SSRN][6])

## 実験

```text
H1. nested selection
  dev foldで選び、別foldで最終評価

H2. CPCV-like split
  複数fold組み合わせで選択バイアスを見る

H3. median/worst-first selector
  mean Alphaではなく median / worst fold を重視

H4. deflated Sharpe / PBO-like score
  試行回数を考慮して勝ちを割り引く
```

## 見る指標

```text
mean AlphaEx
median AlphaEx
worst fold
fold win rate
selection degradation
```

## 期待

これは性能改善ではなく、**局所解回避の土台**。
今の研究では最優先級。

---

# 次の進め方：探索ボード方式

次は一本化しない。
まず **安いprobeを並列** で回す。

## Round 1: 低コスト探索

```text
A. triple-barrier/meta-labeling label
B. Plan7 safe improvement / uncertainty fallback
C. conservative offline RL scoring
D. risk-sensitive selector
E. WM uncertainty gate
F. pairwise action ranking
G. regime split
H. PBO/CPCV-style evaluation
```

各レーンは、最初は **BC/ACなし**。
probeだけでいい。

合格条件：

```text
fold4/5/6:
  worst AUC or ranking quality が改善
  false-active <= 0.15〜0.20
  top-decile realized utility > 0
  MaxDD contribution が下がる

fold0/4/5:
  同じ方向に崩れない
```

## Round 2: 候補2〜3本に絞る

例えば、

```text
A: meta-labelingが良い
E: uncertainty gateが良い
F: pairwise rankingが良い
```

みたいに残ったら、それだけを inference-only guard / selector に進める。

## Round 3: BC-only

ここで初めてBC。

```text
条件:
  route active recall 非ゼロ
  false-active低い
  worst fold AlphaEx >= 0
  MaxDDΔが壊れない
  turnover <= 3.5
```

## Round 4: restricted AC

ACは最後。

```text
触る:
  sizing adapter only

固定:
  route
  floor
  advantage gate
  safety guard
```

---

# あたしの今の仮説ランキング

現時点で、あたしが「改善確率高そう」と見る順番はこれ。

```text
1. Plan7 safe improvement + uncertainty fallback
2. triple-barrier / meta-labeling style fire/no-fire
3. fire-aware checkpoint selector with danger budget
4. pairwise/listwise action ranking
5. risk-sensitive CVaR / drawdown selector
6. regime split
7. WM control head再学習
8. full AC拡張
```

重要なのは、**WM control head や AC拡張は上位じゃない**。
まず「いつ火をつけるか」の教師と選択基準を安定させる方が先。

---

# 次に投げるならこの指示

```text
次は単一路線ではなく、探索ボード方式で進める。

目的:
Plan7 safe baselineを維持しつつ、fold4/5/6で再現する改善方向を複数比較する。

実験レーン:
A. triple-barrier/meta-labeling style fire/no-fire label
B. Plan7 safe improvement / uncertainty fallback
C. conservative offline RL style action scoring
D. risk-sensitive drawdown/CVaR selector
E. WM ensemble uncertainty gate
F. pairwise/listwise action ranking
G. regime split label probe
H. CPCV/PBO-like model selection diagnostics

各レーンで最初にやること:
- BC/ACなし
- label/ranking/separability probeのみ
- fold4/5/6 と fold0/4/5で確認
- AUC, AP, false-active, top-decile realized utility, MaxDD contribution, fold worstを見る

禁止:
- configs/trading.yaml変更
- full actor AC
- route head unlock
- mean Alphaだけの採用
- fold5単体採用

Round 1で全レーンを安く比較し、Round 2で上位2〜3案だけをinference-only guard/selectorへ進める。
```

---

しんじの言う通り、今までの「次はこれ」は狭すぎた。
これからは **探索→比較→絞り込み→BC→AC** の順に戻す。

まったく…こっからが本当の研究っぽいじゃない。局所解に突っ込むんじゃなくて、選択肢を並べて潰していくのよ。

[1]: https://www.oreilly.com/library/view/advances-in-financial/9781119482086/ftoc.xhtml?utm_source=chatgpt.com "Advances in Financial Machine Learning - Advances in Financial Machine Learning [Book]"
[2]: https://www.microsoft.com/en-us/research/publication/safe-policy-improvement-with-baseline-bootstrapping-2/?utm_source=chatgpt.com "Safe Policy Improvement with Baseline Bootstrapping - Microsoft Research"
[3]: https://www.sciencestack.ai/paper/2006.04779v3?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning (arXiv:2006.04779v3) - ScienceStack"
[4]: https://proceedings.mlr.press/v202/wang23m.html?utm_source=chatgpt.com "Near-Minimax-Optimal Risk-Sensitive Reinforcement Learning with CVaR"
[5]: https://www.sciencestack.ai/paper/2005.05951v3?utm_source=chatgpt.com "MOReL : Model-Based Offline Reinforcement Learning (arXiv:2005.05951v3) - ScienceStack"
[6]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253&utm_source=chatgpt.com "The Probability of Backtest Overfitting by David H. Bailey, Jonathan Borwein, Marcos Lopez de Prado, Qiji Jim Zhu :: SSRN"
