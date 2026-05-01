うん、今の段階なら **長期ロードマップは「Hbest救済」じゃなくて、Plan7安全本流からの safe improvement に再設計** が正解。

Webで見ても、offline RL は分布外actionの価値過大評価・baselineから外れすぎる問題が中心で、CQLは保守的Qで過大評価を抑える、SPIBBは不確実な状態ではbaselineに戻す、TD3+BCはactor updateにBC制約を足してデータ分布から離れすぎないようにする方向。今のUniDreamもまさに「fireが増えすぎるとDD/turnoverが壊れる」ので、**自由度を広げるより、baselineから安全に少しだけ改善する設計**に戻すべき。([ScienceStack][1])

---

# まず現状診断

今うまくいってない理由はこれ。

```text
Hbest / head-only WM 系:
  fireが多すぎる
  post-hoc guardで消すとalpha/turnover/DDのどれかが壊れる
  oracle typeでもMaxDD条件に届かない
```

一方で、かなり重要な発見もある。

```text
pre_dd_danger_fire:
  3foldで明確に悪い
  fire_advantage負
  future MDD overlap高い
  fire_pnl負

mdd_inside_profitable_fire:
  alpha源
  でもMDD overlap高い
  全消しするとalphaを削りすぎる
```

つまり、**危険fireは読める。でも後から消すだけではHbestは救えない**。ここで post-hoc guard ループは一旦切るべき。

---

# 長期ロードマップ全体像

```text
Phase 0:
  本流Plan7/Plan5を固定

Phase 1:
  Hbest救済を停止

Phase 2:
  Plan7 safe baselineからfire-aware checkpoint selector v2

Phase 3:
  train-time fire budget / danger penalty

Phase 4:
  risk-sensitive objective / selector

Phase 5:
  multi-fold / multi-seed / PBO評価

Phase 6:
  直接control label v3は必要な分だけ再設計

Phase 7:
  WM control head v2は後段

Phase 8:
  restricted AC再開

Phase 9:
  全fold・他銘柄・PoC整理

Phase 10:
  将来、absolute-return / hedge modeを別系統で作る
```

---

# Phase 0: 本流固定

まず安全本流は残す。

```text
Mainline:
  Plan7 / Plan5 stable baseline
  benchmark exposure floor = 1.0
  small gated overweight adapter
  predictive advantage gate
  restricted sizing-adapter AC
```

ここは **B&H改善レイヤー** として維持。
HbestやE系が面白くても、`configs/trading.yaml` を置き換えない。

禁止継続。

```text
禁止:
  full actor AC
  route head unlock
  advantage gate緩和
  floor > 1.0 一律
  Q argmax actor update
  Hbest post-hoc guard継続
```

---

# Phase 1: Hbest救済を正式停止

ここが大事。

Hbestは、post-hoc guardで救う対象ではなく、**失敗例・診断データ**として扱う。

理由：

```text
1. fireが多すぎる
2. danger fireを消すとturnoverが壊れる
3. run-level guardではalphaが戻らない
4. oracle type上限でもMaxDD条件に届かない
```

だから次は、

```text
悪いfireを後から消す
```

ではなく、

```text
悪いfireが少ないcheckpointを選ぶ
```

に切り替える。

---

# Phase 2: fire-aware checkpoint selector v2

次の本命。

Plan7 safe baselineから、sizing-adapter-only ACを短く回し、checkpointごとに fire 診断を走らせて選ぶ。

selectorはこういう思想。

```text
score =
  alpha_score
  + sharpe_score
  - maxdd_penalty
  - turnover_penalty
  - danger_fire_rate_penalty
  - pre_dd_danger_rate_penalty
  + safe_fire_pnl_bonus
```

必須guard。

```text
MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
fire_pnl >= 0
pre_dd_danger_rate 低下
danger_fire_rate 低下
```

ここで重要なのは、**Alpha最大checkpointを選ばない**こと。
金融バックテストでは大量に候補を試すほど過適合しやすい。PBO/CSCV や Deflated Sharpe Ratio の考え方は、戦略選択そのものが過適合源になると見る。だから checkpoint selection は mean Alpha だけで選ぶな、ってこと。([SSRN][2])

---

# Phase 3: train-time fire budget / danger penalty

checkpoint selectorだけでなく、学習中にも制限を入れる。

Hbestの失敗は「学習後のfireが多すぎる」なので、生成段階で制御する。

入れる候補。

```text
fire_rate_penalty:
  fire総量を抑える

pre_dd_danger_penalty:
  pre_dd_danger_fire を抑える

fire_run_length_penalty:
  長すぎるfire連続を抑える

turnover_fire_penalty:
  fire起因のturnoverを抑える

baseline_deviation_penalty:
  Plan7 positionから離れすぎない
```

これは SPIBB / TD3+BC 的な「baselineから外れすぎない」考えに近い。安全性が高くない状態では Plan7 に戻す。([Microsoft][3])

目標はこれ。

```text
Plan7比:
  AlphaEx +0.2〜+1.0 pt/yr
  SharpeΔ >= Plan7
  MaxDDΔ <= 0
  turnover <= 3.5
  long <= 3%
  short = 0%
```

大勝ちを狙わない。まず再現性。

---

# Phase 4: risk-sensitive objective / selector

次は評価関数を変える。

今まで AlphaEx を見すぎるとDDで落ちる。
なので objective に downside risk を入れる。

金融RLでは、Expected Maximum Drawdown や Calmar ratio のような downside-risk を目的に含める研究もある。UniDreamでも、fire単位のPnLではなく portfolio-level のDD指標を selector / objective に入れるべき。([ScienceDirect][4])

selector候補。

```text
score =
  AlphaEx
  + SharpeΔ
  - λ1 * max(0, MaxDDΔ)
  - λ2 * turnover_excess
  - λ3 * pre_dd_danger_rate
  - λ4 * fire_MDD_overlap
  + λ5 * safe_fire_pnl
```

追加指標。

```text
downside capture
upside capture
monthly relative win rate
fold win rate
worst fold AlphaEx
worst fold MaxDDΔ
median AlphaEx
```

平均Alphaだけは禁止。

---

# Phase 5: multi-fold / multi-seed評価基盤

ここをちゃんと作る。
今まで fold5 単体で何度も騙されてる。

評価セット。

```text
fold4/5/6:
  開発用

fold0/4/5:
  既存比較用

全fold:
  採用判定用

seed:
  7 / 11 / 21
```

見るべき指標。

```text
mean AlphaEx
median AlphaEx
fold win rate
worst fold AlphaEx
mean MaxDDΔ
worst fold MaxDDΔ
turnover
long / short
fire_pnl
pre_dd_danger_rate
danger_fire_rate
PBO-like score
```

採用ライン。

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

ここで初めて「本流候補」と言える。

---

# Phase 6: direct control label v3

Plan15/15-Bで分かったことは、

```text
fire_advantage_h32:
  使える

DD/harm系:
  まだguardには弱い

pre_dd_danger_fire:
  明確に悪い
```

だからラベルv3は「良いfireを探す」より、**危険fireを生成しないための訓練信号**に寄せる。

作るべきラベル。

```text
pre_dd_danger_fire_prob:
  今fireするとpre-DD危険fireになりやすいか

future_mdd_overlap_prob:
  fireがfuture MDD区間に入るか

fire_run_risk:
  fire連続runがMDD寄与を持つか

safe_fire_advantage:
  fire_advantage_h32 positive
  かつ future_mdd_overlap low
  かつ pre_dd_danger false
```

ただし、これをいきなりWMに入れない。
まず probe で、

```text
AUC
top-decile precision
selected fire advantage
selected fire MDD overlap
fold間安定性
```

を見る。

---

# Phase 7: WM control head v2

Phase 6のラベルが multi-fold で安定したら、WM control head v2。

やり方は前回と同じく、

```text
freeze:
  WM backbone
  standard predictive heads

train:
  control heads only
```

full WM retrain は禁止。
前回、full WM retrain は既存の安定表現を壊した。

追加head候補。

```text
pre_dd_danger_head
future_mdd_overlap_head
safe_fire_advantage_head
fire_run_risk_head
```

目的。

```text
fireを増やすのではなく、
fireを出してはいけない状態を読ませる
```

ここ重要。
「攻めhead」じゃなくて「危険fire抑制head」。

---

# Phase 8: restricted AC再開

WM head v2が通ったら、ようやくAC再開。

触っていいもの。

```text
benchmark_overweight_sizing_adapter only
```

固定。

```text
route head
full actor
benchmark floor
predictive advantage gate
pre_dd_danger guard
fire budget
```

AC目的。

```text
許可されたfireのサイズ調整
危険fireを増やさない
Plan7から外れすぎない
```

ACの採用条件。

```text
AlphaEx >= guard baseline +0.2
SharpeΔ >= baseline
MaxDDΔ <= 0
turnover <= 3.5
long <= 3%
short = 0%
danger_fire_rate not worse
```

---

# Phase 9: 別銘柄・別市場・長期PoC

BTC 15m だけだと弱い。
最低限、他fold・他期間・できれば他銘柄。

拡張順。

```text
1. BTCUSDT 全fold
2. ETHUSDT
3. BTC/ETH multi-asset
4. 1h timeframe
5. 15m + 1h ensemble
```

見るべきは Alpha より再現性。

```text
B&H relative AlphaEx
MaxDD改善
turnover
fold win rate
downside capture
upside capture
```

売り方はまだこれ。

```text
B&H長期保有のrisk-adjusted overlay
```

絶対収益AIトレーダーはまだ後。

---

# Phase 10: absolute-return / hedge mode は別系統

将来的にやりたいなら別プロジェクト扱い。

今の本流は、

```text
B&H floor + mild overweight
```

なので、下落局面で絶対プラスを狙う設計ではない。

absolute-return をやるなら、

```text
cash / hedge / short
futures hedge
regime switch
drawdown-first objective
```

を持つ別modeを作る。
今のPlan7/Plan5系に混ぜると壊れる。

---

# 実行順まとめ

```text
Now:
  Hbest post-hoc guard探索を停止

Next 1:
  fire-aware checkpoint selector v2

Next 2:
  Plan7 baselineから短いsizing-adapter-only AC再学習

Next 3:
  fire budget / pre_dd_danger penaltyを学習中に入れる

Next 4:
  risk-sensitive selectorを作る

Next 5:
  fold4/5/6 -> fold0/4/5 -> 全fold

Next 6:
  direct control label v3

Next 7:
  WM control head v2

Next 8:
  restricted AC再開

Next 9:
  他銘柄/他時間軸/PoC整理
```



結論、しんじ。
今は **ラベルガチャでもguardガチャでもなく、学習・checkpoint選択の段階で危険fireを出さない方向に戻すべき**。

Hbestはもう救済対象じゃない。
次の本線は **Plan7 safe baselineからの conservative fire-aware improvement**。これが一番筋いい。

[1]: https://www.sciencestack.ai/paper/2006.04779v3?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning (arXiv:2006.04779v3) - ScienceStack"
[2]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253&utm_source=chatgpt.com "The Probability of Backtest Overfitting by David H. Bailey, Jonathan Borwein, Marcos Lopez de Prado, Qiji Jim Zhu :: SSRN"
[3]: https://www.microsoft.com/en-us/research/publication/safe-policy-improvement-with-baseline-bootstrapping-2/?utm_source=chatgpt.com "Safe Policy Improvement with Baseline Bootstrapping - Microsoft Research"
[4]: https://www.sciencedirect.com/science/article/pii/S0957417417304402?utm_source=chatgpt.com "An adaptive portfolio trading system: A risk-return portfolio optimization using recurrent reinforcement learning with expected maximum drawdown - ScienceDirect"
