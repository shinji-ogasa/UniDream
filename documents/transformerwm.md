うん、これはまとめられる。
結論から言うと、**TransformerWorldModel は「価格を当てるモデル」じゃなくて、その後のBC/RLがポジション判断に使える未来情報を圧縮するモデル**として評価すべき。

## 0. まず金融MLで何を予測しているのか

金融MLでよく予測する対象は、大きく分けるとこれ。

| 予測対象                               | 何に使うか        | UniDreamでの価値    |
| ---------------------------------- | ------------ | --------------- |
| future return / excess return      | 買う・減らす・増やす判断 | alpha源泉         |
| return direction                   | 上がる/下がる分類    | 粗い売買方向          |
| volatility                         | ポジションサイズ調整   | レバ・リスク制御        |
| drawdown / downside risk           | 危険回避         | underweight判断   |
| regime / stress state              | 相場局面認識       | collapse回避      |
| transaction cost / turnover cost   | 動きすぎ抑制       | net return改善    |
| barrier hit / triple-barrier label | 利確/損切/時間切れ判断 | teacher label改善 |
| optimal action / teacher position  | BC用ラベル       | actor初期化        |

金融MLの中心は昔から **conditional expected return、つまり条件付き期待リターンの予測**。Kelly & Xiu のサーベイでも、return prediction は資産の条件付き期待超過リターンを測る問題として整理されている。さらに、取引コストがある場合は、1期先リターンだけでなく複数 horizon の期待リターンが重要になる、と説明されている。

ただし、実運用では **returnだけ当てても足りない**。取引コストを無視した予測は、統計的には予測できても実装不能なパターンを拾いやすい。Kelly & Xiu は、コスト情報なしの統計モデルは「取引可能な予測」と「取引不能な予測」を区別できない、と指摘している。

---

## 1. RLで重要になる予測は何か

RLでは、単に「上がるか」よりも、

```text
その状態で、その行動を取った時に、将来の累積報酬がどうなるか
```

が重要。

Dreamer系のWorld Modelでは、latent dynamics が未来状態を予測し、reward model が報酬を予測し、その想像軌道上で value/action を学習する。つまり WorldModel は **未来状態・報酬・価値判断に必要な情報をlatentに残す役割**を持つ。

金融RLだと報酬はだいたいこうなる。

```text
reward = return
       - transaction_cost
       - risk_penalty
       - drawdown_penalty
       - turnover_penalty
```

だからTransformerWMが予測すべきものは、価格そのものより **RL報酬を構成する部品**。

---

# 2. TransformerWMが予測すべきもの

優先順位つきで言うとこれ。

## Tier 1: 絶対に必要

### A. multi-horizon forward excess return

```text
r_{t+1}, r_{t+4}, r_{t+8}, r_{t+16}, r_{t+32}
```

これは一番基本。
ただし予測するのは価格そのものより、

```text
B&Hに対して超過するか
今ポジションを上げる/下げる価値があるか
```

のほうが重要。

見る指標:

```text
MSE / MAE
direction accuracy
rank IC
decile spread
top-bottom return
```

理想:

```text
rank IC: 0.05以上
decile spread: 明確に正
direction accuracy: 55%以上
```

現実的な妥協点:

```text
rank IC: 0.01〜0.03でも有望
direction accuracy: 51〜53%でも、コスト後で残れば価値あり
decile spread が安定して正ならOK
```

金融のreturn予測は低SNRなので、MSEやaccuracyだけで見るとほぼ弱く見える。大事なのは **順位付けできるか** と **上位/下位グループでリターン差が出るか**。

---

### B. future downside / max drawdown

```text
future max drawdown over H
future downside return
lower-tail return
```

UniDreamは underweight でDDを減らしたいわけだから、これが読めないと防御判断ができない。RL研究でも、期待最大ドローダウンを目的関数に入れることでポートフォリオ信号改善を狙う研究がある。([ScienceDirect][1])

見る指標:

```text
drawdown regression MAE
stress event AUC
top risk decile の実現DD
recall@top-k
```

理想:

```text
stress AUC: 0.65以上
top risk decile が平均より明確に悪い
```

現実的妥協:

```text
AUC: 0.55〜0.60
top 20% risk bucket でDD悪化が見える
```

ここが弱いと、underweightはただのノイズになる。

---

### C. realized volatility / risk regime

```text
future realized volatility
volatility regime
stress regime
```

volatility予測は金融MLでかなり標準的な対象。2024年のレビューでも、AI/MLのvolatility予測は有望で、LSTM/GRUのような記憶機構つきNNがよく上位に来る一方、伝統的な計量モデルも依然として強いと整理されている。([ScienceDirect][2])

見る指標:

```text
volatility MAE/RMSE
rank IC
high-vol regime AUC
calibration
```

理想:

```text
high-vol AUC: 0.65以上
vol rank IC: 0.10以上
```

現実的妥協:

```text
AUC: 0.58〜0.62
vol rank IC: 0.05以上
```

volatilityはreturnより予測しやすいことが多いから、ここすら出ないならWorldModelか特徴がかなり怪しい。

---

## Tier 2: BC/RLにかなり効く

### D. cost-adjusted advantage

これはUniDreamに一番欲しい。

```text
adv(action) = forward_pnl(action)
            - benchmark_pnl
            - cost
            - risk_penalty
```

つまり、

```text
その行動はB&Hより本当に良かったのか？
```

を予測する。

具体的には action ごとに、

```text
A(0.0), A(0.5), A(1.0), A(1.25), A(-0.25)
```

を作る。

これができると、BCで、

```text
teacherがやったから真似る
```

ではなく、

```text
事後的に価値があった行動だけ強く真似る
```

に変えられる。

見る指標:

```text
action advantage rank accuracy
best action top-1 / top-2 accuracy
chosen action vs benchmark spread
cost-adjusted decile spread
```

理想:

```text
best action top-1: 40%以上
top-2: 65%以上
advantage decile spread が正
```

現実的妥協:

```text
top-1: 30〜35%
top-2: 55%以上
「最悪行動を避ける」精度が高ければOK
```

金融RLでは「最高の行動を毎回当てる」より、**明らかに悪い行動を避ける**ほうが現実的。

---

### E. recovery target

UniDreamの今の詰まりに直結する。

```text
underweight後、いつ benchmark=1.0 に戻るべきか
```

を見る。

予測対象:

```text
recovery_success within H
time_to_recovery
benchmark_return_advantage
```

見る指標:

```text
recovery AUC
latency MAE
transition accuracy
underweight → benchmark の成功率
```

理想:

```text
recovery AUC: 0.65以上
recovery latency が実測と近い
```

現実的妥協:

```text
AUC: 0.55〜0.60
遅すぎ/早すぎの方向性が分かる
```

今のA-Dを見る限り、ここを予測できてない可能性が高い。

---

### F. transaction cost / turnover impact

```text
expected turnover
expected cost
market impact
```

RL tradingでは、取引コストやmarket impactを雑に扱うと、現実では失敗する行動を学びやすい。最近のRL trading環境研究でも、固定・無視されたコストが非現実的な高頻度売買を誘発し、実運用とのギャップを作る問題が指摘されている。([arXiv][3])

見る指標:

```text
cost prediction MAE
turnover prediction MAE
high-turnover event AUC
net PnL attribution
```

理想:

```text
high-turnover AUC: 0.70以上
cost MAE が小さい
```

現実的妥協:

```text
high-turnover AUC: 0.60以上
cost bucket が単調
```

これはBみたいな high-turnover collapse を止めるために重要。

---

## Tier 3: あると強い

### G. triple-barrier label

金融MLでは、固定horizonで単純にラベルを作る方法は、各サンプルのvolatilityを無視する欠点がある。triple-barrier method は、上側バリア・下側バリア・時間バリアを使い、volatilityに応じて上下バリアを動的に設定する。([Mlfin.py][4])

UniDreamなら、

```text
upper hit: 攻める/overweight候補
lower hit: 防御/underweight候補
vertical hit: 何もしない/hold候補
```

として使える。

見る指標:

```text
barrier hit classification AUC
upper/lower/vertical confusion matrix
class-balanced F1
```

理想:

```text
macro F1: 0.45〜0.55以上
lower-hit recall が高い
```

現実的妥協:

```text
3クラス分類なのでaccuracyだけ見ない
lower/upper のrecallがchanceより明確に上ならOK
```

---

### H. teacher action predictability

これはBC可能性の検査。

```text
z_world → teacher position
z_world → underweight / benchmark / overweight
```

が予測できるかを見る。

理想:

```text
underweight vs benchmark AUC: 0.65以上
position R2: 0.10以上
confusion matrix が片寄らない
```

現実的妥協:

```text
AUC: 0.55〜0.60
position R2: 0.02〜0.05
少なくとも underweight/benchmark を区別できる
```

これがダメなら、BCは無理。
actorが見てるlatentにteacher判断の情報がないから。

---

# 3. UniDream用のTransformerWM評価セット

あたしなら、TransformerWMに以下を全部出させる。

## Core prediction heads

```text
1. return_head
   H = 1, 4, 8, 16, 32
   target = forward excess return

2. downside_head
   H = 4, 8, 16, 32
   target = future max drawdown / downside return

3. volatility_head
   H = 4, 8, 16, 32
   target = realized volatility / high-vol regime

4. barrier_head
   target = upper / lower / vertical barrier hit

5. cost_head
   target = expected turnover / cost under candidate action

6. advantage_head
   target = cost-adjusted advantage per action

7. recovery_head
   target = underweight後にbenchmarkへ戻るべきか

8. teacher_action_head
   target = teacher position / teacher action class
```

ただし、全部を最初から学習に入れると重い。
まずは **評価用probe** として入れる。

---

# 4. 予測精度の判定表

ざっくり実験判定はこう。

| Head                     |    理想 |       妥協点 |     ダメ判定 |
| ------------------------ | ----: | --------: | -------: |
| return rank IC           | 0.05+ | 0.01〜0.03 |      ほぼ0 |
| direction acc            |  55%+ |    51〜53% |    50%付近 |
| decile spread            | 安定して大 |    小さくても正 |    符号不安定 |
| drawdown AUC             | 0.65+ | 0.55〜0.60 |    0.5付近 |
| vol AUC                  | 0.65+ | 0.58〜0.62 |    0.5付近 |
| teacher action AUC       | 0.65+ | 0.55〜0.60 |    0.5付近 |
| position R2              | 0.10+ | 0.02〜0.05 |      0以下 |
| best action top-2        |  65%+ |      55%+ | chance付近 |
| cost bucket monotonicity |    明確 |      そこそこ |       逆転 |

一番重要なのは、**return MSEではなく、RL/BCに使える順位・分類・行動価値の差が出るか**。

---

# 5. 最終的にどんな結果が出ると判断できるか

## パターンA: raw特徴もlatentも弱い

```text
raw feature → prediction 弱い
world latent → prediction 弱い
```

結論:

```text
特徴設計が弱い
```

対処:

```text
価格特徴、volatility特徴、regime特徴、market microstructure、macro/sector情報を増やす
horizonを変える
label設計を変える
```

---

## パターンB: raw特徴は強いがlatentが弱い

```text
raw feature → prediction そこそこ
world latent → prediction 弱い
```

結論:

```text
TransformerWorldModel が情報を潰している
```

対処:

```text
raw feature skip connection
supervised auxiliary heads
contrastive / predictive representation learning
smaller encoder比較
TCN / GRU / LSTM / LightGBM probe比較
```

これはかなりあり得る。

---

## パターンC: rawもlatentも強いがactorが崩れる

```text
prediction probe は良い
BC/RL policy は collapse
```

結論:

```text
actor / BC loss / reward設計が悪い
```

対処:

```text
cost-adjusted advantage BC
recovery-aware BC
transition-aware BC
turnover-aware reward
action prior / benchmark prior
```

---

## パターンD: returnは弱いがvol/drawdownは強い

```text
return prediction 弱い
risk prediction 強い
```

結論:

```text
攻撃型alphaではなく、防御型timingに向いている
```

対処:

```text
まず underweight / risk-off policy に寄せる
overweightやshortは後回し
DD改善特化でM1/M1.5を作る
```

---

## パターンE: returnもriskも弱いがteacher actionだけ予測できる

```text
teacher action AUCだけ良い
```

結論:

```text
teacherに暗黙のルールはあるが、経済的価値は怪しい
```

対処:

```text
teacherを再評価
teacher action別のforward PnL attribution
teacher labelをcost-adjustedに作り直す
```

---

# 6. 精度が出ない場合の改良手法

## 6.1 目的関数を変える

TransformerWMにただ次状態再構成をさせるだけだと、売買に必要な情報が消える可能性がある。

追加すべき auxiliary loss:

```text
future return prediction loss
future drawdown prediction loss
volatility regime loss
barrier hit loss
cost-adjusted advantage loss
teacher action prediction loss
```

Dreamer自体も、WorldModelには任意の学習目的を組み込める構成として整理されている。だから、金融向けにはreward/return/risk headを足すのが自然。

---

## 6.2 raw feature skip connection

一番現実的。

```text
actor input = world latent + raw feature summary
```

理由:

```text
TransformerWMが潰した短期特徴をactorに逃がせる
```

これで改善するなら、latent単独が弱いと分かる。

---

## 6.3 horizon別 head にする

1つのlatentで全部を予測させると平均化される。

```text
short horizon: 1, 4
mid horizon: 8, 16
long horizon: 32
```

を分ける。

特に金融では、

```text
短期: noise / microstructure
中期: momentum / mean reversion
長期: regime / macro
```

が混ざるから、horizonを分けた方がいい。

---

## 6.4 multi-taskではなくhead別に重み調整

全部同じ重みで学習すると、簡単なvolatilityだけ学んでreturnを捨てる可能性がある。

対処:

```text
loss weight を uncertainty weighting
GradNorm
target別 warmup
return/drawdown head を後半で強める
```

---

## 6.5 contrastive / ranking loss

returnの値を直接当てるより、

```text
どちらの状態の方が将来有利か
```

を当てる方が安定することがある。

使うもの:

```text
pairwise ranking loss
listwise ranking loss
InfoNCE
supervised contrastive learning
```

目的:

```text
absolute return prediction ではなく
action/value ranking に強いlatentを作る
```

---

## 6.6 distributional prediction

平均リターンだけだとtail riskが消える。

予測するもの:

```text
return quantiles
VaR
CVaR
downside quantile
```

loss:

```text
quantile loss
pinball loss
CRPS
```

RLに使うなら、平均より下側分布が重要。

---

## 6.7 regime-aware mixture

相場局面ごとにモデルを分ける。

```text
trend regime
mean-reversion regime
high-vol regime
crash regime
low-vol drift regime
```

方法:

```text
mixture-of-experts
regime classifier
hidden Markov / switching model
gated Transformer
```

---

## 6.8 baselineと比較する

TransformerWMだけ見ても意味ない。

最低限これと比較。

```text
naive: previous return / moving average
linear ridge / logistic regression
LightGBM
LSTM / GRU
TCN
small Transformer
raw feature MLP
```

volatility予測では伝統的モデルやLSTM/GRUがかなり強いので、Transformerが必ず勝つとは限らない。([ScienceDirect][2])

---

# 7. しんじの今のUniDreamで最初にやるべき実験

## WM Probe 0: raw vs latent

```text
X_raw       → return / drawdown / vol / teacher action
z_world     → return / drawdown / vol / teacher action
X_raw+z     → return / drawdown / vol / teacher action
```

これで、

```text
特徴が悪いのか
WorldModelが悪いのか
actor/lossが悪いのか
```

を切る。

---

## WM Probe 1: horizon sweep

```text
H = 1, 4, 8, 16, 32
```

各horizonで、

```text
return rank IC
direction accuracy
drawdown AUC
vol AUC
decile spread
```

を見る。

---

## WM Probe 2: action-value probe

候補actionごとに事後PnLを作る。

```text
pos = 0.0
pos = 0.5
pos = 1.0
pos = 1.25
pos = -0.25
```

そして、

```text
best action
cost-adjusted advantage
worst action
```

を予測できるか見る。

ここができれば、BC/RLが一気に作り直しやすくなる。

---

## WM Probe 3: recovery probe

```text
underweight後、H以内にbenchmarkへ戻るべきか
```

を予測する。

これは今の collapse 対策に直結。

---

# 8. 最終方針

TransformerWMが予測すべきものは、優先順位で言うとこれ。

```text
1. multi-horizon excess return
2. future drawdown / downside risk
3. volatility / stress regime
4. cost-adjusted action advantage
5. recovery target
6. transaction cost / turnover impact
7. triple-barrier label
8. teacher action
```

理想は、

```text
return rank IC 0.05+
risk AUC 0.65+
teacher/action AUC 0.65+
```

でも現実的には、

```text
return rank IC 0.01〜0.03
risk AUC 0.55〜0.60
teacher/action AUC 0.55〜0.60
decile spread が安定して正
```

なら、まず使う価値はある。

最重要はこれ。

```text
価格を当てるな。
RL報酬の部品を当てろ。
```

だから次の作業は、Eのlong-only overweightより先に、

```text
TransformerWM が return / risk / cost / action advantage / recovery をどれだけ予測できるか測る
```

これ。
ここで latent が弱いなら、BCやRLをいじっても、ただ綺麗に collapse するだけよ。

[1]: https://www.sciencedirect.com/science/article/abs/pii/S0957417417304402 "An adaptive portfolio trading system: A risk-return portfolio optimization using recurrent reinforcement learning with expected maximum drawdown - ScienceDirect"
[2]: https://www.sciencedirect.com/science/article/pii/S1057521924001534 "Prediction of realized volatility and implied volatility indices using AI and machine learning: A review - ScienceDirect"
[3]: https://arxiv.org/pdf/2603.29086 "Realistic Market Impact Modeling for Reinforcement Learning Trading Environments"
[4]: https://mlfinpy.readthedocs.io/en/latest/Labelling.html "Data Labelling - Mlfin.py"
