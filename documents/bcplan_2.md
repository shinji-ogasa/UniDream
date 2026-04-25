結論：**次は「WM改善」でも「予測state直結」でもなく、`行動/遷移ごとの価値ラベルを作って、routing policy に分解する` のが本線**。

いまの失敗パターンはかなり明確。

```text id="wtn6ct"
予測特徴をそのまま入れる
→ risk-off shortcut
→ underweight collapse

underweight頻度コピーを弱める
→ active decision消失
→ flat / benchmark hold

overweightを許す
→ long collapse
→ DD悪化
```

報告書でも、predictive state の実装自体は完了しているけど、direct concat は `scale=0` でも short に寄り、adapterでも予測値を入れるほど underweight に寄る、と整理されてる。つまり**特徴が無意味なのではなく、BC loss が特徴を「逃げる理由」として使っている**状態。
さらに検証計画側でも、A-Dは flat/high-turnover/active消失、Eは long collapse になっていて、最終的に `benchmark / de-risk / recovery / overweight` の routing が必要、という結論になってる。

## 次の主仮説

**Hypothesis 1: positionを直接真似るBCが悪い。action/transition別の cost-adjusted advantage で重み付けすれば collapse が減る。**

IQL は、offline RLで未観測actionを直接評価せず、advantage-weighted BCでpolicyを抽出する設計。これは今のUniDreamの「全teacher actionを同じ強さで真似てcollapseする」問題にかなり合う。([Hugging Face][1])
AWACも advantage-weighted な actor update で offline data を使う方向なので、方向性としては同じ。([ScienceStack][2])

やることはこれ。

```text id="qx9dh9"
各tについて候補transitionを作る:
  hold benchmark
  de-risk
  recovery
  overweight
  stay underweight

それぞれに cost_adjusted_advantage を計算:
  future excess return
  - transaction cost
  - turnover penalty
  - drawdown penalty
  - volatility penalty
  - leverage penalty

adv > margin のtransitionだけ強くBC
adv <= 0 は弱くする/真似ない
```

最初の実装は厳密なQ学習じゃなくていい。
まずは realized future path から作る supervised advantage label で十分。

```text id="9veb3c"
w = clip(exp(adv / tau), w_min, w_max)
loss = w * transition_CE + position_loss + turnover_penalty
```

合格条件：

```text id="j2v7e5"
fold4:
  short/underweight <= 70%
  flat 100% ではない
  turnover <= 2
  cost悪化なし
  maxdd_delta <= 0
  alpha_excess >= 0 近辺
```

---

**Hypothesis 2: single actor が全部を1つのpositionに潰している。routing headに分ければ極端解が減る。**

今のActorは、状態を見て最終positionを一発で出してる。これだと予測特徴が入った瞬間、

```text id="vw8f2z"
高vol/DD予測 → とりあえず下げる
```

に潰れる。

だから次は単一headではなく、

```text id="7xo01l"
route:
  neutral
  de-risk
  recovery
  overweight

position:
  routeごとの局所position
```

に分けるべき。

特に predictive state は position head に直結しない。
入れる場所は gate 側だけ。

```text id="8a9r3j"
risk予測:
  de-risk gateへ

return予測:
  overweight gateへ

current position / underweight duration:
  recovery gateへ

latent_zh:
  全体へ
```

合格条件：

```text id="2sndta"
de-risk後にrecoveryが発生する
recovery latencyが短くなる
transition matrixで underweight -> benchmark が増える
long/short/flatの一極集中が消える
```

これは一番本命。
まったく、今の設計は「逃げる」「戻る」「強く張る」を同じ回帰headでやらせてるのが雑すぎるのよ。

---

**Hypothesis 3: TD3+BC / CQL系の conservative critic を軽く足すと、overweight/underweightの過信を抑えられる。**

TD3+BCは、offline RLでOOD actionの価値推定ミスを抑えるために、policy updateへBC termを足すシンプルな手法。実装が軽いのでUniDreamのAC段階に入れやすい。([ResearchGate][3])
CQLは、offline RLの分布シフトやQ過大評価に対して、未知/OOD actionのQを保守的に下げる方法。long-only overweight実験で `long=99%` に倒れた状況にはかなり関係ある。([ScienceStack][4])

ただし、今すぐACを回すのは違う。
まず routing BC が最低限まともになってから。

使うならこう。

```text id="v4wl71"
actor_loss =
  - λ * Q(s, π(s))
  + α * BC_to_valid_transition
  + β * leverage/DD/turnover penalty
```

critic側は、候補actionを離散化して、

```text id="g5cz9t"
a ∈ {0.0, 0.5, 1.0, 1.25}
```

に限定して conservative に学習する。
連続actionで自由に探索させるとまたcollapseしやすい。

合格条件：

```text id="i9hfdk"
overweight使用率が0%でも99%でもない
overweight中のforward net excessが正
maxdd_deltaが悪化しない
critic Qが特定actionだけ異常に高くならない
```

---

**Hypothesis 4: 金融ML的には「returnを直接当てる」より、rank/decile・risk regime・action価値を予測する方が安定する。**

Gu, Kelly, Xiu は、asset pricing を risk premium prediction として扱い、機械学習の価値は非線形な predictor interaction を拾う点にある、としている。支配的なsignalも momentum/liquidity/volatility 系に寄る。([OUP Academic][5])
だからUniDreamでも、`return_h32` の点予測をActorへ直結するより、

```text id="m24lrk"
この局面で
de-risk transition が上位decileか
recovery transition が上位decileか
overweight transition が上位decileか
```

を予測するほうが筋がいい。

先に actor を学習せず、label quality だけ検証する。

```text id="am3mc5"
predicted de-risk score 上位10%:
  future cost-adjusted de-risk advantage が正か

predicted recovery score 上位10%:
  benchmark復帰後のnet excessが改善するか

predicted overweight score 上位10%:
  maxdd悪化なしでexcessが正か
```

ここでtop decileが効かないなら、Actorを回しても無駄。
逆にtop decileが効くなら、それをrouting headに渡す。

---

## 優先順位

次の順番でいい。

```text id="loovkr"
1. diagnostic追加
   transition matrix
   bucket別PnL
   recovery latency
   action別 forward advantage

2. cost_adjusted_advantage dataset作成
   action別 / transition別 / horizon別

3. routing BC
   neutral / de-risk / recovery / overweight

4. predictive stateをgate限定で使用
   position head直結は禁止

5. routing BCが安定したら conservative AC
   TD3+BC or IQL風 or CQL-lite
```

## まず作るべき実験

### Experiment G: transition advantage label probe

学習なし。集計だけ。

```text id="j3scmj"
candidate actions:
  0.0, 0.5, 1.0, 1.25

horizons:
  4, 8, 16, 32

metrics:
  transition別 mean advantage
  top decile advantage
  bottom decile advantage
  turnover/cost
  maxDD contribution
```

目的は、**そもそも de-risk / recovery / overweight に事後的な価値がある局面が存在するか**を見ること。

### Experiment H: routing BC without predictive state

まず predictive state なし。

```text id="9to7ip"
input:
  latent_zh + current_position + inventory_age

target:
  best transition class

loss:
  weighted CE with cost_adjusted_advantage
```

これでcollapseが減らないなら、予測特徴以前にrouting/labelが悪い。

### Experiment I: routing BC with predictive gate

次に predictive state を gate にだけ入れる。

```text id="px9fz8"
risk preds -> de-risk gate
return preds -> overweight gate
position state -> recovery gate
```

比較対象は H。
HよりDD/alpha/recoveryが改善するかを見る。

### Experiment J: conservative AC after routing BC

H/IでまともなBCが出た場合だけ。

```text id="bn3r91"
TD3+BC-lite:
  actor = Q改善 + BC anchor

CQL-lite:
  OOD/未選択actionのQを下げる

IQL-lite:
  advantage-weighted extraction
```

## やらない方がいいこと

```text id="z8aq6n"
predictive state direct concatの継続
scale係数ガチャ
full short解禁
PatchTST-style encoderへの即移行
壊れたBCからのACチューニング
```

今の段階でPatchTSTに行くのは早い。
問題は「情報がない」じゃなくて、**情報を行動価値へ変換する目的関数がない**ことだから。

## 最短の次アクション

これをCodex/GLMに投げるなら、指示はこれでいい。

```text id="58slh1"
次は model architecture 変更ではなく、transition-level cost_adjusted_advantage の検証を行う。

1. 各時刻tで候補action {0.0,0.5,1.0,1.25} を評価する。
2. horizon {4,8,16,32} の realized future return から、transaction cost / turnover / drawdown / volatility / leverage penalty を差し引いた advantage を作る。
3. transition を neutral / de-risk / recovery / overweight に分類する。
4. top-decile advantage, bucket別PnL, transition matrix, recovery latency をreportに出す。
5. fold4で有効なtransition価値が確認できたら、routing BCを実装する。
```

これが一番まとも。
**次の勝ち筋は「もっと強いモデル」じゃなくて、「何を真似るべきかをaction単位で定義し直す」こと。**

[1]: https://huggingface.co/papers/2110.06169?utm_source=chatgpt.com "Paper page - Offline Reinforcement Learning with Implicit Q-Learning"
[2]: https://www.sciencestack.ai/paper/2006.09359?utm_source=chatgpt.com "AWAC: Accelerating Online Reinforcement Learning with Offline Datasets (arXiv:2006.09359v6) - ScienceStack"
[3]: https://www.researchgate.net/publication/352396570_A_Minimalist_Approach_to_Offline_Reinforcement_Learning?utm_source=chatgpt.com "(PDF) A Minimalist Approach to Offline Reinforcement Learning"
[4]: https://www.sciencestack.ai/paper/2006.04779v3?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning (arXiv:2006.04779v3) - ScienceStack"
[5]: https://academic.oup.com/rfs/article/33/5/2223/5758276?utm_source=chatgpt.com "Empirical Asset Pricing via Machine Learning | The Review of Financial Studies | Oxford Academic"
