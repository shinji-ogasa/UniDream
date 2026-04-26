# AC Plan 1 Candidate State-Action Critic Results

作成日: 2026-04-26
対象: `documents/acplan_1.md`
実行範囲: BTCUSDT 15m / seed 7 / fold 4 / 2018-01-01 to 2024-01-01
基準: `configs/bcplan5_phase8_state_machine_s007.yaml`
文字コード: UTF-8

## 結論

ACはまだactor更新へ進めない。

`Q(s, action_candidate)` のprobeは実装できたが、fold4 testで合格条件を満たさなかった。

```text
必要条件:
  Q rank IC / row Spearman > 0
  Q-selected action の realized advantage > 0
  Q-selected action が short / long / flat 一極集中しない

実測:
  row Spearman は一部で正
  しかし Q-selected realized advantage は 0 近辺または負
  rank系は short 100% など片側collapse
```

そのため、今回本流に採用するのは「state-action critic probe / candidate Q diagnostic」まで。AC actor更新、residual adapter学習、route unlockはまだ採用しない。

## Web確認に基づく仮説

実装後の結果を見て、offline RLの既存手法を再確認した。

- TD3+BCは、offline actor更新にBC項を足してdataset actionから外れすぎないようにする方針。UniDreamでもactorをいきなり動かさず、`a_BC` anchorからの小さい候補比較にした。Source: https://arxiv.org/abs/2106.06860
- CQLは、分布外actionの価値過大評価を抑えるために保守的なQを学習する。今回のCQL-liteは過大評価抑制には合うが、実測では改善actionまで押し下げる傾向があった。Source: https://arxiv.org/abs/2006.04779 / https://papers.nips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html
- IQLは未観測actionを直接評価しすぎず、最後はadvantage-weighted BCでpolicy extractionする。今回の結果は、UniDreamでも「Q argmaxでactorを動かす」より「advantageが明確なサンプルだけ重く学ぶ」方が安全という方向を支持する。Source: https://arxiv.org/abs/2110.06169

## 実装したもの

### State-action critic module

追加:

```text
unidream/actor_critic/state_action_critic.py
```

内容:

```text
CandidateQNet
train_candidate_q_ensemble
predict_candidate_q
evaluate_candidate_q
CQL-lite penalty
anchor-advantage target
listwise rank CE
margin-gated rank target
dynamic residual candidate support
```

### Probe CLI

追加:

```text
unidream/cli/ac_candidate_q_probe.py
```

できること:

```text
actor更新なし
Phase 8 BC checkpointを読み込む
通常pipelineと同じval selectorを通す
candidate Q(s,a)をtrainで学習
train/val/testでrankingとrealized advantageを評価
fixed candidates / dynamic residual candidatesを比較
```

候補mode:

```text
fixed:
  0.0 / 0.5 / 1.0 / 1.05 / 1.10 / 1.25

dynamic:
  bc
  bc_minus_0.05
  bc_plus_0.05
  hold_current
  benchmark
  ow_1.05
  ow_1.10
  ow_1.25
```

## 実験ループ

### Loop 1: fixed candidate Q

出力:

```text
documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4.md
documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v2.md
```

初回はval selectorを通していなかったため、Phase 8 test基準よりturnoverが大きく出た。probe CLIを修正し、通常pipelineと同じ`val_adjust_rate_scale_grid`を通して再実行した。

結果:

```text
mse_ensemble_mean test row Spearman: 0.2412
selected adv vs anchor: -0.000215
selected distribution: flat 100%

cql_lite_minq test row Spearman: 0.2608
selected adv vs anchor: -0.000050
selected distribution: flat 100%
```

読み取り:

```text
順位相関は少しある
ただしargmaxにするとbenchmark/flatへ張り付く
CQL-liteはさらに保守化するが、改善actionを選べていない
```

### Loop 2: anchor advantage target

出力:

```text
documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v3_anchor.md
```

絶対valueではなく、`Q(s,a) - Q(s,a_BC)`をtargetにした。

結果:

```text
anchor_adv_mse_mean test row Spearman: 0.2375
selected adv vs anchor: -0.000243
selected distribution: flat 100%

anchor_adv_cql_minq test row Spearman: 0.2439
selected adv vs anchor: -0.000204
selected distribution: flat 100%
```

読み取り:

```text
target中心化だけでは不十分
Qが改善差を安全に拾えていない
```

### Loop 3: listwise rank CE

出力:

```text
documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v4_rank.md
```

候補内の順位を直接学習するrank CEを追加した。

結果:

```text
anchor_rank_ce_mean test row Spearman: 0.1289
selected adv vs anchor: -0.001286
selected distribution: long 37.1% / flat 62.2%

anchor_rank_ce_cql_minq test row Spearman: 0.1719
selected adv vs anchor: -0.000767
selected distribution: long 11.9% / flat 87.3%
```

読み取り:

```text
active候補は出る
しかし実現advantageは悪化
過去のAC microと同じくoverweight/active側へ反転する危険がある
```

### Loop 4: dynamic residual candidates

出力:

```text
documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v5_dynamic.md
```

AC plan通り、固定positionではなく`a_BC`周辺の残差候補に変えた。

Phase 8 baselineは通常test基準に揃った。

```text
test AlphaEx: +0.89 pt/yr
test SharpeDelta: -0.011
test MaxDDDelta: -1.58 pt
test short: 16.5%
test flat: 83.5%
test turnover: 2.65
```

結果:

```text
mse_ensemble_mean test row Spearman: 0.2364
selected adv vs anchor: +0.000000
selected distribution: flat 100%
best possible adv vs anchor: +0.001479

anchor_adv_mse_mean test row Spearman: 0.2364
selected adv vs anchor: +0.000000
selected distribution: flat 100%
best possible adv vs anchor: +0.001479

anchor_rank_ce_mean test row Spearman: 0.0499
selected adv vs anchor: -0.000185
selected distribution: short 100%
```

読み取り:

```text
実現値ベースでは改善余地がある
best actionは主に bc_minus_0.05 に出る
しかしQはそれを選べず、MSE/CQLはanchor維持、rank CEはshort 100%へ崩れる
```

### Loop 5: margin-gated rank target

出力:

```text
documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v6_margin_rank.md
```

rank CEが全局面でshortへ寄るため、best advantageが一定以上ある時だけanchorから動くtargetを試した。

結果:

```text
anchor_margin_rank_m010_mean test row Spearman: 0.1402
selected adv vs anchor: -0.000185
selected distribution: short 100%

anchor_margin_rank_m025_mean test row Spearman: 0.2434
selected adv vs anchor: +0.000001
selected distribution: flat 100%
```

読み取り:

```text
marginを弱くするとshort collapse
marginを強くするとanchor/flat維持
中間に安定改善点は見つからなかった
```

## 総合結果

| 実験 | test row Spearman | selected adv vs anchor | selected dist | 判定 |
|---|---:|---:|---|---|
| fixed mse | 0.2412 | -0.000215 | flat 100% | 不可 |
| fixed CQL-lite | 0.2608 | -0.000050 | flat 100% | 不可 |
| fixed anchor adv | 0.2375 | -0.000243 | flat 100% | 不可 |
| fixed rank CE | 0.1289 | -0.001286 | long 37.1% | 不可 |
| dynamic mse | 0.2364 | +0.000000 | flat 100% | 安全だが改善なし |
| dynamic CQL-lite | 0.2364 | +0.000000 | flat 100% | 安全だが改善なし |
| dynamic rank CE | 0.0499 | -0.000185 | short 100% | collapse |
| dynamic margin rank 0.10 | 0.1402 | -0.000185 | short 100% | collapse |
| dynamic margin rank 0.25 | 0.2434 | +0.000001 | flat 100% | 安全だが改善なし |

## 採用判定

### 採用

```text
state-action critic probe
candidate Q diagnostic CLI
dynamic residual candidate evaluation
CQL-lite / rank CE / margin rank を実験可能なdiagnosticとして保持
```

理由:

```text
AC前の安全診断として有効
V-only criticでは見えなかった action別ランキングを測れる
actor更新禁止の根拠を数値化できる
```

### 非採用

```text
AC actor update
true residual adapter training
route unlock
rank CE policy extraction
CQL-lite argmax policy
full actor AC
```

理由:

```text
Q-selected realized advantageが正にならない
rank系はshort/long collapseを再発
MSE/CQL系はanchor維持で改善しない
```

## 次の仮説

今回の結果から、AC制御より前に以下が問題。

```text
1. Qは候補の順位を少し見ているが、top actionの校正が悪い
2. dynamic候補のbest possible advはあるが、特徴から選別できていない
3. rank lossは分布制御なしだとbest頻度の高いbc_minusへ一極化する
4. Q argmaxはまだ危険。IQL/AWR的なadvantage-weighted extractionの方が合う
```

次に進むなら、actor更新ではなく以下。

```text
A. residual candidateのbest labelを直接BC補助lossに入れる
B. Qではなくrealized advantageでweighted residual BCを作る
C. Q-selectedではなく、Q margin + realized support + regime filterを満たすサンプルだけ使う
D. fold0/fold5へprobeを広げ、Q rankがfold固有でないか確認する
```

## 最終判断

AC Plan 1のAC-A/Bは実装・検証完了。ただし合格しない。

```text
Phase8 BCは維持
AC actor更新は停止
本流にはcandidate-Q診断だけ採用
次はQ argmax型ACではなく、advantage-weighted residual BCかfold拡張probeを優先
```

## 生成物

- `unidream/actor_critic/state_action_critic.py`
- `unidream/cli/ac_candidate_q_probe.py`
- `documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4.md`
- `documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v2.md`
- `documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v3_anchor.md`
- `documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v4_rank.md`
- `documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v5_dynamic.md`
- `documents/ac_candidate_q/20260426_acplan1_candidate_q_probe_fold4_v6_margin_rank.md`
- `documents/logs/20260426_acplan1_candidate_q_probe_fold4*.log`
- `documents/20260426_acplan1_candidate_q_results.md`
