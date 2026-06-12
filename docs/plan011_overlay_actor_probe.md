# Plan011 benchmark-relative overlay actor probe

Plan010 は失敗ログとして固定する。Plan011 は route class (`de_risk/neutral/overweight`) を捨て、
B&H exposure `1.0` を基準にした small continuous overlay actor として分離する。

## 方針

- Actor action は absolute position ではなく `overlay = position - 1.0` を回帰する。
- route head / route label / inventory recovery class は使わない。
- Transformer WM の predictive risk state は Actor 入力として使う。
- 教師は `1.0 + small_overlay` に変換し、BC/AC の oracle anchor を同じ教師に揃える。
- 小さい overlay では `SmoothL1(beta=1)` の勾配が弱すぎるため、Plan011 では residual target loss scale を追加した。

## 実装

- `unidream/experiments/overlay_teacher.py`
  - `benchmark_overlay_teacher`
  - asymmetric up/down scale
  - WM-risk-gated downside overlay
  - WM-risk-budget teacher
  - EMA / hold-band / max-step smoothing
- `unidream/experiments/bc_stage.py`
  - BC teacher positions を Plan011 overlay teacher へ変換
- `unidream/experiments/ac_stage.py`
  - AC oracle anchor も同じ overlay teacher へ変換
- `unidream/actor_critic/bc_pretrain.py`
  - `residual_target_loss_scale` を追加

## fold4 probe

共通条件:

```bash
uv run python -u -m unidream.cli.train \
  --config configs/plan011_overlay_actor_bconly.yaml \
  --start 2018-01-01 --end 2024-01-01 \
  --folds 4 --seed 7 --device mps \
  --start-from bc --stop-after test
```

WM は `checkpoints/plan010_risk_focus_raw_wm_s007/fold_4/world_model.pt` を Plan011 checkpoint dir にコピーして使用した。

| run | teacher | AC | val summary | fold4 test AlphaEx | fold4 test MaxDDDelta | turnover | 所感 |
|---|---|---:|---|---:|---:|---:|---|
| v0 | oracle scaled 0.35, small range | yes | BC flat 100% | +0.38pt | -0.63pt | 0.37 | 壊れないが弱すぎる |
| v2 | oracle scaled 0.75 + scaled BC loss | aborted | short 57%, alpha collapse | - | - | - | BC loss scale は効くが過剰 de-risk |
| v3 BC-only | asymmetric oracle scale | no | val Alpha +20pt | -0.23pt | +0.63pt | 1.69 | val/testズレ、testでDD悪化 |
| v4 BC-only | downside WM risk gate | no | val Alpha -33pt / DD -0.63pt | +0.44pt | -0.88pt | 1.27 | 現時点のbest。弱いが方向は壊れない |
| v5 BC-only | extreme risk gate | no | val DD改善、alpha負 | +0.04pt | -0.50pt | 2.57 | active化したがalphaを削る |
| v6 BC-only | WM risk-budget teacher | no | val Alpha +35pt / DD -0.85pt / TO 45.9 | -1.02pt | +1.00pt | 46.12 | WM riskに反応するが高回転で崩壊 |
| v8 BC-only | v6 + teacher smoothing | no | val Alpha +10pt / DD -0.77pt / TO 1.5 | +0.07pt | -0.53pt | 1.33 | turnoverは制御、test edgeは消える |
| v8 AC | v8 + restore-best AC | yes | ACはvalを壊してBC復元 | +0.07pt | -0.53pt | 1.33 | ACはまだde-risk側に崩す |
| v9 BC-only | v6 + execution slow step | no | val Alpha +60pt / TO 16.4 | -0.20pt | -0.21pt | 16.71 | executionだけ遅くしてもcost過多 |
| v10 BC-only | extreme crash gate | no | val/test高回転 | -2.16pt | +3.37pt | 88.65 | rare gateでもtargetが高周波化して崩壊 |

## 現時点の結論

Plan011 の変更自体は route collapse を回避できた。特に `residual_target_loss_scale` を入れると小さい overlay target をActorが学べる。

ただし fold4 test ではまだ `AlphaEx >= +3pt / MaxDDDelta <= -3pt` に遠い。最良は v4 BC-only の `+0.44pt / -0.88pt`。

観察:

- WM risk signal はActorに伝わる。v6では val で大きく反応した。
- そのまま使うと高回転になり、testでは cost と timing mismatch で崩れる。
- smoothing を強めると turnover は落ちるが、test edge も消える。
- AC は restore-best を入れても改善せず、BCからde-risk方向へ動いてvalを壊す。

次にスケールする価値があるのは、単純な係数拡大ではなく以下。

1. WM risk state を低周波 regime / budget state に圧縮してからActorへ渡す。
2. Actor targetを bar-by-bar risk ではなく、30-100 bar程度の保持前提 overlay path にする。
3. AC rewardに `upside_miss_coef` と turnover/hold制約を入れ、de-risk collapseを明示的に罰する。
4. fold3/4/5で v4 と低周波risk-budget teacher を比較して、fold4専用でないか確認する。

## v4 fold3/4/5 check

v4 は fold4 では最良だったが、fold3/5へ展開すると alpha を大きく壊した。

```bash
uv run python -u -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v4_bconly.yaml \
  --start 2018-01-01 --end 2024-01-01 \
  --folds 3,5 --seed 7 --device mps \
  --start-from bc --stop-after test
```

| fold | AlphaEx | MaxDDDelta | turnover | 所感 |
|---:|---:|---:|---:|---|
| 3 | -47.49pt | -0.95pt | 1.35 | DDは少し改善するがbull局面のunderexposureでalpha破壊 |
| 4 | +0.44pt | -0.88pt | 1.27 | v4 best foldだが目標には遠い |
| 5 | -96.67pt | -0.28pt | 1.79 | 強い上昇局面でsmall underweightでもalphaを大きく失う |

この結果により、v4のfold4改善は採用できる汎化傾向ではない。次は「riskを読んだら一律にunderweight」ではなく、
上昇局面のupside missを目的関数と教師の両方で抑える必要がある。

## 2026-06-05 Plan011 benchmark-relative overlay 追加検証

### 低頻度WM overlay postprocess

`unidream.cli.plan011_lowfreq_overlay_probe` を追加し、WM predictive risk state から低頻度の B&H 相対 overlay を直接生成して、val 選択 -> test 評価を切り分けた。

```bash
.venv/bin/python -u -m unidream.cli.plan011_lowfreq_overlay_probe \
  --config configs/plan011_overlay_actor_v4_bconly.yaml \
  --checkpoint-dir checkpoints/plan010_risk_focus_raw_wm_s007 \
  --folds 3,4,5 --seed 7 --device cpu \
  --output codex_outputs/plan011_lowfreq_overlay_f345
```

| probe | fold | test AlphaEx | test MaxDDDelta | turnover | 所感 |
|---|---:|---:|---:|---:|---|
| risk mean lowfreq | 3 | +28.12pt | +0.35pt | 1.32 | alphaは戻るがDD改善なし |
| risk mean lowfreq | 4 | -0.01pt | -0.09pt | 3.62 | ほぼB&H相当 |
| risk mean lowfreq | 5 | -15.63pt | -0.06pt | 0.51 | alphaを壊す |

robust selection（val前半/後半 + no-op候補）も追加したが、fold3/4ではval上の強い候補がtestへ乗らず、fold5は弱いval候補がtestで崩れた。平均riskをそのまま使う設計はまだ採用不可。

### v11 lowfreq alpha-preserving teacher

低頻度WM overlayをBC teacherに入れ、Actorに実モデル推論として学習させた。

```bash
.venv/bin/python -u -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v11_lowfreq_bconly.yaml \
  --start 2018-01-01 --end 2024-01-01 \
  --folds 3,4,5 --seed 7 --device cpu \
  --start-from bc --stop-after test
```

| fold | AlphaEx | MaxDDDelta | turnover | 所感 |
|---:|---:|---:|---:|---|
| 3 | +20.73pt | +0.18pt | 0.32 | alpha復帰。ただしDD改善なし |
| 4 | -0.30pt | +0.46pt | 0.37 | ほぼ中立だがDD悪化 |
| 5 | +38.27pt | +0.41pt | 0.46 | alpha復帰。ただしDD悪化 |

v11は「route class + AC がalphaを壊す」問題を避け、実モデル推論でalphaを残す方向には進んだ。一方、teacherがoverweight寄りになり、MaxDDDeltaの改善には効いていない。

### v12 downside-only teacher

upside overlayを切り、negative trend下のde-riskだけに寄せた。

| fold | AlphaEx | MaxDDDelta | turnover | 所感 |
|---:|---:|---:|---:|---|
| 3 | -26.12pt | -0.58pt | 0.58 | DDは改善するがalpha破壊 |
| 4 | +0.32pt | -0.75pt | 0.66 | DD改善、alphaは弱い |
| 5 | -102.74pt | -0.23pt | 0.95 | bull局面のunderexposureで崩壊 |

v12はDD方向には動くが、目的の `AlphaEx >= +3pt` と両立しない。特にfold5でupside missが致命的。

### WM predictive feature診断

`unidream.cli.plan011_predictive_feature_probe` を追加し、WM predictive aux featureごとの future return / future drawdown rank を確認した。

```bash
.venv/bin/python -u -m unidream.cli.plan011_predictive_feature_probe \
  --config configs/plan011_overlay_actor_v11_lowfreq_bconly.yaml \
  --checkpoint-dir checkpoints/plan010_risk_focus_raw_wm_s007 \
  --folds 3,4,5 --seed 7 --device cpu \
  --output codex_outputs/plan011_predictive_feature_f345
```

観察:

- fold3: 高vol/DD signalは future DD と正相関、future return と弱い負相関。de-riskに使える。
- fold4: 高vol/DD signalは future DD と正相関だが、future return とも正相関。単純de-riskはupside missになりやすい。
- fold5: 高vol/DD signalの future DD IC がtestでほぼ消え、future returnとは正相関。risk signalを平均で使うとalphaを壊しやすい。

`wm_pred_drawdown_excess_h64` 単独のquick lowfreq probeも実施したが、robust selectionではfold3/4/5すべてno-opを選択した。現時点では平均riskより安全だが、採用できるedgeは出ていない。

### v13 alpha-protected AC

v11 teacherから短いACを入れ、`upside_miss_coef` / turnover / DD budget を強めた。

| fold | 最終挙動 | test AlphaEx | test MaxDDDelta | 所感 |
|---:|---|---:|---:|---|
| 3 | restore-bestでBC復元 | +20.73pt | +0.18pt | AC checkpointはBCよりval悪化 |
| 4 | restore-bestでBC復元 | -0.30pt | +0.46pt | ACはBCを超えない |
| 5 | restore-bestでBC復元 | +38.27pt | +0.41pt | BC valが弱いfoldでもAC改善なし |

現時点の判断:

- TransformerWMのrisk signalは残っているが、foldによって「risk高 = return悪化」ではない。
- そのため risk budget allocator は、risk単独ではなく `risk高 && return/trend悪化` の合成状態を学習する必要がある。
- v11は実モデル推論でalphaを残す方向として有望。ただしMaxDDDelta改善は未達。
- v12はDD改善方向の上限確認として有用だが、alpha破壊が大きく、そのままスケール不可。
- ACは現状のimagination rewardではBCを改善できていない。次はACを強くする前に、teacher/selectorを「upside-protected downside overlay」に作り直す方が筋が良い。

## 2026-06-05 追加: causal regime gate / safe-base / AC risk-budget 再検証

### v14-v18: base/downside tradeoff

| run | 概要 | fold3 | fold4 | fold5 | 判断 |
|---|---|---:|---:|---:|---|
| v14 edge WM | edge条件つき低頻度teacher | +25.06 / +0.37 | -0.24 / +0.40 | +49.87 / +0.29 | alphaは出るがDD悪化。実質小さいoverweight |
| v15 base downside | base +0.015, down 0.06 | +9.42 / +0.01 | -0.10 / +0.10 | +5.14 / +0.25 | 現時点の安定寄り。Aggregate +4.82 / +0.12 |
| v16 down強化 | base +0.015, down 0.08 | -6.01 / -0.26 | +0.02 / -0.20 | -47.25 / +0.16 | DDは動くがalpha崩壊 |
| v17 base強化 | base +0.020, down 0.08 | +3.78 / -0.14 | -0.07 / -0.04 | -28.05 / +0.26 | fold3は近いがfold5崩壊 |
| v18 base過多 | base +0.030, down 0.08 | +26.25 / +0.19 | -0.29 / +0.39 | +27.86 / +0.50 | alpha寄りに戻りDD悪化 |

v15-v18で、base overweightはAlphaExを戻すがMaxDDDeltaを悪化させ、downsideを強めるとfold5のupside missで崩れることが確認できた。

### v19-v22: fold4 deep de-risk

| run | fold4 test AlphaEx | fold4 test MaxDDDelta | val挙動 | 判断 |
|---|---:|---:|---|---|
| v19 | +0.36 | -0.98 | 弱い改善 | DD方向は出るが不足 |
| v20 | +1.18 | -2.94 | まだ不足 | 目標近辺までDDは出る |
| v21 | +3.20 | -6.68 | val Alpha -209.71 / DD -5.52 | testだけなら達成。ただしvalで採用不能 |
| v22 | +1.75 | -3.91 | val Alpha -110.05 / DD -3.08 | trendを厳しくしてもval崩壊 |

v21はfold4 testだけ見ると `AlphaEx >= +3pt / MaxDDDelta <= -3pt` を満たしたが、validationが壊れているためリークなしの採用候補ではない。

同じv21 deep de-riskをfold3/5へ展開すると以下。

| fold | AlphaEx | MaxDDDelta | turnover | 判断 |
|---:|---:|---:|---:|---|
| 3 | -135.79 | -4.67 | 6.64 | DDは出るがalpha崩壊 |
| 5 | -583.31 | -2.01 | 4.10 | alpha崩壊、DDも目標未達 |

結論: `強いde-risk` はDDを作れるが、fold間でそのまま汎化しない。fold4単体の成功を採用するとリークになる。

### causal trailing drawdown gate

`overlay_teacher.py` と `plan011_lowfreq_overlay_probe.py` に、過去リターンだけで作る causal trailing drawdown depth gate を追加した。

追加パラメータ:

- `benchmark_overlay_lowfreq_dd_lookback`
- `benchmark_overlay_lowfreq_dd_min`
- `benchmark_overlay_lowfreq_dd_max`
- `benchmark_overlay_lowfreq_base_mode: safe`

probe結果:

| probe | fold | val AlphaEx / MaxDDDelta | test AlphaEx / MaxDDDelta | 判断 |
|---|---:|---:|---:|---|
| ddgate quick v2 | 4 | +14.27 / -0.69 | +0.07 / -0.12 | DD方向は出るが弱い |
| ddgate quick | 3 | +1111.68 / -0.37 | +43.38 / +0.70 | val過適合、test DD悪化 |
| ddgate quick | 5 | +3.14 / -4.52 | -32.14 / -0.09 | valで選べてもtest alpha崩壊 |

drawdown gate単体では、valでDDが良い候補を選んでもtestへ安定して乗らない。

### v23-v24: safe-base teacher

v23はv17にDD gateを追加、v24はbase overlayを常時ではなくsafe局面だけに制限した。

| run | fold3 | fold4 | fold5 | Aggregate | 判断 |
|---|---:|---:|---:|---:|---|
| v23 config path | +27.30 / +0.48 | -0.49 / +0.90 | +71.31 / +0.41 | - | de-riskが絞られすぎてbase overweight化 |
| v24 BC | +5.69 / -0.00 | -0.12 / +0.18 | +14.92 / +0.20 | +6.83 / +0.13 | alphaは残るがDD未達 |

safe-baseはv15よりalphaを上げたが、MaxDDDeltaは改善しない。

### v25-v26: AC risk-budget再調整

v25はv24を起点に、DD/tail/downside hedge/upside missを強めたAC。v26はrisk_state exposure / risk_tiltをさらに強めたfold4短縮診断。

| run | fold | AlphaEx | MaxDDDelta | 判断 |
|---|---:|---:|---:|---|
| v25 AC | 3 | +15.92 | +0.18 | val改善はtestでoverweight化 |
| v25 AC | 4 | -0.12 | +0.17 | restore-bestでBC相当 |
| v25 AC | 5 | +16.22 | +0.21 | restore-bestでBC相当 |
| v26 AC | 4 | -0.31 | +0.46 | risk項強化でもtest悪化 |

ACは現状、WM risk signalを十分なde-risk行動へ変換できていない。reward係数だけを強めるとval上は改善してもtestでは小さいoverweight/near-B&Hへ戻る。

## 現時点の到達点と次の打ち手

到達点:

- `route class + AC` を捨てた benchmark-relative continuous overlay actor は、AlphaExを壊しにくくなった。
- `v15/v24` 系で実モデル推論のAlphaExは戻る。
- `v21` 系でMaxDDDelta改善能力は存在する。
- ただし両者をリークなしに選ぶselectorがまだない。

次にやるべきこと:

1. per-fold val winnerではなく、複数fold validationで共有specを選ぶ global selector を追加する。
2. selectorの目的を `AlphaEx floor + MaxDDDelta negative + turnover cap` に固定し、固定overweight候補を落とす。
3. strong de-riskを直接使うのではなく、`safe-base / mild-de-risk / deep-de-risk` の3候補を状態別に切り替える小さい gating model を作る。
4. gateの入力はWM riskだけでなく、causal trend、causal drawdown depth、realized vol、risk featureのrank/quantileに限定する。
5. まずfold3/4/5で、global validation selector -> testの順に評価する。ここでfold4だけのtest成功を拾わないことを採用条件にする。

## 2026-06-06 追加: global selector / deep DD guard

### global validation selector

`plan011_lowfreq_overlay_probe.py` に `--global-select` を追加した。これはper-foldのbest specを選ばず、同一specをfold3/4/5のvalidation合算で選び、そのspecだけをtestへ出す。

```bash
.venv/bin/python -u -m unidream.cli.plan011_lowfreq_overlay_probe \
  --config configs/plan011_overlay_actor_v15_base_downside_bconly.yaml \
  --checkpoint-dir checkpoints/plan010_risk_focus_raw_wm_s007 \
  --folds 3,4,5 --seed 7 --device cpu \
  --quick-grid --global-select \
  --output codex_outputs/plan011_lowfreq_global_ddgate_quick_f345
```

結果:

- global selector は no-op を選択した。
- selected: `base_overlay=0.0`, `down_overlay=0.0`, `up_overlay=0.0`
- testは全fold `0.00 / 0.00`

判断:

- 現在のlowfreq grid内には、fold3/4/5のvalidationを同時に満たす非自明なoverlay specがない。
- per-fold val selectionではDD候補が選ばれることはあるが、fold5などでtest alphaが崩れる。
- リークなし採用条件を厳密にすると、現gridはno-opが最良になる。

### v27 deep DD guard

v21 deep de-riskに、過去128本のcausal drawdown depth `>= 0.02` を追加した。

teacher path:

| fold | AlphaEx | MaxDDDelta | 判断 |
|---:|---:|---:|---|
| 3 | +3.00 | +0.15 | alphaは残るがDD悪化 |
| 4 | +0.43 | -0.57 | DD方向は出るが弱い |
| 5 | +49.80 | +0.40 | alphaは残るがDD悪化 |

BC実モデル:

| fold | AlphaEx | MaxDDDelta | turnover | 判断 |
|---:|---:|---:|---:|---|
| 3 | -163.14 | -4.81 | 4.42 | DDは出るがalpha崩壊 |
| 4 | +2.61 | -5.46 | 3.68 | かなり近いがAlphaEx +3ptに未達 |
| 5 | -582.98 | -2.15 | 3.71 | alpha崩壊、DDも未達 |

v27はfold4では目標近辺まで来るが、fold3/5では採用不能。Actorがteacherよりdeep de-riskを増幅しやすく、bull foldで大きくalphaを壊す。

### v28 long-trend deep guard

v27に長期trend条件を追加した。

- `benchmark_overlay_lowfreq_trend_lookback: 512`
- `benchmark_overlay_lowfreq_trend_max: -0.05`

teacher path:

| fold | AlphaEx | MaxDDDelta | 判断 |
|---:|---:|---:|---|
| 3 | +6.77 | +0.18 | alphaは残るがDD悪化 |
| 4 | +0.51 | -0.70 | DD方向は弱い |
| 5 | +61.14 | +0.41 | alphaは残るがDD悪化 |

long trendでbull foldのalpha破壊は抑えられるが、fold4のDD改善も弱くなる。単純な長期trend gateでは目標に届かない。

## 追加後の判断

現時点で見えている境界:

- mild/base系: `v15/v24/v28` はAlphaExを残すがMaxDDDeltaが不足。
- deep de-risk系: `v21/v27` はMaxDDDeltaを作れるがfold3/5でAlphaExを破壊。
- AC係数強化: `v25/v26` はval改善してもtestでは小さいoverweight/near-B&Hへ戻り、DD改善へ変換できない。
- global validation selector: 非自明specを選べずno-op。

次の実装候補:

1. hand-designed threshold gridではなく、`mild/base/deep` の3候補を選ぶ小さいgating modelを作る。
2. gateの教師は「未来リターン」ではなく、`deep候補がB&HよりDDを改善し、かつalphaを壊さない局面` のmeta-labelにする。
3. validationはper-foldではなく、fold集合のglobal selectorで採用する。
4. gate出力は連続overlayではなく、候補mixing weightにして、Actorがdeep de-riskを勝手に常時化するのを防ぐ。

## v14 edge/utility WM + benchmark-relative overlay

v13までの問題は、WM riskだけを読むとbull局面でもunderweightし、Alphaを壊すことだった。v14ではPlan011のまま route class は戻さず、WM補助headに `return` / `position_utility` / `overweight_advantage` / `recovery` を追加した。

追加実装:

- `configs/plan011_overlay_actor_v14_edgewm_bconly.yaml`
  - predictive state: `return, vol, drawdown, crash, drawdown_excess, position_utility, overweight_advantage, recovery`
  - WM: `return_scale=1.0`, `position_utility_scale=1.5`, `overweight_advantage_scale=1.0`, `recovery_scale=0.5`
- `unidream/experiments/overlay_teacher.py`
  - `benchmark_overlay_edge_protect_indices`
  - edge/overweight advantageが強いとき、negative overlayを薄めてupside missを抑える
- `unidream.cli.plan011_lowfreq_overlay_probe`
  - risk + edge + trend + trailing DD guard の低頻度overlay probe
  - val前半/後半を使うrobust selection

### v14 BC actor fold4

```bash
uv run python -u -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v14_edgewm_bconly.yaml \
  --start 2018-01-01 --end 2024-01-01 \
  --folds 4 --seed 7 --device mps \
  --stop-after test
```

| fold | test AlphaEx | test MaxDDDelta | turnover | 所感 |
|---:|---:|---:|---:|---|
| 4 | -0.09pt | +0.04pt | 2.12 | edge-protectでAlpha破壊は抑えたが、DD改善も消えた |

### v14 WM lowfreq direct probe

Actor蒸留前に、v14 WM predictive stateから低頻度overlayを直接生成して切り分けた。

| probe | fold | val AlphaEx | val MaxDDDelta | test AlphaEx | test MaxDDDelta | turnover | 所感 |
|---|---:|---:|---:|---:|---:|---:|---|
| v14 edge lowfreq full | 4 | +35.75pt | -0.81pt | +0.52pt | -1.12pt | 2.54 | v4 bestより少し前進。ただし+3/-3には遠い |
| v14 edge lowfreq quick | 3 | +1246.26pt | -0.37pt | +43.83pt | +0.55pt | 1.80 | edgeでAlphaは強く戻るがDD悪化 |
| v14 edge lowfreq quick | 5 | +3.04pt | -4.27pt | -41.02pt | -0.07pt | 0.96 | valでは+3/-3相当だがtestでAlpha崩壊 |
| v14 strict robust selector | 3 | +0.00pt | +0.00pt | +0.00pt | +0.00pt | 0.00 | 安全側に倒れすぎてB&H選択 |
| v14 strict robust selector | 4 | +0.00pt | +0.00pt | +0.00pt | +0.00pt | 0.00 | 同上 |
| v14 strict robust selector | 5 | +0.00pt | +0.00pt | +0.00pt | +0.00pt | 0.00 | 同上 |

現時点の判断:

- edge/utility headを入れると、v12のようなAlpha破壊はかなり抑えられる。
- ただしDD改善とAlpha維持を同時に満たすval選択がまだ不安定。
- fold5はvalで `+3.04/-4.27` まで届く候補がある一方、testでは `-41.02/-0.07` に崩れた。これは係数不足ではなく、selectorが局所val regimeへ過適合している兆候。
- strict selectorは過適合を避けるが、全foldでno-opを選ぶため改善ゼロ。次は「全候補を厳しく弾く」のではなく、B&Hを基準にした小さな常時edge overlay + crash/DD guard の2層構成に寄せる。

## 2026-06-06 追加: utility allocator / meta gate / DD utility WM

### position_utility raw allocator

`unidream.cli.plan011_utility_allocator_probe` を追加し、v14 WMの `position_utility` raw headから候補positionを直接選んだ。

| fold | val AlphaEx | val MaxDDDelta | test AlphaEx | test MaxDDDelta | 判断 |
|---:|---:|---:|---:|---:|---|
| 3 | +675.43 | +0.00 | +27.90 | +0.21 | alphaは出るがDD改善なし |
| 4 | +43.75 | +0.00 | -0.27 | +0.15 | testで弱い |
| 5 | +0.07 | -0.15 | +0.85 | +0.00 | DD/alphaとも不足 |

結論: v14のposition_utility headは主にalpha/overweight側へ効き、DD改善allocatorとしては弱い。

### supervised meta gate

`unidream.cli.plan011_meta_gate_probe` を追加し、fold内trainの未来windowからcrash/de-risk meta labelを作り、GradientBoostingClassifierでgateを学習した。入力は現行特徴 + WM predictive state + causal trend/vol/DD。

| fold | val AlphaEx | val MaxDDDelta | test AlphaEx | test MaxDDDelta | 判断 |
|---:|---:|---:|---:|---:|---|
| 3 | +1247.44 | +0.09 | +28.47 | +0.44 | alphaは強いがDD悪化 |
| 4 | +30.60 | +0.44 | -0.38 | +0.71 | DD悪化 |
| 5 | +0.15 | -0.24 | -0.13 | -0.00 | ほぼno-op |

結論: train label型gateも、現特徴では「DDを改善しつつalphaを残す」局面をOOSで切り出せていない。

### v29 DD-improving position utility WM

`position_utility` targetに `position_utility_dd_improve_reward` を追加し、B&HよりDDを改善する候補positionを明示的に報酬化した。

追加実装:

- `unidream/world_model/train_wm.py`
  - `position_utility_dd_improve_reward`
  - `utility += dd_improve_reward * relu(bench_dd - candidate_dd)`
- `configs/plan011_overlay_actor_v29_ddutility_bconly.yaml`
  - `position_utility_positions: [0.50, 0.70, 0.85, 0.94, 1.0, 1.06, 1.12]`
  - `position_utility_horizon: 64`
  - `position_utility_dd_improve_reward: 2.5`
  - `position_utility_rank_scale: 0.5`

v29 WMをfold3/4/5で再学習後、utility allocatorで評価。

| fold | val AlphaEx | val MaxDDDelta | test AlphaEx | test MaxDDDelta | 判断 |
|---:|---:|---:|---:|---:|---|
| 3 | +0.00 | +0.00 | +0.00 | +0.00 | no-op選択 |
| 4 | +0.00 | +0.00 | +0.00 | +0.00 | no-op選択 |
| 5 | +3.82 | -5.53 | -373.19 | -3.50 | valでは目標到達、testでalpha崩壊 |

結論: DD改善報酬を入れるとfold5 valでは `+3/-3` を超える候補が出るが、testへは移らず過剰de-riskになる。現状のposition_utility targetはDD改善を入れるほどalpha崩壊リスクが増える。

### 現時点の更新判断

- WMにrisk/edge/utility信号はある。
- ただし `DD改善が必要な局面` をリークなしにOOSで安定抽出するselector/gateがまだない。
- risk-only/deep de-riskはDDを作れるがalpha崩壊。
- edge/utility/vol-managedはalphaを残せるがDD改善不足。
- DD-improve utility targetはvalでは到達候補を作るがtestで崩れる。

次にスケールするなら、単一fold内val選択ではなく、walk-forward内の複数過去foldを使う meta-validation selector が必要。foldごとの1 valだけでは、DDイベントが少なくselectorが局所regimeに過適合する。

## 2026-06-06 追加: alpha-floor selector / edgefix / vol-target / v30 AC

### alpha-floor selector

`unidream.cli.plan011_alpha_floor_selector_probe` を追加した。既存のlowfreq WM overlay生成を使い、選択だけを `AlphaEx floor` と `MaxDDDelta target` に強く寄せるprobe。

```bash
.venv/bin/python -u -m unidream.cli.plan011_alpha_floor_selector_probe \
  --config configs/plan011_overlay_actor_v14_edgewm_bconly.yaml \
  --checkpoint-dir checkpoints/plan011_overlay_actor_v14_edgewm_bconly_s007 \
  --folds 3,4,5 --seed 7 --device cpu \
  --quick-grid --alpha-floor 3.0 --dd-target -3.0 \
  --output codex_outputs/plan011_alpha_floor_quick_edgewm_f345_v2
```

結果:

| run | fold | per-fold test AlphaEx | per-fold test MaxDDDelta | 判断 |
|---|---:|---:|---:|---|
| v14 alpha-floor | 3 | +37.71 | +0.56 | alphaは出るがDD改善なし |
| v14 alpha-floor | 4 | +2.99 | -4.38 | 目標近いがAlpha +3にわずかに未達 |
| v14 alpha-floor | 5 | -14.88 | -0.19 | alpha崩壊 |
| v29 alpha-floor | 3 | +37.54 | +0.45 | DD改善なし |
| v29 alpha-floor | 4 | +0.06 | -0.43 | DD改善が弱い |
| v29 alpha-floor | 5 | -21.06 | -0.11 | alpha崩壊 |

global共有specではv14/v29ともに `AlphaEx >= +3` と `MaxDDDelta <= -3` を全fold validationで満たす候補は0だった。

### edge index fallback修正

`plan011_lowfreq_overlay_probe.py` と `plan011_alpha_floor_selector_probe.py` が `benchmark_overlay_edge_indices` だけを読んでおり、v14/v29 configの `benchmark_overlay_edge_protect_indices` を使えていなかったため修正した。`overlay_teacher.py` のlowfreq branchにも同じfallbackを追加した。

edgefix後:

| run | fold | per-fold test AlphaEx | per-fold test MaxDDDelta | 判断 |
|---|---:|---:|---:|---|
| v14 edgefix | 3 | +35.61 | +0.53 | alpha維持、DD悪化 |
| v14 edgefix | 4 | +2.46 | -3.71 | DDは到達、alpha不足 |
| v14 edgefix | 5 | -12.82 | -0.21 | alpha崩壊 |
| v29 edgefix | 3 | +49.84 | +0.44 | alpha維持、DD悪化 |
| v29 edgefix | 4 | +0.34 | -0.86 | DD不足 |
| v29 edgefix | 5 | -2.35 | -0.11 | alpha不足、DD不足 |

edge-protectはalpha保護には効くが、DD改善との両立には不十分。

### fold3候補分布

fold3単体でalpha-floor selectorの候補数を集計した。

```bash
.venv/bin/python -u -m unidream.cli.plan011_alpha_floor_selector_probe \
  --config configs/plan011_overlay_actor_v14_edgewm_bconly.yaml \
  --checkpoint-dir checkpoints/plan011_overlay_actor_v14_edgewm_bconly_s007 \
  --folds 3 --seed 7 --device cpu \
  --quick-grid --alpha-floor 3.0 --dd-target -3.0 \
  --output codex_outputs/plan011_alpha_floor_counts_v14_f3
```

| fold | total | alpha_floor | dd_target | alpha_and_dd | alpha>=0 & dd<=0 |
|---:|---:|---:|---:|---:|---:|
| 3 | 2561 | 1141 | 0 | 0 | 464 |

fold3では現quick lowfreq policy family内に、validation robustで `MaxDDDelta <= -3` を満たす候補自体がなかった。最もDD方向の候補でもrobust worstは約 `-0.50pt` で、alphaは大きく崩壊した。

### causal realized-vol target baseline

`unidream.cli.plan011_vol_target_probe` を追加し、WMを使わないcausal realized-vol targetingを確認した。

| fold | val AlphaEx | val MaxDDDelta | test AlphaEx | test MaxDDDelta | 判断 |
|---:|---:|---:|---:|---:|---|
| 3 | -3991.17 | -3.48 | -194.74 | -3.42 | DDは作れるがalpha崩壊 |
| 4 | -179.77 | -3.42 | +2.73 | -5.04 | fold4は近い |
| 5 | +6.88 | -10.56 | -715.25 | -0.40 | val到達、test崩壊 |

単純なvol-managed exposureでも同じ構図。DD改善だけなら作れるが、fold3/5でalphaを守れない。

### v30 DD-utility WM + risk-budget AC

`configs/plan011_overlay_actor_v30_ddutility_ac.yaml` を追加。v29のDD-improving position utility WMを使い、v25/v26系のrisk-budget ACを有効化した。v29 WM checkpointをv30 checkpoint dirへコピーしてBC/ACを新規学習。

fold3途中結果:

| stage | AlphaEx | MaxDDDelta | 判断 |
|---|---:|---:|---|
| BC/test | +18.86 | +0.22 | alphaは出るがDD悪化 |
| AC checkpoint step150 val | -421.25 | - | de-risk collapse |
| AC checkpoint step300 val | -1440.22 | - | さらに崩壊 |
| restore-best | BCへ復元 | - | ACはDD方向に動くとalphaを壊す |

fold4は途中で停止。step300 valはAlphaEx +4.44まで残ったが、全体傾向としてv30 ACもfold3でde-risk collapseし、採用不可。

### v31 B&H-relative constrained AC reward

`configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml` を追加。v30のrisk-state直接ペナルティを切り、AC報酬を以下へ寄せた。

- 主目的: WM `excess_bh` reward の相対log wealth
- 制約: rollout内のB&H相対DD、終端alpha shortfall、tail loss
- 補助: turnover / flow / 小さいoverlay L2 / 軽いupside-miss

実行:

```bash
.venv/bin/python -u -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml \
  --start 2018-01-01 --end 2024-01-01 \
  --folds 3 --seed 7 --device cpu \
  --start-from bc --stop-after test
```

fold3一次結果:

| stage | AlphaEx | MaxDDDelta | 判断 |
|---|---:|---:|---|
| BC-only val | +385.94 | -0.10 | low-turnover overlayとして非常に強い |
| AC step150 val | +149.54 | - | v30のようなde-risk崩壊は止まったがalpha低下 |
| AC step300 val | +187.59 | - | alphaは正だがBCを超えない |
| restore-best/test | +18.86 | +0.22 | val最良はBC。testはDD未改善 |

ログ上のAC rollout exposureは `exp=0.996-1.003` で、route-class時代のunderweight collapseとは違う。つまり報酬関数の方向は以前より健全だが、現実のtest position surfaceではBCを上回る改善として出ていない。

小規模probeからの判断:

- `risk_state penalty` を切ったB&H相対constraint rewardは、少なくともfold3でalpha崩壊を大きく緩和した。
- ただしAC更新はBC teacherの強いalphaを削り、val selectionではBC初期状態に負ける。
- 次の検証は、報酬式そのものをさらに強くするより、BC近傍制約を保ったまま「DDイベント時だけoverlayを動かす」actor更新に寄せるべき。具体的には `alpha_final` を高める、AC stepを短くする、またはDDイベント重み付きのconservative policy updateを入れる。

## 更新判断

- `+3/-3` の材料はfold4局所には何度も出ている。
- fold3/5では、DD改善を強くするとalphaが壊れる。alphaを守るとDD改善が消える。
- v29のDD-improve utility targetは、val上のDD候補を作るがtestへ移らない。
- fixed lowfreq selector / GBDT gate / utility allocator / vol-target / risk-budget AC は同じ壁に当たっている。
- 次にやるなら、単一foldのval選択ではなく、過去複数foldでDDイベントを集約するmeta-validation selector、またはWM targetを「future DDそのもの」ではなく「B&H比のDD改善がalpha lossを上回る局面」のevent labelへ作り直す必要がある。
- AC側は、v31でde-risk collapse自体は抑えられた。次はBC近傍のconservative updateとDDイベント重み付けを組み合わせ、BCのalphaを削らずにDD局面だけ動かせるかを見る。
