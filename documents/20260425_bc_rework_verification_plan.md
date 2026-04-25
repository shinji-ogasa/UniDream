# UniDream 今後の方針: BC再設計と検証計画

作成日: 2026-04-25 JST

## 結論

次の主戦場は `teacher 探し直し` ではなく、BC の学習目的を直すこと。

現在の問題は、learner が `underweight に入る` ことは学んでいる一方で、`benchmark=1.0 に戻る` ことを十分に学んでいない点にある。現 config の `short 90-97%` は実ショートではなく benchmark-relative underweight collapse に近い。

最初にやるべきことは以下。

1. 現本線 `stresstri_shiftonly_s007` の fold0 評価を取る。
2. `short_mass_match` / `mode_rate_match` を弱める。
3. `recovery_*` loss を有効化する。
4. cost-adjusted / advantage-weighted BC を入れる。
5. self-conditioned BC で train/test の inventory state 分布ズレを潰す。
6. それでも安定する場合だけ long-only overweight を入れる。

## 現在の有効候補

### 残す本線

- `dualresanchor`
- `feature_stress_tri`
- `residual shift only`
- `stresstri_shiftonly_s007`

対象 config:

- `configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007.yaml`

現在分かっている結果:

| fold | alpha_excess | maxdd_delta | action dist | 評価 |
|---:|---:|---:|---|---|
| 4 | -0.04 pt/yr | -0.85 pt | short 90% / flat 10% | DD改善はあるが underweight 過多 |
| 5 | +1.19 pt/yr | -0.81 pt | short 97% / flat 3% | flat collapse は抜けたが underweight collapse 寸前 |
| 0 | 未確認 | 未確認 | 未確認 | 最優先で確認 |

### いったん切る候補

以下は当面の本線から外す。

- Probe A/B/C
- `feature_dual two-sided`
- direct regime prior
- no-regime
- target bias
- `infer_logits_target_blend >= 0.675`
- full short / full two-sided leverage

理由:

- Probe C は fold4 test で `AlphaEx -29.18 pt/yr`, `MaxDD delta +4.14 pt`, `long 0%`。
- action space を広げても overweight を使わなかった。
- direct regime prior / target bias は fold4 の局所解を壊した。
- no-regime は flat collapse した。

## 現状削除または無効化すべき実装・設定

ここでの `削除` は、すぐ物理削除する意味ではなく、次の本線 config から外す/退役候補にする意味。

### 1. underweight 頻度コピーを強める設定

現在の本線:

```yaml
short_mass_match_coef: 2.0
mode_rate_match_coef: 1.50
mode_regime_rate_match_coef: 0.75
```

問題:

- underweight が事後的に良かったかに関係なく、頻度をコピーする。
- 現在の `short 90-97%` 偏りを助長している可能性が高い。

対応:

- まず `0.25倍` に落とす。
- その後、advantage-weighted に置き換える。

推奨初期値:

```yaml
short_mass_match_coef: 0.50
mode_rate_match_coef: 0.375
mode_regime_rate_match_coef: 0.1875
```

### 2. 効いていない、または意味が薄い設定

該当:

- `label_smoothing`
- `soft_label_temp` + `residual_aux_ce_coef: 0.0`
- `infer_trade_threshold`

現状:

- `label_smoothing` は BCPretrainer に渡されているが、主要 `_bc_loss()` では使われていない。
- `soft_label_temp` は DP soft labels を作るが、residual controller では `residual_aux_ce_coef: 0.0` のためほぼ効かない。
- `infer_trade_threshold` は config にあるが、actor 側の greedy 推論は連続 trade probability で動いており、明示 threshold としては見えていない。

対応:

- 本線 config ではコメントで `currently inactive` と明記する。
- 使うなら実装を接続する。
- 使わないなら次の cleanup で削除対象にする。

### 3. Probe A/B/C 系

状態:

- checkpoint は削除済み。
- config も削除済み。

対応:

- 再作成しない。
- DP soft は teacher 主体ではなく、sample weighting / confidence として再検討する。

### 4. full short / full two-sided leverage

現時点では入れない。

理由:

- 現在の underweight 癖が残ったまま実ショートを許すと、本物の short collapse になる。
- まず long-only mild overweight で `B&H より強く張る` 能力だけ確認する。

## 有効だと思う手法

### 1. Advantage-weighted BC / AWR-style BC

目的:

- teacher action を一律に真似ない。
- 事後的に価値があった teacher action だけ強く学習する。

基本式:

```text
loss = w_adv * BC_or_position_loss
     + lambda_recovery * recovery_loss
     + lambda_turnover * turnover_penalty
     + lambda_prior * benchmark_prior
```

重み案:

```text
w_adv = clip(exp(cost_adjusted_advantage / tau), w_min, w_max)
```

最初は厳密な IQL ではなく、既存の `outcome_edge` / `sample_quality` を使う軽量 AWR でよい。

優先理由:

- 今の UniDream の collapse は teacher そのものより、teacher 頻度の直接コピーが原因っぽい。
- underweight action を `頻度` ではなく `価値` で重み付けできる。

### 2. Cost-aware supervised policy

目的:

- alpha だけでなく、turnover / cost / avg_hold を直接抑える。

必要な評価:

- gross return
- net return
- trade cost
- turnover
- avg_hold
- action transition matrix

追加すべき損失:

```text
turnover_penalty = abs(pos_t - pos_{t-1})
excess_turnover_penalty = max(0, model_turnover - teacher_turnover)
```

現コードには path loss / execution loss の入口があるため、最初は config 有効化で試せる可能性がある。

### 3. Recovery-aware BC

目的:

- underweight に入った後、benchmark に戻る条件を明示的に学ばせる。

既存実装:

- `recovery_trade_coef`
- `recovery_band_coef`
- `recovery_target_coef`
- `recovery_execution_coef`
- `recovery_underweight_margin`
- `recovery_target_margin`

現在の問題:

- 実装はあるが、本線 config では有効化されていない。

初期案:

```yaml
recovery_trade_coef: 0.25
recovery_band_coef: 0.05
recovery_target_coef: 0.50
recovery_execution_coef: 0.25
recovery_underweight_margin: 0.05
recovery_target_margin: 0.05
```

### 4. Self-conditioned BC

目的:

- train では teacher-forced inventory、test では model rollout inventory という分布ズレを減らす。
- 特に underweight に落ちた自分の状態から benchmark に戻る訓練を入れる。

既存実装:

- `self_condition_prob`
- `self_condition_mode`
- `self_condition_blend`
- `self_condition_max_position_gap`
- `self_condition_max_underweight_gap`
- `self_condition_relabel_step`
- `self_condition_relabel_band`

初期案:

```yaml
self_condition_prob: 0.50
self_condition_warmup_epochs: 2
self_condition_interval: 1
self_condition_mode: "dagger"
self_condition_max_underweight_gap: 0.10
self_condition_relabel_step: 0.10
self_condition_relabel_band: 0.02
```

注意:

- いきなり入れると BC が不安定化する可能性がある。
- recovery loss の後に入れる。

### 5. Long-only mild overweight

目的:

- B&H に勝つ手段を `下げる` だけから、`上げる時に少し強く張る` まで広げる。

最小案:

```yaml
actions:
  values: [0.0, 0.5, 1.0, 1.25]
oracle:
  action_values: [0.0, 0.5, 1.0, 1.25]
ac:
  abs_min_position: 0.0
  abs_max_position: 1.25
  residual_min_overlay: -1.0
  residual_max_overlay: 0.25
```

注意:

- action space を広げるだけでは Probe C と同じく overweight 使用率 0% になる可能性が高い。
- overweight teacher label / advantage weight / cost penalty が必要。
- full short はこの後。

## 検証してほしい計算と欲しい結果

### 共通評価指標

全実験で必ず出す。

| 指標 | 目的 |
|---|---|
| alpha_excess | B&H 超過収益 |
| sharpe_delta | risk-adjusted 改善 |
| maxdd_delta | DD 改善が本物か確認 |
| action distribution | underweight / benchmark / overweight collapse 確認 |
| turnover | コスト負け検出 |
| cost | net return 削りの確認 |
| avg_hold | 高頻度 switching 検出 |
| recovery latency | underweight から benchmark に戻る速さ |
| recovery success rate | 戻り行動を学べたか |
| transition matrix | flat/underweight/benchmark/overweight 遷移の偏り確認 |
| fold別 collapse_guard | fold 局所解の排除 |

### fold 優先順位

1. fold0: 未確認なので最優先。
2. fold4: 既存 local best との比較。
3. fold5: flat collapse 改善が出た fold。
4. fold0/fold4/fold5 の 3 fold で再評価。

## 実験リスト

### Experiment 0: 現本線の fold0 評価

目的:

- `stresstri_shiftonly_s007` が fold0 で即 collapse するか確認する。

変更:

- なし。

対象:

```text
config: medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007.yaml
fold: 0
```

欲しい結果:

- alpha_excess
- maxdd_delta
- short/flat/benchmark 分布
- turnover
- cost
- avg_hold
- recovery latency

判定:

- fold0 で `short >= 90%` なら、現本線は追加制約なしでは本線化しない。
- fold0 で `maxdd_delta < 0` かつ `short <= 70%` なら、制約追加の価値あり。

### Experiment A: recovery loss + underweight頻度コピー弱体化

目的:

- 下げるだけでなく戻ることを学ばせる。
- `short_mass_match` / `mode_rate_match` が collapse を助長しているか確認する。

変更案:

```yaml
short_mass_match_coef: 0.50
mode_rate_match_coef: 0.375
mode_regime_rate_match_coef: 0.1875
recovery_trade_coef: 0.25
recovery_band_coef: 0.05
recovery_target_coef: 0.50
recovery_execution_coef: 0.25
recovery_underweight_margin: 0.05
recovery_target_margin: 0.05
```

fold:

- まず fold4。
- 良ければ fold5 / fold0。

欲しい結果:

- short 比率が現本線より下がるか。
- maxdd_delta を維持できるか。
- recovery latency が短くなるか。
- alpha_excess が悪化しすぎないか。

合格目安:

- fold4/fold5 で `maxdd_delta < 0` を維持。
- short 比率が `90-97%` から `70%以下` に低下。
- cost が悪化しない。

### Experiment B: A + cost-adjusted sample weighting

目的:

- underweight を頻度ではなく価値で学ばせる。

既存入口:

```yaml
sample_quality_mode: "outcome_edge"
sample_quality_coef: 1.0
sample_quality_clip: 4.0
```

検証したい派生:

```yaml
sample_quality_mode: "outcome_edge"
sample_quality_coef: 0.50 / 1.00 / 2.00
sample_quality_clip: 2.0 / 4.0
```

欲しい結果:

- underweight が価値のある局面に集中するか。
- action dist が正常化するか。
- turnover/cost が下がるか。
- fold4/fold5/fold0 で再現するか。

追加で欲しい集計:

- underweight 中の forward excess return 平均。
- benchmark 中の forward excess return 平均。
- underweight サンプルの `sample_quality` 分位別成績。

### Experiment C: B + self-conditioned BC

目的:

- train/test の inventory state 分布ズレを減らす。
- 自分で underweight に落ちた状態から benchmark に戻る訓練を入れる。

変更案:

```yaml
self_condition_prob: 0.50
self_condition_warmup_epochs: 2
self_condition_interval: 1
self_condition_mode: "dagger"
self_condition_max_underweight_gap: 0.10
self_condition_relabel_step: 0.10
self_condition_relabel_band: 0.02
```

欲しい結果:

- recovery latency 改善。
- short 比率低下。
- avg_hold が極端に短くならない。
- cost が増えすぎない。

リスク:

- self rollout が悪い初期 policy に引っ張られる。
- BC loss が下がっても test が悪化する可能性あり。

### Experiment D: path / turnover aware BC

目的:

- trade しすぎを抑える。
- net return を削る cost を下げる。

変更案:

```yaml
path_aux_coef: 0.25
path_horizon: 8
path_position_coef: 1.0
path_turnover_coef: 0.25
path_shortfall_coef: 0.25
```

欲しい結果:

- turnover 低下。
- cost 低下。
- avg_hold 上昇。
- maxdd_delta 維持。

注意:

- turnover を抑えすぎると recovery が遅くなる。
- recovery loss と同時に見る。

### Experiment E: long-only mild overweight

前提:

- Experiment A-C のどれかで underweight collapse が改善していること。

目的:

- B&H に勝つ手段を `下げる` だけから `上げる時に強く張る` へ広げる。

変更案:

```yaml
actions:
  values: [0.0, 0.5, 1.0, 1.25]
oracle:
  action_values: [0.0, 0.5, 1.0, 1.25]
ac:
  abs_min_position: 0.0
  abs_max_position: 1.25
  residual_min_overlay: -1.0
  residual_max_overlay: 0.25
```

欲しい結果:

- overweight 使用率。
- overweight 中の forward return / excess return。
- underweight と overweight の遷移。
- alpha_excess 改善。
- maxdd_delta 悪化の有無。
- leverage cost を仮置きした net return。

判定:

- overweight 使用率が 0% なら、action space ではなく teacher/loss が悪い。
- overweight 使用率が高すぎるなら、long collapse。
- `alpha_excess` が改善しても `maxdd_delta` が悪化するなら不採用。

### Experiment F: mild short

前提:

- Experiment E で overweight が使われ、collapse していないこと。

目的:

- 実ショートの価値を確認する。

最小案:

```yaml
actions:
  values: [-0.25, 0.0, 0.5, 1.0, 1.25]
oracle:
  action_values: [-0.25, 0.0, 0.5, 1.0, 1.25]
ac:
  abs_min_position: -0.25
  abs_max_position: 1.25
  residual_min_overlay: -1.25
  residual_max_overlay: 0.25
```

必須追加:

- short borrow cost
- short duration penalty
- short frequency cap
- short recovery loss

欲しい結果:

- real short 使用率。
- real short の平均保有時間。
- real short 中の net PnL。
- short から benchmark への recovery latency。
- liquidation / tail risk proxy。

現時点の優先度:

- 低い。
- underweight collapse が直るまで着手しない。

## 実験の合格基準

最低条件:

- `maxdd_delta <= 0`
- `alpha_excess >= 0` 近辺
- underweight 比率が `70%以下`
- flat collapse なし
- cost が Probe C のように net return を大きく削らない
- fold0/fold4/fold5 のうち少なくとも 2 fold で同方向

M2候補に昇格する条件:

- fold0/fold4/fold5 で `maxdd_delta < 0`
- 3 fold 平均で `alpha_excess > 0`
- collapse_guard 通過
- turnover/cost が許容範囲
- recovery latency が悪化していない

## 次に作るべき config

優先順。

1. `*_recovery_weakshortcopy.yaml`
2. `*_recovery_weighted_outcomeedge.yaml`
3. `*_recovery_weighted_selfcond.yaml`
4. `*_recovery_weighted_pathcost.yaml`
5. `*_longonly_ow125.yaml`

最初の config は現本線から差分最小で作る。

## 実装タスク候補

### すぐできる config-only タスク

- `recovery_*` を有効化。
- `short_mass_match_coef` / `mode_rate_match_coef` を弱める。
- `sample_quality_mode: outcome_edge` と `sample_quality_coef` を有効化。
- `self_condition_*` を有効化。
- `path_aux_coef` / `path_turnover_coef` を有効化。

### 小さな実装タスク

- `label_smoothing` が未使用なら削除または `_bc_loss()` に接続する。
- `infer_trade_threshold` が未使用なら削除または greedy inference に接続する。
- recovery metrics を test report に追加する。
- action transition matrix を test report に追加する。
- underweight / benchmark / overweight 別 PnL attribution を出す。

### 中規模実装タスク

- cost-adjusted advantage を明示的に計算する。
- `w_adv = exp(clip(adv / tau))` 型の weighted BC を追加する。
- benchmark prior KL を追加する。
- long-only overweight teacher を作る。

## 暫定ロードマップ

### Step 1

`stresstri_shiftonly_s007` の fold0 をそのまま評価する。

### Step 2

`Experiment A` を fold4 で実行する。

### Step 3

fold4 が改善したら fold5/fold0 に広げる。

### Step 4

`Experiment B` で sample weighting を入れる。

### Step 5

`Experiment C` で self-conditioned BC を入れる。

### Step 6

collapse が収まった場合だけ `Experiment E` の long-only overweight に進む。

## 現時点の判断

今すぐ full short / full leverage に進むべきではない。

まず直すべき本丸は以下。

```text
underweight を頻度で真似るな。
underweight が事後的に価値を持った時だけ強く真似ろ。
benchmark 復帰を明示的に学習させろ。
train/test の inventory 分布ズレを潰せ。
```

この方針で fold0/fold4/fold5 を順に潰す。

## 実行結果ログ

### 2026-04-25 Experiment 0: `stresstri_shiftonly_s007` fold0 test

実行:

```powershell
uv run python -m unidream.cli.train `
  --config configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007.yaml `
  --start 2020-01-01 `
  --end 2024-01-01 `
  --folds 0 `
  --start-from test `
  --stop-after test `
  --device auto
```

ログ:

- `documents/logs/20260425_exp0_stresstri_s007_fold0_test.log`

前提:

- fold0 は既存の `world_model.pt` / `bc_actor.pt` に加えて、直前の中断実行で `ac.pt` / `ac_best.pt` が作成済みだった。
- 今回は `--start-from test` で test を再実行した。
- device は `cuda` として解決された。

結果:

| fold | stage | alpha_excess | sharpe_delta | maxdd_delta | win_rate | M2 | collapse_guard |
|---:|---|---:|---:|---:|---:|---|---|
| 0 | test | +0.88 pt/yr | +0.009 | -1.65 pt | 49.5% | MISS | pass |

Test metrics:

- Sharpe: `-3.405`
- Sortino: `-4.333`
- MaxDD: `-0.570`
- Calmar: `-3.339`
- TotalRet: `-0.4743`
- PnL attr: `long=-0.6424`, `short=+0.0000`, `cost=0.0007`, `net=-0.6430`
- Test dist: `long=0% short=0% flat=100% mean=-0.045 switches=4 avg_hold=2184.2b turnover=0.21`

Validation adjust:

| scale | alpha | sharpe_delta | maxdd_delta | dist | M2 |
|---:|---:|---:|---:|---|---|
| 0.750 | +0.95 pt | +0.002 | -0.98 pt | long 0% / short 1% / flat 99% | miss |
| 1.000 | +1.16 pt | +0.006 | -0.97 pt | long 0% / short 4% / flat 96% | miss |
| 1.500 | +1.39 pt | +0.011 | -0.97 pt | long 0% / short 6% / flat 94% | miss |

選択:

- `scale=0.750`

判定:

- fold0 は fold4/fold5 のような `underweight 90-97%` ではなく、ほぼ benchmark 近辺の flat/hold に寄った。
- `alpha_excess > 0` と `maxdd_delta < 0` は満たすが、Sharpe が大きくマイナスで M2 は MISS。
- collapse guard は pass だが、実質的には benchmark 近辺の低アクティブ policy に近い。
- fold0 は「short collapse」ではなく「ほぼ benchmark/flat hold」。このため、現本線は fold ごとに `underweight collapse` と `benchmark hold` の両方へ倒れる不安定性がある。

次アクション:

- Experiment A を作る。
- `short_mass_match_coef` / `mode_rate_match_coef` を弱める。
- `recovery_*` loss を有効化する。
- まず fold4 で実行し、fold4 の `short 90%` が下がるかを見る。

### 2026-04-25 Experiment A: recovery loss + weak short-copy fold4

実行:

```powershell
uv run python -m unidream.cli.train `
  --config configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007_recovery_weakshortcopy.yaml `
  --start 2020-01-01 `
  --end 2024-01-01 `
  --folds 4 `
  --device auto
```

ログ:

- `documents/logs/20260425_expA_recovery_weakshortcopy_fold4.log`

config 差分:

```yaml
short_mass_match_coef: 0.50
mode_rate_match_coef: 0.375
mode_regime_rate_match_coef: 0.1875
recovery_trade_coef: 0.25
recovery_band_coef: 0.05
recovery_target_coef: 0.50
recovery_execution_coef: 0.25
recovery_underweight_margin: 0.05
recovery_target_margin: 0.05
```

実行上の補足:

- 新規実験 config は path 長回避のため `checkpoint_dir: checkpoints/expA_recovery_weakshortcopy` に短縮した。
- data cache は `cache_dir: checkpoints/data_cache` を明示した。
- 1回目は長い checkpoint path により parquet cache 保存で失敗。config を短縮して再実行済み。

結果:

| fold | stage | alpha_excess | sharpe_delta | maxdd_delta | win_rate | M2 | collapse_guard |
|---:|---|---:|---:|---:|---:|---|---|
| 4 | test | -0.13 pt/yr | -0.003 | -0.21 pt | 50.0% | MISS | pass |

Test metrics:

- Sharpe: `0.022`
- Sortino: `0.027`
- MaxDD: `-0.182`
- Calmar: `0.049`
- TotalRet: `0.0022`
- PnL attr: `long=+0.0028`, `short=+0.0000`, `cost=0.0006`, `net=+0.0022`
- Test dist: `long=0% short=0% flat=100% mean=-0.015 switches=3 avg_hold=2912.3b turnover=0.14`

Validation adjust:

| scale | alpha | sharpe_delta | maxdd_delta | dist | M2 |
|---:|---:|---:|---:|---|---|
| 0.750 | -6.42 pt | +0.007 | -0.28 pt | long 0% / short 0% / flat 100% | miss |
| 1.000 | -6.28 pt | +0.008 | -0.29 pt | long 0% / short 0% / flat 100% | miss |
| 1.500 | -6.12 pt | +0.009 | -0.29 pt | long 0% / short 0% / flat 100% | miss |

BC / AC observations:

- BC-only val: `AlphaExcess=-30.18pt`, dist `flat=100% mean=-0.029`, turnover `16.30`。
- AC 後の train/val/test はすべてほぼ `flat=100%`。
- `recovery + weak short-copy` は fold4 の `short 90%` を止めたが、benchmark/flat hold 側へ倒した。

判定:

- Experiment A は単独では不採用。
- `short_mass_match` / `mode_rate_match` の急な 0.25倍化と recovery loss の同時投入は、underweight 使用を消しすぎた。
- ただし、underweight collapse を抑える方向には効いている。
- 次は `short_mass/mode_rate` を完全には落としすぎず、価値重み付けを入れる Experiment B に進む。

次アクション:

- Experiment B を作る。
- A をベースに `sample_quality_mode: outcome_edge` と `sample_quality_coef` を入れる。
- ただし A は flat へ寄りすぎたため、B では `short_mass_match_coef` / `mode_rate_match_coef` を少し戻す案も検討する。

### 2026-04-25 Experiment B: recovery + outcome-edge weighted BC fold4

実行:

```powershell
uv run python -m unidream.cli.train `
  --config configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007_recovery_weighted_outcomeedge.yaml `
  --start 2020-01-01 `
  --end 2024-01-01 `
  --folds 4 `
  --device auto
```

ログ:

- `documents/logs/20260425_expB_recovery_weighted_outcomeedge_fold4.log`

config 差分:

```yaml
sample_quality_mode: "outcome_edge"
sample_quality_coef: 1.0
sample_quality_clip: 4.0
short_mass_match_coef: 1.00
mode_rate_match_coef: 0.75
mode_regime_rate_match_coef: 0.375
recovery_*: Experiment A と同じ
```

意図:

- Experiment A は flat 100% へ倒れたため、`short_mass/mode_rate` を元値の半分まで戻した。
- `outcome_edge` で事後的に価値がある underweight sample を強く学習させる狙い。

結果:

| fold | stage | alpha_excess | sharpe_delta | maxdd_delta | win_rate | M2 | collapse_guard |
|---:|---|---:|---:|---:|---:|---|---|
| 4 | test | -8.60 pt/yr | -0.225 | +0.59 pt | 47.6% | MISS | pass |

Test metrics:

- Sharpe: `-0.200`
- Sortino: `-0.247`
- MaxDD: `-0.190`
- Calmar: `-0.410`
- TotalRet: `-0.0194`
- PnL attr: `long=+0.0093`, `short=+0.0000`, `cost=0.0289`, `net=-0.0196`
- Test dist: `long=0% short=4% flat=96% mean=-0.034 switches=1031 avg_hold=8.5b turnover=51.57`

Validation adjust:

| scale | alpha | sharpe_delta | maxdd_delta | dist | 判定 |
|---:|---:|---:|---:|---|---|
| 0.750 | -64.28 pt | -0.226 | -0.20 pt | long 0% / short 3% / flat 97% | reject alpha<-25 |
| 1.000 | -83.08 pt | -0.327 | -0.04 pt | long 0% / short 4% / flat 96% | reject alpha<-25 |
| 1.500 | -83.95 pt | -0.333 | -0.03 pt | long 0% / short 4% / flat 96% | reject alpha<-25 |

BC / AC observations:

- BC-only val: `AlphaExcess=-381.64pt`, dist `short=98% flat=2%`, turnover `638.11`, cost `0.3515`。
- AC は最終的に `short=4% flat=96%` まで戻したが、turnover `51.57` と cost `0.0289` が大きい。
- `outcome_edge` weighting は underweight を選別するより、BC 初期 policy を extreme underweight/high-turnover に押した。

判定:

- Experiment B は不採用。
- `sample_quality_coef: 1.0` + `short_mass/mode_rate` 半戻しは強すぎる。
- cost-aware どころか cost を増やして net return を削った。
- A は underweight を消しすぎ、B は underweight/high-turnover を戻しすぎ。中間探索が必要。

次アクション:

- Experiment C は self-conditioned BC を追加するが、B をそのままベースにすると悪化リスクが高い。
- C は `sample_quality_coef` を弱めた `0.25`、`short_mass/mode_rate` は A 寄りに戻して実行する。
- 目的は、self-conditioned inventory state が recovery と turnover を改善するかを見ること。

### 2026-04-25 Experiment C: recovery + weak outcome-edge + self-conditioned BC fold4

実行:

```powershell
uv run python -m unidream.cli.train `
  --config configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007_recovery_weighted_selfcond.yaml `
  --start 2020-01-01 `
  --end 2024-01-01 `
  --folds 4 `
  --device auto
```

ログ:

- `documents/logs/20260425_expC_recovery_weighted_selfcond_fold4.log`

config 差分:

```yaml
sample_quality_mode: "outcome_edge"
sample_quality_coef: 0.25
sample_quality_clip: 2.0
self_condition_prob: 0.50
self_condition_warmup_epochs: 2
self_condition_interval: 1
self_condition_mode: "dagger"
self_condition_max_underweight_gap: 0.10
self_condition_relabel_step: 0.10
self_condition_relabel_band: 0.02
short_mass_match_coef: 0.50
mode_rate_match_coef: 0.375
mode_regime_rate_match_coef: 0.1875
recovery_*: Experiment A と同じ
```

意図:

- B の `sample_quality_coef: 1.0` は強すぎたため `0.25` に弱めた。
- A の flat 偏りを避けるため、self-conditioned inventory state で underweight からの戻り訓練を入れた。

結果:

| fold | stage | alpha_excess | sharpe_delta | maxdd_delta | win_rate | M2 | collapse_guard |
|---:|---|---:|---:|---:|---:|---|---|
| 4 | test | -0.57 pt/yr | -0.014 | -0.17 pt | 50.0% | MISS | pass |

Test metrics:

- Sharpe: `0.011`
- Sortino: `0.014`
- MaxDD: `-0.183`
- Calmar: `0.025`
- TotalRet: `0.0011`
- PnL attr: `long=+0.0017`, `short=+0.0000`, `cost=0.0006`, `net=+0.0011`
- Test dist: `long=0% short=0% flat=100% mean=-0.014 switches=1 avg_hold=8737.0b turnover=0.03`

Validation adjust:

| scale | alpha | sharpe_delta | maxdd_delta | dist | M2 |
|---:|---:|---:|---:|---|---|
| 0.750 | -7.40 pt | +0.001 | -0.26 pt | long 0% / short 0% / flat 100% | miss |
| 1.000 | -7.94 pt | +0.003 | -0.30 pt | long 0% / short 0% / flat 100% | miss |
| 1.500 | -8.37 pt | +0.005 | -0.33 pt | long 0% / short 0% / flat 100% | miss |

BC / AC observations:

- BC-only val: `AlphaExcess=-7.92pt`, dist `flat=100% mean=-0.013`, turnover `0.03`。
- self-conditioned BC は B の high-turnover collapse を止めた。
- ただし underweight 活用もほぼ消え、A と同じ benchmark/flat hold へ倒れた。
- BC stage が大幅に重くなった。fold4 の BC だけで約27分かかっている。

判定:

- Experiment C は不採用。
- self-conditioned BC は分布ズレ/turnover 抑制には効くが、現設定では active decision を消しすぎる。
- `self_condition_prob: 0.50` は強すぎる可能性が高い。
- 再検討するなら `0.10-0.25` からにする。

次アクション:

- Experiment D で path / turnover aware BC を確認する。
- ただし C で既に turnover は十分低く、問題は active underweight が消えることなので、D は A ベースではなく現本線寄りの active policy に対して path cost を入れる方が良い。

### 2026-04-25 Experiment D: path / turnover aware BC fold4

実行:

```powershell
uv run python -m unidream.cli.train `
  --config configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007_pathcost.yaml `
  --start 2020-01-01 `
  --end 2024-01-01 `
  --folds 4 `
  --device auto
```

ログ:

- `documents/logs/20260425_expD_pathcost_fold4.log`

config 差分:

```yaml
chunk_size: 1
path_aux_coef: 0.25
path_horizon: 8
path_position_coef: 1.0
path_turnover_coef: 0.25
path_shortfall_coef: 0.25
short_mass_match_coef: 2.0
mode_rate_match_coef: 1.50
mode_regime_rate_match_coef: 0.75
```

実装上の注意:

- `path_aux_coef` は現状 `chunk_size > 1` の branch では使われない。
- そのため D では `chunk_size: 1` に変更して path loss を実際に有効化した。

結果:

| fold | stage | alpha_excess | sharpe_delta | maxdd_delta | win_rate | M2 | collapse_guard |
|---:|---|---:|---:|---:|---:|---|---|
| 4 | test | -0.42 pt/yr | -0.010 | -0.70 pt | 50.0% | MISS | pass |

Test metrics:

- Sharpe: `0.016`
- Sortino: `0.019`
- MaxDD: `-0.177`
- Calmar: `0.034`
- TotalRet: `0.0015`
- PnL attr: `long=+0.0021`, `short=+0.0000`, `cost=0.0006`, `net=+0.0015`
- Test dist: `long=0% short=1% flat=99% mean=-0.048 switches=3 avg_hold=2912.3b turnover=0.17`

Validation adjust:

| scale | alpha | sharpe_delta | maxdd_delta | dist | 判定 |
|---:|---:|---:|---:|---|---|
| 0.750 | -26.75 pt | +0.013 | -0.94 pt | long 0% / short 0% / flat 100% | reject alpha<-25 |
| 1.000 | -26.93 pt | +0.013 | -0.94 pt | long 0% / short 1% / flat 99% | reject alpha<-25 |
| 1.500 | -27.10 pt | +0.013 | -0.95 pt | long 0% / short 2% / flat 98% | reject alpha<-25 |

BC / AC observations:

- BC-only val: `AlphaExcess=-24.59pt`, dist `flat=100% mean=-0.043`, turnover `1.76`。
- AC 中間では train/val の short 比率が一時的に上がったが、best checkpoint は最終的に `flat=100%` 側。
- test でも `short=1% flat=99%` で、active underweight はほぼ消えた。

判定:

- Experiment D は不採用。
- path/turnover 制約は turnover/cost を抑えるが、現設定では active decision も消す。
- `chunk_size: 1` への変更自体も挙動を大きく変えている。
- path loss を再検討するなら、chunk branch でも path loss が効くように実装するか、係数をかなり弱くする必要がある。

ここまでの A-D 暫定まとめ:

| 実験 | 主効果 | 問題 | 判定 |
|---|---|---|---|
| A | underweight collapse を止める | flat 100% | 不採用 |
| B | underweight を戻す | BC short 98%, high turnover, alpha/DD悪化 | 不採用 |
| C | turnover を止める | self-condition が重く flat 100% | 不採用 |
| D | path cost で low turnover | active decision 消失 | 不採用 |

次アクション:

- 条件は十分ではないが、表現力不足の切り分けとして Experiment E の long-only mild overweight を実行する。
- 目的は `overweight 使用率 0% になるか` の確認。
