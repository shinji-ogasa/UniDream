# bcplan_2 transition advantage / routing BC 検証結果

作成日: 2026-04-25 JST

## 結論

レビューの主張は概ね正しい。今回の実験でも、`予測stateを強くする` より `transition/action 単位の cost-adjusted advantage を作る` 方が有効だった。

ただし、`best transition をそのまま position label にしてBC` は危険。平滑化なしでは high-turnover short collapse した。平滑化と margin を入れるとBCは安定し、さらに保守的ACを軽く入れると fold4 では改善した。

今回の最良結果は以下。

```text
config: configs/bc_transition_relabel_smooth_m0005_ac500_s007.yaml
fold: 4
stage: transition relabel BC + conservative AC 500 steps
AlphaEx: +3.57 pt/yr
SharpeΔ: +0.082
MaxDDΔ: -0.28 pt
WinRate: 48.3%
Dist: long 6% / short 0% / flat 94%
Turnover: 8.42
M2: MISS
Collapse guard: pass
```

これは前回の no-pred fair baseline より良い。

```text
no-pred fair baseline:
AlphaEx: -0.54 pt/yr
SharpeΔ: -0.013
MaxDDΔ: -0.67 pt
Dist: short 3% / flat 97%
Turnover: 0.99
```

ただし本番候補にはまだ足りない。理由は `turnover=8.42` が高く、`flat=94%` で active decision がまだ薄いこと。ACは改善したが、M2条件には未達。

## Web調査からの判断

- IQL は policy extraction で advantage-weighted behavioral cloning を使う。今回の `transition advantage label` は、この方向性と整合する。Source: https://arxiv.org/abs/2110.06169
- AWAC は advantage-weighted actor update で offline/prior data を活用する。今回の transition value で重み付けする方針を支持する。Source: https://arxiv.org/abs/2006.09359
- TD3+BC は offline RL でBC anchorを残す。今回ACを保守的にした理由はここ。Source: https://arxiv.org/abs/2106.06860
- CQL はOOD actionのQ過大評価を抑える保守的criticの考え方。今後の overweight/short 拡張では必要。Source: https://arxiv.org/abs/2006.04779

## 実装したこと

### 1. transition advantage 計算

追加:

- `unidream/experiments/transition_advantage.py`

各時刻 `t` で候補actionを評価する。

```text
candidate actions: 0.0 / 0.5 / 1.0 / 1.25
horizons: 4 / 8 / 16 / 32
value = future excess return
      - transaction cost
      - volatility penalty
      - drawdown penalty
      - leverage penalty
      - turnover penalty
```

分類:

```text
neutral
de_risk
stay_underweight
recovery
overweight
```

### 2. 診断CLI

追加:

- `unidream/cli/transition_advantage_probe.py`

出力:

- transition class別 advantage
- action別 top/bottom decile value
- transition matrix
- recovery latency
- JSON/Markdown report

### 3. BC relabel 接続

変更:

- `unidream/experiments/fold_inputs.py`

追加config:

```yaml
bc:
  transition_advantage_relabel: true
  transition_candidate_actions: [0.0, 0.5, 1.0, 1.25]
  transition_advantage_horizons: [4, 8, 16, 32]
  transition_advantage_margin: 0.0005
  transition_relabel_smooth: true
```

`best transition` を直接 `oracle_positions` に差し替える。ただし、平滑化なしは崩れるので `transition_relabel_smooth` が重要。

## Experiment G: transition advantage label probe

実行:

```powershell
uv run python -m unidream.cli.transition_advantage_probe `
  --config configs/bc_multitask_aux_nopred_fairwm_s007.yaml `
  --start 2020-01-01 `
  --end 2024-01-01 `
  --folds 4 `
  --candidate-actions 0.0,0.5,1.0,1.25
```

出力:

- `documents/20260425_transition_advantage_probe_fold4.md`
- `documents/20260425_transition_advantage_probe_fold4.json`
- `documents/logs/20260425_transition_advantage_probe_fold4.log`

結果:

| item | value |
|---|---:|
| target underweight | 34.4% |
| target benchmark | 32.9% |
| target overweight | 32.7% |
| mean best advantage | 0.003225 |
| recovery rate from underweight | 41.2% |

Class別:

| class | rate | mean_adv | top_decile | positive_rate |
|---|---:|---:|---:|---:|
| de_risk | 3.3% | 0.002715 | 0.009953 | 100.0% |
| stay_underweight | 31.1% | 0.007966 | 0.028054 | 100.0% |
| recovery | 14.9% | 0.000206 | 0.001904 | 15.4% |
| overweight | 32.7% | 0.001917 | 0.006831 | 100.0% |

判断:

- transition価値は存在する。
- 特に overweight と stay_underweight は top decile が効いている。
- recovery は価値が小さいが、underweightからbenchmarkへ戻る局面はある。
- したがって、レビューの「action/transitionごとに価値ラベルを作れ」は支持された。

## BC / AC 実験結果

| 実験 | config | stage | AlphaEx | SharpeΔ | MaxDDΔ | dist | turnover | guard | 判定 |
|---|---|---|---:|---:|---:|---|---:|---|---|
| baseline | `bc_multitask_aux_nopred_fairwm_s007` | BC | -0.54pt | -0.013 | -0.67pt | short 3% / flat 97% | 0.99 | pass | 比較基準 |
| H1 direct relabel | `bc_transition_relabel_ow125_s007` | BC | -79.19pt | -5.370 | +13.75pt | short 100% | 665.46 | pass | 不採用 |
| H2 smooth m001 | `bc_transition_relabel_smooth_m001_ow125_s007` | BC | +0.08pt | +0.002 | +0.08pt | flat 100% | 0.47 | pass | 安定だがinactive |
| H3 smooth m0005 | `bc_transition_relabel_smooth_m0005_ow125_s007` | BC | +0.14pt | +0.003 | +0.08pt | long 2% / flat 98% | 7.07 | pass | AC候補 |
| J AC500 | `bc_transition_relabel_smooth_m0005_ac500_s007` | AC | +3.57pt | +0.082 | -0.28pt | long 6% / flat 94% | 8.42 | pass | 今回最良 |
| J lowadj | `bc_transition_relabel_smooth_m0005_ac500_lowadj_s007` | AC test-only | +0.02pt | +0.000 | +0.24pt | flat 100% | 2.13 | pass | turnover低下だが弱い |

## 実験別メモ

### H1: direct transition relabel

平滑化なしで `best transition` をそのまま教師にした。

結果:

```text
Transition relabel train dist: short 33% / benchmark 39% / overweight 28%
Oracle aim dist after relabel: long 28% / short 33% / flat 39%
Test: short 100%, turnover 665.46, AlphaEx -79.19pt
```

判定:

- 完全に不採用。
- label distribution は一見よくても、transitionが高頻度すぎてBCが壊れる。
- レビューの「position直接BCが悪い」という指摘を強く支持する結果。

### H2: smoothed relabel m001

変更:

```yaml
transition_candidate_actions: [0.5, 1.0, 1.25]
transition_advantage_margin: 0.001
transition_relabel_smooth: true
transition_relabel_max_step: 0.05
sample_quality_coef: 0.25
```

結果:

```text
AlphaEx +0.08pt
turnover 0.47
flat 100%
```

判定:

- 安定化には成功。
- ただし active decision が消えた。

### H3: smoothed relabel m0005

変更:

```yaml
transition_advantage_margin: 0.0005
transition_relabel_max_step: 0.10
sample_quality_coef: 0.5
```

結果:

```text
AlphaEx +0.14pt
turnover 7.07
long 2% / flat 98%
```

判定:

- BC単体では微改善。
- activeは少し出たが、まだ弱い。
- 安定しているためAC候補として採用。

### J: conservative AC 500step

変更:

```yaml
max_steps: 500
actor_lr: 1.0e-5
td3bc_alpha: 5.0
turnover_coef: 0.50
prior_kl_coef: 0.20
prior_trade_coef: 0.10
prior_flow_coef: 0.30
```

結果:

```text
AlphaEx +3.57pt
SharpeΔ +0.082
MaxDDΔ -0.28pt
long 6% / short 0% / flat 94%
turnover 8.42
```

判定:

- 今回最良。
- BCからACへ進む価値はあった。
- ただし turnover が高く、M2はMISS。
- 本線候補ではあるが、本番採用ではない。

### low adjustment test

AC checkpointはそのまま、val adjust scaleを `[0.25, 0.5, 0.75]` に落とした。

結果:

```text
AlphaEx +0.02pt
turnover 2.13
flat 100%
MaxDDΔ +0.24pt
```

判定:

- turnoverは落ちたがalphaも消えた。
- scaleで雑に抑えるだけではダメ。

## レビューへの回答

レビューの中で正しかった点:

1. `予測state直結` ではなく、transition-level valueが必要。
2. action/transition別の cost-adjusted advantage は有効な診断になる。
3. 壊れたBCからACに進むのは危険。
4. ただし、安定したBCに保守的ACを足すと改善余地がある。

今回わかった追加事項:

1. transition advantage label はそのままBC教師にすると高頻度化して壊れる。
2. margin + smoothing は必須。
3. single actorのままでも改善は出るが、flat 94%でまだroutingとして弱い。
4. 次は本当の `neutral / de-risk / recovery / overweight` routing head が必要。

## 採用判断

採用する実装:

- `transition_advantage.py`
- `transition_advantage_probe.py`
- `transition_advantage_relabel`
- `transition_relabel_smooth`

暫定best config:

```text
configs/bc_transition_relabel_smooth_m0005_ac500_s007.yaml
```

ただし、M2未達なのでproduction configではない。

退役/不採用:

- `bc_transition_relabel_ow125_s007`: direct relabelは危険。
- `bc_transition_relabel_smooth_m001_ow125_s007`: 安定だがinactive。
- `bc_transition_relabel_smooth_m0005_ac500_lowadj_s007`: turnoverは落ちるがalphaが消える。

## 次にやるべきこと

1. true routing actorを実装する。

```text
route heads:
  neutral
  de-risk
  recovery
  overweight

predictive state:
  risk preds -> de-risk gate
  return preds -> overweight gate
  inventory/current position -> recovery gate
```

2. transition labelを position target ではなく route target に使う。
3. `turnover <= 4` をloss/selectorで直接満たすようにする。
4. ACでは critic側に CQL-lite penalty を追加し、overweight/de-riskのQ過大評価を抑える。
5. fold4で `AlphaEx >= +3pt`, `MaxDDΔ <= 0`, `turnover <= 4`, `collapse_guard pass` を満たしてから fold0/fold5へ広げる。

## 実行ログ

- `documents/logs/20260425_transition_advantage_probe_fold4.log`
- `documents/logs/20260425_bc_transition_relabel_ow125_fold4.log`
- `documents/logs/20260425_bc_transition_relabel_smooth_m001_ow125_fold4.log`
- `documents/logs/20260425_bc_transition_relabel_smooth_m0005_ow125_fold4.log`
- `documents/logs/20260425_bc_transition_relabel_smooth_m0005_ac500_fold4.log`
- `documents/logs/20260425_bc_transition_relabel_smooth_m0005_ac500_lowadj_fold4.log`

## 検証

```powershell
uv run python -m py_compile `
  unidream\experiments\transition_advantage.py `
  unidream\cli\transition_advantage_probe.py `
  unidream\experiments\fold_inputs.py
```

結果: compile OK。
