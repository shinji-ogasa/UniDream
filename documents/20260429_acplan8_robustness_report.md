# AC Plan 8 Robustness / Selector / Sizing Adapter Report

対象計画: `documents/acplan_8.md`

対象fold: checkpoint が存在する fold 0 / 4 / 5。

## 結論

Plan 8で確認したこと:

```text
1. AC Plan 7採用checkpointの再現評価
2. adapter attribution診断の追加
3. selector grid改善案の検証
4. cost/slippage stress
5. seed 11でのAC再学習
6. guarded sizing adapter variant
```

採用:

```text
adapter_detail diagnostics
selector scoring / tie-break / hard guard の拡張コード
現行の AC Plan 7: scale=0.5 固定
```

不採用:

```text
val_adjust_rate_scale grid selector
seed 11 AC retrain result
guarded sizing adapter variant
```

一番重要な判断:

```text
AC Plan 7の「採用済みcheckpoint」は再現評価で良い。
ただし、ACをseed違いで再学習すると fold5 が崩れた。
したがって、現時点ではAC範囲拡大は禁止継続。
```

## 実装コミット

```text
e74c6ef Add Plan 8 selector and adapter diagnostics
```

追加した実装:

```text
selector:
  selector_alpha_score_coef
  selector_sharpe_score_coef
  selector_turnover_target
  selector_turnover_excess_score_coef
  selector_period_win_bonus_coef
  selector_max_long_rate
  selector_max_short_rate
  selector_hard_maxdd_delta_pt
  selector_near_best_tiebreak = conservative / balanced / score

adapter diagnostics:
  fire rate
  mean signed delta
  positive delta rate
  long_state rate
  fire時PnL
  non-fire時PnL
```

## コマンド確認

質問: `uv run python -m unidream.cli.train` だけで、途中から再開しなくても今の結果まで行けるか。

答え:

```text
ロジック上は yes。
現在の config は WM -> BC -> AC curriculum -> test を自動で流す。
400stepで止めて手動変更する必要はない。
```

ただし、結果再現という意味では注意が必要。

```text
uv run python -m unidream.cli.train
```

これはデフォルトで:

```text
seed = 42
folds = 全14fold
start_from = wm
stop_after = test
```

なので、今回の報告値とは条件が違う。今回の報告値を再評価するにはこれ。

```powershell
uv run python -m unidream.cli.train `
  --config configs\trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,4,5 `
  --start-from test `
  --stop-after test `
  --seed 7 `
  --device cuda
```

ACを既存BCから同じcurriculumで再学習するならこれ。

```powershell
uv run python -m unidream.cli.train `
  --config configs\trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,4,5 `
  --start-from ac `
  --stop-after test `
  --seed 7 `
  --device cuda
```

ただし、seed 11再学習で崩れたため、現時点では「再学習すれば常にPlan7結果が出る」とは言えない。

## Step 1: AC Plan 7再現評価

実行:

```powershell
uv run python -u -m unidream.cli.train `
  --config configs\trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,4,5 `
  --start-from test `
  --stop-after test `
  --seed 7 `
  --device cuda
```

結果:

| fold | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | PeriodWin | long | short | flat | turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.03 | -0.000 | -0.00 | 0.0% | 0% | 0% | 100% | 0.05 |
| 4 | +0.43 | +0.079 | -0.64 | 66.7% | 1% | 0% | 99% | 2.55 |
| 5 | +41.31 | +0.026 | -0.25 | 50.0% | 2% | 0% | 98% | 1.78 |

Aggregate:

```text
AlphaEx:   +13.91 pt/yr
SharpeΔ:   +0.035
MaxDDΔ:    -0.30 pt
BarWin:    11.0%
PeriodWin: 38.9%
PBO:       0.3333
```

これはAC Plan 7の採用結果と一致。

## Step 2: adapter attribution

新しく出した `adapter_detail`。

| fold | fire | mean_delta | positive | long_state | fire_pnl | nonfire_pnl |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0% | +0.0000 | 0.0% | 7.6% | +0.0000 | +0.2568 |
| 4 | 1.5% | +0.1068 | 75.9% | 50.8% | +0.0759 | -0.7128 |
| 5 | 5.2% | +0.0626 | 96.3% | 14.1% | +0.0887 | +0.5700 |

読み:

```text
adapterは高頻度には発火していない。
fold4/fold5ではfire時PnLは正。
ただしfold4はnon-fire区間が大きく負で、adapterだけで全体を救っているわけではない。
fold5の改善寄与が大きく、fold依存は残る。
```

## Step 3: selector grid改善案

試した設定:

```text
val_adjust_rate_scale: 0.25 / 0.5 / 0.75 / 1.0
near_best_tiebreak: balanced
turnover target: 3.5
long cap: 3%
short cap: 0%
period_win_bonusあり
```

選ばれたscale:

| fold | selected scale | test AlphaEx | test SharpeDelta | test MaxDDDelta |
|---:|---:|---:|---:|---:|
| 0 | 0.25 | -0.01 | -0.000 | -0.00 |
| 4 | 0.75 | +0.45 | +0.081 | -0.67 |
| 5 | 1.00 | +30.98 | +0.021 | -0.19 |

Aggregate:

```text
AlphaEx:   +10.47 pt/yr
SharpeΔ:   +0.034
MaxDDΔ:    -0.28 pt
PeriodWin: 38.9%
```

判断:

```text
不採用。
fold5でval上はscale=1.0が良く見えたが、testではPlan7固定0.5より悪化。
selectorはまだval overfitする。
現時点では scale=0.5 固定が安全。
```

## Step 4: cost / slippage stress

| setting | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | BarWin | PeriodWin | 判定 |
|---|---:|---:|---:|---:|---:|---|
| base | +13.91 | +0.035 | -0.30 | 11.0% | 38.9% | pass |
| cost x1.5 | +12.99 | +0.033 | -0.28 | 10.4% | 38.9% | 条件維持だがSharpeは薄い |
| cost x2.0 | +12.08 | +0.031 | -0.26 | 10.0% | 38.9% | 条件維持だが改善幅は縮小 |
| slippage x2.0 | +13.57 | +0.034 | -0.29 | 10.8% | 38.9% | ほぼ維持 |

読み:

```text
turnoverが低いため、slippage x2には比較的耐える。
cost x2でもAlphaExは正だが、SharpeDeltaは +0.031 まで落ちる。
もともとのSharpe改善が薄いので、cost stressで優位性はかなり細くなる。
```

## Step 5: seed 11 AC retrain

既存WM/BC checkpointをコピーし、ACだけseed 11で再学習した。

結果:

| fold | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | PeriodWin | long | short | turnover |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.04 | -0.000 | +0.00 | 0.0% | 0% | 0% | 0.06 |
| 4 | +0.45 | +0.081 | -0.69 | 33.3% | 1% | 0% | 2.55 |
| 5 | -4.22 | -0.048 | +0.74 | 25.0% | 3% | 0% | 1.61 |

Aggregate:

```text
AlphaEx:   -1.27 pt/yr
SharpeΔ:   +0.011
MaxDDΔ:    +0.02 pt
PeriodWin: 19.4%
```

判断:

```text
不採用。
fold5が崩れた。
short collapseはないが、overweight timingが悪化してB&H比で負ける。
```

重要:

```text
Plan7 checkpointの結果は良い。
しかし、同じロジックをseed違いで再学習するとまだ不安定。
```

## Step 6: guarded sizing adapter variant

seed 11崩壊対策として、fold5だけで以下を試した。

```text
delta_range: 0.03 -> 0.01
actor_lr: 1e-5 -> 5e-6
AC actor steps: 250 -> 150
td3bc_alpha: 3.0 -> 10.0
turnover_coef: 0.35 -> 0.75
long_rate_max: 0.03 -> 0.02
```

fold5結果:

```text
AlphaEx:   -23.36 pt/yr
SharpeΔ:   -0.065
MaxDDΔ:    +0.76 pt
turnover:  1.63
long:      2%
short:     0%
```

判断:

```text
不採用。
単にadapter幅を絞ってもfold5のtiming悪化は直らない。
むしろfire時PnLが負になった。
```

## 採用 / 不採用

採用維持:

```text
AC Plan 7 checkpoint
benchmark exposure floor = 1.0
benchmark-gated small overweight adapter
predictive advantage gate
scale=0.5 fixed selector
adapter attribution diagnostics
```

不採用:

```text
scale grid selector
seed 11 retrained AC
guarded sizing adapter variant
```

解除禁止を継続:

```text
full actor AC
route head unlock
advantage gate緩和
floor > 1.0 一律適用
Q argmax型AC
rank loss criticによるactor更新
```

## 次に必要なこと

現時点でやるべき次の作業:

```text
1. AC再学習のseed varianceを潰す。
2. fold5のoverweight timing崩れをadapter fire別に掘る。
3. selectorはgrid化せず、scale=0.5固定を維持。
4. 全14foldのPhase8/Plan7 checkpoint生成は別ジョブとして実行。
5. ACを広げるのはまだ禁止。
```

全14fold検証について:

```text
データ上は14foldある。
ただし現時点のcheckpointは fold 0 / 4 / 5 のみ。
全fold検証には WM/BC/AC checkpoint生成が必要で、今回のPlan8検証とは別の長時間ジョブとして扱うべき。
```

## 最終判断

```text
AC Plan 7は「採用済みcheckpointの評価」としては維持。
ただし、AC training recipeとしてはまだseed robustではない。
Plan8ではAC拡張ではなく、診断追加と不採用候補の切り分けまでを本流成果とする。
```

