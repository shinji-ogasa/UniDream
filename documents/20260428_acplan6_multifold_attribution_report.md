# AC Plan 6 Multi-fold Attribution Report

対象計画: `documents/acplan_6.md`

実行対象は、現時点で checkpoint が存在する fold 0 / 4 / 5 の全て。`checkpoints/bcplan5_phase8_state_machine_s007` にはこの3foldのみ存在するため、今回は「利用可能fold全部」の検証として扱った。

## 採用中の本流構成

```text
Phase 8 BC
+ predictive state input
+ benchmark-gated small overweight adapter
+ predictive advantage gate
+ benchmark exposure floor = 1.0
+ per-fold deterministic seed reset
```

今回追加した本流実装:

```text
- monthly/period B&H-relative win rate
- upside/downside capture ratio
- max underperformance streak
- selector_win_rate_metric = period
- floor / adapter / gate_off component attribution
- floor / adapter / advantage gate 発動バーの incremental PnL
```

コミット:

```text
aeef6c3 Add benchmark-relative attribution metrics
e16d5e5 Fix period win fallback
700e6eb Add gate attribution diagnostics
```

## 実行コマンド

```powershell
uv run python -u -m unidream.cli.train `
  --config configs\trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,4,5 `
  --start-from test `
  --seed 7 `
  --device cuda
```

`python -u` にしてログを即時flush。実行中は CUDA 使用を確認済み。GPU util はおおむね 37から41%、VRAM は約1.9から2.1GB。

## Phase A: 利用可能fold検証

| fold | test期間 | AlphaEx pt/yr | SharpeΔ | MaxDDΔ pt | PeriodWin | Long | Short | Flat | Turnover | Capture up/down |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2020-04-16 to 2020-07-16 | -0.02 | -0.000 | +0.00 | 0.0% | 0% | 0% | 100% | 0.03 | 1.000 / 1.000 |
| 4 | 2021-04-16 to 2021-07-16 | +0.41 | +0.077 | -0.64 | 33.3% | 1% | 0% | 99% | 2.54 | 1.006 / 1.005 |
| 5 | 2021-07-16 to 2021-10-16 | +40.04 | +0.025 | -0.25 | 50.0% | 2% | 0% | 98% | 1.77 | 1.004 / 1.004 |

Aggregate:

```text
AlphaEx:   +13.48 pt/yr
SharpeΔ:   +0.034
MaxDDΔ:    -0.30 pt
BarWin:    10.6%
PeriodWin: 27.8%
PBO:       0.3333
DSR t-stat:124.0194
```

判定:

```text
平均 AlphaEx > 0: pass
SharpeΔ >= 0: pass
turnover <= 3-4: pass
short collapseなし: pass
fold過半数でAlphaEx >= 0: pass寄り。fold0は -0.02pt で実質B&H近似。
M2: miss。PeriodWin 27.8% が弱い。
```

## Phase B: Component Attribution

### Fold 0

```text
neither:      alpha=-5.13pt   sharpeD=-0.000  maxddD=-0.29pt  turnover=0.43   long=0% short=0% flat=100%
adapter_only: alpha=-5.13pt   sharpeD=-0.000  maxddD=-0.29pt  turnover=0.43   long=0% short=0% flat=100%
floor_only:   alpha=-0.02pt   sharpeD=-0.000  maxddD=+0.00pt  turnover=0.03   long=0% short=0% flat=100%
gate_off:     alpha=-57.81pt  sharpeD=-0.403  maxddD=+0.04pt  turnover=104.60 long=3% short=0% flat=97%
current:      alpha=-0.02pt   sharpeD=-0.000  maxddD=+0.00pt  turnover=0.03   long=0% short=0% flat=100%

effects:
floor_effect=100.0% / +0.017 / pnl=+0.0046
adapter_effect=0.0% / 0.000 / pnl=+0.0000
adv_gate_effect=10.2% / 0.059 / pnl=+0.0302
```

### Fold 4

```text
neither:      alpha=+0.77pt  sharpeD=-0.023  maxddD=-1.49pt  turnover=13.33  long=0% short=17% flat=83%
adapter_only: alpha=+1.22pt  sharpeD=+0.061  maxddD=-2.15pt  turnover=13.69  long=1% short=17% flat=82%
floor_only:   alpha=+0.02pt  sharpeD=+0.014  maxddD=-0.03pt  turnover=1.94   long=0% short=0% flat=100%
gate_off:     alpha=-1.78pt  sharpeD=-0.239  maxddD=+2.26pt  turnover=104.60 long=3% short=0% flat=97%
current:      alpha=+0.41pt  sharpeD=+0.077  maxddD=-0.64pt  turnover=2.54   long=1% short=0% flat=99%

effects:
floor_effect=98.8% / +0.048 / pnl=-0.0247
adapter_effect=2.6% / 0.062 / pnl=+0.0129
adv_gate_effect=52.0% / 0.017 / pnl=+0.0691
```

### Fold 5

```text
neither:      alpha=-81.91pt   sharpeD=-0.006  maxddD=-0.47pt  turnover=1.01   long=0% short=0% flat=100%
adapter_only: alpha=-34.33pt   sharpeD=+0.020  maxddD=-0.72pt  turnover=2.76   long=2% short=0% flat=98%
floor_only:   alpha=-0.17pt    sharpeD=-0.000  maxddD=-0.00pt  turnover=0.09   long=0% short=0% flat=100%
gate_off:     alpha=-324.49pt  sharpeD=-0.430  maxddD=+1.22pt  turnover=105.80 long=3% short=0% flat=97%
current:      alpha=+40.04pt   sharpeD=+0.025  maxddD=-0.25pt  turnover=1.77   long=2% short=0% flat=98%

effects:
floor_effect=96.1% / +0.022 / pnl=+0.0139
adapter_effect=4.9% / 0.066 / pnl=+0.0075
adv_gate_effect=15.8% / 0.058 / pnl=+0.0520
```

## 解釈

### 1. benchmark exposure floor = 1.0 は本流維持

floorなしでは fold5 の upside miss が致命的。`neither alpha=-81.91pt` から `current alpha=+40.04pt` まで戻っている。

ただし、floor単体は alpha を作るというより、B&H未満に落ちない安全床。fold4では floor active PnL が -0.0247 なので、floor自体が常に収益源ではない。役割は alpha generator ではなく、under-benchmark collapse の遮断。

### 2. small overweight adapter は小さいが有効

adapterの発動率は低い。

```text
fold0: 0.0%
fold4: 2.6%, pnl +0.0129
fold5: 4.9%, pnl +0.0075
```

大きな改善源ではないが、fold4/fold5では正の incremental PnL。現状の `long <= 3%` cap の範囲では本流維持でよい。

### 3. predictive advantage gate は解除禁止

`gate_off` は全foldで明確に壊れた。

```text
fold0: AlphaEx -57.81pt, turnover 104.60
fold4: AlphaEx -1.78pt,  turnover 104.60
fold5: AlphaEx -324.49pt, turnover 105.80
```

これは acplan_6 の「advantage gate off は即死」という仮説と一致。ACに進む場合も gate緩和は禁止。少なくとも現状では hard safety 扱いにする。

### 4. selectorの period win 対応は必要だった

`period_win_rate_vs_bh=0.0` が `bar win` にフォールバックするバグを修正した。fold0 の Score 表示は正しく `periodwin=0.0%` になった。

ただし、Aggregate M2 はまだ MISS。

```text
barwin=10.6%
periodwin=27.8%
```

今のpolicyは「毎バー勝つ」型ではなく「upside missを消して大敗を避ける」型なので、M2判定では period/fold-level を見るべきだが、それでも fold0 と fold4 の period win はまだ弱い。

## AC移行判断

結論: 今すぐ full AC には進めない。restricted AC に進むなら、先に isolated overweight sizing adapter を実装する必要がある。

理由:

```text
- 現在の benchmark_overweight_* は主に inference rule / scalar config で、ACが安全にそこだけ学習できる trainable adapter ではない。
- 既存 actor の route/head/full actor を触ると、acplan_6 の「触らない」条件に反する。
- gate_off の結果から、advantage gate / safety を緩める方向は明確に危険。
```

したがって、次の安全な順番はこれ。

```text
1. trainable overweight sizing adapter を追加
2. trainable_actor_prefixes をその adapter だけに限定
3. benchmark floor / state machine / route gate / advantage gate は freeze
4. folds 0/4/5 で AC-0 critic-only
5. AC-1 sizing adapter only
6. turnover <= 3.5, long <= 3%, MaxDDΔ <= 0 を満たした場合だけ採用判断
```

## 採用 / 不採用

採用:

```text
- Phase 8 BC + predictive state
- benchmark exposure floor = 1.0
- benchmark-gated small overweight adapter
- predictive advantage gate
- period-level selector metric
- component attribution diagnostics
```

不採用または解除禁止:

```text
- gate off
- advantage gate threshold relaxation
- full actor AC
- route head unlock
- floor > 1.0 の一律適用
```

## 次にやること

```text
Phase D-0:
  isolated trainable overweight sizing adapter を実装する。

Phase D-1:
  AC critic-only で Q ranking / realized advantage correlation を確認する。

Phase D-2:
  actor側は sizing adapter だけ trainable にして、folds 0/4/5 で検証する。

採用条件:
  AlphaEx >= current + 0.2pt avg
  SharpeΔ >= current
  MaxDDΔ <= 0
  turnover <= 3.5
  long <= 3%
  short = 0%
```

今回の判断としては、現在の本流は available folds では壊れていない。ただし M2 はまだ MISS なので、ACを広げるより先に「ACが触ってよい小さな trainable sizing 部品」を作るのが必要。
