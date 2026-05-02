# Plan 2 Exploration Board Report

Date: 2026-05-02

## Summary

`documents/plan_2.md` を参考に、BC/ACなしの低コスト探索ボードを実装し、fold0/4/5/6で検証した。

結論: **次に残す方向は `D_risk_sensitive + A_triple_barrier_down guard`**。

単体のD selectorは fold6 で微小悪化したが、triple-barrier downside guard を足すと fold6 が alpha+ / MaxDD改善に反転し、fold0/fold5はneutralのまま維持できた。

Best current probe:

```text
D_risk_sensitive_tbguard
folds: 0,4,5,6
Alpha mean: +0.114 pt/yr
Alpha worst: 0.000 pt/yr
MaxDD mean: -0.170 pt
MaxDD worst: 0.000 pt
SharpeDelta mean: +0.012
turnover max: 2.50
long max: 0.0%
```

これはまだ本流採用ではない。BC/ACなしのraw/engineered selector probeなので、次は inference-only guard/selector としてもう少し厳密に検証する段階。

## Implemented

追加CLI:

```powershell
uv run python -m unidream.cli.exploration_board_probe
```

追加ファイル:

```text
unidream/cli/exploration_board_probe.py
```

このCLIは以下を行う。

```text
1. WFO foldごとに raw + engineered state features を作る
2. BC/ACは使わず、future realized utility を教師に候補action selectorをprobeする
3. triple-barrier labels を複数horizon/barrierで評価する
4. D selector と triple-barrier downside guard の合成も検証する
5. validationで閾値を選び、test foldでBacktestする
6. fold0/4/5/6, f456, f045のaggregateを出す
```

最終実行:

```powershell
uv run python -u -m unidream.cli.exploration_board_probe `
  --config configs/trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,4,5,6 `
  --seed 7 `
  --output-json documents/20260502_plan2_exploration_board_f0456_v4.json `
  --output-md documents/20260502_plan2_exploration_board_f0456_v4.md
```

## Tested Lanes

| Lane | Variant | Decision |
|---|---|---|
| A | triple-barrier downside/upside labels | keep as weak guard only |
| B | safe small improvement | drop current form |
| D | risk-sensitive candidate selector | keep, but only with A guard |
| D+A | risk-sensitive + triple-barrier downside guard | best current direction |
| E | bootstrap uncertainty fallback | drop current ridge form |
| F | listwise candidate selector | drop unless event-throttled |
| G | simple volatility regime split | drop current form |

## Selector Aggregate

Full output:

```text
documents/20260502_plan2_exploration_board_f0456_v4.md
documents/20260502_plan2_exploration_board_f0456_v4.json
```

### All folds: 0/4/5/6

| variant | Alpha mean | Alpha worst | MaxDD mean | MaxDD worst | Sharpe mean | Turnover max | Pass rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| D_risk_sensitive | +0.100 | -0.024 | -0.162 | +0.014 | +0.011 | 2.50 | 0.25 |
| D_risk_sensitive_floor005 | +0.096 | -0.041 | -0.159 | +0.024 | +0.012 | 2.50 | 0.25 |
| D_risk_sensitive_floor010 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 |
| **D_risk_sensitive_tbguard** | **+0.114** | **0.000** | **-0.170** | **0.000** | **+0.012** | **2.50** | **0.50** |
| D_risk_sensitive_tbguard_floor005 | +0.096 | -0.041 | -0.159 | +0.024 | +0.012 | 2.50 | 0.25 |
| F_listwise | +0.370 | -0.049 | -0.435 | +0.028 | +0.030 | 42.00 | 0.00 |

Pass rate requires:

```text
alpha > 0
MaxDDDelta <= 0
turnover <= 3.5
long <= 3%
```

### f456

| variant | Alpha mean | Alpha worst | MaxDD mean | MaxDD worst | Turnover max | Pass rate |
|---|---:|---:|---:|---:|---:|---:|
| D_risk_sensitive | +0.133 | -0.024 | -0.216 | +0.014 | 2.50 | 0.33 |
| **D_risk_sensitive_tbguard** | **+0.152** | **0.000** | **-0.227** | **0.000** | **2.50** | **0.67** |
| F_listwise | +0.494 | -0.049 | -0.580 | +0.028 | 42.00 | 0.00 |

### f045

| variant | Alpha mean | Alpha worst | MaxDD mean | MaxDD worst | Turnover max | Pass rate |
|---|---:|---:|---:|---:|---:|---:|
| D_risk_sensitive | +0.141 | 0.000 | -0.220 | 0.000 | 2.50 | 0.33 |
| **D_risk_sensitive_tbguard** | **+0.141** | **0.000** | **-0.220** | **0.000** | **2.50** | **0.33** |
| F_listwise | +0.510 | 0.000 | -0.589 | 0.000 | 42.00 | 0.00 |

## Fold Detail: Best Candidate

`D_risk_sensitive_tbguard`:

| fold | Alpha pt/yr | MaxDDDelta pt | SharpeDelta | Turnover | Flat | Interpretation |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 0.000 | 0.000 | 0.000 | 0.00 | 100.0% | neutral |
| 4 | +0.423 | -0.661 | +0.049 | 2.50 | 99.9% | useful de-risk |
| 5 | 0.000 | 0.000 | 0.000 | 0.00 | 100.0% | neutral |
| 6 | +0.032 | -0.019 | -0.000 | 0.50 | 99.8% | previous bad fire fixed |

The important change is fold6.

Before guard:

```text
D_risk_sensitive fold6:
  Alpha -0.024
  MaxDD +0.014
```

After guard:

```text
D_risk_sensitive_tbguard fold6:
  Alpha +0.032
  MaxDD -0.019
```

So the triple-barrier downside guard did not just reduce activity; it changed the remaining fold6 action set in the right direction.

## Triple-Barrier Label Quality

Best downside labels by worst AUC:

| target | Density | AUC mean | AUC worst | False-active worst | Recall worst |
|---|---:|---:|---:|---:|---:|
| h32_k125_down | 0.177 | 0.603 | 0.599 | 0.256 | 0.031 |
| h64_k150_down | 0.133 | 0.625 | 0.576 | 0.257 | 0.044 |
| h64_k150_up_safe | 0.086 | 0.628 | 0.580 | 0.426 | 0.033 |

Interpretation:

```text
AUCは0.60前後で、単独teacherとしては弱い。
ただし、D selectorのdanger guardとしては有効だった。
```

This is the key distinction:

```text
A_downside as direct route label: not enough
A_downside as no-fire guard: useful
```

## Dropped Directions

### B_safe_small

```text
Alpha mean -0.006
MaxDD worst +0.014
```

Drop current form.

### E_bootstrap_uncertainty

```text
all folds no action
```

Drop ridge-bootstrap version. If uncertainty is revisited, use actual WM ensemble / predictive-head disagreement.

### F_listwise

```text
Alpha mean +0.370
MaxDD mean -0.435
turnover max 42.0
```

The ranking signal exists, but it is not deployable without a much stronger event throttle. Do not use as main selector yet.

### G_vol_regime_safe

```text
Alpha mean -0.059
Alpha worst -0.212
```

Simple volatility gating is not enough.

## Decision

Proceed direction:

```text
D_risk_sensitive candidate selector
+ A_h32_k125_down / h64_k150_down danger guard
+ sparse event throttle
```

Do not proceed direction:

```text
full AC unlock
route head unlock
raw listwise argmax
threshold-only tuning
simple vol regime split
```

## Next Round

The next experiment should convert this probe into an inference-only selector/guard and test stricter deployment constraints.

Required next tests:

```text
1. Use D_risk_sensitive_tbguard as a fixed selector
2. Add event throttle / cooldown so turnover cannot spike
3. Add hard no-short and long-only cap
4. Keep fallback to current safe baseline/benchmark
5. Re-run folds 0/4/5/6 and then broader folds if checkpoints/features allow
```

Target:

```text
Alpha mean > 0
Alpha worst >= 0
MaxDD worst <= 0
turnover <= 3.5
long <= 3%
short = 0
fold4 gain retained
fold6 non-worsening retained
```

Current probe already meets most of this, but it is not yet integrated as a production selector and still uses a simple ridge model over raw/engineered features. Treat it as a validated direction, not final adoption.
