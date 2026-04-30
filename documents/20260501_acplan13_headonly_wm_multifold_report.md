# AC Plan13: Head-only WM Control Multi-fold Validation

Run completed: 2026-05-01 JST

## Purpose

Validate `acplan_12.md` Roadmap B/C:

- keep the Plan7-style mainline intact;
- test head-only TransformerWM control heads on folds 4/5/6;
- sweep only the narrow `benchmark_overweight_advantage_min` / `benchmark_overweight_epsilon` inference controls;
- adopt only if the multi-fold constraints hold.

## Implementation Change

Added fold-matched WM initialization support:

```yaml
world_model:
  init_checkpoint: checkpoints/acplan13_base_wm_s011/fold_{fold}/world_model.pt
```

`prepare_world_model_stage` now expands `{fold}` / `{fold_idx}` before loading the init checkpoint. This avoids the previous fold5-only checkpoint leakage into other folds.

Created a consistent fold-specific base WM config:

```text
configs/trading_wm_base_s011.yaml
```

Then generated base WM checkpoints for folds 4/5/6:

```text
checkpoints/acplan13_base_wm_s011/fold_4/world_model.pt
checkpoints/acplan13_base_wm_s011/fold_5/world_model.pt
checkpoints/acplan13_base_wm_s011/fold_6/world_model.pt
```

## Main Candidate Test

Config:

```text
configs/trading_wm_control_headonly.yaml
```

Candidate parameters:

```yaml
benchmark_overweight_advantage_min: 0.55
benchmark_overweight_epsilon: 0.06
benchmark_overweight_trainable_delta_range: 0.03
```

Fold results:

| fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | fire | mean_delta | judgment |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | +0.19 | +0.034 | +0.09 | 4.41 | 2% | 0% | 12.0% | +0.0260 | fails MaxDD/turnover |
| 5 | -15.09 | -0.042 | +0.46 | 2.11 | 2% | 0% | 10.3% | +0.0292 | fails Alpha/Sharpe/MaxDD |
| 6 | -0.15 | +0.011 | +0.41 | 4.44 | 1% | 0% | 28.0% | +0.0194 | fails Alpha/MaxDD/turnover |
| mean | -5.02 | +0.001 | +0.32 | 3.65 | n/a | 0% | n/a | n/a | FAIL |

Adoption condition was not met:

```text
3fold mean AlphaEx > 0
SharpeDelta >= 0
MaxDDDelta <= 0
turnover <= 3.5
long <= 3%
short = 0%
```

The main failure is not short collapse. It is weak or negative Alpha with positive MaxDDDelta. Fold5 is especially important: the earlier fold5-only success did not reproduce when the base WM was regenerated consistently per fold.

## Narrow Inference Sweep

After training, only inference controls were swept with `--resume --start-from test`.

| tag | adv_min | eps | mean AlphaEx | mean SharpeD | mean MaxDDD | mean turnover | worst AlphaEx | worst MaxDDD | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| adv0p55_eps0p04 | 0.550 | 0.04 | -1.50 | +0.004 | +0.17 | 2.98 | -4.61 | +0.40 | FAIL |
| adv0p55_eps0p05 | 0.550 | 0.05 | -1.09 | +0.002 | +0.22 | 3.65 | -3.38 | +0.50 | FAIL |
| adv0p575_eps0p04 | 0.575 | 0.04 | -1.04 | +0.003 | +0.11 | 2.46 | -3.03 | +0.29 | FAIL |
| adv0p575_eps0p05 | 0.575 | 0.05 | -0.52 | -0.000 | +0.14 | 3.03 | -1.40 | +0.36 | FAIL |
| adv0p6_eps0p04 | 0.600 | 0.04 | -2.49 | -0.002 | +0.23 | 1.73 | -7.34 | +0.45 | FAIL |
| adv0p6_eps0p05 | 0.600 | 0.05 | -2.64 | +0.003 | +0.11 | 2.08 | -8.06 | +0.36 | FAIL |
| adv0p65_eps0p04 | 0.650 | 0.04 | -1.07 | +0.014 | -0.09 | 0.96 | -3.47 | +0.11 | FAIL |

Best safety-like point:

```text
adv_min=0.65, eps=0.04
mean AlphaEx -1.07
mean SharpeD +0.014
mean MaxDDD -0.09
mean turnover 0.96
```

This controls drawdown and turnover, but AlphaEx remains negative and fold5 still fails.

Best Alpha-like points still failed MaxDD or fold-level Alpha:

```text
adv_min=0.575, eps=0.05: mean AlphaEx -0.52, mean MaxDDD +0.14
adv_min=0.55,  eps=0.05: mean AlphaEx -1.09, mean MaxDDD +0.22
```

## Interpretation

The fold5-only head-only WM result was not robust.

The likely reason is that the previous fold5 result depended on the specific `acplan10_fire_selector_s011_on_wmbc_s011/fold_5` WM/BC representation. Once a consistent per-fold base WM is regenerated, the control heads do not learn a stable multi-fold fire boundary.

Important observations:

- Full WM retraining remains rejected.
- Head-only WM control heads are directionally interesting but not enough for adoption.
- The current `overweight_advantage/recovery` labels are too indirect.
- The model often fires in places where the adapter's standalone fire PnL is positive, but total policy Alpha and MaxDD still fail. That means the fire boundary is not aligned with portfolio-level drawdown control.

## Decision

Do not adopt head-only WM control into `configs/trading.yaml`.

Keep the implementation support because it is useful for the next label experiment, but treat current Plan13 candidate as failed.

Mainline remains:

```text
Plan7 / Plan5-style stable baseline
benchmark exposure floor = 1.0
small gated overweight adapter
restricted sizing-adapter AC only
```

## Next Technical Step

Do not continue AC expansion.

The next WM work should replace indirect labels with direct control labels:

```text
fire_harm_prob_h16/h32
trough_exit_prob_h16/h32
drawdown_worsening_prob_h16/h32
fire_advantage_h16/h32
```

The target should be:

```text
allow fire only when:
  fire advantage is positive
  drawdown worsening probability is low
  trough-exit/recovery probability is high
```

Without those direct labels, tuning `adv_min/epsilon` just trades off between no-fire benchmark behavior and noisy fire timing.

## Artifacts

Configs:

- `configs/trading_wm_base_s011.yaml`
- `configs/trading_wm_control_headonly.yaml`

Logs:

- `documents/logs/acplan13_base_wm_folds456.out.log`
- `documents/logs/acplan13_headonly_folds456.out.log`
- `documents/acplan13_headonly_sweep_summary.txt`
- `documents/logs/acplan13_sweep_*.log`
