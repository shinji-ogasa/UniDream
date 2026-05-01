# Reproducibility-First Route Diagnostics

Date: 2026-05-01
Scope: multi-fold reproducibility check after rejecting mean-Alpha-only selection.

## Decision

Current decision:

```text
Prioritize multi-fold reproducibility over best mean Alpha.
Do not proceed to AC expansion yet.
Do not adopt two-stage route v1/v2.
Keep route diagnostics in the main codebase.
Redesign the route teacher/label before more AC work.
```

## Why

The latest results show that current candidates are either:

```text
1. neutral/benchmark-floor collapse, or
2. fold-dependent false-active overfire, or
3. weak single-fold/fold5-driven gains.
```

A model that only looks good because fold5 is strong is not acceptable as the current best direction.

## Route Separability Result

Command family:

```text
python -m unidream.cli.route_separability_probe
```

3fold active/no-active separability on folds 0,4,5:

| feature set | mean test AUC | worst test AUC | mean recall | worst recall | worst false-active |
|---|---:|---:|---:|---:|---:|
| raw | 0.519 | 0.498 | 0.213 | 0.070 | 0.455 |
| wm | 0.524 | 0.518 | 0.173 | 0.099 | 0.293 |
| wm_context | 0.717 | 0.670 | 0.435 | 0.423 | 0.148 |
| raw_wm_context | 0.716 | 0.673 | 0.432 | 0.419 | 0.148 |

Read:

```text
Raw market features alone are not enough.
WM latent alone is not enough.
Adding context makes active/no-active separable enough for a diagnostic model.
```

## Context Ablation

Ablation result:

| feature set | mean test AUC | worst test AUC | mean recall | worst recall | worst false-active |
|---|---:|---:|---:|---:|---:|
| context | 0.738 | 0.694 | 0.434 | 0.343 | 0.137 |
| wm_position | 0.715 | 0.669 | 0.429 | 0.415 | 0.143 |
| wm_position_advantage | 0.714 | 0.666 | 0.427 | 0.406 | 0.145 |
| wm_advantage | 0.525 | 0.516 | 0.167 | 0.098 | 0.273 |
| wm_regime | 0.526 | 0.516 | 0.168 | 0.098 | 0.268 |

Interpretation:

```text
The separability improvement comes mostly from current position/inventory state.
Predictive advantage alone is almost random.
Regime alone is almost random.
```

This is the main finding.

## Route Label Problem

Current route labels are too dependent on the teacher inventory path.

```text
The model is not mainly learning: market state -> useful route.
It is mostly learning: teacher position state -> route label.
```

That creates a distribution mismatch:

```text
BC training/probe uses teacher-forced current positions.
Inference uses the model's own inventory path.
Small mistakes alter inventory state.
Route decisions then move out of the teacher-state distribution.
```

This explains why route training swings between neutral collapse and false-active overfire.

## Two-Stage Route Experiments

Two-stage route head already existed in code but was unused because current config uses `route_dim: 3`.
I tested two variants with `route_dim: 4` and `use_two_stage_route_head: true`.
The temporary configs were not adopted and were removed.

### v1

Route probe:

```text
active recall: 0.0 on folds 0/4/5 test
false-active: 0.0
active AUC test: fold0 0.622, fold4 0.545, fold5 0.502
```

BC-only test:

```text
fold0 AlphaEx +0.00
fold4 AlphaEx +0.31
fold5 AlphaEx +1.59
aggregate AlphaEx +0.63
```

Read:

```text
Safe, but it escapes to benchmark floor. Not an adopted improvement.
```

### v2

Route probe:

```text
fold0 test active recall 0.027, false-active 0.010, active AUC 0.628
fold4 test active recall 0.013, false-active 0.007, active AUC 0.568
fold5 test active recall 0.013, false-active 0.010, active AUC 0.522
```

BC-only test:

```text
fold0 AlphaEx +0.00
fold4 AlphaEx +0.32
fold5 AlphaEx +1.81
aggregate AlphaEx +0.71
```

Read:

```text
Still too inactive. The route head ranking remains weak, especially fold5.
Do not adopt.
```

## Current Engineering Decision

Adopt in mainline:

```text
route_separability_probe diagnostic CLI
route_probe active score AUC/AP diagnostics
deterministic WM probability latent eval/inference
```

Reject:

```text
two-stage route v1/v2 configs
route rebalance strong/mild
Plan17 fire selector v2 as adoption candidate
mean-Alpha-only model selection
```

Hold:

```text
AC expansion
full actor unlock
route head unlock
```

## Next Reproducibility-First Work

The next useful step is not more threshold tuning.

Required next design:

```text
1. Build a market-state active/fire label that is not dominated by teacher current position.
2. Train active/no-active gate first, route type second.
3. Use current position only as controller state, not as the primary route label shortcut.
4. Require 3fold active AUC/worst false-active stability before BC backtest.
5. Only after BC is stable across folds, return to restricted AC.
```

Recommended acceptance gate before AC:

```text
3fold active/no-active test AUC worst >= 0.65
3fold false-active worst <= 0.15 under val-selected threshold
actual BC route active recall not zero on all folds
BC test worst AlphaEx >= 0.0
BC test worst MaxDDDelta <= +0.25pt
turnover <= 3.5
```

## Benchmark-State Label Probe

I also tested `--label-mode benchmark`, which removes the teacher-inventory shortcut by computing route labels from a constant benchmark position.

Result:

| feature set | mean test AUC | worst test AUC | mean recall | worst false-active |
|---|---:|---:|---:|---:|
| raw | 0.527 | 0.502 | 0.202 | 0.412 |
| wm | 0.525 | 0.512 | 0.161 | 0.255 |
| context | 0.516 | 0.488 | 0.118 | 0.191 |
| wm_context | 0.525 | 0.514 | 0.158 | 0.250 |

Read:

```text
Removing the inventory shortcut does not reveal a learnable market-state active label.
The simple benchmark-state forward route label is close to random across folds.
```

Updated conclusion:

```text
The current teacher-forced route label is learnable mostly because it leaks inventory state.
The simple benchmark-state replacement is not learnable from current raw/WM features.
Therefore, route teacher redesign must build a cleaner market-state label, not just replace current_positions with benchmark_position.
```

Next label candidates to test:

```text
future drawdown-window entry label
risk-off event label with minimum post-event avoided drawdown
trend continuation overweight label with volatility and drawdown filters
separate de-risk event labels from overweight event labels before combining into route
```
