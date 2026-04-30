# TransformerWM DD Recovery Guard Probe Report

Date: 2026-04-30

## Purpose

AC Plan11 E/current has high AlphaEx, but fails deployable MaxDDDelta because the overweight/fire adapter remains active inside the worst drawdown interval.
This probe tested whether the existing TransformerWM predictive state can approximate the drawdown/recovery interval well enough to gate that adapter without future leakage.

## Calculation

Implemented an inference-only probe in `unidream.cli.ac_fire_dd_guard_probe`:

- Load the existing Plan10/Plan11 AC checkpoint: `checkpoints/acplan10_fire_selector_s011_on_wmbc_s011`.
- Use the existing TransformerWM predictive state heads: `return`, `vol`, `drawdown`.
- Build a sequential recovery-state guard:
  - Track realized equity and current drawdown online.
  - When adapter fire happens during deep drawdown, allow the adapter only if WM return score is high and WM drawdown-risk score is low.
  - Also test a slope version requiring positive trailing 16-bar realized return.
  - Also test a half-delta version that only applies 50% of the adapter delta.
- Compare against existing guards: fixed pre-DD thresholds, cooldown, predicted drawdown quantile, and oracle diagnostic guards.

Adoption screen for fold5 used the same Plan7-style gate:

- AlphaEx >= +41.31 pt/yr
- SharpeDelta >= +0.026
- MaxDDDelta <= 0
- turnover <= 3.5
- long <= 3%
- short = 0%

## Result

No WM recovery-state variant passed the adoption screen.

| variant | AlphaEx | SharpeD | MaxDDD | turnover | long | short | judgment |
|---|---:|---:|---:|---:|---:|---:|---|
| current | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | fails MaxDD |
| pred_dd_q75 | +68.68 | +0.062 | +0.04 | 3.91 | 1.1% | 0.0% | fails MaxDD/turnover |
| wm_recovery_dd22.50%_r60_d60 | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | same as current; fails MaxDD |
| wm_recovery_dd22.50%_r50_d50 | +54.45 | +0.042 | +0.04 | 1.48 | 1.9% | 0.0% | fails MaxDD |
| wm_recovery_half_dd23.00%_r50_d50 | +56.53 | +0.045 | +0.05 | 1.33 | 1.9% | 0.0% | fails MaxDD |
| wm_recovery_slope_dd22.50%_r50_d50 | +35.89 | +0.025 | -0.03 | 1.45 | 1.8% | 0.0% | passes MaxDD, loses Alpha/Sharpe |
| wm_recovery_slope_dd22.00%_r50_d50 | +29.62 | +0.021 | -0.03 | 1.44 | 1.8% | 0.0% | passes MaxDD, loses Alpha/Sharpe |
| cooldown_32 | +10.80 | +0.011 | -0.06 | 1.91 | 0.1% | 0.0% | too much suppression |
| oracle_mdd_interval | +64.34 | +0.058 | +0.00 | 0.68 | 1.0% | 0.0% | future-leak upper bound only |

Detailed fold result: `documents/20260430_acplan12_wm_recovery_guard_fold5_E.md`.

## Interpretation

The existing TransformerWM outputs are useful as generic predictive state features, but they are not precise enough to solve this specific DD-control problem.

What happened:

- WM permissive gates preserve AlphaEx, but do not remove the drawdown contribution. MaxDDDelta stays positive.
- WM slope/strict gates can fix MaxDDDelta, but suppress too much of the profitable adapter run. AlphaEx falls below the Plan7 fold5 baseline.
- The oracle interval still shows the problem is theoretically solvable: a correctly timed non-leaky DD interval/recovery selector can keep high AlphaEx while controlling MaxDD.

So the bottleneck is not the AC controller alone. The current WM target set lacks a direct notion of `fire is harmful inside upcoming DD` versus `fire is profitable recovery exposure`.

## Decision

Do not adopt the current WM recovery guard into the main path.

WM retraining or a new lightweight WM head is justified before another AC expansion. The next WM target should be route/control-specific, not just generic return/drawdown forecasting.

## Next WM Targets

Recommended additions:

- `recovery_prob_h16/h32`: probability that the next 16/32 bars are a recovery phase after current drawdown.
- `trough_exit_prob_h16/h32`: probability current region is after or near the drawdown trough, not before it.
- `fire_harm_prob_h16/h32`: probability that enabling overweight/fire increases worst future drawdown over the next horizon.
- `fire_advantage_h16/h32`: realized candidate advantage of fire adapter versus no-adapter, net of turnover/cost.
- `drawdown_worsening_prob_h16/h32`: probability that current drawdown deepens materially before recovery.

The key change is to train WM to predict the control decision boundary directly:

```text
allow overweight/fire only when expected fire advantage is positive
and drawdown worsening probability is low
and recovery/trough-exit probability is high
```

## Implementation Status

Added the screening implementation to `unidream/cli/ac_fire_dd_guard_probe.py`:

- `wm_recovery_*` variants
- `wm_recovery_slope_*` variants
- `wm_recovery_half_*` variants

These are probe-only and not adopted into the production train/inference path because no deployable variant passed.
