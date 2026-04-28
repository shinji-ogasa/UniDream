# AC Plan 5 Multi-Fold Generalization Report

Date: 2026-04-28
Config after adoption: `configs/trading.yaml`
Checkpoint dir: `checkpoints/bcplan5_phase8_state_machine_s007`
Device: CUDA
Seed: 7
Folds: 0, 4, 5

## Conclusion

AC Plan 5 found a multi-fold improvement path.

The failure mode was not the overweight adapter itself. The main issue was that fold0/fold5 were slightly under benchmark during strong upside periods, causing large annualized B&H-relative underperformance. Relaxing the predictive advantage gate did not generalize and mostly created churn.

Adopted into mainline:

- `benchmark_exposure_floor` at benchmark position `1.0`.
- Per-fold seed reset so fold results are independent of `--folds` order/list.

Not adopted:

- lower predictive advantage thresholds.
- `floor=1.02` or higher leverage floor.
- `floor=1.05` despite high fold5 Alpha, because fold4 turnover/MaxDD deteriorated.
- AC/full actor unlock.

## Mainline Changes

Commits:

- `9af434c Add benchmark exposure floor control`
- `082e17f Reset seed per WFO fold`
- `e4aaaed Adopt benchmark exposure floor`

Active config addition:

```yaml
ac:
  use_benchmark_exposure_floor: true
  benchmark_exposure_floor_position: 1.0
  benchmark_exposure_floor_advantage_index: -1
```

This is inference-only. It clamps predicted absolute position to at least benchmark when greedy positions are generated. AC training remains disabled in the current mainline because `ac.max_steps: 0`.

## Why Seed Reset Was Needed

Before this change, `--folds 5` and `--folds 0,4,5` could give different fold5 behavior because the folds were processed sequentially with one global RNG stream. That made old checkpoint evaluation order-dependent when some initialized actor components were not fully determined by loaded checkpoint state.

`run_wfo_folds` now resets the requested seed before each fold. This makes fold evaluation independent of fold order and makes multi-fold comparison usable.

## Baseline Before Plan 5

From AC Plan 4:

| fold | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short | flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -5.13 | -0.001 | -0.28 | 0.18 | 0% | 0% | 100% |
| 4 | +1.15 | +0.063 | -1.97 | 3.05 | 1% | 16% | 83% |
| 5 | -51.36 | -0.002 | -0.83 | 1.94 | 2% | 0% | 98% |
| avg | -18.45 | +0.020 | -1.03 | 1.72 | 1.0% | 5.3% | 93.7% |

Interpretation:

- fold4 was good.
- fold0/fold5 were too passive/under-benchmark.
- The average failed because fold5 B&H-relative upside miss was too large.

## Rejected Gate Calibration

Relaxing `benchmark_overweight_advantage_min` by train-fold quantile did not generalize.

Representative failures:

| fold | candidate | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | train q90 | -10.10 | -0.070 | -0.11 | 43.68 | 3.0% | 0.0% |
| 4 | train q95 | +0.95 | +0.089 | +0.05 | 21.85 | 2.2% | 9.9% |
| 5 | train q90 | -288.78 | -0.377 | +0.63 | 67.91 | 3.0% | 0.0% |

Reason:

- Lower thresholds increase adapter firing too much.
- The resulting path churns even when final long percentage remains small.
- This repeats the AC Plan 4 lesson: predictive advantage gate is mandatory and should not be loosened naively.

## Exposure Floor Candidate Search

The useful direction was not more adapter firing. It was preventing the model from being slightly below benchmark in bull/upside folds.

Shortlisted fold5 sizing results:

| candidate | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short | mean overlay |
|---|---:|---:|---:|---:|---:|---:|---:|
| current | -51.04 | -0.001 | -0.84 | 1.94 | 2.2% | 0.0% | -0.016 |
| floor 1.00 | +23.75 | +0.009 | -0.38 | 1.64 | 2.2% | 0.0% | +0.004 |
| floor 1.02 | +92.99 | +0.008 | +0.07 | 1.62 | 2.2% | 0.0% | +0.023 |
| floor 1.05 | +201.80 | +0.005 | +0.75 | 1.62 | 2.3% | 0.0% | +0.052 |

`floor=1.02` and `1.05` were rejected despite strong fold5 Alpha because MaxDD worsened and fold4 degraded.

Fold0/fold4 check:

| fold | candidate | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | floor 1.00 | -0.01 | -0.000 | +0.00 | 0.01 | 0.0% | 0.0% |
| 0 | floor 1.02 | +5.86 | -0.000 | +0.32 | 0.00 | 0.0% | 0.0% |
| 4 | floor 1.00 | +0.31 | +0.064 | -0.48 | 0.87 | 1.1% | 0.0% |
| 4 | floor 1.02 | -0.09 | +0.055 | +0.25 | 0.62 | 1.1% | 0.0% |
| 4 | floor 1.05 | -1.32 | -0.071 | +2.54 | 25.75 | 3.0% | 0.0% |

`floor=1.00` is the safest multi-fold compromise.

## Final CLI Verification After Adoption Logic

Executed with seed reset and temporary floor config, then adopted into `configs/trading.yaml`:

```powershell
uv run python -m unidream.cli.train --config configs\acplan5_floor_tmp.yaml --start 2018-01-01 --end 2024-01-01 --folds 0,4,5 --start-from test --seed 7 --device cuda
```

Final results:

| fold | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short | flat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.02 | -0.000 | +0.00 | 0.03 | 0% | 0% | 100% |
| 4 | +0.41 | +0.077 | -0.64 | 2.54 | 1% | 0% | 99% |
| 5 | +40.04 | +0.025 | -0.25 | 1.77 | 2% | 0% | 98% |
| avg | +13.48 | +0.034 | -0.30 | 1.45 | 1.0% | 0.0% | 99.0% |

Compared with Plan 4 average:

| setup | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | short |
|---|---:|---:|---:|---:|---:|
| Plan 4 current | -18.45 | +0.020 | -1.03 | 1.72 | 5.3% |
| Plan 5 adopted floor | +13.48 | +0.034 | -0.30 | 1.45 | 0.0% |
| delta | +31.93 | +0.014 | +0.73 | -0.27 | -5.3pp |

## Interpretation

This is a real multi-fold improvement, but it changes the policy character.

What improved:

- Multi-fold AlphaEx turned positive.
- Fold5 upside miss was fixed.
- Turnover stayed low.
- Short/underweight collapse was eliminated.
- SharpeD stayed positive on average.

Tradeoff:

- Downside protection is weaker than the original Phase 8 on fold4 because under-benchmark exposure is now floored at benchmark.
- Average MaxDDD remains negative, but less negative than Plan 4.
- The policy is now closer to `B&H + small gated overweight` than `B&H-relative de-risk/underweight`.

Given the current evidence, this is the correct mainline direction because the previous multi-fold failure was dominated by upside miss, not drawdown failure.

## Current Status

Adopted:

- Phase 8 BC.
- predictive state input.
- benchmark-gated small overweight adapter.
- predictive advantage gate.
- benchmark exposure floor at 1.0.
- per-fold deterministic seed reset.

Still blocked:

- AC full actor unlock.
- route head full unlock.
- trainable adapter.
- lower advantage threshold.
- leveraged exposure floor above 1.0.

Next recommended step:

1. Run the adopted config on more folds, preferably all available WFO folds with `--start-from test` where checkpoints exist or after generating missing checkpoints.
2. If average AlphaEx remains positive across more folds, then revisit restricted AC only around overweight sizing, not route/de-risk.
3. Add a stricter selector objective for benchmark-relative win rate, because current M2 still reports miss due low bar-level win rate even when AlphaEx improves.
