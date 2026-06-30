# Plan012 Benchmark-Absolute Constraint Notes

Plan012 is a working branch for final-value AlphaEx optimization:

- AlphaEx is evaluated as `strategy final total return - B&H final total return`.
- MaxDDDelta is `strategy abs(MaxDD) - B&H abs(MaxDD)`, so negative is improvement.
- WM reward training can use `world_model.reward_mode: absolute` while oracle/BC can remain benchmark-relative.
- AC reward can roll out a fixed B&H benchmark path from the same imagined state and optimize B&H-relative log wealth with relative drawdown penalties.

## Current 3-Fold Probe

Config/checkpoint:

- `configs/plan012_benchmark_absolute_constraint_probe.yaml`
- `checkpoints/plan012_benchmark_absolute_constraint_probe_s007`
- folds: `0, 2, 8`
- seed/device: `7` / `cpu`

Direct AC checkpoint replay with validation-selected inference scale:

| actor | fold | AlphaEx | MaxDDDelta | note |
|---|---:|---:|---:|---|
| `ac.pt` | 0 | -0.72pt | -0.45pt | mild DD improvement, Alpha below target |
| `ac.pt` | 2 | -16.17pt | -1.64pt | bull-market underweight still damages Alpha |
| `ac.pt` | 8 | +0.91pt | -1.34pt | correct direction, not enough magnitude |

Mean over folds 0/2/8: `AlphaEx -5.33pt`, `MaxDDDelta -1.14pt`.

The earlier strong de-risk probe showed DD can be forced lower but destroys bull-fold Alpha:

| fold | AlphaEx | MaxDDDelta |
|---:|---:|---:|
| 0 | -2.51pt | -1.23pt |
| 2 | -38.10pt | -3.02pt |
| 8 | +2.95pt | -3.46pt |

Interpretation: WM risk/utility signal is present, but simple hard-gated underweight teachers do not yet separate bull high-risk from true de-risk regimes well enough. Next work should focus on continuous risk-budget sizing and AC reward/selector alignment rather than more threshold-only gates.
