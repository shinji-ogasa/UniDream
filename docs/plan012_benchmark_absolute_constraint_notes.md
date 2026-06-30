# Plan012 Benchmark-Absolute Constraint Notes

Plan012 is a working branch for final-value AlphaEx optimization:

- AlphaEx is evaluated as `strategy final total return - B&H final total return`.
- MaxDDDelta is `strategy abs(MaxDD) - B&H abs(MaxDD)`, so negative is improvement.
- WM reward training can use `world_model.reward_mode: absolute` while oracle/BC can remain benchmark-relative.
- AC reward can roll out a fixed B&H benchmark path from the same imagined state and optimize B&H-relative log wealth with relative drawdown penalties.
- The current probe uses an affine continuous WM overlay teacher instead of hard route classes.

## Current 3-Fold Probe

Config/checkpoint:

- `configs/plan012_benchmark_absolute_constraint_probe.yaml`
- `checkpoints/plan012_benchmark_absolute_constraint_probe_s007`
- folds: `0, 2, 8`
- seed/device: `7` / `cpu`

Direct AC checkpoint replay with validation-selected inference scale:

| actor | fold | AlphaEx | MaxDDDelta | note |
|---|---:|---:|---:|---|
| `ac.pt` | 0 | -1.63pt | -0.34pt | small de-risk, Alpha below target |
| `ac.pt` | 2 | -5.07pt | -0.33pt | bull-fold Alpha damage reduced but still negative |
| `ac.pt` | 8 | +0.36pt | -0.76pt | correct direction, not enough DD improvement |

Mean over folds 0/2/8: `AlphaEx -2.11pt`, `MaxDDDelta -0.47pt`.

BC-only/policy-family replay for the same checkpoint is stored in:

- `docs/plan012_probe_current_compare.md`
- `docs/plan012_probe_current_compare.json`

The earlier strong de-risk probe showed DD can be forced lower but destroys bull-fold Alpha:

| fold | AlphaEx | MaxDDDelta |
|---:|---:|---:|
| 0 | -2.51pt | -1.23pt |
| 2 | -38.10pt | -3.02pt |
| 8 | +2.95pt | -3.46pt |

Interpretation: WM risk/utility signal is present, but hard-gated underweight teachers over-de-risk and the current affine teacher is too weak. The next useful change is not more threshold searching; it is stronger alpha-protected AC optimization and a selector that rejects validation candidates which gain DD by sacrificing final-value AlphaEx.
