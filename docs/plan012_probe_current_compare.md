# Policy Family Holdout Comparison

- period: `2020-04-16 13:45:00` to `2022-07-16 13:45:00`
- folds: `0, 2, 8`
- config: `configs/plan012_benchmark_absolute_constraint_probe.yaml`
- checkpoint dir: `checkpoints/plan012_benchmark_absolute_constraint_probe_s007`
- seed/device: `7` / `cpu`
- selection: train fit + validation selection only; test is report-only
- benchmark: B&H exposure=1.0

## Summary (mean across 3 quarterly test folds)

| method | AlphaEx | MaxDDDelta | median AlphaEx | worst AlphaEx | DD improved | mean turnover |
|---|---:|---:|---:|---:|---:|---:|
| B&H | +0.00pt | +0.00pt | +0.00pt | +0.00pt | 0/3 | 0.00 |
| 単純アルゴリズム (causal vol-target) | -30.94pt | -5.57pt | -3.32pt | -89.52pt | 3/3 | 8.30 |
| ML (HistGradientBoosting) | -0.94pt | -0.68pt | -1.28pt | -2.23pt | 3/3 | 1.61 |
| WMのみ (position-utility allocator) | -36.24pt | -14.07pt | -10.71pt | -117.80pt | 3/3 | 3.83 |
| BCのみ (WM+BC, ACなし) | -5.33pt | -1.14pt | -0.72pt | -16.17pt | 3/3 | 3.10 |

## Fold Results

| fold | test period | method | AlphaEx | MaxDDDelta | turnover |
|---:|---|---|---:|---:|---:|
| 0 | 2020-04-16 13:45:00 to 2020-07-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -3.32pt | -3.78pt | 12.97 |
| 0 | 2020-04-16 13:45:00 to 2020-07-16 13:45:00 | ML (HistGradientBoosting) | -1.28pt | -0.96pt | 2.25 |
| 0 | 2020-04-16 13:45:00 to 2020-07-16 13:45:00 | WMのみ (position-utility allocator) | -10.71pt | -8.21pt | 2.33 |
| 0 | 2020-04-16 13:45:00 to 2020-07-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.72pt | -0.45pt | 2.85 |
| 2 | 2020-10-16 13:45:00 to 2021-01-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -89.52pt | -12.27pt | 10.77 |
| 2 | 2020-10-16 13:45:00 to 2021-01-16 13:45:00 | ML (HistGradientBoosting) | -2.23pt | -0.26pt | 1.52 |
| 2 | 2020-10-16 13:45:00 to 2021-01-16 13:45:00 | WMのみ (position-utility allocator) | -117.80pt | -11.88pt | 1.31 |
| 2 | 2020-10-16 13:45:00 to 2021-01-16 13:45:00 | BCのみ (WM+BC, ACなし) | -16.17pt | -1.64pt | 3.02 |
| 8 | 2022-04-16 13:45:00 to 2022-07-16 13:45:00 | 単純アルゴリズム (causal vol-target) | +0.03pt | -0.66pt | 1.16 |
| 8 | 2022-04-16 13:45:00 to 2022-07-16 13:45:00 | ML (HistGradientBoosting) | +0.69pt | -0.83pt | 1.07 |
| 8 | 2022-04-16 13:45:00 to 2022-07-16 13:45:00 | WMのみ (position-utility allocator) | +19.78pt | -22.12pt | 7.84 |
| 8 | 2022-04-16 13:45:00 to 2022-07-16 13:45:00 | BCのみ (WM+BC, ACなし) | +0.91pt | -1.34pt | 3.41 |

## Definitions

- Simple: past returns only. Realized-vol target and execution parameters are selected on validation.
- ML: HistGradientBoosting learns the training-only oracle position from causal tabular features; execution is selected on validation.
- WM only: the Transformer WM position-utility head selects exposure directly; no actor, BC, or AC is used.
- BC only: the Transformer WM encoder/predictive state and BC actor are used; the AC checkpoint is never loaded.
- AlphaEx = strategy final total return minus B&H final total return. It is not annualized.
- MaxDDDelta = strategy absolute MaxDD minus B&H absolute MaxDD. Negative is improvement.
