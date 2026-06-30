# Policy Family Holdout Comparison

- period: `2024-01-16 13:45:00` to `2026-04-16 13:45:00`
- folds: `15, 16, 17, 18, 19, 20, 21, 22, 23`
- config: `configs/plan011_overlay_actor_v31_holdout.yaml`
- checkpoint dir: `checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007`
- seed/device: `7` / `cpu`
- selection: train fit + validation selection only; test is report-only
- benchmark: B&H exposure=1.0

## Summary (mean across 9 quarterly test folds)

| method | AlphaEx | MaxDDDelta | median AlphaEx | worst AlphaEx | DD improved | mean turnover |
|---|---:|---:|---:|---:|---:|---:|
| B&H | +0.00pt | +0.00pt | +0.00pt | +0.00pt | 0/9 | 0.00 |
| 単純アルゴリズム (causal vol-target) | -1.27pt | -1.66pt | -0.62pt | -14.79pt | 5/9 | 3.63 |
| ML (HistGradientBoosting) | -0.38pt | -0.36pt | -0.12pt | -1.81pt | 9/9 | 1.19 |
| WMのみ (position-utility allocator) | -1.63pt | -1.33pt | -0.48pt | -7.19pt | 9/9 | 2.99 |
| BCのみ (WM+BC, ACなし) | +0.13pt | +0.24pt | -0.12pt | -0.32pt | 0/9 | 0.44 |

## Fold Results

| fold | test period | method | AlphaEx | MaxDDDelta | turnover |
|---:|---|---|---:|---:|---:|
| 15 | 2024-01-16 13:45:00 to 2024-04-16 13:45:00 | 単純アルゴリズム (causal vol-target) | +3.97pt | -0.68pt | 7.60 |
| 15 | 2024-01-16 13:45:00 to 2024-04-16 13:45:00 | ML (HistGradientBoosting) | -0.79pt | -0.10pt | 1.23 |
| 15 | 2024-01-16 13:45:00 to 2024-04-16 13:45:00 | WMのみ (position-utility allocator) | -1.19pt | -0.41pt | 2.30 |
| 15 | 2024-01-16 13:45:00 to 2024-04-16 13:45:00 | BCのみ (WM+BC, ACなし) | +0.90pt | +0.21pt | 0.52 |
| 16 | 2024-04-16 13:45:00 to 2024-07-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -3.73pt | +2.10pt | 2.42 |
| 16 | 2024-04-16 13:45:00 to 2024-07-16 13:45:00 | ML (HistGradientBoosting) | +0.02pt | -0.19pt | 0.62 |
| 16 | 2024-04-16 13:45:00 to 2024-07-16 13:45:00 | WMのみ (position-utility allocator) | -0.32pt | -0.44pt | 4.07 |
| 16 | 2024-04-16 13:45:00 to 2024-07-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.12pt | +0.28pt | 0.41 |
| 17 | 2024-07-16 13:45:00 to 2024-10-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -3.25pt | -3.36pt | 1.33 |
| 17 | 2024-07-16 13:45:00 to 2024-10-16 13:45:00 | ML (HistGradientBoosting) | -0.12pt | -0.57pt | 1.08 |
| 17 | 2024-07-16 13:45:00 to 2024-10-16 13:45:00 | WMのみ (position-utility allocator) | -1.63pt | -2.27pt | 5.14 |
| 17 | 2024-07-16 13:45:00 to 2024-10-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.17pt | +0.21pt | 0.53 |
| 18 | 2024-10-16 13:45:00 to 2025-01-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -14.79pt | -3.15pt | 4.85 |
| 18 | 2024-10-16 13:45:00 to 2025-01-16 13:45:00 | ML (HistGradientBoosting) | -1.62pt | -0.22pt | 1.81 |
| 18 | 2024-10-16 13:45:00 to 2025-01-16 13:45:00 | WMのみ (position-utility allocator) | -7.19pt | -1.29pt | 5.16 |
| 18 | 2024-10-16 13:45:00 to 2025-01-16 13:45:00 | BCのみ (WM+BC, ACなし) | +0.74pt | +0.18pt | 0.45 |
| 19 | 2025-01-16 13:45:00 to 2025-04-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -0.62pt | +0.17pt | 1.30 |
| 19 | 2025-01-16 13:45:00 to 2025-04-16 13:45:00 | ML (HistGradientBoosting) | +0.17pt | -0.22pt | 0.80 |
| 19 | 2025-01-16 13:45:00 to 2025-04-16 13:45:00 | WMのみ (position-utility allocator) | +0.58pt | -1.09pt | 1.28 |
| 19 | 2025-01-16 13:45:00 to 2025-04-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.31pt | +0.35pt | 0.44 |
| 20 | 2025-04-16 13:45:00 to 2025-07-16 13:45:00 | 単純アルゴリズム (causal vol-target) | +2.61pt | +0.61pt | 1.21 |
| 20 | 2025-04-16 13:45:00 to 2025-07-16 13:45:00 | ML (HistGradientBoosting) | -1.81pt | -0.41pt | 1.88 |
| 20 | 2025-04-16 13:45:00 to 2025-07-16 13:45:00 | WMのみ (position-utility allocator) | -6.09pt | -1.79pt | 2.62 |
| 20 | 2025-04-16 13:45:00 to 2025-07-16 13:45:00 | BCのみ (WM+BC, ACなし) | +0.71pt | +0.19pt | 0.40 |
| 21 | 2025-07-16 13:45:00 to 2025-10-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -0.69pt | +1.36pt | 0.94 |
| 21 | 2025-07-16 13:45:00 to 2025-10-16 13:45:00 | ML (HistGradientBoosting) | -0.12pt | -0.07pt | 1.12 |
| 21 | 2025-07-16 13:45:00 to 2025-10-16 13:45:00 | WMのみ (position-utility allocator) | -0.48pt | -0.32pt | 1.32 |
| 21 | 2025-07-16 13:45:00 to 2025-10-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.04pt | +0.11pt | 0.50 |
| 22 | 2025-10-16 13:45:00 to 2026-01-16 13:45:00 | 単純アルゴリズム (causal vol-target) | +2.11pt | -4.34pt | 8.29 |
| 22 | 2025-10-16 13:45:00 to 2026-01-16 13:45:00 | ML (HistGradientBoosting) | +0.31pt | -0.79pt | 1.13 |
| 22 | 2025-10-16 13:45:00 to 2026-01-16 13:45:00 | WMのみ (position-utility allocator) | +0.28pt | -1.71pt | 2.49 |
| 22 | 2025-10-16 13:45:00 to 2026-01-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.20pt | +0.37pt | 0.34 |
| 23 | 2026-01-16 13:45:00 to 2026-04-16 13:45:00 | 単純アルゴリズム (causal vol-target) | +2.91pt | -7.65pt | 4.72 |
| 23 | 2026-01-16 13:45:00 to 2026-04-16 13:45:00 | ML (HistGradientBoosting) | +0.49pt | -0.70pt | 1.06 |
| 23 | 2026-01-16 13:45:00 to 2026-04-16 13:45:00 | WMのみ (position-utility allocator) | +1.42pt | -2.67pt | 2.56 |
| 23 | 2026-01-16 13:45:00 to 2026-04-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.32pt | +0.30pt | 0.41 |

## Definitions

- Simple: past returns only. Realized-vol target and execution parameters are selected on validation.
- ML: HistGradientBoosting learns the training-only oracle position from causal tabular features; execution is selected on validation.
- WM only: the Transformer WM position-utility head selects exposure directly; no actor, BC, or AC is used.
- BC only: the Transformer WM encoder/predictive state and BC actor are used; the AC checkpoint is never loaded.
- AlphaEx = strategy final total return minus B&H final total return. It is not annualized.
- MaxDDDelta = strategy absolute MaxDD minus B&H absolute MaxDD. Negative is improvement.
