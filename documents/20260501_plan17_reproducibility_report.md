# Plan17 Reproducibility Re-evaluation

Deterministic WM eval latent after `ObsEncoder.forward` now uses `mode()` in eval mode.

## Robust Summary

| label | mean Alpha | median Alpha | worst Alpha | win folds | mean SharpeD | worst SharpeD | mean MaxDDD | worst MaxDDD | mean turnover | pass folds | mean fire | mean danger | mean safe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plan17 | +0.07 | -0.00 | -0.22 | 1/3 | +0.007 | -0.000 | +0.16 | +0.48 | 1.23 | 1/3 | 0.0% | 0.0% | 0.0% |
| plan7 | +0.07 | -0.00 | -0.22 | 1/3 | +0.007 | -0.000 | +0.16 | +0.48 | 1.23 | 1/3 | 0.0% | 0.0% | 0.0% |
| plan8 | +0.07 | -0.00 | -0.22 | 1/3 | +0.007 | -0.000 | +0.16 | +0.48 | 1.23 | 1/3 | 0.0% | 0.0% | 0.0% |

## Per Fold

| fold | label | Alpha | SharpeD | MaxDDD | turnover | long | short | fire | danger | safe | fire_pnl |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plan17 | -0.00 | -0.000 | -0.00 | 0.00 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 0 | plan7 | -0.00 | -0.000 | -0.00 | 0.00 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 0 | plan8 | -0.00 | -0.000 | -0.00 | 0.00 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 4 | plan17 | -0.22 | +0.020 | +0.48 | 3.46 | 1.1% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 4 | plan7 | -0.22 | +0.020 | +0.48 | 3.46 | 1.1% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 4 | plan8 | -0.22 | +0.020 | +0.48 | 3.46 | 1.1% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 5 | plan17 | +0.44 | +0.000 | +0.00 | 0.22 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 5 | plan7 | +0.44 | +0.000 | +0.00 | 0.22 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 5 | plan8 | +0.44 | +0.000 | +0.00 | 0.22 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |

## Judgment

- No candidate shows robust positive alpha across all 3 folds.
- Previous high fold5 alpha was not reproducible under deterministic WM evaluation and should not be used as adoption evidence.
- For current evidence, reproducibility-first selection should prefer the simplest deterministic safe baseline, not Plan17 fire selector v2.
- Next optimization should target fold-level win rate / worst-fold alpha, not mean alpha.