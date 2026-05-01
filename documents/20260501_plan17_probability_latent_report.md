# Plan17 Probability-Latent Reproducibility Report

WM eval latent: deterministic category probabilities (`dist.probs`).

## Robust Summary

| label | mean Alpha | median Alpha | worst Alpha | win folds | mean SharpeD | worst SharpeD | mean MaxDDD | worst MaxDDD | mean turnover | pass folds | mean fire | mean danger | mean safe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| det_retrain | +1.27 | +0.36 | +0.00 | 2/3 | +0.005 | -0.045 | -0.11 | +0.24 | 0.93 | 2/3 | 1.9% | 43.6% | 20.1% |
| plan7_old | -6.37 | -0.00 | -19.54 | 1/3 | +0.004 | -0.070 | -0.15 | +0.22 | 1.22 | 1/3 | 1.8% | 40.3% | 18.3% |

## Per Fold

| fold | label | Alpha | SharpeD | MaxDDD | turnover | long | short | fire | danger | safe | fire_pnl |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | det_retrain | +0.00 | +0.000 | +0.00 | 0.00 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 0 | plan7_old | -0.00 | -0.000 | +0.00 | 0.00 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | +0.0000 |
| 4 | det_retrain | +0.36 | +0.059 | -0.57 | 0.80 | 1.1% | 0.0% | 1.4% | 70.3% | 27.1% | +0.0277 |
| 4 | plan7_old | +0.45 | +0.080 | -0.67 | 1.72 | 1.1% | 0.0% | 1.5% | 61.2% | 24.8% | +0.0697 |
| 5 | det_retrain | +3.45 | -0.045 | +0.24 | 2.00 | 2.2% | 0.0% | 4.2% | 60.5% | 33.2% | -0.0051 |
| 5 | plan7_old | -19.54 | -0.070 | +0.22 | 1.95 | 2.2% | 0.0% | 4.0% | 59.6% | 30.2% | -0.0036 |

## Judgment

- Probability latent fixed the non-deterministic probe issue and avoids hard-mode complete collapse.
- It still does not produce reproducible alpha: median Alpha is near zero and worst fold remains around zero/negative.
- det_retrain fold5 creates some alpha but fails MaxDD; fold0/fold4 do not reproduce the effect.
- Current priority should move from AC/checkpoint selection to feature/teacher separability under deterministic latent, then retrain BC.