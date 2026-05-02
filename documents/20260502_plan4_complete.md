# Plan 4 Complete Verification Report

Checkpoint: `checkpoints/acplan13_base_wm_s011` | Device: `cuda` | Folds: `4, 5, 6`

---

## Round A: WM Prediction Calibration

See `documents/20260502_plan4_wm_calibration.md` for full calibration.

**Key finding**: return head random (IC~0), vol head useful (IC 0.3-0.6), dd head weak/random.

---

## Round C: Blocked Event Attribution

| fold | bt_alpha | bt_maxdd | n_blocked_events | danger_blocked | pullback_blocked |
|---|---:|---:|---:|---:|---:|
| 4 | 0.000 | 0.000 | 700 | 468 | 36 |
| 5 | 0.000 | 0.000 | 700 | 495 | 14 |
| 6 | 0.000 | 0.000 | 700 | 484 | 46 |

### Counterfactual: blocked event actual utility

| fold | mean_actual_util | util>0_rate | would_improve |
|---|---:|---:|---:|
| 4 | -0.005 | 0.100 | NO |
| 5 | -0.002 | 0.160 | NO |
| 6 | -0.003 | 0.153 | NO |

---

## Round B: Ridge + WM Ensemble

| fold | variant | AlphaEx | MaxDDΔ | SharpeΔ | turnover | flat |
|---|---:|---:|---:|---:|---:|
| 4 | OR | -1.862 | 1.703 | -0.216 | 80.000 | 0.639 |
| 4 | AND | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 4 | MAX | 1.029 | -2.916 | 0.000 | 177.600 | 0.737 |
| 5 | OR | -1.961 | 0.028 | -0.002 | 0.200 | 1.000 |
| 5 | AND | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 5 | MAX | 1.989 | -0.029 | 0.002 | 0.500 | 1.000 |
| 6 | OR | 0.036 | -0.021 | 0.002 | 0.200 | 1.000 |
| 6 | AND | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 6 | MAX | -0.024 | 0.014 | -0.003 | 0.500 | 0.998 |

---

## Round D: Soft Throttle Guard

| fold | variant | AlphaEx | MaxDDΔ | SharpeΔ | turnover | flat |
|---|---:|---:|---:|---:|---:|
| 4 | WM_hard | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 4 | danger_scale_0.25 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 4 | danger_scale_0.5 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 4 | danger_scale_0.75 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 5 | WM_hard | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 5 | danger_scale_0.25 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 5 | danger_scale_0.5 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 5 | danger_scale_0.75 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 6 | WM_hard | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 6 | danger_scale_0.25 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 6 | danger_scale_0.5 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| 6 | danger_scale_0.75 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

---

## Round E: Utility Parameter Grid (top results per fold)

| fold | params | AlphaEx | MaxDDΔ | SharpeΔ | turnover |
|---|---:|---:|---:|---:|
| 4 | dd0.25_vol0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | dd0.25_vol0.5 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | dd0.5_vol0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | dd0.5_vol0.5 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | dd1.0_vol0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | dd1.0_vol1.0 | 6.312 | -0.091 | 0.006 | 0.400 |
| 5 | dd2.0_vol0.0 | 6.312 | -0.091 | 0.006 | 0.400 |
| 5 | dd0.25_vol0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | dd0.25_vol0.5 | 0.000 | 0.000 | 0.000 | 0.000 |
| 5 | dd0.25_vol1.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | dd0.25_vol0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | dd0.25_vol0.5 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | dd0.25_vol1.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | dd0.5_vol0.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 6 | dd0.5_vol0.5 | 0.000 | 0.000 | 0.000 | 0.000 |
