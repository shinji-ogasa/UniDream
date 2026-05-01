# Route Separability Probe

Config: `configs/acplan17_det_retrain_s007_tmp.yaml`
Checkpoint dir: `checkpoints/acplan17_det_retrain_s007`
Folds: `0, 4, 5`

## Aggregate Active/No-Active Separability

| feature set | folds | test AUC mean | test AUC worst | test AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 3 | 0.519 | 0.498 | 0.508 | 0.213 | 0.070 | 0.197 | 0.455 | 0.203 |
| raw_wm_context | 3 | 0.716 | 0.673 | 0.738 | 0.432 | 0.419 | 0.104 | 0.148 | 0.265 |
| wm | 3 | 0.524 | 0.518 | 0.508 | 0.173 | 0.099 | 0.151 | 0.293 | 0.162 |
| wm_context | 3 | 0.717 | 0.670 | 0.740 | 0.435 | 0.423 | 0.102 | 0.148 | 0.265 |

## Fold Detail

Thresholds are selected on validation with the configured false-active and predicted-active caps, then applied to test.

### Fold 0

| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.483 | 0.558 | 0.451 | 0.112 | 0.067 | 0.085 | 0.245 | 0.621 |
| wm | 0.503 | 0.537 | 0.427 | 0.099 | 0.073 | 0.083 | 0.244 | 0.661 |
| wm_context | 0.654 | 0.755 | 0.700 | 0.430 | 0.062 | 0.207 | 0.351 | 0.469 |
| raw_wm_context | 0.650 | 0.750 | 0.692 | 0.419 | 0.063 | 0.203 | 0.351 | 0.472 |

### Fold 4

| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.518 | 0.498 | 0.582 | 0.070 | 0.069 | 0.069 | 0.228 | 0.589 |
| wm | 0.516 | 0.518 | 0.596 | 0.101 | 0.088 | 0.096 | 0.244 | 0.694 |
| wm_context | 0.581 | 0.727 | 0.818 | 0.452 | 0.097 | 0.305 | 0.354 | 0.450 |
| raw_wm_context | 0.581 | 0.726 | 0.816 | 0.451 | 0.099 | 0.305 | 0.353 | 0.450 |

### Fold 5

| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 0.492 | 0.501 | 0.490 | 0.455 | 0.455 | 0.455 | 0.220 | 0.782 |
| wm | 0.515 | 0.518 | 0.502 | 0.319 | 0.293 | 0.306 | 0.238 | 0.751 |
| wm_context | 0.652 | 0.670 | 0.701 | 0.423 | 0.148 | 0.284 | 0.342 | 0.432 |
| raw_wm_context | 0.652 | 0.673 | 0.705 | 0.427 | 0.148 | 0.286 | 0.340 | 0.436 |

## One-Vs-Rest Test AUC


### Fold 0

| feature set | de_risk | recovery | overweight |
|---|---:|---:|---:|
| raw | 0.549 | 0.704 | 0.617 |
| wm | 0.540 | 0.623 | 0.591 |
| wm_context | 0.881 | 0.599 | 0.725 |
| raw_wm_context | 0.882 | 0.596 | 0.725 |

### Fold 4

| feature set | de_risk | recovery | overweight |
|---|---:|---:|---:|
| raw | 0.510 | 0.652 | 0.537 |
| wm | 0.496 | 0.596 | 0.532 |
| wm_context | 0.869 | 0.578 | 0.776 |
| raw_wm_context | 0.869 | 0.578 | 0.776 |

### Fold 5

| feature set | de_risk | recovery | overweight |
|---|---:|---:|---:|
| raw | 0.525 | 0.688 | 0.526 |
| wm | 0.532 | 0.672 | 0.524 |
| wm_context | 0.856 | 0.663 | 0.726 |
| raw_wm_context | 0.856 | 0.668 | 0.726 |
