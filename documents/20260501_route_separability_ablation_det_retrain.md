# Route Separability Probe

Config: `configs/acplan17_det_retrain_s007_tmp.yaml`
Checkpoint dir: `checkpoints/acplan17_det_retrain_s007`
Folds: `0, 4, 5`

## Aggregate Active/No-Active Separability

| feature set | folds | test AUC mean | test AUC worst | test AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 3 | 0.738 | 0.694 | 0.789 | 0.434 | 0.343 | 0.064 | 0.137 | 0.246 |
| wm_advantage | 3 | 0.525 | 0.516 | 0.510 | 0.167 | 0.098 | 0.148 | 0.273 | 0.156 |
| wm_position | 3 | 0.715 | 0.669 | 0.733 | 0.429 | 0.415 | 0.102 | 0.143 | 0.262 |
| wm_position_advantage | 3 | 0.714 | 0.666 | 0.731 | 0.427 | 0.406 | 0.103 | 0.145 | 0.262 |
| wm_regime | 3 | 0.526 | 0.516 | 0.511 | 0.168 | 0.098 | 0.148 | 0.268 | 0.157 |

## Fold Detail

Thresholds are selected on validation with the configured false-active and predicted-active caps, then applied to test.

### Fold 0

| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 0.670 | 0.780 | 0.771 | 0.423 | 0.030 | 0.185 | 0.347 | 0.453 |
| wm_position | 0.646 | 0.748 | 0.686 | 0.420 | 0.065 | 0.205 | 0.350 | 0.458 |
| wm_regime | 0.500 | 0.539 | 0.429 | 0.117 | 0.084 | 0.097 | 0.252 | 0.646 |
| wm_advantage | 0.500 | 0.537 | 0.427 | 0.107 | 0.078 | 0.089 | 0.253 | 0.650 |
| wm_position_advantage | 0.647 | 0.748 | 0.685 | 0.422 | 0.065 | 0.206 | 0.352 | 0.457 |

### Fold 4

| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 0.580 | 0.741 | 0.838 | 0.537 | 0.137 | 0.371 | 0.337 | 0.453 |
| wm_position | 0.581 | 0.727 | 0.813 | 0.452 | 0.099 | 0.305 | 0.359 | 0.427 |
| wm_regime | 0.521 | 0.524 | 0.601 | 0.098 | 0.091 | 0.095 | 0.237 | 0.691 |
| wm_advantage | 0.521 | 0.524 | 0.600 | 0.098 | 0.093 | 0.096 | 0.238 | 0.693 |
| wm_position_advantage | 0.581 | 0.727 | 0.814 | 0.453 | 0.099 | 0.306 | 0.358 | 0.430 |

### Fold 5

| feature set | val AUC | test AUC | test AP | test recall | test false-active | test pred-active | multiclass test macro-F1 | multiclass false-active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 0.653 | 0.694 | 0.757 | 0.343 | 0.025 | 0.182 | 0.330 | 0.443 |
| wm_position | 0.651 | 0.669 | 0.701 | 0.415 | 0.143 | 0.277 | 0.342 | 0.413 |
| wm_regime | 0.515 | 0.516 | 0.503 | 0.289 | 0.268 | 0.279 | 0.242 | 0.716 |
| wm_advantage | 0.517 | 0.516 | 0.503 | 0.295 | 0.273 | 0.284 | 0.242 | 0.716 |
| wm_position_advantage | 0.650 | 0.666 | 0.695 | 0.406 | 0.145 | 0.274 | 0.342 | 0.416 |

## One-Vs-Rest Test AUC


### Fold 0

| feature set | de_risk | recovery | overweight |
|---|---:|---:|---:|
| context | 0.890 | 0.699 | 0.707 |
| wm_position | 0.881 | 0.591 | 0.730 |
| wm_regime | 0.539 | 0.617 | 0.592 |
| wm_advantage | 0.538 | 0.621 | 0.591 |
| wm_position_advantage | 0.881 | 0.589 | 0.731 |

### Fold 4

| feature set | de_risk | recovery | overweight |
|---|---:|---:|---:|
| context | 0.878 | 0.631 | 0.764 |
| wm_position | 0.869 | 0.581 | 0.779 |
| wm_regime | 0.497 | 0.589 | 0.541 |
| wm_advantage | 0.496 | 0.591 | 0.541 |
| wm_position_advantage | 0.869 | 0.578 | 0.779 |

### Fold 5

| feature set | de_risk | recovery | overweight |
|---|---:|---:|---:|
| context | 0.858 | 0.661 | 0.724 |
| wm_position | 0.856 | 0.651 | 0.728 |
| wm_regime | 0.537 | 0.664 | 0.525 |
| wm_advantage | 0.538 | 0.667 | 0.527 |
| wm_position_advantage | 0.856 | 0.654 | 0.727 |
