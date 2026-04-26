# AC Candidate State-Action Critic Probe

Config: `configs/bcplan5_phase8_state_machine_s007.yaml`
Candidate actions: `0.00, 0.50, 1.00, 1.05, 1.10, 1.25`

## Summary

This probe trains candidate Q(s, position_candidate) from realized transition values. Actor updates are not performed.

## Fold 4

### Baseline Policy

| split | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | long | short | flat | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -19.60 | -0.007 | -1.72 | 0.000 | 0.164 | 0.836 | 20.60 |
| val | -75.09 | -0.014 | -1.11 | 0.000 | 0.150 | 0.850 | 2.46 |
| test | 0.89 | -0.011 | -1.58 | 0.000 | 0.165 | 0.835 | 2.65 |

### Candidate Q Metrics

| variant | split | RMSE | flat corr | row Spearman | top1 match | selected adv vs anchor | best possible adv | top-decile selected adv | selected short | selected flat | selected long | mean abs delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mse_ensemble_mean | train | 0.006394 | 0.3081 | 0.3381 | 0.254 | -0.000082 | 0.002608 | -0.000283 | 0.000 | 1.000 | 0.000 | 0.0675 |
| mse_ensemble_mean | val | 0.008231 | 0.2982 | 0.2922 | 0.223 | -0.000127 | 0.004023 | -0.000429 | 0.000 | 1.000 | 0.000 | 0.0648 |
| mse_ensemble_mean | test | 0.008786 | 0.2404 | 0.2412 | 0.179 | -0.000215 | 0.005108 | -0.000638 | 0.000 | 1.000 | 0.000 | 0.0705 |
| cql_lite_minq | train | 0.006380 | 0.3074 | 0.3527 | 0.368 | -0.000014 | 0.002608 | -0.000014 | 0.000 | 1.000 | 0.000 | 0.0522 |
| cql_lite_minq | val | 0.008260 | 0.2937 | 0.3049 | 0.315 | -0.000009 | 0.004023 | -0.000009 | 0.000 | 1.000 | 0.000 | 0.0503 |
| cql_lite_minq | test | 0.008785 | 0.2436 | 0.2608 | 0.300 | -0.000050 | 0.005108 | -0.000050 | 0.000 | 1.000 | 0.000 | 0.0527 |
| anchor_adv_mse_mean | train | 0.006382 | 0.3099 | 0.3344 | 0.227 | -0.000096 | 0.002608 | -0.000268 | 0.000 | 1.000 | 0.000 | 0.0711 |
| anchor_adv_mse_mean | val | 0.008302 | 0.2956 | 0.2889 | 0.196 | -0.000149 | 0.004023 | -0.000462 | 0.000 | 1.000 | 0.000 | 0.0685 |
| anchor_adv_mse_mean | test | 0.008814 | 0.2400 | 0.2375 | 0.157 | -0.000243 | 0.005108 | -0.000623 | 0.000 | 1.000 | 0.000 | 0.0738 |
| anchor_adv_cql_minq | train | 0.006397 | 0.3046 | 0.3401 | 0.263 | -0.000080 | 0.002608 | -0.000301 | 0.000 | 1.000 | 0.000 | 0.0664 |
| anchor_adv_cql_minq | val | 0.008253 | 0.2900 | 0.2935 | 0.231 | -0.000119 | 0.004023 | -0.000480 | 0.000 | 1.000 | 0.000 | 0.0635 |
| anchor_adv_cql_minq | test | 0.008801 | 0.2414 | 0.2439 | 0.194 | -0.000204 | 0.005108 | -0.000681 | 0.000 | 1.000 | 0.000 | 0.0687 |

### Reading

- Positive row Spearman means the critic can rank candidate actions inside a state.
- `selected adv vs anchor` should be positive before actor residual updates are allowed.
- High selected short/long concentration indicates one-sided Q bias.
