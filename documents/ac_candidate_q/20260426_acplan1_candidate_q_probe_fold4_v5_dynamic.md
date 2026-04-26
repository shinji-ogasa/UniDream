# AC Candidate State-Action Critic Probe

Config: `configs/bcplan5_phase8_state_machine_s007.yaml`
Candidate mode: `dynamic`
Candidate labels: `bc, bc_minus_0.05, bc_plus_0.05, hold_current, benchmark, ow_1.05, ow_1.10, ow_1.25`

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
| mse_ensemble_mean | train | 0.001510 | 0.2206 | 0.2329 | 0.021 | 0.000000 | 0.000969 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| mse_ensemble_mean | val | 0.001867 | 0.2200 | 0.1819 | 0.012 | 0.000000 | 0.001554 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| mse_ensemble_mean | test | 0.002153 | 0.2820 | 0.2364 | 0.011 | 0.000000 | 0.001479 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| cql_lite_minq | train | 0.001508 | 0.2243 | 0.2329 | 0.021 | 0.000000 | 0.000969 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| cql_lite_minq | val | 0.001854 | 0.2232 | 0.1818 | 0.012 | 0.000000 | 0.001554 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| cql_lite_minq | test | 0.002131 | 0.2781 | 0.2364 | 0.011 | 0.000000 | 0.001479 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_mse_mean | train | 0.001775 | -0.0908 | 0.2329 | 0.021 | 0.000000 | 0.000969 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_mse_mean | val | 0.002203 | -0.1260 | 0.1819 | 0.012 | 0.000000 | 0.001554 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_mse_mean | test | 0.002505 | -0.1081 | 0.2364 | 0.011 | 0.000000 | 0.001479 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_cql_minq | train | 0.001777 | -0.0751 | 0.2329 | 0.021 | 0.000000 | 0.000969 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_cql_minq | val | 0.002203 | -0.1157 | 0.1819 | 0.012 | 0.000000 | 0.001554 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_cql_minq | test | 0.002504 | -0.0943 | 0.2364 | 0.011 | 0.000000 | 0.001479 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_rank_ce_mean | train | 0.001800 | -0.1789 | 0.0329 | 0.203 | -0.000164 | 0.000969 | -0.000079 | 1.000 | 0.000 | 0.000 | 0.0337 |
| anchor_rank_ce_mean | val | 0.002235 | -0.1998 | 0.0171 | 0.229 | -0.000254 | 0.001554 | -0.000125 | 0.999 | 0.001 | 0.000 | 0.0351 |
| anchor_rank_ce_mean | test | 0.002559 | -0.1960 | 0.0499 | 0.251 | -0.000185 | 0.001479 | -0.000217 | 1.000 | 0.000 | 0.000 | 0.0336 |
| anchor_rank_ce_cql_minq | train | 0.001800 | -0.1725 | 0.0218 | 0.203 | -0.000164 | 0.000969 | -0.000147 | 1.000 | 0.000 | 0.000 | 0.0337 |
| anchor_rank_ce_cql_minq | val | 0.002235 | -0.1964 | 0.0079 | 0.229 | -0.000254 | 0.001554 | -0.000206 | 0.999 | 0.001 | 0.000 | 0.0351 |
| anchor_rank_ce_cql_minq | test | 0.002563 | -0.1913 | 0.0270 | 0.251 | -0.000185 | 0.001479 | -0.000280 | 1.000 | 0.000 | 0.000 | 0.0336 |

### Reading

- Positive row Spearman means the critic can rank candidate actions inside a state.
- `selected adv vs anchor` should be positive before actor residual updates are allowed.
- High selected short/long concentration indicates one-sided Q bias.
