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
| mse_ensemble_mean | train | 0.001511 | 0.2138 | 0.2329 | 0.021 | 0.000000 | 0.000969 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| mse_ensemble_mean | val | 0.001864 | 0.2195 | 0.1819 | 0.012 | 0.000000 | 0.001554 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| mse_ensemble_mean | test | 0.002145 | 0.2760 | 0.2364 | 0.011 | 0.000000 | 0.001479 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| cql_lite_minq | train | 0.001512 | 0.2118 | 0.2329 | 0.021 | 0.000000 | 0.000969 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| cql_lite_minq | val | 0.001860 | 0.2138 | 0.1819 | 0.012 | 0.000000 | 0.001554 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| cql_lite_minq | test | 0.002138 | 0.2814 | 0.2364 | 0.011 | 0.000000 | 0.001479 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_mse_mean | train | 0.001780 | -0.0968 | 0.2329 | 0.021 | 0.000000 | 0.000969 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_mse_mean | val | 0.002208 | -0.1290 | 0.1819 | 0.012 | 0.000000 | 0.001554 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_mse_mean | test | 0.002523 | -0.1116 | 0.2364 | 0.011 | 0.000000 | 0.001479 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_cql_minq | train | 0.001779 | -0.1153 | 0.2329 | 0.021 | 0.000000 | 0.000969 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_cql_minq | val | 0.002208 | -0.1429 | 0.1819 | 0.012 | 0.000000 | 0.001554 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_adv_cql_minq | test | 0.002514 | -0.1301 | 0.2364 | 0.011 | 0.000000 | 0.001479 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0002 |
| anchor_rank_ce_mean | train | 0.001804 | -0.1759 | -0.0625 | 0.203 | -0.000164 | 0.000969 | -0.000003 | 1.000 | 0.000 | 0.000 | 0.0337 |
| anchor_rank_ce_mean | val | 0.002239 | -0.1981 | -0.0625 | 0.229 | -0.000254 | 0.001554 | -0.000005 | 0.999 | 0.001 | 0.000 | 0.0351 |
| anchor_rank_ce_mean | test | 0.002572 | -0.1948 | -0.0456 | 0.251 | -0.000185 | 0.001479 | -0.000004 | 1.000 | 0.000 | 0.000 | 0.0336 |
| anchor_rank_ce_cql_minq | train | 0.001804 | -0.1831 | -0.0053 | 0.203 | -0.000164 | 0.000969 | -0.000004 | 1.000 | 0.000 | 0.000 | 0.0337 |
| anchor_rank_ce_cql_minq | val | 0.002238 | -0.2025 | -0.0147 | 0.229 | -0.000254 | 0.001554 | -0.000019 | 0.999 | 0.001 | 0.000 | 0.0351 |
| anchor_rank_ce_cql_minq | test | 0.002555 | -0.1998 | -0.0028 | 0.251 | -0.000185 | 0.001479 | -0.000003 | 1.000 | 0.000 | 0.000 | 0.0336 |
| anchor_margin_rank_m010_mean | train | 0.001797 | -0.1656 | 0.0948 | 0.203 | -0.000164 | 0.000969 | -0.000238 | 1.000 | 0.000 | 0.000 | 0.0337 |
| anchor_margin_rank_m010_mean | val | 0.002229 | -0.1893 | 0.0470 | 0.229 | -0.000254 | 0.001554 | -0.000332 | 0.999 | 0.001 | 0.000 | 0.0351 |
| anchor_margin_rank_m010_mean | test | 0.002545 | -0.1799 | 0.1402 | 0.251 | -0.000185 | 0.001479 | -0.000440 | 1.000 | 0.000 | 0.000 | 0.0336 |
| anchor_margin_rank_m025_mean | train | 0.001789 | -0.1419 | 0.2430 | 0.055 | 0.000001 | 0.000969 | 0.000001 | 0.000 | 1.000 | 0.000 | 0.0001 |
| anchor_margin_rank_m025_mean | val | 0.002219 | -0.1692 | 0.1902 | 0.027 | 0.000001 | 0.001554 | 0.000002 | 0.000 | 1.000 | 0.000 | 0.0001 |
| anchor_margin_rank_m025_mean | test | 0.002532 | -0.1583 | 0.2434 | 0.028 | 0.000001 | 0.001479 | 0.000001 | 0.000 | 1.000 | 0.000 | 0.0001 |

### Reading

- Positive row Spearman means the critic can rank candidate actions inside a state.
- `selected adv vs anchor` should be positive before actor residual updates are allowed.
- High selected short/long concentration indicates one-sided Q bias.
