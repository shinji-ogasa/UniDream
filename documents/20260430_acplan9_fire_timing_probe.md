# AC Fire Timing Probe

Fold: `5`

## Performance / Fire Summary

| label | mode | AlphaEx | SharpeD | MaxDDD | turnover | long | short | fire | mean_delta | fire_pnl | nonfire_pnl | fwd16 | incr16 | pred_adv |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_s7wmbc_s7ac | ac | +37.59 | +0.022 | -0.26 | 1.77 | 2.2% | 0.0% | 5.5% | +0.0605 | +0.0732 | +0.5849 | +0.00231 | +0.00032 | +0.0575 |
| B_s7wmbc_s11ac | ac | +23.86 | -0.009 | -0.36 | 2.22 | 2.2% | 0.0% | 2.6% | +0.1187 | +0.0405 | +0.6150 | +0.00157 | +0.00024 | +0.2126 |
| C_s11wmbc_noac | bc | +23.20 | +0.014 | -0.30 | 0.95 | 1.4% | 0.0% | 2.6% | +0.0739 | +0.0596 | +0.5959 | +0.00062 | +0.00009 | +0.1431 |
| D_s11wmbc_s7ac | ac | -7.19 | -0.012 | +0.10 | 0.43 | 0.6% | 0.0% | 1.2% | +0.0712 | +0.0117 | +0.6380 | +0.00167 | +0.00002 | +0.1975 |
| E_s11wmbc_s11ac | ac | +22.13 | +0.016 | +0.04 | 0.79 | 1.1% | 0.0% | 2.3% | +0.0660 | +0.0620 | +0.5933 | +0.00297 | +0.00015 | +0.0935 |
| G_guarded_s11 | ac | -9.51 | -0.036 | +0.13 | 0.89 | 1.5% | 0.0% | 2.4% | +0.0860 | -0.0149 | +0.6642 | -0.00077 | +0.00010 | +0.1927 |

## Fire Overlap

| pair | jaccard | intersection | left_only | right_only |
|---|---:|---:|---:|---:|
| A_s7wmbc_s7ac__B_s7wmbc_s11ac | 0.233 | 133 | 347 | 92 |
| A_s7wmbc_s7ac__C_s11wmbc_noac | 0.221 | 128 | 352 | 100 |
| A_s7wmbc_s7ac__D_s11wmbc_s7ac | 0.224 | 108 | 372 | 2 |
| A_s7wmbc_s7ac__E_s11wmbc_s11ac | 0.204 | 115 | 365 | 85 |
| A_s7wmbc_s7ac__G_guarded_s11 | 0.263 | 144 | 336 | 67 |
| B_s7wmbc_s11ac__C_s11wmbc_noac | 0.171 | 66 | 159 | 162 |
| B_s7wmbc_s11ac__D_s11wmbc_s7ac | 0.151 | 44 | 181 | 66 |
| B_s7wmbc_s11ac__E_s11wmbc_s11ac | 0.118 | 45 | 180 | 155 |
| B_s7wmbc_s11ac__G_guarded_s11 | 0.444 | 134 | 91 | 77 |
| C_s11wmbc_noac__D_s11wmbc_s7ac | 0.463 | 107 | 121 | 3 |
| C_s11wmbc_noac__E_s11wmbc_s11ac | 0.321 | 104 | 124 | 96 |
| C_s11wmbc_noac__G_guarded_s11 | 0.421 | 130 | 98 | 81 |
| D_s11wmbc_s7ac__E_s11wmbc_s11ac | 0.505 | 104 | 6 | 96 |
| D_s11wmbc_s7ac__G_guarded_s11 | 0.514 | 109 | 1 | 102 |
| E_s11wmbc_s11ac__G_guarded_s11 | 0.339 | 104 | 96 | 107 |
