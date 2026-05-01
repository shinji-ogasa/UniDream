# Market Event Label Probe

Config: `configs/trading_wm_base_s011.yaml`
Checkpoint dir: `checkpoints/acplan13_base_wm_s011`
Folds: `4, 5, 6`
Horizon: `32`

## Aggregate

### active

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.410 | 0.572 | 0.512 | 0.477 | 0.288 | 0.183 | 0.193 | 0.253 | 0.243 |
| raw | 3 | 0.410 | 0.573 | 0.532 | 0.478 | 0.294 | 0.198 | 0.214 | 0.354 | 0.256 |
| raw_wm | 3 | 0.410 | 0.568 | 0.519 | 0.473 | 0.286 | 0.248 | 0.197 | 0.278 | 0.242 |
| raw_wm_context_no_position | 3 | 0.410 | 0.569 | 0.519 | 0.475 | 0.291 | 0.250 | 0.196 | 0.272 | 0.244 |
| wm | 3 | 0.410 | 0.568 | 0.518 | 0.474 | 0.285 | 0.250 | 0.195 | 0.273 | 0.241 |

### risk_off

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.201 | 0.568 | 0.511 | 0.271 | 0.294 | 0.165 | 0.187 | 0.214 | 0.219 |
| raw | 3 | 0.201 | 0.584 | 0.537 | 0.280 | 0.307 | 0.187 | 0.197 | 0.224 | 0.230 |
| raw_wm | 3 | 0.201 | 0.574 | 0.520 | 0.275 | 0.308 | 0.203 | 0.208 | 0.214 | 0.240 |
| raw_wm_context_no_position | 3 | 0.201 | 0.576 | 0.520 | 0.276 | 0.313 | 0.211 | 0.211 | 0.216 | 0.243 |
| wm | 3 | 0.201 | 0.575 | 0.522 | 0.276 | 0.308 | 0.199 | 0.208 | 0.211 | 0.240 |

### recovery

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.136 | 0.519 | 0.518 | 0.148 | 0.168 | 0.144 | 0.148 | 0.180 | 0.150 |
| raw | 3 | 0.136 | 0.549 | 0.536 | 0.159 | 0.250 | 0.153 | 0.216 | 0.387 | 0.220 |
| raw_wm | 3 | 0.136 | 0.565 | 0.562 | 0.169 | 0.221 | 0.143 | 0.157 | 0.229 | 0.166 |
| raw_wm_context_no_position | 3 | 0.136 | 0.566 | 0.562 | 0.169 | 0.219 | 0.139 | 0.152 | 0.214 | 0.161 |
| wm | 3 | 0.136 | 0.566 | 0.563 | 0.169 | 0.225 | 0.145 | 0.156 | 0.230 | 0.165 |

### overweight

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.203 | 0.505 | 0.476 | 0.202 | 0.175 | 0.127 | 0.183 | 0.225 | 0.181 |
| raw | 3 | 0.203 | 0.495 | 0.483 | 0.195 | 0.210 | 0.009 | 0.225 | 0.544 | 0.222 |
| raw_wm | 3 | 0.203 | 0.528 | 0.496 | 0.217 | 0.174 | 0.043 | 0.155 | 0.313 | 0.159 |
| raw_wm_context_no_position | 3 | 0.203 | 0.528 | 0.496 | 0.216 | 0.176 | 0.044 | 0.160 | 0.325 | 0.163 |
| wm | 3 | 0.203 | 0.528 | 0.494 | 0.216 | 0.173 | 0.043 | 0.157 | 0.320 | 0.160 |

## Fold Detail

### Fold 4

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.574 | 0.650 | 0.731 | 0.296 | 0.106 | 0.215 |
| raw | risk_off | 0.360 | 0.676 | 0.569 | 0.434 | 0.181 | 0.272 |
| raw | recovery | 0.185 | 0.539 | 0.201 | 0.180 | 0.162 | 0.165 |
| raw | overweight | 0.200 | 0.510 | 0.197 | 0.009 | 0.020 | 0.018 |
| wm | active | 0.574 | 0.654 | 0.733 | 0.324 | 0.122 | 0.238 |
| wm | risk_off | 0.360 | 0.676 | 0.568 | 0.463 | 0.206 | 0.299 |
| wm | recovery | 0.185 | 0.563 | 0.219 | 0.202 | 0.153 | 0.162 |
| wm | overweight | 0.200 | 0.536 | 0.218 | 0.043 | 0.037 | 0.038 |
| raw_wm | active | 0.574 | 0.653 | 0.732 | 0.325 | 0.126 | 0.240 |
| raw_wm | risk_off | 0.360 | 0.675 | 0.567 | 0.458 | 0.208 | 0.298 |
| raw_wm | recovery | 0.185 | 0.562 | 0.217 | 0.206 | 0.156 | 0.165 |
| raw_wm | overweight | 0.200 | 0.538 | 0.220 | 0.043 | 0.035 | 0.037 |
| context_no_position | active | 0.574 | 0.670 | 0.740 | 0.377 | 0.156 | 0.283 |
| context_no_position | risk_off | 0.360 | 0.677 | 0.553 | 0.424 | 0.184 | 0.271 |
| context_no_position | recovery | 0.185 | 0.519 | 0.200 | 0.144 | 0.134 | 0.136 |
| context_no_position | overweight | 0.200 | 0.476 | 0.182 | 0.127 | 0.173 | 0.164 |
| raw_wm_context_no_position | active | 0.574 | 0.656 | 0.735 | 0.335 | 0.129 | 0.247 |
| raw_wm_context_no_position | risk_off | 0.360 | 0.676 | 0.567 | 0.461 | 0.210 | 0.301 |
| raw_wm_context_no_position | recovery | 0.185 | 0.562 | 0.218 | 0.205 | 0.157 | 0.166 |
| raw_wm_context_no_position | overweight | 0.200 | 0.537 | 0.219 | 0.044 | 0.040 | 0.041 |

### Fold 5

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.344 | 0.538 | 0.370 | 0.388 | 0.354 | 0.366 |
| raw | risk_off | 0.108 | 0.538 | 0.132 | 0.299 | 0.224 | 0.232 |
| raw | recovery | 0.097 | 0.571 | 0.132 | 0.153 | 0.098 | 0.103 |
| raw | overweight | 0.234 | 0.491 | 0.225 | 0.548 | 0.544 | 0.545 |
| wm | active | 0.344 | 0.518 | 0.350 | 0.280 | 0.273 | 0.275 |
| wm | risk_off | 0.108 | 0.526 | 0.108 | 0.199 | 0.211 | 0.210 |
| wm | recovery | 0.097 | 0.569 | 0.124 | 0.145 | 0.086 | 0.091 |
| wm | overweight | 0.234 | 0.494 | 0.228 | 0.327 | 0.320 | 0.322 |
| raw_wm | active | 0.344 | 0.519 | 0.349 | 0.286 | 0.278 | 0.281 |
| raw_wm | risk_off | 0.108 | 0.526 | 0.108 | 0.203 | 0.214 | 0.213 |
| raw_wm | recovery | 0.097 | 0.568 | 0.125 | 0.143 | 0.086 | 0.092 |
| raw_wm | overweight | 0.234 | 0.496 | 0.229 | 0.328 | 0.313 | 0.316 |
| context_no_position | active | 0.344 | 0.535 | 0.370 | 0.304 | 0.253 | 0.270 |
| context_no_position | risk_off | 0.108 | 0.517 | 0.123 | 0.294 | 0.214 | 0.222 |
| context_no_position | recovery | 0.097 | 0.519 | 0.108 | 0.156 | 0.130 | 0.133 |
| context_no_position | overweight | 0.234 | 0.525 | 0.241 | 0.238 | 0.225 | 0.228 |
| raw_wm_context_no_position | active | 0.344 | 0.519 | 0.349 | 0.287 | 0.272 | 0.277 |
| raw_wm_context_no_position | risk_off | 0.108 | 0.532 | 0.110 | 0.211 | 0.216 | 0.215 |
| raw_wm_context_no_position | recovery | 0.097 | 0.570 | 0.126 | 0.139 | 0.084 | 0.089 |
| raw_wm_context_no_position | overweight | 0.234 | 0.496 | 0.229 | 0.333 | 0.325 | 0.327 |

### Fold 6

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.313 | 0.532 | 0.332 | 0.198 | 0.183 | 0.188 |
| raw | risk_off | 0.135 | 0.537 | 0.140 | 0.187 | 0.186 | 0.186 |
| raw | recovery | 0.127 | 0.536 | 0.144 | 0.418 | 0.387 | 0.391 |
| raw | overweight | 0.176 | 0.483 | 0.164 | 0.072 | 0.110 | 0.104 |
| wm | active | 0.313 | 0.530 | 0.339 | 0.250 | 0.190 | 0.209 |
| wm | risk_off | 0.135 | 0.522 | 0.151 | 0.262 | 0.205 | 0.213 |
| wm | recovery | 0.127 | 0.564 | 0.163 | 0.326 | 0.230 | 0.242 |
| wm | overweight | 0.176 | 0.553 | 0.203 | 0.150 | 0.113 | 0.120 |
| raw_wm | active | 0.313 | 0.531 | 0.339 | 0.248 | 0.187 | 0.206 |
| raw_wm | risk_off | 0.135 | 0.520 | 0.150 | 0.263 | 0.202 | 0.211 |
| raw_wm | recovery | 0.127 | 0.565 | 0.164 | 0.316 | 0.229 | 0.240 |
| raw_wm | overweight | 0.176 | 0.551 | 0.202 | 0.151 | 0.117 | 0.123 |
| context_no_position | active | 0.313 | 0.512 | 0.320 | 0.183 | 0.171 | 0.175 |
| context_no_position | risk_off | 0.135 | 0.511 | 0.137 | 0.165 | 0.163 | 0.163 |
| context_no_position | recovery | 0.127 | 0.518 | 0.136 | 0.203 | 0.180 | 0.183 |
| context_no_position | overweight | 0.176 | 0.514 | 0.185 | 0.161 | 0.150 | 0.152 |
| raw_wm_context_no_position | active | 0.313 | 0.532 | 0.340 | 0.250 | 0.187 | 0.207 |
| raw_wm_context_no_position | risk_off | 0.135 | 0.520 | 0.151 | 0.266 | 0.206 | 0.214 |
| raw_wm_context_no_position | recovery | 0.127 | 0.566 | 0.163 | 0.312 | 0.214 | 0.227 |
| raw_wm_context_no_position | overweight | 0.176 | 0.550 | 0.202 | 0.150 | 0.116 | 0.122 |

