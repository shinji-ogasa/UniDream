# Market Event Label Probe

Config: `configs/trading_wm_base_s011.yaml`
Checkpoint dir: `checkpoints/acplan13_base_wm_s011`
Folds: `4, 5, 6`
Horizon: `64`

## Aggregate

### active

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.451 | 0.563 | 0.502 | 0.507 | 0.279 | 0.159 | 0.203 | 0.280 | 0.251 |
| raw | 3 | 0.451 | 0.577 | 0.527 | 0.522 | 0.266 | 0.164 | 0.202 | 0.324 | 0.241 |
| raw_wm | 3 | 0.451 | 0.551 | 0.491 | 0.496 | 0.259 | 0.196 | 0.200 | 0.371 | 0.234 |
| raw_wm_context_no_position | 3 | 0.451 | 0.551 | 0.493 | 0.496 | 0.260 | 0.192 | 0.199 | 0.371 | 0.235 |
| wm | 3 | 0.451 | 0.551 | 0.492 | 0.496 | 0.258 | 0.191 | 0.198 | 0.369 | 0.233 |

### risk_off

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.251 | 0.548 | 0.489 | 0.311 | 0.252 | 0.140 | 0.190 | 0.202 | 0.218 |
| raw | 3 | 0.251 | 0.561 | 0.512 | 0.323 | 0.271 | 0.135 | 0.207 | 0.215 | 0.234 |
| raw_wm | 3 | 0.251 | 0.538 | 0.498 | 0.299 | 0.267 | 0.211 | 0.221 | 0.272 | 0.242 |
| raw_wm_context_no_position | 3 | 0.251 | 0.541 | 0.498 | 0.299 | 0.270 | 0.218 | 0.219 | 0.268 | 0.240 |
| wm | 3 | 0.251 | 0.538 | 0.497 | 0.297 | 0.269 | 0.216 | 0.224 | 0.271 | 0.245 |

### recovery

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.132 | 0.507 | 0.466 | 0.137 | 0.158 | 0.076 | 0.141 | 0.177 | 0.144 |
| raw | 3 | 0.132 | 0.510 | 0.488 | 0.139 | 0.168 | 0.058 | 0.156 | 0.276 | 0.157 |
| raw_wm | 3 | 0.132 | 0.562 | 0.552 | 0.157 | 0.200 | 0.150 | 0.148 | 0.213 | 0.154 |
| raw_wm_context_no_position | 3 | 0.132 | 0.562 | 0.551 | 0.156 | 0.206 | 0.148 | 0.149 | 0.213 | 0.156 |
| wm | 3 | 0.132 | 0.562 | 0.550 | 0.156 | 0.198 | 0.141 | 0.146 | 0.204 | 0.153 |

### overweight

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.200 | 0.513 | 0.510 | 0.198 | 0.177 | 0.124 | 0.185 | 0.242 | 0.184 |
| raw | 3 | 0.200 | 0.515 | 0.496 | 0.200 | 0.194 | 0.006 | 0.196 | 0.461 | 0.196 |
| raw_wm | 3 | 0.200 | 0.531 | 0.490 | 0.212 | 0.159 | 0.027 | 0.150 | 0.332 | 0.151 |
| raw_wm_context_no_position | 3 | 0.200 | 0.531 | 0.492 | 0.212 | 0.161 | 0.026 | 0.151 | 0.335 | 0.153 |
| wm | 3 | 0.200 | 0.534 | 0.491 | 0.213 | 0.161 | 0.028 | 0.150 | 0.333 | 0.152 |

## Fold Detail

### Fold 4

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.635 | 0.650 | 0.774 | 0.238 | 0.080 | 0.180 |
| raw | risk_off | 0.431 | 0.637 | 0.598 | 0.405 | 0.215 | 0.297 |
| raw | recovery | 0.175 | 0.507 | 0.176 | 0.058 | 0.058 | 0.058 |
| raw | overweight | 0.204 | 0.519 | 0.205 | 0.006 | 0.014 | 0.012 |
| wm | active | 0.635 | 0.621 | 0.758 | 0.235 | 0.077 | 0.178 |
| wm | risk_off | 0.431 | 0.608 | 0.567 | 0.352 | 0.201 | 0.266 |
| wm | recovery | 0.175 | 0.575 | 0.210 | 0.178 | 0.127 | 0.136 |
| wm | overweight | 0.204 | 0.533 | 0.221 | 0.028 | 0.019 | 0.021 |
| raw_wm | active | 0.635 | 0.621 | 0.758 | 0.233 | 0.081 | 0.178 |
| raw_wm | risk_off | 0.431 | 0.606 | 0.565 | 0.346 | 0.202 | 0.264 |
| raw_wm | recovery | 0.175 | 0.575 | 0.210 | 0.158 | 0.121 | 0.127 |
| raw_wm | overweight | 0.204 | 0.533 | 0.222 | 0.027 | 0.019 | 0.021 |
| context_no_position | active | 0.635 | 0.674 | 0.792 | 0.368 | 0.145 | 0.286 |
| context_no_position | risk_off | 0.431 | 0.656 | 0.613 | 0.401 | 0.202 | 0.288 |
| context_no_position | recovery | 0.175 | 0.552 | 0.196 | 0.193 | 0.145 | 0.153 |
| context_no_position | overweight | 0.204 | 0.510 | 0.198 | 0.124 | 0.155 | 0.148 |
| raw_wm_context_no_position | active | 0.635 | 0.622 | 0.757 | 0.235 | 0.079 | 0.178 |
| raw_wm_context_no_position | risk_off | 0.431 | 0.607 | 0.563 | 0.342 | 0.199 | 0.261 |
| raw_wm_context_no_position | recovery | 0.175 | 0.575 | 0.210 | 0.168 | 0.127 | 0.134 |
| raw_wm_context_no_position | overweight | 0.204 | 0.531 | 0.221 | 0.026 | 0.020 | 0.021 |

### Fold 5

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.400 | 0.555 | 0.455 | 0.396 | 0.324 | 0.353 |
| raw | risk_off | 0.145 | 0.533 | 0.204 | 0.272 | 0.206 | 0.215 |
| raw | recovery | 0.120 | 0.488 | 0.124 | 0.134 | 0.134 | 0.134 |
| raw | overweight | 0.255 | 0.496 | 0.247 | 0.476 | 0.461 | 0.465 |
| wm | active | 0.400 | 0.492 | 0.386 | 0.347 | 0.369 | 0.360 |
| wm | risk_off | 0.145 | 0.510 | 0.145 | 0.241 | 0.271 | 0.267 |
| wm | recovery | 0.120 | 0.550 | 0.136 | 0.141 | 0.109 | 0.113 |
| wm | overweight | 0.255 | 0.491 | 0.248 | 0.323 | 0.333 | 0.330 |
| raw_wm | active | 0.400 | 0.491 | 0.385 | 0.348 | 0.371 | 0.362 |
| raw_wm | risk_off | 0.145 | 0.510 | 0.154 | 0.244 | 0.272 | 0.268 |
| raw_wm | recovery | 0.120 | 0.552 | 0.138 | 0.150 | 0.109 | 0.114 |
| raw_wm | overweight | 0.255 | 0.490 | 0.247 | 0.324 | 0.332 | 0.330 |
| context_no_position | active | 0.400 | 0.514 | 0.410 | 0.310 | 0.280 | 0.292 |
| context_no_position | risk_off | 0.145 | 0.499 | 0.154 | 0.214 | 0.196 | 0.199 |
| context_no_position | recovery | 0.120 | 0.466 | 0.109 | 0.076 | 0.103 | 0.100 |
| context_no_position | overweight | 0.255 | 0.510 | 0.253 | 0.248 | 0.242 | 0.244 |
| raw_wm_context_no_position | active | 0.400 | 0.493 | 0.386 | 0.352 | 0.371 | 0.363 |
| raw_wm_context_no_position | risk_off | 0.145 | 0.517 | 0.152 | 0.249 | 0.268 | 0.265 |
| raw_wm_context_no_position | recovery | 0.120 | 0.551 | 0.136 | 0.148 | 0.108 | 0.113 |
| raw_wm_context_no_position | overweight | 0.255 | 0.492 | 0.248 | 0.332 | 0.335 | 0.335 |

### Fold 6

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.318 | 0.527 | 0.337 | 0.164 | 0.202 | 0.190 |
| raw | risk_off | 0.177 | 0.512 | 0.168 | 0.135 | 0.200 | 0.189 |
| raw | recovery | 0.103 | 0.534 | 0.117 | 0.313 | 0.276 | 0.280 |
| raw | overweight | 0.140 | 0.532 | 0.148 | 0.100 | 0.111 | 0.110 |
| wm | active | 0.318 | 0.538 | 0.344 | 0.191 | 0.147 | 0.161 |
| wm | risk_off | 0.177 | 0.497 | 0.179 | 0.216 | 0.201 | 0.203 |
| wm | recovery | 0.103 | 0.561 | 0.122 | 0.276 | 0.204 | 0.211 |
| wm | overweight | 0.140 | 0.578 | 0.171 | 0.131 | 0.100 | 0.104 |
| raw_wm | active | 0.318 | 0.540 | 0.345 | 0.196 | 0.149 | 0.164 |
| raw_wm | risk_off | 0.177 | 0.498 | 0.179 | 0.211 | 0.190 | 0.194 |
| raw_wm | recovery | 0.103 | 0.559 | 0.122 | 0.292 | 0.213 | 0.221 |
| raw_wm | overweight | 0.140 | 0.569 | 0.167 | 0.126 | 0.100 | 0.104 |
| context_no_position | active | 0.318 | 0.502 | 0.319 | 0.159 | 0.183 | 0.175 |
| context_no_position | risk_off | 0.177 | 0.489 | 0.166 | 0.140 | 0.173 | 0.167 |
| context_no_position | recovery | 0.103 | 0.504 | 0.106 | 0.205 | 0.177 | 0.179 |
| context_no_position | overweight | 0.140 | 0.519 | 0.143 | 0.159 | 0.159 | 0.159 |
| raw_wm_context_no_position | active | 0.318 | 0.537 | 0.344 | 0.192 | 0.148 | 0.162 |
| raw_wm_context_no_position | risk_off | 0.177 | 0.498 | 0.181 | 0.218 | 0.189 | 0.194 |
| raw_wm_context_no_position | recovery | 0.103 | 0.560 | 0.122 | 0.301 | 0.213 | 0.222 |
| raw_wm_context_no_position | overweight | 0.140 | 0.569 | 0.166 | 0.124 | 0.099 | 0.102 |

