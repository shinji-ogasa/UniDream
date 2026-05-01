# Market Event Label Probe

Config: `configs/trading_wm_base_s011.yaml`
Checkpoint dir: `checkpoints/acplan13_base_wm_s011`
Folds: `4, 5, 6`
Horizon: `32`
Probe model: `hgb`

## Aggregate

### active

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.410 | 0.566 | 0.508 | 0.470 | 0.258 | 0.154 | 0.184 | 0.217 | 0.225 |
| raw | 3 | 0.410 | 0.558 | 0.496 | 0.470 | 0.252 | 0.102 | 0.183 | 0.325 | 0.222 |
| raw_wm | 3 | 0.410 | 0.565 | 0.500 | 0.473 | 0.278 | 0.150 | 0.196 | 0.299 | 0.242 |
| raw_wm_context_no_position | 3 | 0.410 | 0.566 | 0.500 | 0.472 | 0.265 | 0.140 | 0.188 | 0.296 | 0.231 |
| wm | 3 | 0.410 | 0.560 | 0.505 | 0.473 | 0.289 | 0.253 | 0.214 | 0.298 | 0.255 |

### risk_off

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.201 | 0.565 | 0.505 | 0.270 | 0.274 | 0.154 | 0.180 | 0.198 | 0.210 |
| raw | 3 | 0.201 | 0.565 | 0.508 | 0.271 | 0.332 | 0.149 | 0.244 | 0.343 | 0.278 |
| raw_wm | 3 | 0.201 | 0.572 | 0.511 | 0.274 | 0.303 | 0.095 | 0.222 | 0.335 | 0.255 |
| raw_wm_context_no_position | 3 | 0.201 | 0.571 | 0.509 | 0.273 | 0.304 | 0.100 | 0.220 | 0.330 | 0.254 |
| wm | 3 | 0.201 | 0.582 | 0.531 | 0.277 | 0.293 | 0.242 | 0.192 | 0.224 | 0.223 |

### recovery

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.136 | 0.519 | 0.508 | 0.144 | 0.159 | 0.153 | 0.145 | 0.166 | 0.147 |
| raw | 3 | 0.136 | 0.544 | 0.495 | 0.166 | 0.224 | 0.048 | 0.201 | 0.336 | 0.204 |
| raw_wm | 3 | 0.136 | 0.543 | 0.512 | 0.158 | 0.201 | 0.054 | 0.175 | 0.333 | 0.179 |
| raw_wm_context_no_position | 3 | 0.136 | 0.547 | 0.516 | 0.159 | 0.207 | 0.055 | 0.175 | 0.334 | 0.179 |
| wm | 3 | 0.136 | 0.547 | 0.526 | 0.159 | 0.185 | 0.112 | 0.148 | 0.270 | 0.153 |

### overweight

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.203 | 0.510 | 0.500 | 0.202 | 0.151 | 0.135 | 0.165 | 0.189 | 0.162 |
| raw | 3 | 0.203 | 0.470 | 0.436 | 0.191 | 0.096 | 0.040 | 0.109 | 0.176 | 0.107 |
| raw_wm | 3 | 0.203 | 0.505 | 0.496 | 0.205 | 0.120 | 0.071 | 0.115 | 0.194 | 0.116 |
| raw_wm_context_no_position | 3 | 0.203 | 0.503 | 0.495 | 0.205 | 0.117 | 0.067 | 0.110 | 0.182 | 0.111 |
| wm | 3 | 0.203 | 0.509 | 0.499 | 0.205 | 0.131 | 0.097 | 0.129 | 0.184 | 0.129 |

## Fold Detail

### Fold 4

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.574 | 0.656 | 0.733 | 0.325 | 0.123 | 0.239 |
| raw | risk_off | 0.360 | 0.677 | 0.575 | 0.519 | 0.229 | 0.333 |
| raw | recovery | 0.185 | 0.495 | 0.198 | 0.255 | 0.252 | 0.253 |
| raw | overweight | 0.200 | 0.458 | 0.177 | 0.040 | 0.075 | 0.068 |
| wm | active | 0.574 | 0.652 | 0.731 | 0.306 | 0.113 | 0.224 |
| wm | risk_off | 0.360 | 0.668 | 0.562 | 0.388 | 0.147 | 0.234 |
| wm | recovery | 0.185 | 0.552 | 0.218 | 0.112 | 0.077 | 0.084 |
| wm | overweight | 0.200 | 0.499 | 0.196 | 0.097 | 0.102 | 0.101 |
| raw_wm | active | 0.574 | 0.670 | 0.745 | 0.367 | 0.129 | 0.265 |
| raw_wm | risk_off | 0.360 | 0.679 | 0.578 | 0.477 | 0.193 | 0.295 |
| raw_wm | recovery | 0.185 | 0.513 | 0.198 | 0.192 | 0.161 | 0.166 |
| raw_wm | overweight | 0.200 | 0.496 | 0.195 | 0.071 | 0.076 | 0.075 |
| context_no_position | active | 0.574 | 0.665 | 0.734 | 0.376 | 0.177 | 0.291 |
| context_no_position | risk_off | 0.360 | 0.672 | 0.551 | 0.428 | 0.190 | 0.276 |
| context_no_position | recovery | 0.185 | 0.521 | 0.193 | 0.153 | 0.149 | 0.150 |
| context_no_position | overweight | 0.200 | 0.500 | 0.193 | 0.151 | 0.163 | 0.161 |
| raw_wm_context_no_position | active | 0.574 | 0.669 | 0.742 | 0.335 | 0.115 | 0.241 |
| raw_wm_context_no_position | risk_off | 0.360 | 0.680 | 0.578 | 0.481 | 0.185 | 0.292 |
| raw_wm_context_no_position | recovery | 0.185 | 0.522 | 0.201 | 0.192 | 0.161 | 0.167 |
| raw_wm_context_no_position | overweight | 0.200 | 0.495 | 0.196 | 0.067 | 0.073 | 0.072 |

### Fold 5

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.344 | 0.496 | 0.347 | 0.102 | 0.100 | 0.101 |
| raw | risk_off | 0.108 | 0.508 | 0.108 | 0.149 | 0.162 | 0.160 |
| raw | recovery | 0.097 | 0.625 | 0.158 | 0.048 | 0.015 | 0.018 |
| raw | overweight | 0.234 | 0.515 | 0.247 | 0.198 | 0.176 | 0.181 |
| wm | active | 0.344 | 0.505 | 0.350 | 0.307 | 0.298 | 0.301 |
| wm | risk_off | 0.108 | 0.546 | 0.129 | 0.242 | 0.205 | 0.209 |
| wm | recovery | 0.097 | 0.564 | 0.122 | 0.139 | 0.096 | 0.100 |
| wm | overweight | 0.234 | 0.507 | 0.237 | 0.190 | 0.184 | 0.186 |
| raw_wm | active | 0.344 | 0.500 | 0.339 | 0.150 | 0.159 | 0.156 |
| raw_wm | risk_off | 0.108 | 0.511 | 0.108 | 0.095 | 0.140 | 0.135 |
| raw_wm | recovery | 0.097 | 0.605 | 0.137 | 0.054 | 0.031 | 0.033 |
| raw_wm | overweight | 0.234 | 0.515 | 0.243 | 0.211 | 0.194 | 0.198 |
| context_no_position | active | 0.344 | 0.526 | 0.364 | 0.245 | 0.217 | 0.227 |
| context_no_position | risk_off | 0.108 | 0.505 | 0.121 | 0.241 | 0.198 | 0.203 |
| context_no_position | recovery | 0.097 | 0.527 | 0.110 | 0.153 | 0.121 | 0.125 |
| context_no_position | overweight | 0.234 | 0.516 | 0.232 | 0.168 | 0.189 | 0.184 |
| raw_wm_context_no_position | active | 0.344 | 0.500 | 0.339 | 0.140 | 0.154 | 0.149 |
| raw_wm_context_no_position | risk_off | 0.108 | 0.509 | 0.107 | 0.100 | 0.145 | 0.140 |
| raw_wm_context_no_position | recovery | 0.097 | 0.604 | 0.136 | 0.055 | 0.031 | 0.033 |
| raw_wm_context_no_position | overweight | 0.234 | 0.514 | 0.243 | 0.202 | 0.182 | 0.186 |

### Fold 6

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.313 | 0.523 | 0.331 | 0.330 | 0.325 | 0.327 |
| raw | risk_off | 0.135 | 0.512 | 0.131 | 0.328 | 0.343 | 0.341 |
| raw | recovery | 0.127 | 0.512 | 0.142 | 0.368 | 0.336 | 0.340 |
| raw | overweight | 0.176 | 0.436 | 0.151 | 0.049 | 0.077 | 0.072 |
| wm | active | 0.313 | 0.524 | 0.336 | 0.253 | 0.233 | 0.239 |
| wm | risk_off | 0.135 | 0.531 | 0.141 | 0.248 | 0.224 | 0.227 |
| wm | recovery | 0.127 | 0.526 | 0.139 | 0.304 | 0.270 | 0.274 |
| wm | overweight | 0.176 | 0.520 | 0.183 | 0.106 | 0.101 | 0.102 |
| raw_wm | active | 0.313 | 0.525 | 0.334 | 0.316 | 0.299 | 0.305 |
| raw_wm | risk_off | 0.135 | 0.525 | 0.136 | 0.337 | 0.335 | 0.335 |
| raw_wm | recovery | 0.127 | 0.512 | 0.139 | 0.358 | 0.333 | 0.336 |
| raw_wm | overweight | 0.176 | 0.505 | 0.176 | 0.076 | 0.074 | 0.074 |
| context_no_position | active | 0.313 | 0.508 | 0.312 | 0.154 | 0.158 | 0.157 |
| context_no_position | risk_off | 0.135 | 0.516 | 0.138 | 0.154 | 0.150 | 0.151 |
| context_no_position | recovery | 0.127 | 0.508 | 0.129 | 0.171 | 0.166 | 0.167 |
| context_no_position | overweight | 0.176 | 0.514 | 0.181 | 0.135 | 0.143 | 0.141 |
| raw_wm_context_no_position | active | 0.313 | 0.527 | 0.335 | 0.320 | 0.296 | 0.304 |
| raw_wm_context_no_position | risk_off | 0.135 | 0.523 | 0.135 | 0.332 | 0.330 | 0.330 |
| raw_wm_context_no_position | recovery | 0.127 | 0.516 | 0.141 | 0.373 | 0.334 | 0.339 |
| raw_wm_context_no_position | overweight | 0.176 | 0.501 | 0.178 | 0.082 | 0.074 | 0.076 |

