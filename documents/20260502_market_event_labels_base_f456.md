# Market Event Label Probe

Config: `configs/trading_wm_base_s011.yaml`
Checkpoint dir: `checkpoints/acplan13_base_wm_s011`
Folds: `4, 5, 6`
Horizon: `32`

## Aggregate

### active

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.995 | 0.755 | 0.675 | 0.998 | 0.267 | 0.252 | 0.019 | 0.056 | 0.266 |
| raw | 3 | 0.995 | 0.655 | 0.604 | 0.997 | 0.298 | 0.178 | 0.111 | 0.208 | 0.297 |
| raw_wm | 3 | 0.995 | 0.659 | 0.455 | 0.996 | 0.287 | 0.256 | 0.149 | 0.321 | 0.287 |
| raw_wm_context_no_position | 3 | 0.995 | 0.662 | 0.462 | 0.996 | 0.289 | 0.258 | 0.149 | 0.321 | 0.288 |
| wm | 3 | 0.995 | 0.661 | 0.462 | 0.996 | 0.290 | 0.257 | 0.149 | 0.321 | 0.289 |

### risk_off

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.958 | 0.679 | 0.631 | 0.975 | 0.282 | 0.245 | 0.115 | 0.157 | 0.276 |
| raw | 3 | 0.958 | 0.667 | 0.618 | 0.976 | 0.310 | 0.194 | 0.170 | 0.344 | 0.304 |
| raw_wm | 3 | 0.958 | 0.604 | 0.520 | 0.967 | 0.287 | 0.176 | 0.202 | 0.318 | 0.285 |
| raw_wm_context_no_position | 3 | 0.958 | 0.604 | 0.523 | 0.967 | 0.286 | 0.180 | 0.202 | 0.318 | 0.283 |
| wm | 3 | 0.958 | 0.603 | 0.521 | 0.966 | 0.289 | 0.181 | 0.203 | 0.320 | 0.286 |

### recovery

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.250 | 0.499 | 0.473 | 0.252 | 0.140 | 0.092 | 0.134 | 0.158 | 0.135 |
| raw | 3 | 0.250 | 0.520 | 0.481 | 0.260 | 0.170 | 0.107 | 0.152 | 0.241 | 0.156 |
| raw_wm | 3 | 0.250 | 0.565 | 0.563 | 0.291 | 0.191 | 0.124 | 0.141 | 0.173 | 0.154 |
| raw_wm_context_no_position | 3 | 0.250 | 0.565 | 0.561 | 0.291 | 0.195 | 0.121 | 0.143 | 0.182 | 0.157 |
| wm | 3 | 0.250 | 0.569 | 0.567 | 0.293 | 0.191 | 0.116 | 0.139 | 0.177 | 0.152 |

### overweight

| feature set | folds | density | AUC mean | AUC worst | AP mean | recall mean | recall worst | false-active mean | false-active worst | pred-active mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| context_no_position | 3 | 0.095 | 0.615 | 0.553 | 0.123 | 0.221 | 0.062 | 0.133 | 0.179 | 0.138 |
| raw | 3 | 0.095 | 0.621 | 0.563 | 0.133 | 0.199 | 0.093 | 0.114 | 0.135 | 0.121 |
| raw_wm | 3 | 0.095 | 0.570 | 0.538 | 0.108 | 0.098 | 0.066 | 0.083 | 0.093 | 0.085 |
| raw_wm_context_no_position | 3 | 0.095 | 0.572 | 0.537 | 0.108 | 0.100 | 0.073 | 0.085 | 0.098 | 0.087 |
| wm | 3 | 0.095 | 0.571 | 0.536 | 0.108 | 0.099 | 0.073 | 0.084 | 0.094 | 0.086 |

## Fold Detail

### Fold 4

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.999 | 0.661 | 0.999 | 0.178 | 0.000 | 0.178 |
| raw | risk_off | 0.987 | 0.721 | 0.995 | 0.194 | 0.063 | 0.193 |
| raw | recovery | 0.280 | 0.481 | 0.265 | 0.107 | 0.130 | 0.124 |
| raw | overweight | 0.047 | 0.705 | 0.081 | 0.224 | 0.118 | 0.123 |
| wm | active | 0.999 | 0.924 | 1.000 | 0.257 | 0.000 | 0.256 |
| wm | risk_off | 0.987 | 0.713 | 0.995 | 0.255 | 0.081 | 0.253 |
| wm | recovery | 0.280 | 0.567 | 0.316 | 0.183 | 0.146 | 0.156 |
| wm | overweight | 0.047 | 0.624 | 0.060 | 0.073 | 0.067 | 0.067 |
| raw_wm | active | 0.999 | 0.925 | 1.000 | 0.256 | 0.000 | 0.256 |
| raw_wm | risk_off | 0.987 | 0.713 | 0.995 | 0.259 | 0.090 | 0.257 |
| raw_wm | recovery | 0.280 | 0.567 | 0.317 | 0.186 | 0.149 | 0.159 |
| raw_wm | overweight | 0.047 | 0.621 | 0.060 | 0.066 | 0.065 | 0.065 |
| context_no_position | active | 0.999 | 0.828 | 1.000 | 0.286 | 0.000 | 0.285 |
| context_no_position | risk_off | 0.987 | 0.775 | 0.996 | 0.314 | 0.081 | 0.311 |
| context_no_position | recovery | 0.280 | 0.504 | 0.277 | 0.147 | 0.158 | 0.155 |
| context_no_position | overweight | 0.047 | 0.717 | 0.090 | 0.395 | 0.179 | 0.189 |
| raw_wm_context_no_position | active | 0.999 | 0.930 | 1.000 | 0.258 | 0.000 | 0.257 |
| raw_wm_context_no_position | risk_off | 0.987 | 0.709 | 0.995 | 0.255 | 0.090 | 0.253 |
| raw_wm_context_no_position | recovery | 0.280 | 0.565 | 0.315 | 0.190 | 0.149 | 0.160 |
| raw_wm_context_no_position | overweight | 0.047 | 0.625 | 0.060 | 0.073 | 0.067 | 0.067 |

### Fold 5

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.994 | 0.701 | 0.998 | 0.511 | 0.208 | 0.509 |
| raw | risk_off | 0.942 | 0.618 | 0.963 | 0.516 | 0.344 | 0.506 |
| raw | recovery | 0.211 | 0.522 | 0.226 | 0.112 | 0.086 | 0.091 |
| raw | overweight | 0.128 | 0.563 | 0.155 | 0.093 | 0.088 | 0.089 |
| wm | active | 0.994 | 0.462 | 0.994 | 0.345 | 0.321 | 0.345 |
| wm | risk_off | 0.942 | 0.576 | 0.957 | 0.430 | 0.320 | 0.423 |
| wm | recovery | 0.211 | 0.567 | 0.245 | 0.116 | 0.093 | 0.098 |
| wm | overweight | 0.128 | 0.552 | 0.140 | 0.087 | 0.092 | 0.091 |
| raw_wm | active | 0.994 | 0.455 | 0.994 | 0.342 | 0.321 | 0.342 |
| raw_wm | risk_off | 0.942 | 0.578 | 0.957 | 0.427 | 0.318 | 0.420 |
| raw_wm | recovery | 0.211 | 0.563 | 0.244 | 0.124 | 0.101 | 0.106 |
| raw_wm | overweight | 0.128 | 0.552 | 0.140 | 0.095 | 0.091 | 0.092 |
| context_no_position | active | 0.994 | 0.763 | 0.998 | 0.263 | 0.000 | 0.261 |
| context_no_position | risk_off | 0.942 | 0.631 | 0.964 | 0.287 | 0.157 | 0.279 |
| context_no_position | recovery | 0.211 | 0.473 | 0.204 | 0.092 | 0.090 | 0.090 |
| context_no_position | overweight | 0.128 | 0.553 | 0.139 | 0.062 | 0.069 | 0.068 |
| raw_wm_context_no_position | active | 0.994 | 0.462 | 0.994 | 0.349 | 0.321 | 0.349 |
| raw_wm_context_no_position | risk_off | 0.942 | 0.580 | 0.957 | 0.422 | 0.318 | 0.416 |
| raw_wm_context_no_position | recovery | 0.211 | 0.561 | 0.244 | 0.121 | 0.099 | 0.103 |
| raw_wm_context_no_position | overweight | 0.128 | 0.554 | 0.141 | 0.101 | 0.091 | 0.092 |

### Fold 6

| feature set | target | density | AUC | AP | recall | false-active | pred-active |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | active | 0.992 | 0.604 | 0.995 | 0.205 | 0.125 | 0.204 |
| raw | risk_off | 0.945 | 0.663 | 0.969 | 0.220 | 0.104 | 0.213 |
| raw | recovery | 0.258 | 0.557 | 0.289 | 0.290 | 0.241 | 0.253 |
| raw | overweight | 0.110 | 0.595 | 0.163 | 0.281 | 0.135 | 0.151 |
| wm | active | 0.992 | 0.598 | 0.995 | 0.267 | 0.125 | 0.266 |
| wm | risk_off | 0.945 | 0.521 | 0.948 | 0.181 | 0.207 | 0.182 |
| wm | recovery | 0.258 | 0.573 | 0.318 | 0.274 | 0.177 | 0.202 |
| wm | overweight | 0.110 | 0.536 | 0.125 | 0.138 | 0.094 | 0.099 |
| raw_wm | active | 0.992 | 0.596 | 0.995 | 0.264 | 0.125 | 0.263 |
| raw_wm | risk_off | 0.945 | 0.520 | 0.948 | 0.176 | 0.199 | 0.177 |
| raw_wm | recovery | 0.258 | 0.566 | 0.311 | 0.263 | 0.173 | 0.196 |
| raw_wm | overweight | 0.110 | 0.538 | 0.125 | 0.135 | 0.093 | 0.098 |
| context_no_position | active | 0.992 | 0.675 | 0.996 | 0.252 | 0.056 | 0.251 |
| context_no_position | risk_off | 0.945 | 0.631 | 0.966 | 0.245 | 0.106 | 0.237 |
| context_no_position | recovery | 0.258 | 0.520 | 0.274 | 0.182 | 0.154 | 0.161 |
| context_no_position | overweight | 0.110 | 0.575 | 0.141 | 0.206 | 0.151 | 0.157 |
| raw_wm_context_no_position | active | 0.992 | 0.593 | 0.995 | 0.259 | 0.125 | 0.258 |
| raw_wm_context_no_position | risk_off | 0.945 | 0.523 | 0.948 | 0.180 | 0.199 | 0.181 |
| raw_wm_context_no_position | recovery | 0.258 | 0.570 | 0.315 | 0.276 | 0.182 | 0.206 |
| raw_wm_context_no_position | overweight | 0.110 | 0.537 | 0.124 | 0.127 | 0.098 | 0.101 |

