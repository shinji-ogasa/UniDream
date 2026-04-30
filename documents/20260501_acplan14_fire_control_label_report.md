# AC Plan14: Direct Fire-Control Label Probe

Run completed: 2026-05-01 JST

## Purpose

`acplan_14.md` の方針に従い、ACを広げる前に「fireしていい局面」が既存特徴から分離できるかを直接検証した。

今回の対象は学習本流ではなく、label quality probe。

```text
fire_harm_prob_h16/h32
trough_exit_prob_h16/h32
drawdown_worsening_prob_h16/h32
fire_advantage_h16/h32
```

## Implementation

追加:

```text
unidream/cli/fire_control_label_probe.py
```

既存probeの修正:

```text
prepare_world_model_stage(..., fold_idx=split.fold_idx)
```

これにより、既存の fire timing / route / candidate Q probe も fold-matched WM checkpoint を正しく読める。

## Calculation

同じ actor で adapter あり/なしの position を作る。

```text
pos_on  = actor current policy
pos_off = actor with benchmark overweight adapter disabled
fire    = abs(pos_on - pos_off) > 1e-6
```

コスト込み1bar PnL:

```text
pnl_t = position_t * return_t
      - spread_cost * abs(position_t - position_{t-1})
      - fee_cost
      - slippage_cost
```

各 fire bar について horizon 16/32 の未来窓で以下を作った。

```text
fire_advantage_h
  = sum_h pnl(pos_on) - sum_h pnl(pos_off)
```

```text
fire_harm_h = 1
  if local_future_maxdd(pos_on) - local_future_maxdd(pos_off) > 0.00025
```

```text
drawdown_worsening_h = 1
  if current equity/peak state から future drawdown が 0.003 超悪化
```

```text
trough_exit_h = 1
  if future trough が horizon 前半にあり、その後 rebound > 0
```

Probe input:

```text
WM latent z/h
existing predictive state
regime probs
current/baseline position
adapter delta/fire
current drawdown
underwater duration
trailing return and volatility
equity/peak state
```

評価方法:

```text
fire bar only
chronological 60% train / 40% eval
ridge linear probe
binary labels: AUC, PR-AUC, top-decile positive rate, Brier
fire_advantage: correlation, top-decile realized advantage, top-bottom spread
```

## E Fold5 Result

Command:

```text
uv run python -m unidream.cli.fire_control_label_probe --config configs/trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 5 --seed 11 --device cuda --run E=checkpoints/acplan10_fire_selector_s011_on_wmbc_s011:ac --output-json documents/plan14_fire_control_labels_E_fold5.json --output-md documents/20260501_plan14_fire_control_labels_E_fold5.md
```

Policy:

| fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | 98.1% | 3.3%/288 | +0.1059 |

Label quality:

| h | fire samples | harm AUC | DD worse AUC | trough AUC | adv corr | adv top10 | adv spread |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 288 | 0.856 | 0.573 | 0.532 | -0.078 | +0.00043 | +0.00007 |
| 32 | 288 | 0.918 | 0.632 | 0.453 | +0.416 | +0.00323 | +0.00293 |

Interpretation:

```text
fire_harm: strong on fold5
drawdown_worsening: usable only on h32
trough_exit: not usable
fire_advantage: h32 is useful, h16 weak
```

E fold5 単体では、h32 の `fire_harm` / `drawdown_worsening` / `fire_advantage` はかなり良い。
ただし fold5 専用 checkpoint なので採用判断には使わない。

## Hbest Fold4/5/6 Result

Command:

```text
uv run python -m unidream.cli.fire_control_label_probe --config configs/trading_wm_control_headonly.yaml --start 2018-01-01 --end 2024-01-01 --folds 4,5,6 --seed 11 --device cuda --run Hbest=checkpoints/acplan13_wm_control_headonly_s011@ac_best.pt:ac --output-json documents/plan14_fire_control_labels_Hbest_folds456.json --output-md documents/20260501_plan14_fire_control_labels_Hbest_folds456.md
```

Policy:

| fold | AlphaEx | SharpeD | MaxDDD | turnover | long | short | flat | fire | fire_pnl |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | +0.05 | +0.017 | +0.35 | 5.69 | 2.7% | 0.0% | 97.3% | 11.4%/994 | -0.0110 |
| 5 | -28.10 | -0.064 | +0.75 | 2.62 | 3.0% | 0.0% | 97.0% | 12.1%/1065 | -0.0407 |
| 6 | -0.60 | -0.021 | +0.61 | 4.72 | 0.9% | 0.0% | 99.1% | 28.4%/2506 | -0.1802 |

Label quality:

| fold | h | fire samples | harm AUC | DD worse AUC | trough AUC | adv corr | adv top10 | adv spread |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 994 | 0.455 | 0.514 | 0.548 | 0.323 | +0.00118 | +0.00114 |
| 4 | 32 | 984 | 0.458 | 0.603 | 0.587 | 0.307 | +0.00152 | +0.00141 |
| 5 | 16 | 1050 | 0.619 | 0.426 | 0.450 | 0.151 | +0.00060 | +0.00061 |
| 5 | 32 | 1047 | 0.620 | 0.421 | 0.502 | 0.230 | +0.00088 | +0.00067 |
| 6 | 16 | 2506 | 0.739 | 0.558 | 0.549 | 0.241 | +0.00004 | +0.00023 |
| 6 | 32 | 2506 | 0.743 | 0.551 | 0.532 | 0.312 | +0.00006 | +0.00035 |

Gate readiness:

| criterion | result |
|---|---:|
| fire_harm AUC >= 0.58 on all folds/horizons | false |
| drawdown_worsening AUC >= 0.58 on all folds/horizons | false |
| trough_exit AUC >= 0.55 on all folds/horizons | false |
| fire_advantage top-decile > 0 on all folds/horizons | true |

## Decision

Plan14 direct labels are directionally useful, but not enough to proceed to WM head integration or AC expansion.

Adopt:

```text
fire_control_label_probe.py
```

Keep as diagnostic/research tooling.

Do not adopt into mainline policy yet:

```text
fire_harm guard
drawdown_worsening guard
trough_exit guard
WM control heads v2
additional AC unlock
```

Reason:

```text
fold5 E looks good, but fold4/5/6 Hbest does not reproduce the required separability.
fire_advantage is the only consistently positive signal.
harm / DD / trough labels are not stable enough across folds.
```

## Next Step

The next experiment should not be full AC.

Most reasonable next step:

```text
Plan15:
  refine direct labels and probe them again

Focus:
  fire_advantage_h32 as primary score
  replace trough_exit with simpler post-trough momentum/recovery label
  make drawdown_worsening horizon-relative and fold-calibrated
  evaluate low-harm ranking, not only high-harm AUC
```

Only if fold4/5/6 label quality passes should we proceed to:

```text
inference-only fire guard
then WM head-only v2
then sizing-adapter-only AC
```
