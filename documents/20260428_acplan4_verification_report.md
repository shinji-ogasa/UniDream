# AC Plan 4 Verification Report

Date: 2026-04-28
Config: `configs/trading.yaml`
Checkpoint dir: `checkpoints/bcplan5_phase8_state_machine_s007`
Device: CUDA
Seed: 7
Period: 2018-01-01 to 2024-01-01

## 結論

`Phase 8 + benchmark-gated overweight adapter + predictive advantage gate` は、fold4単体ではPhase 8を壊さずに上振れを足せている。
ただし、fold0/fold5展開では堅牢性が不足しており、AC Plan 4時点ではこれ以上の制限解除、full actor unlock、route head unlock、adapter trainable化は採用しない。

本流採用済みの current adapter は維持するが、`epsilon=0.25` や `long_rate_max=0.01` などfold4だけで良く見えた変更は未採用。
理由は、fold4単体最適化に寄せるとfold外でのB&H劣後を説明できないため。

## 1. Fold4 Ablation / Sweep

実行条件:

```powershell
uv run python -m unidream.cli.train --config configs\trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 4 --start-from test --seed 7 --device cuda
```

CLI baseline result:

| fold | setting | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short | flat |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | current adapter | +1.15 | +0.063 | -1.97 | 3.05 | 1% | 16% | 83% |

Ablation/sweep was run by loading the same WM/BC checkpoint once and changing only inference adapter settings.

| variant | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short | adapter fires |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase8 adapter off | +0.90 | -0.011 | -1.59 | 2.63 | 0.0% | 16.6% | 0 |
| adapter on, adv gate off | -1.78 | -0.239 | +2.26 | 104.60 | 3.0% | 0.0% | 8708 |
| current gate on | +1.13 | +0.059 | -1.94 | 3.06 | 1.1% | 15.6% | 1405 |
| adv_min 0.5 | +0.75 | +0.065 | +0.23 | 20.97 | 2.2% | 9.9% | 5021 |
| adv_min 1.0 | +1.13 | +0.059 | -1.94 | 3.06 | 1.1% | 15.6% | 1405 |
| adv_min 1.5 | +0.90 | -0.011 | -1.59 | 2.63 | 0.0% | 16.6% | 0 |
| epsilon 0.10 | +0.96 | +0.026 | -1.69 | 2.85 | 1.1% | 15.6% | 1239 |
| epsilon 0.15 | +1.05 | +0.042 | -1.81 | 2.96 | 1.1% | 15.7% | 2469 |
| epsilon 0.20 | +1.13 | +0.059 | -1.94 | 3.06 | 1.1% | 15.6% | 1405 |
| epsilon 0.25 | +1.22 | +0.075 | -2.06 | 3.18 | 1.1% | 15.6% | 1319 |
| long_rate_max 0.01 | +1.25 | +0.047 | -2.10 | 3.04 | 0.4% | 15.9% | 862 |
| long_rate_max 0.03 | +1.13 | +0.059 | -1.94 | 3.06 | 1.1% | 15.6% | 1405 |
| long_rate_max 0.05 | +0.69 | +0.022 | -1.28 | 3.06 | 1.9% | 15.7% | 1581 |
| max_position 1.10 | +0.96 | +0.026 | -1.69 | 2.85 | 1.1% | 15.6% | 1239 |
| max_position 1.15 | +1.05 | +0.042 | -1.81 | 2.96 | 1.1% | 15.7% | 2469 |
| max_position 1.22 | +1.13 | +0.059 | -1.94 | 3.06 | 1.1% | 15.6% | 1405 |

Interpretation:

- Advantage gateなしは即失格。turnover 104.60でadapterがほぼ常時介入し、Phase 8の安全性を破壊した。
- `adv_min=1.0` は必要。`0.5` はturnover 20.97、`1.5` はadapterが実質停止。
- `epsilon=0.25` と `long_rate_max=0.01` はfold4だけなら良いが、fold展開前に採用するほどの根拠はない。
- 現行 `epsilon=0.20 / adv_min=1.0 / long_rate_max=0.03` はfold4では安全性と改善のバランスが最も無難。

## 2. Adapter Fire Attribution

Current gate-on fold4:

| metric | value |
|---|---:|
| adapter fire count | 1405 bars |
| fire rate | 16.1% |
| mean position increment on fire | +0.0255 |
| mean WM predictive advantage[0] on fire | +0.0797 |
| mean realized same-bar return on fire | -0.000017 |

Fire rate自体は16%あるが、long stateとして残るのは約1.1%。つまりadapterは大きなlong転換ではなく、benchmark近傍で小さくpositionを押し上げる補正として働いている。

## 3. Stress Test on Fold4 Current Adapter

| stress | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short |
|---|---:|---:|---:|---:|---:|---:|
| base cost | +1.13 | +0.059 | -1.94 | 3.06 | 1.1% | 15.6% |
| cost x1.5 | +1.11 | +0.056 | -1.91 | 3.06 | 1.1% | 15.6% |
| cost x2.0 | +1.08 | +0.052 | -1.87 | 3.06 | 1.1% | 15.6% |
| slippage x2.0 | +1.12 | +0.058 | -1.93 | 3.06 | 1.1% | 15.6% |

Fold4上ではcost stressに対して壊れていない。turnoverが3付近なので、cost x2でも劣化は限定的。

## 4. Fold Expansion

fold0/fold5 checkpointがなかったため、本流設定で新規にWM/BCを学習した。

実行:

```powershell
uv run python -m unidream.cli.train --config configs\trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 0,5 --seed 7 --device cuda
```

追加でfold5は保存済みcheckpointから単独retestした。

```powershell
uv run python -m unidream.cli.train --config configs\trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 5 --start-from test --seed 7 --device cuda
```

Final fold results used for judgement:

| fold | AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short | flat | judgement |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | -5.13 | -0.001 | -0.28 | 0.18 | 0% | 0% | 100% | fail: almost benchmark, misses upside |
| 4 | +1.15 | +0.063 | -1.97 | 3.05 | 1% | 16% | 83% | pass |
| 5 | -51.36 | -0.002 | -0.83 | 1.94 | 2% | 0% | 98% | fail: severe B&H underperformance |

3fold average:

| AlphaEx pt/yr | SharpeD | MaxDDD pt | turnover | long | short | flat |
|---:|---:|---:|---:|---:|---:|---:|
| -18.45 | +0.020 | -1.03 | 1.72 | 1.0% | 5.3% | 93.7% |

Fold expansion judgement:

- 3fold average AlphaEx is negative and fails the roadmap condition.
- 3fold中2foldでPhase8を上回る条件も満たせない。
- MaxDD and turnover are acceptable, but this is because fold0/fold5 are too close to benchmark/flat and miss upside.
- The adapter is not the main cause of fold0/fold5 failure; the base Phase 8 policy itself becomes too passive in these regimes.

## 5. Implementation / Repro Notes

- Current mainline still runs Phase 8 BC because `ac.max_steps: 0`.
- Current mainline includes the adopted adapter:
  - `use_benchmark_overweight_adapter: true`
  - `benchmark_overweight_epsilon: 0.20`
  - `benchmark_overweight_advantage_min: 1.00`
  - `benchmark_overweight_long_rate_max: 0.03`
- No additional Plan 4 sweep setting was adopted.
- AC curriculum execution support is already implemented for future staged AC runs, but AC is not enabled in the current config.

## 6. Decision

Adopt / keep:

- Phase 8 state machine BC mainline.
- benchmark-gated overweight adapter with predictive advantage gate.
- hard safety constraints, state machine, neutral fallback behavior.

Reject / do not adopt now:

- advantage gate off.
- `adv_min=0.5`.
- `epsilon=0.25` despite fold4 improvement.
- `long_rate_max=0.01` despite fold4 Alpha improvement.
- full actor unlock.
- route head full unlock.
- trainable overweight adapter.

Next useful work:

1. Diagnose why fold0/fold5 become too benchmark-like and miss B&H upside.
2. Compare Phase 8 base vs adapter on newly trained fold0/fold5 with a deterministic evaluation path.
3. Add run-order independent initialization/load checks for optional actor heads if old checkpoints are reused.
4. Only after 3fold AlphaEx turns positive, revisit restricted AC around benchmark-overweight sizing.
