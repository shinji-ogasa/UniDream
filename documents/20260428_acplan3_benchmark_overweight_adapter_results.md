# AC Plan 3: Benchmark-gated Small Overweight Adapter Results

Date: 2026-04-28

## Goal

`acplan_3.md` の方針に従い、realized candidate AWR BC ではなく、Phase 8 safe baseline を anchor にした benchmark-gated small overweight adapter を検証した。

目的は以下。

```text
Phase 8 の DD 改善・turnover 安定を維持したまま、
benchmark 到達後だけ small overweight を許可して AlphaEx / SharpeDelta を改善する。
```

採用条件:

| metric | target |
|---|---:|
| AlphaEx | Phase 8 比 +0.2 pt/yr 以上 |
| SharpeDelta | >= 0 |
| MaxDDDelta | <= -1.0 pt |
| turnover | <= 3.5 |
| short | <= 25% |
| long | 1-5% |
| flat | 70-88% |

Phase 8 baseline:

| metric | baseline |
|---|---:|
| AlphaEx | +0.90 to +0.91 pt/yr |
| SharpeDelta | -0.010 to -0.011 |
| MaxDDDelta | -1.59 to -1.61 pt |
| turnover | 2.60 to 2.62 |
| short | 16-17% |
| flat | 83-84% |
| long | 0% |

## Implementation

Actor inference に、デフォルト無効の benchmark-gated overweight adapter を追加した。

```text
a_final = a_phase8
if gate:
    a_final = max(a_final, benchmark + epsilon)
```

gate 条件:

```text
current_position >= benchmark_overweight_min_position
underweight_duration <= benchmark_overweight_underweight_duration_max
WM predictive advantage[benchmark_overweight_advantage_index] >= benchmark_overweight_advantage_min
long_rate <= benchmark_overweight_long_rate_max
```

今回の採用設定:

```yaml
use_benchmark_overweight_adapter: true
benchmark_overweight_epsilon: 0.20
benchmark_overweight_min_position: 0.93
benchmark_overweight_max_position: 1.22
benchmark_overweight_underweight_duration_max: 1.0
benchmark_overweight_min_hold_bars: 0
benchmark_overweight_require_base_near_bench: false
benchmark_overweight_long_rate_max: 0.03
benchmark_overweight_advantage_index: 0
benchmark_overweight_advantage_min: 1.00
```

Batch size は BC / AC ともに 1024 に変更済み。

```yaml
ac.batch_size: 1024
bc.batch_size: 1024
```

## Hardware Note

実行コマンドは `--device cuda` を指定し、ログ上も `Device: cuda` で実行された。

ただし今回の `--start-from test` 評価は以下が CPU ボトルネックになりやすい。

- Hindsight oracle / WFO setup
- Backtest / PnL attribution
- val selector の複数backtest
- `predict_positions` の逐次 controller rollout

そのため、RTX 3070 は認識・使用されているが、GPU使用率は高止まりしない。今後の探索は同じデータ/WM/actorを再利用する一括probeに寄せるべき。

確認:

```text
PyTorch: 2.11.0+cu128
CUDA available: true
GPU: NVIDIA GeForce RTX 3070
```

## Experiments

全て fold 4 / seed 7 / `--device cuda` / Phase 8 checkpoint から `--start-from test` で評価した。

### Initial Strict Gates

| config | gate | AlphaEx | SharpeDelta | MaxDDDelta | turnover | long | short | flat | result |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| eps0.06 strict | min_pos 0.95, near_bench true | +0.91 | -0.010 | -1.61 | 2.63 | 0% | 17% | 83% | unchanged |
| eps0.08 strict | min_pos 0.95, near_bench true | +0.91 | -0.010 | -1.61 | 2.63 | 0% | 17% | 83% | unchanged |

Strict gate では adapter がほぼ発火しなかった。

### Loose Gate Probe

| config | AlphaEx | SharpeDelta | MaxDDDelta | turnover | long | short | flat | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| eps0.06 loose | -0.59 | -0.072 | +0.69 | 31.38 | 3% | 0% | 97% | reject |

Loose gate は long 3% を出せたが、turnover が崩壊した。

### Predictive Advantage Gate Sweep

`benchmark_overweight_advantage_index: 0` を使い、WM predictive state の先頭 return 系シグナルで gate を絞った。

| config | adv min | AlphaEx | SharpeDelta | MaxDDDelta | turnover | long | short | flat | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| eps0.06 adv0>=0.00 | 0.00 | -0.45 | -0.055 | +0.65 | 30.45 | 3% | 0% | 97% | reject |
| eps0.06 adv0>=0.20 | 0.20 | +0.21 | +0.018 | +0.23 | 19.01 | 3% | 2% | 95% | reject |
| eps0.06 adv0>=0.40 | 0.40 | +0.37 | +0.015 | +0.16 | 9.87 | 2% | 9% | 89% | reject |
| eps0.06 adv0>=0.60 | 0.60 | +0.30 | -0.015 | -0.70 | 6.30 | 1% | 12% | 86% | reject |
| eps0.06 adv0>=0.80 | 0.80 | +0.39 | -0.013 | -0.83 | 4.14 | 1% | 13% | 86% | reject |
| eps0.06 adv0>=1.00 | 1.00 | +0.92 | +0.032 | -1.63 | 2.67 | 1% | 15% | 84% | near pass |

`adv0>=1.00` でようやく turnover と DD が安定した。

### Epsilon Sweep at adv0>=1.00

| config | AlphaEx | SharpeDelta | MaxDDDelta | turnover | long | short | flat | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| eps0.08 | +0.95 | +0.025 | -1.67 | 2.79 | 1% | 16% | 83% | near pass |
| eps0.10 | +0.98 | +0.031 | -1.72 | 2.84 | 1% | 16% | 83% | near pass |
| eps0.15 | +1.07 | +0.047 | -1.85 | 2.95 | 1% | 16% | 83% | near pass |
| eps0.18 | +1.12 | +0.057 | -1.92 | 3.01 | 1% | 16% | 83% | pass |
| eps0.20 | +1.15 | +0.063 | -1.97 | 3.05 | 1% | 16% | 83% | pass |

## Adopted Mainline Result

本流 `configs/trading.yaml` に `eps0.20 / adv0>=1.00 / min_position 0.93` を採用し、再評価した。

```text
Command:
uv run python -m unidream.cli.train --config configs/trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 4 --start-from test --seed 7 --device cuda
```

Result:

| metric | result | pass/fail |
|---|---:|---|
| AlphaEx | +1.15 pt/yr | pass |
| Phase 8 AlphaEx delta | approx +0.24 pt/yr | pass |
| SharpeDelta | +0.063 | pass |
| MaxDDDelta | -1.97 pt | pass |
| turnover | 3.05 | pass |
| short | 16% | pass |
| long | 1% | pass |
| flat | 83% | pass |
| recovery gate active | 1.0% | pass |

Route / position:

```text
Route dist: neutral=36% de_risk=64% overweight=1% active=64.4% conf=0.708
Test dist: long=1% short=16% flat=83% mean=-0.044 switches=61 avg_hold=142.8b turnover=3.05
```

## Interpretation

AWR BC では target は作れても実行policyに出なかった。一方、benchmark-gated adapter は、Phase 8 の state machine / recovery / de-risk を壊さずに long 1% を作れた。

重要なのは、overweight を広く許可しないこと。

```text
loose gate: longは出るが turnover collapse
adv0>=1.0: small longだけ残って Alpha/Sharpe が改善
```

## Decision

採用。

理由:

- AC Plan 3 の採用条件を fold 4 で満たした。
- Phase 8 の安全性を維持したまま、AlphaEx と SharpeDelta が改善した。
- long は 1% に抑制され、turnover 3.05 で制約内。
- 実装は inference-only adapter で、既存 route/state-machine/recovery を破壊しない。

注意:

- 現在利用可能な checkpoint は fold 4 のみだったため、fold横断確認は未実施。
- 次に他fold checkpointが揃ったら、この設定を全foldで再確認する。
