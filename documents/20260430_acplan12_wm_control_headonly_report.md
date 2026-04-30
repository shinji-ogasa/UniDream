# AC Plan12: WM Retraining / DD Control Result

Date: 2026-04-30

## Purpose

Test whether retraining TransformerWM can improve AC drawdown behavior, especially the Plan11 E/current issue where the overweight/fire adapter produced high AlphaEx but left MaxDDDelta positive.

Baseline references on fold5:

| run | AlphaEx | SharpeDelta | MaxDDDelta | turnover | long | short | note |
|---|---:|---:|---:|---:|---:|---:|---|
| Plan11 E/current | +62.09 | +0.050 | +0.04 | 1.18 | 1.9% | 0.0% | high alpha, fails MaxDD <= 0 |
| Phase8 safe baseline | about +0.90 | about -0.010 | about -1.60 | about 2.61 | 0.0% | 16-17% | safe but low alpha |

## Implementation

Added AC-control-oriented WM auxiliary heads:

- `overweight_advantage_head`
- `recovery_head`

The new targets are computed from non-leaky forward return paths during WM training:

```text
overweight_advantage_h
= overweight_delta * future_return_h
  - overweight_delta * drawdown_penalty * future_downside_h
  - one_way_cost

recovery_h
= normalized(future_return_h - recovery_drawdown_penalty * future_downside_h)
```

Also added:

- `world_model.init_checkpoint`
- `world_model.freeze_ensemble`
- `world_model.freeze_standard_predictive_heads`
- configurable `world_model.num_workers`
- configurable `bc.num_workers`

The worker settings are needed because this sandbox rejects Windows multiprocessing pipes.

## Experiments

### A. Full WM Retrain

Config: full WM retrain with `return/vol/drawdown/overweight_advantage/recovery` predictive state.

Result: rejected.

| threshold | AlphaEx | SharpeDelta | MaxDDDelta | turnover | fire | judgment |
|---:|---:|---:|---:|---:|---:|---|
| 0.25 default-ish | -176.05 | -0.232 | +0.67 | 45.94 | 32.1% | collapsed by churn |
| 0.75 | -35.37 | -0.093 | +1.19 | 1.68 | 9.6% | negative alpha |
| 1.00 | -2.37 | -0.014 | +0.16 | 0.96 | 2.6% | too weak |
| 1.25 | +5.67 | +0.001 | +0.04 | 0.59 | 0.0% | adapter off |

Interpretation: moving the whole WM representation broke the existing Phase8/Plan10 behavior. This should not be adopted.

### B. Head-only WM Fine-tune

Config: initialize from existing good WM and freeze the ensemble plus standard heads.

```yaml
world_model:
  init_checkpoint: checkpoints/acplan10_fire_selector_s011_on_wmbc_s011/fold_5/world_model.pt
  freeze_ensemble: true
  freeze_standard_predictive_heads: true
  overweight_advantage_scale: 2.0
  recovery_scale: 1.0
  max_steps: 800

ac:
  wm_predictive_state_heads:
    - return
    - vol
    - drawdown
    - overweight_advantage
    - recovery
  benchmark_overweight_advantage_index: 18
```

This preserves the existing WM representation and only adds DD-aware control signals.

## Best Fold5 Result

Best deployable-ish setting found:

```yaml
benchmark_overweight_advantage_min: 0.55
benchmark_overweight_epsilon: 0.06
benchmark_overweight_trainable_delta_range: 0.03
```

| run | AlphaEx | SharpeDelta | MaxDDDelta | turnover | long | short | fire | mean_delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| head-only WM, thr=0.55, eps=0.06 | +39.00 | +0.026 | -0.00 | 2.55 | 0% | 0% | 17.4% | +0.0188 |
| head-only WM, thr=0.55, eps=0.07 | +48.17 | +0.031 | +0.02 | 3.11 | 1% | 0% | 18.8% | +0.0224 |
| head-only WM, thr=0.55, eps=0.08 | +57.57 | +0.038 | +0.02 | 3.67 | 3% | 0% | 17.2% | +0.0267 |
| head-only WM, thr=0.575, eps=0.08 | +40.39 | +0.023 | +0.14 | 2.71 | 2% | 0% | 14.9% | +0.0256 |
| head-only WM, thr=0.60, eps=0.09 | +36.64 | n/a | n/a | 1.77 | 2% | 0% | 9.0% | +0.0302 |

The best strict candidate is `thr=0.55, eps=0.06` because it keeps turnover under 3.5 and approximately removes MaxDD worsening.

## Interpretation

WM retraining helps only when constrained to head-only fine-tuning.

What improved:

- MaxDDDelta moved from Plan11 E/current `+0.04` to approximately `-0.00`.
- Turnover stayed controlled at `2.55`.
- AlphaEx remained meaningful at `+39.00 pt/yr`.
- The adapter no longer needs large deltas; mean adapter delta was only `+0.0188`.

What did not improve enough:

- It does not beat Plan11 E/current on AlphaEx.
- Slightly higher epsilon gives more AlphaEx, but MaxDDDelta becomes positive again.
- Fold5 only; not enough to adopt into main `configs/trading.yaml` yet.

## Decision

Adopt the code support for WM control heads and head-only WM fine-tuning.

Do not replace the mainline config yet. The candidate must be checked on multiple folds first.

Recommended next validation:

```powershell
.\.venv\Scripts\python.exe -m unidream.cli.train --config configs/trading_wm_control_headonly.yaml --start 2018-01-01 --end 2024-01-01 --folds 4,5,6 --seed 11 --device cuda
```

Current candidate config:

```text
configs/trading_wm_control_headonly.yaml
```

## Files Changed

- `unidream/world_model/train_wm.py`
- `unidream/experiments/wm_stage.py`
- `unidream/actor_critic/bc_pretrain.py`
- `unidream/experiments/bc_stage.py`
- `configs/trading_wm_control_headonly.yaml`

## Commit Status

Commit was not created because `.git` is currently not writable in this sandbox session:

```text
fatal: Unable to create .git/index.lock: Permission denied
```
