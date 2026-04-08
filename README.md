# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

Imagination-based reinforcement learning for crypto trading.

## Current Status

UniDream is a walk-forward crypto trading research project built around:

- a Dreamer-style world model
- behavior cloning from hindsight oracle paths
- imagination-based actor-critic fine-tuning
- backtesting against Buy & Hold with an explicit `M2` scorecard

The codebase has already been refactored into stage-oriented modules under [unidream/experiments](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/experiments), and source-rollout utilities have been centralized under [unidream/source_rollout](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/source_rollout).

## M2 Target

The current target is `M2`. All required conditions are `AND`.

- `alpha_excess >= +5pt/yr`
- `sharpe_delta >= +0.20`
- `maxdd_delta <= -10pt`
- `win_rate_vs_bh >= 60%`
- `collapse_guard = pass`

Stretch conditions are `OR`.

- `alpha_excess >= +8pt/yr`
- `maxdd_delta <= -15pt`

## Latest Results

### 1. Baseline Heavy Run: `medium_v2`

Config: [configs/medium_v2.yaml](C:/Users/Sophie/Documents/UniDream/UniDream/configs/medium_v2.yaml)  
Period: `2020-01-01 -> 2024-01-01`  
Walk-forward: `6 folds`  
Device: `cuda`

Aggregate:

| Metric | Value | Target | Status |
|---|---:|---:|---|
| `alpha_excess` | `-59.61 pt/yr` | `>= +5` | MISS |
| `sharpe_delta` | `-1.010` | `>= +0.20` | MISS |
| `maxdd_delta` | `-10.23 pt` | `<= -10` | PASS |
| `win_rate_vs_bh` | `49.3%` | `>= 60%` | MISS |
| `collapse_guard` | `pass` | `pass` | PASS |

Result: `M2 MISS`

Per-fold:

| Fold | `alpha_excess` | `sharpe_delta` | `maxdd_delta` | `win_rate_vs_bh` | Result |
|---|---:|---:|---:|---:|---|
| 0 | `+16.25 pt` | `-0.402` | `-21.75 pt` | `49.3%` | MISS |
| 1 | `-10.61 pt` | `-1.035` | `-8.64 pt` | `49.3%` | MISS |
| 2 | `-48.17 pt` | `-1.098` | `-10.72 pt` | `48.6%` | MISS |
| 3 | `-287.67 pt` | `-0.815` | `-9.94 pt` | `49.5%` | MISS |
| 4 | `-22.43 pt` | `-1.306` | `-5.01 pt` | `49.5%` | MISS |
| 5 | `-5.04 pt` | `-1.403` | `-5.34 pt` | `49.4%` | MISS |

Main diagnosis:

- final policy collapsed to effectively `short 100%`
- the current source/action family does not survive OOS
- `maxdd` can be improved, but `alpha`, `sharpe`, and `win rate` are still structurally weak

### 2. After WM Indent Fix + External Source Run

Config: [configs/medium_ext_sources.yaml](C:/Users/Sophie/Documents/UniDream/UniDream/configs/medium_ext_sources.yaml)

Latest observed summary:

- `Mean Sharpe = -1.457`
- `Aggregate alpha_excess = -62.67 pt/yr`
- `win_rate_vs_bh = 49.2%`
- positions are no longer stuck at `short 100%`
- `Fold 3` produced positive performance with `Sharpe = 2.146` and about `+13.8%` total return
- `Regime 2` was relatively healthy with `Sharpe = 1.544`
- `PBO = 0.50`

Main diagnosis:

- the WM fix improved policy movement, but did not fix OOS alpha
- `Regime 0` and `Regime 1` are still the main failure zones
- the model still fails to cut long exposure early enough during drawdown regimes
- external sources may help on sharp selloffs, but probably do not solve the full `M2` gap by themselves

## Current Working Hypothesis

The current bottleneck is not just optimizer tuning.

Most likely contributors are:

1. oracle quality is still weak in down-regime handling
2. BC prior is not strong enough before AC starts to drift
3. the world model may still underfit regime transitions
4. external sources such as `signed_order_flow` and `taker_imbalance` may help sharp selloff detection, but not fully fix low-volatility grind-down periods

Current priority for analysis:

- validate oracle action distribution by regime
- compare `medium_v2` vs `medium_ext_sources`
- measure how much external sources improve `Regime 0/1`

## Pipeline

The main training path is:

1. load cached features and returns
2. build walk-forward splits
3. compute hindsight oracle paths
4. train world model
5. behavior clone the oracle path
6. fine-tune with imagination AC
7. run validation selector
8. run test backtest and M2 scorecard

Implementation is now split across:

- [train.py](C:/Users/Sophie/Documents/UniDream/UniDream/train.py)
- [train_app.py](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/experiments/train_app.py)
- [train_pipeline.py](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/experiments/train_pipeline.py)
- [wm_stage.py](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/experiments/wm_stage.py)
- [bc_stage.py](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/experiments/bc_stage.py)
- [ac_stage.py](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/experiments/ac_stage.py)
- [val_selector_stage.py](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/experiments/val_selector_stage.py)
- [test_stage.py](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/experiments/test_stage.py)

## Main Entry Points

### Full BC/AC Pipeline

```bash
uv run python train.py
```

Examples:

```bash
# smoke
uv run python train.py --config configs/smoke_test.yaml --start 2022-01-01 --end 2023-06-01

# medium_v2 all-fold run
uv run python train.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01 --device cuda

# resume
uv run python train.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01 --device cuda --resume

# run only selected folds
uv run python train.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01 --folds 0,1,4 --device cuda

# test-only from checkpoints
uv run python train.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01 --start-from test --stop-after test --resume --device cuda
```

### Risk Controller Probes

```bash
uv run python train_risk_controller.py --config configs/smoke_risk_controller_v5_basis.yaml --start 2021-01-01 --end 2023-06-01
```

### Event Controller Probes

```bash
uv run python train_event_controller.py --config configs/smoke_event_controller_v3_triplebarrier.yaml --start 2021-01-01 --end 2023-06-01
```

### QDT Baseline

```bash
uv run python train_qdt.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01
```

## Source Rollout

Source-family evaluation is now structured around:

- [configs/source_rollout_suite.yaml](C:/Users/Sophie/Documents/UniDream/UniDream/configs/source_rollout_suite.yaml)
- [configs/source_rollout_suite_free.yaml](C:/Users/Sophie/Documents/UniDream/UniDream/configs/source_rollout_suite_free.yaml)
- [unidream/source_rollout/plan.py](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/source_rollout/plan.py)
- [unidream/source_rollout/requirements.py](C:/Users/Sophie/Documents/UniDream/UniDream/unidream/source_rollout/requirements.py)

Useful commands:

```powershell
# rollout diagnostics
powershell -ExecutionPolicy Bypass -File .\scripts\run_source_rollout_checks.ps1 `
  -CacheDir checkpoints\aux_smoke2 `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2

# free source rollout
powershell -ExecutionPolicy Bypass -File .\scripts\run_free_source_rollout_end_to_end.ps1

# source family suite
powershell -ExecutionPolicy Bypass -File .\scripts\run_source_family_suite.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

Documentation:

- [docs/source_rollout_workflow.md](C:/Users/Sophie/Documents/UniDream/UniDream/docs/source_rollout_workflow.md)
- [docs/free_source_rollout.md](C:/Users/Sophie/Documents/UniDream/UniDream/docs/free_source_rollout.md)
- [docs/source_cache_formats.md](C:/Users/Sophie/Documents/UniDream/UniDream/docs/source_cache_formats.md)

## Repository Structure

```text
UniDream/
|- README.md
|- train.py
|- train_qdt.py
|- train_risk_controller.py
|- train_event_controller.py
|- train_ppo.py
|- build_*_source_cache.py
|- build_source_cache_from_manifest.py
|- configs/
|- docs/
|- scripts/
|- tests/
|- checkpoints/
`- unidream/
   |- actor_critic/
   |- baselines/
   |- data/
   |- eval/
   |- experiments/
   |- lore/
   |- online/
   |- source_rollout/
   `- world_model/
```

## Testing

```bash
uv run python -m pytest tests/
```

For source-rollout checks:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_source_rollout_checks.ps1 `
  -CacheDir checkpoints\aux_smoke2 `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

## Notes

- The current best evidence still says `M2` is not close.
- External sources are worth testing, but oracle quality and regime handling remain first-order problems.
- The repository is currently in a better engineering state than the strategy is in a better trading state.
