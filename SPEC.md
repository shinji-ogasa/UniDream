# UniDream Spec

## Scope

UniDream currently contains one supported path:

```text
OHLCV/features -> WFO split -> Hindsight Oracle -> World Model -> BC -> Imagination AC -> validation selector -> test backtest
```

Historical experiment families and compatibility entrypoints have been removed.

## Entrypoint

```bash
uv run python -m unidream.cli.train --config configs/smoke_test.yaml --start 2022-01-01 --end 2023-06-01
```

## Pipeline Modules

- `unidream/cli/train.py`: CLI argument parsing and high-level fold execution
- `unidream/experiments/fold_runtime.py`: stage gating and checkpoint paths
- `unidream/experiments/fold_inputs.py`: oracle/regime/advantage fold inputs
- `unidream/experiments/wm_stage.py`: world model training/loading
- `unidream/experiments/bc_setup.py`: actor construction and BC setup
- `unidream/experiments/bc_stage.py`: BC training/loading
- `unidream/experiments/ac_stage.py`: imagination AC
- `unidream/experiments/val_selector_stage.py`: validation policy selection
- `unidream/experiments/test_stage.py`: test backtest and scorecard

## Active Configs

- `configs/smoke_test.yaml`
- `configs/trading.yaml`
- `configs/medium_v2.yaml`
- current BC stress-regime configs in `configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_*dualresanchor_stresstri_shiftonly_s007.yaml`

## Generated Files

`checkpoints/` contains generated model checkpoints, cached data, and reports. It is not part of the source tree and can be recreated by rerunning the pipeline.
