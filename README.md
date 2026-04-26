# UniDream

BTCUSDT 15分足を対象に、Transformer World Model、Behavior Cloning、Imagination Actor-Critic を検証する研究プロジェクトです。

現在の本流は Phase 8 safe baseline です。通常実行は `configs/trading.yaml` を使います。

## Pipeline

```text
OHLCV / features
  -> Walk-Forward split
  -> Hindsight Oracle / signal_aim teacher
  -> Transformer World Model
  -> WM predictive state bundle (return / vol / drawdown)
  -> Behavior Cloning (route head + inventory recovery + state machine)
  -> Imagination Actor-Critic
  -> Validation selector
  -> Test backtest / M2 scorecard / PBO / regime report
```

## Setup

```bash
uv sync
uv run python -m unidream.cli.train --help
```

## Main Run

Default config is `configs/trading.yaml`, so the latest production path can run without a config argument.

```bash
uv run python -m unidream.cli.train --folds 4 --device auto
```

Period / stage / resume controls remain available:

```bash
uv run python -m unidream.cli.train --start 2018-01-01 --end 2024-01-01 --folds 4 --device auto
uv run python -m unidream.cli.train --stop-after wm --folds 4 --device auto
uv run python -m unidream.cli.train --start-from bc --stop-after bc --resume --folds 4 --device auto
uv run python -m unidream.cli.train --start-from ac --stop-after ac --resume --folds 4 --device auto
uv run python -m unidream.cli.train --start-from test --resume --folds 4 --device auto
```

## Diagnostics

```bash
uv run python -m unidream.cli.route_probe --config configs/trading.yaml --folds 4
uv run python -m unidream.cli.transition_advantage_probe --config configs/trading.yaml --folds 4
uv run python -m unidream.cli.wm_probe --config configs/trading.yaml --folds 4
uv run python -m unidream.cli.ac_candidate_q_probe --config configs/trading.yaml --folds 4
```

## Active Config

- `configs/trading.yaml`: Phase 8 safe baseline. It replaces the old experiment YAML family.

Historical experiment YAMLs were removed from `configs/`. Results and rationale remain in `documents/`.

## Current Baseline

Fold 4 / seed 7 / 2018-01-01 to 2024-01-01:

```text
short 16-17%
flat 83-84%
turnover 2.60-2.62
AlphaEx +0.90 to +0.91 pt/yr
SharpeDelta -0.010 to -0.011
MaxDDDelta -1.59 to -1.61 pt
recovery gate active 0.7-0.9%
```

## Generated Files

- `checkpoints/<logging.checkpoint_dir>/fold_<i>/{world_model.pt, bc_actor.pt, ac.pt}`
- `checkpoints/data_cache/`
- `documents/logs/`, `documents/route_probe/`, `documents/wm_probe/`, `documents/ac_candidate_q/`

`checkpoints/` and `.venv/` are not committed.

## Notes

This repository is for research. It is not investment advice or a production trading system.
