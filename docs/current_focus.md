# Current Focus

UniDream is now scoped to the main World Model -> BC -> AC trading pipeline.

Removed surfaces:

- historical source-rollout tooling
- model-free PPO / QDT / risk-controller / event-controller experiments
- top-level compatibility scripts
- old issue-runner PowerShell scripts
- generated checkpoints and audit outputs

Active entrypoint:

```bash
uv run python -m unidream.cli.train --config configs/smoke_test.yaml --start 2022-01-01 --end 2023-06-01 --device auto
```

Active code areas:

- `unidream/cli`: command-line entrypoints
- `unidream/data`: OHLCV loading, feature construction, oracle labels
- `unidream/world_model`: Transformer world model
- `unidream/actor_critic`: actor, critic, BC pretraining, imagination AC
- `unidream/experiments`: pipeline stages and fold orchestration
- `unidream/eval`: backtest, WFO, PBO, regime helpers

Near-term work is BC prior optimization around the stress-regime dual residual controller.
