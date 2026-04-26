# UniDream Spec

## Scope

UniDream currently supports one production training path plus diagnostic probes.

```text
OHLCV/features
  -> WFO split
  -> Hindsight Oracle / signal_aim teacher
  -> Transformer World Model
  -> WM predictive state bundle (return / vol / drawdown)
  -> Behavior Cloning (route head + inventory recovery + state machine gate)
  -> Imagination Actor-Critic (critic-only / restricted actor unlock when enabled)
  -> Validation selector
  -> Test backtest + M2 scorecard + PBO + regime report
```

Retired teacher/config blocks were removed. Feature-stress logic remains only for regime diagnostics such as `eval.regime_source: feature_stress_tri`.

## Entrypoints

Production training uses `configs/trading.yaml` by default:

```bash
uv run python -m unidream.cli.train --folds 4 --device auto
```

CLI flags kept for the main path:

- `--config`: optional override, default `configs/trading.yaml`
- `--start` / `--end`: data period
- `--start-from {wm,bc,ac,test}` / `--stop-after {wm,bc,ac,test}`: stage gating
- `--resume`: load existing fold checkpoints
- `--folds`: comma-separated subset, for example `--folds 4` or `--folds 0,4,5`
- `--seed`: random seed
- `--device {auto,cpu,cuda,mps}`

Diagnostic probes:

```bash
uv run python -m unidream.cli.route_probe --config configs/trading.yaml --folds 4
uv run python -m unidream.cli.transition_advantage_probe --config configs/trading.yaml --folds 4
uv run python -m unidream.cli.wm_probe --config configs/trading.yaml --folds 4
uv run python -m unidream.cli.ac_candidate_q_probe --config configs/trading.yaml --folds 4
```

## Pipeline Modules

CLI:

- `unidream/cli/train.py`: argparse entrypoint, fold dispatcher, selector / score helpers
- `unidream/cli/route_probe.py`: route classification diagnostics
- `unidream/cli/transition_advantage_probe.py`: realized advantage per candidate action / route
- `unidream/cli/wm_probe.py`: linear probe over WM latents
- `unidream/cli/ac_candidate_q_probe.py`: state-action critic ranking probe

Experiment runtime:

- `runtime.py`: config loading, cost resolution, feature/cache loading, seed
- `train_app.py`: top-level run orchestration
- `train_pipeline.py`: WFO fold loop
- `fold_runtime.py`: stage gating and checkpoint paths
- `fold_inputs.py`: oracle / regime / advantage / route inputs
- `oracle_stage.py` / `oracle_teacher.py` / `oracle_post.py`: DP base + signal_aim teacher + post-processing
- `transition_advantage.py`: route targets, transition advantages, recovery latency
- `regime_runtime.py`: HMM / feature-stress regime fitting
- `predictive_state.py`: WM auxiliary head bundle
- `wm_stage.py`: Transformer World Model training / loading
- `bc_setup.py`: actor construction
- `bc_stage.py`: BC training / loading
- `ac_stage.py`: Imagination AC
- `val_selector_stage.py`: validation policy selection
- `test_stage.py`: test backtest / M2 scorecard

Models:

- `unidream/world_model/transformer.py`: Transformer dynamics with predictive auxiliary heads
- `unidream/actor_critic/actor.py`: inventory controller actor with route / recovery / state-machine controls
- `unidream/actor_critic/critic.py`: value critic
- `unidream/actor_critic/state_action_critic.py`: candidate Q probe model
- `unidream/actor_critic/bc_pretrain.py`: BC trainer
- `unidream/actor_critic/imagination_ac.py`: Dreamer-style lambda-return AC + TD3+BC regularization

## Active Config

- `configs/trading.yaml`: current Phase 8 safe baseline and default train config.

All previous experiment YAMLs were removed from `configs/`. Historical results remain in `documents/`.

## Current Baseline

`configs/trading.yaml` on fold 4 / seed 7 / 2018-01-01 to 2024-01-01:

```text
short 16-17%
flat 83-84%
turnover 2.60-2.62
AlphaEx +0.90 to +0.91 pt/yr
SharpeDelta -0.010 to -0.011
MaxDDDelta -1.59 to -1.61 pt
recovery gate active 0.7-0.9%
```

AC full actor unlock remains forbidden until restricted residual / recovery / sizing experiments satisfy safety gates across validation and test.

## Generated Files

- `checkpoints/<logging.checkpoint_dir>/fold_<i>/{world_model.pt, bc_actor.pt, ac.pt}`
- `checkpoints/data_cache/`
- `documents/logs/<date>_<run>_fold<i>.log`
- `documents/route_probe/`, `documents/wm_probe/`, `documents/ac_candidate_q/`

`checkpoints/` and `.venv/` are generated locally and are not part of the source tree.
