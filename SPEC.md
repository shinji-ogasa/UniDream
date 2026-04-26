# UniDream Spec

## Scope

UniDream currently supports one production training path plus a set of diagnostic probes:

```text
OHLCV/features
  -> WFO split
  -> Hindsight Oracle (signal_aim teacher / feature_stress / dual)
  -> Transformer World Model
  -> WM predictive state bundle (return / vol / drawdown / regime)
  -> Behavior Cloning (route head + inventory recovery + state machine gate)
  -> Imagination Actor-Critic (critic-only / restricted actor unlock)
  -> Validation selector
  -> Test backtest + M2 scorecard + PBO + HMM regime report
```

Historical experiment families and compatibility entrypoints have been removed. Documents under `documents/` track the latest experimental state — `documents/20260426_bcplan5_phase9_constrained_ac_results.md` is the current top-level summary.

## Entrypoints

Production training:

```bash
uv run python -m unidream.cli.train --config configs/<run>.yaml --start 2018-01-01 --end 2024-01-01 --folds 4
```

CLI flags:

- `--start-from {wm,bc,ac,test}` / `--stop-after {wm,bc,ac,test}`: stage gating
- `--resume`: pick up from existing fold checkpoints
- `--folds`: comma-separated subset (e.g. `--folds 4` or `0,4,5`)
- `--cost-profile`: select from `cost_profiles` block (default = `base`)
- `--device {auto,cpu,cuda,mps}`

Diagnostic probes (used before promoting a BC checkpoint to AC):

```bash
uv run python -m unidream.cli.route_probe --config <bc-config> --folds 4
uv run python -m unidream.cli.transition_advantage_probe --config <bc-config> --folds 4
uv run python -m unidream.cli.wm_probe --config configs/wm_probe_multitask_aux_s007.yaml --folds 4
uv run python -m unidream.cli.ac_candidate_q_probe --config <bc-config> --folds 4
```

## Pipeline Modules

CLI:

- `unidream/cli/train.py` — argparse entrypoint, fold dispatcher, selector / score helpers
- `unidream/cli/route_probe.py` — 3-value / 4-value route classification metrics (CE, Macro-F1, per-route recall, ECE, top-decile advantage)
- `unidream/cli/transition_advantage_probe.py` — realized advantage per candidate action / route, recovery latency
- `unidream/cli/wm_probe.py` — linear probe over WM latents (return / vol / drawdown / regime)
- `unidream/cli/ac_candidate_q_probe.py` — state-action critic ranking probe (`Q(s, a)` rank IC, top-decile realized advantage)

Experiment runtime (`unidream/experiments/`):

- `runtime.py` — config loading, cost profile resolution, feature/cache loading, seed
- `train_app.py` — top-level run orchestration (data, splits, fold dispatch, summary)
- `train_pipeline.py` — WFO fold loop
- `train_reporting.py` — stage / training summary, PBO / DSR aggregation
- `wfo_runtime.py` — WFO split building / selection
- `fold_runtime.py` — stage gating, checkpoint paths, `--start-from` / `--stop-after`
- `fold_inputs.py` — oracle / regime / advantage / route inputs for one fold
- `oracle_stage.py` / `oracle_teacher.py` / `oracle_post.py` — oracle pipeline (DP base + signal/feature_stress/dual teachers + post-processing)
- `transition_advantage.py` — route targets, transition advantages, recovery latency, route summarization
- `regime_runtime.py` — HMM / feature_stress regime fitting per fold
- `predictive_state.py` — WM auxiliary head bundle (`use_wm_predictive_state`)
- `wm_stage.py` — Transformer world model training / loading
- `bc_setup.py` — actor construction (route head, inventory recovery head, state machine gate, advantage adapter)
- `bc_stage.py` — BC training / loading, route + transition + state machine losses
- `ac_stage.py` — Imagination AC, supports `critic_only` and `trainable_actor_prefixes`
- `val_selector_stage.py` — validation policy selection over `val_adjust_rate_scale_grid`
- `test_stage.py` — test backtest, attribution, alerts, M2 scorecard
- `m2.py` — collapse guard, M2 scorecard, formatter

Models:

- `unidream/world_model/transformer.py` — Transformer dynamics with twohot heads, predictive auxiliaries (return / vol / drawdown / regime), symlog
- `unidream/world_model/encoder.py` / `ensemble.py` / `train_wm.py` — encoder, ensemble wrapper, trainer
- `unidream/actor_critic/actor.py` — inventory controller actor (trade / target / no-trade band, route head with `route_dim` 3 or 4, inventory recovery head, state machine gate, residual / dual residual controllers, predictive-state advantage gate / adapter)
- `unidream/actor_critic/critic.py` — value critic with reward EMA normalization
- `unidream/actor_critic/state_action_critic.py` — `CandidateQNet` ensemble for `Q(s, a)` ranking, CQL-lite penalty, anchor MSE
- `unidream/actor_critic/bc_pretrain.py` — BC trainer with route / transition advantage / inventory recovery losses
- `unidream/actor_critic/imagination_ac.py` — DreamerV3-style λ-return AC + TD3+BC regularization, `critic_only`, `trainable_actor_prefixes`

Evaluation:

- `unidream/eval/backtest.py` — vectorized backtest, cost model, PnL attribution
- `unidream/eval/pbo.py` — Probability of Backtest Overfitting (Combinatorial Symmetric Cross-Validation simplified)
- `unidream/eval/regime.py` — HMM regime detection and per-regime metrics
- `unidream/eval/wfo.py` — Walk-Forward split definitions

Data:

- `unidream/data/download.py` — Binance OHLCV / funding / open interest / mark price fetchers, parquet caching
- `unidream/data/features.py` — feature engineering, z-scoring, rebound features
- `unidream/data/oracle.py` — DP oracle, signal / feature_stress / dual teachers, smooth aim, soft labels
- `unidream/data/dataset.py` — `WFODataset`, `SequenceDataset`

## Active Configs

Production / current focus:

- `configs/bcplan5_phase8_state_machine_s007.yaml` — current BC baseline (route_dim=3 + inventory recovery + state machine gate)
- `configs/bcplan5_ac{0..3}_*.yaml` — restricted AC curriculum (critic-only → route_delta → recovery → route lite)
- `configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007.yaml` — feature_stress regime + residual-shift baseline
- `configs/bcplan5_phase{6,7}_*.yaml` — exposure loss / inventory recovery experiments
- `configs/bcplan4_phase{1..5}_*.yaml` — earlier route label / two-stage / delta experiments

Smoke / standard:

- `configs/smoke_test.yaml`
- `configs/trading.yaml`

Probes / supporting:

- `configs/wm_probe_multitask_*.yaml`
- `configs/bc_multitask_aux_*.yaml`
- `configs/bc_transition_relabel_*.yaml`
- `configs/bc_true_routing_*.yaml`

The `dualresanchor_stresstri_shiftonly_s007_*.yaml` family covers ablations: `pathcost`, `longonly_ow125`, `recovery_weakshortcopy`, `recovery_weighted_outcomeedge`, `recovery_weighted_selfcond`. Phase 9 cooldown configs (`bcplan5_phase9*`) are kept for reproducibility but are not adopted (flat-100% collapse — see `documents/20260426_bcplan5_phase9_constrained_ac_results.md`).

## Current Baseline (BC Plan 5 Phase 8)

`bcplan5_phase8_state_machine_s007.yaml` on fold 4 / seed 7 / 2018-01-01 to 2024-01-01:

```text
short 16-17%
flat  83-84%
turnover 2.60-2.62
AlphaEx +0.90 to +0.91 pt/yr
SharpeDelta -0.010 to -0.011
MaxDDDelta -1.59 to -1.61 pt
recovery gate active 0.7-0.9%
```

AC curriculum status (`bcplan5_ac{0..3}`): safe but not improving. AC-3 (route lite unlock) regresses validation. AC-4 full actor is forbidden until either:

- AC-3 route lite achieves non-negative validation alpha / sharpe, or
- the state-action critic (`ac_candidate_q_probe`) shows positive Q rank IC vs. realized advantage.

## Generated Files

- `checkpoints/<logging.checkpoint_dir>/fold_<i>/{world_model.pt, bc_actor.pt, ac.pt}` — model artifacts
- `checkpoints/data_cache/` — feature / returns parquet caches
- `documents/logs/<date>_<run>_fold<i>.log` — stdout logs
- `documents/route_probe/<date>_<run>_fold<i>.{md,json}` — route probe outputs
- `documents/wm_probe/<date>_<run>_fold<i>.{md,json}` — WM probe outputs
- `documents/ac_candidate_q/...` — candidate-Q probe outputs
- `documents/transition_advantage_probe/...` (via probe CLI) — transition advantage diagnostics

`checkpoints/` and `.venv/` are not part of the source tree and can be recreated by rerunning the pipeline.
