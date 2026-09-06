# Retained genuine WM + RL candidate audit (read-only, 2026-09-06)

The viable retained runtime foundation is **Plan011 v31 fold23**, a real Transformer WM → BC-initialized RL Actor-Critic policy. It is runnable now as a stored inference bundle, but neither the old development nor holdout evidence meets the required negative MaxDDdelta. No retained WM+RL model was verified to meet both requested economic signs. No new fit, forecast, economic evaluation, gate, deployment or repository mutation was performed in this audit.

## Actual retained candidates

| Candidate | Retained files | Existing evidence / limitation |
|---|---|---|
| Plan011 v31 fold23 | `/Users/sophie/Documents/UniDream/unidream-space/bundles/current/{actor_full.pt,checkpoints/world_model.pt,checkpoints/ac.pt,checkpoints/bc_actor.pt,predictive_state.npz,model_config.yaml,manifest.json,sample_input.npz}`; identical family retained in accuracy-release-space worktree | Full WM+RL runtime. Historical development0–12 +0.41pt Alpha / +0.20pt DD; report-only holdout15–23 +0.11 / +0.20pt. DD fails. Fold23 trained2023-10-16→2025-10-16; validation→2026-01-16; report-only test→2026-04-16. Cannot run its weights over2021–23 and call it OOS. |
| P1 conditional WM8 / BC1 / AC2 | research worktree `codex_outputs/p1_conditional_wm_bc_ac_20260904/checkpoints/{world_model.pt,bc_actor.pt,ac.pt}`, also present in p1-production-chain and other worktrees | Force-tracked actual checkpoints. Registered diagnostic has 100% underweight validation output, five rolling alpha mean−0.041540 additive-log-return. Contract is4-bar, positions.5–1, all-or-none; not current cash/unit six-hour comparison. |
| P1 formal WM700 / BC8 / AC300 | `.worktrees/p1-formal-forecast-wm-bc-ac/codex_outputs/p1_formal_forecast_wm_bc_ac_20260904/checkpoints/{world_model.pt,bc_actor.pt,ac.pt}` plus smoke variants | Actual stronger training retained, but reported AC is exactly flat benchmark:0fills/0overlay, no economic uplift. BC rolling alpha−0.03757496; not a ready improvement candidate. |
| Risk-aware PPO variants | Report only: `UniDream/docs/risk_aware_rl_unidream_holdout_2026.md` | Previously audited checkpoints/code/config removed. Holdout full−2.12/−4.37pt and risk-only−1.74/−4.86pt; Alpha fails. This also is not the UniDream WM+RL pipeline. |

Duplicate paths are copies, not independent trained candidates. The file search covered named checkpoint files under all workspace worktrees, research checkpoints and HF bundles, excluding virtual environments/node_modules. Absence here is not a claim about remote hosts or unmounted stores.

## v31 actual active path

`unidream-space/backend/predictors/plan011.py:58` loads the full learned actor; `:64` builds and loads the trained WM. At`:187`, `WM.encode_sequence(features, actions=None, seq_len=64)` yields z/h. At`:119`, WM predictive auxiliary heads are concatenated, then standardized/clipped with saved predictive state. At`:201`, the learned actor consumes z/h + regime + auxiliary state and outputs target positions. This is not the BTC Ridge/HGB mean pipeline and not the deterministic BNB trend rule.

Direct metadata inspection of the saved actor: `unidream.actor_critic.actor.Actor`, inventory_dim4, advantage_dim42, regime_dim3, infer_adjust_rate_scale0.7, benchmark1, position bounds0.5–1.12, maxstep0.08. Benchmark overweight/floor/trainable-sizing adapters are all disabled. No new predictions were computed during this metadata inspection. Earlier same-turn legacy smoke already passed8641 stored fixture rows with max position difference1.192e−7 and predictive-state difference0.

Feature contract:17 ordered normalized features, sequence64, rolling normalization60days. Names are open_ret, high_ret, low_ret, close_ret, vol_ret, RSI_14, macd, macd_signal, atr_norm_ret, atr, rv_4, rv_16, rv_96, funding_rate, basis, basis_mom, basis_abs. Funding and mark-derived basis are necessary; new Spot/UM quote/taker31-column features cannot be substituted. Missing external sources must not silently become zero. The old retained training cache does not distinguish true zero from old imputation (attribution report feature-quality flag).

## Critical chronology / reproducibility traps

1. Old dev training checkpoints0–12 were deleted (`docs/plan011_v31_investor_evidence.md:11–15`). The retained `docs/figures/plan011_v31_folds0_12/timeseries.npz` contains per-fold timestamp, return, position and strategy/B&H equity arrays only. There are no per-fold saved latent/WM head matrices or validation actor paths. `docs/alpha_attribution_plan011_v31_dev/report.md:14` explicitly notes absent validation paths.
2. Old test folds4–11 cover the same calendar quarters as recent ML val5–12 (2021-04-16→2023-04-16), but the stored metric contracts differ. Old plot source `unidream/cli/plot_plan011_fold_trades.py:80` uses exp cumulative position-weighted log PnL. `unidream/eval/backtest.py:244` computes positions*same-index logreturns minus transition costs. The new common contract is cash/units, next-open fill, borrow, step/deadband, initial B&H inventory. Recompute both candidates and controls under one newly pinned replay contract; do not compare published means as already identical conditions.
3. Feature timestamps are already one-step shifted in old vendor `data/features.py:7–9`; returns are current-bar close/previous-close. Define exactly when a stored actor position becomes an intent and then add the new next-open fill. Never backward-shift the new fill to reproduce old economics. Preserve missing-grid rows and use original exact common support if comparing recent ML evidence.
4. `Actor.predict_positions` resets controller state and rate counters on every call (`actor.py:1496–1500`) and updates controller state from predicted positions (`:1549`). The current convenience `predict_latest` rebuilds an entire rolling path; it does not accept actual cash/unit inventory. A causal risk gate changes actual inventory. An incremental live adapter must define/persist either actual controller state or an explicit shadow-policy state and verify parity; silently reinitializing a rolling window changes the policy. WM chunk context has64-bar warmup and end-aligned last-chunk logic (`train_wm.py:1167`), so prefix/streaming parity also needs testing.
5. Existing attribution already weakens timing-uplift claims: raw actor mean Alpha+0.414pt versus constant actor-mean+0.391pt; lag1/4/16 remain+0.413/+0.407/+0.401pt, DD stays positive (attribution report:20–34). The constant actor-mean uses the whole historical test path and is descriptive, not a deployable selector.

## Minimum bounded next work

Use the retained v31 recipe as the genuine neural foundation. Before spending another training run, a new preregistered **saved-position replay diagnostic** can use old dev4–11 actor positions on the exact target calendar, with original grid/support and canonical cash/unit accounting. It proves only an execution/gate change on saved causal actor outputs, not current-code model reconstruction. Retain pure actor, B&H, and one fixed causal risk-budget gate plus its actor=1 counterfactual. Costs1×/2× must use the same intents; no holdout15+ selection.

If a gate is examined, one coherent form is `target_t=clip(g_t*actor_t, .5, 1.12)` where g depends only on completed-bar risk and uses a predeclared existing risk budget. Do not choose its budget after seeing new scores or substitute a winning price rule and call the actor improved. Compare with exactly the same `clip(g_t*1, .5, 1.12)` control, plus counts/differences in executed orders and paired alpha/DD. A derivative dependence before deadband is insufficient: if both versions execute the same path, the actor is economically decorative. Positive Alpha / negative DD of the full path alone does not establish WM/RL contribution. Risk shrinkage can sacrifice bull returns, as older volatility-target results already show; improvement is unmeasured.

A real revised WM+RL fit can follow only with a separate bounded config/registration and retained checkpoints, because old dev weights cannot be recovered by loading the fold23 bundle. Root is inspecting the fit runner separately. The current contract requires train→BC→AC from scratch; do not quietly warm-start around current CLI restrictions. If the fit/selector changes, all evidence must be labeled new rather than inheriting v31 scores.

## Exact existing commands (not executed in this audit)

Existing bundle smoke, no fit or new outcome scores:

```bash
cd /Users/sophie/Documents/UniDream/unidream-space
/Users/sophie/Documents/UniDream/.worktrees/alpha-dd-goal/.venv/bin/python -m backend.verify_bundle --bundle-dir bundles/current --device cpu
```

Existing full historical v31 reproduction, ONLY if separately authorized as a new expensive run:

```bash
cd /Users/sophie/Documents/UniDream/UniDream
uv run python -m unidream.cli.train --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml --seed 7 --device cuda
```

After a completed fresh dev run only (currently unavailable), saved checkpoint plots:

```bash
uv run python -m unidream.cli.plot_plan011_fold_trades --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml --checkpoint-dir checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007 --folds 0-12 --seed 7 --device cpu --output-dir docs/figures/plan011_v31_folds0_12
```

No existing CLI was verified to perform the proposed canonical six-hour cash/unit saved-position gate replay. That needs a new bounded registered wrapper; inventing an existing command would be misleading.

## Key v31 byte hashes

- actor_full.pt: `35acf3b3c1242b565a9fea0212de53c98371f124464dec63ea34122b14c6d54c`
- checkpoints/ac.pt: `6d53bd94a8c9c19f5c907a8f8f97cb008afed1e5cd44c246001f502e1ab06175`
- checkpoints/world_model.pt: `d97967225515b7d7edfeed335ce2ff5b10df0b236b6d71e1972697b8a8095442`
- checkpoints/bc_actor.pt: `1c856d9baf880ca374afe63257a54c95d98eb0d6a8ad0696718734a4047284b0`
- predictive_state.npz: `48e8e1c31b56764d16d4d2ad0fe81034c1a948f1cfd7a2cef94dbcefabca9250`
- model_config.yaml: `05879c65b377a4a13476863097f69eb4ee69a64cf9fcc53831381ea025a448aa`
- manifest.json: `a89e77009bde351f8d393e459d301d52683d96113ebae619ee28f6af8ad0943e`

Earlier cross-component audit was stopped on user scope change. Its partial source-binding snapshot is `/tmp/btc_cross_component_partial_audit_20260906.json`; it is not release approval. Memory registry was used only to locate historical evidence, then all statements above were checked against current files: MEMORY.md:211–216,237–267.
