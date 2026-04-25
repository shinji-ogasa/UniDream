# 2026-04-25 TransformerWM Probe and Improvement Report

## Conclusion

- The raw feature set is not completely useless. It can separate volatility and drawdown risk. The bigger problem is that the baseline TransformerWorldModel latent state discards much of that predictive information.
- Baseline fold 4 test raw probes reached `vol_h16 AUC=0.6535`, `DD_h16 AUC=0.6316`, and `return_h32 IC=0.0867`. Baseline `latent_zh` dropped to `vol_h16 AUC=0.5849`, `DD_h16 AUC=0.5758`, and `return_h32 IC=0.0092`.
- The best improvement is `configs/wm_probe_multitask_aux_s007.yaml`: multi-horizon future return, future volatility, and future drawdown auxiliary losses. It improved `latent_zh` from `return_h32 IC 0.0092 -> 0.0393`, `vol_h16 AUC 0.5849 -> 0.6083`, `teacher AUC 0.4858 -> 0.5300`, and `recovery AUC 0.5035 -> 0.5329`.
- The `regime_aux` variant improved teacher AUC to `0.5376`, but it degraded return, recovery, and action-advantage metrics. Do not adopt it as the main WM config yet.
- The practical ceiling with the current features is close to the `raw probe` row. Short-horizon return direction remains weak, risk/volatility defense is usable, and action advantage / recovery are still weak.

## Implemented Changes

- Added `unidream/cli/wm_probe.py`.
- The probe evaluates `raw`, `latent_zh`, and `raw_plus_latent` against identical downstream targets, separating raw feature quality from WM latent information retention.
- Probe targets: `return_h{1,4,8,16,32}`, `vol_h{...}`, `drawdown_risk_h{...}`, `teacher_underweight`, `recovery_h16`, and `action_advantage_h16`.
- Updated `unidream/world_model/train_wm.py`.
- Added multi-output `ReturnHead`, future-only multi-horizon return targets, future volatility targets, future drawdown targets, masked SmoothL1 losses, and target scaling.
- Updated checkpoint loading so optimizer-state shape mismatches are skipped for evaluation when auxiliary heads differ.

## Sources Checked

- DreamerV3 learns a world model and improves behavior by imagining future scenarios. This supports forcing UniDream latent state to preserve reward/risk-relevant future information. Source: https://arxiv.org/abs/2301.04104
- DreamSmooth reports that reward prediction can be a bottleneck in model-based RL and uses temporally smoothed reward prediction. This supports multi-horizon return/risk auxiliary prediction. Source: https://arxiv.org/abs/2311.01450
- PatchTST uses patching and channel independence to retain local time-series semantics and attend to longer history. This remains a next architecture candidate, not implemented in this pass. Source: https://arxiv.org/abs/2211.14730
- TS2Vec evaluates timestamp-level representations with simple downstream forecasting probes, matching the `latent -> target` linear probe methodology used here. Source: https://arxiv.org/abs/2106.10466
- Gu, Kelly, and Xiu emphasize low signal-to-noise in expected return prediction and the need for out-of-sample/economic metrics. This is why the report focuses on rank IC, event AUC, decile spread, and action advantage rather than MSE alone. Source: https://academic.oup.com/rfs/article/33/5/2223/5758276

## Baseline: raw vs latent vs raw+latent

| feature set | return h16 IC | return h32 IC | vol h16 AUC | DD h16 AUC | teacher AUC | recovery AUC | action advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | +0.0430 | +0.0867 | +0.6535 | +0.6316 | +0.5444 | +0.5428 | +0.000001 |
| latent_zh | +0.0144 | +0.0092 | +0.5849 | +0.5758 | +0.4858 | +0.5035 | -0.000203 |
| raw_plus_latent | +0.0327 | +0.0490 | +0.6364 | +0.6000 | +0.4910 | +0.5180 | -0.000090 |

Interpretation: raw features already predict risk. Baseline latent is losing useful predictive structure. `raw_plus_latent` does not consistently beat raw, so the baseline latent adds little reliable value.

## Improvement Comparison

| feature set / run | return h16 IC | return h32 IC | vol h16 AUC | DD h16 AUC | teacher AUC | recovery AUC | action advantage |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline latent | +0.0144 | +0.0092 | +0.5849 | +0.5758 | +0.4858 | +0.5035 | -0.000203 |
| multitask latent | +0.0355 | +0.0393 | +0.6083 | +0.5703 | +0.5300 | +0.5329 | -0.000070 |
| multitask+regime latent | +0.0129 | +0.000534 | +0.6059 | +0.5669 | +0.5376 | +0.5085 | -0.000209 |
| baseline raw+latent | +0.0327 | +0.0490 | +0.6364 | +0.6000 | +0.4910 | +0.5180 | -0.000090 |
| multitask raw+latent | +0.0275 | +0.0520 | +0.6269 | +0.5904 | +0.5317 | +0.5371 | -0.000126 |
| multitask+regime raw+latent | +0.0188 | +0.0086 | +0.6260 | +0.5858 | +0.5308 | +0.5176 | -0.000210 |

## Best Detailed Delta: multitask auxiliary vs baseline latent

| metric | baseline latent | multitask latent | delta | raw ceiling |
|---|---:|---:|---:|---:|
| return h16 IC | +0.0144 | +0.0355 | +0.0211 | +0.0430 |
| return h32 IC | +0.0092 | +0.0393 | +0.0302 | +0.0867 |
| vol h16 AUC | +0.5849 | +0.6083 | +0.0234 | +0.6535 |
| DD h16 AUC | +0.5758 | +0.5703 | -0.0054 | +0.6316 |
| teacher underweight AUC | +0.4858 | +0.5300 | +0.0442 | +0.5444 |
| recovery h16 AUC | +0.5035 | +0.5329 | +0.0294 | +0.5428 |
| action advantage | -0.000203 | -0.000070 | +0.000133 | +0.000001 |

## What Improved

- Return: h16/h32 latent rank IC improved. It is still below the raw ceiling, and h1 did not improve, so short-horizon direction is still low-SNR.
- Volatility: h4/h8/h16/h32 latent AUC improved. h4 moved from `0.5858 -> 0.6365`, still below raw `0.6850`.
- Drawdown: h4/h8 improved, while h16/h32 stayed weak or degraded. Long-horizon DD needs a better target or architecture.
- Teacher/recovery: baseline latent was near random. Multitask improved to `teacher AUC=0.5300` and `recovery AUC=0.5329`, close to the raw ceiling around `0.54`.
- Action advantage: the negative edge shrank, but it is not yet a positive or tradable edge.

## Adopt / Hold

- Adopt: `configs/wm_probe_multitask_aux_s007.yaml`.
- Hold: `configs/wm_probe_multitask_regime_aux_s007.yaml`.
- Reason: regime auxiliary improves teacher AUC but hurts return, recovery, and action advantage. For actor input, the pure multitask auxiliary latent is the better compromise.

## Improvement Limit

- Current-feature ceiling from fold 4 raw probe: `return_h32 IC ~0.087`, `vol_h16 AUC ~0.654`, `DD_h16 AUC ~0.632`, `teacher/recovery AUC ~0.54`.
- Multitask latent recovered part of the lost signal, especially risk/vol and teacher/recovery. It did not close the gap to raw for return and long-horizon drawdown.
- Because raw teacher/recovery AUC is only around `0.54`, BC/RL-only changes are unlikely to solve recovery by themselves. Recovery-specific features or teacher redesign are needed.
- Next candidates: `PatchTST-style patch encoder`, native predictive-head outputs appended to actor state, triple-barrier/stress event label heads, and market microstructure / volume / liquidity feature additions. Run fold 0 and fold 5 before larger architecture work.

## Artifacts

- Baseline JSON: `documents/wm_probe/20260425_wm_probe_baseline_fold4.json`.
- Multitask JSON: `documents/wm_probe/20260425_wm_probe_multitask_aux_fold4.json`.
- Multitask+Regime JSON: `documents/wm_probe/20260425_wm_probe_multitask_regime_aux_fold4.json`.
- Multitask train log: `documents/logs/20260425_wm_probe_multitask_aux_fold4_train.log`.
- Multitask+Regime train log: `documents/logs/20260425_wm_probe_multitask_regime_aux_fold4_train.log`.
