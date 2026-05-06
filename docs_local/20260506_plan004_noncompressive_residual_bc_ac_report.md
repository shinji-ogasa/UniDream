# 2026-05-06 Plan004 Non-Compressing Residual BC/AC Report

## Conclusion

- Do not compress the hierarchy policy into one student actor.
- Keep the hierarchy base policy fixed, then learn a small residual policy from realized candidate advantage.
- The selected residual BC / AWR-style extraction reached the target on 13 of 14 folds.
- All 14 folds had AlphaEx > 0. The worst MaxDDDelta was +0.2109 pt.
- The existing scratch neural BC/AC path in train.py was tested on fold8 with CUDA and was not adopted: AlphaEx -0.03 pt, MaxDDDelta +0.04 pt.
- Fold8 remains the only fold below AlphaEx +1.0. Adding WM predictive features to this residual probe did not help.

## Adopted Candidate Implementation

- Code: `unidream/cli/plan004_noncompressive_bc_ac_probe.py`
- Base policy: `_compute_teacher()` recomputes the hierarchy teacher per fold.
- BC target: realized utility for each candidate residual delta.
- Extraction: validation-selected threshold / hold / cooldown.
- Safety: source-specific spec gating, turnover cap, cost stress selection, and MaxDD cap.
- New useful spec: `bc_resid_guarded_twoside_h16`.

## 14-Fold Aggregate: cost_x1

| policy | folds | alpha>0 | alpha>1/dd<=1 | eps pass | Alpha mean | Alpha median | Alpha worst | MaxDD worst | turnover mean | turnover max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_hierarchy | 14 | 12 | 5 | 12 | 4.8335 | 0.4215 | 0.0000 | 0.0000 | 0.7357 | 2.5000 |
| selected_residual_bc_ac | 14 | 14 | 13 | 13 | 16.9584 | 2.6971 | 0.4120 | 0.2109 | 2.1893 | 3.5000 |

## Stress Aggregate: selected_residual_bc_ac

| stress | alpha>0 | alpha>1/dd<=1 | Alpha mean | Alpha worst | MaxDD worst | turnover max |
|---|---:|---:|---:|---:|---:|---:|
| cost_x1 | 14 | 13 | 16.9584 | 0.4120 | 0.2109 | 3.5000 |
| cost_x1_5 | 14 | 13 | 13.3466 | 0.3920 | 0.2188 | 3.5000 |
| cost_x2 | 14 | 13 | 9.7555 | 0.3720 | 0.2268 | 3.5000 |
| slippage_x2 | 14 | 13 | 15.6426 | 0.4047 | 0.2138 | 3.5000 |

## Fold Detail: selected cost_x1

| fold | source | spec | AlphaEx | SharpeDelta | MaxDDDelta | turnover | flat | target |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 0 | GR_baseline | bc_resid_twoside_h16 | 7.439681 | 0.048124 | -0.117196 | 1.500 | 0.994 | PASS |
| 1 | D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly | bc_resid_riskoff_h32 | 2.404245 | 0.198394 | -2.760408 | 2.500 | 0.291 | PASS |
| 2 | micro_triple_fixed_raw | bc_resid_guarded_twoside_h16 | 137.266730 | 0.033867 | 0.000000 | 3.500 | 0.980 | PASS |
| 3 | D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly | bc_resid_riskoff_h32 | 40.176301 | 0.072079 | 0.000000 | 2.100 | 0.993 | PASS |
| 4 | D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly | bc_resid_micro_h32 | 2.013529 | -0.143238 | -3.968637 | 2.350 | 0.323 | PASS |
| 5 | recovery_rescue_fixed_state | bc_resid_twoside_h32 | 23.327155 | 0.028904 | -0.003255 | 1.600 | 0.991 | PASS |
| 6 | D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly | bc_resid_guarded_twoside_h16 | 1.843127 | 0.058961 | -1.869495 | 3.500 | 0.787 | PASS |
| 7 | benchmark | bc_resid_riskoff_h32 | 2.989984 | 0.056449 | -0.203198 | 2.400 | 0.965 | PASS |
| 8 | benchmark | bc_resid_riskoff_h32 | 0.412021 | 0.061573 | -0.598772 | 2.400 | 0.985 | MISS |
| 9 | micro_triple_fixed_raw | bc_resid_guarded_twoside_h16 | 1.601839 | 0.040078 | 0.210850 | 1.300 | 0.997 | PASS |
| 10 | micro_triple_fixed_raw | bc_resid_riskoff_h32 | 5.960813 | 0.091270 | -0.889678 | 2.400 | 0.992 | PASS |
| 11 | D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly | bc_resid_micro_h16 | 8.954289 | 0.047336 | 0.000000 | 1.450 | 0.991 | PASS |
| 12 | GR_baseline | bc_resid_twoside_h16 | 1.535798 | 0.037222 | -0.294291 | 1.200 | 0.995 | PASS |
| 13 | D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly | bc_resid_riskoff_h32 | 1.492071 | -0.074410 | -1.831515 | 2.450 | 0.158 | PASS |

## Additional Checks

- Weak-fold rerun: `codex_outputs/20260506_plan004_fix2_weakfolds.json` fixed the fold2/fold7/fold12 selection failures.
- Guarded h16 check: `codex_outputs/20260506_plan004_guarded_check_folds2691013.json` showed fold6 and fold9 could exceed AlphaEx +1.0.
- Final all-fold run: `codex_outputs/20260506_plan004_fix4_allfold.json`.
- CUDA scratch neural BC/AC on fold8: `docs_local/logs/20260506_train_scratch_fold8_cuda.log`; not adopted.
- Fold8 teacher audit: `codex_outputs/20260506_fold8_teacher_audit.json`; D/GR/recovery/micro teachers were mostly benchmark or negative.
- Fold8 WM predictive feature check: `codex_outputs/20260506_fold8_plan004_wm_features.json`; not adopted.

## Paper-Based Rationale

- IQL extracts policies with advantage-weighted BC and avoids direct evaluation of unseen actions. This supports the residual realized-advantage extraction design. Source: https://arxiv.org/abs/2110.06169
- TD3+BC adds a BC regularizer to offline actor updates. In this project, moving the neural actor broadly was less stable than keeping the hierarchy base fixed and extracting small residual actions. Source: https://arxiv.org/abs/2106.06860
- CQL motivates conservative handling of OOD actions. Source-specific gating, cost stress, turnover caps, and MaxDD caps are used here as conservative offline-RL safety controls. Source: https://arxiv.org/abs/2006.04779

## Adopt / Reject

- Adopt candidate: non-compressive hierarchy base plus realized residual advantage BC plus validation threshold extraction.
- Adopt candidate: source-specific residual spec gating.
- Adopt candidate: guarded two-sided h16 residual.
- Reject: one-actor compression BC of the hierarchy policy.
- Reject for now: existing train.py neural AC path on fold8.
- Reject for now: WM predictive aux added to the fold8 residual probe.

## Commands

```powershell
.venv\Scripts\python.exe -u -m unidream.cli.plan004_noncompressive_bc_ac_probe --folds 0,1,2,3,4,5,6,7,8,9,10,11,12,13 --seed 7 --output-json codex_outputs\20260506_plan004_fix4_allfold.json --output-md codex_outputs\20260506_plan004_fix4_allfold.md
uv run python -u -m unidream.cli.train --config configs\trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 8 --seed 7 --device cuda
```
