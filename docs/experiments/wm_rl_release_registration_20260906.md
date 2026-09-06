# UniDream WM + RL release: required scope and first diagnostic

## User-directed change

On 2026-09-06 the user made genuine UniDream World Model + learned RL mandatory for the VC-facing demo. A Ridge/HGB inventory utility policy is therefore no longer eligible as the primary release. Its research results, frozen production fit and HF `/v3/btc` route remain comparison evidence only. The new ML database run has no observations and its scheduler has not been activated. Web production remains the existing version. Do not present the ML recipe's +4.450/−5.143pt results as WM+RL performance.

The acceptance direction remains mean AlphaEX > 0 and mean MaxDDdelta < 0, with costs, trend slices and limits reported. The implementation must demonstrate an active WM→learned actor path, retained RL updates, and compare against BC and an actor-removed control. A statistical success probability is not established merely by meeting mean signs. The goal is a finite, budget-conscious compromise and a matching research/HF/Web execution path.

## Verified learning mismatch and planned repair

The retained Plan011v31 fold23 actor is a real Imagination AC actor: its state equals the AC checkpoint and differs from BC. However, actionless WM training fixes exposure to B&H and trains benchmark-relative rewards that are zero apart from sequence-entry costs; imagination then queries variable actor exposures. Its auxiliary market heads remain nonconstant, but this does not establish a learned reward response for those exposures. The relative-wealth drawdown objective also differs from strategy maximum drawdown minus B&H maximum drawdown. An untrained done head and zero ensemble disagreement with one member require explicit treatment.

The bounded opt-in repair retains the existing Transformer WM and actor-critic architecture. The WM predicts market log returns under fixed actionless context. A physical portfolio accounting helper computes actor and B&H returns on the same imagined market path using explicit transaction cost, borrowing and evolving inventory. The new objective compares compound strategy and B&H drawdowns including initial capital. This is a market approximation during imagination; actual delayed-open execution must be evaluated separately and used identically in the demo. Legacy behavior remains unchanged for the baseline comparison.

No new family is fitted before source/config/input checks. Full architecture search is outside this scope. New market-mode parameters are to be frozen before its run; reasonable candidates are short imagination and stronger BC anchoring, not an unrestricted grid. Registered P1 manifests/results remain untouched.

## First calculation: unchanged native WM→BC→AC baseline

A new, isolated configuration `configs/wm_rl_v31_v4_baseline_fold8_20260906.yaml` in the prior research worktree fixes a scratch run on official-v4 development fold8, seed7, MPS. It retains WM700 steps, BC5 epochs, AC300 steps, Transformer192/two layers, sequence64, seventeen inputs, legacy reward and native validation selector. Only the explicit v4 cache, fold choice and new non-overwriting output directory differ from the legacy configuration. This first run measures current pipeline behavior and resource use; it is not a selection of a new winning model.

The cache is `/Users/sophie/Documents/UniDream/UniDream/checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_*`. Funding and mark masks remain required. Its first timestamp differs from the older compressed cache, so its fold dates are recorded as a new calendar rather than claimed identical to the old v31 report. Native test results are report-only. These old-contract metrics do not substitute for the later cash/units/delayed-open comparison. No deleted historical checkpoint is reconstructed by applying fold23 weights to earlier periods.

Runtime: local 16GB Apple Silicon, PyTorch2.11.0 with MPS available. Remote sv04 direct SSH authentication failed, so no job or credential change was attempted there. Independent input/source audits and literature checks run alongside the local calculation; concurrent GPU jobs will be bounded to avoid exhausting local memory.

Baseline input preflight confirms fold8 T2020-01-19T17:00→2022-01-19T17:00, V→2022-04-19T17:00, E→2022-07-19T17:00. Train/validation/test contiguous seq64 counts are50,738/8,577/8,673, with finite body cells and returns. The unchanged baseline retains four train sequence edges whose prior mark history is not completely covered by the simple sidecar, and the cache metadata does not bind the TA producer version. These are disclosed imperfections of the baseline, not cleared deployment data-quality gates. The new candidate must enforce its expanded causal dependencies.
