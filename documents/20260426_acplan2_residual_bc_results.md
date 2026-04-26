# AC Plan 2 / Realized Advantage Residual BC Result

Date: 2026-04-26

## Decision

`realized advantage residual BC` は本流採用しない。

理由:

- Phase 8 safe baseline を明確に超えなかった。
- full actor fine-tune は flat 100% に寄って AlphaEx が悪化した。
- route-delta-only は崩壊しないが、Phase 8 とほぼ同等で改善幅が足りない。
- acplan_2 の成功条件 `AlphaEx >= Phase8 +0.2pt`, `SharpeDelta >= 0` を満たさない。

不採用のため、実験実装と実験configは削除した。Phase 8 mainline は維持。

## Baseline

Phase 8 safe baseline, fold 4:

```text
AlphaEx +0.89 to +0.91 pt/yr
SharpeDelta -0.010 to -0.011
MaxDDDelta -1.58 to -1.61 pt
short 16-17%
flat 83-84%
turnover 2.60-2.62
recovery gate active 0.7-0.9%
```

## Experiments

| Run | Change | Residual active | Test AlphaEx | SharpeDelta | MaxDDDelta | Dist | Turnover | Decision |
|---|---|---:|---:|---:|---:|---|---:|---|
| R1-safe | full actor, coef 0.05, max_delta 0.10 | 7.2% | +0.55 | -0.002 | -0.98 | long 0%, short 0%, flat 100% | 0.81 | reject |
| R1-delta | route_delta_head only, max_delta 0.05 | 0.2% | +0.91 | -0.009 | -1.61 | long 0%, short 16%, flat 84% | 2.62 | reject |
| R1-delta-open | route_delta_head only, no max_delta | 30.7% | +0.91 | -0.009 | -1.60 | long 0%, short 16%, flat 84% | 2.62 | reject |

## Interpretation

The candidate advantage signal exists in-sample, but it does not translate into a better test policy.

Observed failure modes:

- Full actor residual BC over-regularizes back toward neutral/flat and loses Phase 8's mild underweight edge.
- Strict support filtering leaves too few active samples, so the policy is effectively unchanged.
- Open delta-only filtering increases active labels, but `route_delta_head` cannot convert those labels into meaningful realized policy improvement.

This matches the offline RL literature direction: AWR/AWAC/IQL-style updates can be stable because they weight high-advantage actions, but they still need the policy update to remain close to the behavior/support distribution. AWR describes supervised value regression plus weighted policy regression; AWAC uses advantage-weighted maximum likelihood updates from prior data; IQL extracts policies with advantage-weighted BC while avoiding direct evaluation of unseen actions. In this UniDream run, the synthetic candidate labels around Phase 8 were not reliable enough to justify adoption.

Sources checked:

- AWR: https://arxiv.org/abs/1910.00177
- AWAC: https://arxiv.org/abs/2006.09359
- IQL: https://arxiv.org/abs/2110.06169

## Implementation Outcome

Kept:

- Phase 8 mainline in `configs/trading.yaml`.
- AC pipeline and diagnostic probes.

Removed after rejection:

- R1 residual BC trainer extension.
- R1 residual candidate/target builder.
- R1 experimental configs.

## Next Recommendation

Do not move to AC actor unlock from R1. The safe AC transition condition is not met.

The next useful path is not more residual BC over synthetic candidates. Better next step:

1. Keep Phase 8 as anchor.
2. Use candidate-Q only as diagnostics/filter, not as action selection.
3. If AC is resumed, restrict it to critic-only or route-delta-only probes and require fold0/fold5 safety before any route unlock.
