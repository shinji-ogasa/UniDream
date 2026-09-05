# Goal continuation: Oracle / features / trend robustness

Overall goal remains active. No candidate has established high-probability joint
AlphaEx>0 and MaxDDDelta<0 across trends. Do not mark this goal achieved because
the exploratory means or an implementation test passed.

Working branch: `exp/oracle-feature-frontier-20260905` in
`/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905`.
Initial experiment source commit: `ed23b49`; risk ablation source: `0272879`.
Runtime: `/Users/sophie/Documents/UniDream/.worktrees/alpha-dd-goal/.venv/bin/python`.

## Completed; do not rerun or retune these locks

- Audited current main and prior alpha/dd work. Older memory of P1 blocked
  execution is stale: formal recoverability tests later ran, but those are not
  natural BTC investment evidence. No old P1 record was modified.
- Added 29-feature technical and 24-feature flow groups, with strict shift1 and
  gap handling. Fixed36 ML family: 234 out-of-time fits on13 actual validation
  quarters, 689 economic rows including diagnostic/control arms.
- RL finite-beam feasible hindsight reference and ML perfect-outcome mapping
  use the same cash/units execution contract as real policies. These are
  future-dependent diagnostics, not global upper bounds or training teachers.
- Six of36 learned policies passed joint mean signs in base and2x costs;
  zero passed all start-of-quarter trend groups. Technical weekly downside
  Ridge improved mean signs over its base16 counterpart but failed bull DD
  and sideways alpha. Fixed minimax selection chose an aggregate-failing flow
  HGB; do not silently switch the selection criterion.
- Technical HGB6h risk forecast improved volatility MSE skill in all three
  regime means (10/13 quarters). Return MSE did not improve. This is the
  actionable positive forecast finding, not an overall accuracy claim.
- Twelve predeclared volatility-ratio controllers were then tested. None
  passed all regimes; adding them worsened average results against their
  own backbones. Reject this direct allocation-scaling recipe.
- Final410 unit tests passed. Audit details and immutable snapshots accompany
  the results report. No production model, HF Space, or paper account changed.

## Next useful investigations, before larger model architectures

1. Measure calibration of the promising6h risk forecast on strictly disjoint
   fit/calibration/validation chronology; compare with a causal persistence
   forecast, not just the training climatology. The current MSE skill reference
   is too weak to establish superiority over a strong risk baseline. Predeclare
   calibration method and all candidates before outcomes, and preserve all
   failed comparisons. Do not treat overlapping6h/7d labels as independent.
2. Measure decision value with a calibrated conditional distribution or joint
   scenarios. The failed ratio controller shows that forecast improvement alone
   is insufficient. Keep transaction costs, endogenous inventory, turnover,
   downside and B&H-relative utility aligned. No hindsight teacher forcing.
3. If return forecasts still fail strong causal baselines, examine new
   information: contemporaneously available Spot/perpetual flow divergence and
   funding/basis, with data availability provenance and matched support. More
   transformations of the same OHLCV cannot create independent information.
4. Freeze any resulting viable policy before prospective data become available.
   Prior historical/fresh quarters, omitted13/14 folds and current development
   periods are not untouched confirmation. A future forward paper protocol
   must define its cutoff, update rule, costs and per-regime sample requirements
   before scoring. A favorable reused-validation result is only a candidate.

No scheduled automation has been created; continuation is through this active
Codex goal. No notification claim should imply an external monitor is running.
