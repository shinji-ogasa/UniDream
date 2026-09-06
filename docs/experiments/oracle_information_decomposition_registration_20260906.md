# Matched-support hindsight information decomposition

Freeze this protocol, source, tests and data-only preflight before new policy
rollouts. The previous additional test15–24 results remain report-only; no
weight, feature, regime threshold or causal model is selected or retuned here.

## Question and fixed scope

On **original development validation folds5–12 only**, determine how the
currently fixed controllers change when supplied with realized return or risk
information. Compare with a finite hindsight search using the same available
decision times, own-inventory action candidates and missing-input rules.
These are hindsight diagnostics, not new causal candidates or training teachers.
The 158 previously explored causal policy names and all locks remain unchanged.

Use the completed `oracle_frozen_procedure_parity_v1` artifacts, their exact
source/data/config bindings and original `2023-04-16T13:45:00Z` cutoff. The
eight periods correspond to original test4–11, not test15–24. No model fitting,
new market acquisition, additional test access or architecture change occurs.

The complete universe is **28 policies × 8 quarters = 224 economic rows**:

- All12 frozen causal controls: B&H, common robust, and scale/full/half means
  with both hold and target-one fallback rules, exactly as previously saved.
- Twelve hybrid information diagnostics: technical half and perpetual half,
  each with return-only, realized-risk-only and both substitutions, each under
  hold and target-one fallback.
- Four matched finite-beam searches: each missing-input rule with fixed
  MaxDD penalty0 and1. Beam width32; no beam-width or penalty search.

## Information substitution

Preserve I=the saved inference mask and O=the saved score-support mask, O⊂I.
On O only, substitute `actual[:,0]` for the mean and/or `actual[:,2]**2` for
risk. No annualization, factor24 or additional floor is applied to that squared
realized volatility. Zero realized risk is allowed by the existing utility
planner; no zero-risk scored rows exist in this registered data preflight.
Do not reapply the half weight after substitution.

Every I\O row retains its own frozen learned mean/risk, and every ~I row
retains NaN forecast availability. The original missing-input rule still uses
I, never O. The full quarterly price path remains unchanged.

The source audit finds **2,586 inference / 2,574 replacement / 12 learned
remainder rows per mean stream**. The 12 remainders comprise eight quarter-end
maturities and four future-gap rows. Retain all332 fallback opportunities and
two missing-current-open scheduled rows. These counts are not multiplied by
the number of controllers and presented as new independent information.

The h24 realized return is `log(close[t+24]/open[t+1])`; it matures at t+25
bars. It is not the true conditional expected return. Likewise realized
volatility squared is an observed quadratic-variation target, not the true
conditional variance. These interventions deliberately reveal future values.
Override inherited causal planner trace labels with explicit hindsight flags,
and expose which decisions received substituted information.

Each controller must create its own evolving cash/units state. Even both-swap
technical/perpetual variants need not have identical paths: the 12 remainder
rows retain different predictions and earlier inventory can differ. Do not
deduplicate these rows or select a favorable missing-input rule.

## Matched action search

At a scheduled, supported decision with observed current open, construct
no-trade plus exposure intents clipped from current exposure±0.04/±0.08 into
[0.5,1.12]. Compute current exposure from its own inventory **before the
decision bar's close mark and borrowing charge**, matching the conditional
planner. Respect max_step0.08, deadband0.01 and the same strict eligible action
set. Unsupported known-open decisions either hold or force target1 according
to the fixed missing rule. A missing current open cannot submit an order.

Fill only at the next bar's observed open; a missing next open skips the fill
without rollover. Keep all price bars and continuous borrowing across gaps.
Initial cash0 and B&H units, one-way fee0.00055, annual borrowing0.10, and all
other accounting conventions remain unchanged.

Rank each beam by `log(marked NAV) − penalty * running MaxDD`, using the same
terminal objective. Retain a separately identified feasible incumbent: hold
on supported opportunities and obey the fixed missing-input rule elsewhere.
For fallback, an unrestricted all-NaN hold path is not a valid incumbent.
Deduplicate only identical accounting states, with stable action tie ordering.

Every resulting path is replayed through canonical accounting. Doubled-cost
stress uses exactly the base intents, without a second search. The finite beam
can prune the best future path, so its objective is a **lower bound on the
optimal hindsight objective over the registered path set**, not a global
optimum or an upper bound on attainable causal model performance.

The full-path objective and future horizon differ from the local h24 utility.
Therefore the gap between hybrid substitution and beam results cannot be
attributed solely to decision quality. The four beams are separate diagnostic
comparators, not winners or evidence that the learned policy can attain them.

## Reporting and validation

Save all224 rows and both cost accounts, six substituted forecast arrays per
fold, all target vectors, all16 hindsight traces per fold and SHA-bound fold
manifests. Show equal-quarter and start-regime AlphaEx, MaxDDDelta, turnover
and trades, plus each hybrid minus its own frozen controller. Preserve failed
or adverse intervention effects; improvement is not guaranteed by revealing a
noisy realized target. Do not report the mechanically substituted zero forecast
error as learned accuracy, and do not express an Oracle gap as a probability.

All original controls must reproduce saved targets/accounting exactly. Tests
must establish unchanged masks/remainder values, future-tail independence,
honest trace labels, dynamic action timing, support/fallback behavior, and
small-case finite-beam parity against an independent exhaustive account search.
Run the required full test suite and `git diff --check` before outcome execution.

Primary references: [Brown, Smith and Sun (2010)](https://doi.org/10.1287/opre.1090.0796)
derive information-relaxation bounds under a different, properly optimized
relaxed problem; a finite heuristic solution alone does not supply their upper
bound. [Brown and Smith (2011)](https://doi.org/10.1287/mnsc.1110.1377)
apply dual bounds to portfolio optimization with transaction costs. Neither
paper establishes the validity or economic accuracy of these BTC diagnostics.

The original development periods remain reused exploratory evidence with
2bull/4bear/2sideways quarters. No independent confirmation, selection-adjusted
inference, prospective receipt proof or high-probability generalization is
claimed by this decomposition.
