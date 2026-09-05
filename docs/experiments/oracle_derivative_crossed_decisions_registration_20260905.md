# Frozen derivative mean / variance crossing registration

This diagnostic is fixed before its four new policy outcomes. It follows 105
observed policy names, including the completed derivative ablation, and is
adaptive research on reused development validation. There is no model selection,
new fitting, recalibration, interval estimation, test/outer evaluation or
deployment in this family. All unsuccessful comparisons remain in the output.

## Question and exact interventions

In the parent ablation, adding only the two perpetual-flow inputs to technical29
reduced the scaled return MSE relative to technical29 but increased scaled
QLIKE. Its allocation results also changed. A model with both altered mean and
variance cannot identify which forecast component affected the decisions.

Freeze all 16 existing `technical_scaled` and `perp_flow_scaled` forecast NPZs
over folds 5–12. Cross their components in exactly two ways:

| Cross | Conditional mean source | Conditional variance source |
| --- | --- | --- |
| `mu_perp_flow_scaled__var_technical_scaled` | perp_flow_scaled | technical_scaled |
| `mu_technical_scaled__var_perp_flow_scaled` | technical_scaled | perp_flow_scaled |

Apply both the unchanged point mapper and the unchanged conditional utility
controller with risk aversion 1 and cost multiplier 2 to each cross: four new
policy names. No risk-0 cross is run, because the same mean with risk aversion 0
already exists and ignores variance. No all8 representation is introduced.
Neither lower forecast loss nor one favorable economic result is a selection
criterion. This isolates decision responses to swapping frozen components; it
does not establish causal economic effects outside these observed paths.

The point mapper uses `mu / max(sqrt(variance), .001)` through its existing tanh
map, with exposure bounds [.5, 1.12]. The utility controller uses actual own
cash and units and the current open before current-close marking or borrowing.
It compares the fixed bounded intents with the endogenous no-trade action,
using the existing approximate six-hour mean-variance score. This is not an
exact log-wealth, Bayes, global return or MaxDD optimum.

## Immutable evidence and support

The config pins the parent registration JSON SHA-256
`5c1fcda4bf329c6e099adada30de2fc525bb8aa50eb25e5753092a47d5147891`,
the parent result JSON SHA-256
`2360421c9436416e149bf30642caea4c019f0b4b21ab61d488b62694f14a660a`,
and all 16 source forecast SHA-256 values. Verify the result's canonical
registration binding, original configuration, original helper hashes and
Spot artifact proof before computing new policies. Runtime registration also
records the new runner and every executed local helper source hash, versions,
commit revision, all source-control target hashes and full-trace hashes.

Use the original validation windows for folds 5–12: 2021-04-16 13:45 UTC through
2023-04-16 13:45 UTC exclusive, each three months, on the complete 15-minute
calendar. The 18/3/3-month fitting and two calibration segments are already
frozen in the source artifacts; this runner does not revisit them. Each pair
must have exactly equal timestamps, inference masks, scoring masks and actual
outcome arrays. The expected common support is 2,587 inference rows and 2,575
scoring rows. Preserve all inference rows even where future outcomes cannot be
scored; never use the scoring mask to cancel an order. Forecasts are set to NaN
outside the existing causal six-hour inference mask before either policy sees
them. Features underlying those forecasts use only completed bars through t−1.

The existing two bull, four bear and two sideways quarters remain unchanged.
The minimum of three quarters per regime therefore fails in advance, including
when all observed directional signs are favorable. No all-trend or
high-probability generalization claim can pass this diagnostic.

## Accounting, controls and output inventory

Keep the parent cash/units simulator, initial B&H inventory, six-hour UTC
decisions, next-bar open fills, max step .08, deadband .01, one-way cost .00055,
annual borrowing .10, and original gaps. Unknown future fill prices are handled
by the existing simulator's actual-fill clipping. Plan utility targets once
using the base execution costs; replay the same submitted targets under base
and doubled transaction cost / borrowing. Do not replan under stress costs.

Copy and verify these eight existing controls on identical support: B&H,
`common_robust`, and technical_scaled / perp_flow_scaled each with point,
utility risk 0 and utility risk 1. Before copying a control row, verify its
target digest, calendar, causal mask, regime, and every economic metric by
canonical base/stress replay. Copy its full trace when present and retain its
original hash. No control is retrained or newly selected.

Expected output is 96 economic rows and target NPZs: (eight existing controls +
four new crosses) × eight folds. Save full utility decision traces separately
and compact hash-bound diagnostics in rows. Report the maximum source-control
replay discrepancy, source links, support counts and all 12 summaries. Preserve
per-fold output and immutable registration/results. Existing output cannot be
silently replaced with different content.

For each of the four crosses, compare against both technical_scaled and
perp_flow_scaled using the same policy, giving eight fixed paired comparisons.
Report candidate-minus-reference AlphaEx, MaxDDDelta, turnover, trades, fees
and borrowing under both cost assumptions, averaging quarters equally within
all / bull / bear / sideways. Economic summaries are point estimates only;
there is no new forecast or economic interval estimation and no
selection-adjusted inference. Preserve the original failed
sample-coverage gate. New rankings, adaptive retries, alternate risk weights
and retraining after outcomes are outside this registration.

Sources of the frozen hypotheses and contract are the
[parent registration](oracle_derivative_ablation_registration_20260905.md),
[crossing config](../../configs/oracle_derivative_crossed_decisions_20260905.yaml)
and [runner](../../unidream/experiments/oracle_derivative_crossed_decisions.py).
No external result is treated as proof that either crossing will improve BTC
investment performance.
