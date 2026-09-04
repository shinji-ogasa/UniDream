# P1 action primitive implementation notes

These notes describe the implementation boundary for the immutable
`p1-action-primitive-v1` record schema.  The external schema and the
preregistration manifest remain authoritative and are not modified here.

## Causal masks and state

For every complete scheduled h4 row, the producer derives masks in this
order:

```
decision = origin_eligible_mask AND forecast_finite_mask
executed = decision AND fill_complete_mask
scored_action = executed AND outcome_complete_mask
```

The scheduled starts remain the chronological `0, 4, ...` grid from
`complete_decision_starts`.  A decision is mapped against the current
position, but the effective `selected_delta`/`selected_position` are emitted
only when the delayed fill is complete.  A fill gap therefore emits a zero
effective delta and holds the previous position; it never fabricates a fill.
When a fill exists but a future outcome bar is missing, the effective action
and carried position remain in the row and feed the next chronological row.
Only the retrospective score is removed: outcome-dependent values are NaN and
`scored_action_mask` is false.

## Metric domains

`common_mask` is the fixed paired-grid mask, not a universal finite-value
mask.  The producer and validator use the registered per-field domains:

| Fields | Required mask |
| --- | --- |
| `candidate_utility`, `benchmark_hold_utility`, `same_state_local_hold_utility` | `outcome_complete_mask AND common_mask` |
| `clairvoyant_utility`, `regret`, `opportunity`, `agreement` | `scored_action_mask AND common_mask` |

This keeps a finite hold utility available for a no-decision/active-hold row
while excluding action agreement and regret.  A downstream bootstrap or
comparison must apply the same field-specific masks; selecting only
`scored_action_mask` for every field would incorrectly drop valid hold PnL,
while selecting only `outcome_complete_mask` for every field would invent an
action comparison.

The v1 record has no separate `pnl_scored_mask` field.  A source adapter that
cannot map an active-commitment feature gap to the outcome-complete hold
utility domain must fail closed rather than overload `common_mask` or invent a
new schema field.

The authenticated action capability exports a full-grid mask-hash registry for
`origin_eligible_mask`, `forecast_finite_mask`, `fill_complete_mask`,
`outcome_complete_mask`, `scored_action_mask`, `common_mask`,
`utility_metric_mask`, and `action_metric_mask`.  A production MBB result marks
itself as `authenticated_capability`, records the selected field mask and its
digest, and must be persisted/reloaded with an independent copy of that
registry.  A result that only repeats hashes from its own payload is not a
production binding.

The persistence adapter names the source-bar availability vector
`bar_available`; the legacy `score_eligible` spelling is fixture-only and is
rejected at the production save/load boundary to avoid confusing it with a
derived block-level score mask.

## Hash boundary

The canonical artifact has exactly these three inferential hashes:

* `action_primitive_schema_sha256`
* `action_primitive_content_sha256`
* `action_primitive_payload_sha256`

`action_primitive_envelope_sha256` is retained only as a compatibility helper
for callers that need a deterministic pre-write envelope.  It is not emitted
or validated as part of a canonical artifact.  A persisted-file/storage
identity should instead be computed as a SHA-256 over the final written file
bytes after materialization.
