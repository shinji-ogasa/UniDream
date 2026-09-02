# P0-C action/execution contract

Status: legacy implementation snapshot; it is retained for historical
provenance and is not the current P1 production contract. The current
mask-aware implementation lives on
`exp/p1-production-chain-20260831`; its outcome-gap policy and counts supersede
this snapshot. Do not use this document as evidence for an experiment result
or as the source of truth for replay behavior.

Current source of truth:
[`p1_validation_integration_contract.md`](p1_validation_integration_contract.md)
and [`action_primitive_implementation_notes.md`](action_primitive_implementation_notes.md).

## Canonical contract

The immutable `ActionExecutionContract` in
[`unidream/eval/action_execution.py`](../../unidream/eval/action_execution.py)
is the source of truth. Its deterministic SHA-256 contract hash in this
revision is:

```text
6f5beb7865fceac5ecbcfbb31dd11e8fdada02e1841fecac1c17e22377bb624f
```

The tracked machine-readable artifact is
[`action_execution_contract.json`](action_execution_contract.json). The
availability/skip policies are part of the hashed contract. A missing decision
feature is an explicit scored hold commitment. A missing delayed fill prevents
execution, while an outcome gap after a successful fill preserves the executed
inventory and excludes only retrospective scoring. This prevents a consumer
from treating either case as an implicit legacy fallback.

The canonical fields are:

```json
{
  "position_min": 0.5,
  "position_max": 1.0,
  "candidate_deltas": [-0.08, -0.04, 0.0, 0.04, 0.08],
  "h_decision": 4,
  "commitment_bars": 4,
  "execution_delay_bars": 1,
  "fill_policy": "all_or_none",
  "partial_fill_policy": "unsupported",
  "tail_policy": "exclude_incomplete",
  "spread_bps": 3.0,
  "spread_convention": "full_quoted",
  "spread_side": "half_transition",
  "slippage_bps": 1.0,
  "fee_rate": 0.0003,
  "return_unit": "additive_log_return",
  "funding_included": false,
  "p_start": 1.0,
  "initial_countdown": 0,
  "countdown_decrement": 1,
  "boundary_cost_policy": "fill_only",
  "feature_unavailable_policy": "hold_and_score_commitment",
  "outcome_unavailable_policy": "exclude_block",
  "execution_skip_policy": "hold_commitment",
  "eligibility_masks_required": true
}
```

The JSON artifact also records the derived `transition_cost_rate`,
`commitment_countdown_reset`, `commitment_countdown_decrement`, and
`spread_side` values; `from_config()` ignores only these derivable fields and
recomputes the hash from the semantic contract.

`spread_bps=3` is the full quoted spread, therefore each transition charges
half-spread `1.5 bps`, slippage `1 bp`, and fee `3 bps`. The resulting cost is
`0.00055 * abs(next_position - previous_position)`. Funding and borrow are
excluded from this spot first-pass PnL contract.

## Timeline and replay

The shared replay is `decision t -> fill t+1 -> earn returns[t+1:t+5]`.
`candidate_positions()` clips each requested delta to `[0.50, 1.00]` and then
deduplicates the resulting absolute positions. A decision is accepted only at
the start of a complete four-bar block. During the block the feasible position
is unchanged. The first bar before the delayed fill and any incomplete final
block are outside the scored mask; candidate and benchmark paths use the same
mask. Cost is charged once at the fill bar. A partial fill, non-finite scored
input, unsupported delta, or unsupported semantic field raises; non-finite
cells outside an eligible block are ignored under the explicit masks.

`replay_action_path()` returns decision, fill, effective-position, countdown,
cost, gross-PnL, net-PnL, and `scored_mask` arrays. This makes boundary
handling inspectable rather than implicit. `select_block_decisions()` uses the
same feasible grid and replay geometry for the causal forecast-score teacher.
Its full-length `decision_block_scores` input contains one scalar cumulative
4-bar forecast at each complete decision start `t`; only that start cell is
read. Blocked/outcome-bar cells are ignored, so future per-bar features cannot
change an earlier teacher decision. U0 is kept separate through
`select_hindsight_block_decisions()`, an iterative bottom-up DP that may
inspect all realized complete bars for an upper-bound-only diagnostic. Both
paths share bounds, delay, commitment, cost, tail mask, and replay geometry,
but action equality is not a contract.

The historical path described by this snapshot required full-length boolean
`decision_eligible[T]` and `score_eligible[T]` masks. In the current contract,
the shared helper derives `decision`, `fill_complete`, `outcome_complete`,
`executed`, and `scored_action` from explicit origin/forecast/bar-availability
inputs plus realized-return finiteness. A feature gap holds and can score its
finite outcome; a fill gap emits no fill; and a later outcome gap carries the
filled position but is excluded from PnL. The schedule remains `t+4` and is
never compressed. See the current source-of-truth documents linked above.

Counts are intentionally distinct: `scheduled_decisions` is the fixed
schedule, `eligible_decisions` is the decision-feature count,
`eligible_blocks` is the executable count, `scorable_blocks`/`complete_blocks`
are the outcome denominator, `filled_blocks` counts non-zero fills,
`execution_skipped_blocks` counts scored hold commitments, and
`excluded_blocks` counts outcome gaps. Metrics are conditional on the shared
scored bars; excluded wall-clock gaps are not silently treated as zero-return
observations.
After a fill, `effective_positions` retains the live inventory through an
unscored incomplete tail; the tail contributes zero PnL via `scored_mask`.
For teacher diagnostics, replay expands each decision-start scalar evenly
across its four scored bars only to populate trajectory utility arrays; the
scalar itself remains the selector's sole forecast input.

## Oracle and Backtest integration

- [`unidream/data/oracle.py`](../../unidream/data/oracle.py) adds
  `conditional_oracle_teacher_path()` and `hindsight_upper_bound_path()`.
  The teacher calls the causal local selector, while U0 calls the separate
  iterative hindsight selector. U0 is explicitly diagnostic-only and must not
  feed a training feature, weight, threshold, or target.
- [`unidream/eval/backtest.py`](../../unidream/eval/backtest.py) adds
  `ActionExecutionBacktest`. The existing `Backtest` is unchanged unless an
  explicit `action_execution_contract` is supplied. Contract metrics are
  computed only over the shared scored mask and record both the contract and
  eligibility-mask hashes/counts. Contract mode requires both masks and passes
  the same masks to strategy and benchmark replay.
  When the contract is supplied, `action_positions_are_deltas` must be
  explicitly `True` or `False`; no value-based delta/absolute inference is
  permitted. `True` denotes decision deltas and `False` denotes a strict
  absolute committed-position path.
  `run_contract_backtest()` converts an absolute actor path strictly into
  contract deltas before invoking the opt-in Backtest and rejects any
  explicitly supplied legacy spread/fee/slippage/delay/initial-position
  override instead of silently dropping it.
- [`unidream/experiments/transition_advantage.py`](../../unidream/experiments/transition_advantage.py)
  has a separate contract branch. The generic `compute_transition_advantage()`
  rejects a contract because its realized-return computation is not a causal
  teacher. `compute_hindsight_transition_advantage()` is an explicit,
  `diagnostic_only=True` row-wise diagnostic that requires an independently
  supplied benchmark/causal-policy inventory trace and returns
  `future_derived=true`, `training_eligible=false`, and `replayable=false`.
  Sequential realized-future selection belongs only to U0's iterative DP;
  neither diagnostic output may be used as a teacher. Missing contract fields
  fail before computation.
- Validation/test/AC stage adapters use the contract branch only when the
  manifest explicitly includes `action_execution_contract` (or explicitly
  opts into the contract path). The historical `policy_fire` every-bar
  evaluator is disabled for this path rather than being allowed to select an
  AC checkpoint under legacy geometry.

The stage adapter also requires both masks; stage callers that do not yet have
the availability sidecar fail closed instead of entering a legacy replay.

The conditional path does not infer costs from the historical `costs` section.
`ActionExecutionContract.from_config()` requires the complete semantic field
set and, by default, requires the canonical P0-C values. Consequently an
omitted contract cannot fall back to historical `5 bps / 2 bps / 0.0004`,
delay-0, flat-start, or partial-fill behavior.

## Tests and verification

Dedicated contract tests are in
[`tests/test_action_execution_contract.py`](../../tests/test_action_execution_contract.py).
They cover:

- clip-then-unique candidate positions and bounds;
- one-bar fill alignment and four-bar countdown;
- blocked-bar immutability and incomplete-tail exclusion;
- live inventory retention through an unscored incomplete tail;
- all-in cost arithmetic and additive-log-return PnL;
- immutable/hashable round trip and missing-config fail-closed behavior;
- unsupported funding/partial/simple-return semantics;
- strict explicit delta/absolute Backtest modes and the absolute-position
  adapter;
- causal teacher invariance to future-block score perturbations;
- cumulative decision-start score API, including ignored blocked/outcome cells;
- separate iterative hindsight U0 and causal-teacher masks/constraints;
- U0 long-window replay without recursive stack growth;
- U0/teacher/Backtest replay geometry parity;
- row-wise hindsight transition diagnostics that require explicit opt-in and
  persist future-derived/non-training provenance;
- sequential U0 inventory persistence across excluded schedules and iterative
  replay without recursive stack growth;
- full-length availability masks: required/strict-boolean validation,
  decision-feature hold commitments versus outcome-gap exclusion without
  schedule compression, strategy/U0/teacher/benchmark parity, mask
  counts/hash, and fail-closed stage adaptation;
- strict rejection of legacy adapter overrides, strict numeric/boolean/config
  validation, canonical non-cost geometry, and a tracked JSON contract
  artifact that round-trips to the canonical hash.

Commands run in the worktree:

```text
uv run python -m unittest tests.test_action_execution_contract -v
Ran 30 tests ... OK

uv run python -m unittest tests.test_action_execution_contract tests.test_backtest_final_excess tests.test_leak_discipline -v
Ran 39 tests ... OK

uv run python -m unittest discover -s tests -q
Ran 153 tests ... OK

git diff --check
OK
```

The full suite includes pre-existing data-quality diagnostics that print a
failed availability gate for their intentional negative fixtures; the unittest
result itself is `OK` (153/153).

## Boundary and non-goals

- The contract is opt-in because the current repository configuration and
  `fold_inputs.py` ownership remain historical. A production conditional-
  Oracle manifest must explicitly carry the serialized contract and persist
  its hash with every teacher/backtest artifact.
- The stage adapters require actor absolute paths to already obey the block
  contract; they reject unsupported per-bar changes instead of silently
  clipping them. A future student should emit contract deltas directly or
  call the strict absolute-path adapter.
- Availability policy is explicit in the contract:
  `feature_unavailable_policy=hold_and_score_commitment`,
  `outcome_unavailable_policy=exclude_block`, and
  `execution_skip_policy=hold_commitment`. The canonical path requires both
  full-length masks and rejects missing/non-boolean/incorrect-length inputs.
  Existing stage adapters do not yet produce those masks, so their contract
  opt-in is intentionally blocked until a P0-A sidecar is wired in.
- Legacy direct encoding and legacy every-bar paths are not integrated with
  this mask-aware contract and are blocked from being used as its adapter.
- No production Supabase/Space deployment, live execution, funding cash flow,
  borrow cost, partial fill, or database transaction was changed or verified.
- Existing historical APIs retain their legacy defaults for reproducibility;
  callers must not use them as evidence for the P0-C contract unless they pass
  the explicit contract adapter.
