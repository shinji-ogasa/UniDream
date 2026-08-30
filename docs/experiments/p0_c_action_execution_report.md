# P0-C action/execution contract

Status: implemented as an explicit opt-in path; no existing Plan011 run is
silently migrated. This report describes the implementation at commit
`2e11cb9` on branch `exp/p0-c-action-execution-20260830`.

## Canonical contract

The immutable `ActionExecutionContract` in
[`unidream/eval/action_execution.py`](../../unidream/eval/action_execution.py)
is the source of truth. Its deterministic SHA-256 contract hash in this
revision is:

```text
26ac8e44fac6de2ad55f83179acc3d2033cc943a1edeff498f92f9174b19015b
```

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
  "boundary_cost_policy": "fill_only"
}
```

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
mask. Cost is charged once at the fill bar. A partial fill, non-finite input,
unsupported delta, or unsupported semantic field raises.

`replay_action_path()` returns decision, fill, effective-position, countdown,
cost, gross-PnL, net-PnL, and `scored_mask` arrays. This makes boundary
handling inspectable rather than implicit. `select_block_decisions()` uses the
same feasible grid and replay geometry for a forecast-score teacher or the
realized-future U0 diagnostic.

## Oracle and Backtest integration

- [`unidream/data/oracle.py`](../../unidream/data/oracle.py) adds
  `conditional_oracle_teacher_path()` and `hindsight_upper_bound_path()`.
  Both call the shared block selector; U0 is explicitly diagnostic-only and
  must not feed a training feature, weight, threshold, or target.
- [`unidream/eval/backtest.py`](../../unidream/eval/backtest.py) adds
  `ActionExecutionBacktest`. The existing `Backtest` is unchanged unless an
  explicit `action_execution_contract` is supplied. Contract metrics are
  computed only over the shared scored mask and record the contract hash.
  `run_contract_backtest()` converts an absolute actor path strictly into
  contract deltas before invoking the opt-in Backtest.
- [`unidream/experiments/transition_advantage.py`](../../unidream/experiments/transition_advantage.py)
  has a separate contract branch. It uses delayed four-bar block utility and
  does not call legacy min-hold smoothing, same-bar reward, or the historical
  action grid. Missing contract fields fail before computation.
- Validation/test/AC stage adapters use the contract branch only when the
  manifest explicitly includes `action_execution_contract` (or explicitly
  opts into the contract path). The historical `policy_fire` every-bar
  evaluator is disabled for this path rather than being allowed to select an
  AC checkpoint under legacy geometry.

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
- all-in cost arithmetic and additive-log-return PnL;
- immutable/hashable round trip and missing-config fail-closed behavior;
- unsupported funding/partial/simple-return semantics;
- absolute-position adapter and explicit delta path;
- U0/teacher/Backtest trajectory parity;
- transition-advantage contract rows and legacy-cost non-inheritance.

Commands run in the worktree:

```text
uv run python -m unittest tests.test_action_execution_contract -v
Ran 11 tests ... OK

uv run python -m unittest tests.test_action_execution_contract tests.test_backtest_final_excess tests.test_leak_discipline -v
Ran 20 tests ... OK

uv run python -m unittest discover -s tests -v
Ran 134 tests ... OK

git diff --check
OK
```

The full suite includes pre-existing data-quality diagnostics that print a
failed availability gate for their intentional negative fixtures; the unittest
result itself is `OK` (134/134).

## Boundary and non-goals

- The contract is opt-in because the current repository configuration and
  `fold_inputs.py` ownership remain historical. A production conditional-
  Oracle manifest must explicitly carry the serialized contract and persist
  its hash with every teacher/backtest artifact.
- The stage adapters require actor absolute paths to already obey the block
  contract; they reject unsupported per-bar changes instead of silently
  clipping them. A future student should emit contract deltas directly or
  call the strict absolute-path adapter.
- Availability/missing-feature skip policy is outside this action replay
  module. Non-finite returns fail closed; data eligibility must be established
  before replay.
- No production Supabase/Space deployment, live execution, funding cash flow,
  borrow cost, partial fill, or database transaction was changed or verified.
- Existing historical APIs retain their legacy defaults for reproducibility;
  callers must not use them as evidence for the P0-C contract unless they pass
  the explicit contract adapter.
