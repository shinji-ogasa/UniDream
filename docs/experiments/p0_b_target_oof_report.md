# P0-B target / chronological OOF / inventory report

STATUS: partial

OBJECTIVE: Make target/gradient coverage auditable, provide a chronological
OOF contract for future-target-derived teacher inputs, and reject hindsight
teacher inventory in the new conditional path without changing historical
artifacts.

## Boundary and judgment

The legacy `hindsight_oracle_dp` path remains unchanged and is not a P0-B
pass.  It may be used for historical diagnostics, but its realized-future
position path cannot be reused as a conditional current inventory or as a
same-row predictive state.

The new conditional opt-in is fail-closed.  `run_fold`,
`prepare_fold_inputs`, `prepare_world_model_stage`, `run_bc_stage`, and the
predictive-state builder raise `ConditionalPathBlocked` unless a separately
provenanced chronological OOF bundle is supplied.  Full per-fold WM
retraining, early stopping, calibration, and student replay are not wired
through the current fold runner; silently routing them through the legacy
state would be a false P0 pass.

## Implemented

### Target and gradient coverage

`WorldModelTrainer` now emits one JSONL row per enabled output head/horizon
next to a supplied `world_model.pt` checkpoint:

`target_gradient_coverage.jsonl`

Each row contains `head`, `horizon`, `sequence_length`, `target_count`,
`mask_fraction`, `finite_loss_steps`, `gradient_steps`,
`nonzero_gradient_steps`, `target_coverage`, `gradient_coverage`, `status`,
and `block_reason`.  A row passes only when it has finite masked targets and an
actual nonzero gradient step.  Multi-output heads inspect the final projection
row, so a valid h4 output cannot hide a zero-gradient h64 sibling.

For future-only `seq_len=64`, h64 and `position_utility_horizon=64` report
`target_count=0`, `mask_fraction=0`, and `status=block` with
`block_reason=zero_valid_targets`.

### Chronological OOF contract

`unidream.experiments.chronological_oof.chronological_oof_predict` uses an
expanding or rolling-origin prefix.  A target is label-complete at
`target_end` (default `row + horizon`) and is eligible only when
`target_end <= prediction_index - purge`; random K-fold is not used.  Early
rows stay NaN/false in `prediction_mask`; no in-sample fill is performed.
Callback metadata is retained per origin for model, normalizer, calibrator,
and teacher-weight provenance.

`chronological_oof_standardize` fits each row's mean/std from prior OOF rows
only.  It leaves rows without enough prior OOF history unavailable.

### Current inventory contract

`current_inventory_from_replay` and
`validate_current_inventory_source` accept only `actual_replay`,
`benchmark_replay`, or `policy_replay`, with an explicit initial position.
Teacher/hindsight/oracle/signal provenance and a teacher-shaped benchmark path
are rejected.  The existing `current_positions_from_path` call remains in the
legacy path but is unreachable from the conditional opt-in because its input
builder is blocked.

## Verification

Scoped tests:

```text
uv run python -m unittest tests.test_world_model_target_coverage tests.test_chronological_oof_teacher tests.test_teacher_inventory_contract tests.test_world_model_gate0_validation tests.test_world_model_gate0_action_context -v
```

The new tests cover h64 zero target/gradient detection, output-specific
gradient coverage, same-row future-label perturbation, horizon/purge
eligibility, no early-row fill, and hindsight inventory rejection.  The two
existing WM gate modules remain green.

Observed result: 11 new contract tests passed; combined with the two existing
WM gate modules, 16 scoped tests passed.  The complete repository suite passed
(132 tests).

The full suite is required before promotion:

```text
uv run python -m unittest discover -s tests -v
```

`git diff --check` is also required.  No historical config or artifact is
rewritten by this change.

## Gaps / promotion gate

P0-B remains `partial` until a caller supplies and verifies full chronological
OOF WM retraining for each enabled future head, plus OOF normalizer/calibrator
and teacher-weight provenance, then consumes the row masks in BC/student
replay.  Conditional runs currently stop with an explicit blocker; this is
intentional and must not be reported as a conditional-Oracle result.
