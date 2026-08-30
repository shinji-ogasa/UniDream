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
`block_reason`, and optional run/fold/phase context fields.  A row passes only
when it has finite masked targets and an actual nonzero gradient step.
`finite_loss_steps` and `gradient_steps` count only steps on which that head's
loss actually ran; aggregate WM loss no longer credits a skipped head.
Multi-output heads inspect the final projection row, so a valid h4 output
cannot hide a zero-gradient h64 sibling.

For future-only `seq_len=64`, h64 and `position_utility_horizon=64` report
`target_count=0`, `mask_fraction=0`, and `status=block` with
`block_reason=zero_valid_targets`.

The default remains diagnostic-only for legacy callers.  Conditional runs, or
an explicit `require_target_gradient_coverage: true`, write the JSONL artifact
first and then raise `TargetGradientCoverageError` when any enabled row is
blocked.  If a checkpoint was written before the gate, a
`world_model.pt.blocked.json` marker records `status=blocked` and
`promotable=false`.  `WorldModelTrainer.load(path)` treats that marker as an
active consumer-side block and raises `TargetGradientCoverageError` by
default, so a marker cannot be ignored by normal promotion/evaluation.  Only
an explicit `allow_blocked_legacy=True` load is permitted for historical
diagnostics; it does not make the checkpoint promotable.
`require_target_gradient_coverage: false` preserves the legacy
diagnostic-only continuation.

### Chronological OOF contract

`unidream.experiments.chronological_oof.chronological_oof_predict` uses an
expanding or rolling-origin prefix.  A target is label-complete at
`target_end` (default `row + horizon`) and is eligible only when
`target_end <= prediction_index - purge`; random K-fold is not used.  Early
rows stay NaN/false in `prediction_mask`; no in-sample fill is performed.
Mask外に一つでも finite な成分がある行（例: `[finite, NaN]`）も拒否し、
partial fill を許さない。OOF validator、expanding standardizer、conditional
predictive-state bundle の三つの入口でこの契約を検証する。
All availability masks are strict boolean arrays; integer, float/NaN, and
string masks are rejected instead of implicitly coerced.  An optional strict
boolean `row_eligibility_mask` from the caller is combined with finite
features for prediction-origin eligibility.  Training-label eligibility is a
separate mask that additionally requires `valid_target_mask` and finite
targets.  The result records `prediction_eligibility` and
`training_label_eligibility` counts and provenance separately.  Thus a
future-target value or mask on the origin row cannot suppress that row's
decision-time OOF state; it can affect only later training prefixes.  A false
origin never calls the callback and remains NaN/false.  An incomplete target
tail can still receive a prediction when its decision-time feature/window is
eligible.  `prediction_mask` means only finite callback output, not score/eval
label completeness; downstream scoring/evaluation must apply a separate
label-completeness mask rather than reusing the target-training mask.  Its
caller-supplied provenance is retained in the OOF result.  For
window/sequence inputs, the caller must pass one eligibility value per window;
no sidecar is auto-zero-filled.
The OOF validator requires both full-row eligibility masks, their count and
provenance details, and an explicit `provenance.in_sample: false`; it also
rejects prediction/training masks that fall outside origin eligibility.  The
producer persists the exact per-row `target_end_exclusive` vector.  Consumers
must validate that vector (and may only supply an identical external vector as
a cross-check), then automatically check every origin's strictly increasing,
unique training indices for `idx < t`, `training_label_eligibility_mask`, and
`target_end_exclusive[idx] <= t - purge`.  Origin records are required for
every finite `prediction_mask` row, with unique prediction indices and aligned
counts, so a fabricated finite state without fit provenance cannot pass.
P1 callers must make the explicit `target_end=t+h+1` construction part of
their producer manifest; this implementation still records the equivalent
default vector for the model-agnostic helper.
The conditional boundary requires this raw validated OOF result in addition
to the train/val/test state views.  Split-only state masks therefore cannot
bypass raw OOF provenance.  Each split must carry strict increasing integer
`{train,val,test}_row_indices`; its values, state mask, and both eligibility
masks must equal the corresponding indexed rows of the raw OOF result.  The
current safe contract accepts raw views only; transformed/standardized state
is blocked until an explicit causal transform artifact and input-row mapping
are implemented.
Horizon, purge, train-size/window, step, target-end cutoff, and standardizer
history options require actual integer types; bool, fractional, and string
coercion are rejected. Conditional flags and mapping `enabled` values likewise
require actual booleans.
Callback metadata is retained per origin for model, normalizer, calibrator,
and teacher-weight provenance.

`chronological_oof_standardize` fits each row's mean/std from prior OOF rows
only.  It leaves rows without enough prior OOF history unavailable.

### Conditional OOF artifact boundary (schema v1)

The conditional path now has a separate, hashable artifact boundary in
`unidream.experiments.chronological_oof`.  `build_conditional_oof_artifact`
requires the producer to persist the explicit future-only rule
`target_end_exclusive[t] = t + horizon + 1`; the historical helper's
model-agnostic default is intentionally not reused here.  The artifact keeps
the full-row prediction mask and training-label mask separately, retains NaN
outside the prediction mask, requires `provenance.in_sample: false`, and
records strict SHA-256 fields for the origin records, checkpoint, normalizer,
calibrator, teacher weight, and the P0-C `ActionExecutionContract`.

`validate_conditional_oof_artifact` re-runs the raw chronological checks and
also rejects duplicate/overlapping origins, any `idx >= origin`, a target end
after `origin - purge`, wrong one-bar decision-to-fill delay, root/provenance
hash disagreement, content tampering, or a contract-hash mismatch.  Both the
root and provenance carry the canonical `ActionExecutionContract` mapping;
the mapping is reparsed with `ActionExecutionContract.from_config` and its
content-derived hash must match every alias.  Coverage is represented as an
explicit head-by-horizon list.  A zero-covered h64 row is kept in the artifact
with its block reason; it is never silently removed.  The strict consumer gate
rejects that row, while the relaxed producer mode exists only to preserve the
diagnostic evidence.

`write_conditional_oof_artifact`/`load_conditional_oof_artifact` use typed
base64 array payloads so NaN and mask dtypes survive JSON round trips.  Array
dtype, dimensions, element/byte counts, and JSON nesting are bounded before
allocation, and writes use a same-directory unique temporary file followed by
an atomic replace.  A strict conditional config must set
`require_conditional_oof_artifact: true` (or
`conditional_oof_artifact_required: true`) and supply all external bindings:
`expected_heads_horizons`, an `expected_hashes` mapping containing
`checkpoint_sha256`, `normalizer_sha256`, `calibrator_sha256`, and
`teacher_weight_sha256` (the `teacher_sha256` alias is accepted), plus
`expected_action_execution_contract_hash` (and optionally the full canonical
`expected_action_execution_contract` mapping).  Omitting any binding is a
configuration blocker; values are never copied from the artifact itself.  A
nested envelope may add only indexed `train`/`val`/`test` views and their masks;
outer predictions, origins, provenance, targets, coverage, hashes, or schema
keys are rejected even when their values happen to match.  The existing
raw-bundle path remains for the current integration fixture and historical
diagnostics; it is not a claim that full per-fold WM/normalizer/calibrator/
student replay is complete.

Implementation commits for this contract are `8357ab1` (schema, validator,
hash and persistence), `950d426` (fixture tests), and `a3912c4` (predictive
state artifact-envelope gate), branch
`exp/p0b-oof-artifact-contract-20260830`.  No WM/BC/AC result was run or
promoted by this work.

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
uv run python -m unittest tests.test_chronological_oof_teacher tests.test_teacher_inventory_contract tests.test_world_model_target_coverage -v
```

The new tests cover h64 zero target/gradient detection, output-specific and
per-head step coverage, coverage-gate blocking/markers, run/fold/phase
context, strict mask and integer types, same-row future-label perturbation,
explicit origin/window eligibility, horizon/purge eligibility, no early-row
fill, partial-finite values outside an OOF mask, missing eligibility/in-sample
provenance, alias consistency, origin/target-cutoff integrity, exact indexed
split-view matching, split-only conditional bundles, and hindsight inventory
rejection.

Observed result: 26 scoped contract tests passed.  The complete repository
suite passed (149 tests).

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
