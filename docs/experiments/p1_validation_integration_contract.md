# P1 validation integration contract

Status: implementation boundary; no P1 validation or outer result has been
executed by this document.  The immutable experiment definition remains
`p1_recovery_prereg_manifest.json` and its two authenticated registries.

## Artifact chain

The production path is one directional.  A downstream artifact may bind an
upstream digest; an upstream producer must never require or copy its own output
digest as an input expectation.

```text
fixed manifest + 56-trial registry + 16-comparison registry
  -> authenticated validation forecast artifact
  -> canonical h4 action primitive artifact
  -> persisted fixed MBB-start artifacts for L=8,16,32
  -> persisted comparison bootstrap result
  -> fixed-family Holm + Wilson/coverage decision
  -> frozen validation decision
  -> at most one report-only outer operation, only if validation passed
```

Every arrow is checked by an exact external file SHA-256 plus the content
digest declared by the loaded upstream artifact.  A hash read only from the
candidate payload is not an external binding.  The initial producer is bound
by source/registry expectations; its output digests are recorded only after an
atomic write, then required when that file is loaded for the next stage.

The persisted MBB index and result writers return the SHA-256 of the complete
post-rename NPZ bytes.  That file digest is ledger metadata only: it is never
written into the NPZ payload, whose internal `artifact_sha256`/`result_sha256`
digests remain canonical content bindings.  Production path loaders require
both the caller-pinned `expected_file_sha256` and the corresponding external
internal digest, hash the exact regular-file bytes before parsing, and fail
closed on omission, replay, or byte tampering.  The loaded index capability
exposes `file_sha256`; production MBB provenance records it alongside the
internal index digest.

The authenticated action-artifact path now exposes an identity-sealed
production capability.  MBB accepts a typed action input only after that
capability has been loaded with the externally pinned file SHA, three
inferential hashes, source binding, and field-specific mask registry.  Fixture
loads, direct dataclass construction, and self-declared file hashes remain
fail-closed and cannot promote.  A bare `LoadedP1ActionArtifact` is therefore
not sufficient evidence; production callers must pass the sealed capability
returned by the strict loader.

## Result-state semantics

The frozen preregistration always echoes
`prereg_results_observed=false`.  This describes the state at registration;
it must not be reused to claim that a produced result is unobserved.

- Before any validation fit: `validation_results_observed=false` and
  `outer_results_observed=false`.
- A persisted validation forecast/action/bootstrap result:
  `validation_results_observed=true` and `outer_results_observed=false`.
- The outer operation remains unavailable until all fixed validation gates are
  frozen.  Once attempted, success or failure consumes the single report-only
  operation and sets `outer_results_observed=true`; it cannot be retried or
  replaced.

Thresholds, registry membership, model/horizon/task coverage, source paths,
and action/bootstrap semantics cannot change at either state transition.

## Forecast sources

Synthetic and S3 results have different realized-return authorities but share
one downstream action-source interface.

- Synthetic S0/S1/S2: the persisted source is the exact registered
  seed/beta dataset and fixed validation fit at origin 90,000 over support
  `[90000,100000)`.  BTC/v4 returns are forbidden in this path.
- S3 injected/control: the persisted source is the sealed authenticated v4
  body plus the selected arm's injected or zero-injection-control returns,
  fixed fit origin 104,528, train start 52,492, and support
  `[104528,139568)`.  An injected arm may not silently fall back to raw v4
  returns.

The forecast artifact binds timestamps, realized returns, targets/labels,
context/target/prediction masks, every required model/task/horizon output,
train-mask counts, registry digests, source provenance, and future-perturbation
evidence.  Its production loader returns the only capability accepted by the
action producer; a plain mapping or directly constructed dataclass is a
fixture and cannot promote.

## Action and bootstrap boundaries

The action producer consumes the loaded forecast capability's h4 continuous
forecast, timestamps, selected-arm realized returns, and fixed masks.  It
cannot select a different source array.  Inventory is carried only over the
stored chronological scheduled grid.  Bootstrap never replays inventory over
resampled indices; it resamples the stored primitive metrics.

At the action-artifact storage boundary the full-bar availability input is
named `bar_available`.  The historical `score_eligible` argument is accepted
only by fixture/compatibility paths and is rejected for production writes and
loads; it must never be interpreted as a block-level score mask.

The persisted action artifact retains the current exact top-level contract and
exactly the three inferential action hashes (schema, content, and payload).
Its exact header also retains `metric_mask_registry` and, for production
artifacts, the registered `trial_id`, `source_binding`,
`source_binding_sha256`, and `paired_common_mask_sha256`.  The source binding
is a canonical projection of the sealed forecast capability: its source
schema/role, selected scenario/arm/seed/model/support identity, result-state
fields, capability binding digest, and complete registered source-hash map are
all pinned externally.  The source-binding digest is canonical JSON metadata;
it is not an action-output hash and it is never used as a self-referential file
digest.

Production action persistence requires the caller-pinned file SHA, all three
action hashes, exact production metadata, exact source binding, and the
identity-sealed `ForecastActionSource`.  The raw arrays passed to the wrapper
must be byte-for-byte/dtype-equal to that sealed capability; fixture arrays,
directly constructed forecast capabilities, and caller-selected action deltas
are rejected.  A successful production load registers an identity-sealed
`LoadedP1ActionArtifact` capability.  Fixture loads, direct constructors, and
`dataclasses.replace` copies cannot promote, even if they copy the private
seal fields.

The authenticated capability exposes an immutable typed MBB input.  Each
selected metric returns only its declared value fields and its effective mask:
utility fields use `outcome_complete_mask AND common_mask`, while action
comparison fields use `scored_action_mask AND common_mask`.  The input carries
the external file/action/source provenance alongside read-only arrays, so the
downstream MBB boundary cannot substitute a bare raw array or a common mask
from a different action artifact.

Action input expectations bind source/forecast/support/timestamp/common-mask
digests.  The producer computes exactly the three inferential output hashes:
schema, content, and payload.  The historical envelope helper is storage-only
and is not part of this contract.  The action writer returns the exact
post-rename file SHA-256 to the external ledger, never embedding it in the JSON
payload.  The subsequent persistence/load boundary requires the three output
hashes, the exact action-artifact file digest, the complete action mask-hash
registry, and (for paired results) the matching registry for the second arm.

For every unit, support, seed ordinal, and `L in {8,16,32}`, all 2,000
non-circular block-start vectors are atomically stored before a production
comparison can be promoted.  Production inference must load those starts with
an externally recorded digest and verify them against the fixed RNG formula;
internally rebuilt, unsaved, or self-declared starts are fixture-only.

The synthetic action aggregate additionally binds ordinal `0..9` to the
registered source seeds `20260830..20260839` in each sealed action artifact
(and checks candidate/baseline identities independently).  Reusing one
authenticated artifact under multiple ordinals is rejected before MBB input
validation.

## Promotion stop conditions

Promotion stops on any missing/N/A arm, model/task/horizon, source digest,
full-grid mask, action hash, persisted MBB index, comparison result, or fixed
coverage field.  The 16 comparison raw p-values must all exist before Holm is
evaluated.  S0 remains a negative-control safety gate; S1 recovery, both S2
adjacent monotonic contrasts, and both S3 injected-control comparisons must
pass their registered point, coverage, per-seed, Holm, Wilson, and
clairvoyant-sanity conditions.  A working pipeline or passing unit tests alone
is not an accuracy claim.
