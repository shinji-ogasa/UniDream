# P0-A availability propagation report

Status: complete for the owned runtime/dataset surface.

## Objective

Propagate the schema-v4 Spot, funding, and mark availability sidecar into row
and sequence-window eligibility without changing the canonical 17-column model
observation.  Keep the historical v3 path readable but explicitly
`legacy_v3_unverified`.

## Contract implemented

- `spot_bar_observed` is always required when a sidecar is consumed.
- `funding_rate_available` and `mark_close_available` are required only when
  their corresponding feature groups are enabled.
- Required and present known mask columns must be pandas boolean dtype with no
  missing values.  Missing columns, non-boolean values, duplicate or unsorted
  indexes, and body timestamps absent from the sidecar fail closed.
- A v4 sidecar may cover the complete expected grid while the feature body is
  a strict subset.  Body timestamps are selected by label only after complete
  membership validation; the sidecar is never sorted, interpolated, or
  compacted.
- `valid_sequence_starts` returns offsets into the original body.  A window is
  valid only when every row has all required availability flags and every
  adjacent timestamp is exactly one configured interval apart.  Therefore an
  unresolved Spot row represented by a missing body timestamp also invalidates
  windows crossing the gap.
- `SequenceDataset` maps its dataset index to the original valid offset.  It
  retains all body rows in `features` and keeps observation width unchanged.
- `WFODataset` validates the full sidecar once, carries split-local timestamps,
  masks, and sidecars, and constructs eligible train/validation/test windows.
  The existing numpy-valued stage boundary receives a `FeatureArray` carrying
  the same metadata, so existing `wm_stage` call sites consume the contract
  without adding sidecar columns to the model input.

## Runtime propagation

`load_training_features` still returns `(features, returns)` by default for
the v3-compatible public API.  On a validated v4 cache hit it attaches the
sidecar and contract settings to both pandas objects via attrs; callers may
request `(features, returns, availability)` with
`return_availability=True`.  A v4 request with missing/incomplete artifacts is
fail-closed and cannot fall back to the legacy downloader.  The official v4
rebuild remains the producer of the four validated cache artifacts; its
sidecar is consumed by the same runtime path as a cache hit.

## Files changed

- `unidream/data/availability_contract.py`: sidecar schema, alignment, and
  row eligibility validation.
- `unidream/data/window_quality.py`: optional full-sidecar window checks.
- `unidream/data/dataset.py`: original-offset sequence filtering and WFO
  propagation.
- `unidream/experiments/runtime.py`: v4 sidecar attachment, explicit optional
  third return, and fail-closed missing-v4 handling.
- `tests/test_availability_contract.py`, `tests/test_runtime_v4.py`,
  `tests/test_window_quality.py`: regression coverage.

## Verification evidence

Executed from the P0-A worktree:

```text
uv run python -m unittest tests.test_runtime_v4 tests.test_window_quality -v
Ran 11 tests in 0.148s
OK

./.venv/bin/python -m unittest tests.test_window_quality tests.test_runtime_v4 tests.test_availability_contract -v
Ran 13 tests in 4.368s
OK

./.venv/bin/python -m unittest discover -s tests -v
Ran 128 tests in 3.982s
OK
```

`git diff --check` also passed before commit.  No raw data, checkpoint, or
generated cache artifact is included.

## Deliberate boundaries

The v3 body remains unchanged and is not promoted to verified.  The existing
test-stage code still materializes the full body for direct encoded-path
diagnostics; consumers that bypass `SequenceDataset` should use the exposed
`test_dataset().valid_starts`/row mask when evaluating only eligible windows.
