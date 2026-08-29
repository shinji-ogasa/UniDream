# Research cache v4 contract

Schema v4 is a future cache format. The existing development `*_v3_*`
parquet files are not upgraded in place and do not pass the availability
quality gate.

## Files and contract

For a cache tag, v4 writes four files:

- `<tag>_features.parquet`: exactly the canonical 17 model feature columns,
  in metadata order. Availability flags are never appended to this body.
- `<tag>_returns.parquet`: one `returns` column with the exact body index.
- `<tag>_availability.parquet`: the complete expected interval grid, with
  boolean `spot_bar_observed`, `funding_rate_available`, and
  `mark_close_available` columns. Additional availability columns must be
  declared in metadata and remain boolean.
- `<tag>_metadata.json`: schema version, feature/sidecar columns, row counts,
  interval, explicit gap list and policy, source provenance and digest,
  schema digest, and deterministic content digests for all three DataFrames.

The sidecar may mark an unresolved spot timestamp false while the feature body
omits that row. Body and returns must still align exactly. Duplicate,
non-increasing, non-contiguous, non-finite, reordered, stale, or mixed-generation
files fail closed. The loader never sorts, fills, interpolates, or silently
rebuilds an invalid v4 hit. Availability masks are metadata only; they are not
mixed into model values.

`unidream.data.cache_v4.write_cache_v4` is the explicit writer and
`load_cache_v4` validates every hit. The training runtime reports a complete
v4 hit as `v4_verified`, rejects an incomplete/invalid v4 set, and reports the
historical v3 path as `legacy_v3_unverified`. Pass `require_v4_cache=True` to
reject that legacy path.

## Gap policy

The required policy is `exclude_windows_crossing_gaps`; interpolation is
forbidden. `unidream.data.window_quality.valid_sequence_starts` filters
sequence offsets using the original timestamps and optional spot mask. A
future training integration must pass those offsets before constructing a
`SequenceDataset`; compacting a gapped array first would hide the outage.

The official development-cache audit is recorded in
[data_quality_gap_recovery_2018_2024.md](data_quality_gap_recovery_2018_2024.md)
and its JSONL ledger. It found 18 of 542 expected timestamps in official Spot
sources and 524 still unresolved. The 18 recovered timestamps are eligible
for a future v4 raw-row regeneration only after official OHLCV and as-of
external inputs are recomputed. The current v3 body was not modified.

The unresolved 524 timestamps must remain explicit sidecar gaps. Training
windows crossing them are ineligible. Execution/evaluation must either segment
metrics at a gap or record a separate, explicit policy attributing a return
spanning the gap to the position held immediately before it; a missing return
must not be silently dropped or assigned to a post-gap position.

No v4 dataset has been generated from the current cache yet; this document and
the synthetic tests validate the contract only. Do not interpret the existing
v3 model cache as v4-compliant until an official-source regeneration completes.

## Reproduce the official audit

This command is intentionally read-only with respect to the cache and exits
non-zero while official gaps remain unresolved:

```bash
uv run python -m unidream.cli.audit_official_gap_recovery
```

Use `--allow-unresolved` only when collecting evidence. It still writes the
JSONL/Markdown evidence and never writes feature or returns rows.
