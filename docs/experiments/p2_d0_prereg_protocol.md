# P2-D0 corrected full17 versus OHLCV13 preregistration

Status: \`preregistered\`. This amendment is a result-free protocol boundary;
\`results_observed=false\` is fixed in the manifest and both registries. This
branch contains no experiment runner and must not fit, score, calculate
accuracy, or start an outer operation. The machine-readable sources of truth
are [\`p2_d0_prereg_manifest.json\`](p2_d0_prereg_manifest.json), the trial
ledger [\`p2_d0_trial_registry.jsonl\`](p2_d0_trial_registry.jsonl), and the
primary comparison ledger
[\`p2_d0_comparison_registry.jsonl\`](p2_d0_comparison_registry.jsonl).

The independent P2 manifest digest is
\`a0ac7357abadb4b459f0687b12fb5926089fe9e1bd0987990ede82750b952cd2\`.
It records the P1 predecessor digest
\`d1854827bd4aa204cc2b5cde375edf62583bf0d164b39e8ac25a6c10ad7dc0c4\` as
history/provenance only. P2 has its own registered digest and does not
authorize itself by reading or recomputing P1 results.

## Question and data boundary

The comparison is BTCUSDT at exactly 15-minute UTC timestamps using the
authenticated v4 body. \`full17\` is the canonical ordered feature vector:

\`\`\`text
open_ret, high_ret, low_ret, close_ret, vol_ret, RSI_14, macd,
macd_signal, atr_norm_ret, atr, rv_4, rv_16, rv_96, funding_rate,
basis, basis_mom, basis_abs
\`\`\`

\`ohlcv13\` is exactly the first 13 entries of that vector, in the same order.
It is a projection of the canonical full17 body, not an independently loaded
OHLCV dataset. The omitted funding/basis columns are never zero-filled,
interpolated, synthesized, or reordered.

For every horizon and split, eligibility is constructed from full17 context,
target, spot-bar, availability, finite-return, and finite-feature checks first.
Both arms then use the exact intersection of those masks and identical
timestamp rows, horizon, split, purge, and finite-prediction support. The
OHLCV13 arm cannot recover rows unavailable to full17, and neither arm may
compact the original grid or silently repair a gap. The common mask is:

\`\`\`text
common_mask[t,h] = context_eligible_full17[t]
                  AND target_complete_full17[t,h]
                  AND finite_ohlcv13[t]
                  AND finite_full17[t]
                  AND finite_return_window[t,h]
\`\`\`

The v4 parent is pinned by frozen metadata, source-provenance, schema, and
feature/returns/availability content SHA-256 values in the manifest. The
feature body has 173111 rows and its availability sidecar has 210336 rows;
those counts and all required availability columns are part of the binding.

## Forecast and action semantics

The only forecast horizons are \`h=1,4,8,16\`; \`h=64\` and a utility head
are excluded. A decision at row \`t\` observes the close of \`t\`, and an
action-capable h4 block fills at the close of \`t+1\`. Forecast targets are:

\`\`\`text
y[t,h] = sum(return[t+1:t+h+1])
target_end[t,h] = t+h+1       # right-exclusive
\`\`\`

Every exact 15-minute edge from \`t\` to \`t+1\` through \`t+h-1\` to \`t+h\`
is required. The following edge \`t+h\` to \`t+h+1\` is not required. A
missing, non-finite, or non-contiguous required row invalidates that horizon
while retaining its timestamp and false mask. The context is the
current-inclusive \`[t-63,t]\` window with all exact edges and complete v4
availability; it is not flattened into the model input.

The h4 action arm is specified but blocked until a canonical action-primitive
producer and P1-specific moving-block bootstrap are implemented and audited.
h1, h8, and h16 are forecast-only. Existing generic MBB and generic WM→BC→AC
application are forbidden for this preregistration.

## Splits and causal fitting

All ranges are UTC timestamp \`[start,end)\` ranges. Target, fill, and outcome
tails crossing a right boundary are excluded rather than clipped or moved:

| split | interval | role |
| --- | --- | --- |
| \`train_2018_2021\` | \`2018-01-19T17:00:00Z\` to \`2021-01-01T00:00:00Z\` | fit prefix |
| \`inner_calibration_2021\` | \`2021-01-01T00:00:00Z\` to \`2022-01-01T00:00:00Z\` | nested calibration only |
| \`outer_validation_2022\` | \`2022-01-01T00:00:00Z\` to \`2023-01-01T00:00:00Z\` | primary inferential gate |
| \`historical_report_2023\` | \`2023-01-01T00:00:00Z\` to \`2024-01-01T00:00:00Z\` | report-only |

Purge is 16 bars. Fit rows must be in the fixed causal common-row mask,
strictly before the origin, and satisfy \`target_end <= origin - 16 bars\`.
StandardScaler is fit separately per arm/model/task/origin/horizon on those
rows only; target and baseline outputs are not scaled. Calibration, if used,
is fit only inside \`inner_calibration_2021\` and frozen before outer
validation. Outer and historical rows cannot fit, scale, calibrate, select,
or tune.

Every arm/model/task/horizon/origin must have at least 16384 eligible history
rows; otherwise that fit is N/A and cannot be promoted. One-class binary
prefixes are N/A; no oversampling, class repair, or synthetic class is
allowed.

The fixed candidate set is zero-return, last-observed persistence, Ridge,
LogisticRegression, and a shallow fixed-budget HistGradientBoosting
regressor/classifier. Zero-return emits continuous \`0.0\` and binary \`0.5\`;
persistence emits \`h*return[t]\` and \`1-eps\`/\`eps\` with \`eps=1e-6\`.
Ridge, LogisticRegression, and HGB hyperparameters, seeds, tasks, and horizons
are machine-readable in the manifest and trial registry. MLP, deep neural,
transformer, deep boosting, and unregistered candidates are forbidden.

## Outcomes, coverage, and multiplicity

Primary forecast comparisons use proper MSE for continuous output and log loss
for binary output on identical common rows. Paired h4 utility is defined for
Ridge and HGB only after the action blocker is cleared; h1/h8/h16 never produce
utility. Active rate and turnover remain defined on the complete scheduled h4
grid, while scored utility uses only the fixed complete scored-action mask.

Required coverage gates are common-row, context-complete, and label-complete
fractions at least \`0.90\`, finite-prediction fraction at least \`0.95\`, and
scored-action fraction at least \`0.80\`. Any missing field, non-finite
denominator, or required N/A cell blocks that comparison. N/A is never
dropped, imputed, compacted, or converted to zero. If every primary cell is
N/A/undefined, status is \`blocked_no_inferential_result\` and no claim or
promotion is permitted.

The 14 rows in the primary comparison registry form one fixed
\`p2_d0_full17_vs_ohlcv13_primary\` Holm-Bonferroni family. Paired
moving-block bootstrap uses 2000 replicates and block lengths 8, 16, and 32,
with identical sampled indices for both arms and the original grid retained.
A replicate with \`N < block_length\` is N/A; there is no circular wrap, gap
compression, or denominator repair. The conservative maximum raw p-value over
block lengths enters the fixed Holm step-down procedure. The historical 2023
report arm never enters this family.

The 2023 interval is not a true untouched holdout if it was already observed
during development. It is historical report-only: it cannot train, calibrate,
select, gate, or be called prospective accuracy evidence. Before any future
promotion or prospective claim, a newly acquired, separately frozen,
post-registration holdout with registered timestamp, source, content/schema
hashes, and access boundary is required. If that boundary is absent, stop.

## Authenticated execution boundary

A future runner must invoke
\`unidream.experiments.p2_d0_prereg.load_authenticated_v4_runtime\`, which
first loads and validates this fixed manifest and then calls the authenticated
P1 v4 wrapper
\`unidream.experiments.runtime.validate_p1_v4_runtime_inputs\`. The generic
body validator is not a production boundary. All four explicit body paths,
frozen metadata, cache-local metadata, content/schema/source provenance hashes,
row counts, regular-file identity, and before/after TOCTOU snapshots are
pinned. Missing, unknown, fallback, symlinked, changed, or hash-mismatched
inputs block before any fit or score. Returned provenance is scalar/immutable;
mutable feature and pandas objects are not authentication evidence.

This branch excludes D1 signed-flow acquisition. It also excludes every
WM/BC/AC execution, result artifact, accuracy statement, and outer run. The
future outer operation is report-only and may be attempted once after all
prerequisites pass; a failed, incomplete, or blocked attempt remains N/A and
must not be rerun or replaced.

