# P1 recovery preregistration protocol

Status: preregistered, no experiment result is attached.  This document amends
the immediately preceding manifest digest
`de422979bf263677d10c689beb77b2c6ec44c26aec458779cce01083d3ceb481` under the
reason `fourth pre-execution independent audit`; `results_observed=false` is
fixed.  The amendment history retains the original pre-execution amendment
digests `9ba18e3e1226cbcbe57e6dfc40050036b1e70b92e58a75e73f8e6ad6c3bc747d`,
`5f8dbd798cf6dc44e15c94b45bc49081c1f7eefea2b89369b682e8e1c7f5d0cc`,
`1ea702af170408f023f7c7b6e83eef2056df9523259b0fd9812ee99946a1c485`, and
`de422979bf263677d10c689beb77b2c6ec44c26aec458779cce01083d3ceb481`.

The machine-readable source of truth is
[`p1_recovery_prereg_manifest.json`](p1_recovery_prereg_manifest.json).  A
future runner must enter through
`unidream.experiments.runtime.validate_p1_v4_runtime_inputs`, which first calls
`load_fixed_manifest` (canonical digest, independently pinned digest, critical
fields, and `results_observed=false`) and only then delegates to the generic
`validate_v4_runtime_inputs` body validator.  The registered base is
`origin/main` at `881e5e08e9b413b51b0a2faf5c49592ce13329d1`; the manifest
digest `d1854827bd4aa204cc2b5cde375edf62583bf0d164b39e8ac25a6c10ad7dc0c4` is
pinned independently in
`unidream/experiments/p1_recovery_prereg.py`.

This wave is a recovery/implementation test, not investment evidence.  S0–S2
are synthetic; S3 is a semi-synthetic injection into a fixed v4 BTC parent.
No result, outer-test metric, or apparent winner may alter this protocol.

## Common fixed contract

- Symbol/frequency: `BTCUSDT`, `15m`, UTC; returns are additive log returns
  `log(close[t] / close[t-1])`.
- The canonical v4 feature body remains the 17 columns listed in the
  manifest.  Availability is a separate strict-boolean sidecar containing
  `spot_bar_observed`, `funding_rate_available`, and
  `mark_close_available`.
- A decision origin is an output-coordinate row `t`; it requires `t >= 63`
  and the current-inclusive context `[t-63,t]` (64 rows) to have consecutive
  15m timestamps, finite canonical 17 features, and all three strict sidecar
  flags true.  This indexing is applied after the raw burn-in slice and only
  `X[t]` is passed to the model.  A target/outcome bar requires only
  `spot_bar_observed=true`, a finite return, and contiguous 15m adjacency;
  future funding/mark flags do not invalidate a return label.  A
  false/missing/non-contiguous required row invalidates the complete context
  or target window.  Timestamp rows remain in place; no sorting, compression,
  interpolation, forward fill, or missing-to-zero conversion is allowed.
- The model input is exactly the current-row canonical 17-feature vector
  `X[t]`; the 64-bar context is an eligibility requirement only.  No context
  flattening, lagged feature, or additional rolling feature is constructed.
- Binary labels are `label[t,h] = 1 iff y[t,h] > 0`; exact zero is class 0.
  The fixed probability clip is `eps=1e-6`: zero-return emits 0.5,
  persistence emits `1-eps` only when `return[t] > 0` and `eps` otherwise,
  and Logistic class-1 probabilities are clipped to `[eps,1-eps]`.  A
  one-class Logistic training prefix is N/A for that origin/horizon and is
  never oversampled, repaired, or promoted.
- For each learned Ridge/Logistic origin and horizon, the train mask is
  `context_eligible AND target_complete[h] AND target_end <= origin-purge_bars
  AND row < origin`.  Fit `StandardScaler(with_mean=True,with_std=True)` on
  those rows only (`ddof=0`, zero-variance scale 1), then transform evaluation
  rows only; targets are never scaled and fixed baselines use no scaler.
  `min_history_rows=16384` counts eligible rows satisfying this exact train
  mask, not raw prefix rows; a model/horizon/origin with fewer rows is N/A and
  cannot promote.
- For every `h` in `{1,4,8,16}`, the target is exactly
  `y[t,h] = sum(return[t+1:t+h+1])` and its right-exclusive availability marker
  is `target_end[t,h] = t+h+1`.  `target_complete[t,h]` requires decision row
  `t`, target rows `[t+1,...,t+h]`, and exactly the `h` contiguous 15m edges
  `t→t+1` through `t+h-1→t+h`; returns and Spot masks are required only on
  `t+1..t+h`.  The edge `t+h→t+h+1` is not required.  Funding/mark
  availability is not a target-label requirement.
- OOF origins are chronological and processed in exactly eight fixed batches:
  `origin[k] = 20000 + 10000*k` for `k=0..7`, with origins
  `[20000,30000,40000,50000,60000,70000,80000,90000]`.  The first seven
  origins are OOF-development batches predicting `[20000,90000)`; the
  origin-90,000 batch is the sole validation operation predicting
  `[90000,100000)`.  Each batch uses one fit at its origin and an expanding
  eligible prefix with no cap.  The minimum history is 16,384 rows; the purge
  is 16 bars; labels satisfy `target_end <= origin - 16` and label row `<
  origin`.  Later origins and labels never enter an earlier batch.  Scaling is
  fit on eligible rows `u < origin` only.  Early rows remain false/NaN.  The
  OOF `target_mask_rule` references the canonical
  `common.availability.target_window_rule` exactly: for every horizon `h`, all
  `h` edges `t->t+1` through `t+h-1->t+h` are required, while
  `t+h->t+h+1` is not.
  Synthetic fit/OOF-development/validation/outer ranges are `[0,20000)`,
  `[20000,90000)`, `[90000,100000)`, and `[100000,120000)`; these ranges are
  disjoint.  OOF-development is diagnostic-only.  All 16 primary comparisons
  use only the fixed validation support `[90000,100000)` after their declared
  masks; no OOF-development row enters a primary inferential gate.
  After all thresholds and manifest fields are fixed, synthetic outer reporting
  performs exactly one fit at origin 100,000 on the admissible prefix
  `[0,100000)` (with the same purge and target-end rules) and exactly one
  prediction pass over `[100000,120000)`.  There is no refit at 110,000.  The
  outer pass is report-only and cannot tune, select, revise a threshold, or
  enter a primary comparison.
  Every numeric `split_range`, `support_range`, `fit_prefix_range`, and
  `prediction_range` is zero-based `[start,end)` with the end excluded; the
  origin row is never included in its fit prefix.
- For every diagnostic-development, primary-validation, and outer-report
  support, potential origins require the 64-bar history and the requested
  target/fill/outcome tail to end before that split's right-exclusive end.
  Cross-boundary target or action tails are excluded rather than borrowed from
  another split.  Reset each independent split to `p_start=1.0`, countdown 0,
  and position 1.0; only policy inventory carries across non-overlapping
  batches within a split, and model/seed/cost/injection-control arms have
  separate policy state.
- The decision contract is four-bar block commitment only: decision `t`, full
  fill `t+1`, fixed position returns `t+1..t+4`, then next decision at `t+4`
  and next fill at `t+5`.  Positions are clipped to `[0.50,1.00]` from
  `previous_position + {-0.08,-0.04,0,0.04,0.08}`.  `p_start=1.0` and
  countdown zero are fixed.  h1/h8/h16 are forecast diagnostics only; Q and
  backtest action utility use h4 only.
- All forecast action choices call the canonical
  `unidream.eval.action_execution.select_block_decisions`.  Its tie key is
  `max(value, -abs(delta), -delta)`: highest value, then smallest absolute
  delta, then more-negative delta.  Feasible absolute candidates use
  clip-then-round-to-12-decimals followed by `np.unique`; no alternate grid or
  hidden action deduplication is allowed.
- Cost-off is zero spread, slippage, fee.  Cost-on is full spread 3 bps
  (1.5 bps half-spread), 1 bp one-way slippage, and 0.0003 one-way fee,
  giving the fixed transition approximation
  `0.00055 * abs(action - previous_position)`.  Accounting is
  `net_log = allocation * bar_log_return - transition_cost`.  Both modes are
  run; neither may be changed after output.
- Cost-on uses the canonical action contract JSON.  Cost-off uses the separate
  derived JSON with identical geometry, timing, masks, and inventory semantics
  and only `spread_bps`, `slippage_bps`, `fee_rate`, and
  `transition_cost_rate` replaced by zero.  Each mode has its own pinned
  canonical SHA-256, and every optimizer/teacher/student-replay/U0/Q/Backtest
  artifact must echo and verify the mode-specific contract hash.
- The canonical action contract is the pinned
  `docs/experiments/action_execution_contract.json`.  It requires strict
  availability masks, an all-or-none fill, `fill_only` boundary charging,
  `hold_commitment` execution skips, `hold_and_score_commitment` for a feature
  becoming unavailable after a valid fill, and `exclude_block` for an
  unavailable/non-contiguous outcome window.  These policies are distinct:
  an ineligible decision origin is retained with a false mask and is not
  scored for forecasts or action agreement, whereas a feature gap during an
  already valid four-bar commitment holds and is included in PnL scoring when
  the four-bar outcome is complete.  The forecast-origin mask is
  `origin_eligible AND finite_forecast`; the action-agreement mask additionally
  requires complete fill and outcome; the PnL-scored mask is
  `valid_fill_or_active_hold_commitment AND four_bar_outcome_complete`.
- The primary utility benchmark is an independent `benchmark_hold_path`: reset
  to `p_start=1`, countdown 0, position 1, hold `delta=0` throughout, use the
  same score mask, and charge cost 0.  For regret/opportunity only, the
  candidate policy's own carried `p_{t-1}` is used with a separate
  `same_state_local_hold` at `delta=0`; this is never substituted for the
  independent benchmark.  S3 timing DID subtracts the independent benchmark
  in both injected and control arms.

For action agreement and regret, the clairvoyant comparison is recomputed at
each origin from the forecast policy's actual current inventory `p_{t-1}` and
the same feasible action set.  Agreement compares the forecast-optimal next
position with the realized four-bar one-block optimum from that same `p_{t-1}`;
regret is realized best utility minus chosen utility from that same state.
Only the chosen forecast policy action advances its inventory into the next
block.  Hindsight/U0-path inventory and the U0 global dynamic-programming path
are never fed into row scoring or policy state; U0 is a separate report-only
upper bound.

Action bootstrap never replays a policy over resampled rows.  First produce the
full original validation action primitive grid once in chronological order:
split-local scheduled starts are `0,4,...` from the canonical
`complete_decision_starts`, and global decision index is
`support_start + local decision index`.  There is one record for every
structurally complete, scheduled, non-overlapping h4 block, including
false-mask/N/A records for forecast or outcome gaps.  The exact record-field
list and nested schema are identical and include `common_mask`:
`primitive_index`, `decision_index`, `fill_index`, `end_index`,
`previous_position`, `selected_delta`, `selected_position`,
`candidate_utility`, `benchmark_hold_utility`, `same_state_local_hold_utility`,
`clairvoyant_utility`, `regret`, `opportunity`, `agreement`, `turnover`,
`active_indicator`, `origin_eligible_mask`, `forecast_finite_mask`,
`fill_complete_mask`, `outcome_complete_mask`, `scored_action_mask`,
`common_mask`, `scenario_id`, `seed`, `split_id`, `support_id`, `model_id`,
`cost_mode`, and `cost_contract_hash`.  Index fields and `seed` are strict
little-endian int64; value fields are strict little-endian IEEE-754 float64;
mask fields are strict bool; arm strings are UTF-8.  Record fields are encoded
in this order and rows remain in original `primitive_index` order, with every
full-grid gap row retained physically.  C-order shape `(record_count,)`, field
name/dtype length-prefixes, little-endian ndim/shape uint64 framing, data
byte-length framing, one-byte bool encoding (`0x00`/`0x01`),
and UTF-8 length framing are fixed.  Finite float bits are preserved, every
NaN is encoded as `0x7ff8000000000000`, and infinities are rejected.  JSON
framing uses UTF-8, `ensure_ascii=false`, `sort_keys=true`, compact separators,
and `allow_nan=false`.

`selected_delta` is the canonical chosen delta, `selected_position` is the
clipped/deduplicated chosen position, `previous_position` is the policy state
before the block, `turnover=abs(selected_position-previous_position)`, and
`active_indicator=1` iff `turnover>0`.

`primitive_index` is the zero-based full-grid row ordinal; `decision_index` is
the output-coordinate `t`; `fill_index=decision_index+1`; and
`end_index=decision_index+4` is the inclusive final return bar.  Scheduled
decision indices advance by four.

Three hashes have disjoint, exact scopes.  `action_primitive_schema_sha256`
is SHA-256 over the external canonical JSON schema
`docs/experiments/action_primitive_schema.json`, whose pinned digest is
`d0520b3dbc3c444e2efe5a55e175e96b662f97fb404d901ea51e1c32e5bb9955` outside
the payload.  `action_primitive_content_sha256` is SHA-256 over the
canonical framed bytes for every listed field and every original full-grid row;
hash declarations are excluded.  `action_primitive_payload_sha256` is
SHA-256 over payload magic, a canonical JSON header containing
`record_count`, `record_fields`, `schema_sha256`, and `content_sha256`, and the
framed content bytes; the payload hash itself is excluded.  Moving-block
replicates resample record indices and recompute the declared means, sums,
ratios, or DiD from those stored values; they do not carry inventory across
bootstrap boundaries, duplicate records, or replay a nonchronological
sequence.  The payload, schema, and content SHA-256 values are echoed in every
result artifact.

The action primitive producer and P1-specific moving-block implementation are
not implemented on this preregistration branch.  A runner must stop with a
blocked/N/A status until both are separately implemented and validated; the
existing generic MBB path is forbidden.

## Fixed models and metrics

The only model candidates are zero/persistence, Ridge, and logistic regression.
Ridge is `alpha=1.0`, intercept on, `solver=lsqr`, `tol=1e-12`,
`max_iter=10000`.  Logistic is L2, `C=1.0`, `solver=lbfgs`, `tol=1e-10`,
`max_iter=1000`, `class_weight=null`, `random_state=0`.  A one-class logistic
prefix is N/A; it is not oversampled or otherwise repaired.

Continuous forecasts report MSE and MAE.  The S2 primary continuous metric is
normalized MSE skill `1 - MSE(model)/MSE(zero_return)` on the same complete
target rows; raw MSE is report-only for monotonicity because target variance
changes with beta.  Binary forecasts report log loss and Brier score.  The h4
 action mapper reports mean net-log utility,
 paired utility delta against the independent benchmark hold, regret against the same-contract
clairvoyant action, feasible-action agreement, active rate, turnover, and all
eligibility denominators.  Sign accuracy alone is diagnostic and cannot
promote.

The action utility is O1 point forecast plus transition cost.  S2 primary
regret is normalized per seed and per bootstrap replicate as
`sum(regret) / sum(clairvoyant_net_utility - same_state_local_hold_net_utility)`, requiring a
strictly positive aggregate opportunity denominator; no per-row tiny-denominator
division is used.  S3 primary forecast recovery is
`skill(injected Ridge vs injected zero) - skill(control Ridge vs control zero)`
where `skill(A vs B) = 1 - MSE(A)/MSE(B)`, not a raw injected-vs-control MSE
difference.  No path risk is invented from a point forecast.  If O2/O3 is
later enabled, it must use a
calibrated joint return-path scenario with the same four-bar trajectory; no
independent marginal synthesis is allowed.

## Scenarios

### S0 — zero signal

Generate raw arrays of length 120,512, then discard raw rows `[0,512)` so the
output has 120,000 rows and 17 observed features.  Draw from
`default_rng(seed+100)` in this exact order: one scalar `z0`, `xi` with shape
`(120511,)`, C-order noise features with shape `(120512,16)`, then `epsilon`
with shape `(120512,)`.  Every `z0`, `xi` entry, `noise_features` entry, and
`epsilon` entry is an independent float64 `N(0,1)` draw from
`np.random.Generator.standard_normal`; the four draw groups are mutually
independent and entries within each group are iid.  Set `z_raw[0]=z0` and
`z_raw[k]=0.95*z_raw[k-1]+sqrt(1-0.95^2)*xi[k-1]`; set
`r_raw[0]=0.001*epsilon[0]` (beta-independent sentinel) and
`r_raw[k+1]=beta*z_raw[k]+0.001*epsilon[k+1]`.  The same base arrays are reused
for every beta; only beta changes.  Ten fixed seeds are used:
`20260830,20260831,20260832,20260833,20260834,20260835,20260836,20260837,20260838,20260839`.

S0 is a null control.  Only action-capable Ridge and persistence under
cost-on are in its safety scope; zero_return is the hold baseline, Logistic
action is N/A, and cost-off is diagnostic-only.  No model/cost combination may be promoted for a
  positive utility or high-agreement result.  Its safety gate is the
  direction-aware candidate-minus-independent-benchmark-hold lower bound `<= 0` (not an upper-bound
claim that the true edge is negative).  An apparent promotion pass is a
contract or implementation failure and must be recorded, not tuned away.

### S1 — known high-SNR DGP

Use the same fixed state and return process with `beta=0.004`, giving the
registered SNR of 4.0 because `std(z)=1` and return noise standard deviation is
0.001.  The signal is observable as `x[t,0]` and all target values are created
after the feature at t.  Ten fixed seeds and the common splits are used.

High-SNR recovery is inferred only on the fixed synthetic validation support
`[90000,100000)`, using the one fit at origin 90,000.  It requires per-seed
feasible-action agreement of at least 90%, pooled Wilson lower bound at least
90%, and a Holm-adjusted one-sided bootstrap p-value `<=0.05` from the
conservative `raw_p` for validation-minus-independent-benchmark-hold h4 net utility under cost-on,
with a favorable point delta `>0`.  In addition, every one of the ten fixed
seed-level validation utility deltas must be strictly positive and non-N/A,
  and, for every seed on the identical scored mask, the mean realized
  same-state clairvoyant net utility/value must be strictly greater than the
  Ridge mean realized net utility/value.  A mask mismatch, N/A value, or
  non-strict comparison fails.
OOF-development scores are diagnostic-only;
the single synthetic outer pass is report-only.  The realized-path clairvoyant
is report-only and must remain above the validation decision on the same
support.

### S2 — monotonic SNR

Use exactly the same DGP, 120,000 rows, seeds, latent state, 16 noise
features, return-noise draws, two-bar availability gaps, models, and splits as
S1; only beta changes.  The base arrays are drawn once per seed from
`default_rng(seed + 100)` and reused for high, medium, and low, so the
monotonic comparisons are paired rather than three independent simulations:

| level | beta | SNR |
| --- | ---: | ---: |
| high | 0.004 | 4.0 |
| medium | 0.001 | 1.0 |
| low | 0.00025 | 0.25 |

The primary monotonic comparisons use only the identical seed/support rows in
the fixed synthetic validation interval `[90000,100000)`; OOF-development is
diagnostic-only and the outer pass is report-only.  On those comparisons, the
fixed order must be high >= medium
>= low for normalized MSE skill, high <= medium <= low for log loss and
normalized action regret, and high >= medium >= low for net utility and action
agreement.  Raw MSE and raw regret are report-only because their scales change
with beta.  Timing/net-utility monotonicity uses the per-level
`Ridge-minus-independent-benchmark-hold` delta, not absolute utility.  Medians over the ten seeds and
paired block intervals are reported.
Any order violation fails the gate; no SNR level may be removed or retuned.

### S3 — semi-synthetic BTC with an observable injection

Use the immutable v4 BTC parent described below.  The model input remains the
canonical 17-column body; the named signal source is the existing feature
`close_ret`, and no generated latent feature is added or passed to the model.
At decision time t, define the only injected
signal by a prefix-only fit:

```text
z[t] = (close_ret[t] - np.mean(close_ret[u] for context_eligible[u] and u<t))
       / max(np.std(close_ret[u] for context_eligible[u] and u<t, ddof=0), 1e-12)
```

At least 256 context-eligible prefix rows are required.  The future target is
never used for this fit or scaling.  Inject only when `context_eligible[t]`,
`t+1` is inside the body, `t+1` is contiguous with `t`, and
`spot_bar_observed[t+1]` is true:
`returns_injected[t+1] = returns_v4[t+1] + 0.0005*z[t]`; the paired zero
injection control is exactly `returns_control[t+1] = returns_v4[t+1]`.
The original feature body is not recomputed.  Invalid or gapped origins remain
false and are not repaired.  S3 uses one
deterministic run with seed `20260830`; uncertainty is supplied by the fixed
paired moving-block bootstrap, not by pretending identical deterministic runs
are independent seed replications.  The v4 parent rows and sidecar are
identical across the injection/control pair, but each injected/control
candidate has its own reset-and-carry policy inventory; their inventories are
never shared or forced equal.  Only the common timestamp score mask is shared.
The exact v4 raw-body boundary
indices are 52,491 for `2020-01-01T00:00:00Z`, 52,492 for the first fully
available `2020-01-01T00:15:00Z`, 104,528 for `2022-01-01T00:00:00Z`, 139,568
for `2023-01-01T00:00:00Z`, and 173,111 for the exclusive
`2024-01-01T00:00:00Z` endpoint.  OOF-development origins resolve after the
timestamp join at raw indices `[72492,82492,92492,102492]`; their spans are
`[72492,82492)`, `[82492,92492)`, `[92492,102492)`, and the truncated final
span `[102492,104528)`.  The primary inferential validation operation is one
fit at raw 104,528 using only its admissible pre-validation prefix, followed by
one prediction pass over `[104528,139568)`.  Validation rows are never refit.
After all thresholds and manifest fields are fixed, the outer operation is one
report-only fit at raw 139,568 using its admissible prefix and one pass over
`[139568,173111)`; there is no outer refit.  The eighth synthetic-relative
origin would resolve to raw 142,492, which lies inside the sealed outer range
and is therefore excluded.  Raw body thirds are not valid split boundaries.

S3 promotes only if both the Holm-adjusted one-sided p-value from conservative
`raw_p` and favorable point delta pass for the injected-control h4 timing
net-utility difference-in-differences and the h4 MSE-skill
difference-in-differences on the fixed S3 validation support
`[104528,139568)`.  Compute candidate-minus-independent-benchmark utility in
each injected/control arm with its own reset-and-carry policy inventory; do not
share or force candidate inventories equal.  Only the common timestamp mask is
paired.  OOF-development is diagnostic-only and the outer pass is
report-only, never a gate or selection support.  Future perturbation after an origin's
`target_end` must not change any earlier prediction, mask, or fitted-prefix
digest.

## Availability and computation

Synthetic masks are deterministic two-bar blocks.  For each source, use the
exact call
`rng=np.random.default_rng(seed+50000+source_offset);`
`relative=rng.choice(119998-512,size=40,replace=False,shuffle=True);`
`starts=np.asarray(relative,dtype=np.int64)+512`.  These are output-coordinate
indices after the raw burn-in slice, not pre-slice raw indices; retain the
returned choice order in the artifact and never sort it.  Each source's false
mask is the union of half-open output intervals `[start,start+2)`; starts are
unique, but adjacent starts may overlap bars and the union is applied.  Raw
arrays have length 120,512 and output rows are raw `[512,120512)`; the same
starts are reused by S0, S1, and all S2 levels for a seed.  S3 uses the v4
sidecar directly.  A context or target window crossing a gap is N/A; rows are
never compacted to conceal it.  Context windows require all three source
masks, while target labels require only the Spot observation mask plus finite,
contiguous returns.  A conservative
mask-only reference count with 40 two-bar gaps per source, context 64, and
maximum target horizon 16 gives a minimum eligible fraction of 0.9245 across
the ten fixed synthetic seeds; this derivation inspects no model output or
target outcome and is frozen in the manifest.

The synthetic run contains five scenario instances (S0, S1, and S2 high,
medium, low), ten seeds, four fixed model IDs (zero return, persistence, Ridge,
logistic), two cost modes, four forecast horizons, and one h4 action mapper.
Exactly seven chronological OOF-development fits (origins 20,000..80,000) and
one validation fit (origin 90,000) are specified per synthetic
scenario/model/seed.  Primary inferential/gate statistics use only the
validation interval `[90000,100000)`; OOF-development is diagnostic-only.
After the protocol is frozen, the synthetic outer report is one fit at origin
100,000 and one prediction pass over `[100000,120000)`, with no refit.  S3 uses
one deterministic injected/control pair over the fixed body: its four
development spans end at raw 104,528, its validation fit at raw 104,528 is the
sole primary operation, and its raw [139568,173111) outer fit/pass is
report-only.  No window bound, seed count, or scenario arm may be selected from
results.

## Fixed gates and stopping rules

Coverage is machine-reported before any score is interpreted.  Synthetic
eligible-origin fraction must be at least 0.90; S3 must be at least 0.50;
label-complete fraction at least 0.90; finite OOF prediction fraction at least
0.95; and scored-action fraction at least 0.80.  Any enabled neural future
head would additionally require target/gradient coverage 1.0 per head; a
zero-valid-target/gradient row is contract failure/N/A, never accuracy.
Potential origins are all rows inside a split's prediction support with a
64-bar past and a complete requested target tail before the split end.  For
each horizon, `context_fraction = context_complete/potential_origins`,
`label_fraction = target_complete/potential_origins`,
`eligible_fraction = (context_complete AND target_complete)/potential_origins`,
and `finite_prediction_fraction = finite_prediction/(context_complete AND
target_complete)`.  `scored_action_fraction` is the number of scheduled
complete canonical four-bar blocks with eligible origin, finite h4 forecast,
and complete realized h4 outcome divided by all scheduled complete canonical
blocks inside that split.  These full-grid masks are reported for every
required horizon/model/seed/cost mode and injected/control arm before fixed
thresholds are applied; an undefined/N/A denominator blocks promotion.

Primary bootstrap inference uses 2,000 non-circular moving-block replicates at
`L` in `{8,16,32}`.  The forecast primitive grid is the complete validation
split time-series row grid in original order; missing/N/A rows remain in place
and are represented by masks.  The action primitive grid is one record per
canonical non-overlapping complete four-bar scheduled decision block.  `L` is
  measured in primitive records, not compacted valid rows.  Require `n >= L`; for
  each replicate draw exactly
  `starts=rng.integers(low=0,high=n-L+1,size=ceil(n/L),endpoint=False,dtype=np.int64)`.
  Materialize `indices = starts[:,None] + np.arange(L,dtype=np.int64)`, flatten
  in C order, and take the first `n` indices.  This is non-circular; gaps are
  never crossed by physical compression.  The seed is
  `20260830 + 100000*unit_code + 1000*L + seed_ordinal`, with unit codes 1, 2,
  3, and 4 for synthetic forecast/action and S3 forecast/action respectively;
  synthetic seed ordinal is 0..9 and S3 ordinal is 0.  Within a fixed
  unit/support/seed/L, create `default_rng(derived_seed)` exactly once, draw
  all starts for replicates `b=0..1999` in ascending order, and reuse the same
  sampled indices for every arm and comparison.  Do not reinitialize the RNG
  per replicate, arm, or comparison.  Percentiles use
  `np.quantile(values, q, method='linear')`.
Each replicate resamples primitive arrays and recomputes the metric before
forming its paired contrast: MSE is the mean squared-error difference, skill is
`1-sum(SE_model)/sum(SE_zero)`, log loss/agreement are means, utility is the
mean candidate-minus-independent-benchmark-hold value, S2 forms the directed
level contrast after recomputing each level, normalized regret is
`sum(regret)/sum(opportunity)` with a positive aggregate opportunity, and S3
recomputes the injected/control skill and utility DID.  The shared common
eligible mask stays a mask; sampled N/A records are not dropped or compacted.
  The entire comparison is N/A only when `n<L`, valid primitive count is zero,
  an arm's required metric is unavailable, or a denominator is zero/nonpositive.
  If any required denominator is zero or nonpositive in a replicate or required
  arm, the entire comparison is N/A/blocked; it is never repaired, omitted, or
  removed by resampling.
For each `L`, the two-sided percentile interval `[quantile_0.025,
quantile_0.975]` is diagnostic.  The one-sided raw p-value used for inference is
`raw_p=max(p_8,p_16,p_32)`, a conservative intersection-union rule; no gate is
formed by mixing unadjusted confidence limits.  Every non-S0 primary gate uses
this raw p after the fixed Holm–Bonferroni family correction (`alpha=0.05`,
family size 16).
The immutable reporting-arm ledger `p1_recovery_trial_registry.jsonl`
enumerates all 56 execution arms (seven scenario arms x four model IDs x two
cost modes).  The separate
`p1_recovery_primary_comparisons.jsonl` enumerates the 16 executable paired
comparisons; only its `primary=true` records define the multiplicity family.
Every primary record contains its fixed `support_id`, right-exclusive
`support_range`, the exact `support_range_semantics` value `zero-based
[start,end) right-exclusive; end excluded`, and `support_role`; action-capable
records additionally carry the exact stored-record/no-replay
`action_bootstrap_replay_policy`.  All 16 records point to the validation
support for their scenario (synthetic `[90000,100000)`, S3 raw
`[104528,139568)`).  OOF-development rows are diagnostic-only, and outer rows
are report-only and never enter Holm or any gate.
Both exact hashes are pinned in the manifest.  An execution arm is not counted
as a statistical comparison unless it has an explicit candidate, baseline,
metric, horizon, cost mode, direction, and gate in that comparison registry.
For each primary contrast, orient the paired bootstrap means so that favorable
evidence is positive.  With `B=2000` at each `L`, its one-sided bootstrap
p-value is `p_L=(1 + count_b(T_b,L <= 0))/(B+1)` for a positive contrast and
`p_L=(1 + count_b(T_b,L >= 0))/(B+1)` for a negative contrast; the `+1`
correction is mandatory.  The comparison's `raw_p=max(p_8,p_16,p_32)` is the
sole p-value entering Holm.  For high-vs-medium or medium-vs-low `<=` metrics,
the orientation is respectively medium-minus-high or low-minus-medium.  Sort
the 16 raw p-values ascending, break ties by lexicographic `comparison_id`, and
apply Holm step-down: rank `r` is rejected only if every rank `j<=r` satisfies
`p_(j) <= 0.05/(16-j+1)`; stop at the first failure and use
`adjusted_p_(r)=min(1,max_{j<=r}((16-j+1)*p_(j)))`.  Positive/negative gates
require the adjusted p-value and favorable point direction.  S0 is a safety
exception: for every `L` its positive-direction lower percentile is computed
at the Holm-rank-adjusted `alpha_r=0.05/(16-r+1)` and must be `<=0`, and its
positive-edge Holm rejection must be false.  It is never a promotion claim;
its p-value is not interpreted as proof of a negative edge.
Agreement and coverage use the 95% Wilson score interval with
`z=1.959963984540054`; normal approximations are not substituted.  For
successes `x` and denominator `n`, `p=x/n`, `den=1+z^2/n`,
`center=(p+z^2/(2n))/den`, and
`half=z*sqrt(p*(1-p)/n+z^2/(4n^2))/den`; the interval is
`[center-half,center+half]`.  A 90% agreement gate requires every fixed-seed
point estimate `x/n >= 0.90` and the pooled Wilson lower bound `>= 0.90`.
For S2, display point estimates are the median of the ten per-seed metrics;
each adjacent point contrast must satisfy its registry direction after the
`1e-12` tie tolerance, and its Holm-adjusted bootstrap p-value must also pass.
The bootstrap statistic equal-weights the ten seed-level primitive metrics at
`1/10`; it is not the row-pooled mean.  Monotonicity passes only when both
adjacent inequalities high-vs-medium and medium-vs-low meet both conditions;
no level may be removed or retuned.

Missing rows, insufficient class support, undefined metrics, missing masks,
missing provenance, altered manifest fields, and failed coverage are N/A or
blocked.  The runner must stop before fitting or promotion, and it must not
change a threshold, seed, feature, horizon, split, cost, or outer-row status
after seeing any output.  The existing generic
`unidream.experiments.train_app.run_training_app` is not a P1 research runner;
it is explicitly unsuitable because it owns the generic WM→BC→AC pipeline.
The P1 runner is blocked unless it uses the authenticated v4 wrapper and the
separately implemented P1 action primitive/MBB components.

## Immutable v4 input provenance

S3 references the committed metadata file
`docs/data_quality_v4_rebuild_2018_2024_metadata.json` with SHA-256
`2c9db28deebe7e6b08f4ffedf65c3cdb51a78cfd7ee7d6580f76a62cc424bdcb`.
The parent is cache tag
`BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official`, schema digest
`1c1c41a9aca3e8af22b357a8483ea6419745ee4b24c10c09c47289df3744c616`, and
source provenance digest
`aa320222dca0a46b2a0730f17bb1665f31a70074aa3bafcc6bff58ca21618fad`.
Its immutable content digests are:

| input | SHA-256 | rows |
| --- | --- | ---: |
| features | `8a7aad5809c7a21e614da7d836629309cda9c2de74553bf1fbc6934f7b07f5e2` | 173111 |
| returns | `c33a00cac4cf169f01e3ba5823a3f6d9bae17da5add5f8d5a3538d4142a0fabb` | 173111 |
| availability sidecar | `630de125ae9bc04cd0376404c7cff07f8e7d06c3bec2eece1b546e05959e292f` | 210336 |

The v4 body has 119,849 rows with all three availability flags true and
209,805 observed Spot rows.  These counts and hashes are input provenance,
not experiment outcomes; a mismatch blocks S3.

The S3 runner must call `unidream.data.cache_v4.load_cache_v4` with all four
explicit artifact paths and the expected cache tag from the manifest:

```text
load_cache_v4(
    cache_tag="BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official",
    feature_path="checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_features.parquet",
    returns_path="checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_returns.parquet",
    availability_path="checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_availability.parquet",
    metadata_path="docs/data_quality_v4_rebuild_2018_2024_metadata.json",
)
```

Before fitting or scoring, the production runner must call the authenticated
`unidream.experiments.runtime.validate_p1_v4_runtime_inputs` entrypoint.  It
must first call `load_fixed_manifest` and reject any forged manifest digest,
`results_observed` flag, critical field, or frozen v4 digest.  Only the frozen
manifest may then be passed to the generic body validator
`unidream.experiments.runtime.validate_v4_runtime_inputs` with all four
explicit paths.  That body validator must call the explicit `load_cache_v4(...)`,
then verify the loaded body against the frozen metadata's
content digests, schema digest, cache tag, feature/sidecar row counts, and
canonical columns.  Missing or unknown provenance, missing body files, or any
mismatch blocks S3.  The validator returns the mandatory disposition fields
`status`, `reason`, `body_match`, and `source_provenance_match`.  If only the
known cache-local source-provenance digest differs while all body, schema,
cache-tag, and row-count checks match, the run may proceed but must echo the
`source_provenance_only_difference` disposition; an unknown source digest
blocks it.  A cache-local file is audit-only and may be absent, in which case
the disposition is `absent` and the frozen metadata remains authoritative.
The authenticated wrapper snapshots the manifest, all four explicit body
paths, and any existing cache-local path with `lstat`/`stat` identity
(`dev`, `ino`, `size`, `mtime_ns`, regular-file mode) before and after the
read/load/hash sequence; symlinks, non-regular files, and any identity change
block the run.  Absolute paths are allowed when they resolve to regular files;
the generic body validator retains its fixture-only API without this wrapper
authentication.

The `cache_dir`/`cache_tag`-only default lookup is forbidden for S3.  Any
`path_overrides` supplied to the validator must provide the complete four-path
set (`feature_path`, `returns_path`, `availability_path`, `metadata_path`);
partial overrides are rejected.  The
repository-frozen metadata is authoritative and its SHA-256 is pinned above;
the cache-local metadata at
`checkpoints/data_cache/BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official_metadata.json`
is audit-only and must never replace the frozen `metadata_path`.  If present,
the runner must echo its file SHA, source-provenance digest, schema/content
digests, and row counts separately from the frozen values.  No local-versus-
frozen difference may be hidden: content, schema, cache-tag, or row-count
mismatch blocks S3, while a source-provenance revision difference is retained
for explicit disposition before promotion.  The raw v4 body files are not
committed to this preregistration branch; their caller-supplied explicit paths
and all echoed digests are mandatory runtime inputs.

The cache-local snapshot available during preregistration review had metadata
SHA-256 `bade1775884cd22c8675af225b429976aa6b2c60b859b4a591c76f8a87d17450`
and source-provenance digest
`1e78ccf3162567e799b05a1c25dbe12a1c4c37e8e5a2abf2f9b95a70c380e2db`.  The
repo-frozen values are the SHA-256 and source-provenance digest given above;
the known snapshot's schema/content digests and row counts match.  This
difference is recorded as provenance evidence only, not as permission to
select the local metadata or to claim a new dataset revision.

The canonical action execution
contract is `action_execution_contract.json` (the path and canonical-content
SHA-256 are pinned in the manifest) and is shared by optimizer, teacher,
student replay, U0, Q, and Backtest.  Action results must also echo the
canonical primitive payload, schema, and content SHA-256 fields named in the
manifest; a missing or mismatched hash is a fail-closed contract error.
Logistic regression is a binary
proper-score diagnostic only; Ridge is the sole learned h4 action mapper.
Zero-return and persistence-last-observed are fixed comparators: persistence
predicts `h * return[t]` continuously and clips its binary probability to
`1-1e-6` only for a strictly positive last return and `1e-6` otherwise (exact
zero is class 0).  Logistic class-1 probabilities are clipped to
`[1e-6,1-1e-6]`; Logistic action
utility is N/A, so no post hoc probability-to-return conversion is permitted.
