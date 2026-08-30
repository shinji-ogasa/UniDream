# P1 recovery preregistration protocol

Status: preregistered, no experiment result is attached.

The machine-readable source of truth is
[`p1_recovery_prereg_manifest.json`](p1_recovery_prereg_manifest.json).  A
future runner must load that file through the fail-closed validator before it
creates data, fits a model, or opens an outer row.  The registered base is
`origin/main` at `881e5e08e9b413b51b0a2faf5c49592ce13329d1`; the manifest
digest is pinned independently in
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
- A decision origin requires all three strict sidecar flags and finite model
  features for its 64-bar forecast context.  A target/outcome bar requires
  only `spot_bar_observed=true`, a finite return, and contiguous 15m
  adjacency; future funding/mark flags do not invalidate a return label.  A
  false/missing/non-contiguous required row invalidates the complete context
  or target window.  Timestamp rows remain in place; no sorting, compression,
  interpolation, forward fill, or missing-to-zero conversion is allowed.
- For every `h` in `{1,4,8,16}`, the target is exactly
  `y[t,h] = sum(return[t+1:t+h+1])` and its right-exclusive availability marker
  is `target_end[t,h] = t+h+1`.  The future bars must be contiguous, have
  finite returns, and have `spot_bar_observed=true`; funding/mark
  availability is not a target-label requirement.
- OOF origins are chronological and processed in exactly eight fixed batches:
  `origin[k] = 20000 + 10000*k` for `k=0..7`, with origins
  `[20000,30000,40000,50000,60000,70000,80000,90000]`.  Each batch predicts
  the next fixed 10,000-row interval using one fit at its origin and an
  expanding eligible prefix with no cap.  The minimum history is 16,384 rows;
  the purge is 16 bars; labels satisfy `target_end <= origin - 16` and label
  row `< origin`.  Later origins and labels never enter an earlier batch.
  Scaling is fit on eligible rows `u < origin` only.  Early rows remain
  false/NaN.  Synthetic fit/OOF-development/validation/outer ranges are
  `[0,20000)`, `[20000,90000)`, `[90000,100000)`, and `[100000,120000)`;
  these ranges are disjoint, and the origin-90,000 batch is the validation
  batch.
  Every outer-test row is report-only and cannot tune, select, or revise a
  threshold.
- The decision contract is four-bar block commitment only: decision `t`, full
  fill `t+1`, fixed position returns `t+1..t+4`, then next decision at `t+4`
  and next fill at `t+5`.  Positions are clipped to `[0.50,1.00]` from
  `previous_position + {-0.08,-0.04,0,0.04,0.08}`.  `p_start=1.0` and
  countdown zero are fixed.  h1/h8/h16 are forecast diagnostics only; Q and
  backtest action utility use h4 only.
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

For action agreement and regret, the clairvoyant comparison is recomputed at
each origin from the forecast policy's actual current inventory `p_{t-1}` and
the same feasible action set.  Agreement compares the forecast-optimal next
position with the realized four-bar one-block optimum from that same `p_{t-1}`;
regret is realized best utility minus chosen utility from that same state.
Only the chosen forecast policy action advances its inventory into the next
block.  Hindsight/U0-path inventory and the U0 global dynamic-programming path
are never fed into row scoring or policy state; U0 is a separate report-only
upper bound.

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
paired utility delta against hold, regret against the same-contract
clairvoyant action, feasible-action agreement, active rate, turnover, and all
eligibility denominators.  Sign accuracy alone is diagnostic and cannot
promote.

The action utility is O1 point forecast plus transition cost.  S2 primary
regret is normalized per seed and per bootstrap replicate as
`sum(regret) / sum(clairvoyant_net_utility - hold_net_utility)`, requiring a
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

Generate 120,000 rows after a 512-row burn-in with 17 observed features.  The
observed state is an AR(1) feature with `rho=0.95`; the other 16 features are
independent standard normal noise.  The return formula is
`r[t+1] = 0 + 0*z[t] + 0.001*epsilon[t+1]`.  Ten fixed seeds are used:
`20260830,20260831,20260832,20260833,20260834,20260835,20260836,20260837,20260838,20260839`.

S0 is a null control.  No model/cost combination may be promoted for a
positive utility or high-agreement result.  Its safety gate is the
direction-aware candidate-minus-hold lower bound `<= 0` (not an upper-bound
claim that the true edge is negative).  An apparent promotion pass is a
contract or implementation failure and must be recorded, not tuned away.

### S1 — known high-SNR DGP

Use the same fixed state and return process with `beta=0.004`, giving the
registered SNR of 4.0 because `std(z)=1` and return noise standard deviation is
0.001.  The signal is observable as `x[t,0]` and all target values are created
after the feature at t.  Ten fixed seeds and the common splits are used.

High-SNR recovery requires per-seed feasible-action agreement of at least
90%, pooled Wilson lower bound at least 90%, and a Holm-adjusted one-sided
bootstrap p-value `<=0.05` from the conservative `raw_p` for OOF-minus-hold
h4 net utility under cost-on, with a favorable point delta `>0`.  The
realized-path clairvoyant is report-only and must remain above the OOF decision
on the same support.

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

On identical seed/support comparisons, the fixed order must be high >= medium
>= low for normalized MSE skill, high <= medium <= low for log loss and
normalized action regret, and high >= medium >= low for net utility and action
agreement.  Raw MSE and raw regret are report-only because their scales change
with beta.  Timing/net-utility monotonicity uses the per-level
`Ridge-minus-hold` delta, not absolute utility.  Medians over the ten seeds and
paired block intervals are reported.
Any order violation fails the gate; no SNR level may be removed or retuned.

### S3 — semi-synthetic BTC with an observable injection

Use the immutable v4 BTC parent described below.  The model input remains the
canonical 17-column body; the named signal source is the existing feature
`close_ret`, and no generated latent feature is added or passed to the model.
At decision time t, define the only injected
signal by a prefix-only fit:

```text
z[t] = (close_ret[t] - mean(close_ret[u] for eligible u<t))
       / max(std(close_ret[u] for eligible u<t), 1e-12)
```

At least 256 eligible prefix rows are required.  The future target is never
used for this fit or scaling.  The injected parent is
`returns_injected[t+1] = returns_v4[t+1] + 0.0005*z[t]`; the paired zero
injection control is exactly `returns_control[t+1] = returns_v4[t+1]`.
Invalid or gapped origins remain false and are not repaired.  S3 uses one
deterministic run with seed `20260830`; uncertainty is supplied by the fixed
paired moving-block bootstrap, not by pretending identical deterministic runs
are independent seed replications.  The v4 parent rows and sidecar are
identical across the injection/control pair.  The exact v4 raw-body boundary
indices are 52,491 for `2020-01-01T00:00:00Z`, 52,492 for the first fully
available `2020-01-01T00:15:00Z`, 104,528 for `2022-01-01T00:00:00Z`, 139,568
for `2023-01-01T00:00:00Z`, and 173,111 for the exclusive
`2024-01-01T00:00:00Z` endpoint.  Development origins resolve after the
timestamp join at raw indices `[72492,82492,92492,102492,112492,122492,132492]`;
the final development batch is truncated at raw 139,568.  A single fit at raw
139,568 followed by one pass over `[139568,173111)` is the report-only outer
operation.  The eighth synthetic-relative origin would resolve to raw 142,492,
which lies inside the sealed outer range and is therefore excluded.  Raw body
thirds are not valid split boundaries.

S3 promotes only if both the Holm-adjusted one-sided p-value from conservative
`raw_p` and favorable point delta pass for the injected-control h4 timing
net-utility difference-in-differences and the h4 MSE-skill
difference-in-differences.  Future perturbation after an origin's
`target_end` must not change any earlier prediction, mask, or fitted-prefix
digest.

## Availability and computation

Synthetic masks are deterministic two-bar blocks: 40 starts per source are
sampled without replacement from the fixed range after burn-in using
`default_rng(seed + 50000 + source_offset)`, with source offsets 11, 23, and
37.  The same starts are reused by S0, S1, and all S2 levels for a seed.  S3
uses the v4 sidecar directly.  A context or target
window crossing a gap is N/A; rows are never compacted to conceal it.  Context
windows require all three source masks, while target labels require only the
Spot observation mask plus finite, contiguous returns.  A conservative
mask-only reference count with 40 two-bar gaps per source, context 64, and
maximum target horizon 16 gives a minimum eligible fraction of 0.9245 across
the ten fixed synthetic seeds; this derivation inspects no model output or
target outcome and is frozen in the manifest.

The synthetic run contains five scenario instances (S0, S1, and S2 high,
medium, low), ten seeds, four fixed model IDs (zero return, persistence, Ridge,
logistic), two cost modes, four forecast horizons, and one h4 action mapper.
Exactly eight chronological batch fits are specified by the origin schedule per
synthetic scenario/model/seed.  S3 uses one deterministic injected/control pair
over the fixed body and the same h4 contract; its outer fit is a single
report-only operation.  No window bound, seed count, or scenario arm may be
selected from results.

## Fixed gates and stopping rules

Coverage is machine-reported before any score is interpreted.  Synthetic
eligible-origin fraction must be at least 0.90; S3 must be at least 0.50;
label-complete fraction at least 0.90; finite OOF prediction fraction at least
0.95; and scored-action fraction at least 0.80.  Any enabled neural future
head would additionally require target/gradient coverage 1.0 per head; a
zero-valid-target/gradient row is contract failure/N/A, never accuracy.

Utility intervals use 2,000 moving-block bootstrap replicates at primary block
length 16 and fixed sensitivity lengths 8 and 32.  For each paired row
difference `d_i = candidate_i - baseline_i`, contiguous blocks are sampled with
identical indices for the pair.  Within each synthetic seed, the resampled
statistic is computed first and the ten seed statistics are then averaged with
equal `1/10` weight (never row-count weighting); S3 is one timestamp stratum.
For every length `L` in `{8,16,32}`, the two-sided percentile interval
`[quantile_0.025, quantile_0.975]` is diagnostic.  The one-sided raw p-value
used for inference is `raw_p=max(p_8,p_16,p_32)`, a conservative
intersection-union rule; no gate is formed by mixing unadjusted confidence
limits.  Every non-S0 primary gate uses this raw p after the fixed
Holm–Bonferroni family correction (`alpha=0.05`, family size 16).
The immutable reporting-arm ledger `p1_recovery_trial_registry.jsonl`
enumerates all 56 execution arms (seven scenario arms x four model IDs x two
cost modes).  The separate
`p1_recovery_primary_comparisons.jsonl` enumerates the 16 executable paired
comparisons; only its `primary=true` records define the multiplicity family.
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
Monotonicity passes only when the two adjacent inequalities high-vs-medium
and medium-vs-low hold after the fixed `1e-12` tie tolerance; no level may be
removed or retuned.

Missing rows, insufficient class support, undefined metrics, missing masks,
missing provenance, altered manifest fields, and failed coverage are N/A or
blocked.  The runner must stop before fitting or promotion, and it must not
change a threshold, seed, feature, horizon, split, cost, or outer-row status
after seeing any output.

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

The `cache_dir`/`cache_tag`-only default lookup is forbidden for S3.  The
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

The canonical action execution
contract is `action_execution_contract.json` (the path and canonical-content
SHA-256 are pinned in the manifest) and is shared by optimizer, teacher,
student replay, U0, Q, and Backtest.  Logistic regression is a binary
proper-score diagnostic only; Ridge is the sole learned h4 action mapper.
Zero-return and persistence-last-observed are fixed comparators: persistence
predicts `h * return[t]` continuously and clips its binary probability to
`1-1e-6` for a nonnegative last return and `1e-6` otherwise.  Logistic action
utility is N/A, so no post hoc probability-to-return conversion is permitted.
