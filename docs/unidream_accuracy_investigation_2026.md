# UniDream accuracy investigation (2026)

**Report state:** Stage 3 — complete evidence audit and report update (Wave3C
development result, hypotheses, checklist, literature, and agent plan). The
evidence-only skeleton was first pushed as `66d6298`, and the Stage 2 plan as
`8737581`; this report remains a report-only artifact. It does not retrain a
model, apply a Supabase migration, deploy a Space, or write production data.

**Evidence cutoff:** 2026-08-30. The Research, Space, and Web repositories are
separate repositories. The snapshots inspected for this report were:

| repository | inspected revision | role |
| --- | --- | --- |
| `UniDream` | `330c712` Wave3D artifact audit; v4 provenance `d06bfa4`; Wave3C artifact `c3b4ccb` (run `8e9e0fc`, baseline evidence begins at `a871439`) | Research training/evaluation, official-source readers, and contract tests |
| `unidream-space` | public SHA prefix `60052bc5…` | Plan011 v31 inference service and feature parity |
| `unidream-demo-web` | `271baf7` | Edge data fetch, paper trading, Supabase writes, dashboard |

All cited forecast, backtest, source-rebuild, and Wave3D changes are audited as
committed artifacts and are intentionally not modified by this report. Revision
`157e408` adds the official-source reader and probe; the later v4 run and its
explicit-gap provenance are audited below and are not model results. Wave3C is
audited from the latest committed artifact at `c3b4ccb`; its underlying
tournament code/data commit is recorded in the Wave3C section below.

## Reading contract

The labels below separate what the repository proves from what it merely
suggests.

* **CONFIRMED** — stated by a committed artifact, source inspection, or a
  passing deterministic test at the cited revision.
* **INFERENCE** — a technically plausible explanation or experiment, not a
  measured causal result.
* **UNVERIFIED** — requires a new run, an external provider, a local database,
  or production observation.
* **GATE STOPPED** — a required contract currently fails; no accuracy claim
  should be promoted through that gate.
* **PENDING** — implementation or review exists, but the required numerical
  evidence is not yet complete.

Wave3C is now a numerical source for a narrowly scoped, development-only
report. Its values are not holdout or production accuracy, and `status:
complete` means the artifact was generated—not that a candidate passed
promotion. Wave3D is a separate, returns-only constant-exposure diagnostic:
its artifact is complete, but its gate failed, and it is not forecast/model
evidence. A constant exposure has no temporal timing path, so Wave3D reports
timing increment as `N/A` rather than substituting a Wave3C value.

## Executive evidence summary

The current evidence supports a pipeline-contract and out-of-sample
generalization investigation, not a single identified model defect.

1. **Research data quality is still a stopped gate.** The original v3 audit
   for `[2018-01-01, 2024-01-01)` and 15-minute bars reports an overall `FAIL`:
   29 non-15m steps and an estimated 542 missing bars, with no availability
   mask. A subsequent official v4 run generated a 17-column body and a
   210,336-row sidecar, but its raw gap ledger summed to 611 missing Spot bars
   before REST recovery: 80 were recovered, 531 remain unresolved, and 81
   off-grid rows were quarantined. It is therefore useful audit evidence, not
   a complete full17 training window; the availability-aware training path is
   still required before promotion.
2. **The retained model result is modest and not drawdown-improving, but is
   stale for the corrected Plan011 contract.** Development folds 0–12 have
   mean AlphaEx `+0.41pt`, median `+0.16pt`, and MaxDDDelta `+0.20pt`; the
   2024–2026 holdout folds 15–23 have mean AlphaEx `+0.11pt`, median `-0.04pt`,
   and MaxDDDelta `+0.20pt`. Negative MaxDDDelta is the improvement direction.
   These are historical v3 snapshots produced before commit `c8f6914` fixed
   the Plan011 World Model future-action context. They are retained for
   provenance only and are **not valid accuracy estimates for the corrected
   Plan011 implementation**; retraining is required before comparing corrected
   accuracy.
3. **Wave3C found no usable winner under the corrected development contract.**
   All six feature-set × candidate combinations failed the signal/timing
   gate. The apparent roughly `+20pt` AlphaEx is the validation-selected
   constant exposure baseline (`0.5 / 1.12 / 0.5` for folds `0 / 2 / 8`), not
   a dynamic timing edge: every candidate's median dynamic-minus-constant
   timing increment is negative. OHLCV13 forecast diagnostics are modest and
   report-only; the historical full17 rows are blocked because the v3 cache
   had no availability mask, while the generated v4 sidecar is not yet
   consumable by the training path. The corrected Wave3A replay confirms the
   same interpretation. A separate Wave3D returns-only constant-exposure
   diagnostic over development folds 0–11 also failed its gate: selected
   constant median AlphaEx was `+1.5644pt` (6/12 positive; mean `+0.8584pt`)
   versus the previous-fold comparator median `+4.0307pt` (mean `+6.0872pt`),
   with median MaxDDDelta `-4.1566pt`; it has no timing increment and makes no
   predictive claim.
4. **The live contracts are stricter than the old research cache.** Space
   requires all 17 Plan011 inputs and rejects missing/non-finite funding or
   mark data. Its public deployment was observed at SHA prefix `60052bc5`,
   stage `RUNNING`, one `cpu-basic` replica: `/health` returned `ok=true` with
   the exact 17-feature schema, and `/sample/verify` returned `ok=true`,
   `strict_ok=true`, `n=8641`, and position max difference
   `1.1920929e-7`. The live advantage max difference was `1.531839e-5`, within
   the diagnostic tolerance `2e-5`, but its strict flag was false. Web
   excludes partial candles, joins funding as-of and mark by candle timestamp,
   and commits prediction/state/snapshot/trade through one CAS-protected RPC.
   The deployed Space contract/sample parity is therefore confirmed, while
   `/predict/latest` returned `502` because official FAPI funding/mark calls
   returned HTTP `451` from that region; provider-backed latest inference and
   production database execution remain unverified.
5. **Deep architecture and offline-RL follow-ups are gate-stopped.** C8 deep
   architecture and C9 offline-RL experiments were not run after the Wave3C
   signal gate failed; they are not completed results. C10/C11 statistical
   promotion is also intentionally not claimed. The development-only
   statistical gate accepts explicit per-bar paths and has moving/stationary
   block bootstrap, DSR with an explicit trial count, CSCV/PBO, sign tests, and
   cost/regime stress. Wave3D applied that contract to a returns-only constant
   exposure diagnostic: its additive AlphaEx CI was `[-66.0753,+75.3420]pt`,
   DSR probability was `0.8880` versus `0.95`, PBO was `0.4524` (pass), cost
   stress passed but regime stress passed only `2/3`; the overall gate failed.
   This is not a candidate-model significance claim, and no holdout/deep/RL
   result was run.

## System and evidence map

```text
official Spot OHLCV + USDⓈ-M funding/mark
        │
        ├── Research v4 cache + availability sidecar ──┐
        │                                                │
        └── Web closed-candle derivative join             │
                                                         ▼
         right-exclusive WFO ──> Hindsight Oracle ──> Transformer WM
                                                         │
                              WM predictive state ──> BC ──> Imagination AC
                                                         │
                              validation selector ──> test/report-only metrics
                                                         │
                           Space Plan011 v31 bundle ──> Web Edge RPC/dashboard
```

The causal training convention is important: Research documents that the
feature row at bar `t` uses information known by the start of bar `t` (most
technical features are shifted by one bar), while `returns[t]` is the realized
bar-`t` return. A forecast target and an execution delay therefore need an
explicit alignment proof; a positive result without that proof is not
accepted as causal.

## Confirmed evidence by repository

### Research: feature, data, and model contract

**CONFIRMED — canonical schema.** `unidream/data/cache_v4.py` defines the
17-column model order:

```text
open_ret, high_ret, low_ret, close_ret, vol_ret,
RSI_14, macd, macd_signal, atr_norm_ret, atr,
rv_4, rv_16, rv_96, funding_rate, basis, basis_mom, basis_abs
```

The first 13 columns are OHLCV-derived. `funding_rate` is sourced from
funding observations; `basis`, `basis_mom`, and `basis_abs` are sourced from
mark close. The cache-v4 contract keeps these values in a 17-column feature
body and stores `spot_bar_observed`, `funding_rate_available`, and
`mark_close_available` in a separate complete-grid sidecar.

**CONFIRMED — causality probes.** `unidream/data/features.py` shifts the
OHLCV technical inputs and external series, uses forward/as-of alignment, and
does not backfill a mark series from its future first observation. The data
quality report records passing `future_perturbation_prefix`,
`prefix_invariance`, `mark_offset_no_future_bfill`, and
`funding_offset_asof` probes, each with max difference `0.0`.

**GATE STOPPED — current development cache and v4 availability.**
`docs/data_quality_gate_2018_2024.md` records:

* scope `[2018-01-01, 2024-01-01)`, 15m, overall `FAIL`;
* cache contract `fail`, causal probes `pass`, OHLCV13/full17 same-row
  eligibility `pass`, external availability mask `fail`;
* 29 non-15m steps and 542 estimated missing bars; non-finite count `0`;
* 18 of 542 expected timestamps covered by official gap recovery and 524
  unresolved; no interpolation or in-place v3 repair;
* OHLCV13 and full17 eligible rows both `208299`, with the same row mask.

**CONFIRMED — actual official-source v4 rebuild audit.** The final run record
is `2fc795c` (the off-grid rejection/quarantine policy was introduced in
`7e0dfb9`; final provenance/evidence is recorded at `d06bfa4`) and uses the
same runtime data contract as the inspected Plan011 config:
`BTCUSDT`, `15m`, scope `[2018-01-01, 2024-01-01)`, z-score window `60`,
`extra_series_mode=derived`, no extra series, `include_funding=true`,
`include_oi=false`, and `include_mark=true`. Its cache tag is
`BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official`, schema `4`, with 17
feature columns, `173111` body rows, and a complete `210336`-row availability
sidecar. The source probe passed (`spot 4/4` HTTP 200; UM mark `1/4` and
funding `1/4`, with the pre-2020 archive probes returning 404 and masks false).

The run status is `generated_with_explicit_spot_gaps`: its pre-REST gap ledger
sums to `611` missing Spot bars; `209805/210336` Spot bars are observed after
`80` missing bars are recovered by official REST, `531` remain unresolved, and
`81` off-grid Spot rows are quarantined without timestamp remapping (the
explicit quarantine mode is recorded as enabled).
The sidecar reports funding as-of availability
`140255/210336`, exact causal mark availability `139485/210336`, and all three
Spot/funding/mark flags true for `139333/210336`; only `119849/173111` body
rows have all three flags true. The body is intentionally not filtered by the
external masks, so an availability-aware sequence builder must exclude unsafe
windows. The final source/provenance digest is
`aa320222dca0a46b2a0730f17bb1665f31a70074aa3bafcc6bff58ca21618fad`.

The metadata parameters match the rebuild CLI and the config values above;
this is a parameter-consistent generated artifact, not a metadata mismatch.
However, the model config does not set `data.cache_schema: v4`, so the training
entrypoint defaults to the legacy `v3` tag. Explicit v4 selection is required;
when funding or mark inputs are enabled, the current loader rejects promotion
because it cannot propagate the sidecar into `SequenceDataset/WFODataset`.
This is the intended fail-closed boundary from `f37a202`, not a successful
full17 training run. The run ledger has one run record plus source, gap, and
off-grid provenance records; its metadata path and source/content digests are
retained in `docs/data_quality_v4_rebuild_2018_2024.jsonl` and
`docs/data_quality_v4_rebuild_2018_2024_metadata.json`. No model result was
read, and no v3 file was overwritten.

The audited contract chain includes `664d656` (off-grid source rejection),
`f37a202` (fail-closed full17 promotion), `bf54ad2` (initial blocked-run
evidence), and the subsequent explicit quarantine, schema-normalization,
coverage, report, and provenance revisions through `d06bfa4`. A read-only
parameter check returned `metadata_parameters_equal_config_runtime=True`, the
repository metadata copy matched the generated cache metadata, and
the full Research unittest suite completed `123/123` tests `OK`. These checks
do not verify a deployed training run or production runtime.

### Research: retained performance evidence

`docs/plan011_v31_investor_evidence.md` is a historical result snapshot. Its
locked setup is BTCUSDT/15m, Plan011 v31 WM → BC → AC, folds 0–12 for
development and folds 15–23 for the untouched 2024–2026 holdout, with train
2 years, validation 3 months, test 3 months, and 3-month slides. Validation
selects; test is report-only. The default one-way full position-change cost is
5.50 bps (`fee 0.0003`, full spread 3 bps, slippage 1 bps).

**Important validity boundary:** the retained development and holdout
snapshots were produced before `c8f6914` corrected Plan011's World Model
action-context handling. Therefore they are historical pre-correction
artifacts, not evidence for the current corrected pipeline. They also use the
v3 cache contract described above. A new corrected run on the repaired v4
cache is required; this report deliberately does not invent its values.

The retained aggregate values are:

| scope | AlphaEx mean | AlphaEx median | AlphaEx positive | MaxDDDelta mean | goal `+3/-3` |
| --- | ---: | ---: | ---: | ---: | ---: |
| development folds 0–12 | `+0.41pt` | `+0.16pt` | `7/13` | `+0.20pt` | `0/13` |
| holdout folds 15–23 | `+0.11pt` | `-0.04pt` | `3/9` | `+0.20pt` | `0/9` |

Here AlphaEx is strategy final total return minus B&H final total return;
MaxDDDelta is absolute strategy MaxDD minus absolute B&H MaxDD, so a negative
value is better. These values are evidence of weak/unstable benefit and no
drawdown improvement, not evidence of a causal root cause.

`docs/alpha_attribution_plan011_v31_dev/report.md` records a constant-exposure
component and a timing component under the same cost contract, but its
retained actor-mean row is diagnostic-only because no validation actor path
was saved. Any new comparison must keep validation selection separate from
test reporting and must not silently use an unselected test mean.

### Wave3A, Wave3C, and Wave3D status

**CONFIRMED — Wave3A frozen development screen, with an erratum boundary.**
`docs/forecast_tournament_plan011_dev/report.md` covers development folds
0, 2, and 8, horizons 4/16/64, fixed operational delay 1 bar, and sensitivity
lags 1/16. The screen tested full17 and OHLCV13 with a causal trend/vol rule,
Ridge, and HistGradientBoosting. All six combinations failed. Examples from
the retained aggregate table are full17 causal `IC -0.0329`, AlphaEx
`-1.3651pt`; full17 HistGB `IC +0.0448`, AlphaEx `-0.7256pt`; and full17 Ridge
`IC +0.0187`, AlphaEx `+0.8335pt`, which still failed the complete gate.

The original Wave3A result remains a historical artifact. It predates the
baseline-exposure matching, leading-padding removal, and turnover-semantics
audit, so it is not corrected timing-superiority evidence. The committed
Wave3A corrected replay below is the appropriate erratum comparison and is
report-only.

**CONFIRMED — Wave3C development result (artifact `c3b4ccb`).** This refresh
supersedes the initial `de66ff0` artifact for numeric claims; no `de66ff0`
replay value is used below. The committed artifact
`docs/forecast_context_tournament_plan011_dev/result.json` has
`status: complete`, seed `7`, exact development folds `[0, 2, 8]`, horizons
`[4, 16, 64]`, feature sets `ohlcv13` and `full17`, and candidates Ridge,
HistGB, and downside classifier. It fits on train, selects horizon/policy on
validation, and reports the development test only; no fold 15+ result is read.
The fixed operational execution delay is one bar, with timing lags `1/16` and
null shifts `1/16/64`. Artifact completion is not candidate promotion: all six
candidate gate rows fail, and `next_wave_candidates` is exactly `[]`.

The selected OHLCV13 forecast quality is:

| fold | Ridge (selected horizon; IC / sign accuracy) | HistGB (selected horizon; IC / sign accuracy) | downside classifier (selected horizon; AUC) |
| ---: | --- | --- | --- |
| 0 | h16; `+0.033066 / 0.503925` | h4; `+0.046314 / 0.518442` | h16; `0.622610` |
| 2 | h64; `+0.071069 / 0.542780` | h16; `+0.029532 / 0.519935` | h4; `0.645271` |
| 8 | h64; `-0.002432 / 0.495964` | h64; `-0.027998 / 0.497002` | h4; `0.678548` |
| median | —; `+0.033066 / 0.503925` | —; `+0.029532 / 0.518442` | —; `0.645271` |

The selected-quality-positive fold counts are Ridge `2/3`, HistGB `2/3`, and
classifier AUC `3/3`. Classifier sign accuracy is not defined by this
one-sided downside-event target; its sparse precision/recall are diagnostics,
not a replacement for AUC. These are development-test forecast diagnostics,
not holdout or production accuracy.

The corrected economic aggregates (percentage-point AlphaEx/MaxDDDelta and
position-path cost turnover) are:

| feature set | candidate | median forecast quality | median dynamic AlphaEx | median constant AlphaEx | median timing increment | median MaxDDDelta | median cost turnover | gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ohlcv13 | Ridge risk-adjusted context | `+0.033066 IC` | `+20.059139` | `+20.167520` | `-0.108381` | `-8.237799` | `10.065126` | **fail** |
| ohlcv13 | HistGB risk-adjusted context | `+0.029532 IC` | `+18.496408` | `+20.167520` | `-1.671112` | `-8.313140` | `8.916144` | **fail** |
| ohlcv13 | downside classifier | `0.645271 AUC` | `+20.093023` | `+20.167520` | `-0.074498` | `-8.313140` | `3.130630` | **fail** |
| full17 | Ridge risk-adjusted context | `+0.053675 IC` | `+20.076113` | `+20.167520` | `-0.091407` | `-8.384446` | `9.682861` | **blocked: historical v3 has no availability mask** |
| full17 | HistGB risk-adjusted context | `+0.061134 IC` | `+19.780975` | `+20.167520` | `-0.386545` | `-8.064934` | `5.467703` | **blocked: historical v3 has no availability mask** |
| full17 | downside classifier | `0.626428 AUC` | `+19.714931` | `+20.167520` | `-0.452590` | `-8.313140` | `2.991194` | **blocked: historical v3 has no availability mask** |

The interpretation is decisive for this screen: the apparently roughly
`+20pt` AlphaEx is the validation-selected constant exposure component, not a
dynamic timing improvement. The selected constant exposures are fold `0`
`0.5`, fold `2` `1.12`, and fold `8` `0.5` (`0.5 / 1.12 / 0.5`) for every
candidate. All six dynamic-minus-constant timing medians are negative. OHLCV13
is the formal promotion feature set for the artifact, but every OHLCV13
candidate still fails its signal/timing gate. Full17 is additionally blocked
in the historical screen because the v3 cache has no availability mask to
distinguish external zero from missing/imputed derivative values. The later v4
sidecar does distinguish them, but the current training loader fail-closes
until it propagates those flags into sequence eligibility; this is not evidence
that full17 is accurate or inaccurate.

**Wave3A corrected replay (confirmed, report-only).** The replay in the
Wave3C result has source artifact
`docs/forecast_tournament_plan011_dev/result.json`, source commit
`e0ab435ab6601ce49b4f6c28bdb15504d2c57315`, nine rows, and zero failures. It
uses validation-selected constant exposure, a common right-side return
window, and fixed delay one. Its medians are dynamic AlphaEx
`+18.761945pt`, constant AlphaEx `+20.167520pt`, and timing increment
`-1.405575pt`. It is excluded from Wave3C candidate selection/gate; it exists
to document the corrected interpretation of the historical Wave3A screen.

**Wave3C provenance and integrity audit (confirmed).** The result records
tournament source commit `8e9e0fcfaed92c99522aa22f724943c89484aee3`, seed `7`,
config SHA-256
`5c9599bdb2b5495ad525715291fba7812232b24353c4ccc16d42f1d5ad15a944`, source
SHA-256 `b1038b7b7af457c29f660f6b117b903b1c436535c0edf9e8e860367f805a3aa1`,
data-contract SHA-256
`bc8a53c87eb196d443b09a491cfd0803ec4ecef30c9524246dacb690f14bb2cb`, and
data SHA-256
`027b476f209d41f477623d92cd3e2f5061976840fd682874588fdcebdeab8704`.
The committed artifact hashes are result
`2966130a88b203b8250786d30ddc95f9d937b96d83f3fd2e1f2d405d39675d3d`, report
`e3859393b622d57df1e0f65fffae3b8a2e7e0b849b2cef13e36075840232abe8`, ledger
`a4e707fdd2af292b24395dd9c5d5aabd4b2deb5aa3337689c48b3b36b86885c1`, and
errata `878f518a0b936c512701680f615feae6e917c011bd0d6f14eb4d12d03b17fce0`.
The recursive result JSON audit found `0` non-finite values (the committed
result is 14,795 lines with 18 candidate rows and 18 feature-quality rows).
All `376` ledger lines parse as JSON and are finite; record counts are one run,
`18` feature-coverage, `18` candidate, `180` forecast-metric, `126`
economic-metric, `18` timing-attribution, `6` successive-halving-gate, and
`9` corrected-replay records. The artifact runtime is `179.88257045800128s`.
For the formal OHLCV13 rows, runtimes in seconds were:

| fold | Ridge | HistGB | classifier |
| ---: | ---: | ---: | ---: |
| 0 | `12.797059` | `7.691945` | `6.743973` |
| 2 | `12.708218` | `7.896540` | `7.127292` |
| 8 | `13.426194` | `7.580536` | `6.556530` |

No statistical gate was applied to these candidate paths. `full17` has
promotion status `blocked_by_data_quality`, `ohlcv13` is only the formal
feature-set designation, and `all_candidates_failed_gate` is true.

**CONFIRMED — Wave3D constant-exposure diagnostic (artifact complete; gate
failed; not model evidence).** The committed artifact at
`docs/constant_exposure_plan011_dev/` was generated from source commit
`f6d273b` and published in `330c712`. It is a returns-only diagnostic over
exact development folds `0–11`, with fixed execution delay `1`, no delay
tuning, fixed exposure grid `[0.5, 0.75, 1.0, 1.05, 1.1, 1.12]`, and costs
`fee 0.0003`, full spread `3 bps`, slippage `1 bps`. It excludes fold `12+`
and holdout data; features and model artifacts were not used.

The selected constant path has median AlphaEx `+1.5644pt`, mean `+0.8584pt`,
`6/12` positive folds, and median MaxDDDelta `-4.1566pt`. The prior-fold
comparator on folds `1–11` has median `+4.0307pt`, mean `+6.0872pt`, while the
selected path on those same folds is median `+3.1289pt`, mean `+2.3851pt`.
The selected path's additive AlphaEx estimate is `+1.7774pt` with a primary
block-bootstrap CI `[-66.0753,+75.3420]pt`; all block-length sensitivity lower
bounds are negative. Its fold sign test is `N/A`/fail (`6` positive, `5`
negative, `1` zero; only `11` non-zero folds versus the required `12`). DSR
uses explicit `n_trials=7` and returns p-value `0.11197` (probability
`0.8880 < 0.95`, fail). CSCV/PBO is `0.4524` over six candidates and twelve
even subperiods (diagnostic pass, report-only). Cost stress passes `3/3`, but
regime stress passes only `2/3`, so the overall Wave3D gate is **FAIL** and
`next_wave_candidates=[]`.

A constant exposure has no temporal timing path; its timing increment is
`N/A`, not a substitute for Wave3C's dynamic-minus-constant values. Wave3D
therefore cannot establish forecast quality, dynamic timing superiority, or a
model promotion case. Its result and ledger hashes are committed and match;
the result JSON and all `135` ledger lines are finite, and the diagnostic
remains development-only and report-only.

### Space inference service

**CONFIRMED at committed code/test level.** `unidream-space` describes the
current bundle as Plan011 v31 fold23 Transformer WM + BC + Imagination AC for
BTCUSDT 15m. `backend/feature_contract.py` separates 13 OHLCV-derived
features from the four derivative-source features and rejects wrong order,
missing columns, or non-finite matrices. `backend/feature_pipeline.py`:

* derives an injected interval-boundary cutoff and rejects an unclosed latest
  candle or non-contiguous input;
* requires finite funding observations that cover the first OHLCV candle;
* requires finite, positive mark closes for every OHLCV timestamp;
* computes metadata with `closed_candle_only=True` and fails closed on source
  fetch failure or missing derivative coverage.

`/predict` raw candles therefore require timestamp/open/high/low/close/volume
plus funding and mark inputs; the service does not zero-fill missing derivative
inputs. The public deployment observation above confirms the deployed health
and sample contract, but its provider-backed latest path remains unavailable
from the observed region because FAPI returned HTTP `451`.

### Web Edge, paper trading, and dashboard

**CONFIRMED at committed code/test level.** `unidream-demo-web` has the
following contracts:

* `fetchCandles` fixes one observation cutoff, keeps only Spot candles whose
  reported close is at or before that cutoff, paginates deterministically, and
  returns exactly `TARGET_BARS` when coverage is complete;
* USDⓈ-M funding is aligned using the latest publication at or before the
  candle timestamp, and mark close is joined by exact candle timestamp;
* every candle sent to Space contains the six OHLCV fields plus
  `funding_rate` and `mark_close`; missing/invalid coverage fails closed;
* `callPredict` retains the existing Space POST shape (`symbol`, `timeframe`,
  `candles`, `tail`);
* `record_unidream_inference` commits prediction, strategy state, equity
  snapshot, and trade in one Postgres transaction with a strategy-state row
  lock, expected-state CAS, duplicate keys, and explicit finite predicates;
* research-aligned costs are fee `0.0003`, full spread `3 bps` (half spread
  `1.5 bps`), and slippage `1 bps` applied to changed exposure. The legacy DB
  `fee` column stores the all-in quote cost and must not be displayed as fee
  alone;
* dashboard contracts distinguish the last closed bar timestamp from RPC
  record time. Public badges are source-configured declarations, not row-level
  live health proof.

**UNVERIFIED.** No production Supabase database or Edge deployment was
modified or exercised for this report. Local SQL execution, Edge provider
reachability, Cron scheduling, HF Space latency beyond the observed endpoints,
and real fill/slippage are external runtime boundaries unless separately
recorded by an operator. The Space `HTTP 451` observation is regional and does
not establish a general Binance or Web availability result.

### Statistical robustness gate

**CONFIRMED — independent API/contract only.** `unidream/eval/statistical_gate.py`
and `docs/statistical_gate_contract.md` consume explicit, already-aligned
development fold paths and reject fold 15 or later. The fixed contract covers:

* moving-block (default) or stationary bootstrap confidence intervals with
  preregistered block length/sensitivity and deterministic seed;
* Bailey/López de Prado Deflated Sharpe Ratio with per-bar Sharpe, display-only
  annualization, and an explicit `n_trials` requirement for promotion;
* CSCV/PBO, requiring at least two candidates and an even number of at least
  four subperiods (the current 13-fold development layout is intentionally
  `N/A` for CSCV);
* paired fold sign/binomial tests and required cost/regime stress cases;
* fail-closed handling where `N/A` never passes and CLI rejection is explicit.

The repository contains synthetic unit tests for the API. Wave3D applied the
contract to a returns-only constant-exposure diagnostic and failed its overall
gate; no forecast/model candidate input was applied, so this does not
manufacture a significance claim for Plan011.

## Gate status at this cutoff

| gate | status | what is established | what remains |
| --- | --- | --- | --- |
| Gate0 WM validation/action context | **complete/pass** | ordered full validation evaluation, coherent best-state restore, and Plan011 actionless WM context are tested | a new model run on v4 data |
| Research causal feature probes | **complete/pass** | four perturbation/prefix/as-of probes pass with max diff 0 | availability and gap-complete data |
| Research v3 data quality | **GATE STOPPED/fail** | audit records gaps and zero/missing ambiguity | official v4 regeneration and window exclusion |
| v4 cache contract | **generated artifact / GATE STOPPED** | writer, loader, sidecar, digest, gap policy, and official run with explicit gaps are recorded | exclude windows crossing 531 unresolved Spot gaps and add availability-aware sequence consumption before full17 promotion |
| Plan011 historical evidence | **complete pre-correction snapshot / not current-accuracy evidence** | dev/holdout aggregates and report-only semantics before `c8f6914`, on v3 | corrected retraining on v4; do not compare as current Plan011 |
| Wave3A forecast tournament | **frozen pre-errata diagnostic; corrected replay complete** | original six-combination screen is historical; nine-row corrected replay records the common-window/constant-baseline interpretation | no promotion from the historical screen; v4 rerun if a new candidate is justified |
| Wave3C | **complete artifact / promotion stopped** | six development candidates ran on folds 0/2/8; all failed the signal/timing gate; `next_wave_candidates=[]`; full17 is data-quality blocked | corrected v4 data and a new preregistered run; no holdout or production claim |
| Wave3D constant exposure | **complete artifact / gate fail** | returns-only diagnostic on folds 0–11: selected median AlphaEx `+1.5644pt`, mean `+0.8584pt`, positive `6/12`, median MaxDDDelta `-4.1566pt`; prior comparator median/mean `+4.0307pt`/`+6.0872pt`; `next_wave_candidates=[]` | no model/timing claim; any future dynamic comparison must use a separate common-window path |
| Space 17-feature parity | **deployed health/sample parity verified; latest provider path unavailable** | public SHA prefix `60052bc5`, `RUNNING` cpu-basic/one replica; `/health` exact 17-feature schema; `/sample/verify` strict pass over `n=8641` | `/predict/latest` provider-backed call returned `502` after FAPI funding/mark HTTP `451` from the observed region |
| Web closed-candle/atomic/cost | **complete code/tests** | closed data, derivative join, one RPC, cost contract | local SQL and production runtime verification |
| Statistical gate | **complete Wave3D diagnostic / model application blocked** | CI/DSR/PBO/sign/stress contract exercised on returns-only constant exposure; overall diagnostic gate failed | apply only to a new forecast/model candidate after C0/C5/C7 pass, with full trial count |

## What this evidence does not prove

It does not prove that funding/mark features improve or harm accuracy, that a
Transformer is inferior to a simpler model on BTCUSDT, that AC caused the
pre-correction holdout shortfall, or that any Wave3C candidate passes. The
roughly `+20pt` Wave3C AlphaEx is explained by the selected constant exposure
baseline in this development screen, not by a measured dynamic timing edge.
It does not prove live profitability, execution capacity, actual exchange fills, Supabase atomicity
under concurrent production requests, or statistical significance. Those are
questions for the controlled checklist below.

## Source index for the confirmed evidence

* Research scope and pipeline: [`SPEC.md`](../SPEC.md),
  [`docs/project_map.md`](project_map.md), and [`docs/README.md`](README.md).
* Feature causality and schema: `unidream/data/features.py`,
  `unidream/data/cache_v4.py`,
  [`docs/data_quality_cache_v4.md`](data_quality_cache_v4.md).
* Data gate and gaps: [`docs/data_quality_gate_2018_2024.md`](data_quality_gate_2018_2024.md)
  and [`docs/data_quality_gap_recovery_2018_2024.md`](data_quality_gap_recovery_2018_2024.md).
* Actual official v4 rebuild audit: [`docs/data_quality_v4_rebuild_2018_2024.md`](data_quality_v4_rebuild_2018_2024.md),
  [`docs/data_quality_v4_rebuild_2018_2024.jsonl`](data_quality_v4_rebuild_2018_2024.jsonl),
  and [`docs/data_quality_v4_rebuild_2018_2024_metadata.json`](data_quality_v4_rebuild_2018_2024_metadata.json).
* Result snapshots: [`docs/plan011_v31_investor_evidence.md`](plan011_v31_investor_evidence.md),
  [`docs/alpha_attribution_plan011_v31_dev/report.md`](alpha_attribution_plan011_v31_dev/report.md),
  and [`docs/forecast_tournament_plan011_dev/report.md`](forecast_tournament_plan011_dev/report.md).
* Wave3D constant-exposure diagnostic: [`docs/constant_exposure_plan011_dev/report.md`](constant_exposure_plan011_dev/report.md),
  [`docs/constant_exposure_plan011_dev/result.json`](constant_exposure_plan011_dev/result.json),
  and [`docs/constant_exposure_plan011_dev/ledger.jsonl`](constant_exposure_plan011_dev/ledger.jsonl),
  published at `330c712` from source commit `f6d273b`.
* Statistical contract: [`docs/statistical_gate_contract.md`](statistical_gate_contract.md).
* Cross-repository implementation evidence is retained in the corresponding
  `unidream-space/README.md`, `backend/feature_contract.py`,
  `backend/feature_pipeline.py`, `unidream-demo-web/README.md`,
  `supabase/functions/run-unidream-inference/binance.ts`,
  `supabase/functions/run-unidream-inference/atomic.ts`, and
  `supabase/migrations/0003_atomic_inference.sql` at the revisions listed at
  the top of this report.

## Accuracy-decline hypotheses

The following are ranked working hypotheses. A ranking is a decision aid, not
a measured attribution. The numerical values treated as evidence are the
committed snapshots and the scoped Wave3C development artifact above; none is
a production or corrected-v4 accuracy estimate.

### P0 — data and contract validity

**H1: gaps and derivative zero/missing ambiguity contaminate comparability.**
The v3 gate failure, 542 estimated missing bars, 524 unresolved gaps, and lack
of availability flags are confirmed. The official v4 rebuild now records a
generated body plus sidecar, but still has 531 unresolved Spot bars and only
partial external availability; it is not a complete training window. It is an
**INFERENCE**, not proof that this caused the historical AlphaEx decline: the
committed results were computed on the old cache, and no paired
v3-versus-v4 experiment exists. Repairing this first is mandatory because
every later model comparison could otherwise measure a data-generation change.

**H2: target, feature, and execution timing may be misaligned.** The Research
feature contract says row `t` is known through the start of bar `t`, while
`returns[t]` is realized at bar `t`. A target written as `t+1..t+h`, a policy
delay, and the backtest position path must be right-aligned exactly once. A
positive delay must trim `positions[:-d]` against `returns[d:]`; padding with
`p0` is not causal. The Web/Space closed-candle contracts are confirmed in
code/tests, but cross-repository end-to-end alignment for every research
experiment is **UNVERIFIED**. This is a P0 audit item rather than a claimed
cause.

**H3: live/demo and research inputs can drift despite a shared 17-column
name/order.** Space now requires raw derivative sources and Web supplies them,
but research v3 has ambiguous external availability and a historical snapshot
with a different data contract. A schema match is necessary, not sufficient:
source, as-of time, cutoff, normalization, row eligibility, and model/bundle
hashes must all match. The drift risk is **INFERENCE** until a same-candle
parity replay produces a signed digest.

### P1 — signal, exposure, and objective

**H4: the apparent benefit is exposure, not timing.** The retained attribution
report defines constant exposure as the actor-mean constant path and timing as
the actor sequence minus that constant under the same costs. Wave3C now
provides the scoped test: validation-selected constants were `0.5 / 1.12 /
0.5` for folds `0 / 2 / 8`, and all six dynamic timing medians were negative
(`-1.6711` to `-0.0745pt` across OHLCV13/full17). This interpretation is
**CONFIRMED for the Wave3C screen**, not a universal causal attribution. The
Wave3D constant-exposure diagnostic is also complete, but it has no temporal
timing path and therefore cannot provide the paired dynamic-vs-constant
timing interval requested here; any future dynamic/common-window comparison
must report mean exposure, paired block intervals, and cost turnover separately.

**H5: the training objective and promotion objective are not identical.** The
Plan011 config uses B&H-relative rewards, relative drawdown and turnover
penalties, predictive-state heads, and an AC horizon of 8, while the declared
promotion target is final-value AlphaEx `>= +3pt` together with MaxDDDelta
`<= -3pt`. That configuration mismatch is **CONFIRMED**; its causal effect is
**UNVERIFIED**. A fixed reward/selector ablation must measure the final metrics
under the same cost and timing contract.

**H6: offline RL extrapolation or world-model error may erase the BC signal.**
The current pipeline uses WM → BC → imagination AC, with one ensemble member in
the inspected Plan011 config and no CQL/IQL conservative objective. Offline
distribution shift is a known mechanism in the literature, but there is no
local OPE proof that it is the cause. This remains an **INFERENCE**; C9 was
stopped before execution after the Wave3C signal gate failed, so no CQL/IQL
result exists. Any future test needs action-support diagnostics,
shuffled-latent/no-WM ablations, and conservative baselines; the untouched
test/holdout must remain report-only.

**H7: forecast signal is weak or regime-specific.** Wave3C's selected OHLCV13
forecast metrics are modest: Ridge median IC `+0.033066` with sign accuracy
`0.503925`, HistGB median IC `+0.029532` with sign accuracy `0.518442`, and
classifier median AUC `0.645271`; all six economic candidates nevertheless
failed the signal/timing gate. These forecast values are **CONFIRMED for the
development screen**, but do not identify whether weak IC, policy translation,
costs, or regime change is dominant. Long BTCUSDT history crosses materially
different regimes, so non-stationarity remains an **INFERENCE**, not evidence
of model failure. Regime-stratified development-only paths and fixed cost
stress are required.

**H8: derivative features may add variance without incremental signal.** The
Wave3C full17 rows are blocked because the historical v3 cache lacks an
availability mask; the generated v4 sidecar is not yet wired into sequence
eligibility. The OHLCV13 rows fail the timing/economic gate. This is not a
causal derivative ablation because the data-quality contract is not v4-clean.
The only safe statement is that derivative usefulness is **UNVERIFIED**. In addition,
Research and Space currently derive `basis` with an effective raw lag of one
bar, while `basis_abs` and `basis_mom` have further shifts (raw lags two and
two-to-three respectively); this is a confirmed timing detail, not an
accuracy attribution. C6 must compare the current compatible definition with
any single-shift alternative on paired v4 rows, and must update the bundle
and research contract together.

### P2 — architecture and economic measurement

**H9: transformer inductive bias/capacity is not optimal for this horizon.**
The current model is a Transformer WM plus downstream actors. Simpler linear,
patch, inverted-variate, non-stationary, and multiscale models are credible
baselines, but architecture superiority is **UNVERIFIED** until the same folds,
targets, costs, and selection rules are used. C8 was stopped before execution
because the preceding Wave3C signal gate failed; no architecture comparison
was completed.

**H10: costs and initial-entry conventions can hide a small edge.** Research
and demo now document fee `0.0003`, full spread `3 bps` (half spread `1.5 bps`),
and slippage `1 bps` on absolute position delta. The Web implementation keeps
the B&H entry cost as fixed cash and explicitly documents that window return
excludes initial cost. Research turnover historically excludes synthetic
initial entry for compatibility, while a separate cost-turnover path is
required for cost calculations. These semantics are **CONFIRMED as current
contracts**, but historical cross-repo parity is **UNVERIFIED**. A numerical
fixture must assert both strategy and B&H initial-entry convention before any
small AlphaEx is interpreted.

## Canonical metric and timing contract for new experiments

All rows in the checklist below must state these quantities in machine-readable
metadata. If a caller cannot satisfy one, it records `N/A` and stops promotion.

| item | fixed contract |
| --- | --- |
| bar | BTCUSDT, 15m, right-exclusive intervals |
| feature timing | row `t` uses only information available by bar-`t` start; shifted indicators retain the `shift(1)` semantics |
| target | for horizon `h`, strictly future `t+1..t+h`; no target element may cross a split boundary |
| execution | fixed non-negative integer delay `d`; for `d>0`, evaluate `positions[:-d]` against `returns[d:]`; reject negative, non-integer, boolean, or `d >= len` |
| benchmark | B&H position `1.0`, trimmed to the same effective bars as the strategy |
| cost | fee `0.0003` + half-spread `0.00015` + slippage `0.0001`, multiplied by `abs(position_delta)` and quote price/notional according to the shared implementation |
| turnover | preserve legacy turnover for compatibility when it excludes synthetic entry; expose `cost_turnover = sum(abs(diff(concat([0], positions))))` separately and never use the names interchangeably |
| AlphaEx | strategy final total return minus B&H final total return, in percentage points; not annualized |
| MaxDDDelta | `abs(strategy MaxDD) - abs(B&H MaxDD)`; negative improves drawdown |
| return window | any “window return” must identify whether the initial entry cost is excluded; the Web contract excludes it from return while retaining it in cash/equity |
| selection | fit on train only; choose checkpoint/hyperparameters/policy/threshold on validation only; development test is report-only; folds 15+ are never consumed by development selection |
| statistical gate | development paths only; explicit full `n_trials`; block bootstrap and paired tests; odd/too-short CSCV inputs are N/A, never pass |

### Synthetic numerical assertions required before a candidate run

These are not candidate results. They are small contract fixtures that every
implementation may reproduce independently:

1. With initial capital `10000`, first price `100`, target moving from flat to
   `1.0`, one-way cost is `5.50` quote units under the fixed profile. At price
   `110`, the B&H equity is `10994.50` when the same flat-to-1.0 entry cost is
   applied as fixed cash: `10000 * 110/100 - 5.50`. The alternative
   `(10000-5.50) * 110/100` is a different contract and must not be mixed in.
2. A three-return path and delay `1` use decisions `positions[:-1]` with
   `returns[1:]`; no synthetic leading position is added. Delay `0` preserves
   all rows. Negative, boolean, fractional, and too-large delays reject.
3. Funding at publication times `t-1` and `t+1` must align to candle `t` with
   the `t-1` value only. A mark row at an exact candle timestamp is accepted;
   a later timestamp or missing exact row rejects.
4. A null or zero external value is not silently converted into an observed
   derivative. v4 sidecar flags must distinguish it, and an unresolved row
   makes a crossing sequence ineligible.

## Experiment checklist

The checklist is intentionally gate-oriented. “Pass” means the stated
criterion is met for the named artifact; it never means a positive trading
result is guaranteed. Every experiment must retain the input/config/source
hashes and a reason for every `N/A`.

| ID / owner | status now | input | metric and pass criterion | stop condition | artifact | dependency |
| --- | --- | --- | --- | --- | --- | --- |
| **C0 Data QA / Research** | **GATE STOPPED (v4 generated with explicit gaps)** | official Spot + USDⓈ-M mark/funding, v3 audit, generated v4 body/sidecar | exact 17 columns, complete sidecar, finite values, no unresolved window crossing; raw gap ledger totals 611 Spot bars, 80 recovered, 531 unresolved, with 81 off-grid rows quarantined; cannot pass full17 until sequence masks are consumed | non-official source, redirect, interpolation, bfill, unresolved required input, mixed digest | `2fc795c` final run record; final provenance/evidence `d06bfa4`; policy chain includes `7e0dfb9`; v4 parquet trio + metadata, gap/off-grid ledger, source response hashes (source digest `aa320222…`) | official Binance archives/REST; source reader at `157e408` |
| **C1 Closed-candle / Space + Edge** | **complete code/tests; deployed Space health/sample verified; latest provider path regional fail** | injected `now`, deterministic Spot/mark/funding fixtures, `TARGET_BARS`, public Space SHA prefix `60052bc5` | all bars close at or before cutoff; exact count; mark timestamp equality; funding publication `<=` candle; no current partial bar; `/health` and `/sample/verify` pass; latest `/predict/latest` fails closed on provider error | latest partial included, future funding, mark fill, pagination duplicate/gap, missing derivative, provider HTTP `451` treated as valid input | fixture tests; deployed `/health` and `/sample/verify` evidence; `/predict/latest` `502`/FAPI `451` observation | C0 source semantics; official Binance docs |
| **C2 Research timing / Eval** | **complete helper contract; Wave3C contract recorded** | positions, returns, benchmark paths, integer delay | d>0 uses `positions[:-d]` vs `returns[d:]` and same benchmark period; all metric arrays trimmed; strict type validation; pass synthetic boundary tests | p0 padding, negative/fractional/bool delay, cross-split target, `d>=len` | alignment JSON + numerical fixture report | C0; shared backtest/action_stats implementation |
| **C3 Web atomicity / Supabase** | **code/static tests complete; SQL/runtime unverified** | migration `0003`, RPC payloads, duplicate/stale/partial-failure fixtures | one RPC; row lock + CAS; unique `(run_id, latest_timestamp)` and trade key; duplicate idempotent; any error leaves no partial four-table write; pass local Postgres or explicit N/A | service secret exposure, public write grant, NaN/Inf accepted, partial row, duplicate trade | migration review, local SQL transcript or blocked report, Edge logs | Supabase CLI/Docker/local Postgres; no production writes |
| **C4 Cost/equity / Web + Research** | **contract/tests complete; historical parity unverified** | shared costs, flat/1.0/changed-position fixtures, B&H path | fee .0003, spread half 1.5bps, slippage 1bp times `abs(delta)`; strategy/B&H initial-entry symmetry; fixed-cash equity fixture; pass exact expected values | all-in cost labeled fee-only; initial cost double-counted/omitted; benchmark different window/unit | cost fixture JSON + metric contract report | C2; shared `paper_trading` and dashboard metrics |
| **C5 Target/forecast / Research** | **complete Wave3C report-only; candidate gate fail** | committed Wave3C dev artifact, folds 0/2/8, horizons 4/16/64, train/validation/development-test timestamps | targets exactly `t+1..t+h`; no split crossing; validation selects model/policy; development test only reports; Wave3C structural contract passes but all six economic gates fail | any future leakage, target overlap, fold15+ read, execution delay selected on test, baseline/padding/turnover mismatch | `c3b4ccb` forecast ledger/result/report; corrected replay/errata; values remain v3-cache evidence pending v4 rerun | C0, C2; corrected forecast implementation |
| **C6 Feature ablation / Research Data + Model** | **planned; blocked by C0** | paired v4 rows, full17, OHLCV13, explicit availability flags | compare AlphaEx, MaxDDDelta, IC/MAE, sign and turnover under same folds/cost/delay; pass only preregistered superiority with no missingness confound | unpaired rows, zero/missing ambiguity, different selector, holdout selection | ablation ledger + per-fold paths + mask report | C0, C5 |
| **C7 Exposure/timing / Research Eval** | **complete Wave3C report-only; gate fail; Wave3D constant baseline fail** | validation-selected constant, dynamic path, lag/shift nulls, common evaluation start | Wave3C records mean exposure, AlphaEx, timing increment, cost turnover, and common-window comparisons; all six timing medians are negative; Wave3D separately finds selected constant median AlphaEx `+1.5644pt` but is not a timing test because constant exposure has no temporal path | exposure difference explains result, period mismatch, delay mixed into selection, null beats dynamic | `c3b4ccb` attribution ledger/report; Wave3D `330c712` result/report/ledger | C2, C4, C5; v4 paired paths for any rerun |
| **C8 Architecture / Research Model** | **stopped before execution** | same v4 folds/targets and training budget for DLinear, PatchTST, iTransformer, Nonstationary Transformer, TimeMixer, current WM | no architecture result was run; resume only after a passing signal gate with split/cost parity and validation-only selection | architecture gets extra tuning/data, test-driven selection, no deployment parity | explicit stopped-run record; no checkpoints/results claimed | C0, C5; primary literature below |
| **C9 Offline-RL / Research RL** | **stopped before execution** | same behavior data/WM and action support; current BC/AC, CQL, IQL candidates | no offline-RL result was run; resume only after a passing signal gate with conservative OPE and action-support checks | unseen action mass, world-model disagreement unreported, holdout feedback, policy instability | explicit stopped-run record; no OPE/checkpoint/result claimed | C0, C2, C5; architecture results optional |
| **C10 Stress / Research Stats** | **complete Wave3D diagnostic; gate fail** | returns-only selected-constant development paths, folds 0–11; cost and regime cases | block-bootstrap primary/sensitivity lower bounds are negative; fold sign test is N/A (11 non-zero folds, need 12); cost stress passes `3/3`, regime stress `2/3`, so the required stress gate fails | missing path, future fold, omitted stress group, CI sensitivity fails, N/A treated as pass | Wave3D machine-readable result/report/ledger at `330c712`; no model significance claim | C0, C4, C5, C7; explicit `n_trials` |
| **C11 DSR/PBO / Research Stats** | **complete Wave3D diagnostic; gate fail** | six fixed exposure candidates plus selected-constant path, explicit `n_trials=7`, twelve even subperiods | DSR probability `0.8880 < 0.95` fails; CSCV/PBO `0.4524 <= 0.5` passes report-only; no model promotion follows | `n_trials` inferred from retained candidates, odd/insufficient CSCV, test/holdout consumed | Wave3D result/report/ledger at `330c712`; no holdout/deep/RL result | C10; candidate search registry |
| **C12 Cross-repo parity / Space + Web + Research** | **deployed Space contract/sample parity verified; latest provider path unavailable** | same candle bundle, 17 features, model/schema/source hashes; public Space SHA prefix `60052bc5` | `/health` exact 17-feature schema and `/sample/verify` `strict_ok=true` over `n=8641`; position max difference `1.1920929e-7`; live advantage max difference `1.531839e-5` is within diagnostic tolerance `2e-5` but strict flag is false; no claim beyond sample parity | schema/order/hash/timestamp mismatch, partial candle, dashboard status overclaims health, regional FAPI `451` | deployed endpoint evidence; signed replay and Web/Research same-candle parity remain required | C0, C1, C4; deployed revisions |
| **C13 Shadow/replay / Web Ops** | **N/A — no production authorization or run** | immutable provider responses, model hash, Edge logs, RPC rows | no production/shadow result is claimed; execution requires lead/operator authorization after C0–C12 gates | any unapproved provider/production access, live partial bar, 409 storm, RPC partiality, unmodeled fill/slippage, secret/log leak | no production artifact; authorization and redacted replay remain required | C1, C3, C12 |
| **C14 Wave3C/Wave3D freeze / Research lead + independent auditor** | **complete artifact audit; promotion stopped** | signed `c3b4ccb` and `330c712` config/data/source hashes, finite result JSONs/ledgers, dev-only fold paths | Wave3C six candidate gate rows fail and corrected replay is nine rows/zero failures; Wave3D constant diagnostic gate fails with `next_wave_candidates=[]`; result/ledger hashes match; no holdout/deep/RL promotion claim | any invented number, fold15+ selection, missing provenance, test result changes selector | signed Wave3C/Wave3D reports/ledgers, hashes, finite audits, errata | C0–C11; independent review |

### Checklist status interpretation

* **Complete/pass** means a contract or synthetic test passed; it is not a
  profitability claim.
* **Complete/fail** means the experiment ran and did not meet its preregistered
  gate (Wave3C is the current example); it must not be silently retried with
  changed thresholds until the change is documented as a new candidate.
* **Gate stopped** means downstream numerical comparison is not promoted until
  the blocking contract is repaired.
* **N/A** is a valid result with a reason, never a zero and never a pass.

## Primary-literature map and testable implications

The papers below provide hypotheses and statistical safeguards. None is
evidence that its method will improve BTCUSDT or UniDream. Each proposed model
must use the same repaired v4 data, WFO splits, target masks, execution delay,
costs, and validation-only selection.

| literature | primary source | implication for UniDream | what would count as evidence |
| --- | --- | --- | --- |
| DLinear / LTSF-Linear | [Zeng et al., *Are Transformers Effective for Time Series Forecasting?*](https://arxiv.org/abs/2205.13504) | a very simple linear baseline can beat complex Transformer forecasters on some benchmark datasets; test whether current capacity is unnecessary | same fold/horizon/feature contract; improvement in net dev IC and downstream AlphaEx without extra tuning; no claim from paper alone |
| PatchTST | [Nie et al., *A Time Series is Worth 64 Words*](https://arxiv.org/abs/2211.14730) | patching and channel-independent tokens may improve local semantics and attention efficiency for a 64-bar context | predeclare patch length/stride; compare compute and net metrics against current WM under identical data and selection |
| iTransformer | [Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*](https://arxiv.org/abs/2310.06625) | variate tokens may model cross-feature dependencies differently from time tokens, relevant to OHLCV + derivatives | same 17 columns, normalization, target mask, and training budget; measure whether cross-variate gains survive costs |
| Non-stationary Transformer | [Liu et al., *Non-stationary Transformers*](https://arxiv.org/abs/2205.14415) | reversible stationarization/de-stationary attention is a hypothesis for regime/level shifts; current rolling z-score may remove or retain different information | regime-stratified dev ablation with no future statistics; compare stability, not only best fold |
| TimeMixer | [Wang et al., *TimeMixer* (ICLR 2024)](https://arxiv.org/abs/2405.14616) | decomposable multiscale mixing may match 15m horizons spanning 4/16/64 bars | same multiscale target set and fixed compute; predeclare whether long-horizon improvements translate to net policy paths |
| Short-term Bitcoin market prediction | [Jaquart, Dann, and Weinhardt, *Short-term bitcoin market prediction via machine learning*](https://publikationen.bibliothek.kit.edu/1000150665) | a BTC-specific study reports classification above random while a very short-horizon strategy became negative after transaction costs; this supports separating IC/AUC/sign from net timing, without assuming transfer to UniDream | same BTC horizon, costs, and execution contract; require forecast and net economic gates independently, with no claim from the paper alone |
| CQL | [Kumar et al., *Conservative Q-Learning*](https://arxiv.org/abs/2006.04779) | conservative Q regularization is a candidate response to offline action-distribution shift and overestimated unseen actions | action-support and conservative OPE diagnostics plus development gate; no holdout tuning |
| IQL | [Kostrikov, Nair, and Levine, *Offline Reinforcement Learning with Implicit Q-Learning*](https://arxiv.org/abs/2110.06169) | expectile value learning and advantage-weighted behavior cloning avoid direct maximization over unseen actions | compare to BC/AC with identical behavior data and cost; report unsupported-action rate and stability |
| Deflated Sharpe Ratio | [Bailey and López de Prado, *The Deflated Sharpe Ratio* (author PDF)](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) and [SSRN version](https://doi.org/10.3905/jpm.2014.40.5.094) | multiple trials, non-normality, and sample length must deflate apparent Sharpe; annualization is display-only | explicit full trial registry, per-bar returns, deterministic DSR output; omitted trial count is non-promotion |
| Probability of Backtest Overfitting | [Bailey et al., *The Probability of Backtest Overfitting* (author PDF)](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) | CSCV/PBO is a guard against selecting a lucky in-sample combination; odd/insufficient subperiods are N/A | at least two candidates and a preregistered even subperiod count (for example 12); no 13-fold shortcut or holdout use |

## Agent organization and hand-off plan

The plan assigns one owner per file family and an independent audit role. The
parent/lead arbitrates contracts and claims; no agent may broaden its scope by
editing another owner’s files or by deploying production state.

| role | owned surface | deliverable and acceptance |
| --- | --- | --- |
| **Lead / contract arbiter (parent)** | cross-repo decisions, release claims, gate ledger | approves fixed metric/timing/schema contract; records status and prevents unverified or Wave3C numbers from promotion |
| **Research Data QA** | `UniDream/unidream/data`, v4 tests, data-quality docs | official-only source reconstruction, sidecar/gap ledger, exact digests, no v3 overwrite; C0 pass artifact |
| **Research Evaluation/Stats** | `unidream/eval/backtest.py`, attribution, statistical gate and dedicated tests | shared delay/cost alignment, timing/common-window report, DSR/PBO/block bootstrap with full trial registry; no holdout reads |
| **Research Forecast/Architecture** | forecast tournament/context files and architecture experiments | target mask/split proof, DLinear/PatchTST/iTransformer/Nonstationary/TimeMixer candidates; validation-only selection; report-only development test |
| **Research RL** | WM/BC/AC experiment configs and RL-specific tests | current baseline versus CQL/IQL or conservative diagnostics; action support and model disagreement evidence |
| **Space owner** | `unidream-space/backend`, bundle verification and tests | 17-feature raw contract, closed candle, derivative as-of join, parity digest; runtime/deployment observation separately labeled |
| **Web Edge/Supabase owner** | `unidream-demo-web/supabase/functions`, migration, backfill tests | Binance closed/derivative fetch, one RPC/CAS, exact costs, no public write privilege; local SQL evidence or explicit boundary |
| **Dashboard/metrics owner** | Web `src/lib/metrics.ts`, contract/UI tests | truthful AlphaEx/MaxDD/B&H/return windows, cost units, timestamps, model/schema/parity/cutoff/atomic status; no source-configured badge presented as live health |
| **Independent auditor** | read-only review across all three repositories | verifies fold scope, target offsets, timing, parity, hashes, and claim wording; must not edit a concurrently owned file |
| **Ops/replay owner** | redacted provider/RPC logs, shadow replay only | validates latency/error/fill observability after code gates; never accesses production secrets in Research experiments |

### Sequencing and branch discipline

```text
C0 v4/data gate
  -> C1/C2 closed + timing fixtures
  -> C3/C4 atomic + cost/equity verification
  -> C5 target/forecast structural audit
  -> C6/C7 ablation and mean-matched timing
  -> C8/C9 architecture and offline-RL candidates
  -> C10/C11 independent statistical gate
  -> C12 same-candle parity
  -> C13 shadow replay
  -> lead sign-off / optional deployment
```

Each hand-off must include: commit hash, config hash, data/source/bundle
digests, fold and timestamp range, target/delay/cost contract, per-fold paths,
tests and counts, gate output, `N/A` reasons, and unverified boundaries. Use
small commits with exact-file staging; never force-push, rewrite a concurrent
branch, or stage another owner’s untracked work. A failed gate produces a
diagnostic artifact and stops downstream promotion.

## Promotion, rollback, and reporting decision tree

1. **Data:** if C0 fails, stop all accuracy comparisons and report the gap and
   source failure. Do not interpolate, backfill from future data, or call a
   v3 cache v4.
2. **Causality:** if C1/C2/C5 fails, stop model selection. Fix the timestamp,
   target, or execution contract and regenerate the affected artifact.
3. **Economics/state:** if C3/C4 fails, stop dashboard or paper-trading
   promotion. Keep production untouched; use a local SQL/synthetic fixture and
   record whether DB execution remains unavailable.
4. **Model:** if a candidate fails its predeclared development gate, retain it
   as a failed candidate. Do not change the gate after reading its test or
   holdout result.
5. **Statistics:** if DSR/PBO, block CI, sign, or stress is N/A/failed, the
   result may be described as exploratory only. A synthetic fixture passing the
   API is not candidate evidence.
6. **Runtime:** only after all data/model gates pass may a redacted shadow
   replay be considered. Any schema/hash/cutoff/RPC mismatch pauses Cron and
   returns to the last compatible Edge revision; never drop the migration or
   uniqueness keys while a revision may still run.
7. **Claims:** report development, untouched holdout, live shadow, and
   production observations in separate sections. Wave3C values in this report
   come only from the signed `c3b4ccb` artifact; any future rerun must replace
   them only with a new complete-provenance artifact.

## Known limitations and open questions

* The controlled official v4 run generated a body and full-grid availability
  sidecar, but it still has 531 unresolved Spot bars and external availability
  gaps. The current full17 loader cannot propagate the sidecar into sequence
  eligibility, so no corrected-v4 model result exists; a complete
  availability-aware training run remains required.
* The public Space deployment was observed at SHA prefix `60052bc5`, stage
  `RUNNING`, one cpu-basic replica; `/health` and `/sample/verify` passed the
  deployed 17-feature/sample contract (`n=8641`, position max difference
  `1.1920929e-7`). The live advantage max difference was `1.531839e-5`, within
  diagnostic tolerance `2e-5`, but strict parity was false. `/predict/latest`
  returned `502` because official FAPI funding/mark calls returned HTTP `451`
  from that region. This verifies a deployed sample boundary, not a
  provider-backed latest result, Web execution, Supabase Cron, or a production
  Postgres transaction.
* Research historical result snapshots use the prior v3 data contract. Their
  exact numbers are retained for honesty but are not a clean estimate under
  the future v4 contract.
* CSCV/PBO over the current 13 development folds is intentionally N/A because
  subperiods must be even. Wave3D used a separate, explicit twelve-subperiod
  constant-exposure diagnostic, but that report-only result does not upgrade
  the three-fold Wave3C model screen. A future formal gate should preregister
  an even count, preserve all trial candidates, and record the full search
  count for DSR.
* Wave3C is a three-fold development screen on the prior v3 cache, not a
  corrected-v4 retraining or holdout result. Its candidate gate is complete
  and failed; its statistical gate remains unapplied. Wave3D is complete as a
  returns-only constant-exposure diagnostic but its overall gate failed, with
  no timing increment or model-promotion claim.

## Current report completion state

| stage | status | evidence |
| --- | --- | --- |
| Stage 1 skeleton + confirmed evidence | **complete/pushed** | commit `66d6298`, now an ancestor of the current `origin/main` |
| Stage 2 hypotheses + checklist + literature + agent plan | **complete/pushed** | commit `8737581`; this report retains the contract and updates its evidence state |
| Official v4 source rebuild audit | **generated artifact / C0 stopped** | `2fc795c` run record, final provenance/evidence `d06bfa4`; 17-column body plus full-grid sidecar, raw gap ledger 611 bars (80 recovered, 531 unresolved), 81 off-grid quarantined; no model run |
| Wave3C numerical outcome | **complete artifact / promotion stopped** | `c3b4ccb`; six candidate rows all fail, corrected replay is nine rows with zero failures, `next_wave_candidates=[]` |
| Wave3D constant-exposure diagnostic | **complete artifact / gate failed** | `330c712` published from source `f6d273b`; selected median AlphaEx `+1.5644pt`, mean `+0.8584pt`, `6/12` positive, median MaxDDDelta `-4.1566pt`; DSR/CI/stress gates fail; `next_wave_candidates=[]`; no timing/model claim |
| C8 deep architecture / C9 offline RL | **stopped before execution** | upstream Wave3C signal gate failed; no completion or result is claimed |
| Candidate statistical promotion | **blocked / no model candidate applied** | Wave3D diagnostic applied CI/DSR/PBO/stress to a returns-only constant path and failed overall; Wave3C model paths remain unapplied; no model significance or promotion claim |

## Official and primary references consulted

Operational contracts were checked against the current official pages below.
The Supabase changelog was fetched on 2026-08-30; its recent logs API breaking
change is relevant to operators using the Management API, but no production
Supabase operation was performed for this report.

* [Binance Spot Kline/Candlestick data](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)
* [Binance USDⓈ-M Kline/Candlestick data](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data)
* [Binance USDⓈ-M Mark Price Kline/Candlestick data](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price-Kline-Candlestick-Data)
* [Binance USDⓈ-M Funding Rate History](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)
* [Binance USDⓈ-M general information and limits](https://developers.binance.com/docs/derivatives/usds-margined-futures/general-info)
* [Supabase changelog](https://supabase.com/changelog.md)
* [Supabase Edge Functions](https://supabase.com/docs/guides/functions)
* [Supabase JavaScript RPC](https://supabase.com/docs/reference/javascript/rpc)
* [Supabase Row Level Security](https://supabase.com/docs/guides/database/postgres/row-level-security)

The primary model/statistical references are listed in the literature table
above. Links are included to author-hosted PDFs or the paper landing pages so
that the claims can be checked without relying on model memory.
