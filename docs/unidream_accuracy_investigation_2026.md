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
| `UniDream` | `8e9e0fc` for the Wave3C run; artifact refresh `c3b4ccb` (baseline evidence begins at `a871439`) | Research training/evaluation, official-source readers, and contract tests |
| `unidream-space` | `60052bc` | Plan011 v31 inference service and feature parity |
| `unidream-demo-web` | `271baf7` | Edge data fetch, paper trading, Supabase writes, dashboard |

The Research working tree also contains concurrent, uncommitted forecast,
backtest, and source-rebuild edits. They were read only and are intentionally
not part of this report commit. Revision `157e408` adds official-source reader
and probe code; it does not constitute a generated v4 cache or a new model
result. Wave3C is audited from the latest committed artifact at `c3b4ccb`; its
underlying tournament code/data commit is recorded in the Wave3C section below.

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
* **IN PROGRESS** — implementation or review exists, but the required
  numerical evidence is not yet complete.

Wave3C is now a numerical source for a narrowly scoped, development-only
report. Its values are not holdout or production accuracy, and `status:
complete` means the artifact was generated—not that a candidate passed
promotion. The separate Wave3D low-frequency exposure diagnostic remains
unconfirmed and is intentionally not inferred from Wave3C.

## Executive evidence summary

The current evidence supports a pipeline-contract and out-of-sample
generalization investigation, not a single identified model defect.

1. **Research data quality is a stopped gate.** The development data-quality
   audit for `[2018-01-01, 2024-01-01)` and 15-minute bars reports an overall
   `FAIL`: 29 non-15m steps and an estimated 542 missing bars. The v3 cache
   has no availability mask, so a zero derivative value cannot be
   distinguished from missing or imputed data. Causal probes pass, but this
   does not make the cache v4-compliant.
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
   report-only; full17 is blocked by the missing availability mask. The
   corrected Wave3A replay confirms the same interpretation.
4. **The live contracts are stricter than the old research cache.** Space
   requires all 17 Plan011 inputs and rejects missing/non-finite funding or
   mark data. Web excludes partial candles, joins funding as-of and mark by
   candle timestamp, and commits prediction/state/snapshot/trade through one
   CAS-protected RPC. Code/tests establish these contracts; provider reachability,
   deployed revision, and production database execution remain unverified.
5. **Deep architecture and offline-RL follow-ups are gate-stopped.** C8 deep
   architecture and C9 offline-RL experiments were not run after the Wave3C
   signal gate failed; they are not completed results. C10/C11 statistical
   promotion is also intentionally not claimed. The development-only
   statistical gate accepts explicit per-bar paths and has moving/stationary
   block bootstrap, DSR with an explicit trial count, CSCV/PBO, sign tests, and
   cost/regime stress. It currently has synthetic API tests only and has not
   been applied to a candidate result.

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

**GATE STOPPED — current development cache.**
`docs/data_quality_gate_2018_2024.md` records:

* scope `[2018-01-01, 2024-01-01)`, 15m, overall `FAIL`;
* cache contract `fail`, causal probes `pass`, OHLCV13/full17 same-row
  eligibility `pass`, external availability mask `fail`;
* 29 non-15m steps and 542 estimated missing bars; non-finite count `0`;
* 18 of 542 expected timestamps covered by official gap recovery and 524
  unresolved; no interpolation or in-place v3 repair;
* OHLCV13 and full17 eligible rows both `208299`, with the same row mask.

The v4 writer/loader and gap policy are implemented and tested as a contract,
but `docs/data_quality_cache_v4.md` explicitly says that no v4 dataset has
yet been generated from the current cache. The existing v3 model cache must
not be described as v4-compliant.

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
| full17 | Ridge risk-adjusted context | `+0.053675 IC` | `+20.076113` | `+20.167520` | `-0.091407` | `-8.384446` | `9.682861` | **blocked: no availability mask** |
| full17 | HistGB risk-adjusted context | `+0.061134 IC` | `+19.780975` | `+20.167520` | `-0.386545` | `-8.064934` | `5.467703` | **blocked: no availability mask** |
| full17 | downside classifier | `0.626428 AUC` | `+19.714931` | `+20.167520` | `-0.452590` | `-8.313140` | `2.991194` | **blocked: no availability mask** |

The interpretation is decisive for this screen: the apparently roughly
`+20pt` AlphaEx is the validation-selected constant exposure component, not a
dynamic timing improvement. The selected constant exposures are fold `0`
`0.5`, fold `2` `1.12`, and fold `8` `0.5` (`0.5 / 1.12 / 0.5`) for every
candidate. All six dynamic-minus-constant timing medians are negative. OHLCV13
is the formal promotion feature set for the artifact, but every OHLCV13
candidate still fails its signal/timing gate. Full17 is additionally blocked
because the v3 cache has no availability mask to distinguish external zero
from missing/imputed derivative values; this is not evidence that full17 is
accurate or inaccurate.

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

**Wave3D low-frequency exposure diagnostic — IN PROGRESS / UNCONFIRMED.** No
committed Wave3D result was part of this audit. The low-frequency dynamic
AlphaEx, timing increment, exposure distribution, turnover, and MaxDD values
remain `[WAVE3D_UNCONFIRMED]`; no Wave3C value is substituted for them. This
diagnostic must use the shared delay/cost contract and a common evaluation
window before it can change the interpretation above.

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
inputs. `/health`, `/sample/verify`, and bundle metadata expose schema/digest
information, but they do not prove that a future provider call or deployed
revision is healthy.

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
modified or exercised for this report. Local SQL execution, provider
reachability, Cron scheduling, HF Space latency, and real fill/slippage are
external runtime boundaries unless separately recorded by an operator.

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

The repository contains synthetic unit tests for the API. It contains no
candidate input result, and the gate has not been used to manufacture a
significance claim for Plan011.

## Gate status at this cutoff

| gate | status | what is established | what remains |
| --- | --- | --- | --- |
| Gate0 WM validation/action context | **complete/pass** | ordered full validation evaluation, coherent best-state restore, and Plan011 actionless WM context are tested | a new model run on v4 data |
| Research causal feature probes | **complete/pass** | four perturbation/prefix/as-of probes pass with max diff 0 | availability and gap-complete data |
| Research v3 data quality | **GATE STOPPED/fail** | audit records gaps and zero/missing ambiguity | official v4 regeneration and window exclusion |
| v4 cache contract | **complete contract / no dataset** | writer, loader, sidecar, digest, gap policy | generate and audit the actual v4 cache |
| Plan011 historical evidence | **complete pre-correction snapshot / not current-accuracy evidence** | dev/holdout aggregates and report-only semantics before `c8f6914`, on v3 | corrected retraining on v4; do not compare as current Plan011 |
| Wave3A forecast tournament | **frozen pre-errata diagnostic; corrected replay complete** | original six-combination screen is historical; nine-row corrected replay records the common-window/constant-baseline interpretation | no promotion from the historical screen; v4 rerun if a new candidate is justified |
| Wave3C | **complete artifact / promotion stopped** | six development candidates ran on folds 0/2/8; all failed the signal/timing gate; `next_wave_candidates=[]`; full17 is data-quality blocked | corrected v4 data and a new preregistered run; no holdout or production claim |
| Wave3D low-frequency exposure | **IN PROGRESS / unconfirmed** | no committed result audited | common-window, cost-aligned diagnostic and independent report |
| Space 17-feature parity | **complete code/tests** | derivative inputs required; causal closed path | deployed provider/runtime observation |
| Web closed-candle/atomic/cost | **complete code/tests** | closed data, derivative join, one RPC, cost contract | local SQL and production runtime verification |
| Statistical gate | **complete fixture only** | independent API and fail-closed contract | candidate dev paths, explicit full trial count |

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
* Result snapshots: [`docs/plan011_v31_investor_evidence.md`](plan011_v31_investor_evidence.md),
  [`docs/alpha_attribution_plan011_v31_dev/report.md`](alpha_attribution_plan011_v31_dev/report.md),
  and [`docs/forecast_tournament_plan011_dev/report.md`](forecast_tournament_plan011_dev/report.md).
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
of availability flags are confirmed. It is an **INFERENCE**, not proof that
this caused the historical AlphaEx decline: the committed results were
computed on that cache, but no paired v3-versus-regenerated-v4 experiment
exists. Repairing this first is mandatory because every later model comparison
could otherwise measure a data-generation change.

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
low-frequency/common-window diagnostic remains unconfirmed and must report
mean exposure, paired block intervals, and cost turnover separately.

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
Wave3C full17 rows are blocked because the cache lacks an availability mask;
the OHLCV13 rows fail the timing/economic gate. This is not a causal derivative
ablation because the data-quality contract is not v4-clean. The only safe
statement is that derivative usefulness is **UNVERIFIED**. In addition,
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
| **C0 Data QA / Research** | **GATE STOPPED** | official Spot + USDⓈ-M mark/funding, v3 audit, v4 writer/reader | exact 17 columns, complete sidecar, finite values, no unresolved window crossing; pass only after official v4 regeneration and audit | non-official source, redirect, interpolation, bfill, unresolved required input, mixed digest | v4 parquet trio + metadata, gap ledger, source response hashes | official Binance archives/REST; source reader at `157e408` |
| **C1 Closed-candle / Space + Edge** | **complete code/tests; runtime unverified** | injected `now`, deterministic Spot/mark/funding fixtures, `TARGET_BARS` | all bars close at or before cutoff; exact count; mark timestamp equality; funding publication `<=` candle; no current partial bar; pass unit suite | latest partial included, future funding, mark fill, pagination duplicate/gap, missing derivative | fixture test output and input digest; `/health`/parity evidence | C0 source semantics; official Binance docs |
| **C2 Research timing / Eval** | **complete helper contract; Wave3C contract recorded** | positions, returns, benchmark paths, integer delay | d>0 uses `positions[:-d]` vs `returns[d:]` and same benchmark period; all metric arrays trimmed; strict type validation; pass synthetic boundary tests | p0 padding, negative/fractional/bool delay, cross-split target, `d>=len` | alignment JSON + numerical fixture report | C0; shared backtest/action_stats implementation |
| **C3 Web atomicity / Supabase** | **code/static tests complete; SQL/runtime unverified** | migration `0003`, RPC payloads, duplicate/stale/partial-failure fixtures | one RPC; row lock + CAS; unique `(run_id, latest_timestamp)` and trade key; duplicate idempotent; any error leaves no partial four-table write; pass local Postgres or explicit N/A | service secret exposure, public write grant, NaN/Inf accepted, partial row, duplicate trade | migration review, local SQL transcript or blocked report, Edge logs | Supabase CLI/Docker/local Postgres; no production writes |
| **C4 Cost/equity / Web + Research** | **contract/tests complete; historical parity unverified** | shared costs, flat/1.0/changed-position fixtures, B&H path | fee .0003, spread half 1.5bps, slippage 1bp times `abs(delta)`; strategy/B&H initial-entry symmetry; fixed-cash equity fixture; pass exact expected values | all-in cost labeled fee-only; initial cost double-counted/omitted; benchmark different window/unit | cost fixture JSON + metric contract report | C2; shared `paper_trading` and dashboard metrics |
| **C5 Target/forecast / Research** | **complete Wave3C report-only; candidate gate fail** | committed Wave3C dev artifact, folds 0/2/8, horizons 4/16/64, train/validation/development-test timestamps | targets exactly `t+1..t+h`; no split crossing; validation selects model/policy; development test only reports; Wave3C structural contract passes but all six economic gates fail | any future leakage, target overlap, fold15+ read, execution delay selected on test, baseline/padding/turnover mismatch | `c3b4ccb` forecast ledger/result/report; corrected replay/errata; values remain v3-cache evidence pending v4 rerun | C0, C2; corrected forecast implementation |
| **C6 Feature ablation / Research Data + Model** | **planned; blocked by C0** | paired v4 rows, full17, OHLCV13, explicit availability flags | compare AlphaEx, MaxDDDelta, IC/MAE, sign and turnover under same folds/cost/delay; pass only preregistered superiority with no missingness confound | unpaired rows, zero/missing ambiguity, different selector, holdout selection | ablation ledger + per-fold paths + mask report | C0, C5 |
| **C7 Exposure/timing / Research Eval** | **complete Wave3C report-only; gate fail; Wave3D pending** | validation-selected constant, dynamic path, lag/shift nulls, common evaluation start | Wave3C records mean exposure, AlphaEx, timing increment, cost turnover, and common-window comparisons; all six timing medians are negative, so the promotion criterion fails; paired block CI remains unrun | exposure difference explains result, period mismatch, delay mixed into selection, null beats dynamic | `c3b4ccb` attribution ledger/report; separate Wave3D diagnostic | C2, C4, C5; v4 paired paths for any rerun |
| **C8 Architecture / Research Model** | **stopped before execution** | same v4 folds/targets and training budget for DLinear, PatchTST, iTransformer, Nonstationary Transformer, TimeMixer, current WM | no architecture result was run; resume only after a passing signal gate with split/cost parity and validation-only selection | architecture gets extra tuning/data, test-driven selection, no deployment parity | explicit stopped-run record; no checkpoints/results claimed | C0, C5; primary literature below |
| **C9 Offline-RL / Research RL** | **stopped before execution** | same behavior data/WM and action support; current BC/AC, CQL, IQL candidates | no offline-RL result was run; resume only after a passing signal gate with conservative OPE and action-support checks | unseen action mass, world-model disagreement unreported, holdout feedback, policy instability | explicit stopped-run record; no OPE/checkpoint/result claimed | C0, C2, C5; architecture results optional |
| **C10 Stress / Research Stats** | **statistical API complete; candidate application blocked** | development fold paths only, cost and regime cases | moving/stationary block CI with fixed/sensitivity blocks; sign/binomial; all required cost/regime stress pass; N/A rejects | missing path, future fold, omitted stress group, CI sensitivity fails, N/A treated as pass | machine-readable statistical gate output | C0, C4, C5, C7; explicit `n_trials` |
| **C11 DSR/PBO / Research Stats** | **contract complete; no result** | all tried candidate paths/count, even pre-registered subperiods | DSR with per-bar/annualized distinction and complete `n_trials`; CSCV/PBO with >=2 candidates and even >=4 subperiods; pass only if preregistered | `n_trials` inferred from retained candidates, 13-fold odd CSCV, test/holdout consumed | gate JSON + trial registry + PBO reason | C10; candidate search registry |
| **C12 Cross-repo parity / Space + Web + Research** | **code fixtures complete; live unverified** | same candle bundle, 17 features, model/schema/source hashes | max feature/position discrepancy within declared tolerance; same latest closed timestamp and cost metadata; pass signed replay | schema/order/hash/timestamp mismatch, partial candle, dashboard status overclaims health | parity bundle/report and status payload | C0, C1, C4; deployed revisions |
| **C13 Shadow/replay / Web Ops** | **planned; no production run** | immutable provider responses, model hash, Edge logs, RPC rows | no missing/duplicate bar; latency, error class, fills, equity/B&H, and cost audit; pass only after operator sign-off | live partial bar, 409 storm, RPC partiality, unmodeled fill/slippage, secret/log leak | redacted replay + observability dashboard | C1, C3, C12 |
| **C14 Wave3C freeze / Research lead + independent auditor** | **complete artifact audit; promotion stopped** | signed `c3b4ccb` config/data/source hashes, finite result JSON, 376-line finite ledger, dev-only fold paths | all Wave3C placeholders replaced from the reproducible artifact; six candidate gate rows fail; corrected replay is nine rows/zero failures; no statistical candidate gate claimed | any invented number, fold15+ selection, missing provenance, test result changes selector | signed Wave3C report/ledger, hashes, finite audit, errata | C0–C11; independent review |

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

* The v4 source readers and contract tests do not by themselves generate or
  validate a complete historical v4 feature cache. Provider availability,
  archive semantics, and all 2018–2024 rows still require a controlled run.
* Space and Edge tests are local code-level evidence. This report did not
  invoke a deployed HF Space, Binance from the production region, Supabase
  Cron, or a production Postgres transaction.
* Research historical result snapshots use the prior v3 data contract. Their
  exact numbers are retained for honesty but are not a clean estimate under
  the future v4 contract.
* CSCV/PBO over the current 13 development folds is intentionally N/A because
  subperiods must be even. A future formal gate should preregister an even
  count (for example 12), preserve all trial candidates, and record the full
  search count for DSR.
* Wave3C is a three-fold development screen on the prior v3 cache, not a
  corrected-v4 retraining or holdout result. Its candidate gate is complete
  and failed; its statistical gate remains unapplied. The separate Wave3D
  low-frequency diagnostic is still `[WAVE3D_UNCONFIRMED]`.

## Current report completion state

| stage | status | evidence |
| --- | --- | --- |
| Stage 1 skeleton + confirmed evidence | **complete/pushed** | commit `66d6298`, now an ancestor of the current `origin/main` |
| Stage 2 hypotheses + checklist + literature + agent plan | **complete/pushed** | commit `8737581`; this report retains the contract and updates its evidence state |
| Wave3C numerical outcome | **complete artifact / promotion stopped** | `c3b4ccb`; six candidate rows all fail, corrected replay is nine rows with zero failures, `next_wave_candidates=[]` |
| C8 deep architecture / C9 offline RL | **stopped before execution** | upstream Wave3C signal gate failed; no completion or result is claimed |
| Candidate statistical promotion | **not started** | statistical API/tests exist, but no candidate input is supplied |

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
