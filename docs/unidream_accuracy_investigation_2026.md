# UniDream accuracy investigation (2026)

**Report state:** Stage 1 — skeleton and confirmed evidence. This is a
report-only artifact; it does not retrain a model, apply a Supabase migration,
deploy a Space, or write production data.

**Evidence cutoff:** 2026-08-30. The Research, Space, and Web repositories are
separate repositories. The snapshots inspected for this report were:

| repository | inspected revision | role |
| --- | --- | --- |
| `UniDream` | `a871439` | Research training/evaluation and contract tests |
| `unidream-space` | `60052bc` | Plan011 v31 inference service and feature parity |
| `unidream-demo-web` | `271baf7` | Edge data fetch, paper trading, Supabase writes, dashboard |

The Research working tree also contains concurrent, uncommitted forecast and
backtest edits. They were read only and are intentionally not part of this
report commit.

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

Wave3C is not a numerical source in this report. Every Wave3C value remains an
explicit placeholder such as `[WAVE3C_UNCONFIRMED_ALPHAEX_MEAN_PT]`; no value
from Wave3A, the historical v31 snapshot, or a future run may be substituted.

## Executive evidence summary

The current evidence supports a pipeline-contract and out-of-sample
generalization investigation, not a single identified model defect.

1. **Research data quality is a stopped gate.** The development data-quality
   audit for `[2018-01-01, 2024-01-01)` and 15-minute bars reports an overall
   `FAIL`: 29 non-15m steps and an estimated 542 missing bars. The v3 cache
   has no availability mask, so a zero derivative value cannot be
   distinguished from missing or imputed data. Causal probes pass, but this
   does not make the cache v4-compliant.
2. **The model result is modest and not drawdown-improving in the retained
   snapshot.** Development folds 0–12 have mean AlphaEx `+0.41pt`, median
   `+0.16pt`, and MaxDDDelta `+0.20pt`; the 2024–2026 holdout folds 15–23 have
   mean AlphaEx `+0.11pt`, median `-0.04pt`, and MaxDDDelta `+0.20pt`. Negative
   MaxDDDelta is the improvement direction. These are historical evidence
   snapshots, not a new v4 rerun.
3. **The Wave3A forecast screen found no winner.** Its six feature-set ×
   candidate combinations all failed their preregistered gate. This is a
   development screen, not Wave3C, and it does not prove that derivative
   features or a particular architecture caused the decline.
4. **The live contracts are stricter than the old research cache.** Space
   requires all 17 Plan011 inputs and rejects missing/non-finite funding or
   mark data. Web excludes partial candles, joins funding as-of and mark by
   candle timestamp, and commits prediction/state/snapshot/trade through one
   CAS-protected RPC. Code/tests establish these contracts; provider reachability,
   deployed revision, and production database execution remain unverified.
5. **Statistical promotion is intentionally not claimed.** The development-only
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

### Wave3A and Wave3C status

**CONFIRMED — Wave3A development screen.**
`docs/forecast_tournament_plan011_dev/report.md` covers development folds
0, 2, and 8, horizons 4/16/64, fixed operational delay 1 bar, and sensitivity
lags 1/16. The screen tested full17 and OHLCV13 with a causal trend/vol rule,
Ridge, and HistGradientBoosting. All six combinations failed. Examples from
the retained aggregate table are full17 causal `IC -0.0329`, AlphaEx
`-1.3651pt`; full17 HistGB `IC +0.0448`, AlphaEx `-0.7256pt`; and full17 Ridge
`IC +0.0187`, AlphaEx `+0.8335pt`, which still failed the complete gate.

These numbers are Wave3A only. They are not Wave3C results and cannot fill
the placeholders below.

**IN PROGRESS / GATE STOPPED — Wave3C result freeze.** The exact Wave3C
selection, timing, cost, and statistical values are not confirmed in the
current evidence set:

```text
WAVE3C_ALPHAEX_MEAN_PT                 = [UNCONFIRMED]
WAVE3C_ALPHAEX_MEDIAN_PT               = [UNCONFIRMED]
WAVE3C_TIMING_INCREMENT_MEAN_PT        = [UNCONFIRMED]
WAVE3C_CONSTANT_MEAN_EXPOSURE          = [UNCONFIRMED]
WAVE3C_DYNAMIC_MEAN_EXPOSURE           = [UNCONFIRMED]
WAVE3C_COST_TURNOVER                   = [UNCONFIRMED]
WAVE3C_MAXDD_DELTA_PT                  = [UNCONFIRMED]
WAVE3C_STATISTICAL_GATE                = [UNCONFIRMED]
```

Until a signed artifact records the config/data/source hashes, fold scope,
selection split, execution delay, and exact paths, Wave3C has no pass/fail
accuracy claim.

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
| Plan011 historical evidence | **complete snapshot** | dev/holdout aggregates and report-only semantics | re-run only after data contract is repaired |
| Wave3A forecast tournament | **complete screen/fail** | all six tested combinations failed | Wave3C artifact and any revised screen |
| Wave3C | **IN PROGRESS / stopped** | placeholders only | signed exact numerical result |
| Space 17-feature parity | **complete code/tests** | derivative inputs required; causal closed path | deployed provider/runtime observation |
| Web closed-candle/atomic/cost | **complete code/tests** | closed data, derivative join, one RPC, cost contract | local SQL and production runtime verification |
| Statistical gate | **complete fixture only** | independent API and fail-closed contract | candidate dev paths, explicit full trial count |

## What this evidence does not prove

It does not prove that funding/mark features improve or harm accuracy, that a
Transformer is inferior to a simpler model on BTCUSDT, that AC caused the
holdout shortfall, or that any Wave3C candidate passes. It does not prove live
profitability, execution capacity, actual exchange fills, Supabase atomicity
under concurrent production requests, or statistical significance. Those are
questions for the controlled checklist in the next stage of this report.

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

---

The hypotheses, primary-literature map, executable experiment checklist,
agent organization plan, and promotion/rollback decision tree are added in
Stage 2 after this evidence-only commit is pushed.
