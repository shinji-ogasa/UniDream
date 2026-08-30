# Predictable Oracle hypothesis registry

**Registry opened:** 2026-08-30

**Base revision:** `8e95cda`

**Scope:** hypotheses in `predictable_data_conditional_oracle_roadmap_2026.md`; BTCUSDT 15m first

**Selection rule:** validation selects, development test reports once, future temporal holdout remains sealed

This registry defines what “all hypotheses” means for the current research program. A hypothesis is not marked passed because code runs or one metric improves. Each row must produce the named artifact, use the same eligible rows and execution contract as its parent, and satisfy its gate without reading a report-only test to choose the next candidate.

## Status vocabulary

| status | meaning |
| --- | --- |
| `confirmed-defect` | current implementation violates the new contract; evidence already exists |
| `running` | an isolated branch is implementing or testing the preregistered check |
| `ready` | prerequisites pass and the experiment manifest is frozen |
| `not-started` | testable later, but an earlier gate has not passed |
| `blocked-data` | required point-in-time data is not locally available or historically verifiable |
| `failed` | preregistered gate ran and did not pass |
| `passed` | preregistered gate ran and passed; exact artifact and commit must be linked |

## P0 — correctness hypotheses

| id | hypothesis / falsifier | required evidence | current status |
| --- | --- | --- | --- |
| P0-A1 | v4 availability can reach row and sequence eligibility without changing the canonical 17 model columns. Falsified if unavailable rows enter a training window, timestamps are compacted, or `obs_dim != 17`. | sidecar/window fixtures, full test suite, P0-A report | `running` |
| P0-A2 | missing external data remains distinguishable from an observed numeric zero. Falsified by fill/backfill or a missing/non-boolean mask being accepted. | zero-vs-missing and fail-closed fixtures | `running` |
| P0-B1 | every enabled WM head has nonzero valid-target and gradient coverage. Current h64 and position-utility-h64 are expected to falsify the legacy configuration, not to report low accuracy. | per-head machine-readable coverage artifact | `confirmed-defect`; remediation `running` |
| P0-B2 | a train row's future label cannot influence its own predictive state, normalizer, calibrator, Q, teacher weight or action. | chronological OOF perturbation test with purge | `confirmed-defect`; remediation `running` |
| P0-B3 | current inventory comes from benchmark/policy replay rather than a hindsight teacher path. | inventory provenance guard and trajectory fixture | `confirmed-defect`; remediation `running` |
| P0-C1 | target, U0, O1-O3 and Backtest share `decision t -> fill t+1 -> returns t+1..t+4`, four-bar commitment and identical incomplete-tail exclusion. | hand-calculated trajectory fixture and contract hash | `confirmed-defect`; remediation `running` |
| P0-C2 | the new path uses one explicit cost/initial-state contract and cannot fall back to legacy `5/2/0.0004`, delay 0 or flat start. | fail-closed config tests and cost fixture | `confirmed-defect`; remediation `running` |

P0 passes only if every P0 row is `passed`. A partial implementation or explicit blocker is useful evidence but does not unlock market-accuracy experiments.

## P1 — controlled recoverability hypotheses

| id | hypothesis | fixed comparison | promotion evidence | current status |
| --- | --- | --- | --- | --- |
| P1-S0 | zero-signal data does not create a promoted timing edge | unconditional, persistence, Ridge/logistic; cost off/on | trial ledger shows no candidate whose preregistered paired/block interval clears zero after multiplicity adjustment | `not-started` |
| P1-S1 | a high-SNR causal state is recoverable under the full timing/action contract | known DGP, horizons 1/4/8/16, ten fixed seeds | at least 90% feasible-action agreement and positive OOF utility on every high-SNR seed; clairvoyant remains strictly above OOF | `not-started` |
| P1-S2 | medium/low SNR degrades monotonically rather than collapsing through a hidden contract bug | same DGP with only SNR changed | median proper score and decision regret ordered high -> medium -> low; violations recorded, not tuned away | `not-started` |
| P1-S3 | semi-synthetic BTC recovers a known injected signal on real gaps, volatility and costs | v4-eligible BTC parent rows; zero injection control | injected-signal OOF score and net utility beat zero-injection parent on paired blocks; future perturbation leaves prefix unchanged | `not-started` |

## P2 — real-data predictability hypotheses

All P2 families must freeze exact folds, horizons, seeds, model hyperparameters and a family-wise multiplicity rule before outputs are opened. The default family is horizons 4/8/16/32 with unconditional/persistence, Ridge/logistic and one fixed-budget tree model. Deep models are not candidates until a simple model passes.

| id | hypothesis | parent/common-row comparison | required data | current status |
| --- | --- | --- | --- | --- |
| P2-D0 | corrected full17 adds stable outcome information beyond OHLCV13 | full17 minus OHLCV13 on identical availability-eligible rows | rebuilt local v4 cache | `not-started`; P0-A required |
| P2-D1 | signed/taker flow adds short-horizon return or downside information | D1 minus promoted D0 parent | checksumed Spot and USD-M `aggTrades`/kline metadata with right-exclusive aggregation | `blocked-data` |
| P2-D2 | spot-perp premium/flow divergence adds value after D1 | D2 minus D1 on common rows | audited mark/index/premium/funding timing | `blocked-data` |
| P2-D3 | OI/crowding improves downside/crash forecasts | D3 minus promoted parent | point-in-time OI/ratio history with revision/availability ledger | `blocked-data` |
| P2-D4 | L2 spread/depth/OFI survives 15m aggregation and one-bar delay | D4 minus promoted parent | reconstructable order-book updates and staleness ledger | `blocked-data`; forward collector required |
| P2-D5 | ETH/multi-asset state improves stability rather than fitting BTC only | pooled/leave-one-asset-out minus BTC-only parent | aligned ETH and later liquid assets | `blocked-data` |
| P2-D6 | options/macro/liquidation context improves calibrated tail risk | D6 minus promoted parent | release/vintage-aware external data | `blocked-data` |

For every P2 row, primary evidence is an OOS proper-score delta and the net utility of one frozen mapper. Sign accuracy, IC or one profitable fold alone cannot promote a feature group.

## P3-P6 — decision and learning hypotheses

| id | hypothesis | falsifier / promotion evidence | current status |
| --- | --- | --- | --- |
| P3-O1 | OOF point forecasts support a cost-positive four-bar spot allocation decision | O1 fails if it cannot beat hold/B&H and matched constant exposure on preregistered paired blocks | `not-started`; P2 required |
| P3-O2 | calibrated joint scenarios improve utility/risk over O1 | joint path calibration must pass; O2 must beat O1/O0 after multiplicity adjustment | `not-started` |
| P3-O3 | abstention improves the risk-coverage frontier without all-hold collapse | validation freezes threshold; active-rate/turnover bounds and OOS utility must pass | `not-started` |
| P4-DIRECT | direct O2/O3 is at least as good as a distilled student | student must retain utility under its own inventory trajectory; otherwise direct policy remains selected | `not-started` |
| P4-IL | self-conditioning/DAgger reduces trajectory error when BC compounds | must improve active-transition error and net utility over plain BC on the same support | `not-started` |
| P5-DFL | Q-ranking/SPO/direct-utility loss improves decision regret over outcome loss | common folds/features/action/cost; forecast score alone is insufficient | `not-started` |
| P5-AC | AC adds stable value beyond direct O3 and BC | must improve both comparators and pass support/OPE/disagreement checks | `not-started` |
| P6-STAT | the selected family survives trial-count, paired-block, DSR/PBO and regime/cost/delay stress | any required N/A remains exploratory and cannot promote | `not-started` |
| P6-EXT | the selected relationship transfers to ETH or leave-one-asset-out data | failure limits the claim to BTC development evidence | `blocked-data` |
| P7-HOLDOUT | the frozen candidate passes one new untouched temporal holdout and shadow replay | the already inspected 2024-2026 folds are not reused as pristine holdout | `blocked-data`; future accrual required |

## Active branches

| branch | owner | scope |
| --- | --- | --- |
| `exp/p0-a-availability-20260830` | Luna max P0-A | availability sidecar -> row/window eligibility |
| `exp/p0-b-target-oof-20260830` | Luna max P0-B | target/gradient coverage, OOF teacher gate, inventory provenance |
| `exp/p0-c-action-execution-20260830` | Luna max P0-C | shared action/execution/cost/commitment contract |

Each owner commits and pushes coherent units on its branch. The lead inspects the complete diff, reruns tests and merges only after the branch's scoped gate passes. Failed and blocked rows remain in this registry; they are not deleted from the candidate history.
