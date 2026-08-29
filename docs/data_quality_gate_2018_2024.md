# Development data / feature quality gate

This report audits only the materialized development cache. No model, forecast, or future-return result was read.

- Scope: `[2018-01-01 00:00:00, 2024-01-01 00:00:00)`
- Interval: `15m`
- Overall gate: **FAIL**
- JSONL ledger: `docs/data_quality_gate_2018_2024.jsonl`
- Metadata digest: `f80a1f2ec971b9b1c51c30387103a4614b5d967f30e731ee57b7b7e89c9dfdc0`
- Schema digest: `d6a1ad5904a1de9f53377217cdc3cdff3012c0b0879ec1d092f26733cefde9ef`

## Gate matrix

| Gate | Status |
| --- | --- |
| cache contract | fail |
| causal feature probes | pass |
| OHLCV13/full17 same-row eligibility | pass |
| external availability mask | fail |

## Schema and alignment

- Metadata is authoritative: `True`
- Feature columns: `['open_ret', 'high_ret', 'low_ret', 'close_ret', 'vol_ret', 'RSI_14', 'macd', 'macd_signal', 'atr_norm_ret', 'atr', 'rv_4', 'rv_16', 'rv_96', 'funding_rate', 'basis', 'basis_mom', 'basis_abs']`
- Features/returns exact index alignment: `True`
- Feature timestamp diagnostics: `fail`
- Returns timestamp diagnostics: `fail`
- Non-15m steps: features `29`, returns `29`; missing-bar estimate `542`
- Non-finite values: features `0`, returns `0`

## Causality probes

- Overall: `pass`
- `future_perturbation_prefix`: pass, max difference `0.0`
- `prefix_invariance`: pass, max difference `0.0`
- `mark_offset_no_future_bfill`: pass, max difference `0.0`
- `funding_offset_asof`: pass, max difference `0.0`

## External coverage

Coverage records finite/nonfinite, zero, and nonzero values independently. The v3 cache has no availability mask, so the zero-vs-missing gate is intentionally failed even when the observed rows are finite.

| Year | Rows | funding zero/nonzero/missing | basis zero/nonzero/missing | basis_mom zero/nonzero/missing | basis_abs zero/nonzero/missing |
| --- | ---: | --- | --- | --- | --- |
| 2018 | 33290 | 33290/0/0 | 33290/0/0 | 33290/0/0 | 33290/0/0 |
| 2019 | 34918 | 24120/10798/0 | 34102/816/0 | 34104/814/0 | 34103/815/0 |
| 2020 | 35053 | 0/35053/0 | 0/35053/0 | 0/35053/0 | 0/35053/0 |
| 2021 | 34969 | 0/34969/0 | 0/34969/0 | 0/34969/0 | 0/34969/0 |
| 2022 | 35040 | 0/35040/0 | 0/35040/0 | 0/35040/0 | 0/35040/0 |
| 2023 | 35029 | 0/35029/0 | 0/35029/0 | 0/35029/0 | 0/35029/0 |

## OHLCV13 vs full17 fairness

- Rule: `finite intersection of all full17 columns within the development scope`
- OHLCV13 eligible rows: `208299`
- Full17 eligible rows: `208299`
- Same row mask: `True`
- Full17 eligible period: `[2018-01-16 13:45:00, 2024-01-01 00:00:00)`

## WFO coverage

WFO rows are reported by configured fold and right-exclusive train/val/test phase; no performance metric is computed.

| Fold | Phase | Rows | availability-mask gate |
| ---: | --- | ---: | --- |
| 0 | train | 69703 | fail |
| 0 | val | 8701 | fail |
| 0 | test | 8712 | fail |
| 1 | train | 69900 | fail |
| 1 | val | 8712 | fail |
| 1 | test | 8832 | fail |
| 2 | train | 69953 | fail |
| 2 | val | 8832 | fail |
| 2 | test | 8808 | fail |
| 3 | train | 69953 | fail |
| 3 | val | 8808 | fail |
| 3 | test | 8623 | fail |
| 4 | train | 69971 | fail |
| 4 | val | 8623 | fail |
| 4 | test | 8708 | fail |
| 5 | train | 69978 | fail |
| 5 | val | 8708 | fail |
| 5 | test | 8806 | fail |
| 6 | train | 69999 | fail |
| 6 | val | 8806 | fail |
| 6 | test | 8832 | fail |
| 7 | train | 70005 | fail |
| 7 | val | 8832 | fail |
| 7 | test | 8640 | fail |
| 8 | train | 70022 | fail |
| 8 | val | 8640 | fail |
| 8 | test | 8736 | fail |
| 9 | train | 69961 | fail |
| 9 | val | 8736 | fail |
| 9 | test | 8832 | fail |
| 10 | train | 69985 | fail |
| 10 | val | 8832 | fail |
| 10 | test | 8832 | fail |
| 11 | train | 69985 | fail |
| 11 | val | 8832 | fail |
| 11 | test | 8629 | fail |
| 12 | train | 70009 | fail |
| 12 | val | 8629 | fail |
| 12 | test | 8736 | fail |

## Blocking limitation

v3 cache has no availability mask; external zero and missing values cannot be distinguished

The pre-declared fair comparison rule is to use the intersection of rows finite for all full17 features for both ablations. If external availability is required for a future cache, add explicit per-column availability masks before treating zero as observed data.
