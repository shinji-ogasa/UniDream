# Official Binance gap-recovery audit

This report probes only Binance-owned Spot market-data sources around the development cache. It does not read model results and never interpolates or writes cache rows.

- Scope: `[2018-01-01 00:00:00, 2024-01-01 00:00:00)`
- Symbol / interval: `BTCUSDT` / `15m`
- Allowed hosts: `data-api.binance.vision, data.binance.vision`
- REST base: `https://data-api.binance.vision`
- Archive base: `https://data.binance.vision/data/spot/monthly/klines`
- Non-official provider used: `False`
- Interpolation used: `False`
- Cache feature rows / index digest: `208299` / `654674119f9a358ddfee40d9fce2c3434d13bd8482f35088f49a2a56f21a6c67`
- Returns rows / index digest: `208299` / `654674119f9a358ddfee40d9fce2c3434d13bd8482f35088f49a2a56f21a6c67`
- Ledger: `docs/data_quality_gap_recovery_2018_2024.jsonl`
- Probe git commit: `bd0b7e90ef8b354a2b0143d687364f5aa7ea67da`

## Summary

- Status: **UNRESOLVED_OFFICIAL_GAP**
- Gaps: `29`
- Expected missing bars: `542`
- Officially covered bars: `18`
- Unresolved after official probes: `524`

An unresolved bar is retained as a data-quality gap. The next cache generation may include an observed-bar availability sidecar, but it must not synthesize the missing OHLCV row.

The 18 officially covered timestamps are eligible for a future v4 regeneration only after their official OHLCV rows and as-of external inputs are recomputed into the new body. This audit intentionally did not mutate the v3 body.
The remaining 524 timestamps are retained as unresolved exchange/source outages: v4 should mark them `spot_bar_observed=False`, keep external availability masks separate, and exclude every sequence window crossing them.

## Per-gap coverage

| Gap | Left | Right | Expected | Covered | Unresolved | Coverage |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | 2018-02-08 00:15:00 | 2018-02-09 10:00:00 | 134 | 0 | 134 | 0.000 |
| 1 | 2018-02-10 05:45:00 | 2018-02-10 06:15:00 | 1 | 0 | 1 | 0.000 |
| 2 | 2018-02-11 04:00:00 | 2018-02-11 04:30:00 | 1 | 0 | 1 | 0.000 |
| 3 | 2018-06-26 01:45:00 | 2018-06-26 12:00:00 | 40 | 0 | 40 | 0.000 |
| 4 | 2018-06-27 12:45:00 | 2018-06-27 14:45:00 | 7 | 0 | 7 | 0.000 |
| 5 | 2018-07-04 00:15:00 | 2018-07-04 08:00:00 | 30 | 0 | 30 | 0.000 |
| 6 | 2018-10-19 05:45:00 | 2018-10-19 09:30:00 | 14 | 0 | 14 | 0.000 |
| 7 | 2018-11-14 01:45:00 | 2018-11-14 09:00:00 | 28 | 0 | 28 | 0.000 |
| 8 | 2019-03-12 01:45:00 | 2019-03-12 08:00:00 | 24 | 0 | 24 | 0.000 |
| 9 | 2019-05-15 02:45:00 | 2019-05-15 13:00:00 | 40 | 0 | 40 | 0.000 |
| 10 | 2019-06-07 20:15:00 | 2019-06-07 22:45:00 | 9 | 5 | 4 | 0.556 |
| 11 | 2019-08-15 01:45:00 | 2019-08-15 10:00:00 | 32 | 0 | 32 | 0.000 |
| 12 | 2019-11-13 01:45:00 | 2019-11-13 04:15:00 | 9 | 0 | 9 | 0.000 |
| 13 | 2019-11-25 01:45:00 | 2019-11-25 04:00:00 | 8 | 0 | 8 | 0.000 |
| 14 | 2020-02-09 01:45:00 | 2020-02-09 03:00:00 | 4 | 0 | 4 | 0.000 |
| 15 | 2020-02-19 11:30:00 | 2020-02-19 17:30:00 | 23 | 0 | 23 | 0.000 |
| 16 | 2020-03-04 09:15:00 | 2020-03-04 11:30:00 | 8 | 0 | 8 | 0.000 |
| 17 | 2020-04-25 01:45:00 | 2020-04-25 04:30:00 | 10 | 0 | 10 | 0.000 |
| 18 | 2020-06-28 01:45:00 | 2020-06-28 05:30:00 | 14 | 0 | 14 | 0.000 |
| 19 | 2020-11-30 05:45:00 | 2020-11-30 07:00:00 | 4 | 0 | 4 | 0.000 |
| 20 | 2020-12-21 13:45:00 | 2020-12-21 18:00:00 | 16 | 1 | 15 | 0.062 |
| 21 | 2020-12-25 01:45:00 | 2020-12-25 03:00:00 | 4 | 0 | 4 | 0.000 |
| 22 | 2021-02-11 02:30:00 | 2021-02-11 05:30:00 | 11 | 6 | 5 | 0.545 |
| 23 | 2021-03-06 01:45:00 | 2021-03-06 03:30:00 | 6 | 0 | 6 | 0.000 |
| 24 | 2021-04-20 01:45:00 | 2021-04-20 04:30:00 | 10 | 0 | 10 | 0.000 |
| 25 | 2021-04-25 04:00:00 | 2021-04-25 08:45:00 | 18 | 0 | 18 | 0.000 |
| 26 | 2021-08-13 01:45:00 | 2021-08-13 06:30:00 | 18 | 0 | 18 | 0.000 |
| 27 | 2021-09-29 06:45:00 | 2021-09-29 09:00:00 | 8 | 0 | 8 | 0.000 |
| 28 | 2023-03-24 11:30:00 | 2023-03-24 14:30:00 | 11 | 6 | 5 | 0.545 |

## Future v4 remediation policy

- Keep the feature body at the exact 17 model columns; store `spot_bar_observed` and external-source availability in a separate sidecar.
- Preserve official source/provenance hashes and the explicit gap list in v4 metadata.
- Exclude sequence windows crossing unresolved gaps; do not sort, drop, fill, or interpolate rows during cache validation.
- If official recovery remains incomplete, execution/evaluation must either segment metrics at the gap or explicitly attribute a return spanning the gap to the position held immediately before the gap. That attribution is a separate contract and must not silently become a post-gap position.
