# Official-source schema-v4 cache rebuild

This report reads no model results. It uses only Binance-owned sources and never interpolates missing bars.

- Cache output directory: `/tmp/unidream-v4-actual`
- Cache tag: `BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official`
- Scope: `[2018-01-01, 2024-01-01)` / `15m`
- Source probe status: **PASS**
- Source ledger: `docs/data_quality_v4_rebuild_2018_2024.jsonl`

## Official source probe

| Source | Probe responses | HTTP 200 | HTTP 404 |
| --- | ---: | ---: | ---: |
| `spot_klines` | 3 | 3 | 0 |
| `um_mark_price_klines` | 3 | 1 | 2 |
| `um_funding_rate` | 3 | 1 | 2 |

UM mark/funding archives before `2020-01-01` are treated as unavailable and their masks are false; no future value is backfilled into that period.

## Rebuild status

- **BLOCKED**: `OfficialSourceError: official Spot source returned an off-grid timestamp: first=2018-02-09 09:58:14.789000; source=spot_klines_archive; source_month=2018-02; range=[2018-01-01 00:00:00, 2024-01-01 00:00:00); previous=2018-02-08 00:15:00; delta_from_previous=1 days 09:43:14.789000; next=2018-02-09 10:00:00; delta_to_next=0 days 00:01:45.211000; interval=15m; grid_remainder=0 days 00:13:14.789000`
