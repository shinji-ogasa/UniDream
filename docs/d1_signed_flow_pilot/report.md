# D1 signed-flow acquisition pilot

This is a data-only pilot. No model, prediction result, or P2 tournament was run.

- Scope: `BTCUSDT` `15m` monthly kline metadata, `2024-01`
- Row semantics: `decision_ts = bar_open_ts + 15m`; each feature row covers Binance `[bar_open_ts, decision_ts)` with inclusive `close_time=decision_ts-1ms`.
- Leakage rule: a bar is eligible only after its close; no next bar is read while constructing a row.
- Feature artifact: `docs/d1_signed_flow_pilot/pilot_features.csv`
- Capacity artifact: `docs/d1_signed_flow_pilot/aggtrade_capacity.json`
- Append-only ledger: `docs/d1_signed_flow_pilot/availability_revision_ledger.jsonl`
- Acquisition code commit: `97bb6f2da7f2db0b4ac032654669797fafb6d62e`

## Official sources

- [Binance Public Data README](https://github.com/binance/binance-public-data/blob/master/README.md)
- [Spot market-data REST specification](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints)
- [USD-M Futures market-data REST specification](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)

The README documents monthly/daily archives, Spot/Futures klines and checksum sidecars. The USD-M specification documents the corresponding market-data endpoints. Archive publication/download timestamps are recorded separately from live observation timestamps; this archive pilot does not claim live causal availability.

## Download and revision evidence

| source | HTTP | checksum | archive revision | parsed rows | live causal eligible |
| --- | ---: | --- | --- | ---: | --- |
| `spot_klines` | `200` | `True` | `20c617e34b350f61d4bba493154ddec76b9f2ecd449a94410a60ba75efd488ce` | `2976` | `False` |
| `um_klines` | `200` | `True` | `76953983fcd4cc35ac181c4a1c69d28cbb4ef8b983021aac84a111ea4e82ef69` | `2976` | `False` |

`archive_published_ts` is unknown for these downloaded files and `collector_observed_ts` is null. A later archive revision is never silently substituted: the ledger records previous and replacement revision IDs.

## D1 feature contract

- Spot and USD-M fields: trade count, quote volume, taker-buy base volume and taker-buy quote volume.
- Taker imbalance per venue: `(2 * taker_buy_quote / quote_volume) - 1` when quote volume is positive; otherwise the value is NaN and its mask is false.
- Spot-perp basis: `log(perp_close / spot_close)` at the completed bar close.
- Spot-perp return divergence: `log(perp_close_t/perp_close_{t-1}) - log(spot_close_t/spot_close_{t-1})`, requiring adjacent observed bars on both venues.
- Missing rows remain NaN. A numeric zero is retained as a value and is not used as a missing sentinel.

| item | value |
| --- | ---: |
| rows | `2976` |
| fully available D1 rows | `2975` (99.97%) |
| NaN feature cells | `1` |
| literal zero feature cells | `4` |

Availability columns: `spot_bar_observed, perp_bar_observed, spot_taker_imbalance_available, perp_taker_imbalance_available, spot_perp_basis_available, spot_perp_return_divergence_available, d1_features_available`.
Feature columns: `spot_trade_count, spot_quote_volume, spot_taker_buy_base, spot_taker_buy_quote, spot_taker_imbalance, perp_trade_count, perp_quote_volume, perp_taker_buy_base, perp_taker_buy_quote, perp_taker_imbalance, spot_perp_basis, spot_perp_return_divergence`.

## Aggregate-trade capacity check

Method: `HTTP HEAD Content-Length; no aggregate-trade payload downloaded`
Known compressed bytes across requested Spot + USD-M monthly archives: `101938925277`

No aggregate-trade payload was downloaded. The estimate is based on official `Content-Length` values for available monthly ZIPs; unknown/404 months remain explicit in the capacity JSON and ledger.

This artifact is feasibility evidence only. It does not establish that any D1 feature predicts returns or improves trading utility.
