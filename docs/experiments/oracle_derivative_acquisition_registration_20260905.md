# Derivative raw-data acquisition registration

This data-only protocol is committed before downloading any new archive. It
does not change existing model registrations or selection locks.

- Source: official Binance USD-M BTCUSDT monthly traded klines, 15-minute bars.
- Fixed requested archive months: September 2019 through April 2023 inclusive
  (44 months). Missing months remain explicit; no substitute source is used.
- Persist raw open, high, low, close, base/quote volume, trade count, and
  taker-buy base/quote volume. These support a later spot/perpetual flow and
  traded-price-divergence ablation; no features or models are fitted here.
- The existing `download_d1_kline_month` helper verifies the official CHECKSUM
  and applies its strict field, timestamp and OHLC parser. Invalid rows are not
  silently quarantined or remapped. A non-404 source/checksum/parser error fails
  closed. A 404 is retained as an unavailable month with NaN grid observations.
- Retain original ZIP payloads, parsed monthly Parquet, monthly source records,
  and an append-only source ledger. Bind these to config/source hashes before
  download. Existing data are verified and reused or rejected, never overwritten.
- Final raw data retain the full 15-minute UTC bar-open grid. The observed
  inclusive exchange close is open + 15 minutes - 1 millisecond; completed-bar
  decision time is open + 15 minutes. Missing source bars have NaN raw fields and
  false observation masks; clock timestamps are still defined on the grid.
- Feature-decision eligibility requires an observed bar whose completed-bar
  decision timestamp is strictly before **2023-04-16 13:45 UTC**. The remainder
  of April is retained as raw archive content only and cannot enter this
  development feature scope. No prediction, performance or test selection is
  performed on any date by this acquisition.
- Source records preserve archive URL, download time, response/checksum digests,
  archive revision and parser evidence. Historical exchange/collector/publication
  timestamps remain null; this retrospective archive has no live-causality claim.
- No OI endpoint, aggregate-trade payload, account API, credentials or paid
  infrastructure is used. Follow-up model experiments need their own frozen
  common-row, training, purge and validation protocol.

Run from this worktree after the protocol/code/config commit:

```sh
python -m unidream.experiments.oracle_derivative_acquisition --config configs/oracle_derivative_acquisition_20260905.yaml
```
