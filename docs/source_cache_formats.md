# Source Cache Formats

UniDream can rebuild features and run probes fully offline when raw source caches exist in a cache directory.

## Cache Tag

The cache tag must match:

`<SYMBOL>_<INTERVAL>_<START>_<END>_z<ZSCORE_WINDOW>_v2`

Example:

`BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2`

## Required File Names

Place files under the same cache directory using these names:

- `<cache_tag>_ohlcv.parquet`
- `<cache_tag>_mark.parquet`
- `<cache_tag>_funding.parquet`
- `<cache_tag>_oi.parquet`
- `<cache_tag>_series_<name>.parquet`

Only `ohlcv` is required for a full offline rebuild. The others are optional.

## File Schemas

### Spot OHLCV

Filename:

- `<cache_tag>_ohlcv.parquet`

Columns:

- `open`
- `high`
- `low`
- `close`
- `volume`

Index or time column:

- Datetime index, or one of `time`, `open_time`, `timestamp`

### Mark Price

Filename:

- `<cache_tag>_mark.parquet`

Columns:

- `mark_close`

Accepted alternatives for import:

- `close`
- `markPrice`
- `mark_price`

Index or time column:

- Datetime index, or one of `time`, `open_time`, `timestamp`

### Funding Rate

Filename:

- `<cache_tag>_funding.parquet`

Columns:

- `funding_rate`

Accepted alternative:

- `fundingRate`

Index or time column:

- Datetime index, or one of `time`, `fundingTime`, `timestamp`

### Open Interest

Filename:

- `<cache_tag>_oi.parquet`

Columns:

- `open_interest`

Accepted alternatives:

- `sumOpenInterest`
- `openInterest`

Index or time column:

- Datetime index, or one of `time`, `timestamp`
- `t` is also accepted for Glassnode-style exports (unix seconds or milliseconds)

Value column detection:

- explicit `name=path:column`
- otherwise a numeric column named `<name>`
- otherwise `v`
- otherwise `value`
- otherwise the single numeric column in the file

### Generic External Series

Filename:

- `<cache_tag>_series_<name>.parquet`

Columns:

- one numeric column named `<name>`

Examples:

- `exchange_netflow`
- `stablecoin_inflow`
- `signed_order_flow`

When a generic external series is present, feature rebuild now creates both the
raw shifted level and simple context features automatically:

- `<name>`
- `<name>_delta1`
- `<name>_abs`
- `<name>_mean_<bars>`
- `<name>_std_<bars>`
- `<name>_z_<bars>`
- `<name>_impulse_<bars>`

Default context windows are 4h / 24h / 72h.

Index or time column:

- Datetime index, or one of `time`, `timestamp`

## Builders

### Import from Existing Files

Use:

```powershell
.\.venv\Scripts\python.exe build_aux_source_cache.py `
  --cache-dir checkpoints\basis_source_cache `
  --cache-tag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2 `
  --spot-file path\to\spot.csv `
  --mark-file path\to\mark.csv `
  --funding-file path\to\funding.csv `
  --oi-file path\to\oi.csv `
  --extra-series exchange_netflow=path\to\netflow.csv:netflow
```

### Download from Binance

Use:

```powershell
.\.venv\Scripts\python.exe build_binance_source_cache.py `
  --cache-dir checkpoints\basis_source_cache `
  --cache-tag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2 `
  --symbol BTCUSDT `
  --interval 15m `
  --start 2021-01-01 `
  --end 2023-06-01
```

## Offline Runners

After caches exist, run:

```powershell
.\scripts\run_basis_source_probe.ps1
```

That script:

1. Builds raw source cache
2. Rebuilds basis-aware training cache
3. Runs the basis-aware risk probe
