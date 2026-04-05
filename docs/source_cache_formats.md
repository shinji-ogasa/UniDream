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

### Build from a Manifest

Use:

```powershell
.\.venv\Scripts\python.exe build_source_cache_from_manifest.py `
  --manifest configs\source_manifest_example.yaml
```

This is the shortest path when you have multiple market and external-source files.
To validate a manifest without fetching or writing files:

```powershell
.\.venv\Scripts\python.exe build_source_cache_from_manifest.py `
  --manifest configs\source_manifest_remote_example.yaml `
  --dry-run
```

The same manifest builder also supports remote provider sections:

- `binance`
- `coinmetrics`
- `glassnode`

See:

- [source_manifest_example.yaml](/C:/Users/Sophie/Documents/UniDream/UniDream/configs/source_manifest_example.yaml)
- [source_manifest_remote_example.yaml](/C:/Users/Sophie/Documents/UniDream/UniDream/configs/source_manifest_remote_example.yaml)

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

This builder now also writes, when available:

- `<cache_tag>_series_signed_order_flow.parquet`
- `<cache_tag>_series_taker_imbalance.parquet`
- `<cache_tag>_series_buy_sell_ratio.parquet`

### Download from Coin Metrics

Use:

```powershell
.\.venv\Scripts\python.exe build_coinmetrics_source_cache.py `
  --cache-dir checkpoints\basis_source_cache `
  --cache-tag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2 `
  --asset btc `
  --start 2021-01-01 `
  --end 2023-06-01 `
  --frequency 1h `
  --metric active_address_growth=AdrActCnt:logdiff
```

### Download from Glassnode

Use:

```powershell
.\.venv\Scripts\python.exe build_glassnode_source_cache.py `
  --cache-dir checkpoints\basis_source_cache `
  --cache-tag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2 `
  --asset BTC `
  --start 2021-01-01 `
  --end 2023-06-01 `
  --pit `
  --interval 1h `
  --api-key <glassnode_key> `
  --metric exchange_netflow=transactions/transfers_volume_exchanges_net
```

Provider builders accept optional transforms on metrics:

- `:diff`
- `:pct_change`
- `:logdiff`

## Offline Runners

After caches exist, run:

```powershell
.\scripts\run_basis_source_probe.ps1
```

Or, if you are importing from a manifest and want the external-series probe in one step:

```powershell
.\scripts\run_manifest_external_probes.ps1 `
  -Manifest configs\source_manifest_example.yaml `
  -RiskConfig configs\smoke_risk_controller_v8_orderflow_ctx.yaml
```

To build from a manifest and then compare source families in one step:

```powershell
.\scripts\run_manifest_source_family_suite.ps1 `
  -Manifest configs\source_manifest_remote_example.yaml
```

To compare source families side by side after caches exist:

```powershell
.\scripts\run_source_family_suite.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

This runs the basis / orderflow / on-chain probes and writes:

- `checkpoints\source_family_suite\<config>\risk_controller_summary.csv`
- `checkpoints\source_family_suite\suite_summary.csv`

The default suite now also includes a hybrid config that combines basis,
order-flow, and on-chain context features.
By default, configs whose raw source dependencies are still missing are skipped.

To generate a manifest stub from those suite configs:

```powershell
.\scripts\generate_suite_manifest_stub.ps1
```

This writes a placeholder manifest listing the raw source files and extra series
needed by the selected probe configs.

For staged rollout by source priority:

```powershell
.\scripts\run_priority_source_rollout.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

This runs, in order:

1. basis
2. order-flow
3. on-chain
4. hybrid

To generate a config-to-source requirement matrix:

```powershell
.\scripts\generate_source_requirements_matrix.ps1
```

To see which stages are unlocked by the current cache:

```powershell
.\scripts\report_source_stage_status.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

To get the next raw sources to fetch in rollout order:

```powershell
.\scripts\recommend_next_source_step.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

To get concrete fetch/import command hints for that next blocked stage:

```powershell
.\scripts\recommend_source_fetch_commands.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

To generate a manifest stub for that next blocked stage:

```powershell
.\scripts\generate_next_source_manifest.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

To run the full planning loop in one step:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\plan_next_source_rollout.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

To print the full rollout diagnosis in one step:

```powershell
.\scripts\diagnose_source_rollout.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

If PowerShell script execution is blocked, call wrappers as:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\diagnose_source_rollout.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

To save the rollout status as markdown:

```powershell
.\scripts\write_source_rollout_report.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

That script:

1. Builds raw source cache
2. Rebuilds basis-aware training cache
3. Runs the basis-aware risk probe
