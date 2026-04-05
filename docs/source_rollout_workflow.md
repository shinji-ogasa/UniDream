# Source Rollout Workflow

This repo uses a staged source rollout:

1. basis
2. order-flow
3. on-chain
4. hybrid

The canonical config order and stage grouping live in:

- [source_rollout_suite.yaml](/C:/Users/Sophie/Documents/UniDream/UniDream/configs/source_rollout_suite.yaml)

## Main Commands

### 1. Diagnose current cache

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\diagnose_source_rollout.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

### 2. Generate the next-stage manifest

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\generate_next_source_manifest.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

### 3. Dry-run the manifest

```powershell
.\.venv\Scripts\python.exe build_source_cache_from_manifest.py `
  --manifest configs\source_manifest_next_stage.yaml `
  --dry-run
```

### 4. Build source cache from the manifest

```powershell
.\.venv\Scripts\python.exe build_source_cache_from_manifest.py `
  --manifest configs\source_manifest_next_stage.yaml
```

### 5. Run the source family suite

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_source_family_suite.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

### 6. Save a markdown rollout report

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\write_source_rollout_report.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

### 7. Run the staged rollout

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_priority_source_rollout.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

## Provider Builders

- Binance: [build_binance_source_cache.py](/C:/Users/Sophie/Documents/UniDream/UniDream/build_binance_source_cache.py)
- Coin Metrics: [build_coinmetrics_source_cache.py](/C:/Users/Sophie/Documents/UniDream/UniDream/build_coinmetrics_source_cache.py)
- Glassnode: [build_glassnode_source_cache.py](/C:/Users/Sophie/Documents/UniDream/UniDream/build_glassnode_source_cache.py)

## Notes

- Use `--dry-run` before touching remote providers.
- `run_source_family_suite.ps1` skips configs whose raw source dependencies are still missing.
- Glassnode exchange-flow examples use `--pit`.
