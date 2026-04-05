param(
  [string]$Symbol = "BTCUSDT",
  [string]$Interval = "15m",
  [string]$Start = "2021-01-01",
  [string]$End = "2023-06-01",
  [string]$CacheDir = "checkpoints\\basis_source_cache",
  [string]$RiskConfig = "configs\\smoke_risk_controller_v6_orderflow.yaml",
  [string]$Folds = "4"
)

$ErrorActionPreference = "Stop"
$Python = ".\\.venv\\Scripts\\python.exe"
$CacheTag = "${Symbol}_${Interval}_${Start}_${End}_z60_v2"

Write-Host "[1/2] Inspecting source cache..."
& $Python inspect_source_cache.py `
  --cache-dir $CacheDir `
  --cache-tag $CacheTag `
  --interval $Interval `
  --zscore-window-days 60

Write-Host "[2/2] Running external-series risk probe..."
& $Python train_risk_controller.py `
  --config $RiskConfig `
  --start $Start `
  --end $End `
  --folds $Folds `
  --device cuda `
  --checkpoint_dir checkpoints\\external_series_risk_probe `
  --data_cache_dir $CacheDir
