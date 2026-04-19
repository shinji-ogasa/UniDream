param(
  [string]$Symbol = "BTCUSDT",
  [string]$Interval = "15m",
  [string]$Start = "2021-01-01",
  [string]$End = "2023-06-01",
  [string]$CacheDir = "checkpoints\\basis_source_cache",
  [string]$TrainConfig = "configs\\smoke_signal_teacher_v21_basis_source.yaml",
  [string]$RiskConfig = "configs\\smoke_risk_controller_v5_basis.yaml",
  [string]$Folds = "4",
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Python = ".\\.venv\\Scripts\\python.exe"
$CacheTag = "${Symbol}_${Interval}_${Start}_${End}_z60_v2"

Write-Host "[1/3] Building raw source cache..."
& $Python build_binance_source_cache.py `
  --cache-dir $CacheDir `
  --cache-tag $CacheTag `
  --symbol $Symbol `
  --interval $Interval `
  --start $Start `
  --end $End

Write-Host "[2/3] Rebuilding basis-aware training cache..."
& $Python train.py `
  --config $TrainConfig `
  --symbol $Symbol `
  --start $Start `
  --end $End `
  --device $Device `
  --stop-after bc `
  --folds $Folds `
  --checkpoint_dir checkpoints\\basis_source_train `
  --resume

Write-Host "[3/3] Running basis-aware risk probe..."
& $Python train_risk_controller.py `
  --config $RiskConfig `
  --start $Start `
  --end $End `
  --folds $Folds `
  --device $Device `
  --checkpoint_dir checkpoints\\basis_source_risk_probe `
  --data_cache_dir $CacheDir
