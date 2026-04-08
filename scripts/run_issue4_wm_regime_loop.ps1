param(
  [string]$Config = "configs/medium_v2.yaml",
  [string]$Start = "2020-01-01",
  [string]$End = "2024-01-01",
  [string]$CacheDir = "checkpoints/data_cache",
  [string]$CheckpointDir = "checkpoints",
  [string]$Folds = "4",
  [string]$Device = "cuda",
  [double]$Ridge = 1e-2
)

$ErrorActionPreference = "Stop"

Write-Host "[Issue4] WM regime audit"
uv run python audit_wm_regime.py `
  --config $Config `
  --start $Start `
  --end $End `
  --cache-dir $CacheDir `
  --checkpoint-dir $CheckpointDir `
  --folds $Folds `
  --device $Device `
  --ridge $Ridge

Write-Host "[Issue4] next-step candidates are documented in docs/optimization_issue4_wm_regime.md"
