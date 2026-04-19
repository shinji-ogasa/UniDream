param(
  [string]$Config = "configs/medium_v2.yaml",
  [string]$Start = "2020-01-01",
  [string]$End = "2024-01-01",
  [string]$CacheDir = "checkpoints/data_cache",
  [string]$CheckpointDir = "checkpoints",
  [string]$Folds = "4",
  [string]$Device = "auto",
  [string]$CheckpointName = "ac_best.pt"
)

$ErrorActionPreference = "Stop"

Write-Host "[Issue3] AC support audit"
uv run python audit_ac_support.py `
  --config $Config `
  --start $Start `
  --end $End `
  --cache-dir $CacheDir `
  --checkpoint-dir $CheckpointDir `
  --folds $Folds `
  --device $Device `
  --checkpoint-name $CheckpointName

Write-Host "[Issue3] next-step candidates are documented in docs/optimization_issue3_ac_support.md"
