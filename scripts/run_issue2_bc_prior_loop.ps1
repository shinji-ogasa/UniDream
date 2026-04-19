param(
  [string]$Config = "configs/medium_v2.yaml",
  [string]$Start = "2020-01-01",
  [string]$End = "2024-01-01",
  [string]$CacheDir = "checkpoints/data_cache",
  [string]$CheckpointDir = "checkpoints",
  [string]$Folds = "4",
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

Write-Host "[Issue2] BC prior audit"
uv run python audit_bc_prior.py `
  --config $Config `
  --start $Start `
  --end $End `
  --cache-dir $CacheDir `
  --checkpoint-dir $CheckpointDir `
  --folds $Folds `
  --device $Device

Write-Host "[Issue2] weighted BC"
uv run python train.py `
  --config configs/medium_l1_bc_weighted.yaml `
  --start $Start `
  --end $End `
  --folds $Folds `
  --device $Device `
  --checkpoint_dir checkpoints/medium_l1_bc_weighted_fold4

Write-Host "[Issue2] sequence BC"
uv run python train.py `
  --config configs/medium_l1_bc_sequence.yaml `
  --start $Start `
  --end $End `
  --folds $Folds `
  --device $Device `
  --checkpoint_dir checkpoints/medium_l1_bc_sequence_fold4

Write-Host "[Issue2] residual BC"
uv run python train.py `
  --config configs/medium_l1_bc_residual.yaml `
  --start $Start `
  --end $End `
  --folds $Folds `
  --device $Device `
  --checkpoint_dir checkpoints/medium_l1_bc_residual_fold4
