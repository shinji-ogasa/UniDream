param(
  [string]$Start = "2020-01-01",
  [string]$End = "2024-01-01",
  [string]$Folds = "4",
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

Write-Host "[Issue5] KL budget"
uv run python train.py `
  --config configs/medium_l1_ac_klbudget.yaml `
  --start $Start `
  --end $End `
  --folds $Folds `
  --device $Device `
  --checkpoint_dir checkpoints/medium_l1_ac_klbudget_fold4

Write-Host "[Issue5] support budget"
uv run python train.py `
  --config configs/medium_l1_ac_supportbudget.yaml `
  --start $Start `
  --end $End `
  --folds $Folds `
  --device $Device `
  --checkpoint_dir checkpoints/medium_l1_ac_supportbudget_fold4

Write-Host "[Issue5] conservative AC"
uv run python train.py `
  --config configs/medium_l1_ac_conservative.yaml `
  --start $Start `
  --end $End `
  --folds $Folds `
  --device $Device `
  --checkpoint_dir checkpoints/medium_l1_ac_conservative_fold4
