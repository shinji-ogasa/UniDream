param(
  [string]$Start = "2021-01-01",
  [string]$End = "2023-06-01",
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"

Write-Host "[ISSUE7] medium_l0_bc_sequence"
.\.venv\Scripts\python.exe .\audit_policy_collapse.py `
  --config .\configs\medium_l0_bc_sequence.yaml `
  --start $Start `
  --end $End `
  --checkpoint-dir checkpoints\medium_l0_bc_sequence_fold4 `
  --folds 4 `
  --splits val `
  --max-bars 4096 `
  --device $Device

Write-Host "[ISSUE7] medium_l0_bc_weighted"
.\.venv\Scripts\python.exe .\audit_policy_collapse.py `
  --config .\configs\medium_l0_bc_weighted.yaml `
  --start $Start `
  --end $End `
  --checkpoint-dir checkpoints\medium_l0_bc_weighted_fold4 `
  --folds 4 `
  --splits val `
  --max-bars 4096 `
  --device $Device

Write-Host "[ISSUE7] medium_l0_ac_conservative_rawonly"
.\.venv\Scripts\python.exe .\audit_policy_collapse.py `
  --config .\configs\medium_l0_ac_conservative_rawonly.yaml `
  --start $Start `
  --end $End `
  --checkpoint-dir checkpoints\medium_l0_ac_conservative_rawonly_fold4 `
  --folds 4 `
  --splits val `
  --max-bars 4096 `
  --device $Device

Write-Host "[ISSUE7] medium_l0_bc_sequence_regimedist"
.\.venv\Scripts\python.exe .\audit_policy_collapse.py `
  --config .\configs\medium_l0_bc_sequence_regimedist.yaml `
  --start "2020-01-01" `
  --end "2024-01-01" `
  --cache-dir checkpoints\medium_l0_bc_sequence_regimedist\data_cache `
  --checkpoint-dir checkpoints\medium_l0_bc_sequence_regimedist `
  --folds 4 `
  --splits val `
  --max-bars 4096 `
  --device $Device

Write-Host "[ISSUE7] medium_l0_bc_rawonly_shortmass"
.\.venv\Scripts\python.exe .\audit_policy_collapse.py `
  --config .\configs\medium_l0_bc_rawonly_shortmass.yaml `
  --start $Start `
  --end $End `
  --cache-dir checkpoints\medium_l0_bc_rawonly_shortmass\data_cache `
  --checkpoint-dir checkpoints\medium_l0_bc_rawonly_shortmass `
  --folds 4 `
  --splits val `
  --max-bars 4096 `
  --device $Device
