param(
  [string]$CacheDir = "checkpoints\\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$Interval = "15m",
  [string]$Folds = "4",
  [string]$RootDir = "checkpoints\\priority_source_rollout"
)

$ErrorActionPreference = "Stop"
$Python = ".\\.venv\\Scripts\\python.exe"
New-Item -ItemType Directory -Force -Path $RootDir | Out-Null

$Stages = @(
  @{ Name = "stage1_basis"; Configs = @("configs\\smoke_risk_controller_v5_basis.yaml") },
  @{ Name = "stage2_orderflow"; Configs = @("configs\\smoke_risk_controller_v8_orderflow_ctx.yaml") },
  @{ Name = "stage3_onchain"; Configs = @("configs\\smoke_risk_controller_v9_onchain_ctx.yaml") },
  @{ Name = "stage4_hybrid"; Configs = @(
      "configs\\smoke_risk_controller_v10_hybrid_ctx.yaml",
      "configs\\smoke_risk_controller_v11_hybrid_linear.yaml"
    )
  }
)

foreach ($Stage in $Stages) {
  $StageDir = Join-Path $RootDir $Stage.Name
  Write-Host "[ROLLOUT] Running $($Stage.Name) ..."
  & .\scripts\run_source_family_suite.ps1 `
    -CacheDir $CacheDir `
    -CacheTag $CacheTag `
    -Interval $Interval `
    -Folds $Folds `
    -SuiteDir $StageDir `
    -Configs $Stage.Configs
}

Write-Host "[ROLLOUT] Aggregating all stages ..."
& $Python summarize_probe_suite.py `
  --suite-dir $RootDir
