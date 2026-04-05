param(
  [string]$CacheDir = "checkpoints\\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string[]]$Configs = @(
    "configs\\smoke_risk_controller_v5_basis.yaml",
    "configs\\smoke_risk_controller_v8_orderflow_ctx.yaml",
    "configs\\smoke_risk_controller_v9_onchain_ctx.yaml",
    "configs\\smoke_risk_controller_v10_hybrid_ctx.yaml",
    "configs\\smoke_risk_controller_v11_hybrid_linear.yaml"
  )
)

$ErrorActionPreference = "Stop"
$Python = ".\\.venv\\Scripts\\python.exe"
$ArgsList = @(
  "recommend_next_source_step.py",
  "--cache-dir", $CacheDir,
  "--cache-tag", $CacheTag
)
foreach ($Config in $Configs) {
  $ArgsList += @("--config", $Config)
}
& $Python @ArgsList
