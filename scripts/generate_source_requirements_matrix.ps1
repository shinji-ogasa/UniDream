param(
  [string]$Output = "docs\\source_requirements_matrix.md",
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
  "summarize_source_requirements.py",
  "--output", $Output
)
foreach ($Config in $Configs) {
  $ArgsList += @("--config", $Config)
}
& $Python @ArgsList
