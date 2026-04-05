param(
  [string]$Output = "docs\\source_requirements_matrix.md",
  [string]$SuiteConfig = "configs\\source_rollout_suite.yaml",
  [string[]]$Configs = @()
)

$ErrorActionPreference = "Stop"
$Python = ".\\.venv\\Scripts\\python.exe"
if (($Configs.Count -eq 0) -and (Test-Path $SuiteConfig)) {
  $SuiteYaml = (& $Python yaml_to_json.py --file $SuiteConfig) | ConvertFrom-Json
  if ($SuiteYaml.ordered_configs) {
    $Configs = @($SuiteYaml.ordered_configs)
  }
}
$ArgsList = @(
  "summarize_source_requirements.py",
  "--output", $Output
)
foreach ($Config in $Configs) {
  $ArgsList += @("--config", $Config)
}
& $Python @ArgsList
