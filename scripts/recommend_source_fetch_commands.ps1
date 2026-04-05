param(
  [string]$CacheDir = "checkpoints\\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
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
  "recommend_source_fetch_commands.py",
  "--cache-dir", $CacheDir,
  "--cache-tag", $CacheTag
)
foreach ($Config in $Configs) {
  $ArgsList += @("--config", $Config)
}
& $Python @ArgsList
