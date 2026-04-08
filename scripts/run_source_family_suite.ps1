param(
  [string]$CacheDir = "checkpoints\\basis_source_cache",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$Interval = "15m",
  [string]$Folds = "4",
  [bool]$SkipMissing = $true,
  [string]$SuiteConfig = "configs\\source_rollout_suite.yaml",
  [string[]]$Configs = @(),
  [string]$SuiteDir = "checkpoints\\source_family_suite"
)

$ErrorActionPreference = "Stop"
$Python = ".\\.venv\\Scripts\\python.exe"
New-Item -ItemType Directory -Force -Path $SuiteDir | Out-Null

if (($Configs.Count -eq 0) -and (Test-Path $SuiteConfig)) {
  $SuiteYaml = (uv run python yaml_to_json.py --file $SuiteConfig) | ConvertFrom-Json
  if ($SuiteYaml.ordered_configs) {
    $Configs = @($SuiteYaml.ordered_configs)
  }
}

foreach ($Config in $Configs) {
  $Stem = [System.IO.Path]::GetFileNameWithoutExtension($Config)
  $RunDir = Join-Path $SuiteDir $Stem
  New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

  if ($SkipMissing) {
    & $Python check_config_source_requirements.py `
      --cache-dir $CacheDir `
      --cache-tag $CacheTag `
      --config $Config
    if ($LASTEXITCODE -ne 0) {
      Write-Host "[SUITE] Skipping $Stem due to missing raw source dependencies."
      continue
    }
  }

  Write-Host "[SUITE] Inspecting cache for $Stem ..."
  & $Python inspect_source_cache.py `
    --cache-dir $CacheDir `
    --cache-tag $CacheTag `
    --interval $Interval `
    --zscore-window-days 60 `
    --config $Config

  Write-Host "[SUITE] Running $Stem ..."
  & $Python train_risk_controller.py `
    --config $Config `
    --folds $Folds `
    --device cuda `
    --checkpoint_dir $RunDir `
    --data_cache_dir $CacheDir
}

Write-Host "[SUITE] Aggregating summaries ..."
& $Python summarize_probe_suite.py `
  --suite-dir $SuiteDir
