param(
  [string]$Manifest = "configs\\source_manifest_example.yaml",
  [string]$RiskConfig = "configs\\smoke_risk_controller_v8_orderflow_ctx.yaml",
  [string]$Folds = "4",
  [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Python = ".\\.venv\\Scripts\\python.exe"

Write-Host "[1/3] Building source cache from manifest..."
& $Python build_source_cache_from_manifest.py `
  --manifest $Manifest

$ManifestYaml = (uv run python yaml_to_json.py --file $Manifest) | ConvertFrom-Json
$CacheDir = $ManifestYaml.cache_dir
$CacheTag = $ManifestYaml.cache_tag
$Interval = "15m"
if ($ManifestYaml.data -and $ManifestYaml.data.interval) {
  $Interval = $ManifestYaml.data.interval
}

Write-Host "[2/3] Inspecting source cache..."
& $Python inspect_source_cache.py `
  --cache-dir $CacheDir `
  --cache-tag $CacheTag `
  --interval $Interval `
  --zscore-window-days 60 `
  --config $RiskConfig

Write-Host "[3/3] Running external-series risk probe..."
& $Python train_risk_controller.py `
  --config $RiskConfig `
  --folds $Folds `
  --device $Device `
  --checkpoint_dir checkpoints\\manifest_external_risk_probe `
  --data_cache_dir $CacheDir
