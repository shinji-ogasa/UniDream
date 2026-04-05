param(
  [string]$Manifest = "configs\\source_manifest_example.yaml",
  [string]$Folds = "4"
)

$ErrorActionPreference = "Stop"
$Python = ".\\.venv\\Scripts\\python.exe"

Write-Host "[1/2] Building source cache from manifest..."
& $Python build_source_cache_from_manifest.py `
  --manifest $Manifest

$ManifestYaml = (& $Python yaml_to_json.py --file $Manifest) | ConvertFrom-Json
$CacheDir = $ManifestYaml.cache_dir
$CacheTag = $ManifestYaml.cache_tag
$Interval = "15m"
if ($ManifestYaml.binance -and $ManifestYaml.binance.interval) {
  $Interval = $ManifestYaml.binance.interval
}

Write-Host "[2/2] Running source family suite..."
& .\scripts\run_source_family_suite.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -Interval $Interval `
  -Folds $Folds
