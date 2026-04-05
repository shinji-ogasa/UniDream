param(
  [string]$CacheDir = "checkpoints\\aux_smoke2",
  [string]$CacheTag = "BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2",
  [string]$SuiteConfig = "configs\\source_rollout_suite.yaml",
  [string]$Manifest = "configs\\source_manifest_remote_example.yaml",
  [string]$SnapshotOutput = "checkpoints\\source_rollout_snapshot.json",
  [string]$Python = ".\\.venv\\Scripts\\python.exe"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Unit tests"
& $Python -m unittest tests.test_source_rollout_helpers
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/5] Py-compile"
& $Python -m py_compile `
  source_rollout_plan.py `
  check_config_source_requirements.py `
  validate_source_rollout_suite.py `
  validate_source_manifest.py `
  build_source_cache_from_manifest.py `
  build_coinmetrics_source_cache.py `
  build_glassnode_source_cache.py `
  build_binance_source_cache.py `
  tests\test_source_rollout_helpers.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/6] Validate suite YAML"
& $Python validate_source_rollout_suite.py `
  --suite-config $SuiteConfig
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/7] Validate remote manifest"
& $Python validate_source_manifest.py `
  --manifest $Manifest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[5/7] Rollout diagnosis"
powershell -ExecutionPolicy Bypass -File .\scripts\diagnose_source_rollout.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -SuiteConfig $SuiteConfig
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[6/7] Next-stage planning"
powershell -ExecutionPolicy Bypass -File .\scripts\plan_next_source_rollout.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -SuiteConfig $SuiteConfig `
  -OutManifest configs\source_manifest_next_stage.yaml `
  -Python $Python
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[7/8] Remote manifest dry-run"
& $Python build_source_cache_from_manifest.py `
  --manifest $Manifest `
  --dry-run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[8/8] Write rollout snapshot"
powershell -ExecutionPolicy Bypass -File .\scripts\write_source_rollout_snapshot.ps1 `
  -CacheDir $CacheDir `
  -CacheTag $CacheTag `
  -SuiteConfig $SuiteConfig `
  -Output $SnapshotOutput
exit $LASTEXITCODE
