param(
  [string]$Output = "checkpoints\\uv_runtime_check.txt",
  [string]$ProjectCacheDir = ".uv-cache",
  [string]$PythonPath = "C:\\Users\\Sophie\\anaconda3\\envs\\UniDream\\python.exe"
)

$ErrorActionPreference = "Continue"
$projectCache = Join-Path (Get-Location) $ProjectCacheDir
New-Item -ItemType Directory -Force -Path $projectCache | Out-Null
$env:UV_CACHE_DIR = $projectCache

$lines = New-Object System.Collections.Generic.List[string]

function Add-Result {
  param(
    [string]$Label,
    [scriptblock]$Action
  )

  $lines.Add("## $Label")
  try {
    $result = & $Action 2>&1 | Out-String
    $lines.Add($result.TrimEnd())
  } catch {
    $err = ($_ | Out-String).TrimEnd()
    $lines.Add($err)
  }
  $lines.Add("")
}

Add-Result "uv --version" { uv --version }
Add-Result "UV_CACHE_DIR" { Write-Output $env:UV_CACHE_DIR }
Add-Result "uv python find" { uv python find }
Add-Result "uv run python -V" { uv run python -V }
Add-Result "uv run python -c import sys; print(sys.executable)" { uv run python -c "import sys; print(sys.executable)" }
Add-Result "uv run --python <UniDream> python -V" { uv run --python $PythonPath python -V }

$outPath = Join-Path (Get-Location) $Output
$outDir = Split-Path -Parent $outPath
if ($outDir) {
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}
$lines -join "`r`n" | Set-Content -Path $outPath -Encoding UTF8
Write-Host "[UV] wrote $outPath"
