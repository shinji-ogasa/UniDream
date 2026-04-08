param(
  [string]$Output = "checkpoints\\uv_runtime_check.txt"
)

$ErrorActionPreference = "Continue"

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
    $lines.Add($_ | Out-String)
  }
  $lines.Add("")
}

Add-Result "uv --version" { uv --version }
Add-Result "uv python find" { uv python find }
Add-Result "uv run python -V" { uv run python -V }
Add-Result "uv run python -c import sys; print(sys.executable)" { uv run python -c "import sys; print(sys.executable)" }

$outPath = Join-Path (Get-Location) $Output
$outDir = Split-Path -Parent $outPath
if ($outDir) {
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}
$lines -join "`r`n" | Set-Content -Path $outPath -Encoding UTF8
Write-Host "[UV] wrote $outPath"
