param(
  [string]$SuiteDir = "checkpoints\\source_family_suite",
  [string]$Summary = "",
  [string]$Output = "checkpoints\\source_family_suite\\best_source_family.md"
)

$ErrorActionPreference = "Stop"

if (-not $Summary) {
  $Summary = Join-Path $SuiteDir "suite_summary.csv"
}

uv run python select_best_source_family.py `
  --summary $Summary `
  --output $Output
