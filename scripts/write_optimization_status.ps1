param(
  [string]$JsonOutput = "checkpoints\\optimization_status.json",
  [string]$MdOutput = "checkpoints\\optimization_status.md"
)

$ErrorActionPreference = "Stop"

uv run python write_optimization_status.py `
  --json-output $JsonOutput `
  --md-output $MdOutput
