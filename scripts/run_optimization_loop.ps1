param(
  [switch]$CheckUv,
  [switch]$RunIssue2,
  [switch]$RunIssue3,
  [switch]$RunIssue4,
  [switch]$RunIssue5,
  [switch]$RunIssue6
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not ($CheckUv -or $RunIssue2 -or $RunIssue3 -or $RunIssue4 -or $RunIssue5 -or $RunIssue6)) {
  Write-Host "[Loop] No issue flag selected."
  Write-Host "[Loop] Available flags: -CheckUv -RunIssue2 -RunIssue3 -RunIssue4 -RunIssue5 -RunIssue6"
  exit 0
}

if ($CheckUv) {
  Write-Host "[Loop] UV runtime check"
  & powershell -ExecutionPolicy Bypass -File .\scripts\check_uv_runtime.ps1
}

if ($RunIssue2) {
  Write-Host "[Loop] Issue2"
  & powershell -ExecutionPolicy Bypass -File .\scripts\run_issue2_bc_prior_loop.ps1
}

if ($RunIssue3) {
  Write-Host "[Loop] Issue3"
  & powershell -ExecutionPolicy Bypass -File .\scripts\run_issue3_ac_support_loop.ps1
}

if ($RunIssue4) {
  Write-Host "[Loop] Issue4"
  & powershell -ExecutionPolicy Bypass -File .\scripts\run_issue4_wm_regime_loop.ps1
}

if ($RunIssue5) {
  Write-Host "[Loop] Issue5"
  & powershell -ExecutionPolicy Bypass -File .\scripts\run_issue5_conservative_ac_loop.ps1
}

if ($RunIssue6) {
  Write-Host "[Loop] Issue6"
  & powershell -ExecutionPolicy Bypass -File .\scripts\run_issue6_external_source_loop.ps1
}
