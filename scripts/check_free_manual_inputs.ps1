param(
  [string]$ZipDir = "raw\binance_klines",
  [string]$CoinMetricsJson = "raw\coinmetrics\btc_adractcnt.json",
  [string]$Symbol = "BTCUSDT",
  [string]$Interval = "15m",
  [string]$Start = "2021-01-01",
  [string]$End = "2023-06-01"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

& .\.venv\Scripts\python.exe .\inspect_free_manual_inputs.py `
  --zip-dir $ZipDir `
  --coinmetrics-json $CoinMetricsJson `
  --symbol $Symbol `
  --interval $Interval `
  --start $Start `
  --end $End
