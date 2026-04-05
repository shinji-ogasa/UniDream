# Free Source Rollout

This path is for the zero-cost source family rollout:

- `basis` from Binance spot/futures public REST
- `signed_order_flow` / `taker_imbalance` from Binance spot kline taker-buy fields
- `active_address_growth` from Coin Metrics Community API

Use the one-shot runner:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_free_source_rollout_end_to_end.ps1
```

This performs:

1. Fetch free raw sources into `checkpoints\basis_source_cache`
2. Run the free source-family suite from `configs\source_rollout_suite_free.yaml`
3. Select the best free source family into `checkpoints\source_family_suite_free\best_source_family.md`
4. Write a rollout snapshot into `checkpoints\source_family_suite_free\free_source_rollout_snapshot.json`

If Coin Metrics fetch fails via Python, the fetch path falls back to PowerShell download plus local import.
