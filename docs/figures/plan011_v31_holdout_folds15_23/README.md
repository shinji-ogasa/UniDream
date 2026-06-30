# Plan011 v31 Fold15-23 Trade Charts

保存済み checkpoint から実モデル推論を再実行し、test split の資産推移、B&H、position 変更点を可視化した。

## Reproduction

```bash
uv run python -m unidream.cli.plot_plan011_fold_trades \
  --config configs/plan011_overlay_actor_v31_holdout.yaml \
  --checkpoint-dir checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007 \
  --folds 15-23 \
  --seed 7 \
  --device cpu \
  --output-dir docs/figures/plan011_v31_holdout_folds15_23
```

## Files

- metrics: `docs/figures/plan011_v31_holdout_folds15_23/metrics.csv`
- trades: `docs/figures/plan011_v31_holdout_folds15_23/trades.csv`
- compressed time series: `docs/figures/plan011_v31_holdout_folds15_23/timeseries.npz`
- trade_eps: `0.0005`
- active_eps: `0.005`

## Fold Summary

| fold | period | AlphaEx | MaxDDDelta | turnover | trade points | active blocks | chart |
|---:|---|---:|---:|---:|---:|---:|---|
| 15 | 2024-01-16 13:45:00 to 2024-04-16 13:45:00 | +0.90pt | +0.21pt | 0.52 | 32 | 7 | [fold_15_equity_trades.png](fold_15_equity_trades.png) |
| 16 | 2024-04-16 13:45:00 to 2024-07-16 13:45:00 | -0.12pt | +0.28pt | 0.41 | 0 | 5 | [fold_16_equity_trades.png](fold_16_equity_trades.png) |
| 17 | 2024-07-16 13:45:00 to 2024-10-16 13:45:00 | -0.02pt | +0.03pt | 1.09 | 206 | 58 | [fold_17_equity_trades.png](fold_17_equity_trades.png) |
| 18 | 2024-10-16 13:45:00 to 2025-01-16 13:45:00 | +0.65pt | +0.12pt | 0.14 | 0 | 6 | [fold_18_equity_trades.png](fold_18_equity_trades.png) |
| 19 | 2025-01-16 13:45:00 to 2025-04-16 13:45:00 | -0.31pt | +0.35pt | 0.44 | 0 | 8 | [fold_19_equity_trades.png](fold_19_equity_trades.png) |
| 20 | 2025-04-16 13:45:00 to 2025-07-16 13:45:00 | +0.32pt | +0.14pt | 1.73 | 464 | 18 | [fold_20_equity_trades.png](fold_20_equity_trades.png) |
| 21 | 2025-07-16 13:45:00 to 2025-10-16 13:45:00 | -0.04pt | +0.11pt | 0.50 | 4 | 7 | [fold_21_equity_trades.png](fold_21_equity_trades.png) |
| 22 | 2025-10-16 13:45:00 to 2026-01-16 13:45:00 | -0.20pt | +0.37pt | 0.34 | 0 | 1 | [fold_22_equity_trades.png](fold_22_equity_trades.png) |
| 23 | 2026-01-16 13:45:00 to 2026-04-16 13:45:00 | -0.14pt | +0.17pt | 0.23 | 0 | 19 | [fold_23_equity_trades.png](fold_23_equity_trades.png) |

AlphaEx は strategy の最終total return minus B&Hの最終total return。年率換算ではない。
MaxDDDelta は strategy の絶対MaxDD minus B&Hの絶対MaxDD。マイナスが改善。
取引点は exposure が前バーから `trade_eps` 以上変化したバー。active block は `abs(exposure - 1.0) > active_eps` の連続区間。
全バーのposition系列は `timeseries.npz` に保存している。
