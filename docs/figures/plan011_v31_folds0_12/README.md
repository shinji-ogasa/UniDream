# Plan011 v31 Fold0-12 Trade Charts

保存済み checkpoint から実モデル推論を再実行し、test split の資産推移、B&H、position 変更点を可視化した。

## Reproduction

```bash
uv run python -m unidream.cli.plot_plan011_fold_trades \
  --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml \
  --checkpoint-dir checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007 \
  --folds 0-12 \
  --seed 7 \
  --device cpu \
  --output-dir docs/figures/plan011_v31_folds0_12
```

## Files

- metrics: `docs/figures/plan011_v31_folds0_12/metrics.csv`
- trades: `docs/figures/plan011_v31_folds0_12/trades.csv`
- compressed time series: `docs/figures/plan011_v31_folds0_12/timeseries.npz`
- trade_eps: `0.0005`
- active_eps: `0.005`

## Fold Summary

| fold | period | AlphaEx | MaxDDDelta | turnover | trade points | active blocks | chart |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | 2020-04-16 13:45:00 to 2020-07-16 13:45:00 | +2.44pt | +0.02pt | 1.06 | 176 | 68 | [fold_00_equity_trades.png](fold_00_equity_trades.png) |
| 1 | 2020-07-16 13:45:00 to 2020-10-16 13:45:00 | +1.64pt | +0.06pt | 0.19 | 7 | 6 | [fold_01_equity_trades.png](fold_01_equity_trades.png) |
| 2 | 2020-10-16 13:45:00 to 2021-01-16 13:45:00 | +477.79pt | +0.05pt | 0.77 | 33 | 25 | [fold_02_equity_trades.png](fold_02_equity_trades.png) |
| 3 | 2021-01-16 13:45:00 to 2021-04-16 13:45:00 | +18.86pt | +0.22pt | 0.56 | 15 | 8 | [fold_03_equity_trades.png](fold_03_equity_trades.png) |
| 4 | 2021-04-16 13:45:00 to 2021-07-16 13:45:00 | -0.20pt | +0.32pt | 0.53 | 4 | 9 | [fold_04_equity_trades.png](fold_04_equity_trades.png) |
| 5 | 2021-07-16 13:45:00 to 2021-10-16 13:45:00 | +39.17pt | +0.25pt | 0.48 | 1 | 4 | [fold_05_equity_trades.png](fold_05_equity_trades.png) |
| 6 | 2021-10-16 13:45:00 to 2022-01-16 13:45:00 | -0.32pt | +0.28pt | 0.52 | 28 | 14 | [fold_06_equity_trades.png](fold_06_equity_trades.png) |
| 7 | 2022-01-16 13:45:00 to 2022-04-16 13:45:00 | -1.28pt | +0.24pt | 0.55 | 18 | 13 | [fold_07_equity_trades.png](fold_07_equity_trades.png) |
| 8 | 2022-04-16 13:45:00 to 2022-07-16 13:45:00 | -0.22pt | +0.39pt | 0.09 | 0 | 1 | [fold_08_equity_trades.png](fold_08_equity_trades.png) |
| 9 | 2022-07-16 13:45:00 to 2022-10-16 13:45:00 | -0.53pt | +0.26pt | 0.07 | 0 | 1 | [fold_09_equity_trades.png](fold_09_equity_trades.png) |
| 10 | 2022-10-16 13:45:00 to 2023-01-16 13:45:00 | +0.81pt | +0.15pt | 0.35 | 0 | 11 | [fold_10_equity_trades.png](fold_10_equity_trades.png) |
| 11 | 2023-01-16 13:45:00 to 2023-04-16 13:45:00 | +5.61pt | +0.17pt | 0.08 | 0 | 1 | [fold_11_equity_trades.png](fold_11_equity_trades.png) |
| 12 | 2023-04-16 13:45:00 to 2023-07-16 13:45:00 | -0.45pt | +0.22pt | 0.49 | 8 | 5 | [fold_12_equity_trades.png](fold_12_equity_trades.png) |

MaxDDDelta は strategy の絶対MaxDD minus B&Hの絶対MaxDD。マイナスが改善。
取引点は exposure が前バーから `trade_eps` 以上変化したバー。active block は `abs(exposure - 1.0) > active_eps` の連続区間。
全バーのposition系列は `timeseries.npz` に保存している。
