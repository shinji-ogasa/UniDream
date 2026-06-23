# Plan011 v31 Investor Evidence Snapshot

このメモはVC/外部説明向けの検証証跡。LP本文用のコピーではなく、結果・再現性・限界を確認するためのソースとして扱う。

## 主成果

Plan011 v31 は、B&H exposure `1.0` を基準にした低回転 overlay actor。主張できる成果は以下に絞る。

> B&H近傍の低回転overlayとして、13fold testで大崩れを避けながらAlphaExの右側テールを獲得した。

現時点では、DD改善AIやリスク調整後リターン改善AIとしては主張しない。0-12集計の `MaxDDDelta` はプラス寄りで、DDを下げるよりAlphaExを取りにいく挙動になっている。

2024-2026 の完全未使用 holdout fold15-23 では aggregate `AlphaEx +2.32pt`、`SharpeDelta -0.003`、`MaxDDDelta +0.20pt`。低回転と一部foldのAlphaEx右テールは残ったが、DD改善は残っていない。

## Locked Spec

- Config: `configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml`
- Checkpoint dir: `checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007`
- Result commit before evidence docs: `caa2695`
- Symbol / interval: `BTCUSDT`, `15m`
- Date range: `2018-01-01` to `2024-01-01`
- Fold range: `0-12`
- Seed: `7`
- Device used for final reproduction: `mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Costs: default profile, one-way full `Δpos=1` cost `5.50bps`

Each fold uses:

- train: 2 years
- validation: 3 months
- test: 3 months
- slide: 3 months

Validation selects only inference adjustment scale. Test remains report-only; the strict reproduction run recreates WM/BC/AC before evaluating it.

## Reproduction Commands

The strict runner retrains WM, BC and AC from scratch. Reproduce fold0-12 with the self-contained v31 config:

```bash
uv run python -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml \
  --seed 7 \
  --device cuda
```

CPU fallback:

```bash
uv run python -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml \
  --seed 7 \
  --device cpu
```

## Fold Results

| fold | test period | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | turnover | verdict |
|---:|---|---:|---:|---:|---:|---|
| 0 | 2020-04-16 to 2020-07-16 | +2.44 | +0.009 | +0.02 | 1.06 | small positive |
| 1 | 2020-07-16 to 2020-10-16 | +1.64 | +0.009 | +0.06 | 0.19 | small positive |
| 2 | 2020-10-16 to 2021-01-16 | +477.79 | +0.020 | +0.05 | 0.77 | large positive outlier |
| 3 | 2021-01-16 to 2021-04-16 | +18.86 | +0.009 | +0.22 | 0.56 | positive |
| 4 | 2021-04-16 to 2021-07-16 | -0.20 | -0.010 | +0.32 | 0.53 | flat / near neutral |
| 5 | 2021-07-16 to 2021-10-16 | +39.17 | -0.005 | +0.25 | 0.48 | positive |
| 6 | 2021-10-16 to 2022-01-16 | -0.32 | -0.000 | +0.28 | 0.52 | flat / near neutral |
| 7 | 2022-01-16 to 2022-04-16 | -1.28 | -0.021 | +0.24 | 0.55 | worst AlphaEx |
| 8 | 2022-04-16 to 2022-07-16 | -0.22 | -0.006 | +0.39 | 0.09 | flat / near neutral |
| 9 | 2022-07-16 to 2022-10-16 | -0.53 | -0.006 | +0.26 | 0.07 | flat / near neutral |
| 10 | 2022-10-16 to 2023-01-16 | +0.81 | +0.007 | +0.15 | 0.35 | small positive |
| 11 | 2023-01-16 to 2023-04-16 | +5.61 | -0.004 | +0.17 | 0.08 | positive |
| 12 | 2023-04-16 to 2023-07-16 | -0.45 | -0.011 | +0.22 | 0.49 | flat / near neutral |

Aggregate reported by the evaluation run:

- `AlphaEx +41.79pt`
- `SharpeDelta -0.001`
- `MaxDDDelta +0.20pt`
- `PBO 0.420`
- worst AlphaEx: `-1.28pt`

## Distribution Stats

AlphaEx distribution:

- mean: `+41.79pt`
- median: `+0.81pt`
- worst: `-1.28pt` (fold7)
- best: `+477.79pt` (fold2)
- mean excluding fold2: `+5.46pt`
- median excluding fold2: `+0.31pt`
- AlphaEx `> 0`: `7/13`
- AlphaEx `>= +3pt`: `4/13`
- `abs(AlphaEx) <= 1pt`: `6/13`
- AlphaEx `>= 0` or `abs(AlphaEx) <= 1pt`: `12/13`
- fold-resampling bootstrap 95% CI of mean: `[+1.03pt, +116.88pt]`
- fold2-excluded bootstrap 95% CI of mean: `[+0.30pt, +13.03pt]`

MaxDDDelta distribution:

- mean: `+0.20pt`
- median: `+0.22pt`
- best: `+0.02pt` (fold0)
- worst: `+0.39pt` (fold8)
- MaxDDDelta `<= 0`: `0/13`
- MaxDDDelta `<= +0.30pt`: `11/13`

Interpretation:

- The mean AlphaEx is strong, but fold2 is a large positive outlier.
- The robust read is not "all folds produce large alpha"; it is "most folds stay near neutral or positive, with right-tail upside."
- The model does not improve drawdown yet. DD is slightly worse than B&H in every fold, although the deterioration is small in this v31 run.

## Reproducibility Notes

- `checkpoints/` is git-ignored. Reproduce checkpoints with the config above, or preserve the local checkpoint directory when auditing.
- Validation is used for inference adjustment scale selection. Test split is report-only within each fold.
- Fold0-12 are development WFO folds, not a pristine post-lock holdout.
- Fold15-23 are the current untouched 2024-2026 holdout, ending at `2026-04-16` and excluding roughly the latest 60 days at run time.
- Live / Space adoption is wired to the Plan011 v31 fold23 neural bundle. `/sample/verify` passes with `strict_ok=True`.
- This result should be presented as model research evidence, not as a live trading track record.

## Fold0-12 Trade / Equity Charts

Saved checkpoint inference has been replayed for fold0-12, including validation-time `infer_adjust_rate_scale` selection. The generated artifacts are:

- Chart index: `docs/figures/plan011_v31_folds0_12/README.md`
- Per-fold equity / B&H / position-change PNGs: `docs/figures/plan011_v31_folds0_12/fold_XX_equity_trades.png`
- Metrics CSV: `docs/figures/plan011_v31_folds0_12/metrics.csv`
- Position-change events CSV: `docs/figures/plan011_v31_folds0_12/trades.csv`
- Full per-bar arrays: `docs/figures/plan011_v31_folds0_12/timeseries.npz`

Reproduce the charts:

```bash
uv run python -m unidream.cli.plot_plan011_fold_trades \
  --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml \
  --checkpoint-dir checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007 \
  --folds 0-12 \
  --seed 7 \
  --device cpu \
  --output-dir docs/figures/plan011_v31_folds0_12
```

## Untouched Holdout 2024-2026

The locked v31 spec was retrained/evaluated on folds 15-23 with data through `2026-04-17`. Current date during the run was `2026-06-16 JST`, so the latest roughly 60 days were left outside this holdout and live bundle export.

Reproduction:

```bash
uv run python -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v31_holdout.yaml \
  --seed 7 \
  --device cuda
```

| fold | test period | AlphaEx pt/yr | SharpeDelta | MaxDDDelta pt | turnover |
|---:|---|---:|---:|---:|---:|
| 15 | 2024-01-16 to 2024-04-16 | +11.35 | +0.006 | +0.21 | 0.52 |
| 16 | 2024-04-16 to 2024-07-16 | -0.51 | -0.012 | +0.28 | 0.41 |
| 17 | 2024-07-16 to 2024-10-16 | -0.10 | -0.001 | +0.03 | 1.09 |
| 18 | 2024-10-16 to 2025-01-16 | +8.17 | +0.003 | +0.12 | 0.14 |
| 19 | 2025-01-16 to 2025-04-16 | -0.74 | -0.014 | +0.35 | 0.44 |
| 20 | 2025-04-16 to 2025-07-16 | +3.64 | -0.014 | +0.14 | 1.73 |
| 21 | 2025-07-16 to 2025-10-16 | -0.14 | +0.003 | +0.11 | 0.50 |
| 22 | 2025-10-16 to 2026-01-16 | -0.51 | -0.001 | +0.37 | 0.34 |
| 23 | 2026-01-16 to 2026-04-16 | -0.26 | -0.002 | +0.17 | 0.23 |

Aggregate:

- `AlphaEx +2.32pt`
- `SharpeDelta -0.003`
- `MaxDDDelta +0.20pt`
- `PBO 0.400`
- AlphaEx `> 0`: `3/9`
- AlphaEx `>= +3pt`: `3/9`
- MaxDDDelta `<= 0`: `0/9`
- turnover mean: `0.60`

Interpretation:

- Low turnover survived on untouched data.
- Aggregate AlphaEx stayed positive, but the median fold was slightly negative.
- DD improvement did not survive. Do not claim `MaxDDDelta <= -3pt`, or even `MaxDDDelta <= 0`, on untouched holdout.
- The live Space bundle now uses fold23 as the latest no-leak deployment candidate.

## What This Does Not Prove Yet

- It does not prove drawdown improvement: aggregate `MaxDDDelta` is `+0.20pt`.
- It does not prove Sharpe improvement: aggregate `SharpeDelta` is `-0.001`.
- It does not isolate Transformer WM contribution. WM/no-WM and shuffled-latent ablations are still needed.
- It does not prove production-grade robustness. The untouched holdout is positive on aggregate AlphaEx but not on drawdown or Sharpe.
- It does not prove execution capacity, market impact robustness, or live-only performance.

## Next Evidence To Add

Highest-priority additions for investor diligence:

1. Policy-family ablation completed: see `docs/policy_family_holdout_comparison.md` for simple vol-target vs tabular ML vs WM-only vs WM+BC (ACなし).
2. Same-turnover / same-exposure random overlay baseline.
3. Live paper-trading immutable log: timestamp, model hash, input timestamp, position, fee/slippage assumption, realized B&H comparison.
4. Fold-level cumulative wealth and drawdown charts.
5. A reward/selector revision specifically targeting MaxDDDelta, because v31 does not reduce DD on holdout.
