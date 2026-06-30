# Plan011 v31 Investor Evidence Snapshot

このメモはVC/外部説明向けの検証証跡。LP本文用のコピーではなく、結果・再現性・限界を確認するためのソースとして扱う。

## 主成果

Plan011 v31 は、B&H exposure `1.0` を基準にした低回転 overlay actor。現在の正しい評価定義では、主張できる成果は以下に絞る。

> 実モデル推論の WM -> BC -> AC pipeline を live/demo runtime に接続し、B&H近傍の低回転 neural overlay として collapse を避ける研究証跡を作った。

現時点では、DD改善AI、リスク調整後リターン改善AI、または `AlphaEx >= +3pt` / `MaxDDDelta <= -3pt` 達成モデルとしては主張しない。

AlphaEx は strategy の最終total return minus B&Hの最終total return。年率換算ではない。MaxDDDelta は strategy の絶対MaxDD minus B&Hの絶対MaxDDで、マイナスが改善。

## Locked Spec

- Config: `configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml`
- Checkpoint dir: `checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007`
- Symbol / interval: `BTCUSDT`, `15m`
- Development date range: `2018-01-01` to `2024-01-01`
- Development fold range: `0-12`
- Holdout config: `configs/plan011_overlay_actor_v31_holdout.yaml`
- Holdout range: `2024-01-16` to `2026-04-16` (`fold15-23`)
- Seed: `7`
- Replay device used for evidence: `cpu`
- Costs: default profile, one-way full `Delta position = 1` cost `5.50bps`

Each fold uses:

- train: 2 years
- validation: 3 months
- test: 3 months
- slide: 3 months

Validation selects inference adjustment scale. Test remains report-only.

## Reproduction Commands

The strict runner retrains WM, BC and AC from scratch. Reproduce fold0-12 with the self-contained v31 config:

```bash
uv run python -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml \
  --seed 7 \
  --device cuda
```

Replay saved fold0-12 checkpoint inference and regenerate charts:

```bash
uv run python -m unidream.cli.plot_plan011_fold_trades \
  --config configs/plan011_overlay_actor_v31_relative_constraint_ac.yaml \
  --checkpoint-dir checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007 \
  --folds 0-12 \
  --seed 7 \
  --device cpu \
  --output-dir docs/figures/plan011_v31_folds0_12
```

Replay saved holdout fold15-23 checkpoint inference and regenerate charts:

```bash
uv run python -m unidream.cli.plot_plan011_fold_trades \
  --config configs/plan011_overlay_actor_v31_holdout.yaml \
  --checkpoint-dir checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007 \
  --folds 15-23 \
  --seed 7 \
  --device cpu \
  --output-dir docs/figures/plan011_v31_holdout_folds15_23
```

## Fold0-12 Results

| fold | test period | AlphaEx | SharpeDelta | MaxDDDelta | turnover | verdict |
|---:|---|---:|---:|---:|---:|---|
| 0 | 2020-04-16 to 2020-07-16 | +0.28pt | +0.009 | +0.02pt | 1.06 | small positive |
| 1 | 2020-07-16 to 2020-10-16 | +0.21pt | +0.009 | +0.06pt | 0.19 | small positive |
| 2 | 2020-10-16 to 2021-01-16 | +3.46pt | +0.020 | +0.05pt | 0.77 | best alpha |
| 3 | 2021-01-16 to 2021-04-16 | +1.05pt | +0.009 | +0.22pt | 0.56 | positive |
| 4 | 2021-04-16 to 2021-07-16 | -0.35pt | -0.010 | +0.32pt | 0.53 | small negative |
| 5 | 2021-07-16 to 2021-10-16 | +1.40pt | -0.005 | +0.25pt | 0.48 | positive |
| 6 | 2021-10-16 to 2022-01-16 | -0.23pt | -0.000 | +0.28pt | 0.52 | small negative |
| 7 | 2022-01-16 to 2022-04-16 | -0.38pt | -0.021 | +0.24pt | 0.55 | small negative |
| 8 | 2022-04-16 to 2022-07-16 | -0.40pt | -0.006 | +0.39pt | 0.09 | worst alpha |
| 9 | 2022-07-16 to 2022-10-16 | -0.17pt | -0.006 | +0.26pt | 0.07 | near neutral |
| 10 | 2022-10-16 to 2023-01-16 | +0.16pt | +0.007 | +0.15pt | 0.35 | near neutral |
| 11 | 2023-01-16 to 2023-04-16 | +0.44pt | -0.004 | +0.17pt | 0.08 | small positive |
| 12 | 2023-04-16 to 2023-07-16 | -0.11pt | -0.011 | +0.22pt | 0.49 | near neutral |

Aggregate:

- AlphaEx mean: `+0.41pt`
- AlphaEx median: `+0.16pt`
- AlphaEx best / worst: `+3.46pt / -0.40pt`
- AlphaEx `> 0`: `7/13`
- AlphaEx `>= +3pt`: `1/13`
- MaxDDDelta mean: `+0.20pt`
- MaxDDDelta median: `+0.22pt`
- MaxDDDelta `<= 0`: `0/13`
- MaxDDDelta `<= -3pt`: `0/13`
- Goal pass (`AlphaEx >= +3pt` and `MaxDDDelta <= -3pt`): `0/13`
- Turnover mean: `0.44`

## Untouched Holdout 2024-2026

| fold | test period | AlphaEx | SharpeDelta | MaxDDDelta | turnover |
|---:|---|---:|---:|---:|---:|
| 15 | 2024-01-16 to 2024-04-16 | +0.90pt | +0.006 | +0.21pt | 0.52 |
| 16 | 2024-04-16 to 2024-07-16 | -0.12pt | -0.012 | +0.28pt | 0.41 |
| 17 | 2024-07-16 to 2024-10-16 | -0.02pt | -0.000 | +0.03pt | 1.09 |
| 18 | 2024-10-16 to 2025-01-16 | +0.65pt | +0.003 | +0.12pt | 0.14 |
| 19 | 2025-01-16 to 2025-04-16 | -0.31pt | -0.014 | +0.35pt | 0.44 |
| 20 | 2025-04-16 to 2025-07-16 | +0.32pt | -0.014 | +0.14pt | 1.73 |
| 21 | 2025-07-16 to 2025-10-16 | -0.04pt | +0.003 | +0.11pt | 0.50 |
| 22 | 2025-10-16 to 2026-01-16 | -0.20pt | -0.001 | +0.37pt | 0.34 |
| 23 | 2026-01-16 to 2026-04-16 | -0.14pt | -0.002 | +0.17pt | 0.23 |

Aggregate:

- AlphaEx mean: `+0.11pt`
- AlphaEx median: `-0.04pt`
- AlphaEx best / worst: `+0.90pt / -0.31pt`
- AlphaEx `> 0`: `3/9`
- AlphaEx `>= +3pt`: `0/9`
- MaxDDDelta mean: `+0.20pt`
- MaxDDDelta median: `+0.17pt`
- MaxDDDelta `<= 0`: `0/9`
- MaxDDDelta `<= -3pt`: `0/9`
- Goal pass (`AlphaEx >= +3pt` and `MaxDDDelta <= -3pt`): `0/9`
- Turnover mean: `0.60`

## Policy-Family Ablation

完全未使用 holdout fold15-23 を同一cost・B&H基準で比較した。testはreport-only。

| method | AlphaEx | MaxDDDelta | median AlphaEx | worst AlphaEx | DD improved |
|---|---:|---:|---:|---:|---:|
| B&H | +0.00pt | +0.00pt | +0.00pt | +0.00pt | 0/9 |
| 単純アルゴリズム (causal vol-target) | -1.27pt | -1.66pt | -0.62pt | -14.79pt | 5/9 |
| ML (HistGradientBoosting) | -0.38pt | -0.36pt | -0.12pt | -1.81pt | 9/9 |
| WMのみ (position-utility allocator) | -1.63pt | -1.33pt | -0.48pt | -7.19pt | 9/9 |
| BCのみ (WM+BC, ACなし) | +0.13pt | +0.24pt | -0.12pt | -0.32pt | 0/9 |

Read:

- WM-only / tabular ML / simple vol-target はDD改善方向のsignalを持つが、alphaを失う。
- WM+BC はalphaを残すが、DD改善を失う。
- AC込みv31はBC-onlyに対する優位性が限定的。
- 次の改善対象は、報酬関数とselectorを final AlphaEx + DD改善の同時達成へ寄せること。

## Artifacts

- Fold0-12 chart index: `docs/figures/plan011_v31_folds0_12/README.md`
- Holdout chart index: `docs/figures/plan011_v31_holdout_folds15_23/README.md`
- Policy-family comparison: `docs/policy_family_holdout_comparison.md`
- Metrics CSVs:
  - `docs/figures/plan011_v31_folds0_12/metrics.csv`
  - `docs/figures/plan011_v31_holdout_folds15_23/metrics.csv`
- Position-change events CSVs:
  - `docs/figures/plan011_v31_folds0_12/trades.csv`
  - `docs/figures/plan011_v31_holdout_folds15_23/trades.csv`

## What This Does Not Prove Yet

- It does not prove drawdown improvement: aggregate `MaxDDDelta` is positive in both fold0-12 and holdout.
- It does not prove Sharpe improvement.
- It does not meet `AlphaEx >= +3pt` and `MaxDDDelta <= -3pt`.
- It does not prove production-grade robustness.
- It does not prove execution capacity, market impact robustness, or live-only performance.

## Next Evidence To Add

Highest-priority additions:

1. Reward/selector revision that directly optimizes final-value AlphaEx and MaxDDDelta.
2. Same-turnover / same-exposure random overlay baseline.
3. Shuffled-latent and no-WM ablations to isolate Transformer WM contribution.
4. Live paper-trading immutable log: timestamp, model hash, input timestamp, position, fee/slippage assumption, realized B&H comparison.
