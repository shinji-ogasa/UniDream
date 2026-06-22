# Policy Family Holdout Comparison

- period: `2024-01-16 13:45:00` to `2026-04-16 13:45:00`
- folds: `15, 16, 17, 18, 19, 20, 21, 22, 23`
- config: `configs/plan011_overlay_actor_v31_holdout.yaml`
- seed/device: `7` / `cpu`
- selection: train fit + validation selection only; test is report-only
- benchmark: B&H exposure=1.0
- checkpoint source: `checkpoints/plan011_overlay_actor_v31_relative_constraint_ac_s007/fold_{15..23}`

## Reproduction

```bash
uv run python -m unidream.cli.compare_policy_families \
  --config configs/plan011_overlay_actor_v31_holdout.yaml \
  --seed 7 \
  --device cpu \
  --output docs/policy_family_holdout_comparison
```

`docs/policy_family_holdout_comparison.json`にdata/source/config SHA256とfold別WM/BC checkpoint semantic SHA256を保存する。checkpoint自体はGit管理外なので、外部再現ではsemantic hashが一致するartifactを使用する。

## Summary (mean across 9 quarterly test folds)

| method | AlphaEx | MaxDDDelta | median AlphaEx | worst AlphaEx | DD improved | mean turnover |
|---|---:|---:|---:|---:|---:|---:|
| B&H | +0.00pt | +0.00pt | +0.00pt | +0.00pt | 0/9 | 0.00 |
| 単純アルゴリズム (causal vol-target) | -1.14pt | -1.44pt | -2.14pt | -90.36pt | 5/9 | 4.64 |
| ML (HistGradientBoosting) | -6.11pt | -0.34pt | -0.18pt | -26.32pt | 9/9 | 1.59 |
| WMのみ (position-utility allocator) | -17.41pt | -1.28pt | -1.33pt | -92.30pt | 9/9 | 4.93 |
| BCのみ (WM+BC, ACなし) | +2.83pt | +0.24pt | -0.51pt | -0.77pt | 0/9 | 0.44 |

## Findings

- 4方式のうち、平均AlphaExが正だったのはBC-onlyだけ。ただし中央値は`-0.51pt`で、プラスは3/9foldに限られる。
- BC-onlyはMaxDDDeltaが9/9foldでプラス。alphaを残す代わりにDD改善能力はない。
- WM-onlyはMaxDDDeltaを9/9foldで改善したが、fold18 `-92.30pt`、fold20 `-42.39pt`のAlphaEx損失が大きい。WM risk signalはDD方向に機能する一方、position-utilityを直接positionへ変換すると防御過多になる。
- tabular MLもMaxDDDeltaは9/9foldで改善したが、fold18/20の上昇取り逃しで平均AlphaExが`-6.11pt`。中央値`-0.18pt`なので通常期はB&H近傍だが、強い上昇局面への追随が課題。
- simple vol-targetはDD平均を改善したが、AlphaExの分散が最大。validation選択だけではregime間の安定性を確保できていない。
- 現時点では4方式のどれも、平均で`AlphaEx > 0`かつ`MaxDDDelta < 0`を同時達成していない。

## Fold Results

| fold | test period | method | AlphaEx | MaxDDDelta | turnover |
|---:|---|---|---:|---:|---:|
| 15 | 2024-01-16 13:45:00 to 2024-04-16 13:45:00 | 単純アルゴリズム (causal vol-target) | +51.63pt | -0.68pt | 7.60 |
| 15 | 2024-01-16 13:45:00 to 2024-04-16 13:45:00 | ML (HistGradientBoosting) | -9.72pt | -0.10pt | 1.23 |
| 15 | 2024-01-16 13:45:00 to 2024-04-16 13:45:00 | WMのみ (position-utility allocator) | -14.65pt | -0.41pt | 2.30 |
| 15 | 2024-01-16 13:45:00 to 2024-04-16 13:45:00 | BCのみ (WM+BC, ACなし) | +11.35pt | +0.21pt | 0.52 |
| 16 | 2024-04-16 13:45:00 to 2024-07-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -11.39pt | +2.28pt | 1.25 |
| 16 | 2024-04-16 13:45:00 to 2024-07-16 13:45:00 | ML (HistGradientBoosting) | -0.17pt | -0.18pt | 2.59 |
| 16 | 2024-04-16 13:45:00 to 2024-07-16 13:45:00 | WMのみ (position-utility allocator) | -1.33pt | -0.44pt | 4.07 |
| 16 | 2024-04-16 13:45:00 to 2024-07-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.51pt | +0.28pt | 0.41 |
| 17 | 2024-07-16 13:45:00 to 2024-10-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -2.14pt | -0.18pt | 0.94 |
| 17 | 2024-07-16 13:45:00 to 2024-10-16 13:45:00 | ML (HistGradientBoosting) | -0.56pt | -0.57pt | 1.08 |
| 17 | 2024-07-16 13:45:00 to 2024-10-16 13:45:00 | WMのみ (position-utility allocator) | -7.39pt | -2.27pt | 5.14 |
| 17 | 2024-07-16 13:45:00 to 2024-10-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.77pt | +0.21pt | 0.53 |
| 18 | 2024-10-16 13:45:00 to 2025-01-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -90.36pt | -2.76pt | 1.32 |
| 18 | 2024-10-16 13:45:00 to 2025-01-16 13:45:00 | ML (HistGradientBoosting) | -26.32pt | -0.05pt | 3.55 |
| 18 | 2024-10-16 13:45:00 to 2025-01-16 13:45:00 | WMのみ (position-utility allocator) | -92.30pt | -0.64pt | 9.68 |
| 18 | 2024-10-16 13:45:00 to 2025-01-16 13:45:00 | BCのみ (WM+BC, ACなし) | +9.31pt | +0.18pt | 0.45 |
| 19 | 2025-01-16 13:45:00 to 2025-04-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -3.73pt | +1.29pt | 1.32 |
| 19 | 2025-01-16 13:45:00 to 2025-04-16 13:45:00 | ML (HistGradientBoosting) | +0.40pt | -0.22pt | 0.80 |
| 19 | 2025-01-16 13:45:00 to 2025-04-16 13:45:00 | WMのみ (position-utility allocator) | -0.27pt | -0.50pt | 8.75 |
| 19 | 2025-01-16 13:45:00 to 2025-04-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.74pt | +0.35pt | 0.44 |
| 20 | 2025-04-16 13:45:00 to 2025-07-16 13:45:00 | 単純アルゴリズム (causal vol-target) | +30.32pt | +0.61pt | 1.21 |
| 20 | 2025-04-16 13:45:00 to 2025-07-16 13:45:00 | ML (HistGradientBoosting) | -20.10pt | -0.41pt | 1.88 |
| 20 | 2025-04-16 13:45:00 to 2025-07-16 13:45:00 | WMのみ (position-utility allocator) | -42.39pt | -0.95pt | 2.55 |
| 20 | 2025-04-16 13:45:00 to 2025-07-16 13:45:00 | BCのみ (WM+BC, ACなし) | +8.09pt | +0.19pt | 0.40 |
| 21 | 2025-07-16 13:45:00 to 2025-10-16 13:45:00 | 単純アルゴリズム (causal vol-target) | -2.23pt | +1.36pt | 0.94 |
| 21 | 2025-07-16 13:45:00 to 2025-10-16 13:45:00 | ML (HistGradientBoosting) | -0.18pt | -0.07pt | 0.93 |
| 21 | 2025-07-16 13:45:00 to 2025-10-16 13:45:00 | WMのみ (position-utility allocator) | +0.18pt | -0.28pt | 4.35 |
| 21 | 2025-07-16 13:45:00 to 2025-10-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.14pt | +0.11pt | 0.50 |
| 22 | 2025-10-16 13:45:00 to 2026-01-16 13:45:00 | 単純アルゴリズム (causal vol-target) | +8.61pt | -5.57pt | 18.13 |
| 22 | 2025-10-16 13:45:00 to 2026-01-16 13:45:00 | ML (HistGradientBoosting) | +0.78pt | -0.79pt | 1.13 |
| 22 | 2025-10-16 13:45:00 to 2026-01-16 13:45:00 | WMのみ (position-utility allocator) | -1.24pt | -3.40pt | 4.99 |
| 22 | 2025-10-16 13:45:00 to 2026-01-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.51pt | +0.37pt | 0.34 |
| 23 | 2026-01-16 13:45:00 to 2026-04-16 13:45:00 | 単純アルゴリズム (causal vol-target) | +9.07pt | -9.28pt | 9.04 |
| 23 | 2026-01-16 13:45:00 to 2026-04-16 13:45:00 | ML (HistGradientBoosting) | +0.92pt | -0.70pt | 1.06 |
| 23 | 2026-01-16 13:45:00 to 2026-04-16 13:45:00 | WMのみ (position-utility allocator) | +2.69pt | -2.67pt | 2.56 |
| 23 | 2026-01-16 13:45:00 to 2026-04-16 13:45:00 | BCのみ (WM+BC, ACなし) | -0.59pt | +0.30pt | 0.41 |

## Definitions

- Simple: past returns only. Realized-vol target and execution parameters are selected on validation.
- ML: HistGradientBoosting learns the training-only oracle position from causal tabular features; execution is selected on validation.
- WM only: the Transformer WM position-utility head selects exposure directly; no actor, BC, or AC is used.
- BC only: the Transformer WM encoder/predictive state and BC actor are used; the AC checkpoint is never loaded.
- MaxDDDelta = strategy absolute MaxDD minus B&H absolute MaxDD. Negative is improvement.
