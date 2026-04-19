# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

暗号資産トレード向けの World Model + Imagination RL 研究プロジェクトです。  
Hindsight Oracle から BC で初期方策を作り、Dreamer 系の世界モデル上で AC を回し、Buy & Hold に対する超過成績を狙います。

---

## 概要

UniDream は、BTCUSDT 15 分足を中心にした Walk-Forward 検証型の研究コードです。現在の主な構成は次の通りです。

- **入力**: OHLCV、最小 TA、basis、funding、order flow、on-chain 系の拡張ソース
- **世界モデル**: DreamerV3 系 + Block-Causal Transformer
- **方策**: inventory controller
- **学習**: Hindsight Oracle -> BC 初期化 -> Imagination AC
- **評価**: Walk-Forward、PBO、Deflated Sharpe、HMM レジーム別分析、M2 scorecard

直近では、実験系の責務を [unidream/experiments](unidream/experiments) に、source rollout の中核を [unidream/source_rollout](unidream/source_rollout) に切り出して、大規模リファクタリングを完了しています。

---

## M2 目標

現在の到達目標は `M2` です。必須条件はすべて `AND` です。

- `alpha_excess >= +5pt/年`
- `sharpe_delta >= +0.20`
- `maxdd_delta <= -10pt`
- `win_rate_vs_bh >= 60%`
- `collapse_guard = pass`

加点条件は `OR` です。

- `alpha_excess >= +8pt/年`
- `maxdd_delta <= -15pt`

---

## 最適化ループ

学習原理側の改善は、次の順で回します。

1. `issue1 teacher audit`
2. `issue2 BC prior`
3. `issue3 AC support`
4. `issue4 WM regime`
5. `issue5 conservative AC`
6. `issue6 external source`

入口:

- 全体 overview: [docs/optimization_loop_overview.md](docs/optimization_loop_overview.md)
- top-level runner: [scripts/run_optimization_loop.ps1](scripts/run_optimization_loop.ps1)
- current status: [docs/optimization_status.md](docs/optimization_status.md)

issue ごとのメモ:

- [docs/optimization_issue1_teacher_audit.md](docs/optimization_issue1_teacher_audit.md)
- [docs/optimization_issue2_bc_prior.md](docs/optimization_issue2_bc_prior.md)
- [docs/optimization_issue3_ac_support.md](docs/optimization_issue3_ac_support.md)
- [docs/optimization_issue4_wm_regime.md](docs/optimization_issue4_wm_regime.md)
- [docs/optimization_issue5_conservative_ac.md](docs/optimization_issue5_conservative_ac.md)
- [docs/optimization_issue6_external_sources.md](docs/optimization_issue6_external_sources.md)

---

## 最新結果

### 1. `medium_v2` 全 fold 重実験

設定: [configs/medium_v2.yaml](configs/medium_v2.yaml)  
期間: `2020-01-01 -> 2024-01-01`  
WFO: `6 folds`  
実行: `auto`（CUDA 環境では `cuda`、Apple Silicon では `mps`）

#### Aggregate

| Metric | Value | M2 Target | Status |
|---|---:|---:|---|
| `alpha_excess` | `-59.61 pt/年` | `>= +5` | MISS |
| `sharpe_delta` | `-1.010` | `>= +0.20` | MISS |
| `maxdd_delta` | `-10.23 pt` | `<= -10` | PASS |
| `win_rate_vs_bh` | `49.3%` | `>= 60%` | MISS |
| `collapse_guard` | `pass` | `pass` | PASS |

**結果: `M2 MISS`**

#### Fold 別

| Fold | `alpha_excess` | `sharpe_delta` | `maxdd_delta` | `win_rate_vs_bh` | Result |
|---|---:|---:|---:|---:|---|
| 0 | `+16.25 pt` | `-0.402` | `-21.75 pt` | `49.3%` | MISS |
| 1 | `-10.61 pt` | `-1.035` | `-8.64 pt` | `49.3%` | MISS |
| 2 | `-48.17 pt` | `-1.098` | `-10.72 pt` | `48.6%` | MISS |
| 3 | `-287.67 pt` | `-0.815` | `-9.94 pt` | `49.5%` | MISS |
| 4 | `-22.43 pt` | `-1.306` | `-5.01 pt` | `49.5%` | MISS |
| 5 | `-5.04 pt` | `-1.403` | `-5.34 pt` | `49.4%` | MISS |

#### 診断

- 最終 policy はほぼ `short 100%` に崩壊
- 現行の source/action family では OOS の優位が残っていない
- `maxdd` は一定改善できても、`alpha`、`sharpe`、`win rate` が弱い

### 2. WM indent 修正後 + 外部ソース追加後

設定: [configs/medium_ext_sources.yaml](configs/medium_ext_sources.yaml)

最新の観測結果:

- `Mean Sharpe = -1.457`
- `Aggregate alpha_excess = -62.67 pt/年`
- `win_rate_vs_bh = 49.2%`
- ポジションは `short 100%` 固定ではなくなった
- Fold 3 は `Sharpe = 2.146`、約 `+13.8%` の黒字
- Regime 2 は `Sharpe = 1.544`
- `PBO = 0.50`

診断:

- WM 修正は効いているが、OOS alpha は改善不足
- `Regime 0` と `Regime 1` が主な失敗領域
- 下落局面でロングを十分に切れていない
- 外部ソースは急落初動には効く余地があるが、M2 までの決定打にはまだ見えていない

---

## 現在の仮説

現時点のボトルネックは、単純な optimizer 調整ではありません。主に以下を疑っています。

1. Oracle 自体が下落局面で十分な行動を出せていない
2. BC prior が弱く、AC に入る前に土台が崩れている
3. World Model の regime transition 表現力が不足している
4. `signed_order_flow` や `taker_imbalance` は急落検知には効くが、低ボラ下落局面まではカバーしきれない

直近の優先分析は以下です。

- Oracle の行動分布を regime 別に確認する
- `medium_v2` と `medium_ext_sources` を比較する
- 外部ソースが `Regime 0/1` にどれだけ効いているかを測る

---

## パイプライン

現在のメイン学習パスは次の通りです。

1. feature / returns を cache から読み込む
2. Walk-Forward split を組む
3. hindsight oracle path を計算する
4. World Model を学習する
5. oracle path を BC で模倣する
6. imagination AC で微調整する
7. val selector で候補を選ぶ
8. test backtest と M2 scorecard を出す

この実行責務は現在、以下に分割されています。

- [train.py](train.py)
- [train_app.py](unidream/experiments/train_app.py)
- [train_pipeline.py](unidream/experiments/train_pipeline.py)
- [wm_stage.py](unidream/experiments/wm_stage.py)
- [bc_stage.py](unidream/experiments/bc_stage.py)
- [ac_stage.py](unidream/experiments/ac_stage.py)
- [val_selector_stage.py](unidream/experiments/val_selector_stage.py)
- [test_stage.py](unidream/experiments/test_stage.py)

---

## 実装状況

### Phase 1: バックテスト基盤 + WM + BC + AC

完了済み:

1. WFO ベースのバックテスト基盤
2. Hindsight Oracle -> BC -> WM -> AC の一連実装
3. Risk Controller 系 probe
4. Event Controller 系 probe
5. QDT baseline
6. source rollout 基盤
7. 実験ランタイムの大規模リファクタリング

### Phase 2: LoRe 統合

現在は未着手寄りです。候補として以下のスタブがあります。

- [llm_embed.py](unidream/lore/llm_embed.py)
- [risk_gate.py](unidream/lore/risk_gate.py)

### Phase 3: オンライン fine-tune

現在は本格運用前のスタブ段階です。

- [finetune.py](unidream/online/finetune.py)

---

## セットアップ

### 前提

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

### インストール

```bash
git clone https://github.com/shinji-ogasa/UniDream.git
cd UniDream
uv sync
```

Apple Silicon では `uv sync` で標準の PyTorch wheel を入れ、実行時デバイスは `--device auto` か `--device mps` を使います。
実行時デバイスの共通管理は [unidream/device.py](unidream/device.py) に寄せています。

---

## 使い方

### メインパイプライン

```bash
uv run python train.py
```

主な例:

```bash
# smoke
uv run python train.py --config configs/smoke_test.yaml --start 2022-01-01 --end 2023-06-01

# medium_v2 を全 fold 実行
uv run python train.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01 --device auto

# resume
uv run python train.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01 --device auto --resume

# 特定 fold のみ
uv run python train.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01 --folds 0,1,4 --device auto

# checkpoint から test だけ再実行
uv run python train.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01 --start-from test --stop-after test --resume --device auto
```

主なオプション:

| Flag | 説明 |
|---|---|
| `--config` | 設定ファイル |
| `--start` / `--end` | 実験期間 |
| `--device` | `auto` / `mps` / `cuda` / `cpu` |
| `--resume` | 既存 checkpoint から再開 |
| `--start-from` | `wm / bc / ac / test` のどこから再開するか |
| `--stop-after` | `wm / bc / ac / test` のどこで止めるか |
| `--cost-profile` | `base / stress` のコスト設定 |
| `--folds` | 実行 fold の絞り込み |

### Risk Controller

```bash
uv run python train_risk_controller.py --config configs/smoke_risk_controller_v5_basis.yaml --start 2021-01-01 --end 2023-06-01
```

### Event Controller

```bash
uv run python train_event_controller.py --config configs/smoke_event_controller_v3_triplebarrier.yaml --start 2021-01-01 --end 2023-06-01
```

### QDT

```bash
uv run python train_qdt.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01
```

### PPO baseline

```bash
uv run python train_ppo.py
```

---

## Source Rollout

source family 比較は以下を中心に回します。

- [configs/source_rollout_suite.yaml](configs/source_rollout_suite.yaml)
- [configs/source_rollout_suite_free.yaml](configs/source_rollout_suite_free.yaml)
- [unidream/source_rollout/plan.py](unidream/source_rollout/plan.py)
- [unidream/source_rollout/requirements.py](unidream/source_rollout/requirements.py)

よく使うコマンド:

```powershell
# source rollout 診断
powershell -ExecutionPolicy Bypass -File .\scripts\run_source_rollout_checks.ps1 `
  -CacheDir checkpoints\aux_smoke2 `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2

# free source rollout
powershell -ExecutionPolicy Bypass -File .\scripts\run_free_source_rollout_end_to_end.ps1

# source family suite
powershell -ExecutionPolicy Bypass -File .\scripts\run_source_family_suite.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

関連ドキュメント:

- [docs/source_rollout_workflow.md](docs/source_rollout_workflow.md)
- [docs/free_source_rollout.md](docs/free_source_rollout.md)
- [docs/source_cache_formats.md](docs/source_cache_formats.md)
- [docs/source_requirements_matrix.md](docs/source_requirements_matrix.md)

---

## 出力

- checkpoint: `checkpoints/fold_{idx}/`
- Risk/Event probe summary: `checkpoints/*_summary.csv`
- source rollout summary: `checkpoints/source_family_suite/suite_summary.csv`
- best source family report: `checkpoints/source_family_suite/best_source_family.md`

---

## テスト

```bash
uv run python -m pytest tests/
```

source rollout の確認:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_source_rollout_checks.ps1 `
  -CacheDir checkpoints\aux_smoke2 `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

---

## リポジトリ構成

```text
UniDream/
|- README.md
|- CLAUDE.md
|- SPEC.md
|- pyproject.toml
|- train.py
|- train_qdt.py
|- train_risk_controller.py
|- train_event_controller.py
|- train_ppo.py
|- build_*_source_cache.py
|- build_source_cache_from_manifest.py
|- validate_source_*.py
|- inspect_*.py
|- select_best_source_family.py
|- recommend_*.py
|- write_source_rollout_*.py
|- configs/
|- docs/
|- scripts/
|- tests/
|- checkpoints/
`- unidream/
   |- actor_critic/
   |- baselines/
   |- data/
   |- eval/
   |- experiments/
   |- lore/
   |- online/
   |- source_rollout/
   `- world_model/
```

`unidream/experiments` 配下には、現在の stage 分割された実験実行コードがあります。

---

## 既知の課題

2026-04-06 時点のコードレビュー上の主な懸念:

### Critical

| # | Module | Issue |
|---|---|---|
| 1 | `eval/backtest.py` | perfect strategy 時に `Sortino/Calmar = np.inf` となり JSON serialization を壊す |
| 2 | `eval/pbo.py` | Bailey / Lopez de Prado の定義と完全一致していない |
| 3 | `actor_critic/imagination_ac.py` | `_compute_lambda_returns()` の symlog 重複適用疑い |
| 4 | `world_model/train_wm.py` | batch に `actions` が無い時の fallback が弱い |
| 5 | `data/oracle.py` | `ACTIONS` に `0.0` が無いと `np.where(...)[0][0]` が落ちる |

### High Priority

| # | Module | Issue |
|---|---|---|
| 6 | `eval/backtest.py` | `equity = exp(cumsum(pnl))` が数理的に不整合な可能性 |
| 7 | `eval/pbo.py` | `skew=0, kurtosis=3` 固定が crypto に対して弱い |
| 8 | `actor_critic/actor.py` | `target_soft_labels()` の bucketize 境界 |
| 9 | `data/download.py` | retry / validation が弱い |
| 10 | `data/dataset.py` | `right_inclusive=True` が test のみに入っている |

### Medium

- config schema validation が無い
- helper の重複がまだ一部残っている
- テストは `tests/test_source_rollout_helpers.py` 偏重

---

## 主な依存

- `torch`
- `numpy`
- `pandas`
- `scipy`
- `pandas-ta`
- `scikit-learn`
- `hmmlearn`
- `matplotlib`
- `requests`
- `pyyaml`
- `pyarrow`

---

## 参考

- [Dreamer 4](https://github.com/nicklashansen/dreamer4)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [IRIS](https://arxiv.org/abs/2209.00588)
- [TWM](https://arxiv.org/abs/2209.14855)
- [SIRL](https://arxiv.org/abs/2209.02276)
- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)

---

## メモ

- 現状の最有力な結論は、`M2` はまだ遠いということです。
- 外部ソース追加は意味がありますが、Oracle 品質と regime handling の問題がまだ一段重いです。
- いまの repo は、戦略成績よりも先に実装基盤の整理がかなり進んだ状態です。
