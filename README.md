# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

Imagination-based reinforcement learning for crypto trading.

DreamerV3 ベースで RSSM を Transformer に置換した世界モデルを学習し、Imagination 上で Actor-Critic を訓練してトレーディング方策を獲得する。

---

## Overview

UniDream は、暗号資産トレード向けに **World Model + Imagination RL** を適用するプロジェクト。

- **入力**: OHLCV（対数リターン）+ 最小TA（RSI, MACD, ATR）をフラット concat、オプションで外部ソース（funding rate, basis, order flow, on-chain）
- **世界モデル**: DreamerV3 ベース、RSSM を Block-Causal Transformer に置換、MLP encoder（2 層 256-256）、離散カテゴリカル潜在（32x32）、アンサンブル 2-5
- **方策**: `trade / target inventory / no-trade band` を出力する inventory controller
- **報酬**: Buy & Hold 超過収益（`excess_bh`）を基準に DSR / drawdown shaping を追加
- **学習**: Hindsight Oracle → 世界モデル学習 → BC 初期化 → Imagination AC
- **評価**: Walk-Forward（四半期ロール、10 期間以上）、PBO / Deflated Sharpe、HMM レジーム別メトリクス
- **M2 目標**: alpha_excess >= +5pt/年, sharpe_delta >= +0.20, maxdd_delta <= -10pt, win_rate_vs_bh >= 60%, collapse_guard = pass

---

## Pipeline

```
OHLCV取得（Binance API）+ オプション外部ソース（funding, basis, order flow, on-chain）
  ↓
rolling z-score正規化（60日窓）+ ATR割りリターン + TA計算（shift(1)適用）
  ↓
時系列WFO分割（train 2年 / val 3ヶ月 / test 3ヶ月、四半期ロール）
  ↓
train期間でhindsight oracle DP計算（inventory path、コスト込み）
  ↓
Transformer世界モデル学習（MLP encoder、離散カテゴリカル潜在、アンサンブル）
  ↓
BC初期化（controller を oracle inventory path に模倣）
  ↓
AC fine-tune（imagination rollout、excess_bh + turnover / flow 制約、BC prior 混合）
  ↓
Val Selector（複数候補からスコアベースで最良方策を選択）
  ↓
test期間バックテスト（スプレッド+手数料+スリッページモデル込み）
  ↓
PBO/DSR選別 → レジーム別メトリクス → M2 スコアカード判定
```

---

## Key Specifications

| Item | Value |
|------|-------|
| 対象 | BTCUSDT (Binance Futures) |
| 時間足 | 15 分足 |
| 方策 | `trade / target inventory / band` controller |
| 目標 inventory | config の `oracle.action_values` で指定 |
| 世界モデル | DreamerV3 + Block-Causal Transformer |
| エンコーダ | MLP 2 層（256-256） |
| 潜在空間 | 離散カテゴリカル（32x32） |
| Imagination horizon | 1-5 ステップ |
| アンサンブル | 2-5 モデル |
| 学習報酬 | `excess_bh` + DSR / DD shaping |
| DL Framework | PyTorch (>= 2.0) |

---

## Latest Results (medium_v2, 2026-04-06)

`medium_v2.yaml` で 2020-01-01 → 2024-01-01、6 fold WFO、3 行動 long-only（0.0 / 0.5 / 1.0）を CUDA で全 fold 実行。

### Aggregate

| Metric | Value | M2 Target | Status |
|--------|-------|-----------|--------|
| alpha_excess | -59.61 pt/年 | >= +5 pt | MISS |
| sharpe_delta | -1.010 | >= +0.20 | MISS |
| maxdd_delta | -10.23 pt | <= -10 pt | PASS |
| win_rate_vs_bh | 49.3% | >= 60% | MISS |
| collapse_guard | pass | pass | PASS |

**M2 = MISS**

### Per-Fold

| Fold | alpha_excess | sharpe_delta | maxdd_delta | win_rate | M2 |
|------|-------------|-------------|-------------|----------|-----|
| 0 | +16.25 pt | -0.402 | -21.75 pt | 49.3% | MISS |
| 1 | -10.61 pt | -1.035 | -8.64 pt | 49.3% | MISS |
| 2 | -48.17 pt | -1.098 | -10.72 pt | 48.6% | MISS |
| 3 | -287.67 pt | -0.815 | -9.94 pt | 49.5% | MISS |
| 4 | -22.43 pt | -1.306 | -5.01 pt | 49.5% | MISS |
| 5 | -5.04 pt | -1.403 | -5.34 pt | 49.4% | MISS |

### Diagnosis

全 fold で最終 policy が実質 short 100% に張り付き。現在の source/action family（OHLCV + TA のみ、long-only 3 行動）では OOS 優位が確認できず。モデル微調整ではなく source family 拡張（basis, order-flow, on-chain）が次のステップ。

---

## Implementation Status

### Phase 1 — バックテスト基盤 + World Model + AC（現フェーズ）

1. バックテスト基盤（WFO・コスト・PBO）
2. Hindsight Oracle → BC 初期化 → 世界モデル → AC fine-tune
3. Risk Controller（v1-v5: linear, context, funding, basis）
4. Event Controller（v1-v3: sparse, triple barrier）
5. QDT ベースライン（Decision Transformer + Oracle Q）
6. Source Rollout（basis → order-flow → on-chain → hybrid の段階的データソース拡張）

### Phase 2 — LoRe 統合

Phase 1 で alpha が確認できてから導入。

- **LLM Embed**: ニュース・センチメントを LLM で embed → 世界モデルの入力に結合
- **Risk Gate**: 重大イベント検知時に actor のポジションを強制縮小 / フラット化

### Phase 3 — オンライン fine-tune

Backtest 結果を見てから検討。実環境データで世界モデルを逐次更新。

---

## Setup

### 前提条件

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

### インストール

```bash
git clone https://github.com/shinji-ogasa/UniDream.git
cd UniDream
uv sync
```

---

## Usage

### メインパイプライン（World Model + Imagination AC）

Oracle → 世界モデル学習 → BC 初期化 → AC fine-tune → バックテスト → PBO/DSR 評価 を一括実行する。

```bash
uv run python train.py
```

オプション:

| フラグ | デフォルト | 説明 |
|--------|-----------|------|
| `--config` | `configs/trading.yaml` | 設定ファイルパス |
| `--symbol` | config の値 (`BTCUSDT`) | Binance シンボル |
| `--start` | `2018-01-01` | データ取得開始日 |
| `--end` | `2024-01-01` | データ取得終了日 |
| `--device` | auto (`cuda` / `cpu`) | CUDA 自動検出、フォールバック CPU |
| `--seed` | `42` | 乱数シード |
| `--checkpoint_dir` | `checkpoints` | チェックポイント保存先 |
| `--resume` | off | チェックポイントから再開 |
| `--start-from` | `wm` | `wm / bc / ac / test` から開始 |
| `--stop-after` | `test` | `wm / bc / ac / test` で停止 |
| `--cost-profile` | config 既定 (`base`) | `base / stress` のコスト設定を選択 |
| `--folds` | all | 実行する fold を指定（例: `0,1,4`） |

```bash
# 途中で落ちた場合の再開
uv run python train.py --resume

# World Model だけ学習して停止
uv run python train.py --stop-after wm

# 既存の WM checkpoint を使って BC だけ実行
uv run python train.py --start-from bc --stop-after bc

# 既存 checkpoint から backtest だけ再実行
uv run python train.py --start-from test

# コストを stress 条件で評価
uv run python train.py --cost-profile stress

# Smoke test（パイプライン動作確認、数分で完了）
uv run python train.py --config configs/smoke_test.yaml --start 2022-01-01 --end 2023-06-01

# Medium run（2020-2024、6 fold WFO）
uv run python train.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01
```

### Risk Controller

Funding rate / basis / order flow 等を使ったリスクコントロール方策の学習。

```bash
uv run python train_risk_controller.py --config configs/smoke_risk_controller_v5_basis.yaml \
  --start 2021-01-01 --end 2023-06-01
```

### Event Controller

イベント駆動型コントローラの学習。

```bash
uv run python train_event_controller.py --config configs/smoke_event_controller_v3_triplebarrier.yaml \
  --start 2021-01-01 --end 2023-06-01
```

### QDT ベースライン

Decision Transformer + Oracle Q-value による比較対象。

```bash
uv run python train_qdt.py --config configs/medium_v2.yaml --start 2020-01-01 --end 2024-01-01
```

### PPO ベースライン

Model-free PPO を WFO で学習・評価する。世界モデルの寄与を差分で確認するために使用。

```bash
uv run python train_ppo.py
```

### Source Rollout（外部データソース段階導入）

basis → order-flow → on-chain → hybrid の順でデータソースを段階的に追加・評価する。

```bash
# Free source（Binance public + Coin Metrics community）の一括ロールアウト
powershell -ExecutionPolicy Bypass -File .\scripts\run_free_source_rollout_end_to_end.ps1

# Source family suite 実行
powershell -ExecutionPolicy Bypass -File .\scripts\run_source_family_suite.ps1 `
  -CacheDir checkpoints\basis_source_cache `
  -CacheTag BTCUSDT_15m_2021-01-01_2023-06-01_z60_v2
```

詳細は `docs/source_rollout_workflow.md` および `docs/free_source_rollout.md` を参照。

### 出力

- チェックポイント: `checkpoints/fold_{idx}/` 以下に `world_model.pt`, `bc_actor.pt`, `ac.pt`, `ac_best.pt`
- Risk/Event Controller: `checkpoints/{controller}_summary.csv`
- Source Rollout: `checkpoints/source_family_suite/suite_summary.csv`, `best_source_family.md`
- 評価指標（stdout）: Fold 別 Sharpe / MaxDD / Calmar / Total Return、PBO、Deflated Sharpe、レジーム別メトリクス、M2 スコアカード

### テスト

```bash
uv run python -m pytest tests/
```

---

## Repository Structure

```
UniDream/
├── README.md                       # プロジェクト概要（本ファイル）
├── CLAUDE.md                       # 開発方針
├── SPEC.md                         # 技術仕様詳細
├── pyproject.toml                  # プロジェクト設定・依存パッケージ（uv）
│
├── train.py                        # メインパイプライン（WM + BC + AC）
├── train_ppo.py                    # PPO ベースライン
├── train_risk_controller.py        # Risk Controller 学習
├── train_event_controller.py       # Event Controller 学習
├── train_qdt.py                    # QDT (Decision Transformer) 学習
│
├── build_*_source_cache.py         # データソースキャッシュ構築（Binance / CoinMetrics / Glassnode）
├── build_source_cache_from_manifest.py  # マニフェストからソースキャッシュ構築
├── validate_source_*.py            # マニフェスト・ロールアウト検証
├── inspect_*.py                    # ソースキャッシュ・手動入力検査
├── select_best_source_family.py    # 最良ソースファミリ選択
├── recommend_*.py                  # 次ステップ・フェッチコマンド推薦
├── write_source_rollout_*.py       # ロールアウトレポート・スナップショット出力
│
├── unidream/                       # メインパッケージ
│   ├── data/
│   │   ├── download.py             # Binance OHLCV / funding / OI / mark price 取得
│   │   ├── features.py             # TA計算・rolling z-score正規化・rebound features
│   │   ├── oracle.py               # Hindsight oracle DP（離散行動、コスト込み）
│   │   └── dataset.py              # データローダー（WFO分割対応）
│   ├── world_model/
│   │   ├── encoder.py              # MLP encoder（2層 256-256）
│   │   ├── transformer.py          # Block-Causal Transformer 世界モデル
│   │   ├── ensemble.py             # アンサンブル + 不一致ペナルティ
│   │   └── train_wm.py             # 世界モデル学習（IDM / return 補助損失含む）
│   ├── actor_critic/
│   │   ├── actor.py                # Inventory controller actor（trade / target / band）
│   │   ├── critic.py               # Critic（twohot value function）
│   │   ├── bc_pretrain.py          # BC初期化（KL損失、状態依存重み、action chunking）
│   │   └── imagination_ac.py       # Imagination AC（BC損失減衰混合、online WM update）
│   ├── baselines/
│   │   └── ppo.py                  # Model-free PPO ベースライン
│   ├── experiments/                # パイプライン実行管理
│   │   ├── train_app.py            # 共通学習アプリケーションランタイム
│   │   ├── train_pipeline.py       # WFO fold ループ
│   │   ├── train_reporting.py      # 集約レポート・PBO/DSR 診断
│   │   ├── runtime.py              # config / seed / cost 解決
│   │   ├── fold_runtime.py         # fold 単位のチェックポイント管理
│   │   ├── fold_inputs.py          # fold 入力データ準備
│   │   ├── wfo_runtime.py          # WFO 分割・fold 選択
│   │   ├── oracle_stage.py         # Oracle ステージ
│   │   ├── oracle_teacher.py       # Oracle 教師ラベル生成
│   │   ├── wm_stage.py             # 世界モデルステージ
│   │   ├── bc_stage.py             # BC ステージ
│   │   ├── ac_stage.py             # AC ステージ
│   │   ├── val_selector_stage.py   # Val selector ステージ
│   │   ├── test_stage.py           # テストステージ
│   │   ├── m2.py                   # M2 スコアカード・目標判定
│   │   └── probe_common.py         # プローブ共通ユーティリティ
│   ├── source_rollout/             # 外部ソース段階導入
│   │   ├── plan.py                 # ロールアウト計画
│   │   └── requirements.py         # ソース要件定義
│   ├── lore/                       # Phase 2: LoRe 統合（スタブ）
│   │   ├── llm_embed.py            # ニュース/イベント → LLM embedding
│   │   └── risk_gate.py            # Uncertainty gating
│   ├── online/
│   │   └── finetune.py             # オンライン fine-tune（Phase 3 スタブ）
│   └── eval/
│       ├── backtest.py             # バックテスト（コスト・スリッページモデル込み）
│       ├── wfo.py                  # Walk-Forward Optimization
│       ├── pbo.py                  # PBO・Deflated Sharpe計算
│       └── regime.py               # HMMレジーム検出・レジーム別メトリクス
│
├── configs/                        # 実験設定
│   ├── trading.yaml                # フル設定テンプレート
│   ├── medium_v2.yaml              # 中規模実験（3行動 long-only）
│   ├── smoke_test.yaml             # パイプライン動作確認
│   ├── smoke_risk_controller_*.yaml  # Risk Controller バリアント（v1-v11）
│   ├── smoke_event_controller_*.yaml # Event Controller バリアント（v1-v3）
│   ├── source_rollout_suite*.yaml  # ソースロールアウトスイート定義
│   └── source_manifest_*.yaml      # ソースマニフェスト
│
├── scripts/                        # PowerShell ヘルパースクリプト
│
├── docs/                           # 補足ドキュメント
│   ├── source_rollout_workflow.md
│   ├── free_source_rollout.md
│   ├── source_cache_formats.md
│   └── source_requirements_matrix.md
│
├── tests/
│   └── test_source_rollout_helpers.py
│
└── checkpoints/                    # 学習済みモデル・評価結果
```

---

## Code Review Summary (2026-04-06)

### Critical Issues

| # | Module | Issue |
|---|--------|-------|
| 1 | `eval/backtest.py` | Sortino/Calmar return `np.inf` for perfect strategies — breaks JSON serialization |
| 2 | `eval/pbo.py` | PBO logic differs from Bailey/Lopez de Prado reference (compares IS best vs OOS best, not IS-best's OOS rank) |
| 3 | `actor_critic/imagination_ac.py` | `_compute_lambda_returns()` may double-apply symlog |
| 4 | `world_model/train_wm.py` | Missing action fallback when batch lacks "actions" key |
| 5 | `data/oracle.py` | `np.where(...)[0][0]` crashes if ACTIONS lacks 0.0 |

### High Priority

| # | Module | Issue |
|---|--------|-------|
| 6 | `eval/backtest.py` | Equity uses `exp(cumsum(pnl))` but pnl is not log-scale — math inconsistency |
| 7 | `eval/pbo.py` | Hardcoded skew=0, kurtosis=3 underestimates deflation for crypto |
| 8 | `actor_critic/actor.py` | `target_soft_labels()` bucketize off-by-one edge case |
| 9 | `data/download.py` | No API retry logic or input validation |
| 10 | `data/dataset.py` | `right_inclusive=True` only for test — potential boundary overlap |

### Medium Priority

- Config management: no schema validation, silent config errors
- DRY violations: `resolve_costs()` duplicated across train scripts, risk/event controller grid search nearly identical
- Test coverage: only 1 test file (`test_source_rollout_helpers.py`)
- `source_rollout/plan.py`: hardcoded Windows paths

---

## Dependencies

`pyproject.toml` で管理。主な依存:

- **torch** (>= 2.0) — DL フレームワーク（CUDA 12.8 対応）
- **numpy / pandas / scipy** — 数値計算・データ操作
- **pandas-ta** — テクニカル指標
- **scikit-learn** — 正規化等
- **hmmlearn** — HMM レジーム検出
- **matplotlib** — 評価・可視化
- **requests** — Binance API
- **pyyaml** — 設定ファイル読み込み
- **pyarrow** — Parquet I/O

---

## References

- [Dreamer 4](https://github.com/nicklashansen/dreamer4) — Block-Causal Transformer 実装参考
- [DreamerV3](https://arxiv.org/abs/2301.04104) — 離散カテゴリカル潜在・Actor-Critic 設計
- [IRIS](https://arxiv.org/abs/2209.00588) — Transformer world model for Atari
- [TWM](https://arxiv.org/abs/2209.14855) — Transformer-based World Models
- [SIRL](https://arxiv.org/abs/2209.02276) — 状態依存 BC 重み
- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) — 多重比較補正
