# UniDream

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Non--Commercial-orange)

Imagination-based reinforcement learning for crypto trading.

DreamerV3 ベースで RSSM を Transformer に置換した世界モデルを学習し、Imagination 上で Actor-Critic を訓練してトレーディング方策を獲得する。

---

## Overview

UniDream は、暗号資産トレード向けに **World Model + Imagination RL** を適用するプロジェクト。

- **入力**: OHLCV（対数リターン）＋ 最小TA（RSI, MACD, ATR）をフラット concat
- **世界モデル**: DreamerV3 ベース、RSSM を Block-Causal Transformer に置換、MLP encoder（2 層 256-256）、離散カテゴリカル潜在（32×32）、アンサンブル 3〜5
- **行動空間**: 離散 5 択（-1, -0.5, 0, +0.5, +1）
- **報酬**: DSR（Differential Sharpe Ratio）－ β · ドローダウン増分
- **学習**: Hindsight Oracle → BC 初期化 → Imagination AC（BC 損失減衰混合）
- **評価**: Walk-Forward（四半期ロール、10 期間以上）、PBO / Deflated Sharpe、HMM レジーム別メトリクス

---

## Pipeline

```
OHLCV取得（FX: OANDA API / 仮想通貨: Binance API）
  ↓
rolling z-score正規化（60日窓）+ ATR割りリターン + TA計算（shift(1)適用）
  ↓
時系列WFO分割（train 2年 / val 3ヶ月 / test 3ヶ月、四半期ロール）
  ↓
train期間でhindsight oracle DP計算（離散5行動、コスト込み）
  ↓
BC初期化（KL損失、状態依存重み、数エポック）
  ↓
Transformer世界モデル学習（MLP encoder、離散カテゴリカル潜在、アンサンブル3-5）
  ↓
AC fine-tune（imagination 1-3step、DSR-コスト-DD罰、BC損失減衰混合）
  ↓
test期間バックテスト（スプレッド+手数料+スリッページモデル込み）
  ↓
PBO/DSR選別 → レジーム別メトリクス → 生き残りのみペーパートレード
```

---

## Key Specifications

| Item | Value |
|------|-------|
| 対象 | BTCUSDT (Binance Futures) |
| 時間足 | 15 分足 |
| 行動空間 | 離散 5 択（-1, -0.5, 0, +0.5, +1） |
| 世界モデル | DreamerV3 + Block-Causal Transformer |
| エンコーダ | MLP 2 層（256-256） |
| 潜在空間 | 離散カテゴリカル（32×32） |
| Imagination horizon | 1〜3 ステップ |
| アンサンブル | 3〜5 モデル |
| 報酬 | DSR − β · ΔDD |
| DL Framework | PyTorch (>= 2.0) |

---

## Implementation Phases

### Phase 1 — バックテスト基盤 + Model-Free ベースライン（現フェーズ）

1. バックテスト基盤（WFO・コスト・PBO）を先に固める
2. model-free PPO をベースラインとして実行し、alpha を確認
3. Hindsight Oracle → BC 初期化 → 世界モデル → AC fine-tune

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

Oracle → BC 初期化 → 世界モデル学習 → AC fine-tune → バックテスト → PBO/DSR 評価 を一括実行する。

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

```bash
# 途中で落ちた場合の再開
uv run python train.py --resume

# World Model だけ学習して停止
uv run python train.py --stop-after wm

# 既存の WM checkpoint を使って BC だけ実行
uv run python train.py --start-from bc --stop-after bc

# 既存の WM / BC checkpoint を使って AC だけ実行
uv run python train.py --start-from ac --stop-after ac

# 既存 checkpoint から backtest だけ再実行
uv run python train.py --start-from test

# コストを stress 条件で評価
uv run python train.py --cost-profile stress

# Smoke test（パイプライン動作確認、数分で完了）
uv run python train.py --config configs/smoke_test.yaml --start 2022-01-01 --end 2023-06-01

# Medium run（2020-2024、6 fold WFO、~3 時間）
uv run python train.py --config configs/medium.yaml --start 2020-01-01 --end 2024-01-01
```

### PPO ベースライン

Model-free PPO を WFO で学習・評価する。世界モデルの寄与を差分で確認するために使用。

```bash
uv run python train_ppo.py
```

オプションは `train.py` と同じ（`--checkpoint_dir` のデフォルトは `checkpoints/ppo`）。

```bash
uv run python train_ppo.py --seed 123

# 途中から再開
uv run python train_ppo.py --resume
```

### 出力

- チェックポイント: `checkpoints/fold_{idx}/` 以下に `world_model.pt`, `bc_actor.pt`, `ac.pt`
- PPO チェックポイント: `checkpoints/ppo/ppo_fold_{idx}.pt`
- 評価指標（stdout）: Fold 別 Sharpe / MaxDD / Calmar / Total Return、PBO、Deflated Sharpe、レジーム別メトリクス

### テスト

現時点ではユニットテストは未整備。検証はバックテスト（WFO + PBO + Deflated Sharpe）で行う。

```bash
# メインパイプラインの結果で検証
uv run python train.py

# PPO ベースラインとの比較で妥当性を確認
uv run python train_ppo.py
```

---

## Repository Structure

```
UniDream/
├── README.md                      # プロジェクト概要（本ファイル）
├── CLAUDE.md                      # 開発方針
├── SPEC.md                        # 技術仕様詳細
├── pyproject.toml                 # プロジェクト設定・依存パッケージ（uv）
├── train.py                       # メインパイプライン実行スクリプト
├── train_ppo.py                   # PPO ベースライン実行スクリプト
├── unidream/
│   ├── data/
│   │   ├── download.py            # Binance / OANDA OHLCV取得
│   │   ├── features.py            # TA計算・rolling z-score正規化
│   │   ├── oracle.py              # Hindsight oracle DP（離散5行動、コスト込み）
│   │   └── dataset.py             # データローダー（WFO分割対応）
│   ├── world_model/
│   │   ├── encoder.py             # MLP encoder（2層 256-256）
│   │   ├── transformer.py         # Block-Causal Transformer 世界モデル
│   │   ├── ensemble.py            # アンサンブル（3-5モデル）+ 不一致ペナルティ
│   │   └── train_wm.py            # 世界モデル学習エントリポイント
│   ├── actor_critic/
│   │   ├── actor.py               # Actor（MLP / Transformer）
│   │   ├── critic.py              # Critic（value function）
│   │   ├── bc_pretrain.py         # BC初期化（KL損失、状態依存重み）
│   │   └── imagination_ac.py      # Imagination AC（BC損失減衰混合）
│   ├── baselines/
│   │   └── ppo.py                 # Model-free PPO ベースライン
│   ├── lore/                      # Phase 2: LoRe 統合
│   │   ├── llm_embed.py           # ニュース/イベント → LLM embedding
│   │   └── risk_gate.py           # Uncertainty gating
│   ├── online/
│   │   └── finetune.py            # オンライン fine-tune（Phase 3）
│   └── eval/
│       ├── backtest.py            # バックテスト（コスト・スリッページモデル込み）
│       ├── wfo.py                 # Walk-Forward Optimization
│       ├── pbo.py                 # PBO・Deflated Sharpe計算
│       └── regime.py              # HMMレジーム検出・レジーム別メトリクス
└── configs/
    └── trading.yaml
```

---

## Dependencies

`pyproject.toml` で管理。主な依存:

- **torch** (>= 2.0) — DL フレームワーク
- **numpy / pandas / scipy** — 数値計算・データ操作
- **pandas-ta** — テクニカル指標
- **scikit-learn** — 正規化等
- **hmmlearn** — HMM レジーム検出
- **matplotlib** — 評価・可視化
- **requests** — Binance / OANDA API
- **pyyaml** — 設定ファイル読み込み

---

## References

- [Dreamer 4](https://github.com/nicklashansen/dreamer4) — Block-Causal Transformer 実装参考
- [DreamerV3](https://arxiv.org/abs/2301.04104) — 離散カテゴリカル潜在・Actor-Critic 設計
- [IRIS](https://arxiv.org/abs/2209.00588) — Transformer world model for Atari
- [TWM](https://arxiv.org/abs/2209.14855) — Transformer-based World Models
- [SIRL](https://arxiv.org/abs/2209.02276) — 状態依存 BC 重み
- [Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) — 多重比較補正
