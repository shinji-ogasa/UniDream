# UniDream

World-model-based reinforcement learning for trading.

## Project Goal

Oracle trajectory から Transformer 世界モデルを学習し、Imagination 上で Actor-Critic を訓練することで、トレーディング方策を獲得する。
Dreamer の imagination-based RL パイプラインを金融時系列に適用する。

---

## Architecture Decisions

### DL Framework: PyTorch
- Dreamer 4 (`nicklashansen/dreamer4`) が PyTorch (torch 2.8) — ベースとして利用
- 世界モデル・Actor・Critic すべて PyTorch で統一

### 世界モデル: Transformer（IRIS / TWM 系）
- 状態トークン列 → 次状態 + 報酬予測
- RSSM ではなく Transformer ベースのシーケンスモデルを採用
- 損失: next state prediction + reward prediction + termination の 3 ヘッド

### 行動空間（未確定）
二つの候補:

| Option | Pros | Cons |
|--------|------|------|
| 離散（buy / hold / sell） | DP で oracle 計算が容易、実装シンプル | ポジションサイジング不可 |
| 連続（ポジション比率 -1〜+1） | 柔軟なポジション管理 | oracle 計算が複雑、方策学習が難しい |

**未決定: 離散から始めるのが安全。TradeRL との差分で判断。**

### Actor アーキテクチャ
- MLP or 小さめ Transformer
- 入力は過去〜現在の状態のみ（未来情報なし）

### Oracle Trajectory 生成
- 未来価格既知の状態で DP または貪欲法により最適行動列を算出
- 世界モデルの学習データおよび Actor BC の教師ラベルとして使用

---

## Data Pipeline

- **入力**: OHLCV + テクニカル指標（MA, RSI, ボリンジャーバンド等）数十次元
- **行動空間**: 離散（buy / hold / sell）または連続（ポジション比率 -1〜+1）
- **学習データ混合**: ランダム方策・ε-greedy・ヒューリスティック・oracle を混合
  - 初期比率: ランダム:oracle = 6:4（要チューニング）

---

## Repository Structure

```
unidream/
├── data/
│   ├── oracle.py              # Oracle trajectory 生成（DP / 貪欲法）
│   ├── features.py            # OHLCV + テクニカル指標の特徴量計算
│   └── dataset.py             # データローダー（混合方策データ）
├── world_model/
│   ├── transformer.py         # Transformer 世界モデル（IRIS/TWM 系）
│   └── train_wm.py            # 世界モデル事前学習エントリポイント
├── actor_critic/
│   ├── actor.py               # Actor（MLP / Transformer）
│   ├── critic.py              # Critic（value function）
│   ├── bc_pretrain.py         # Phase 2: Actor BC 事前学習
│   └── imagination_ac.py      # Phase 3: Imagination AC 学習
├── online/
│   └── finetune.py            # Phase 4: オンライン fine-tune
├── configs/
│   └── trading.yaml
└── eval/
    └── backtest.py            # バックテスト評価
```

---

## Dependencies

```yaml
# Core
- torch >= 2.0
- numpy
- pandas

# テクニカル指標
- ta-lib          # or: pandas-ta
- scikit-learn    # 特徴量正規化等

# データ取得
- yfinance        # or: ccxt for crypto

# 評価
- matplotlib
- seaborn
```

---

## Implementation Phases

### Phase 1 — 世界モデル事前学習
- Transformer 世界モデルの実装（状態トークン列 → 次状態 + 報酬 + termination）
- データ収集: ランダム・ε-greedy・ヒューリスティック・oracle の混合データ生成
- 3 ヘッド損失で学習

### Phase 2 — Actor BC 事前学習
- Oracle 行動を教師ラベルにして教師あり学習（Behavioral Cloning）
- 入力は過去〜現在の状態のみ（未来情報なし）

### Phase 3 — Imagination 上で Critic warmup → AC 学習
- 世界モデルで imagination rollout（horizon 15〜50 ステップ）
- Critic をまず数エポック warmup（actor 凍結）
- その後 actor 解放、KL 制約付きで BC 方策から徐々に離れる
- λ-return で value target 計算（Dreamer 方式）

### Phase 4 — オンライン fine-tune
- 実環境データで世界モデルを逐次更新
- Imagination AC も継続学習
- Replay buffer に実データを蓄積、世界モデルのカバレッジを拡大

---

## Open Questions

- **行動空間**: 離散 vs 連続（oracle の DP 計算と方策設計の両方に影響）
- **時間足**: 日足から始めるのが安全（分足はノイズが多い）
- **対象銘柄**: まず 1 銘柄で検証

---

## Key References

- [Dreamer 4](https://github.com/nicklashansen/dreamer4)
- [IRIS](https://arxiv.org/abs/2209.00588) — Transformer world model for Atari
- [TWM](https://arxiv.org/abs/2209.14855) — Transformer-based World Models
- [Dynalang](https://arxiv.org/abs/2308.01399) — multi-modal world model（コンセプト参考）
