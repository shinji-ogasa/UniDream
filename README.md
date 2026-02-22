# UniDream

World-model-based reinforcement learning for trading.

Oracle trajectory から世界モデルを学習し、Imagination 上で Actor-Critic を訓練するアプローチ。

---

## データパイプライン

- **入力**: OHLCV + テクニカル指標（MA, RSI, ボリンジャーバンド等）数十次元
- **行動空間**: 離散（buy / hold / sell）または連続（ポジション比率 -1〜+1）
- **Oracle trajectory 生成**: 未来価格既知の状態で DP または貪欲法により最適行動列を算出

---

## 実装フェーズ

### Phase 1 — 世界モデル事前学習

- **アーキテクチャ**: Transformer（IRIS / TWM 系）。状態トークン列 → 次状態 + 報酬予測
- **データ**: ランダム方策・ε-greedy・ヒューリスティック・oracle を混合。比率は要チューニング（ランダム:oracle = 6:4 あたりから開始）
- **損失**: next state prediction + reward prediction + termination の 3 ヘッド

### Phase 2 — Actor BC 事前学習

- Oracle 行動を教師ラベルにして教師あり学習（Behavioral Cloning）
- 入力は過去〜現在の状態のみ（未来情報なし）
- Actor アーキテクチャ: MLP or 小さめ Transformer

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

## 最初に決めるべきこと

- **行動空間**: 離散 vs 連続（oracle の DP 計算と方策設計の両方に影響）
- **時間足**: 日足から始めるのが安全（分足はノイズが多い）
- **対象銘柄**: まず 1 銘柄で検証
