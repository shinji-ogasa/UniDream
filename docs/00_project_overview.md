# UniDream プロジェクト概要

## 目的

BTCUSDT 15分足を対象に、Transformer World Model + Behavior Cloning + Imagination Actor-Critic で取引ポリシーを学習する研究プロジェクト。Buy & Hold をベンチマークとし、OOS 超過成績・Sharpe改善・最大ドローダウン改善・collapse回避を同時に評価する。

## パイプライン

```
OHLCV / features → Walk-Forward split → Hindsight Oracle (signal_aim teacher)
→ Transformer World Model → WM predictive state (return/vol/drawdown)
→ Behavior Cloning (route head + inventory recovery + state machine)
→ Imagination Actor-Critic → Validation selector
→ Test backtest / M2 scorecard / PBO / regime report
```

## M2 目標

| 指標 | 目標値 |
|---|---|
| alpha_excess | >= +5pt/年 |
| sharpe_delta | >= +0.20 |
| maxdd_delta | <= -10pt |
| win_rate_vs_bh | >= 60% |
| collapse_guard | pass |

加点条件: alpha_excess >= +8pt/年 OR maxdd_delta <= -15pt

前提: まず BC-only で M2、その後に AC で向上を狙う。

## 本流構成

- **Phase 8 safe baseline**: 3値 route head (de_risk/neutral/overweight) + state machine gate + inventory recovery controller
- **configs/trading.yaml**: 本流 config。改変禁止
- **checkpoints/acplan13_base_wm_s011**: f456 の WM checkpoint。return/vol/drawdown aux head あり
- **主要評価 folds**: dev=f456, legacy=f045, adoption=全fold

## ディレクトリ構造

```
unidream/
  cli/           # CLI エントリポイント（train, probes）
  data/          # Binance データ取得・特徴量・Oracle
  world_model/   # Transformer WM (GPT-style causal, symlog twohot)
  actor_critic/  # Actor, Critic, BC pretrain, Imagination AC
  experiments/   # パイプライン制御・WFO・評価
  eval/          # バックテスト・PBO・regime分析
```

## 実験ファイル一覧

| CLI | Plan | 内容 |
|---|---|---|
| `exploration_board_probe.py` | Plan2 | ridge D+A+pullback 探索ボード（本命） |
| `plan3_wm_overlay.py` | Plan3 | WM推論ベース overlay（/100 fix済み） |
| `plan4_calibration.py` | Plan4 A | WM 予測値校正 |
| `plan4_verify.py` | Plan4 B-E | ensemble, blocked attribution, throttle, grid |
| `plan5_verify.py` | Plan5 B-C | ridge+WM veto, pipeline breakdown |
| `plan5_laneF.py` | Plan5 F | pullback_recovery label evaluator |
