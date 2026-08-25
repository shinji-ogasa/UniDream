# Risk-aware RL / UniDream common-action holdout experiment

実験目的は、論文の「Agent が B&H を約 +15pt 上回る」という主張を、UniDream の holdout・コスト・action 制約に置き換えたときに再現できるかを確認することだった。

論文: [Reinforcement Learning for Enhancing Bitcoin Risk-Aware Trading with Predictive Signals](https://www.mdpi.com/2079-9292/15/4/793)

## 実験契約

- データ: BTCUSDT / 15m、固定済み feature・return cache
- WFO: folds 15–23、各 fold は train 2年 → validation 3か月 → test 3か月
- 評価 test: 2024-01-16 から 2026-04-16（right-exclusive slice）
- コスト: fee 3.0bps + half-spread 1.5bps + slippage 1.0bps = `5.5bps * abs(delta_position)`
- action: target position `[0.50, 1.12]`、1 step の変化量 `<= 0.08`、初期 position は B&H の `1.0`
- seed: `7`
- device: CPU、deterministic algorithms 有効

## リーク防止契約

- feature の時刻 `t` はバー `t-1` までの情報だけを使う既存の causal cache を利用。
- forecaster は各 fold の train return だけで fit。future label も train の終端を越えない。
- forecaster の時刻 `t` の入力は `return[:t]` のみで、`return[t]` と future return を含めない。
- policy の更新は train episode のみ。
- checkpoint 選択は validation のみ。test は選択後に一度だけ report 用に評価。
- action bounds、step cap、cost、rolling volatility の causal 性を unit test で検証。

## 比較条件

| condition | forecast state | risk-aware reward |
|---|---:|---:|
| full | yes | yes |
| forecast-only | yes | no |
| risk-only | no | yes |
| baseline | no | no |

## 結果

`AlphaEx` は B&H 超過リターン（positive が良い）、`MaxDDDelta` は strategy の最大 drawdown − B&H の最大 drawdown（negative が良い）。

| condition | AlphaEx mean | median | best / worst | positive folds | MaxDDDelta mean | DD improved | Sharpe mean / delta | turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | -2.12pt | 0.00pt | +6.23 / -19.96pt | 4/9 | -4.37pt | 5/9 | 0.558 / -0.019 | 5.71 |
| forecast-only | -2.35pt | 0.00pt | +6.27 / -19.63pt | 3/9 | -2.61pt | 4/9 | 0.650 / +0.073 | 4.98 |
| risk-only | -1.74pt | 0.00pt | +8.90 / -19.82pt | 4/9 | -4.86pt | 5/9 | 0.573 / -0.005 | 6.70 |
| baseline | -1.87pt | 0.00pt | +6.25 / -19.39pt | 4/9 | -3.80pt | 5/9 | 0.647 / +0.070 | 5.79 |

full の fold 別 test 結果:

| fold | period | AlphaEx | MaxDDDelta | Sharpe | turnover |
|---:|---|---:|---:|---:|---:|
| 15 | 2024-01-16 → 2024-04-16 | +0.00pt | +0.00pt | 2.641 | 1.00 |
| 16 | 2024-04-16 → 2024-07-16 | +0.01pt | +2.54pt | 0.161 | 1.28 |
| 17 | 2024-07-16 → 2024-10-16 | -1.51pt | -12.20pt | 0.508 | 3.70 |
| 18 | 2024-10-16 → 2025-01-16 | -8.74pt | +1.05pt | 3.105 | 9.83 |
| 19 | 2025-01-16 → 2025-04-16 | +6.23pt | -8.75pt | -0.913 | 4.86 |
| 20 | 2025-04-16 → 2025-07-16 | -19.96pt | -5.56pt | 3.982 | 8.22 |
| 21 | 2025-07-16 → 2025-10-16 | -0.40pt | +1.53pt | -0.692 | 3.24 |
| 22 | 2025-10-16 → 2026-01-16 | +1.06pt | -7.04pt | -1.813 | 10.84 |
| 23 | 2026-01-16 → 2026-04-16 | +4.22pt | -10.87pt | -1.954 | 8.40 |

## 判定

この契約では、論文で示された +15pt 級の優位性は残らなかった。full の AlphaEx 平均は **-2.12pt**、Sharpe 差は **-0.019** だった。risk-only が比較条件の中では最も良い AlphaEx 平均（-1.74pt）と最大 drawdown 改善（-4.86pt）を示したが、平均リターンはなお B&H を下回る。

したがって今回の結果は「Risk-aware RL が弱い」と断定するものではなく、少なくともこの holdout・cost・action 制約下では +15pt を再現できず、評価設計やデータ頻度・action 表現の違いが結果に大きく効いていることを示す。

なお、これは paper の hourly cash/holdings + `[ActionType, Amount]` 環境を、UniDream の 15m target-position action に写像した common-action probe であり、論文環境の literal reproduction ではない。複数 seed と paper-native hourly 環境での追試が必要。

## 保存方針

このファイルを実験結果の記録として push する。実験用コード・設定・ローカル checkpoint は現行プロジェクトには残さない。
