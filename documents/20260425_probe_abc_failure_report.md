# Probe A/B/C 検証結果と失敗判定

作成日: 2026-04-25 JST

## 結論

Probe A/B/C 系は、現時点では本線化しない。

特に正式に再実行した Probe C は fold 4 test で明確に失敗した。two-sided action space を用意しても、policy は overweight を使わず、underweight/flat 寄りに倒れた。AlphaEx は大きくマイナスで、MaxDD も B&H より悪化した。

## 検証目的

既存本線は `benchmark_position=1.0`, `abs_max_position=1.0`, `action_values=[0.0, 0.5, 1.0]` のため、B&H より強く張る overweight action が存在しない。超過リターン獲得手段が underweight のみに偏り、fold によって collapse しやすい。

Probe A/B/C は以下を切り分けるための検証だった。

- Probe A: DP soft teacher に戻すだけで改善するか
- Probe B: two-sided action space にすれば改善するか
- Probe C: DP ではなく feature-based two-sided teacher なら BC しやすいか

## Probe 定義

| Probe | teacher | action space | 目的 | 判定 |
|---|---|---|---|---|
| A | DP soft | `[0.0, 0.5, 1.0]` | DP teacher の効果確認 | 比較用。overweight 不可能 |
| B | DP soft | `[0.0, 0.5, 1.0, 1.25]` | two-sided action の効果確認 | teacher 分布が極端で危険 |
| C | feature_dual | `[0.0, 0.5, 1.0, 1.25]` | BC しやすい two-sided teacher 検証 | 正式実行で失敗 |

## 実行環境確認

MPS は使用可能であることを確認済み。

- `resolve_device("auto") == "mps"`
- `torch.backends.mps.is_available() == True`
- 実テンソル演算の結果デバイス: `mps:0`
- 学習ログ上も `Device: mps`

また、以前の MPS 最適化で入っていた `torch.compile` は checkpoint の state dict を `_orig_mod.*` キーにして通常ロード不能にしていたため撤去済み。修正後の checkpoint は通常ロードできることを確認した。

関連コミット:

- `b4bd675 Fix MPS training checkpoint compatibility`
- `79b3116 Remove retired probe configs`

## Probe C 正式実行

実行条件:

- config: `configs/probe_C_dual_twosided.yaml`
- fold: `4`
- device: `mps`
- stage 1: `wm -> bc`
- stage 2: `test -> test`
- WM: `max_steps=1000`
- BC: `n_epochs=8`

Teacher 分布:

- Oracle DP dist: `long=50% short=50% flat=0%`
- Feature dual teacher: `range=[+0.00,+1.25]`
- Oracle aim dist: `long=6% short=15% flat=79%`

WM / BC:

- WM best val loss: `0.9276`
- BC loss:
  - epoch 1: `1.3587`
  - epoch 8: `0.7033`
- checkpoint load: OK
- `_orig_mod` 汚染: なし

## Validation Adjust 結果

全候補が reject。

| scale | alpha | sharpe delta | maxdd delta | action dist | 判定 |
|---|---:|---:|---:|---|---|
| 0.750 | `-209.20pt` | `-0.883` | `-0.84pt` | `long=0% short=37% flat=63%` | reject |
| 1.000 | `-218.39pt` | `-0.966` | `-0.67pt` | `long=0% short=34% flat=66%` | reject |
| 1.500 | `-232.33pt` | `-1.103` | `-0.32pt` | `long=0% short=29% flat=71%` | reject |

選択された scale は `0.750` だが、これは成功ではなく「最も悪くない」候補に過ぎない。

## Test 結果

fold 4 test:

- Sharpe: `-0.907`
- Sortino: `-1.104`
- MaxDD: `-0.226`
- Calmar: `-1.406`
- TotalRet: `-0.0791`
- AlphaEx: `-29.18 pt/yr`
- Sharpe delta: `-0.933`
- MaxDD delta: `+4.14 pt`
- WinRate: `48.8%`
- M2: `MISS`
- Test dist: `long=0% short=24% flat=76%`
- PnL attr: `long=-0.0516 short=+0.0000 cost=0.0308 net=-0.0824`

## 失敗理由

1. overweight が全く使われていない

   two-sided action space を追加しても、test の `long=0%` であり、目的だった overweight 活用が起きていない。

2. alpha が大きくマイナス

   `AlphaEx=-29.18 pt/yr` で、B&H に対して明確に劣後している。

3. MaxDD が悪化

   `MaxDD delta=+4.14 pt` で、撤退条件に該当する。

4. policy が underweight/flat に偏った

   Test dist は `short=24% flat=76%`。overweight を含む two-sided 設計の検証としては失敗。

5. コスト負けしている

   `cost=0.0308` が net return を削っており、頻繁な調整が利益に結びついていない。

## 削除済み生成物

失敗 probe の checkpoint は削除済み。

削除済み:

- `checkpoints/probe_A_dp_soft`
- `checkpoints/probe_B_dp_soft_twosided`
- `checkpoints/probe_C_dual_twosided`
- その他過去の `checkpoints/probe_*`

現在残している checkpoint は本線の以下のみ。

- `checkpoints/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007`

削除済み config:

- `configs/probe_A_dp_soft.yaml`
- `configs/probe_B_dp_soft_twosided.yaml`
- `configs/probe_C_dual_twosided.yaml`
