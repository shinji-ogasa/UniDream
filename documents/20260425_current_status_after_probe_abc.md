# UniDream 現状整理: Probe A/B/C 失敗後

作成日: 2026-04-25 JST

## 結論

現時点で M2 達成候補はまだない。

ただし、探索はかなり絞れている。Probe A/B/C の結果により、`DP soft` や `two-sided action space` をそのまま入れるだけでは解決しないことが分かった。今残すべき本線は、`dualresanchor + feature_stress_tri + residual shift only` 系である。

今の最重要課題は、`alpha_excess` や `maxdd_delta` 以前に、policy が `short` または `flat` に寄りすぎること。特に最新の本線候補は `fold4/fold5` で drawdown 改善は出るが、行動分布が `short 90-97%` まで偏っている。

## 最良の結果

### 1. 元の fold4 local best: dualresanchor

config:

- `configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor.yaml`

fold4 では最も良い局所解だった。

主要結果:

- `alpha_excess +0.06 pt/yr`
- `maxdd_delta -0.81 pt`
- `short 58% / flat 42%`

評価:

- fold4 単体では一番バランスが良い。
- ただし fold0/fold5 で opposite collapse が出る。
- multi-fold 本線には昇格できない。

### 2. 現在の新本線候補: feature_stress_tri + residual shift only

config:

- `configs/medium_l1_bc_continuous_exec_shortmass_regimebias_shift15_blend625_bandtarget_tradeonly_dualresanchor_stresstri_shiftonly_s007.yaml`

fold4:

- `alpha_excess -0.04 pt/yr`
- `maxdd_delta -0.85 pt`
- `short 90% / flat 10%`

fold5:

- `alpha_excess +1.19 pt/yr`
- `maxdd_delta -0.81 pt`
- `short 97% / flat 3%`

評価:

- fold5 の `flat 100%` collapse を抜けた点は大きい。
- `alpha >= 0` と `maxdd_delta < 0` に近い。
- ただし行動が `short` に寄りすぎており、collapse guard 的にはまだ危険。
- 現時点では「採用候補」ではなく「一番見込みがある探索軸」。

## 良さそうな方向性

### 1. feature_stress_tri + residual shift only

HMM regime ではなく、feature stress 由来の regime を使う方向は一定の改善があった。特に `target bias` ではなく `residual shift` のみに限定した場合、fold5 の collapse が改善した。

今後見る価値があるもの:

- `shift_scale 0.06 / 0.07 / 0.08` の再確認
- `short` 偏りを抑える制約
- `fold0` での proper WM + BC 評価

### 2. teacher を変えるより routing を安定化する方向

Probe A/B/C の結果から、teacher/action space を大きく変えるだけでは policy が素直に改善しないことが分かった。

今の問題は、teacher 自体よりも learner が「いつ benchmark に戻るか」を安定して学べていない点が大きい。

見る価値があるもの:

- underweight から benchmark へ戻る recovery 条件の強化
- trade cost を意識した低頻度 switching
- residual shift に上限・平滑化・戻り制約を入れる設計

### 3. DP soft は補助的に再検討

Probe A/B/C では本線化できなかったが、DP soft そのものが完全に無価値とは言い切れない。問題は、two-sided action space や feature_dual と組み合わせたときに、overweight が使われず、コスト負けしたこと。

再検討するなら、teacher 主体ではなく confidence / sample weighting として使う方が良い。

## ダメだったこと

### 1. Probe A/B/C

参照:

- `documents/20260425_probe_abc_failure_report.md`

Probe A/B/C は現時点では本線化しない。

特に正式実行された Probe C は fold4 test で明確に失敗した。

Probe C fold4 test:

- `AlphaEx -29.18 pt/yr`
- `Sharpe delta -0.933`
- `MaxDD delta +4.14 pt`
- `long 0% / short 24% / flat 76%`
- `cost 0.0308`

失敗理由:

- two-sided action space を入れても overweight を使わなかった。
- alpha が大きくマイナス。
- MaxDD が B&H より悪化。
- trade cost が net return を削った。

### 2. regime prior 系

以下は fold4 の時点で short collapse した。

- regime gate prior
- regime residual shift prior
- prior alpha blend

評価:

- teacher 由来の regime prior を直接 gate/shift に入れると強すぎる。
- fold4 の局所解を壊すため、いったん撤退。

### 3. no-regime 系

`dualresanchor` から regime 条件を外すと flat collapse した。

代表結果:

- `bc_short 2.5%`
- `bc_flat 97.5%`

評価:

- regime 情報は必要。
- ただし HMM regime や target bias の使い方は不安定。

### 4. feature_stress target bias

`feature_stress_tri` でも `target bias` を使うと short collapse した。

代表結果:

- `targetonly`: `bc_short 99.7%`
- `shiftonly`: `bc_short 85.2%`

評価:

- feature_stress は使える可能性がある。
- 使い道は target logits ではなく residual shift 側。

### 5. blend 上げ

`stresstri_shiftonly_s007` で `infer_logits_target_blend` を上げると flat collapse した。

結果:

- `blend675`: fold4/fold5 とも `flat 100%`
- `blend800`: fold4/fold5 とも `flat 100%`

評価:

- 現在の有効域は `blend 0.625` 近辺。
- inference blend で行動偏りを直す余地は小さい。

## 今の課題

### 1. 行動分布がまだ偏りすぎる

現在の本線候補は fold4/fold5 で `maxdd_delta < 0` を出せているが、`short 90-97%` まで偏っている。

これは M2 条件の `collapse_guard` や実運用上の安定性に引っかかる。

### 2. benchmark への復帰が弱い

今の learner は underweight に入ることは学び始めているが、benchmark に戻る条件が弱い。

そのため、下げ回避の形が `de-risk` ではなく `short-heavy hold` に近くなる。

### 3. overweighing を許しても使われない

Probe B/C で action space に `1.25` を足しても、正式実行された Probe C では `long 0%` だった。

つまり、単に action space を広げるだけでは不十分。overweight を使う teacher / loss / routing が必要。

### 4. cost-aware な切り替えが弱い

Probe C では `cost=0.0308` が net return を大きく削った。

今後は alpha だけでなく、turnover / cost / avg_hold を明示的に見る必要がある。

### 5. fold0 が未確認

現在の本線候補 `stresstri_shiftonly_s007` は fold4/fold5 まで見えているが、fold0 の proper 評価がまだ不足している。

fold0 で short collapse が強く出るなら、この方向は追加制約なしでは本線化できない。

## 次にやるべきこと

優先順は以下。

1. `stresstri_shiftonly_s007` を fold0 に当てる
2. fold0 で `alpha`, `maxdd`, `short/flat`, cost を確認する
3. `short` 偏りを抑える residual shift 制約を入れる
4. recovery / benchmark 復帰を強める loss または routing を再設計する
5. 良化が出た場合だけ fold4/fold5/fold0 の 3 fold で再評価する

今すぐ Probe A/B/C 系に戻る優先度は低い。

## 現在の暫定判断

残す:

- `dualresanchor`
- `feature_stress_tri`
- `residual shift only`
- `stresstri_shiftonly_s007`

切る:

- Probe A/B/C
- feature_dual two-sided
- direct regime prior
- no-regime
- target bias
- blend 0.675 以上

次の主戦場:

- `fold0` での `stresstri_shiftonly_s007` 評価
- `short` 偏り抑制
- benchmark 復帰の安定化
- cost-aware な低頻度切り替え
