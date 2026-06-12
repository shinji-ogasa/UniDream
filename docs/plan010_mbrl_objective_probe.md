# Plan010: Transformer WM -> Risk-Budget AC Probe

## 目的

Transformer World Model (WM) で vol / drawdown / crash / utility state を読み、
BC/AC Actor が `cost-adjusted wealth growth under risk budget` を最適化できるかを検証する。

最終目標は実モデル推論で `AlphaEx >= +3pt` かつ `MaxDDDelta <= -3pt`。
この文書の結果はすべて validation で設定を選び、test は report-only として扱う。

## 実装変更

- `ImagACTrainer` に `reward_objective: risk_budget` を追加。
- AC reward に log wealth, drawdown budget, terminal drawdown, downside/tail, turnover/overlay penalty, WM predictive risk/edge term を追加。
- Actor / predictive state / imagination rollout の NaN guard を追加。
- `EnsembleWorldModel` の `disagreement` を `unbiased=False` に修正。
- WM に `crash_head`, `drawdown_excess_head`, `position_utility_head` を追加。
- WM auxiliary head が `aux_use_raw_features: true` で raw feature skip を使えるようにした。
- `predict_auxiliary_from_encoded(..., features=...)` を追加し、AC predictive state 生成でも raw skip head を正しく使うようにした。

## 主要結果

### 1. AC objective only

| config / run | fold | test AlphaEx | test MaxDDDelta | 所感 |
|---|---:|---:|---:|---|
| `plan010_edge_allocator_v3_300.yaml` | 0 | +0.93pt | -1.87pt | 改善方向。ただし常時 de-risk 寄り。 |
| `plan010_edge_allocator_v3_300.yaml` | 1 | +0.20pt | -1.02pt | 弱い改善。 |
| `plan010_edge_allocator_v3_300.yaml` | 2 | -4.66pt | -1.12pt | 回復局面で alpha を削る。 |
| `plan010_crash_risk_allocator.yaml` raw-risk AC v1 | 4 | -0.28pt | -0.81pt | route de-risk 100%。ほぼ 0.95 exposure。 |
| raw-risk AC v2 edge/risk強化 | 4 | -0.95pt | -0.85pt | risk signal は入ったが、さらに underweight 寄り。 |
| raw-risk AC v3 class-balanced BC | 4 | -2.07pt | -0.57pt | BC collapse 形状は変化したが alpha 悪化。 |

### 2. Crash/DDEx WM head

fold3-5 の weak fold に対して risk target を重くした。

| WM | fold | h16 crash AUC | h32 crash AUC | h32 DD corr | 所感 |
|---|---:|---:|---:|---:|---|
| crash/DDEx baseline | 3 | 0.546 | 0.536 | ~0.020 | ほぼランダム。 |
| crash/DDEx baseline | 4 | 0.559 | 0.550 | ~0.002 | ほぼランダム。 |
| crash/DDEx baseline | 5 | 0.641 | 0.624 | ~0.071 | やや使える。 |
| risk-focus WM | 3 | 0.628 | 0.629 | 0.231 | 改善。 |
| risk-focus WM | 4 | 0.525 | 0.517 | 0.022 | 改善せず。 |
| risk-focus WM | 5 | 0.659 | 0.648 | 0.211 | 改善。 |
| risk-focus raw-skip WM | 3 | 0.611 | 0.610 | 0.195 | latent-onlyより少し低下。 |
| risk-focus raw-skip WM | 4 | 0.588 | 0.574 | 0.122 | fold4 は明確に改善。 |
| risk-focus raw-skip WM | 5 | 0.632 | 0.636 | 0.176 | 使えるが強信号ではない。 |

raw skip は fold4 の risk signal には効いた。ただし risk-only selector は上昇局面で underexposure になり、test alpha が崩れた。

### 3. Position utility WM head

| WM / selector | folds | pass +3/-3 | Alpha mean | Alpha worst | MaxDD mean | MaxDD worst | 所感 |
|---|---:|---:|---:|---:|---:|---:|---|
| position utility baseline / small_overlay | 3-5 | 0/3 | +7.24pt | -2.91pt | +0.34pt | +0.64pt | alpha は出るが DD は悪化。 |
| ranked utility CE追加 / small_overlay | 3-5 | 0/3 | -1.00pt | -4.76pt | +0.47pt | +1.30pt | rank補助は悪化。 |
| raw-skip utility / small_overlay | 3-5 | 0/3 | -35.72pt | -131.59pt | -1.48pt | -0.53pt | DD方向に寄りすぎ、alpha崩壊。 |

utility head の順位診断は多くの fold で Pearson が負になった。
平均 utility は学ぶが、actionable な rare state の順位付けが汎化していない。

## 診断

現時点で `+3/-3` を 0-12 にスケールする見込みは低い。

理由:

1. WM は vol/crash risk をある程度読めるが、alpha を作る return/utility edge は汎化していない。
2. risk-only は DD を少し改善できても、B&H 対比 alpha を削りやすい。
3. AC は risk_budget 目的にしても route が de-risk に潰れやすい。
4. BC prior が fold4 で弱く、AC が悪い prior から抜けられていない。
5. raw feature skip は risk AUC を改善したが、utility allocation では underweight 過多を誘発した。

## 次に試す価値がある方向

係数ガチャではなく、構造変更が必要。

1. Actor action space を `target exposure = 1.0 + small overlay` に制限し、full route/de-risk class を廃止する。
2. BC teacher を raw Oracle aim ではなく、`benchmark-relative overlay teacher` に作り直す。
3. WM は utility scalar ではなく、`risk state` と `safe-to-overweight state` を別 head / 別 loss で学習する。
4. AC reward に `low-risk underweight penalty` と `high-risk-only de-risk allowance` を明示する。
5. fold4 を gate fold とし、fold4 で AlphaEx >= 0 かつ MaxDDDelta <= -1pt を超えない限り、0-12 へ広げない。

## 現時点の保存判断

実モデル推論で demo に移すべきモデルはまだない。

最も情報価値が高かった成果は `checkpoints/plan010_risk_focus_raw_wm_s007` の raw-skip risk WM。
ただし AC policy は未達なので、`unidream-space` へ差し替える段階ではない。
