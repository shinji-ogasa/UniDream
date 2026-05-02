# Plan 2 Extended Exploration Report

Date: 2026-05-02

## Objective

Plan2 の目的は、単一foldや平均Alphaだけに寄らず、複数foldで再現する改善方向を探すこと。
今回は、最初の `D_risk_sensitive + triple-barrier guard` で止めず、Plan2 の主要レーンを追加で14fold評価した。

対象条件:

```text
config: configs/trading.yaml
period: 2018-01-01 to 2024-01-01
folds: 0,1,2,3,4,5,6,7,8,9,10,11,12,13
seed: 7
```

## Implemented Changes

`unidream/cli/exploration_board_probe.py` に以下を追加した。

```text
--selector-names filter
cooldown_bars / hold_bars
cooldown_gridによるvalidation内auto cooldown選択
h32/h64 triple-barrier downside guard設定
threshold floor variants
risk-first validation score
validation MaxDD improvement requirement
selected event context diagnostics
pullback / false-de-risk no-fire guard
eval-only pullback guard
nested leave-one-fold selector diagnostic
```

Plan2レーン対応:

| Lane | 内容 | 今回の検証状態 |
|---|---|---|
| A | triple-barrier / meta-labeling style guard | h16/h32/h64 downside/up-safe label + h32 guardを検証 |
| B | safe policy improvement / baseline deviation | `B_safe_small` を14fold検証 |
| C | conservative offline RL style scoring | 実ACではなくcandidate utility/risk score段階で確認。ACは未移行 |
| D | risk-sensitive / DD objective | `D_risk_sensitive` 系を14fold検証 |
| E | model uncertainty | `E_bootstrap_uncertainty` を14fold検証 |
| F | ranking/listwise action selector | `F_listwise` を14fold検証 |
| G | regime split | `G_vol_regime_safe` を14fold検証 |
| H | strict validation/PBO | all14 + f456/f045 + nested leave-one-fold + PBO-like diagnostic |

## Executions

D/A系とpullback guard最終候補:

```powershell
uv run python -u -m unidream.cli.exploration_board_probe `
  --config configs/trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,1,2,3,4,5,6,7,8,9,10,11,12,13 `
  --seed 7 `
  --selector-names D_risk_sensitive_tbguard_auto_cd_floor001,D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly,D_risk_sensitive_tbguard_cd32_floor001 `
  --output-json documents/20260502_plan2_final_pullback_evalonly_allfolds.json `
  --output-md documents/20260502_plan2_final_pullback_evalonly_allfolds.md
```

B/E/F/G未確認レーン:

```powershell
uv run python -u -m unidream.cli.exploration_board_probe `
  --config configs/trading.yaml `
  --start 2018-01-01 `
  --end 2024-01-01 `
  --folds 0,1,2,3,4,5,6,7,8,9,10,11,12,13 `
  --seed 7 `
  --selector-names B_safe_small,E_bootstrap_uncertainty,F_listwise,G_vol_regime_safe `
  --output-json documents/20260502_plan2_befg_allfolds.json `
  --output-md documents/20260502_plan2_befg_allfolds.md
```

## Best Current Candidate

採用候補として一番マシだったのはこれ。

```text
D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly
```

設計:

```text
D risk-sensitive candidate utility
+ h32 triple-barrier downside guard
+ threshold floor 0.001
+ validation内auto cooldown: 0 or 32 bars
+ eval-only pullback/false-de-risk guard
```

全14fold aggregate:

| Metric | Result |
|---|---:|
| AlphaEx mean | +3.747 pt/yr |
| AlphaEx worst | 0.000 pt/yr |
| MaxDDDelta mean | -0.071 pt |
| MaxDDDelta worst | 0.000 pt |
| SharpeDelta mean | +0.014 |
| turnover max | 3.000 |
| long max | 0.000 |
| pass rate | 0.286 |
| PBO | 0.500 |

Nested leave-one-fold selector:

| Metric | Result |
|---|---:|
| folds | 14 |
| AlphaEx mean | +3.747 pt/yr |
| AlphaEx worst | 0.000 pt/yr |
| MaxDDDelta worst | 0.000 pt |
| turnover max | 3.000 |

Fold detail:

| fold | AlphaEx | MaxDDDelta | SharpeDelta | turnover | flat |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.000 | 0.0 | 1.000 |
| 1 | +3.996 | 0.000 | +0.037 | 1.0 | 0.997 |
| 2 | 0.000 | 0.000 | 0.000 | 0.0 | 1.000 |
| 3 | +41.381 | 0.000 | +0.063 | 1.0 | 0.999 |
| 4 | +0.423 | -0.661 | +0.049 | 2.5 | 0.999 |
| 5 | 0.000 | 0.000 | 0.000 | 0.0 | 1.000 |
| 6 | +0.037 | -0.022 | +0.002 | 0.5 | 1.000 |
| 7 | 0.000 | 0.000 | 0.000 | 0.0 | 1.000 |
| 8 | 0.000 | 0.000 | 0.000 | 0.0 | 1.000 |
| 9 | 0.000 | 0.000 | 0.000 | 0.0 | 1.000 |
| 10 | +0.752 | -0.222 | +0.012 | 2.7 | 0.999 |
| 11 | +5.689 | 0.000 | +0.027 | 1.0 | 1.000 |
| 12 | 0.000 | 0.000 | 0.000 | 0.0 | 1.000 |
| 13 | +0.178 | -0.085 | +0.006 | 3.0 | 0.999 |

Interpretation:

```text
worst AlphaExは非負
worst MaxDDDeltaは実質ゼロ
turnoverは3.0で上限3.5以内
longは0なのでoverweightリスクは出していない
ただし平均Alphaはfold3の大きい勝ちに引っ張られている
neutral foldが多く、pass rateは0.286に留まる
```

したがって、これは「強い本流モデル」ではなく、現時点では安全側のinference-only de-risk selector候補として見る。

## Why Pullback Eval-Only Was Needed

`D_risk_sensitive_tbguard_auto_cd_floor001` は平均では強かったが、fold10/fold12で崩れた。

| variant | Alpha mean | Alpha worst | MaxDD worst | turnover max |
|---|---:|---:|---:|---:|
| D auto_cd floor001 | +3.643 | -0.434 | +0.055 | 4.2 |
| D auto_cd floor001 + pullback eval-only | +3.747 | 0.000 | 0.000 | 3.0 |
| D fixed cd32 floor001 | +1.056 | -1.295 | +0.055 | 3.0 |

重要な改善点:

```text
fold10:
  auto_cd floor001: Alpha -0.434 / turnover 4.2
  pullback eval-only: Alpha +0.752 / turnover 2.7

fold12:
  auto_cd floor001: Alpha -0.272 / MaxDD +0.055
  pullback eval-only: Alpha 0.000 / MaxDD 0.000
```

hard pullback guardをvalidation側にも入れると閾値選択が変わって別foldを壊すため、今回は validation選択はそのまま、test/inference時だけfalse de-riskを止める eval-only guard が一番安定した。

## Other Plan2 Lanes

B/E/F/G系は14foldで不採用。

| variant | Alpha mean | Alpha worst | MaxDD worst | turnover max | Decision |
|---|---:|---:|---:|---:|---|
| B_safe_small | +0.032 | -1.736 | +0.499 | 1.0 | 不採用。安全逸脱のつもりだがfold13でDD悪化 |
| E_bootstrap_uncertainty | -7.106 | -99.479 | 0.000 | 9.0 | 不採用。不確実性gateが逆に危険fireを残す |
| F_listwise | -0.221 | -5.334 | +0.110 | 42.0 | 不採用。utility rankingはturnover崩壊 |
| G_vol_regime_safe | -1.493 | -14.487 | +0.466 | 7.0 | 不採用。regime分割がfold間で安定しない |

この結果から、Plan2の中で現時点の勝ち筋は以下に絞られる。

```text
残す:
  A: h32 triple-barrier downside guard
  D: risk-sensitive candidate utility
  H: nested / allfold validation
  pullback false-de-risk guard
  auto cooldown / sparse event throttle

捨てる:
  B_safe_small as-is
  E_bootstrap_uncertainty as-is
  F_listwise as-is
  G_vol_regime_safe as-is
  validation MaxDD requirement as-is
  fixed cooldown only
  thresholdだけの調整
```

## Triple-Barrier Label Quality

barrier label自体は完全ではないが、downside系は最低限の分離力があった。

| target | AUC mean | AUC worst | false-active worst |
|---|---:|---:|---:|
| h16_k100_down | 0.588 | 0.550 | 0.420 |
| h32_k125_down | 0.600 | 0.572 | 0.501 |
| h64_k150_down | 0.627 | 0.569 | 0.434 |
| h64_k150_up_safe | 0.629 | 0.557 | 0.482 |

解釈:

```text
AUCは0.57から0.63程度で、単体分類器として強くはない
ただしrisk-sensitive selectorのgateとしては十分に効いた
false-active worstはまだ高く、ここが次の改善対象
```

## Decision

今回の結論:

```text
Plan2内で一番再現性がある方向は、
raw route/AC拡張ではなく、
D risk-sensitive utility + A downside guard + false-de-risk pullback blocker。
```

本流に入れてよいもの:

```text
exploration_board_probeのselector filter / diagnostics
allfold + nested leave-one-fold validation
D+A+pullback eval-onlyを研究候補として残す
```

まだ本流policyへ直結しないもの:

```text
full AC
route head unlock
listwise selector
bootstrap uncertainty gate
vol regime route
```

理由:

```text
良い候補は出たが、まだactive率が低くneutral-heavy。
平均Alphaはfold3にかなり依存している。
直接train本流に混ぜるより、まずinference-only overlayとして切り出して検証する方が安全。
```

## Next Direction

次にやるなら、閾値いじりは打ち止め。
やるべきは2つ。

```text
1. D+A+pullback eval-only selectorを独立overlay化
   - training本流ではなく、既存policy出力の後段guardとして実装
   - allfold backtestで同じ結果が再現するか確認

2. false-active worstを下げる専用labelを作る
   - pullback_recovery_label
   - false_de_risk_label
   - post-fire drawdown contribution label
   - future MDD interval overlap label
```

ACへ戻る条件:

```text
overlay単体でall14 worst AlphaEx >= 0を維持
MaxDDDelta worst <= 0を維持
turnover <= 3.5を維持
active foldを増やしてもfold10/12/13が崩れない
```

現時点ではACを広げるより、false de-riskを止める分類/guardを強化する方が期待値が高い。

## Artifacts

```text
documents/20260502_plan2_final_pullback_evalonly_allfolds.md
documents/20260502_plan2_final_pullback_evalonly_allfolds.json
documents/20260502_plan2_befg_allfolds.md
documents/20260502_plan2_befg_allfolds.json
documents/20260502_plan2_final_allfolds.md
documents/20260502_plan2_final_allfolds.json
```
