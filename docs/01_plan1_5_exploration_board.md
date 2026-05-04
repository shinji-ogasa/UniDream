# Plan 1-5 探索ボード 全結果サマリ

## Plan 1: 市場イベントラベル検証 → **失敗**

**目的**: route teacher を teacher inventory shortcut から脱却させ、market-state event label として再設計。BC/ACへ戻る前の教師ラベル separability 確認。

### 作ったラベル
- `risk_off`: 将来DD窓に入る前にde-riskでB&HよりMaxDDを改善できる局面
- `recovery`: DD後にbenchmark復帰価値のある局面
- `overweight`: B&H以上に張ってよい局面
- `active`: risk_off OR recovery OR overweight

### 結果

| target | best feature | density | AUC mean | AUC worst | false-active worst |
|---|---|---|---|---|---|
| active | context_no_position | 0.995 | 0.755 | 0.675 | 0.056 |
| risk_off | context_no_position | 0.958 | 0.679 | 0.631 | 0.157 |
| recovery | wm | 0.250 | 0.569 | 0.567 | 0.177 |
| overweight | raw | 0.095 | 0.621 | 0.563 | 0.135 |

### tightened v2 / h64 / HGB も全滅

| variant | active AUC worst | 判定 |
|---|---|---|
| v2 h32 (tightened) | 0.532 | fail |
| h64 logistic | 0.527 | fail |
| v2 h32 HGB | 0.508 | fail |

### 結論

market event label は fold4 では一部読めるが fold5/fold6 で AUC 0.50 付近に落ちる。非線形 probe（HGB）でも改善しない。BC/AC最適化へは進めないと判断。event label の複数 fold 再現性が未達。

### 次方向

maxDD window overlap label、post-fire drawdown contribution label を作り直す提案あり。
risk_off threshold の細かい探索、AC制限解除、route head unlock、fold4単体採用は禁止とされた。

---

## Plan 2: 8レーン探索ボード → **D+A+pullback が最良**

**目的**: 単一路線ではなく、複数仮説を並列に安く検証して勝ち筋だけ残す。

### 8レーン

| Lane | 内容 | 14fold結果 | 判定 |
|---|---|---|---|
| A | triple-barrier / meta-labeling guard | h32/h64 downside guard として有効 | 残す（guard用途） |
| B | safe policy improvement (Plan7逸脱制限) | Alpha mean +0.032, worst -1.736 | **不採用** |
| C | conservative offline RL scoring | — | AC未移行 |
| D | risk-sensitive / DD objective | Alpha mean +3.747, worst 0.000 | **残す** |
| E | model uncertainty (bootstrap) | Alpha mean -7.106, worst -99.479 | **不採用** |
| F | ranking/listwise action selector | Alpha mean -0.221, worst -5.334, turnover 42.0 | **不採用** |
| G | regime split (vol regime) | Alpha mean -1.493, worst -14.487 | **不採用** |
| H | strict validation/PBO (nested leave-one-fold) | — | **残す** |

### 最良候補: `D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly`

設計: D risk-sensitive candidate utility + h32 triple-barrier downside guard + threshold floor 0.001 + auto cooldown (0 or 32 bars) + eval-only pullback/false-de-risk guard

**全14fold aggregate:**

| 指標 | 値 |
|---|---|
| AlphaEx mean | +3.747 pt/yr |
| AlphaEx worst | 0.000 pt/yr |
| MaxDDΔ mean | -0.071 pt |
| MaxDDΔ worst | 0.000 pt |
| SharpeΔ mean | +0.014 |
| turnover max | 3.000 |
| long max | 0.000 |
| pass rate | 0.286 |
| PBO | 0.500 |

**fold別詳細:**

| fold | AlphaEx | MaxDDΔ | turnover | flat |
|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.0 | 1.000 |
| 1 | +3.996 | 0.000 | 1.0 | 0.997 |
| 2 | 0.000 | 0.000 | 0.0 | 1.000 |
| 3 | +41.381 | 0.000 | 1.0 | 0.999 |
| 4 | +0.423 | -0.661 | 2.5 | 0.999 |
| 5 | 0.000 | 0.000 | 0.0 | 1.000 |
| 6 | +0.037 | -0.022 | 0.5 | 1.000 |
| 7~9,12 | 0.000 | 0.000 | 0.0 | 1.000 |
| 10 | +0.752 | -0.222 | 2.7 | 0.999 |
| 11 | +5.689 | 0.000 | 1.0 | 1.000 |
| 13 | +0.178 | -0.085 | 3.0 | 0.999 |

### pullback guard の効果

fold10/12 で `auto_cd floor001` が崩れたのを `pullback eval-only guard` が止めた:

| variant | fold10 Alpha | fold12 Alpha |
|---|---|---|
| auto_cd floor001 (pullbackなし) | -0.434 | -0.272 |
| + pullback eval-only | +0.752 | 0.000 |

### 勝ち筋

```
残す: A (h32 guard), D (risk-sensitive utility), H (nested/allfold), pullback guard, auto cooldown
捨てる: B, E, F, G (as-is), validation MaxDD requirement, fixed cooldown only, threshold-only tuning
```

### triple-barrier label 品質

| target | AUC mean | AUC worst | false-active worst |
|---|---|---|---|
| h16_k100_down | 0.588 | 0.550 | 0.420 |
| h32_k125_down | 0.600 | 0.572 | 0.501 |
| h64_k150_down | 0.627 | 0.569 | 0.434 |
| h64_k150_up_safe | 0.629 | 0.557 | 0.482 |

AUC 0.57-0.63 で単体分類器としては強くないが、risk-sensitive selector の gate としては十分に効いた。

### 判断

「raw route/AC拡張ではなく、D risk-sensitive utility + A downside guard + false-de-risk pullback blocker」が Plan2 内で最も再現性のある方向。平均Alphaはfold3依存があるので、まだ full actor や route unlock には戻さない。

---

## Plan 3: WM Overlay 独立実装

**目的**: Plan2 の ridge ベース候補を実 WM 推論に置き換え、独立 overlay 化。

### Round 1 結果

**Ridge 版** (`plan3_overlay_probe.py`): Plan2 と一致確認（f0456 で再現）。D_baselineだけfold6で崩れ、guard/cooldownで修復。

**WM 版** (`plan3_wm_overlay.py`): f456 で実行

| fold | AlphaEx | MaxDDΔ | SharpeΔ | turnover | active |
|---|---|---|---|---|---|
| 4 | 0.000 | 0.000 | 0.000 | 0.0 | 0 |
| 5 | **+8.431** | -0.121 | +0.008 | 0.2 | 1 |
| 6 | 0.000 | 0.000 | 0.000 | 0.0 | 0 |

Aggregate: AlphaEx mean +2.810, worst 0.000, MaxDDΔ mean -0.040

### Ridge vs WM 比較

| fold | Ridge (Plan2) | WM (Plan3) |
|---|---|---|
| 4 | +0.423 | 0.000 |
| 5 | 0.000 | +8.431 |
| 6 | +0.037 | 0.000 |

WM と ridge は**異なる fold に反応**。相補的。

### 注意点

WM fold5 +8.431 は後日再現不能と判明。リファクタリングで `/100` スケール修正・`_threshold_grid` 変更・`danger_guard` quantile 変更により結果が変わった。修正後も WM return head がランダム（IC~0）のため utility 改善はノイズ。

---

## Plan 4: WM 校正・統合検証

### A: WM 予測値校正

WM aux head 予測値 vs 実績値 h32 の相関（f456、risk_target_scale=100 で割り戻し後）:

| 信号 | pearson mean | spearman mean | scale_ratio | 判定 |
|---|---|---|---|---|
| return | 0.002 | 0.015 | 0.098 | **ランダム。使えない** |
| vol | 0.320 | 0.342 | 0.207 | **使える。IC 0.3-0.6 (fold4で強い)** |
| dd | NA | NA | NA | **弱い (fold4 pearson 0.208, 他はランダム)** |

### C: Blocked Event Attribution

block されたイベントの counterfactual:

| fold | blocked events | mean_actual_util | util>0 rate | 判定 |
|---|---|---|---|---|
| 4 | 700 | -0.005 | 10% | **正ブロック** |
| 5 | 700 | -0.002 | 16% | **正ブロック** |
| 6 | 700 | -0.003 | 15% | **正ブロック** |

guard は過剰抑制ではなく、危険イベントを正しく止めている。

### B: Ridge+WM Ensemble

| variant | fold4 Alpha | fold4 turnover | fold5 Alpha | fold6 Alpha | 判定 |
|---|---|---|---|---|---|
| OR | -1.862 | 80.0 | -1.961 | +0.036 | **爆発** |
| AND | 0.000 | 0.0 | 0.000 | 0.000 | 全部neutral |
| MAX | +1.029 | 177.6 | +1.989 | -0.024 | **爆発** |

単純な OR/AND/MAX は全滅。ridge が fold4/5 で暴発。

### D/E: Soft Throttle / Utility Grid

- Soft throttle: 全 neutral（guard 緩和の前に候補が threshold 通過しない）
- Utility grid: fold5 のみ dd1.0_vol1.0 で +6.312。fold4/6 は全パラメータで neutral

---

## Plan 5: 信号再解釈・ridge+WM統合2.0

### Lane 0: Plan3 再現性確認

Plan3 +8.431 再現不能。3つのバグを特定・修正:
1. `_threshold_grid` → リファクタで粗くなった（実測値が候補から消えた）
2. `danger_guard` → quantile(0.25) が 旧 median*0.5 より厳しくなった
3. WM予測値 `/100` 未実施 → `risk_target_scale=100` で訓練、実績値と10-100倍ずれ

修正後も WM return head がランダムのため utility 改善はノイズ（improve mean=0.00005）。

### Lane C: パイプライン分解

| fold | WM_pre | WM_post_thr | WM_danger | WM_final | R_pre | R_post_thr | R_final |
|---|---|---|---|---|---|---|---|
| 4 | 5119 | 2 | 0 | 0 | 6189 | 4071 | 4071 |
| 5 | 2378 | 3 | 0 | 0 | 1255 | 1052 | 1040 |
| 6 | 3134 | 0 | 0 | 0 | 2247 | 19 | 19 |

WM: threshold=0.001 で 99.97% が消滅。Ridge: 候補多数だが暴発。

### Lane A: Vol-Only Utility

return head を utility から外し vol head のみ → 全 fold で neutral。vol-only utility は常に負（非benchmarkのcost > 期待リターン0）。

### Lane B: Ridge Primary + WM Vol Veto

ridge 単体で fold4 turnover=187, fold5 alpha=-397 と暴発。vol veto でも止められず。

### Lane F: Pullback Recovery Label → **本命成果**

| label | fold | AUC | density | false_active |
|---|---|---|---|---|
| pullback_recovery | 4 | **0.909** | 0.017 | 0.156 |
| pullback_recovery | 5 | **0.910** | 0.016 | 0.121 |
| pullback_recovery | 6 | **0.930** | 0.021 | 0.108 |
| false_derisk | 4 | 0.495 | 0.264 | 0.161 |
| false_derisk | 5 | 0.517 | 0.329 | 0.194 |
| false_derisk | 6 | 0.531 | 0.311 | 0.280 |

`pullback_recovery` label が **AUC 0.91-0.93** で高精度識別可能。state features から pullback 後の回復局面（de-risk すべきでない局面）を検出できる。
`false_derisk` は読めず（AUC~0.5）。

### Plan 5 結論

- WM return head: 使えない（IC~0）
- WM vol head: risk veto に使える
- WM dd head: 弱い（fold4のみ pearson 0.2）
- Ridge: Plan2 の探査ボード内では安定、簡易実装では暴発
- **pullback_recovery label (AUC 0.91-0.93) が最大の成果**

---

## 全体結論

| 手法 | 状態 |
|---|---|
| Plan2 ridge D+A+pullback | ✅ 全14fold worst AlphaEx>=0。安全だが neutral-heavy |
| pullback_recovery label | ✅ AUC 0.91-0.93。guard改善に有望 |
| WM return head | ❌ ランダム（IC~0） |
| WM vol head | ⚠️ IC 0.3-0.6。リスクフィルタとして利用可能 |
| WM dd head | ❌ ほぼ読めず |
| AC / route unlock | ❌ 禁止中。教師ラベルが不安定 |
| BC 単体 | ⚠️ 条件付き。market event label の separability が未達 |

次ステップ候補: pullback_recovery label を guard に組み込む、Plan2 ridge overlay の active fold 増加、WM all14（checkpoint 不足）
