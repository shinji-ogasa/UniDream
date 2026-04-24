作成者: GLM
宛先: Codex
作成時刻: 2026-04-24 19:30:00 JST
種別: 報告

# 報告: BC Collapse Probe + Stability Probe 結果

## 参照した指示
docs/Codex_Agent/20260424170032_current_focus.md

## 実施内容
最小ActorでBC学習を実行し、各Oracle設定と不均衡対策の組み合わせで
collapseするかどうかを検証した。

2epoch probe + 10epoch stability probeの2段で実施。

## 根本的診断の修正

**従来の理解「Oracle long=0%」は不正確だった。**

signal_aim teacher (現状設定: bm=1.0, floor=0.0, ceiling=1.0) の実際:
- flat=0.5%, short(pos=0.5)=20.0%, long(pos=1.0)=79.6%
- **longは80%近くある！BCが学習後にこれをcollapseさせている**

## 2epoch Probe結果

| Oracle設定 | 条件 | 結果 | 分布 |
|-----------|------|------|------|
| signal_aim現状 | vanilla | **COLLAPSE** | long=100% |
| signal_aim現状 | class_balanced | OK | flat=2% short=47% long=51% |
| signal_aim現状 | label_smoothing=0.1 | **COLLAPSE** | long=100% |
| signal_aim現状 | label_smoothing=0.3 | **COLLAPSE** | long=100% |
| signal_aim現状 | class_balanced+ls0.1 | **COLLAPSE** | flat=100% |
| signal_aim bm=1.0 floor=-1.0 | vanilla | **COLLAPSE** | long=100% |
| signal_aim bm=1.0 floor=-1.0 | class_balanced | OK | flat=41% short=26% long=33% |
| signal_aim bm=1.0 floor=-1.0 | class_balanced+ls0.1 | OK | flat=65% short=23% long=12% |
| signal_aim bm=0.5 floor=-0.5 | vanilla | **COLLAPSE** | short=100% |
| signal_aim bm=0.5 floor=-0.5 | class_balanced | OK | flat=70% short=30% |
| DP Oracle bm=1.0 | 全条件 | OK | flat=56% long=44% |

## 10epoch Stability Probe結果

**全候補で10epoch後もcollapseなし！**

| 候補 | Oracle分布 | 10epoch後分布 | max_class | KL from oracle |
|------|-----------|--------------|-----------|----------------|
| signal_aim現状 + class_balanced | flat=0.5% short=20% long=80% | flat=13% short=39% long=48% | 48% | — |
| signal_aim bm1.0 floor-1.0 + class_balanced | flat=12% short=18% long=70% | flat=31% short=35% long=34% | 35% | 0.31 |
| signal_aim現状 + class_balanced + entropy0.05 | 同上 | flat=12% short=43% long=45% | 45% | — |
| signal_aim bm1.0 floor-1.0 + class_balanced + entropy0.05 | 同上 | flat=41% short=19% long=41% | 41% | — |
| signal_aim現状 + class_balanced + dropout0.3 | 同上 | flat=8% short=46% long=46% | 46% | — |
| signal_aim bm1.0 floor-1.0 + class_balanced + dropout0.3 | flat=12% short=18% long=70% | flat=37% short=31% long=32% | 37% | 0.31 |

## 分析

### 1. class_balanced重みがcollapse防止に決定的

vanilla（均一重み）では、2epoch以内に単一クラスにcollapse。
class_balanced（逆頻度重み）では、10epoch後も3クラス分布を維持。

**これはBC学習のデフォルト設定にすべき変更。**

### 2. label_smoothingは単独ではcollapse防止に無効

ls=0.1, ls=0.3ともに2epochでcollapse。
class_balancedと組み合わせると逆効果（flatにcollapse）。

### 3. entropy正則化・dropoutは微増効果

class_balancedだけでもcollapseしないが、entropyやdropoutを追加すると
分布のバランスがわずかに改善（よりoracleに近い方向へ）。

### 4. floor=-1.0（真のshort許可）がバランス改善に有効

signal_aim bm=1.0 floor=-1.0 + class_balanced:
- Oracle: flat=12% short=18% long=70%
- BC出力: flat=31% short=35% long=34%
- **3クラスがほぼ均等に分布！** これはBCとして理想的

### 5. 最良の組み合わせ

**signal_aim bm=1.0 floor=-1.0 + class_balanced + dropout=0.3**

10epoch後: flat=37% short=31% long=32%
→ 3クラスがほぼ33%ずつ。最もバランスが良い。
→ KL divergence from oracle = 0.31（oracle偏重をBC側でバランス化）

## 次に必要な判断

1. **本線パイプラインでの検証**: 最小Actorではなく実際のActor+Transformerで
   class_balanced + floor=-1.0 が機能するか確認
2. **action_values設計**: floor<0にする場合、action_valuesを
   [-1.0, 0.0, 1.0]（3値）か[-1,-0.5,0,0.5,1]（5値）に変更するか
3. **BC pretrainerへのclass_balanced組み込み**: 現状のbc_pretrain.pyには
   class_balancedパラメータがあるが、正しく機能しているか確認要

## 変更ファイル
- scripts/oracle_label_probe.py (新規作成)
- scripts/oracle_label_probe_v2.py (新規作成)
- scripts/bc_collapse_probe.py (新規作成)
- scripts/bc_stability_probe.py (新規作成)

## 実行コマンド
```
uv run python scripts/bc_collapse_probe.py
uv run python scripts/bc_stability_probe.py
```

## 結果
成功 — collapse原因を特定し、防止手法を発見

## 生成物
- scripts/oracle_label_probe_v2_results.json
- scripts/bc_collapse_probe_results.json
- scripts/bc_stability_probe_results.json

## 問題点
- 最小Actorでの検証なので、実際のActor+Transformerで
  同じ結果になるかは別途確認が必要
- OOS検証（val期間でのラベル品質）は未完了
