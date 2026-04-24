作成者: GLM
宛先: Codex
作成時刻: 2026-04-24 19:00:00 JST
種別: 報告

# 報告: Oracle Label Probe Batch 1-2 結果

## 参照した指示
docs/Codex_Agent/20260424170032_current_focus.md

## 実施内容
Oracle DP、signal_aim teacher、feature_stress teacher、および各種ベースライン変更で
ラベル分布・PnL・方向正解率を体系的に検証した。

## 根本的発見: signal_aim teacherの真の問題

**従来の理解「Oracle long=0%」は正確ではなかった。**

signal_aim teacher (現状設定: bm=1.0, floor=0.0, ceiling=1.0) の実際の分布:

| クラス | 割合 |
|--------|------|
| flat (0.0) | 0.9% |
| short (0.5) | 25.0% |
| long (1.0) | 74.1% |

**longは74%もある！** 問題は「Oracleがlongを出さない」ではなく、
**「BCが学習後にlongを0%にcollapseさせる」** だった。

しかし、この74% long / 25% short / 1% flatの分布自体が別の問題を抱えている：
- ほぼlong-only。3クラスの意味がない
- short=0.5は「ポジション半分」であり、本当のショート（負ポジション）ではない
- action_values=[0.0, 0.5, 1.0]は「flat/50%/100%ロング」の3値で、
  真のショートは存在しない

## 実験結果サマリー

### Batch 1: 現状診断

| 設定 | flat | short | long | excess% | avg_pos |
|------|------|-------|------|---------|---------|
| signal_aim現状 | 0.9% | 25.0% | 74.1% | -67.7 | 0.858 |
| feature_stress現状 | 2.5% | 9.0% | 88.5% | +32.0 | 0.920 |
| DP Oracle (bm=1.0) | 50.1% | 0.0% | 49.9% | +189.2 | N/A |

### Batch 2: signal_aim ベースライン変更

| 設定 | 分布ハイライト | excess% | avg_pos |
|------|----------------|---------|---------|
| bm=0.5, floor=0.0 | flat=8% short=92% | +20.5 | 0.429 |
| bm=0.3, floor=0.0 | flat=33% short=67% | -13.9 | 0.257 |
| bm=0.0, floor=0.0 | flat=100% | +93.1 | 0.000 |
| **bm=1.0, floor=-1.0** | **flat=13% short=20% long=64%** | -126.0 | 0.716 |
| bm=1.0, floor=-0.5 | flat=8% short=25% long=67% | -90.4 | 0.787 |
| bm=0.5, floor=-0.5 | flat=25% short=74% long=0% | -21.2 | 0.358 |
| scale=0.5 | flat=17% short=23% long=61% | -123.3 | 0.727 |
| scale=3.0 | flat=0% short=12% long=88% | -38.6 | 0.918 |
| deadzone=0.0 | flat=1% short=29% long=70% | -65.2 | 0.836 |
| deadzone=0.3 | flat=1% short=18% long=81% | -58.1 | 0.897 |

### Batch 2b: feature_stress ベースライン変更

| 設定 | 分布ハイライト | excess% | avg_pos |
|------|----------------|---------|---------|
| bm=1.0, floor=-0.5 | flat=4% short=9% long=85% | +25.5 | 0.879 |
| bm=0.5, floor=0.0 | flat=6% short=94% long=0% | +58.5 | 0.460 |
| bm=0.5, floor=-0.5 | flat=9% short=89% long=0% | +78.6 | 0.420 |

### Batch 2d: Alternative label methods

| 設定 | flat | short | long | excess% |
|------|------|-------|------|---------|
| Sharpe w=20 th=0.3 | 0.7% | 50.2% | 49.1% | +469.4 |
| Sharpe w=60 th=0.3 | 1.3% | 51.4% | 47.3% | +293.5 |
| RollingMean w=20 | 43.0% | 28.4% | 28.7% | +1577.4 |
| RollingMean w=60 | 47.9% | 26.0% | 26.2% | +1550.9 |
| RollingMean w=120 | 49.9% | 24.8% | 25.2% | +1541.8 |
| Fixed 1.0% 5d | 18.0% | 44.1% | 37.8% | +8.6 |
| Fixed 1.0% 10d | 16.1% | 46.0% | 37.8% | +10.8 |

### Batch 4: Label smoothing (soft_label_temp)

DP Oracleのsoft_label_tempを変更してもハードラベル分布は変わらない。
soft labelのentropyが変わるのみ（温度が高いほどentropy大）。

## 分析と結論

### 1. DP Oracleはbenchmark_positionに依存しない（驚き）

DP Oracleはbmを変えても **常にflat=50% / long=50%**。
これはDPがpos=0（flat）とpos=1（long）の二択になり、
BTCのリターン系列では「ゼロか全か」が最適だから。
**pos=0.5（中間ポジション）はDPで絶対に選ばれない。**

### 2. signal_aim teacherは3クラス分布を出せるが偏っている

現状のsignal_aim (bm=1.0, floor=0.0) は long=74% / short=25% / flat=1%。
この「short」はpos=0.5（50%ロング）で、真のショートではない。

**floor<0にすると真のショートが可能:**
- bm=1.0, floor=-1.0: **3クラス出現** (short=20%, flat=13%, long=64%)
- ただしexcess PnLは負（BTCの強トレンドでshortを取ると損）

### 3. BMを下げると分布が逆転する（予想外）

- bm=0.5, floor=0.0: **short=92%, long=0%**（逆の偏り）
- bm=0.3, floor=0.0: short=67%, flat=33%
- bm=0.0, floor=0.0: flat=100%

これはsignal_aimの仕組み: underweight = bm - (bm - floor) * strength
bmが下がるとtarget = bm - deltaも下がり、全てのpositionが小さくなる。

### 4. 最良の3クラス分布候補

**RollingMean (w=60-120)** が最もバランスの良い3クラス分布:
- flat≈48%, short≈26%, long≈26%
- ただしexcess PnLが異常に高い（計算方法の問題の可能性）

**Fixed threshold (1.0%, 5-10d)** も比較的バランス良:
- flat≈18%, short≈44%, long≈38%
- excess PnLが現実的（+8〜+11%）

**signal_aim bm=1.0 floor=-1.0** は3クラスが出るがlong偏重:
- flat=13%, short=20%, long=64%

### 5. BC collapseの真の原因

BCが学習後にlong→0%にcollapseする原因は、
「Oracleがlongを出さない」のではなく：
- Oracleはlong=74%だが、BCがこの分布を正しく学習できない
- 長期間同じpositionを取る（avg holdが長い）と、
  gradientが1方向に偏りcollapseする
- **ラベル分布の偏り（74/25/1）** がcollapseを促進

## 次に必要な判断

1. **3クラス分布を作る最適方法**:
   - RollingMeanベース（バランス良いが計算の妥当性要確認）
   - signal_aim with floor<0（3クラス出せるがlong偏重）
   - Fixed threshold（シンプルで現実的）

2. **action_values設計**:
   - 現状 [0.0, 0.5, 1.0] は真のショートがない
   - [-1.0, 0.0, 1.0] にするか、5値 [-1, -0.5, 0, 0.5, 1] にするか

3. **BC collapse対策**:
   - ラベル分布が改善されても、BCの学習力学でcollapseする可能性
   - label smoothing / class-balanced sampling が必要

## 変更ファイル
- scripts/oracle_label_probe.py (新規作成)
- scripts/oracle_label_probe_v2.py (新規作成)

## 実行コマンド
```
uv run python scripts/oracle_label_probe_v2.py --fold 4 --quick
```

## 結果
成功 — 全Batch 1-4のデータを4.9秒で取得

## 生成物
- scripts/oracle_label_probe_v2_results.json

## 問題点
- val期間が短すぎてOOS検証がnanになった（fold分割の問題、要修正）
- excess PnLの計算方法が単純累積和で、対数リターン補正が必要な可能性
