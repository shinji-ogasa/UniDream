# Codex Outputs サマリ (2026-05-06 一斉実験)

本ドキュメントは `codex_outputs/` 配下の全出力ファイルを要約したものです。

---

## 1. プローブシリーズ概要

`codex_outputs/` には以下のプローブ系列が格納されています：

| 系列 | 説明 | ファイル数 |
|---|---|---|
| PhaseA | Round2 Selector Audit (purged境界) | 4 |
| Plan003 BC Student | Ridge Student vs Teacher Hierarchy 比較 | 8 |
| Plan003 Teacher Export | Hierarchy Teacher npz 出力 | 4 |
| Plan003 Policy Blend | D/GR/Micro/Recovery 階層選択 | 48 |
| Plan003 Round2 Tightcap | Tightcap 閾値スイープ | 6 |
| Plan003 その他 | Pullback Audit, Overlay Recheck, Recovery | 4 |
| Plan004 | Non-Compressive BC/AC (全14fold + fix series) | 22 |
| その他 | WM Features, Selector TPEN, Logs, YAML, Reviews | 多数 |

---

## 2. PhaseA: Round2 Selector Audit (Purged)

**設定**: `configs/trading.yaml`, folds 0-13, purge/embargo/lookback = 32/128/256

**主要結果 (cost_x1)**:

| Selector | Alpha pass | Alpha median | Alpha worst | turnover max |
|---|---|---|---|---|
| priority_recovery_veto | 7/14 | +0.020 | -1.905 | 6.5 |
| simple_volshock_t8 | 8/14 | +0.026 | -2.634 | 9.0 |
| fixed_recovery_raw | 7/14 | +0.014 | -4.953 | 5.6 |

**コストストレス**: `priority_recovery_veto` は cost_x2 で崩壊（worst -41.686）。slippage_x2 は pass。

**リーク監査**: `time_shift` で pass が増加 → selector の時間的過学習リスクあり。

**fold2 が外れ値**: fold2 AlphaEx +117.73（vol_shock/recovery/state）。

---

## 3. Plan003 BC Student Probe

BC Ridge Student の全14fold評価。teacher_hierarchy をベースラインとして、ridge 回帰 student がどの程度再現できるかを検証。

**Teacher Hierarchy (ベースライン)**:
- Alpha mean: +4.881, median: +0.443, pass: 12/14
- Cost_x2 でも pass 11/14 → 頑健

**BC Ridge Student 4 variant 比較 (cost_x1)**:

| Variant | pass | Alpha mean | Alpha median | Alpha worst | turnover max |
|---|---|---|---|---|---|
| raw | 6 | +0.224 | 0.000 | -11.552 | 2.515 |
| quantile | 8 | +4.163 | +0.065 | -0.868 | 4.650 |
| hgb_val_quant | 1 | +3.680 | 0.000 | -0.291 | 3.500 |
| trainval_quant | 4 | +0.582 | 0.000 | -2.831 | 7.700 |

quantile 版が最も安定（pass 8/14, worst -0.868）。ただし teacher には遠く及ばず。

---

## 4. Plan003 Teacher Export

全14fold の Teacher bundle (.npz) を生成。

**通常 bundle**:
- Alpha mean: +4.881, pass: 12/14, worst: 0.000
- 選択 source: D_risk_sensitive (主要), GR_baseline, micro_triple_fixed_raw, recovery_rescue_fixed_state

**TurnGuard25 bundle**:
- Alpha mean: +4.833, pass: 12/14
- fold10 が D から micro_triple_fixed_raw に変更（turnguard 制限による fallback）

---

## 5. Plan003 Policy Blend Probe

D safe overlay をベースに、validation に応じて GR/micro/rescue に fallback する階層的ポリシー選択。

### 5.1 Base Blend (D + GR fallback)

| variant | pass | Alpha mean | Alpha median | Alpha worst | MaxDD worst |
|---|---|---|---|---|---|
| base | 9/14 | +3.806 | +0.295 | 0.000 | 0.000 |
| safety_fullfit (seed7) | 9/14 | +2.849 | +0.367 | 0.000 | 0.000 |
| d_auto | 11/14 | +4.785 | +0.422 | -0.272 | 0.055 |
| d_cd32 | 9/14 | +2.993 | +0.264 | -1.905 | 0.055 |

### 5.2 Rescue Chain 追加

Micro triple rescue + recovery rescue を階層に追加すると大幅改善：

| variant | pass | Alpha mean | Alpha median | Alpha worst | turnover max |
|---|---|---|---|---|---|
| micro_auc050 (seed7) | **12/14** | +4.881 | +0.443 | 0.000 | 2.700 |
| micro_fullfit (seed7) | **13/14** | +4.883 | +0.443 | 0.000 | 2.700 |
| turnguard25 (seed7) | **12/14** | +4.833 | +0.422 | 0.000 | 2.500 |
| micro_fullfit (seed11) | **13/14** | +4.883 | +0.443 | 0.000 | 2.700 |
| micro_fullfit (seed21) | **13/14** | +4.883 | +0.443 | 0.000 | 2.700 |

### 5.3 Seed 比較

turnguard25 は seed7/11/21 で同一結果（決定論的）。fullfit 系も seed 間で一致。

### 5.4 問題点

- D safe 単体は fold2 で崩壊（AlphaEx -166.04）→ safety fallback なしでは危険
- blend fullfit seed7 で collapsed fold2 が通過（D安全制限に引っかからず）→ -166.04 が混入
- 最良構成は **micro_fullfit rescue chain + turnover guard**（13/14 pass）

---

## 6. Plan004 Non-Compressive BC/AC Probe

**アイデア**: Hierarchy base policy を固定したまま、realized residual-advantage を学習。BC で残差モデルを訓練し、AC スタイルで threshold/hold/cooldown を validation 選択。

### 6.1 Fix Series (修正の軌跡)

| variant | folds | selected pass | Alpha mean | Alpha median | Alpha worst | MaxDD worst |
|---|---|---|---|---|---|---|
| smoke (f045) | 3 | 3/3 | +3.060 | +0.420 | +0.401 | 0.000 |
| fix1_weakfolds | 8 | 6/8 | +3.324 | +0.671 | -2.057 | 0.094 |
| fix2_weakfolds | 8 | **8/8** | +4.161 | +1.514 | +0.346 | 0.000 |
| fix3_allfold | 14 | **13/14** | +16.076 | +2.502 | -6.975 | 0.211 |
| fix4_allfold | 14 | **14/14** | +16.958 | +2.697 | +0.412 | 0.211 |

fix1→fix2 で weakfolds の pass 率が 6→8 に向上。fix4 では全14fold で pass（Alpha worst +0.412）。

### 6.2 Smoke 系列の進行

smoke (f045) → smoke2 → smoke3 → smoke4 → smoke5（全て fold0,4,5）で段階的に改善されたが、数値は smoke のみ記録（他は同条件の反復テストの可能性）。

### 6.3 問題点

- residual_bc_ac 全体では fold2 などで **-4000 以上の暴落**（twoside_h32 等）
- selected で救われるが、選択ロジックに依存
- fix4 では `selected_residual_bc_ac` が Alpha worst +0.412 で **全 fold pass** → 有望

---

## 7. Plan004 補助プローブ

### 7.1 Grid (folds689)

閾値グリッド探索。fix 適用前の状態。

### 7.2 Guarded Check (folds2691013, folds78)

twoside に risk-off guard を追加した `guarded_twoside` の検証。fix3/fix4 で採用された `bc_resid_guarded_twoside_h16` の原型。

### 7.3 AllowWide D Bench (folds67813)

D 安全ポリシーに allow_wide フラグを追加したベンチマーク。

---

## 8. WM Features / Teacher Audit (fold8)

### WM Features
fold8 の WM feature extraction 結果（JSON + log）。

### Teacher Audit
fold8 の teacher ラベル監査（JSON + log）。教師ラベルの品質確認。

---

## 9. Round2 Tightcap (Plan003)

Tightcap 閾値のスイープ（0.005, 0.001, 0.002）：

| variant | folds | 内容 |
|---|---|---|
| tightcap0005_allfold | 0-13 | 閾値 0.005 全fold |
| tightcap0005_audit | 0-13 | 同上 + 監査 |
| tightcap0005_folds2789 | 2,7,8,9 | subset |
| tightcap001_folds2789 | 同上 | 閾値 0.001 |
| tightcap002_folds2789 | 同上 | 閾値 0.002 |

---

## 10. その他ファイル

### レビュー
- `openclaude_noncompressive_bc_ac_prompt.txt + review.json`: Plan004 設計の外部レビュー
- `openclaude_plan004_fix_prompt.txt + review.json`: Plan004 fix の外部レビュー

### YAML 設定
- `plan003_hierarchy_bc_*.yaml` (7個): Plan003 BC 実験設定
- `plan003_hierarchy_direct_policy_eval.yaml`: ダイレクトポリシー評価設定

### ログファイル (選択)
- `*_bc_cuda.log`: BC training + CUDA logs
- `*_test_cuda.log`: Test evaluation + CUDA logs
- `*_test.log`: Test evaluation (CPU) logs
- `20260506_fold8_plan004_wm_features.log`: WM feature extraction log
- `20260506_fold8_teacher_audit.log`: Teacher audit log

### Checkpoint ディレクトリ
- `plan003_hierarchy_bc_*_ckpt/` (6個): BC チェックポイント
- `plan003_hierarchy_teacher_bundle/` (14個): Teacher bundle .npz
- `plan003_hierarchy_teacher_bundle_turnguard25/` (14個): TurnGuard25 bundle .npz

### Selector TPEN
- `20260506_selector_tpen_smoke_fold10.json + .md`: Selector TPEN スモークテスト

---

## 11. 総括

### 達成されたこと

| 成果 | 詳細 |
|---|---|
| Policy Blend 13/14 fold pass | micro_fullfit rescue chain が最良 |
| Plan004 Fix4 全14fold pass | Selected residual BC/AC が Alpha worst +0.412 |
| Teacher Export 完了 | 全14fold の teacher bundle 生成（2 variant） |
| BC Student quant 版改善 | pass 8/14 まで向上 |
| Non-compressive BC/AC パイプライン確立 | ニューラル actor 圧縮不要の残差学習 |

### 未解決の課題

| 課題 | 該当 |
|---|---|
| D safe 単体の崩壊リスク | fold2 で -166 AlphaEx |
| BC Ridge Student の pass 率 | 最大でも 8/14 |
| Plan004 residual の暴落 | twoside で -4515 |
| fold7,8 の不活性 | Teacher / D 共に benchmark に fallback |
| Shuffle/time_shift 監査の弱い偽陽性 | 時間的過学習の可能性 |
