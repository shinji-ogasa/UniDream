# Plan004 Production Adoption Report (2026-05-06)

## 結論

Plan004 residual BC/AC は no-leak 条件で本流候補に昇格。主指標では全14foldで `AlphaEx > +1` かつ `MaxDDDelta <= +1` を達成した。

Plan004 は本流 `uv run python -m unidream.cli.train --device cuda` に統合済み。各foldで WM/BC/AC checkpoint を作った後、train/valのみで Plan004 residual BC/AC をfreshにfit/extractし、test評価へ渡す。

HF Spaces 推論repoには Plan004 完全bundleを移行済み。UniDream本体APIには接続せず、`unidream-space/bundles/current` の成果物だけで推論する。

## 採用ロジック

- 単一actor圧縮は不採用。
- 階層base policyを固定し、residualのみを学習する。
- residual BCは realized candidate advantage を教師にする。
- AC相当の制御は neural full actor unlock ではなく、validation-only の threshold / hold / cooldown extraction として制限する。
- source/spec選択はvalidation情報だけで行い、testは評価専用にした。

## no-leak 条件

| 項目 | 内容 |
|---|---|
| model fit | train split only |
| extraction selection | validation split only |
| test split | 評価とsample verificationのみ |
| checkpoint resume | なし |
| neural actor compression | なし |
| source selection | `multi_source_val` |
| teacher selection | `val_only` |
| selection stress | `primary` |

## 本流統合

標準config:

- `configs/trading.yaml`
- checkpoint dir: `checkpoints/main_plan004_residual_bc_ac_s007`
- Plan004 stage: `plan004_residual_bc_ac.enabled=true`

fold13 scratch実行:

```powershell
uv run python -u -m unidream.cli.train --config configs/trading.yaml --start 2018-01-01 --end 2024-01-01 --folds 13 --seed 7 --device cuda
```

生成物:

- `checkpoints/main_plan004_residual_bc_ac_s007/fold_13/world_model.pt`
- `checkpoints/main_plan004_residual_bc_ac_s007/fold_13/bc_actor.pt`
- `checkpoints/main_plan004_residual_bc_ac_s007/fold_13/ac.pt`
- `checkpoints/main_plan004_residual_bc_ac_s007/fold_13/plan004_policy.npz`
- `checkpoints/main_plan004_residual_bc_ac_s007/fold_13/plan004_summary.json`

fold13 本流評価:

| stress | AlphaEx | SharpeDelta | MaxDDDelta | turnover |
|---|---:|---:|---:|---:|
| cost_x1 | +1.492 | -0.074 | -1.832 | 2.45 |
| cost_x2 | +1.093 | -0.092 | -1.718 | 2.45 |
| cost_x3 | +0.699 | -0.109 | -1.605 | 2.45 |

## 全14fold 結果

出力:

- `codex_outputs/20260506_211338_plan004_current_no_leak_allfold.json`
- `codex_outputs/20260506_211338_plan004_current_no_leak_allfold.md`
- `codex_outputs/20260506_211338_plan004_current_no_leak_allfold.log`

| stress | AlphaEx>0 | AlphaEx>+1/DD<=+1 | eps pass | Alpha mean | Alpha median | Alpha worst | MaxDD worst | turnover max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cost_x1 | 14/14 | 14/14 | 14/14 | +103.582 | +3.721 | +1.171 | +0.000 | 3.50 |
| cost_x1_5 | 14/14 | 14/14 | 13/14 | +99.577 | +2.856 | +1.145 | +0.014 | 3.50 |
| cost_x2 | 14/14 | 14/14 | 12/14 | +95.595 | +2.732 | +1.093 | +0.030 | 3.50 |
| cost_x3 | 13/14 | 11/14 | 12/14 | +87.696 | +2.487 | -2.180 | +0.064 | 3.50 |
| slippage_x2 | 14/14 | 14/14 | 13/14 | +102.123 | +3.399 | +1.162 | +0.003 | 3.50 |

## fold別 selected policy (cost_x1)

| fold | source | spec | AlphaEx | MaxDDDelta | turnover |
|---:|---|---|---:|---:|---:|
| 0 | GR_baseline | bc_resid_twoside_h16 | +7.440 | -0.117 | 1.50 |
| 1 | benchmark | bc_resid_riskoff_h32 | +6.788 | -2.760 | 2.40 |
| 2 | benchmark | bc_resid_wide_riskoff_h16 | +1347.877 | -4.682 | 3.40 |
| 3 | D_risk_sensitive | bc_resid_riskoff_h32 | +40.176 | +0.000 | 2.10 |
| 4 | GR_baseline | bc_resid_twoside_h16 | +2.971 | -5.006 | 3.50 |
| 5 | recovery_rescue_fixed_state | bc_resid_twoside_h32 | +23.327 | -0.003 | 1.60 |
| 6 | D_risk_sensitive | bc_resid_guarded_twoside_h16 | +1.843 | -1.869 | 3.50 |
| 7 | benchmark | bc_resid_riskoff_h32 | +2.990 | -0.203 | 2.40 |
| 8 | D_inactive_residual | bc_resid_wide_riskoff_h16 | +1.171 | -1.241 | 2.80 |
| 9 | benchmark | bc_resid_wide_riskoff_h16 | +2.123 | +0.000 | 0.80 |
| 10 | micro_triple_fixed_raw | bc_resid_riskoff_h32 | +5.961 | -0.890 | 2.40 |
| 11 | GR_baseline | bc_resid_twoside_h16 | +4.451 | -0.002 | 3.50 |
| 12 | GR_baseline | bc_resid_twoside_h16 | +1.536 | -0.294 | 1.20 |
| 13 | D_risk_sensitive | bc_resid_riskoff_h32 | +1.492 | -1.832 | 2.45 |

## cost stress判断

`cost_x2` では全14foldで目標を維持。`cost_x3` は13/14でAlphaEx正、11/14で目標通過。fold11だけ `cost_x3` でAlphaExが負になるため、高コスト運用ではこのfold相当のGR two-sided過信を監査対象に残す。

## Space移行

出力先:

`C:/Users/Sophie/Documents/UniDream/unidream-space/bundles/current`

bundle:

- `manifest.json`
- `model_config.yaml`
- `policy_model.npz`
- `plan004_summary.json`
- `actor_full.pt`
- `checkpoints/world_model.pt`
- `checkpoints/bc_actor.pt`
- `checkpoints/ac.pt`
- `sample_input.npz`
- `sample_output.json`

Space runtime:

- `bundle_type=plan004_residual_bc_ac` の場合、推論はPlan004 residual policyをロードする。
- `policy_model.npz` には fold13 selected の `D_risk_sensitive_tbguard_auto_cd_floor001_pullback_evalonly + bc_resid_riskoff_h32` を再現するbase selector係数、danger gate係数、residual係数、threshold/hold/cooldownを保存した。
- neural checkpointは推論bundle内に同梱し、監査・将来の比較・デバッグに使える。
- `POST /predict` は `features + returns` または `candles` を受け取れる。
- `GET /predict/latest` はWebから最新OHLCV/funding/OI/markを取得し、Space内で特徴量とreturnsを作る。

## Space検証

コマンド:

```powershell
$env:PYTHONPATH='C:\Users\Sophie\Documents\UniDream\unidream-space'
uv run python -m backend.verify_bundle --bundle-dir C:\Users\Sophie\Documents\UniDream\unidream-space\bundles\current
```

結果:

```text
n=8833 max_abs_diff=0.0000000000 mean_abs_diff=0.0000000000
expected_last=0.85000002 actual_last=0.85000002
```

FastAPI sample:

```text
GET /health        200
GET /sample/verify 200 strict_ok=True max_abs_diff=0.0
POST /predict      200 position=0.85000002 signal=underweight
```

## 実装整理

- `unidream/research/plan004_residual_bc_ac.py`: Plan004検証本体。
- `unidream/experiments/plan004_stage.py`: 本流trainからPlan004をfreshにfit/extractし、`plan004_policy.npz` を保存。
- `unidream/deploy/plan004_space_bundle.py`: 本流checkpointからSpace bundleを作成。
- `unidream/cli/plan004_noncompressive_bc_ac_probe.py`: thin CLI wrapper。
- `unidream/cli/export_plan004_space_bundle.py`: thin CLI wrapper。
- `unidream-space/backend/runtime.py`: D-risk base + residual Plan004 bundle loaderを追加。
- `unidream-space/backend/feature_pipeline.py`: candlesからreturnsも返せるように変更。
- `unidream-space/backend/schemas.py`: `returns` 入力を追加。

## 残す監査

productionでは以下を継続監査する。

- validation選択過適合: threshold / hold / cooldown のfold別偏り。
- cost_x3耐性: fold11相当の高コスト崩れ。
- live feature parity: candles入力時のfunding/OI/mark欠損時の挙動。
