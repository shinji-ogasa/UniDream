# 因果 ML/RL の採用判断用・既存 evidence 監査（2026-09-06）

新規 fit・予測・損益経路は計算していない。完了済み成果物の保存済み集計値とモデル所在を確認した。Stage20 は未完了として含めない。Oracle/hindsight/将来符号・値の置換は候補から除外した。旧 P1 と過去の選択 lock は変更しない。

## 採用判断に直結する結論

既存の因果 ML には、現在の最低条件「validation の四半期平均 AlphaEX > 0、MaxDDdelta < 0」を通常・費用2倍の両方で満たす候補がある。主な妥協候補は Perp reliability / hold、Technical reliability / hold、既に固定した Perp half / hold。いずれも観測した開始時トレンド3分類の平均符号を満たす。これは繰り返し見た validation の記述値で、高確率保証ではない。

**既存の学習済み RL では、同じ最低条件を満たすことと配布可能な checkpoint の両方を確認できない。** ML を採用して RL も改善済みと表示することはできない。現在の BNB trend bundle は固定ルールであり、ML/RL 学習成果ではない。

Perp reliability は全体 Alpha が最大の候補だが、half より DD 改善幅が小さく、四半期単位の両条件達成は3/8対4/8。Technical reliability は Spot 特徴だけの平均・分散モデルを使える実装上の利点がある。ただし研究の共通支持集合には UM を含む依存群があり、配布時に Spot-only の別マスクへ無断で広げて「同条件」としてはいけない。

## 同一 validation の最低条件通過候補

すべて BTCUSDT15m、validation5–12 = [2021-04-16 13:45Z, 2023-04-16 13:45Z)、8四半期を等重み。単位は pt、年率・連結曲線ではない。base fee0.00055/借入0.10、stress は同じ target を fee/借入2倍で再生。開始時分類 bull2/bear4/sideways2。hold は予測なしで保有継続、fallback は既知現在始値で target1。

| 因果候補ID | base Alpha / DD | stress Alpha / DD | 3分類平均も両費用で通過 | 同一四半期で両符号・両費用 |
|---|---:|---:|---|---:|
| `perp_delay0_reliability_utility_risk1` | +4.450249 / -5.143249 | +4.247804 / -5.055918 | yes | 3/8 |
| `perp_delay0_reliability_utility_risk1_fallback_bh` | +4.207554 / -5.854364 | +4.039490 / -5.779119 | yes | 3/8 |
| `technical_reliability_utility_risk1` | +3.750066 / -5.695064 | +3.530854 / -5.640501 | yes | 3/8 |
| `technical_reliability_utility_risk1_fallback_bh` | +3.704393 / -6.378202 | +3.545658 / -6.311185 | yes | 3/8 |
| `perp_delay0_half_utility_risk1` | +3.612942 / -5.387702 | +3.348978 / -5.304506 | yes | 4/8 |
| `perp_delay0_half_utility_risk1_fallback_bh` | +3.379453 / -6.159801 | +3.143856 / -6.090210 | no | 4/8 |
| `technical_half_utility_risk1` | +1.148958 / -6.031374 | +0.943271 / -5.953139 | no | 3/8 |
| `technical_half_utility_risk1_fallback_bh` | +1.006518 / -6.782863 | +0.821266 / -6.718393 | no | 3/8 |
| `perp_delay0_magnitude_direction_utility_risk1` | +3.335463 / -3.113130 | +2.811854 / -3.007415 | yes | 5/8 |
| `perp_delay0_magnitude_direction_utility_risk1_fallback_bh` | +1.687686 / -2.528492 | +1.198274 / -2.430444 | yes | 5/8 |
| `perp_delay0_ordinary_direction_utility_risk1` | +3.086659 / -2.788922 | +2.566077 / -2.672388 | yes | 5/8 |
| `perp_delay0_ordinary_direction_utility_risk1_fallback_bh` | +2.080828 / -2.778153 | +1.585923 / -2.664776 | yes | 5/8 |
| `perp_delay0_scaled_utility_risk1` | +2.130806 / -3.129558 | +1.722574 / -3.000522 | yes | 4/8 |
| `perp_delay0_scaled_utility_risk1_fallback_bh` | +1.917732 / -3.776697 | +1.546061 / -3.663649 | no | 4/8 |
| `technical_scaled_utility_risk1` | +0.595094 / -4.322763 | +0.276755 / -4.210764 | no | 4/8 |
| `technical_scaled_utility_risk1_fallback_bh` | +0.705421 / -5.079134 | +0.409777 / -4.981111 | no | 4/8 |
| `scale_mean_utility_risk1` | +3.128084 / -6.531401 | +3.061185 / -6.494211 | no | 3/8 |
| `scale_mean_utility_risk1_fallback_bh` | +3.541396 / -6.716185 | +3.486235 / -6.688096 | no | 3/8 |

`scale_mean` は校正期間の平均だけを使う対照で、学習特徴の予測改善とは区別する。上表は完了済み Stage19 と Stage15 の保持ファミリーの和集合にある通過18候補であり、全歴史探索の全候補を再採点した順位表ではない。18候補から test を使って選び直していない。

## 主要候補の開始時トレンド別成績

| 候補 | 分類 | base Alpha / DD | stress Alpha / DD |
|---|---|---:|---:|
| `perp_delay0_reliability_utility_risk1` | bull | +4.062991 / -4.749015 | +3.776810 / -4.547813 |
| `perp_delay0_reliability_utility_risk1` | bear | +2.894529 / -3.966984 | +2.659914 / -3.904311 |
| `perp_delay0_reliability_utility_risk1` | sideways | +7.948947 / -7.890012 | +7.894579 / -7.867240 |
| `perp_delay0_reliability_utility_risk1_fallback_bh` | bull | +6.020865 / -6.681983 | +5.812628 / -6.542008 |
| `perp_delay0_reliability_utility_risk1_fallback_bh` | bear | +0.921381 / -4.228697 | +0.732601 / -4.170765 |
| `perp_delay0_reliability_utility_risk1_fallback_bh` | sideways | +8.966589 / -8.278081 | +8.880131 / -8.232935 |
| `technical_reliability_utility_risk1` | bull | +3.773439 / -5.352392 | +3.523008 / -5.174817 |
| `technical_reliability_utility_risk1` | bear | +2.221394 / -3.920687 | +1.936235 / -3.909537 |
| `technical_reliability_utility_risk1` | sideways | +6.784037 / -9.586491 | +6.727939 / -9.568115 |
| `technical_reliability_utility_risk1_fallback_bh` | bull | +5.762933 / -7.310367 | +5.590344 / -7.193596 |
| `technical_reliability_utility_risk1_fallback_bh` | bear | +0.446757 / -4.220500 | +0.260384 / -4.164258 |
| `technical_reliability_utility_risk1_fallback_bh` | sideways | +8.161124 / -9.761442 | +8.071521 / -9.722629 |
| `perp_delay0_half_utility_risk1` | bull | +4.221752 / -6.472994 | +4.038359 / -6.350385 |
| `perp_delay0_half_utility_risk1` | bear | +1.724013 / -2.951472 | +1.428680 / -2.882012 |
| `perp_delay0_half_utility_risk1` | sideways | +6.781988 / -9.174870 | +6.500194 / -9.103617 |

開始時の過去90日 momentum/volatility による分類で、四半期の実現方向ではない。各分類最低3期という元の確認 gate は2/4/2のため未達。Perp reliability の MSE は元予測比3.990%改善、half比0.304%改善だが scale 平均より1.236%悪い。Technical reliability は元予測比5.162%改善、half比1.063%改善だが scale 平均より0.809%悪い。経済条件の妥協採用と、リターン予測能力の確立は別。

## 追加 test は report-only として分離

固定済み half ファミリーだけを、再利用された original test15–24 [2024-01-16 13:45Z, 2026-07-16 13:45Z) の10四半期で報告した。独立 holdout ではない。**Stage13 reliability と Stage17 direction の追加 test 成績はこの実験には存在しない。half の追加 test を reliability の裏付けへ流用しない。**

| 固定候補 | base Alpha / DD (pt) | stress Alpha / DD (pt) |
|---|---:|---:|
| `technical_half_utility_risk1` | +1.336426 / -2.992183 | +1.162111 / -2.970762 |
| `technical_half_utility_risk1_fallback_bh` | +1.542313 / -3.015473 | +1.371206 / -2.995026 |
| `perp_delay0_half_utility_risk1` | +0.047672 / -2.825694 | -0.129695 / -2.804205 |
| `perp_delay0_half_utility_risk1_fallback_bh` | +0.866052 / -2.852600 | +0.680330 / -2.830673 |

Perp half / hold は追加 test で通常平均 Alpha +0.048pt と僅少、stress Alpha −0.130pt。全4候補は bull 平均 Alpha 負、sideways 平均 DD差正で、追加 test の全トレンド条件は未達。これを隠さず Web に validation と分けて表示する必要がある。追加 test の良さで Technical half や fallback へ選び直す判断は本監査では行わない。

## RL の確認結果

| 候補 | 確認できた期間・契約 | Alpha / DD | 配布・採用条件 |
|---|---|---:|---|
| Plan011 v31 WM→BC→AC | historical development config の outer fold0–12。現行 val5–12 と同一比較ではない | +0.41 / +0.20pt | DD条件未達。旧訓練checkpointは削除済みだが HF repo に fold23 inference bundle は存在 |
| Risk-aware PPO full | validation-selected report-only test15–23、15m target-position common-action | −2.12 / −4.37pt | Alpha条件未達。checkpoint・実験コード・config非保持 |
| PPO risk-only | 同じ report-only test15–23 | −1.74 / −4.86pt | Alpha条件未達。新fitなしのexport不可 |
| P1 WM700 / BC8 / AC300 diagnostic | 別の4-bar additive-log-return契約 | BC rolling alpha −0.03757496、AC 0 | DDの同条件資格を主張できず、ACは全窓fill0。正式P1 promotion不可 |

上記 RL に stress2×と現在の開始時3分類で最低条件を満たす証拠は確認していない。原本: `docs/plan011_v31_investor_evidence.md`, `docs/figures/plan011_v31_folds0_12/summary.json`, `docs/risk_aware_rl_unidream_holdout_2026.md`, `docs/experiments/p1_formal_forecast_wm_bc_ac_20260905.md`。

## 実 artifact と export 境界

学習済み ML の配布候補は実 joblib と校正値が存在するため、**新しい predictor/export adapter を作れば再学習なしの凍結スナップショット配布は可能**。既存 `unidream.cli.export_inference_bundle` と HF `backend/runtime.py` は Plan011 用であり、Ridge/HGB/reliability をそのまま読み込めない。HF の BNB `/v2` 系も trend rule の別契約。既存ファイル名だけ差し替える操作では同条件にならない。

| 種類 | 実モデル/校正/トレース | 必要な推論処理 |
|---|---|---|
| Half/full ML | `codex_outputs/oracle_frozen_procedure_parity_v1/models/fold{5..12}_{technical_mean,perp_delay0_mean,technical_variance}.joblib`; `calibration/fold{f}_provenance.json` | StandardScaler+Ridge100 mean、Technical HGB100 variance、保存bias/variance_multiplier、保存scale_mean、halfなら固定0.5混合 |
| Reliability ML | 上記モデル + `codex_outputs/oracle_mean_reliability_decisions_v1/weights/fold{f}_{technical,perp_delay0}.json` | `p=raw+saved_bias`, `mu=w*p+(1-w)*scale_mean`; exact0/1 branches、sharedtechnical_scaledvariance |
| C1 direction | `codex_outputs/oracle_direction_decisions_v1/models/fold{f}_{technical,perp_delay0}_{ordinary,magnitude}.joblib` + half mean/risk | `sign(logit)*abs(own_half_mu)`。これは soft-mapping Stage19 と別候補 |
| より新しい half/full の凍結モデル | `codex_outputs/oracle_additional_window_replay_v1/models/fold24_{technical_mean,perp_delay0_mean,technical_variance}.joblib`; `calibration/fold24_provenance.json` | 元から固定した same procedure の最新評価用snapshot。選択法を変えずartifactの時系列位置を記録する。reliability用の新しいwは存在しない |

**時点の罠:** validationの最終fold12用モデルは2023-01-16評価開始で、T=[2021-01-16,2022-07-16)、S=[2022-07-16,2022-10-16)、I=[2022-10-16,2023-01-16)。Perp reliabilityのfold12保存wは厳密に0、従ってそのsnapshotの平均予測は定数scale_meanへ一致する。Technicalは0.09332864097215063。平均成績が良いから別foldのwだけを移植してはいけない。現時点用に推奨するなら、新しいproduction cutoffを先に固定し同じ18/3/3手続きの一回fitとparityが必要（本監査は実行していない）。

追加test最終fold24用モデルのT=[2024-04-16,2025-10-16)、S=[2025-10-16,2026-01-16)、I=[2026-01-16,2026-04-16)、E=[2026-04-16,2026-07-16)。2026-09-06現在の新しいfitとは表示しない。

## 固定すべき production feature / action contract

- Canonical source: `alpha_dd_features.py`16列 + `oracle_frontier_features.py`13列 = Technical29。Perp31は `oracle_derivative_features.py` の UM weighted flow24/96だけを追加。37列のStage20は別候補。
- Spot/UMとも BTCUSDT15m raw OHLC、quotevolume、taker-buy quote、ntrades等。完全15mグリッド、NaNを保持、完成済みt−1までから一度だけshift。90日窓を含むため少なくとも8641本程度と全依存のwarmupを実装で確認。既存17列/64step Plan011またはclose-only BNB契約へのゼロ埋め変換は禁止。
- Fit18月、scale3月、interval3月、評価3月。h24 return=`log(close[t+24]/open[t+1])`、ラベル成熟375分。モデルfitとbias/variance/weight校正は境界以前。`oracle_frozen_forecasts.fit_frozen_forecasts` と `oracle_mean_reliability.fit_reliability` が実手続き。
- `oracle_frozen_procedure_parity.py` runがfull mean/risk復元; `oracle_mean_reliability_decisions.py` runが保存raw/biasからwを推定。CLIはそれぞれ `python -m unidream.experiments.<module> --config configs/<same>_20260906.yaml`。完了済みoutputを再実行しない。新production runは別namespace/config/registration。
- 売買候補は6h UTC決定、next15m open fill。utilityは自身のcash/units/NAVに基づき、risk1/costbudget2、maxstep0.08、deadband0.01、intent[0.5,1.12]。missing current openはhold、missing forecast ruleは選んだ候補に固定。無状態の毎回position1入力では検証済み経済経路にならない。
- shared inference2,586と score2,574を区別し、future label availabilityで注文を消さない。研究common maskは過去archiveの比較用supportで、receipt-time証明ではない。ライブでは観測可能なavailabilityのみで実装し、差分を明示してfixture/accounting parityする。
- Webの見出し・モデル名・銘柄・rawfeature契約・最終学習日・validation期間・追加test期間・費用を同じmanifestから読む。平均DD差を連結資産曲線のDD改善と混同しない。無条件の「全トレンド保証」「RL改善済み」は表示できない。

## 元ファイルの SHA256

| ファイル | SHA256 |
|---|---|
| `codex_outputs/oracle_soft_direction_decisions_v1/results.json` | `d85e14e1a8249601a1f28c0d4fa29b1fbb23a3571232dcedc5edee0644b82cc5` |
| `codex_outputs/oracle_short_feature_decisions_v1/results.json` | `2f65382455e1266e035d38007a8dc939efb74d84df676c44da8ed3de9cd77218` |
| `codex_outputs/oracle_mean_reliability_decisions_v1/results.json` | `333f88d4bc06f671d552d8ca70470ee60ecd67074812f6aa248b26f1b94562f1` |
| `codex_outputs/oracle_mean_shrinkage_decisions_v1/results.json` | `a06c1ed0d6b85eb4d808c2741dca9cfd19b9d1f79023043ade3fbbb5897d212e` |
| `codex_outputs/oracle_additional_window_replay_v1/results.json` | `8579e40b5be9ed737acf6c633c92d33b383d0fb338c75832f7b3a125bdd474d1` |
| `codex_outputs/oracle_frozen_procedure_parity_v1/models/fold12_perp_delay0_mean.joblib` | `52debb1ef0ecda2f6815713b4a31fbdbe4d3e5e02037f88170d87b53301ec2a3` |
| `codex_outputs/oracle_frozen_procedure_parity_v1/models/fold12_technical_mean.joblib` | `456570b9f0bc81b4543e36b623d89914fbaa5303a32e89d502668f43eb30538b` |
| `codex_outputs/oracle_frozen_procedure_parity_v1/models/fold12_technical_variance.joblib` | `51a3246272be20d3e03894fd08a719b29a1844f41cdc552a04a5dda1dc0c282d` |
| `codex_outputs/oracle_frozen_procedure_parity_v1/calibration/fold12_provenance.json` | `2dec741db59c32f8b02f8e17b9761a95995c7ddd93a21136b3d3f15e0ec80899` |
| `codex_outputs/oracle_mean_reliability_decisions_v1/weights/fold12_perp_delay0.json` | `24db8245ed33294778155505c8c3d819d38403b1e784fd96aa76ccb9dcc6ac7a` |
| `codex_outputs/oracle_mean_reliability_decisions_v1/weights/fold12_technical.json` | `a1646ad9a83b0607f1e633f1c0c35be7b9f4cae93ce736edfe57802ff3ac6bf1` |
| `codex_outputs/oracle_additional_window_replay_v1/models/fold24_perp_delay0_mean.joblib` | `dcf39f6d8cafe37a463467300083704723757f8dd02f426e942bc065a1f048e5` |
| `codex_outputs/oracle_additional_window_replay_v1/models/fold24_technical_variance.joblib` | `c116d810e2d2637e4ed62ee17b1598b3b49f22f48802d6ab0e7d89955d8ada2f` |
| `codex_outputs/oracle_additional_window_replay_v1/calibration/fold24_provenance.json` | `c4478ef5c37a59256c3191f576f3b8d3d05c977efe671a928e52b344b03915fc` |

関連 source/報告の検索位置は現在のworkspaceで確認した。過去メモはRL report所在の発見にのみ使い、上のRL条件・数値・checkpoint非保持は現行repo本文へ照合した。メモ参照: MEMORY.md:181–220（rollout01a03504-87f0-7dd1-b9df-ef37cf68f50d）。
