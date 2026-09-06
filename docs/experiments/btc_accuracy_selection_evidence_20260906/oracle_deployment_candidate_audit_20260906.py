"""Read-only candidate comparison of already completed development results."""
from pathlib import Path
import hashlib,json,math
W=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
def read(p):return json.loads((W/p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths={'stage19':'codex_outputs/oracle_soft_direction_decisions_v1/results.json','stage15':'codex_outputs/oracle_short_feature_decisions_v1/results.json','stage13':'codex_outputs/oracle_mean_reliability_decisions_v1/results.json','half':'codex_outputs/oracle_mean_shrinkage_decisions_v1/results.json','additional':'codex_outputs/oracle_additional_window_replay_v1/results.json'}
results={k:read(p) for k,p in paths.items()};entries={}
for stage in ('stage19','stage15'):
 r=results[stage];s=r['summary'];rows={(x['fold'],x['candidate_id']):x for x in r['rows']}
 for cid,v in s['economics']['all'].items():
  if v.get('hindsight_only') or any(x in cid for x in ('oracle','hindsight','beam','sign_only','magnitude_only')):continue
  if all(v[c]['alpha_ex']>0 and v[c]['maxdd_delta']<0 for c in ('base','stress_2x')):
   assert all(not rows[f,cid].get('hindsight_only',False) for f in range(5,13))
   x={'candidate':cid,'source':paths[stage],'base':v['base'],'stress_2x':v['stress_2x'],'regimes':{g:s['economics'][g][cid] for g in ('bull','bear','sideways')},'joint_quarters_both_costs':sum(all(rows[f,cid][c]['alpha_ex']>0 and rows[f,cid][c]['maxdd_delta']<0 for c in ('base','stress_2x')) for f in range(5,13))}
   x['all_start_regime_signs_both_costs']=all(x['regimes'][g][c]['alpha_ex']>0 and x['regimes'][g][c]['maxdd_delta']<0 for g in x['regimes'] for c in ('base','stress_2x'))
   if cid in entries:
    assert entries[cid]['base']==x['base'] and entries[cid]['stress_2x']==x['stress_2x']
   else:entries[cid]=x
assert len(entries)==18
order=['perp_delay0_reliability_utility_risk1','perp_delay0_reliability_utility_risk1_fallback_bh','technical_reliability_utility_risk1','technical_reliability_utility_risk1_fallback_bh','perp_delay0_half_utility_risk1','perp_delay0_half_utility_risk1_fallback_bh','technical_half_utility_risk1','technical_half_utility_risk1_fallback_bh','perp_delay0_magnitude_direction_utility_risk1','perp_delay0_magnitude_direction_utility_risk1_fallback_bh','perp_delay0_ordinary_direction_utility_risk1','perp_delay0_ordinary_direction_utility_risk1_fallback_bh','perp_delay0_scaled_utility_risk1','perp_delay0_scaled_utility_risk1_fallback_bh','technical_scaled_utility_risk1','technical_scaled_utility_risk1_fallback_bh','scale_mean_utility_risk1','scale_mean_utility_risk1_fallback_bh']
assert set(order)==set(entries)
fmt=lambda v:f'{100*v:+.6f}'
lines=['# 因果 ML/RL の採用判断用・既存 evidence 監査（2026-09-06）','',
'新規 fit・予測・損益経路は計算していない。完了済み成果物の保存済み集計値とモデル所在を確認した。Stage20 は未完了として含めない。Oracle/hindsight/将来符号・値の置換は候補から除外した。旧 P1 と過去の選択 lock は変更しない。','',
'## 採用判断に直結する結論','',
'既存の因果 ML には、現在の最低条件「validation の四半期平均 AlphaEX > 0、MaxDDdelta < 0」を通常・費用2倍の両方で満たす候補がある。主な妥協候補は Perp reliability / hold、Technical reliability / hold、既に固定した Perp half / hold。いずれも観測した開始時トレンド3分類の平均符号を満たす。これは繰り返し見た validation の記述値で、高確率保証ではない。','',
'**既存の学習済み RL では、同じ最低条件を満たすことと配布可能な checkpoint の両方を確認できない。** ML を採用して RL も改善済みと表示することはできない。現在の BNB trend bundle は固定ルールであり、ML/RL 学習成果ではない。','',
'Perp reliability は全体 Alpha が最大の候補だが、half より DD 改善幅が小さく、四半期単位の両条件達成は3/8対4/8。Technical reliability は Spot 特徴だけの平均・分散モデルを使える実装上の利点がある。ただし研究の共通支持集合には UM を含む依存群があり、配布時に Spot-only の別マスクへ無断で広げて「同条件」としてはいけない。','',
'## 同一 validation の最低条件通過候補','',
'すべて BTCUSDT15m、validation5–12 = [2021-04-16 13:45Z, 2023-04-16 13:45Z)、8四半期を等重み。単位は pt、年率・連結曲線ではない。base fee0.00055/借入0.10、stress は同じ target を fee/借入2倍で再生。開始時分類 bull2/bear4/sideways2。hold は予測なしで保有継続、fallback は既知現在始値で target1。','',
'| 因果候補ID | base Alpha / DD | stress Alpha / DD | 3分類平均も両費用で通過 | 同一四半期で両符号・両費用 |','|---|---:|---:|---|---:|']
for cid in order:
 e=entries[cid];lines.append(f"| `{cid}` | {fmt(e['base']['alpha_ex'])} / {fmt(e['base']['maxdd_delta'])} | {fmt(e['stress_2x']['alpha_ex'])} / {fmt(e['stress_2x']['maxdd_delta'])} | {'yes' if e['all_start_regime_signs_both_costs'] else 'no'} | {e['joint_quarters_both_costs']}/8 |")
lines+=['','`scale_mean` は校正期間の平均だけを使う対照で、学習特徴の予測改善とは区別する。上表は完了済み Stage19 と Stage15 の保持ファミリーの和集合にある通過18候補であり、全歴史探索の全候補を再採点した順位表ではない。18候補から test を使って選び直していない。','',
'## 主要候補の開始時トレンド別成績','', '| 候補 | 分類 | base Alpha / DD | stress Alpha / DD |','|---|---|---:|---:|']
for cid in order[:5]:
 for g,v in entries[cid]['regimes'].items():lines.append(f"| `{cid}` | {g} | {fmt(v['base']['alpha_ex'])} / {fmt(v['base']['maxdd_delta'])} | {fmt(v['stress_2x']['alpha_ex'])} / {fmt(v['stress_2x']['maxdd_delta'])} |")
lines+=['','開始時の過去90日 momentum/volatility による分類で、四半期の実現方向ではない。各分類最低3期という元の確認 gate は2/4/2のため未達。Perp reliability の MSE は元予測比3.990%改善、half比0.304%改善だが scale 平均より1.236%悪い。Technical reliability は元予測比5.162%改善、half比1.063%改善だが scale 平均より0.809%悪い。経済条件の妥協採用と、リターン予測能力の確立は別。','',
'## 追加 test は report-only として分離','',
'固定済み half ファミリーだけを、再利用された original test15–24 [2024-01-16 13:45Z, 2026-07-16 13:45Z) の10四半期で報告した。独立 holdout ではない。**Stage13 reliability と Stage17 direction の追加 test 成績はこの実験には存在しない。half の追加 test を reliability の裏付けへ流用しない。**','',
'| 固定候補 | base Alpha / DD (pt) | stress Alpha / DD (pt) |','|---|---:|---:|']
for cid in ['technical_half_utility_risk1','technical_half_utility_risk1_fallback_bh','perp_delay0_half_utility_risk1','perp_delay0_half_utility_risk1_fallback_bh']:
 rr=[x for x in results['additional']['rows'] if x['candidate_id']==cid];assert len(rr)==10
 v={c:{k:math.fsum(x[c][k]/10 for x in rr) for k in ('alpha_ex','maxdd_delta')} for c in ('base','stress_2x')}
 lines.append(f"| `{cid}` | {fmt(v['base']['alpha_ex'])} / {fmt(v['base']['maxdd_delta'])} | {fmt(v['stress_2x']['alpha_ex'])} / {fmt(v['stress_2x']['maxdd_delta'])} |")
lines+=['','Perp half / hold は追加 test で通常平均 Alpha +0.048pt と僅少、stress Alpha −0.130pt。全4候補は bull 平均 Alpha 負、sideways 平均 DD差正で、追加 test の全トレンド条件は未達。これを隠さず Web に validation と分けて表示する必要がある。追加 test の良さで Technical half や fallback へ選び直す判断は本監査では行わない。','',
'## RL の確認結果','',
'| 候補 | 確認できた期間・契約 | Alpha / DD | 配布・採用条件 |','|---|---|---:|---|',
'| Plan011 v31 WM→BC→AC | historical development config の outer fold0–12。現行 val5–12 と同一比較ではない | +0.41 / +0.20pt | DD条件未達。旧訓練checkpointは削除済みだが HF repo に fold23 inference bundle は存在 |',
'| Risk-aware PPO full | validation-selected report-only test15–23、15m target-position common-action | −2.12 / −4.37pt | Alpha条件未達。checkpoint・実験コード・config非保持 |',
'| PPO risk-only | 同じ report-only test15–23 | −1.74 / −4.86pt | Alpha条件未達。新fitなしのexport不可 |',
'| P1 WM700 / BC8 / AC300 diagnostic | 別の4-bar additive-log-return契約 | BC rolling alpha −0.03757496、AC 0 | DDの同条件資格を主張できず、ACは全窓fill0。正式P1 promotion不可 |','',
'上記 RL に stress2×と現在の開始時3分類で最低条件を満たす証拠は確認していない。原本: `docs/plan011_v31_investor_evidence.md`, `docs/figures/plan011_v31_folds0_12/summary.json`, `docs/risk_aware_rl_unidream_holdout_2026.md`, `docs/experiments/p1_formal_forecast_wm_bc_ac_20260905.md`。','',
'## 実 artifact と export 境界','',
'学習済み ML の配布候補は実 joblib と校正値が存在するため、**新しい predictor/export adapter を作れば再学習なしの凍結スナップショット配布は可能**。既存 `unidream.cli.export_inference_bundle` と HF `backend/runtime.py` は Plan011 用であり、Ridge/HGB/reliability をそのまま読み込めない。HF の BNB `/v2` 系も trend rule の別契約。既存ファイル名だけ差し替える操作では同条件にならない。','',
'| 種類 | 実モデル/校正/トレース | 必要な推論処理 |','|---|---|---|',
'| Half/full ML | `codex_outputs/oracle_frozen_procedure_parity_v1/models/fold{5..12}_{technical_mean,perp_delay0_mean,technical_variance}.joblib`; `calibration/fold{f}_provenance.json` | StandardScaler+Ridge100 mean、Technical HGB100 variance、保存bias/variance_multiplier、保存scale_mean、halfなら固定0.5混合 |',
'| Reliability ML | 上記モデル + `codex_outputs/oracle_mean_reliability_decisions_v1/weights/fold{f}_{technical,perp_delay0}.json` | `p=raw+saved_bias`, `mu=w*p+(1-w)*scale_mean`; exact0/1 branches、sharedtechnical_scaledvariance |',
'| C1 direction | `codex_outputs/oracle_direction_decisions_v1/models/fold{f}_{technical,perp_delay0}_{ordinary,magnitude}.joblib` + half mean/risk | `sign(logit)*abs(own_half_mu)`。これは soft-mapping Stage19 と別候補 |',
'| より新しい half/full の凍結モデル | `codex_outputs/oracle_additional_window_replay_v1/models/fold24_{technical_mean,perp_delay0_mean,technical_variance}.joblib`; `calibration/fold24_provenance.json` | 元から固定した same procedure の最新評価用snapshot。選択法を変えずartifactの時系列位置を記録する。reliability用の新しいwは存在しない |','',
'**時点の罠:** validationの最終fold12用モデルは2023-01-16評価開始で、T=[2021-01-16,2022-07-16)、S=[2022-07-16,2022-10-16)、I=[2022-10-16,2023-01-16)。Perp reliabilityのfold12保存wは厳密に0、従ってそのsnapshotの平均予測は定数scale_meanへ一致する。Technicalは0.09332864097215063。平均成績が良いから別foldのwだけを移植してはいけない。現時点用に推奨するなら、新しいproduction cutoffを先に固定し同じ18/3/3手続きの一回fitとparityが必要（本監査は実行していない）。','',
'追加test最終fold24用モデルのT=[2024-04-16,2025-10-16)、S=[2025-10-16,2026-01-16)、I=[2026-01-16,2026-04-16)、E=[2026-04-16,2026-07-16)。2026-09-06現在の新しいfitとは表示しない。','',
'## 固定すべき production feature / action contract','',
'- Canonical source: `alpha_dd_features.py`16列 + `oracle_frontier_features.py`13列 = Technical29。Perp31は `oracle_derivative_features.py` の UM weighted flow24/96だけを追加。37列のStage20は別候補。',
'- Spot/UMとも BTCUSDT15m raw OHLC、quotevolume、taker-buy quote、ntrades等。完全15mグリッド、NaNを保持、完成済みt−1までから一度だけshift。90日窓を含むため少なくとも8641本程度と全依存のwarmupを実装で確認。既存17列/64step Plan011またはclose-only BNB契約へのゼロ埋め変換は禁止。',
'- Fit18月、scale3月、interval3月、評価3月。h24 return=`log(close[t+24]/open[t+1])`、ラベル成熟375分。モデルfitとbias/variance/weight校正は境界以前。`oracle_frozen_forecasts.fit_frozen_forecasts` と `oracle_mean_reliability.fit_reliability` が実手続き。',
'- `oracle_frozen_procedure_parity.py` runがfull mean/risk復元; `oracle_mean_reliability_decisions.py` runが保存raw/biasからwを推定。CLIはそれぞれ `python -m unidream.experiments.<module> --config configs/<same>_20260906.yaml`。完了済みoutputを再実行しない。新production runは別namespace/config/registration。',
'- 売買候補は6h UTC決定、next15m open fill。utilityは自身のcash/units/NAVに基づき、risk1/costbudget2、maxstep0.08、deadband0.01、intent[0.5,1.12]。missing current openはhold、missing forecast ruleは選んだ候補に固定。無状態の毎回position1入力では検証済み経済経路にならない。',
'- shared inference2,586と score2,574を区別し、future label availabilityで注文を消さない。研究common maskは過去archiveの比較用supportで、receipt-time証明ではない。ライブでは観測可能なavailabilityのみで実装し、差分を明示してfixture/accounting parityする。',
'- Webの見出し・モデル名・銘柄・rawfeature契約・最終学習日・validation期間・追加test期間・費用を同じmanifestから読む。平均DD差を連結資産曲線のDD改善と混同しない。無条件の「全トレンド保証」「RL改善済み」は表示できない。','',
'## 元ファイルの SHA256','', '| ファイル | SHA256 |','|---|---|']
for name,p in paths.items():lines.append(f'| `{p}` | `{sha(W/p)}` |')
artifacts=['codex_outputs/oracle_frozen_procedure_parity_v1/models/fold12_perp_delay0_mean.joblib','codex_outputs/oracle_frozen_procedure_parity_v1/models/fold12_technical_mean.joblib','codex_outputs/oracle_frozen_procedure_parity_v1/models/fold12_technical_variance.joblib','codex_outputs/oracle_frozen_procedure_parity_v1/calibration/fold12_provenance.json','codex_outputs/oracle_mean_reliability_decisions_v1/weights/fold12_perp_delay0.json','codex_outputs/oracle_mean_reliability_decisions_v1/weights/fold12_technical.json','codex_outputs/oracle_additional_window_replay_v1/models/fold24_perp_delay0_mean.joblib','codex_outputs/oracle_additional_window_replay_v1/models/fold24_technical_variance.joblib','codex_outputs/oracle_additional_window_replay_v1/calibration/fold24_provenance.json']
for p in artifacts:assert (W/p).is_file();lines.append(f'| `{p}` | `{sha(W/p)}` |')
lines+=['','関連 source/報告の検索位置は現在のworkspaceで確認した。過去メモはRL report所在の発見にのみ使い、上のRL条件・数値・checkpoint非保持は現行repo本文へ照合した。メモ参照: MEMORY.md:181–220（rollout01a03504-87f0-7dd1-b9df-ef37cf68f50d）。']
p=Path('/tmp/oracle_deployment_candidate_audit_20260906.md');p.write_text('\n'.join(lines)+'\n')
j=Path('/tmp/oracle_deployment_candidate_audit_20260906.json');j.write_text(json.dumps({'status':'pass','scope':'existing completed development evidence only; no new fits or policy paths','source_results':{p:sha(W/p) for p in paths.values()},'minimum_sign_witnesses':[entries[cid] for cid in order],'eligible_development_witness_count':18,'learned_rl_joint_sign_and_exportable_artifact_confirmed':False,'report_sha256':sha(p),'script_sha256':sha(__file__)},sort_keys=True,indent=2,allow_nan=False)+'\n')
print(p,sha(p));print(j,sha(j))
