from pathlib import Path
import json, hashlib
from unidream.experiments import oracle_short_direction_decisions as m
R=Path(m.FIXED['output_dir']);E=Path('docs/experiments/oracle_short_direction_evidence_20260906')
d=json.loads((R/'results.json').read_text());s=d['summary'];reg=json.loads((R/'registration.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
f=lambda v:'null' if v is None else f'{v:.6f}'
pt=lambda v:f(v*100)
lines=['# Stage20 results: fixed short8 features for direction prediction','',
'2026-09-06. Development comparison only. No high-probability generalization is established and no candidate is promoted. The formal P1 state remains `results_observed=false`.',
'', '[Frozen protocol](oracle_short_direction_registration_20260906.md) · [Primary-source rationale](oracle_short_direction_research_20260906.md) · [Machine-readable results](oracle_short_direction_evidence_20260906/results.json)', '',
'## Registered comparison and evidence boundaries','',
'Added the complete existing short8 block to Technical29, producing Technical37. Exactly16 ordinary/magnitude C1 logistic fits were run. Only magnitude probability enters the unchanged `a_T*(2*q-1)` mean mapping and the two hold/fallback policies. No old models, unique priors, risk models or calibrations were fitted. All80 old policies,24 means and10 classification streams were preserved.',
'',
'Both tasks use the same original T rows, labels and weighting arithmetic. The29 inherited feature columns and37 selected fit/predict matrices were checked against independent Stage17/Stage15 records. Added features were not imputed and original supports were not narrowed. The comparison measures the whole regularized procedure; added correlated columns also change effective L2 geometry.',
'',
'Eight reused development quarters have2 bull,4 bear and2 sideways regimes. Original test(f) aliases validation(f+1). I strata use the later E-start regime retrospectively. Additional-test15–24 was not modeled or scored. Thus the inequalities below are descriptive comparisons, not a confidence level or all-trend guarantee. The bound Spot Parquet is decoded before strict semantic cutoff; archive event time is not receipt-time proof.',
'',
'## New policy economics','',
'AlphaEX and MaxDDdelta are percentage points, equal-weighted across quarters, relative to B&H. Positive AlphaEX and negative MaxDDdelta are desired. Stress doubles costs on the same targets. Joint counts require both signs under both costs in the same quarter.', '',
table(['Stratum','Rule','Quarters','Base AlphaEX','Base DDdelta','Stress AlphaEX','Stress DDdelta','Joint quarters'],
 [[g,rule,s['economics'][g][m.NEW_MEAN+'_'+rule]['quarters'],*[pt(s['economics'][g][m.NEW_MEAN+'_'+rule][co][k]) for co in ('base','stress_2x') for k in ('alpha_ex','maxdd_delta')],s['economics'][g][m.NEW_MEAN+'_'+rule]['joint_positive_quarters_both_costs']] for g in ('all','bull','bear','sideways') for rule in m.RULES]),'',
'All per-quarter new outcomes:', '',
table(['Fold','Trend','Rule','Base AlphaEX','Base DDdelta','Stress AlphaEX','Stress DDdelta'],[[r['fold'],r['regime']['trend'],r['candidate_id'].removeprefix(m.NEW_MEAN+'_'),*[pt(r[co][k]) for co in ('base','stress_2x') for k in ('alpha_ex','maxdd_delta')]] for r in d['rows'] if r['candidate_id'] in m.NEW_IDS]),'',
'## Probability and mapped-return evidence','',
'Ordinary losses assess ordinary direction probability; magnitude-weighted losses assess a different, absolute-return-weighted target. Neither is a guarantee of conditional-mean accuracy. This section reports both tasks even if only one improves.', '',
table(['Stratum','Segment','Task','Brier Δ vs29','Logloss Δ vs29','Brier Δ vsprior','Logloss Δ vsprior'],[[g,seg,w,*[f(s['classification_paired'][g][seg][m.GROUP+'_'+w][ref][k]) for ref in ('technical_'+w,'prior_'+w) for k in (('brier','log_loss') if w=='ordinary' else ('weighted_brier','weighted_log_loss'))]] for g in ('all','bull','bear','sideways') for seg in m.SEGMENTS for w in m.WEIGHTINGS]),'',
'Above, negative differences are improvement. The full JSON retains both weighted and ordinary scoring families for every classifier, including zeros/ties and denominator records.', '',
'Equal-quarter mapped MSE ×1,000,000:', '',
table(['Stratum','Segment',m.NEW_MEAN,*m.REFERENCES],[[g,seg,*[f(s['prediction'][g][seg][mid]['return_mse']*1e6) for mid in (m.NEW_MEAN,*m.REFERENCES)]] for g in ('all','bull','bear','sideways') for seg in m.SEGMENTS]),'',
'## Independent descriptive gates','',
table(['Task','I matched proper-loss gate','E matched proper-loss gate'],[[mid,*[s['probability_gates'][mid][seg] for seg in m.SEGMENTS]] for mid in m.MODEL_IDS]),'',
'Both-task conjunction: `'+str(s['both_classifier_families_improve_matched_losses_all_strata_both_segments']).lower()+'`.', '',
table(['Rule','Absolute economics','All6 paired economics','I mapped MSE','E mapped MSE','I source magnitude losses','E source magnitude losses','High-probability'],[[rule,*[s['short_direction'][m.NEW_MEAN+'_'+rule][k] for k in ('economic_means_all_strata_both_costs','economic_improvement_vs_all_six_references_all_strata_both_costs')],*[s['short_direction'][m.NEW_MEAN+'_'+rule]['mapped_mse_vs_all_six_references_improved_all_strata'][seg] for seg in m.SEGMENTS],*[s['short_direction'][m.NEW_MEAN+'_'+rule]['magnitude_probability_losses_vs_Technical29_and_prior_improved_all_strata'][seg] for seg in m.SEGMENTS],False] for rule in m.RULES]),'',
'These strict conditions cover every all/bull/bear/sideways stratum. A failure means the registered procedure did not demonstrate that improvement on the reused periods; it does not prove absence of information. Ordinary remains diagnostic for the policy. High-probability confirmation stays false regardless of the descriptive outcomes.', '',
'## Verification and provenance','',
'Run inventory:656 economic rows/1312 accounts,400 return-score records,192 classification-score records,32 direction diagnostics,16 mapping diagnostics,95 artifacts/fold. All640 old economic records,384 old return records,160 old classification records and the complete Stage19 summary were checked exactly. The adaptive causal-name ledger is220.', '',
'Original E inference2586/score2574 and I mapped inference2537/score2523 remain distinct; unscored origins still receive predictions. Missing forecasts and missing opens preserve332 fallback rows and2 gaps. Zero-mean controls retain the same risk controller and are not B&H.', '',
'Freeze revision: `'+reg['source_revision']+'`.', '',
table(['Binding','SHA256'],[[name,sha(path)] for name,path in [('config','configs/oracle_short_direction_decisions_20260906.yaml'),('protocol','docs/experiments/oracle_short_direction_registration_20260906.md'),('research','docs/experiments/oracle_short_direction_research_20260906.md'),('registration',R/'registration.json'),('preflight',R/'preflight.json'),('results',R/'results.json')]]), '',
'[Evidence directory](oracle_short_direction_evidence_20260906/) contains runtime manifests/logs and independent audit sources/results. Large binaries remain local hash-bound. Independent audit details and numerical-warning counts are recorded in `publication_verification.json`.', '',
'## Interpretation','',
'This experiment closes the fixed Technical37 task comparison. Any subsequent hypothesis must receive a separate pre-outcome registration. No threshold, feature subset, C or model structure was changed after these outcomes.', '']
Path('docs/experiments/oracle_short_direction_results_20260906.md').write_text('\n'.join(lines))
print('report_written')
