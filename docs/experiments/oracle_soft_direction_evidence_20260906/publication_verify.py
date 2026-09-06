from pathlib import Path
import hashlib,json,re,shutil,subprocess
root=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
ev=root/'docs/experiments/oracle_soft_direction_evidence_20260906'
out=root/'codex_outputs/oracle_soft_direction_decisions_v1'
report=root/'docs/experiments/oracle_soft_direction_results_20260906.md'
frozen='9b4f6a0e5606831a26a8f2a7c401e05c52d41f6f'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
bound={}
def verify(path,h):
 p=Path(path);p=p if p.is_absolute() else root/p;p=p.resolve()
 if str(p) in bound:assert bound[str(p)]==h,(p,'conflicting binding')
 else:assert sha(p)==h,(p,'digest mismatch');bound[str(p)]=h
regv=read(ev/'registration_verification.json');pre=read(out/'preflight.json');r=read(out/'results.json');reg=read(out/'registration.json')
for key in ['source_bindings','direct_source_bindings','source_artifact_bindings']:
 for p,h in pre[key].items():verify(p,h)
assert len(pre['source_bindings'])==33 and len(pre['source_artifact_bindings'])==3488
own={}
for f in range(5,13):
 p=out/f'fold_{f}.json';d=read(p);assert len(d['artifact_sha256'])==121
 for q,h in d['artifact_sha256'].items():
  assert q not in own;own[q]=h;verify(q,h)
 assert p.read_bytes()==(ev/p.name).read_bytes()
assert len(own)==968
for name in ['preflight.json','registration.json','results.json','run.log']:
 assert (out/name).read_bytes()==(ev/name).read_bytes()
assert sha(out/'preflight.json')==regv['preflight_sha256']
assert sha(ev/'full-tests.log')==regv['full_test_log_sha256']
assert re.search(r'(?m)^Ran 752 tests in 58\.614s\n\nOK$',(ev/'full-tests.log').read_text())
assert not re.search(r'(?m)^FAILED(?: |$)',(ev/'full-tests.log').read_text())
lines=(out/'run.log').read_text().splitlines();assert len(lines)==8
assert [json.loads(v)['fold'] for v in lines]==list(range(5,13))
assert all(json.loads(v)['event']=='fold_complete' for v in lines)
assert r['registration_sha256']==hashlib.sha256(json.dumps(reg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',report.read_text()):
 if not target.startswith(('https://','http://','#')):assert (report.parent/target.split('#')[0]).exists(),target
assert not re.search(r'PENDING|TODO|FIXME',report.read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==frozen
subprocess.run(['git','diff','--exit-code',frozen,'--','unidream','tests','configs','docs/experiments/oracle_soft_direction_registration_20260906.md','docs/experiments/oracle_soft_direction_research_20260906.md'],cwd=root,check=True,capture_output=True)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
for stem in ['audit','source_audit','mapping_audit','score_audit','preflight_binding_audit']:
 name='oracle_soft_direction_'+stem+'_20260906';d=read(ev/(name+'.json'))
 assert d.get('status')=='pass_pre_freeze_input_bindings' if stem=='preflight_binding_audit' else d.get('passed',d.get('status')) in (True,'pass')
 expected=d.get('script_sha256',d.get('audit_script',{}).get('sha256'))
 assert expected==sha(ev/(name+'.py'))
print(json.dumps({'verified':True,'source_files':33,'ancestor_artifacts':3488,'new_artifacts':968,'distinct_hashed_files':len(bound),'report_sha256':sha(report),'copied_runtime_exact':True,'source_unchanged':True}))
# Requires the completed independent report audit; no fit or economic replay.
ra_name='oracle_soft_direction_report_audit_20260906';ra_path=Path('/tmp')/(ra_name+'.json');assert ra_path.exists(),'report audit pending'
ra=read(ra_path);assert ra.get('passed',ra.get('status')) in (True,'pass')
assert ra['report_sha256']==sha(report)
assert ra['audit_script_sha256']==sha(Path('/tmp')/(ra_name+'.py'))
for p,h in ra['source_sha256'].items():verify(p,h)
for suffix in ['.py','.json']:shutil.copyfile(Path('/tmp')/(ra_name+suffix),ev/(ra_name+suffix))
audits={}
for p in sorted(ev.glob('oracle_soft_direction*.json')):
 audits[p.stem]={'result_sha256':sha(p)}
 if p.with_suffix('.py').exists():audits[p.stem]['script_sha256']=sha(p.with_suffix('.py'))
metadata={'source_revision':frozen,'source_unchanged_since_freeze':True,'run_attempts':1,'run_session':91810,'run_exit_code':0,
 'new_model_fits':0,'new_probability_prediction_calls':0,'new_unique_priors':0,
 'economic_rows':640,'economic_accounts_checked':1280,'return_scores_checked':384,'classification_scores_checked':160,
 'mapping_diagnostics_checked':64,'new_mapping_npz_checked':160,'new_policy_paths_checked':160,'new_decisions_checked':55040,
 'new_causal_names':20,'new_learned_policy_names':8,'new_constant_control_policy_names':12,'total_adaptively_explored_causal_names':218,
 'registered_sources_verified':33,'ancestor_artifacts_verified':3488,'new_artifacts_verified':968,
 'complete_old_rows_exact':480,'complete_old_return_scores_exact':224,'complete_old_classification_scores_exact':160,
 'copied_runtime_manifests_exact':True,'relative_report_links_resolve':True,
 'full_test_count':752,'full_test_exit_code':0,'full_test_seconds':58.614,'full_test_log_sha256':sha(ev/'full-tests.log'),
 'config_sha256':reg['config_sha256'],'registration_file_sha256':sha(out/'registration.json'),
 'registration_canonical_sha256':r['registration_sha256'],'results_file_sha256':sha(out/'results.json'),
 'run_log_sha256':sha(out/'run.log'),'runtime_warning_lines':0,'report_sha256':sha(report),
 'formatted_total_table_cells_checked':ra['numeric_and_joint_table_cells'],
 'formatted_performance_and_joint_cells_checked':404,'formatted_fold_ID_cells_checked':8,
 'formatted_rows_checked':ra['total_table_rows'],'formatted_report_hashes_checked':ra['report_direct_hash_count'],
 'report_audit_result':ra,'audits':audits,'additional_test_used':False,'new_real_model_selected':False,
 'new_probability_accuracy_improvement':False,'high_probability_generalization_established':False,
 'outcome':'All eight learned mappings rejected: absolute/paired economic and mapped-MSE gates fail; unchanged probability evidence.',
 'publication_script_sha256':sha(__file__)}
shutil.copyfile(__file__,ev/'publication_verify.py')
metadata['evidence_file_sha256']={p.name:sha(p) for p in sorted(ev.iterdir()) if p.is_file() and p.name!='publication_verification.json'}
(ev/'publication_verification.json').write_text(json.dumps(metadata,sort_keys=True,indent=2,allow_nan=False)+'\n')
print('Publication verified',len(metadata['evidence_file_sha256']),'evidence files',sha(ev/'publication_verification.json'))
