from pathlib import Path
import hashlib,json,subprocess,re,shutil
root=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
ev=root/'docs/experiments/oracle_regularized_direction_evidence_20260906'
out=root/'codex_outputs/oracle_regularized_direction_decisions_v1'
report=root/'docs/experiments/oracle_regularized_direction_results_20260906.md'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
bound={}
def verify(path,expected):
 p=Path(path); p=p if p.is_absolute() else root/p; p=p.resolve()
 if str(p) in bound:assert bound[str(p)]==expected,(p,'conflicting binding')
 else:assert sha(p)==expected,(p,'digest mismatch');bound[str(p)]=expected
regv=read(ev/'registration_verification.json')
src=read(ev/'oracle_regularized_direction_source_audit_20260906.json')
for p,h in regv['source_bindings'].items():verify(p,h)
for key in ['source_artifact_bindings','direct_source_bindings']:
 for p,h in src[key].items():verify(p,h)
assert len(src['source_artifact_bindings'])==2840
own={}
for f in range(5,13):
 p=out/f'fold_{f}.json';d=read(p)
 assert len(d['artifact_sha256'])==81
 for q,h in d['artifact_sha256'].items():
  assert q not in own;own[q]=h;verify(q,h)
 assert p.read_bytes()==(ev/p.name).read_bytes()
assert len(own)==648
for p in [out/'preflight.json',out/'registration.json',out/'results.json',out/'run.log']:
 assert p.read_bytes()==(ev/p.name).read_bytes()
assert sha(ev/'full-tests.log')==regv['full_test_log_sha256']
assert 'Ran 733 tests in 57.350s' in (ev/'full-tests.log').read_text()
assert re.search(r'(?m)^Ran 733 tests in 57\.350s\n\nOK$', (ev/'full-tests.log').read_text())
assert not re.search(r'(?m)^FAILED(?: |$)', (ev/'full-tests.log').read_text())
for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',report.read_text()):
 if not target.startswith(('https://','http://','#')):assert (report.parent/target.split('#')[0]).exists(),target
frozen='5a82c270c64a342ab7e9df8105b7d23d1336d876'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==frozen
subprocess.run(['git','diff','--exit-code',frozen,'--','unidream','tests','configs','docs/experiments/oracle_regularized_direction_registration_20260906.md','docs/experiments/oracle_regularized_direction_research_20260906.md'],cwd=root,check=True,capture_output=True)
subprocess.run(['git','diff','--check'],cwd=root,check=True)
for stem in ['audit','source_audit','model_audit','score_audit','old_objective_audit']:
 name='oracle_regularized_direction_'+stem+'_20260906';d=read(ev/(name+'.json'))
 assert d.get('passed',d.get('status')) in (True,'pass')
 expected=d.get('script_sha256',d.get('audit_script',{}).get('sha256'))
 assert expected==sha(ev/(name+'.py'))
print(json.dumps({'verified':True,'inherited_artifacts':2840,'own_artifacts':648,'source_files':len(regv['source_bindings']),'distinct_hashed_files':len(bound),'report_sha256':sha(report),'copied_runtime_exact':True,'source_unchanged':True}))

# Final report audit must have completed before publication.
ra_name='oracle_regularized_direction_report_audit_20260906'
ra_path=Path('/tmp')/(ra_name+'.json')
assert ra_path.exists(), 'Final independent report audit still pending'
ra=read(ra_path)
assert ra.get('passed',ra.get('status')) in (True,'pass')
assert ra['report_sha256']==sha(report)
assert ra['script_sha256']==sha(Path('/tmp')/(ra_name+'.py'))
for p,h in ra['bound_input_sha256'].items():verify(p,h)
for suffix in ['.py','.json']:
 shutil.copyfile(Path('/tmp')/(ra_name+suffix),ev/(ra_name+suffix))
audits={}
for p in sorted(ev.glob('oracle_regularized_direction*.json')):
 companion=p.with_suffix('.py')
 audits[p.stem]={'result_sha256':sha(p)}
 if companion.exists():audits[p.stem]['script_sha256']=sha(companion)
r=read(out/'results.json');reg=read(out/'registration.json')
assert r['registration_sha256']==hashlib.sha256(json.dumps(reg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
metadata={
 'source_revision':frozen,'source_unchanged_since_freeze':True,
 'run_attempts':1,'run_session':98384,'run_exit_code':0,
 'new_model_fits':32,'new_unique_priors':0,'frozen_prior_verifications':16,
 'economic_rows':480,'economic_accounts_checked':960,'return_scores_checked':224,'classification_scores_checked':160,
 'direction_diagnostics_checked':64,'new_policy_paths_checked':64,'new_decisions_checked':22016,
 'new_causal_names':8,'total_adaptively_explored_causal_names':198,
 'registered_sources_verified':31,'ancestor_artifacts_verified':2840,'new_artifacts_verified':648,
 'copied_runtime_manifests_exact':True,'relative_report_links_resolve':True,
 'full_test_count':733,'full_test_exit_code':0,'full_test_seconds':57.350,
 'full_test_log_sha256':sha(ev/'full-tests.log'),
 'config_sha256':reg['config_sha256'],'registration_file_sha256':sha(out/'registration.json'),
 'registration_canonical_sha256':r['registration_sha256'],'results_file_sha256':sha(out/'results.json'),
 'run_log_sha256':sha(out/'run.log'),'report_sha256':sha(report),
 'formatted_performance_cells_checked':ra['performance_cells'],
 'formatted_performance_rows_checked':ra['total_performance_rows'],
 'formatted_output_hash_cells_checked':ra['hash_claims'],
 'report_audit_result':ra,'audits':audits,
 'runtime_warnings_retained':384,'warning_cause_established':False,
 'additional_test_used':False,'new_real_model_selected':False,'high_probability_generalization_established':False,
 'outcome':'All eight new policies rejected: overall AlphaEX<0 and DDdelta>0; all registered improvement gates fail.',
 'publication_script_sha256':sha(__file__)
}
shutil.copyfile(__file__,ev/'publication_verify.py')
metadata['evidence_file_sha256']={str(p.relative_to(ev)):sha(p) for p in sorted(ev.iterdir()) if p.is_file() and p.name!='publication_verification.json'}
(ev/'publication_verification.json').write_text(json.dumps(metadata,sort_keys=True,indent=2,allow_nan=False)+'\n')
print('Publication verified',len(metadata['evidence_file_sha256']),'evidence files',sha(ev/'publication_verification.json'))
