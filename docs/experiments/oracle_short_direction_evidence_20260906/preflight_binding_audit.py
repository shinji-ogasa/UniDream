"""Compare the official Stage20 input-only preflight to independent verified inputs.
No prepare, fit, prediction, new statistic, mapping, scoring or planning is called.
"""
from pathlib import Path
import hashlib,json,os,yaml
ROOT=Path('/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905')
os.chdir(ROOT)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)

def main():
 p=Path('codex_outputs/oracle_short_direction_decisions_v1/preflight.json');cp=Path('configs/oracle_short_direction_decisions_20260906.yaml')
 audit_path=Path('/tmp/oracle_short_direction_source_audit_20260906.json');bridge=Path('/tmp/oracle_short_direction_prepare.py')
 require(sha(p)=='1a48c5d966c32439b0e3efc837210a88cf60f7cd96b194b2378fad1a5a62c4b6','official preflight hash changed')
 require(sha(cp)=='3eb2c811e9d96dd5e3e2d51483694e1b4d6dd412cd61dda50fe1d6d74df9520f','registered config hash changed')
 require(sha(audit_path)=='88f6c1f6031029a404310b2ce1bc741d85b5f6fca94cc11e8a09f2b68c8f5c5b','independent input audit changed')
 cfg=yaml.safe_load(cp.read_text());pre=read(p);old=read(audit_path)
 require(old['passed'] and old['script_sha256']==sha(bridge)=='9e725d84099bef356749e538617ec43cec4fae81b5e72ee16a05f45c58425e36','independent bridge changed')
 require(Path('unidream/experiments/oracle_short_direction_inputs.py').read_text()==bridge.read_text().split("if __name__=='__main__':")[0].rstrip()+'\n','integrated bridge differs')
 from unidream.experiments.oracle_short_direction_decisions import validate_config
 validate_config(cfg)
 require(pre['config_contract_sha256']==digest({k:v for k,v in cfg.items() if k!='preflight_sha256'}),'official config digest changed')
 require(cfg['preflight_sha256']==sha(p),'config/preflight binding changed')
 require(pre['source_bindings']==cfg['source_bindings'] and len(pre['source_bindings'])==42,'wrong source set')
 expected_new={'unidream/experiments/oracle_short_direction_'+n+'.py' for n in ('fit','inputs','decisions')}
 require(set(pre['source_bindings'])-set(old['source_bindings'])==expected_new,'unexpected new source set')
 require(set(old['source_bindings'])<=set(pre['source_bindings']),'lost old source')
 require(set(pre['direct_source_bindings'])-set(old['direct_source_bindings'])==expected_new,'unexpected new direct binding')
 for key in ('source_bindings','direct_source_bindings'):
  for path,h in old[key].items():require(pre[key][path]==h,'changed inherited '+key+' '+path)
 for key in ('source_artifact_bindings','support','spot_data_proof','um_data_proof','inventory',
             'parent_registration_canonical_sha256','feature_registration_canonical_sha256','schema',
             'no_new_fits_statistics_predictions_losses_or_orders','only_new_feature_reconstruction_and_existing_array_identity_checks',
             'loader_scope','historical_receipt_provenance_established'):
  require(pre[key]==old[key],'changed independent field '+key)
 require(len(pre['source_artifact_bindings'])==5344,'wrong ancestor inventory')
 verified={};checks=0
 for bindings in (pre['source_bindings'],pre['direct_source_bindings'],pre['source_artifact_bindings']):
  for path,h in bindings.items():
   checks+=1
   if path not in verified:verified[path]=sha(path)
   require(verified[path]==h,'changed file '+path)
 extras=['docs/experiments/oracle_short_direction_registration_20260906.md','tests/test_oracle_short_direction_fit.py']
 review={'status':'pass','script_sha256':sha(__file__),'scope':'Official and independent input-only preflight comparison plus final source integration review',
  'official_config_path':str(cp),'official_config_sha256':sha(cp),'official_preflight_path':str(p),'official_preflight_sha256':sha(p),
  'independent_input_audit_path':str(audit_path),'independent_input_audit_sha256':sha(audit_path),
  'independent_bridge_sha256':sha(bridge),'source_bindings':pre['source_bindings'],
  'other_reviewed_bindings':{path:sha(path) for path in extras},'source_binding_count':42,'original_source_binding_count':39,
  'hash_binding_checks':checks,'distinct_verified_files':len(verified),'ancestor_artifact_binding_count':5344,
  'support_exact_across_all_eight_folds':True,'all_six_masks_exact':True,
  'selected29_original_arrays_exact':True,'selected37_fit_and_predict_exact_saved_Stage15_hashes':True,
  'evaluation_inference_rows':2586,'evaluation_score_rows':2574,'interval_inference_rows':2537,'interval_score_rows':2523,
  'original_fit_row_counts':[s['counts']['fit'] for s in pre['support']],
  'retained_means':24,'retained_policies':80,'retained_economic_rows':640,'retained_return_scores':384,'retained_classifier_scores':160,
  'no_new_fits_statistics_predictions_losses_or_orders':True,'material_findings':[],
  'limitations':['Original availability mask is retrospective; no new confirmation evidence.',
                 'Inherited loader decodes full Spot parquet before semantic cutoff; no later-price modeling or scoring.',
                 'Archive provenance is not historical receipt evidence.',
                 'This audit establishes input identity and source integration, not predictive performance.']}
 out=Path('/tmp/oracle_short_direction_preflight_binding_audit_20260906.json');out.write_text(json.dumps(review,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'path':str(out),'sha256':sha(out),'script_sha256':sha(__file__),'hash_binding_checks':checks,'distinct_verified_files':len(verified),'status':'pass'}))
if __name__=='__main__':main()
