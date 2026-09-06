"""Stage20 input-only bridge between immutable Stage19 and Stage15.
No estimator fit/predict, new binary statistic, mapping, loss or order is called.
Caller must validate its fixed experiment config; this validates source/data inputs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml
from unidream.experiments import oracle_soft_direction_decisions as soft
from unidream.experiments import oracle_short_feature_decisions as short
from unidream.experiments.oracle_confirmation_contract import calendar

FOLDS=tuple(range(5,13))
SIX=('fit','predict','scale','interval','inference','score')
GROUP='technical_short_both'
ROOT19=Path('codex_outputs/oracle_soft_direction_decisions_v1')
ROOT15=Path('codex_outputs/oracle_short_feature_decisions_v1')
ROOT17=Path('codex_outputs/oracle_direction_decisions_v1')
REV19='9b4f6a0e5606831a26a8f2a7c401e05c52d41f6f'
REV15='d11cfee15b77f773e353c7ecb1ba4729c0b4abe7'
PACK_KEYS={'fit_positions','timestamps','returns','binary_labels','predict_positions','predict_timestamps',
 'fit_features_technical','fit_features_perp_delay0','predict_features_technical','predict_features_perp_delay0',
 'weights_ordinary','weights_magnitude'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(condition,message):
 if not condition:raise ValueError(message)
def exact(a,b,name):
 a,b=np.asarray(a),np.asarray(b)
 require(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b,equal_nan=True),'changed '+name)
def matrixsha(a):
 a=np.asarray(a,dtype='<f8',order='C')
 return hashlib.sha256(np.asarray([a.ndim,*a.shape],dtype='<i8').tobytes()+a.tobytes()).hexdigest()
def masksha(index,mask):return hashlib.sha256(index.asi8.astype('<i8').tobytes()+np.asarray(mask,'u1').tobytes()).hexdigest()
def positionmasksha(mask):return hashlib.sha256(np.asarray([len(mask)],dtype='<i8').tobytes()+np.asarray(mask,'u1').tobytes()).hexdigest()
def indexsha(index):
 header=json.dumps({'type':type(index).__name__,'dtype':str(index.dtype),'length':len(index)},sort_keys=True).encode()
 return hashlib.sha256(header+b'\n'+pd.util.hash_pandas_object(index,index=False).to_numpy('<u8').tobytes()).hexdigest()
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def merge(*parts):
 out={};resolved={}
 for part in parts:
  require(isinstance(part,dict),'binding map required')
  for path,value in part.items():
   require(isinstance(path,str) and isinstance(value,str) and len(value)==64,'malformed binding')
   canonical=str(Path(path).resolve())
   require(canonical not in resolved or resolved[canonical]==path,'aliased binding '+path)
   require(path not in out or out[path]==value,'conflicting binding '+path)
   out[path]=value;resolved[canonical]=path
 return out

def manifest_paths(root):return {str(root/n) for n in ['registration.json','preflight.json','results.json']+[f'fold_{f}.json' for f in FOLDS]}

def complete_stage(root,cfg,prepared_pre,manifest_bindings,revision,kind):
 require(set(manifest_bindings)==manifest_paths(root),'incomplete '+kind+' manifests')
 reg=read(root/'registration.json');res=read(root/'results.json');saved_pre=read(root/'preflight.json')
 require(reg['config']==cfg and reg['source_revision']==revision,'changed '+kind+' registration')
 require(reg['preflight_sha256']==cfg['preflight_sha256']==manifest_bindings[str(root/'preflight.json')],'changed '+kind+' preflight binding')
 require(saved_pre==prepared_pre and res['registration_sha256']==digest(reg),'changed '+kind+' completed chain')
 own={}
 for f in FOLDS:
  fold=read(root/f'fold_{f}.json');require(fold['registration_sha256']==digest(reg),'changed fold registration')
  if kind=='Stage19':
   specs=[('forecasts',soft.NEW_MEANS,'npz'),('calibration',soft.NEW_MEANS,'npz'),('targets',soft.POLICIES,'npz'),('traces',soft.NEW_IDS,'json'),('provenance',('mapping',),'json')]
   keys=('rows','scores','classification_scores','mapping_diagnostics')
  else:
   specs=[('models',short.GROUPS,'joblib'),('provenance',('fit',),'json'),('calibration',short.GROUPS,'npz'),('forecasts',tuple(g+'_raw' for g in short.GROUPS),'npz'),('targets',short.POLICIES,'npz'),('traces',short.NEW_IDS,'json')]
   keys=('rows','scores')
  expected={str(root/folder/f'fold{f}_{name}.{ext}') for folder,names,ext in specs for name in names}
  require(set(fold['artifact_sha256'])==expected,'changed '+kind+' own artifact inventory')
  for key in keys:require(fold[key]==[r for r in res[key] if r['fold']==f],'changed '+kind+' fold '+key)
  if kind=='Stage15':require(fold['baseline_parity']==next(r for r in res['baseline_parity'] if r['fold']==f),'changed baseline provenance')
  require(not set(own).intersection(fold['artifact_sha256']),'duplicate own artifact')
  own.update(fold['artifact_sha256'])
 require(not set(own).intersection(prepared_pre['source_artifact_bindings']),'duplicate ancestry/own artifact')
 return reg,res,merge(prepared_pre['source_artifact_bindings'],own),own

def prepare_sources(cfg):
 """Return original inputs and a JSON-safe preflight, with no new model arithmetic.

 Required cfg keys: parent_config, feature_config, source_bindings,
 parent_manifest_bindings, feature_manifest_bindings, parent_config_sha256,
 feature_config_sha256. preflight_sha256 and experiment-specific keys are retained
 only in the config digest. Run from the research worktree, as existing prepares do.
 """
 required={'parent_config','feature_config','source_bindings','parent_manifest_bindings','feature_manifest_bindings','parent_config_sha256','feature_config_sha256'}
 require(required<=set(cfg),'missing input provenance config')
 require(cfg['parent_config']=='configs/oracle_soft_direction_decisions_20260906.yaml' and cfg['feature_config']=='configs/oracle_short_feature_decisions_20260906.yaml','changed source configurations')
 old_sources=set(soft.SOURCES)|set(short.SOURCES)
 require(len(old_sources)==39 and old_sources<=set(cfg['source_bindings']),'incomplete original source union')
 direct=merge(cfg['source_bindings'],cfg['parent_manifest_bindings'],cfg['feature_manifest_bindings'],
  {cfg['parent_config']:cfg['parent_config_sha256'],cfg['feature_config']:cfg['feature_config_sha256']})
 verified={}
 def verify(bindings):
  for path,value in bindings.items():
   if path not in verified:verified[path]=sha(path)
   require(verified[path]==value,'changed bound file '+path)
 verify(direct)
 pc,bars,old_masks,scalars,evaluation,calibration,_,_,pp=soft.prepare(cfg['parent_config'])
 sc,fc,sbars,groups,old_y,masks,_,_,_,sp=short.prepare(cfg['feature_config'])
 require(bars.equals(sbars) and pc['data_cutoff']==sc['data_cutoff']==fc['data_cutoff']=='2023-04-16T13:45:00Z','changed common market grid')
 reg19,res19,a19,own19=complete_stage(ROOT19,pc,pp,cfg['parent_manifest_bindings'],REV19,'Stage19')
 reg15,res15,a15,own15=complete_stage(ROOT15,sc,sp,cfg['feature_manifest_bindings'],REV15,'Stage15')
 require(reg19['config_sha256']==cfg['parent_config_sha256'] and reg15['config_sha256']==cfg['feature_config_sha256'],'changed source config file binding')
 require(len(a19)==4456 and len(a15)==2216 and len(set(a19)&set(a15))==1328,'changed two-chain inventories')
 bindings=merge(a19,a15);require(len(bindings)==5344,'changed union inventory')
 direct=merge(direct,pp['direct_source_bindings'],sp['direct_source_bindings']);verify(merge(direct,bindings))
 controls={(r['fold'],r['candidate_id']):r for r in res19['rows']}
 require(len(controls)==len(res19['rows'])==640 and set(controls)=={(f,c) for f in FOLDS for c in soft.POLICIES},'changed 80-control inventory')
 require(len(res19['scores'])==384 and len(res19['classification_scores'])==160,'changed old score inventories')
 require(len(res15['rows'])==272 and len(res15['scores'])==184,'changed feature-stage record inventory')
 require(tuple(groups[GROUP].columns)==tuple(groups['technical'].columns)+tuple(short.PRICE_FEATURE_NAMES)+tuple(short.FLOW_FEATURE_NAMES),'changed exact37 order')
 require(len(groups[GROUP].columns)==37 and len(groups['technical'].columns)==29,'changed feature dimensions')
 packs={};fit_y={};old_provenance={};support=[]
 for f in FOLDS:
  m=masks[f];dates=calendar(f-1);ix=np.asarray((bars.index>=dates['evaluation_start'])&(bars.index<dates['evaluation_end']));ci=np.asarray((bars.index>=dates['scale_start'])&(bars.index<dates['evaluation_start']))
  ppack=ROOT17/'fit_data'/f'fold{f}_training.npz';p17=ROOT17/'provenance'/f'fold{f}_fit.json';p15=ROOT15/'provenance'/f'fold{f}_fit.json'
  require(all(str(p) in bindings for p in (ppack,p17,p15)),'unbound selected input provenance')
  pack=arrays(ppack);packs[f]=pack;require(set(pack)==PACK_KEYS,'changed original training pack schema')
  old_provenance[f]=read(p17);prov17=old_provenance[f]['fit_provenance'];prov15=read(p15)['fit_provenance'];s15=next(v for v in sp['support'] if v['fold']==f)
  require(prov17['index_sha256']==prov15['index_sha256']==indexsha(bars.index),'changed full-grid index')
  for key in SIX:
   exact(m[key],old_masks[f][key],f'{f} '+key)
   require(m[key].dtype==bool and m[key].shape==(len(bars),),'invalid support')
   require(np.isfinite(groups[GROUP].to_numpy()[m[key]]).all(),'nonfinite exact37 on original '+key)
   require(masksha(bars.index,m[key])==s15['mask_sha256'][key],'changed feature preflight mask')
  for name,poskey,timekey in [('fit','fit_positions','timestamps'),('predict','predict_positions','predict_timestamps')]:
   exact(np.flatnonzero(m[name]),pack[poskey],name+' selected positions')
   exact(bars.index[m[name]].asi8,pack[timekey],name+' timestamps')
   exact(groups['technical'].to_numpy()[m[name]],pack[name+'_features_technical'],name+' frozen technical29')
   exact(groups[GROUP].to_numpy()[m[name],:29],pack[name+'_features_technical'],name+' exact37 baseline prefix')
   require(positionmasksha(m[name])==prov17['mask_position_sha256'][name]==prov15['mask_position_sha256'][name],'changed positional support')
   for g in ('technical',GROUP):require(matrixsha(groups[g].to_numpy()[m[name]])==prov15[name+'_features_sha256'][g],'changed saved Stage15 '+name+' '+g+' matrix')
   require(matrixsha(pack[name+'_features_technical'])==prov17[name+'_features_sha256']['technical'],'changed saved technical29 matrix')
  require(list(groups['technical'].columns)==prov17['feature_columns']['technical'] and list(groups[GROUP].columns)==prov15['feature_columns'][GROUP],'changed bound feature names')
  exact(old_y[m['fit'],0],pack['returns'],'frozen T returns')
  require(matrixsha(pack['returns'])==prov17['fit_return_sha256']==prov15['fit_return_sha256'],'changed T return provenance')
  require(matrixsha(pack['binary_labels'])==prov17['fit_binary_labels_sha256'],'changed saved T binary labels')
  for weighting in ('ordinary','magnitude'):require(matrixsha(pack['weights_'+weighting])==prov17['sample_weights'][weighting]['weight_sha256'],'changed saved T weights')
  fy=np.full((len(bars),3),np.nan);fy[m['fit'],0]=pack['returns'];fit_y[f]=fy
  for mean in soft.NEW_MEANS:
   source=soft.MAPPING[mean]['source_mean']
   for folder,destination,selected in [('forecasts',evaluation,ix),('calibration',calibration,ci)]:
    path=ROOT19/folder/f'fold{f}_{mean}.npz';require(str(path) in bindings,'unbound retained Stage19 forecast')
    a=arrays(path);ref=destination[f,source];require(set(a)==set(ref),'changed new parent forecast schema')
    for key in a:
     if key!='mu':exact(a[key],ref[key],'retained paired '+key)
    exact(a['timestamps'],bars.index[selected].asi8,'retained forecast timestamps')
    active=a['inference_mask'] if folder=='forecasts' else a['mapped_inference_mask']
    require(a['mu'].dtype==np.float64 and np.isfinite(a['mu'][active]).all() and np.isnan(a['mu'][~active]).all(),'changed retained forecast support')
    destination[f,mean]=a
  require({mean for ff,mean in evaluation if ff==f}==set(soft.MEANS) and {mean for ff,mean in calibration if ff==f}==set(soft.MEANS),'incomplete 24-mean inventory')
  ref=evaluation[f,'technical_half'];ca=calibration[f,'technical_half']
  exact(old_y[m['score']],ref['actual'][m['score'][ix]],'reconstructed E score labels')
  for key in ('scale','interval'):exact(old_y[m[key]],ca['actual'][m[key][ci]],'reconstructed '+key+' labels')
  require(np.all(bars.index[m['fit']]+pd.Timedelta(minutes=375)<dates['scale_start']),'T label maturity violation')
  require(np.all(bars.index[m['scale']]+pd.Timedelta(minutes=375)<dates['interval_start']),'S label maturity violation')
  require(np.all(bars.index[m['interval']]+pd.Timedelta(minutes=375)<dates['evaluation_start']),'I label maturity violation')
  require(np.all(bars.index[m['score']]+pd.Timedelta(minutes=375)<=dates['evaluation_end']),'E label maturity violation')
  parent_support=next(v for v in pp['support'] if v['fold']==f)
  support.append({**parent_support,'short_feature_columns':list(groups[GROUP].columns),
   'short_fit_features_sha256':matrixsha(groups[GROUP].to_numpy()[m['fit']]),
   'short_predict_features_sha256':matrixsha(groups[GROUP].to_numpy()[m['predict']]),
   'short_fit_canonical_json_sha256':digest(groups[GROUP].to_numpy()[m['fit']].tolist()),
   'fit_return_float64le_sha256':matrixsha(pack['returns']),
   'original_fit_data_path':str(ppack),'original_fit_data_sha256':bindings[str(ppack)],
   'short_fit_provenance_path':str(p15),'short_fit_provenance_sha256':bindings[str(p15)],
   'finite_rows_on_required_masks':{k:int(m[k].sum()) for k in SIX},
   'selected_technical29_exact':True,'selected37_exact_saved_Stage15_provenance':True,
   'all_six_original_masks_exact':True,'support_narrowed':False,'scheduled_finiteness_required':False,
   'evaluation_inference_rows':int(ref['inference_mask'].sum()),'evaluation_score_rows':int(ref['score_support'].sum()),
   'interval_inference_rows':int(ca['mapped_inference_mask'].sum()),'interval_score_rows':int(ca['interval_mask'].sum())})
 require(sum(v['evaluation_inference_rows'] for v in support)==2586 and sum(v['evaluation_score_rows'] for v in support)==2574,'changed E support totals')
 require(sum(v['interval_inference_rows'] for v in support)==2537 and sum(v['interval_score_rows'] for v in support)==2523,'changed I support totals')
 pre={'schema':'oracle-short-direction-input-bridge-v1','config_contract_sha256':digest({k:v for k,v in cfg.items() if k!='preflight_sha256'}),
  'source_bindings':cfg['source_bindings'],'direct_source_bindings':direct,'source_artifact_bindings':bindings,
  'support':support,'spot_data_proof':pp['spot_data_proof'],'um_data_proof':pp['um_data_proof'],
  'parent_registration_canonical_sha256':digest(reg19),'feature_registration_canonical_sha256':digest(reg15),
  'inventory':{'parent_complete_artifacts':4456,'feature_complete_artifacts':2216,'shared_artifacts':1328,'union_artifacts':5344,'original_source_files':39,'parent_rows':640,'parent_return_scores':384,'parent_classification_scores':160,'retained_means':24,'retained_policies':80,'feature_stage_rows':272,'feature_stage_return_scores':184},
  'no_new_fits_statistics_predictions_losses_or_orders':True,'only_new_feature_reconstruction_and_existing_array_identity_checks':True,
  'loader_scope':'Both inherited prepares decode Spot before strict semantic cutoff; reconstructed features/labels used only on original development masks; no additional-test semantic use',
  'historical_receipt_provenance_established':False}
 return {'bars':bars,'feature_config':fc,'features37':groups[GROUP],'y_fit_only':fit_y,'masks':masks,'packs':packs,'old_provenance':old_provenance,'scalars':scalars,'evaluation':evaluation,'calibration':calibration,'controls':controls,'parent':res19,'preflight':pre}
