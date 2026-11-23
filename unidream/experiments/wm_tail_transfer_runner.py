"""Execute the registered staged comparison without Test-based reselection."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing
from pathlib import Path
import pickle
import time
import traceback

import numpy as np
import pandas as pd

from . import wm_tail_transfer_benchmark as campaign
_THREAD_LIMIT = None


def _initialize_worker(threads):
    global _THREAD_LIMIT
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(int(threads))
    torch.set_num_interop_threads(int(threads))
    _THREAD_LIMIT = threadpool_limits(limits=int(threads))


def _status(cfg, phase, completed, total, running=()):
    value=dict(at=campaign.now(),phase=phase,completed=int(completed),total=int(total),running=list(running),
               test_selection_allowed=False,independently_unused_holdout=False)
    campaign.write_json(Path(cfg['output_dir'])/'status.json',value)
    print(json.dumps(value),flush=True)


def _fit_worker(cfg,task,regsha):
    from .wm_tail_transfer_training import train
    return train(cfg,task,regsha)


def _cache_worker(cfg,job,regsha):
    from .wm_tail_transfer_training import prepare_producer_cache
    return prepare_producer_cache(cfg,job['fold'],job['seed'],regsha,job['sources'],split=job.get('split','train_val'))


def _policy_worker(cfg,job,regsha):
    from .wm_tail_transfer_training import load_producer_cache
    from .wm_tail_transfer_policy import train_cached
    arrays=load_producer_cache(cfg,job['fold'],job['seed'],split='train_val')
    return train_cached(cfg,job,arrays,Path(cfg['output_dir'])/'policies'/job['job'],regsha)


def _stat_fit_worker(cfg,job,regsha):
    from . import wm_tail_transfer_data as data_module
    from .wm_tail_transfer_baselines import fit_statistical_baselines
    folder=Path(cfg['output_dir'])/'statistical'/job['job'];folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'fit.json'
    if marker.exists():
        fit=json.loads(marker.read_text())
        if fit['registration_sha256']!=regsha or campaign.sha(folder/'bundle.pkl')!=fit['artifact_sha256']['bundle.pkl']:
            raise ValueError('statistical fit changed')
        return fit
    data=data_module.load_data(cfg)
    meta=next(x for x in campaign.fold_definitions(cfg)if x['fold']==job['fold'])
    train=campaign.split_ids(data,'D03',meta['train_start'],meta['train_end'])
    val=campaign.split_ids(data,'D03',meta['earlystop_start'],meta['earlystop_end'])
    bundle=fit_statistical_baselines(data,train,val,cfg)
    with (folder/'bundle.pkl').open('wb')as stream:pickle.dump(bundle,stream,protocol=pickle.HIGHEST_PROTOCOL)
    fit=dict(complete=True,task=job,registration_sha256=regsha,test_read=False,
             artifact_sha256={'bundle.pkl':campaign.sha(folder/'bundle.pkl')})
    campaign.write_json(marker,fit)
    return fit


def _forecast_eval_worker(cfg,job,regsha):
    from .wm_tail_transfer_report import evaluate_forecast_job
    return evaluate_forecast_job(cfg,job,regsha)


def _policy_eval_worker(cfg,job,regsha):
    from .wm_tail_transfer_report import evaluate_policy_path
    return evaluate_policy_path(cfg,job,regsha)


def _stat_eval_worker(cfg,job,regsha):
    from .wm_tail_transfer_report import evaluate_statistical_job
    return evaluate_statistical_job(cfg,job,regsha)


def _baseline_eval_worker(cfg,job,regsha):
    from .wm_tail_transfer_report import evaluate_baseline_paths
    return evaluate_baseline_paths(cfg,job,regsha)


def _ensemble_eval_worker(cfg,job,regsha):
    from .wm_tail_transfer_report import evaluate_ensemble_job
    return evaluate_ensemble_job(cfg,job,regsha)


def _rollout_eval_worker(cfg,job,regsha):
    from .wm_tail_transfer_report import evaluate_rollout_job
    return evaluate_rollout_job(cfg,job,regsha)


def parallel_stage(cfg,phase,jobs,function,regsha,*,neural_workers=None,classical_workers=None):
    """Bound shared-MPS workers independently from CPU-native workers."""
    jobs=list(jobs)
    if not jobs:
        _status(cfg,phase,0,0)
        return []
    ncpu=int(classical_workers or cfg['runtime']['classical_workers'])
    ngpu=int(neural_workers or cfg['runtime']['neural_workers'])
    queues={'neural':[],'classical':[]}
    for job in jobs:
        device_task=job.get('source',job)
        if function is _cache_worker:
            device_task={**job,'model':'wm'}
        forecast=(device_task.get('model')in('wm','transformer','lstm')or function in (_cache_worker,_rollout_eval_worker))
        kind='neural' if forecast and campaign.execution_device(cfg,device_task)!='cpu' else 'classical'
        queues[kind].append(job)
    limits={'neural':ngpu,'classical':ncpu}
    results=[];completed=0
    options=dict(mp_context=multiprocessing.get_context('spawn'),initializer=_initialize_worker,
                 initargs=(int(cfg['runtime']['cpu_threads']),))
    with ProcessPoolExecutor(max_workers=ngpu,**options)as gpu, \
            ProcessPoolExecutor(max_workers=ncpu,**options)as cpu:
        pools={'neural':gpu,'classical':cpu};pending={};active={'neural':0,'classical':0}
        def fill():
            for kind in queues:
                while queues[kind]and active[kind]<limits[kind]:
                    job=queues[kind].pop(0)
                    future=pools[kind].submit(function,cfg,job,regsha)
                    pending[future]=(job,kind);active[kind]+=1
        fill();_status(cfg,phase,0,len(jobs),[v[0]['job']for v in pending.values()])
        while pending:
            ready,_=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in ready:
                job,kind=pending.pop(future);active[kind]-=1
                try:results.append(future.result())
                except BaseException:
                    campaign.write_json(Path(cfg['output_dir'])/'failure.json',dict(phase=phase,job=job,at=campaign.now(),traceback=traceback.format_exc()))
                    for remaining in pending:remaining.cancel()
                    raise
                completed+=1
            fill();_status(cfg,phase,completed,len(jobs),[v[0]['job']for v in pending.values()])
    return results


def select_inputs(cfg,jobs):
    """Use only screen validation available before each fold's Test starts."""
    folders=Path(cfg['output_dir'])/'fits'
    folds={x['fold']:x for x in campaign.fold_definitions(cfg)}
    rows=[]
    candidates=['D04','D05','D07_dense']
    for task in jobs:
        if task['role']!='benchmark'or task.get('cohort')!='dense_common':continue
        fit=json.loads((folders/task['job']/'fit.json').read_text())
        score=fit['validation_score']
        if score is None or not np.isfinite(score):continue
        rows.append(dict(group=task['group'],model=task['model'],fold=task['fold'],seed=task['seed'],score=score,
                         validation_end=folds[task['fold']]['val_end']))
    frame=pd.DataFrame(rows);selection={}
    for fold,meta in folds.items():
        past=frame.loc[pd.to_datetime(frame.validation_end,utc=True)<=pd.Timestamp(meta['test_start'])].copy()
        past['rank']=past.groupby(['model','fold','seed']).score.rank(method='average',ascending=True)
        ranks=past[past.group.isin(candidates)].groupby('group')['rank'].mean()
        if len(ranks)!=len(candidates):raise ValueError('every registered dense candidate needs past validation evidence')
        chosen=sorted(ranks.index,key=lambda name:(ranks[name],name))[0]
        selection[str(fold)]=dict(groups=cfg['confirmation']['always_input_groups']+[chosen],selected=chosen,
            eligible_screen_folds=sorted(past.fold.unique().tolist()),validation_rank={k:float(v)for k,v in ranks.items()},
            latest_used_validation_end=past.validation_end.max(),test_start=meta['test_start'])
    campaign.write_json(Path(cfg['output_dir'])/'input_selection.json',dict(selection=selection,Test_used=False))
    return selection


def selected_confirmation_jobs(cfg,selection):
    answer=[]
    for fold in cfg['splits']['folds']:
        answer += [j for j in campaign.confirmation_jobs(cfg,selection[str(fold)]['groups'])if j['fold']==fold]
    return answer


def producer_sources(cfg,jobs):
    by_fold={};out=Path(cfg['output_dir'])
    for fold in cfg['splits']['folds']:
        candidates=[j for j in jobs if j['role']=='producer'and j['fold']==fold and j['model']in('linear','lightgbm')]
        forecast=min(candidates,key=lambda j:float(json.loads((out/'fits'/j['job']/'fit.json').read_text())['earlystop_score']))
        all_tail={j['seed']:j for j in jobs if j['role']=='producer'and j['fold']==fold and j.get('representation')=='tail'}
        all_old={j['seed']:j for j in jobs if j['role']=='producer'and j['fold']==fold and j.get('representation')=='old'}
        for seed in cfg['seeds']:
            by_fold[(fold,seed)]=dict(old=all_old[seed],tail=all_tail[seed],forecast=forecast,
                ensemble=[all_tail[s]for s in cfg['ablation']['ensemble_members']])
    serial={f'{f}:{s}':v for(f,s),v in by_fold.items()}
    campaign.write_json(out/'producer_selection.json',dict(sources=serial,selection='producer first-Val-month native score only',Test_used=False))
    return by_fold


def probe_jobs(cfg,sources):
    result=[]
    for fold in cfg['splits']['folds']:
        for seed in cfg['seeds']:
            source=sources[(fold,seed)]['old']
            result.append(campaign.make_job('ablation','D03','wm',fold,seed,architecture='frozen_probe',source_job=source['job'],item='W03',cohort=cfg['ablation']['cohort']))
            for model in ('linear','lightgbm'):
                result.append(campaign.make_job('ablation','D03',model,fold,seed,architecture='latent_probe',source_job=source['job'],item='W02',cohort=cfg['ablation']['cohort']))
    return result


def path_specs(cfg,tasks,sources):
    groups=defaultdict(list)
    for task in tasks:
        key=json.dumps({k:v for k,v in task.items()if k not in('job','fold')},sort_keys=True)
        groups[key].append(task)
    answer=[]
    for key,group in groups.items():
        group.sort(key=lambda x:x['fold'])
        if (sorted(j['fold'] for j in group)!=sorted(cfg['splits']['folds'])
                or len({j['job'] for j in group})!=len(group)):
            raise ValueError('full unique fold coverage required before continuous replay')
        condition=json.loads(key)
        for stage in ('BC','RL'):
            digest=__import__('hashlib').sha256((key+stage).encode()).hexdigest()[:16]
            answer.append(dict(job=f'path_{stage}_{digest}',tasks=group,stage=stage,condition=condition,
                sources={str(j['fold']):sources[(j['fold'],j['seed'])]for j in group},sensitivity=None))
        if condition['control']['account_mode']=='linear_perp':
            for mmr in cfg['perpetual']['maintenance_margin_rates']:
                for slip in cfg['perpetual']['slippage_rates']:
                    if mmr==cfg['perpetual']['primary_maintenance_margin_rate']and slip==cfg['perpetual']['primary_slippage_rate']:continue
                    sensitivity=dict(mmr=mmr,slippage=slip)
                    digest=__import__('hashlib').sha256((key+json.dumps(sensitivity,sort_keys=True)).encode()).hexdigest()[:16]
                    answer.append(dict(job=f'path_sensitivity_{digest}',tasks=group,stage='RL',condition=condition,
                        sources={str(j['fold']):sources[(j['fold'],j['seed'])]for j in group},sensitivity=sensitivity))
    return answer


def _freeze_barrier(cfg,regsha,forecast_jobs,policy_jobs,stat_jobs,paths):
    from .wm_tail_transfer_report import comparison_plan, baseline_path_specs, ensemble_jobs, rollout_jobs
    out=Path(cfg['output_dir']);frozen={}
    if (out/'test_barrier.json').exists():
        return campaign.verify_test_barrier(cfg)
    for directory,jobs in [('fits',forecast_jobs),('policies',policy_jobs),('statistical',stat_jobs)]:
        for task in jobs:
            path=out/directory/task['job']/'fit.json';fit=json.loads(path.read_text())
            if not fit.get('complete')or fit.get('registration_sha256')!=regsha:
                raise ValueError('all declared Train/Val fits must complete before any Test inference')
            frozen[str(path.relative_to(out))]=campaign.sha(path)
            for relative,expected in fit['artifact_sha256'].items():
                artifact=path.parent/relative
                if campaign.sha(artifact)!=expected:raise ValueError('selected fit artifact changed')
                frozen[str(artifact.relative_to(out))]=expected
    # Producer calibration references remain part of the Train/Val state.
    # Bind the manifests themselves as well as every referenced cache array.
    for pattern in ('producer_caches/*/train_val/manifest.json',
                    'latent_probe_features/*/train_val/manifest.json'):
        for marker in sorted(out.glob(pattern)):
            manifest=json.loads(marker.read_text())
            if manifest.get('registration_sha256')!=regsha:
                raise ValueError('foreign Train/Val feature cache at Test barrier')
            frozen[str(marker.relative_to(out))]=campaign.sha(marker)
            for name,expected in manifest['artifact_sha256'].items():
                artifact=marker.parent/name
                if campaign.sha(artifact)!=expected:
                    raise ValueError('Train/Val feature cache changed before Test barrier')
                frozen[str(artifact.relative_to(out))]=expected
    plan=comparison_plan(cfg,forecast_jobs,paths)
    campaign.write_json(out/'comparison_plan.json',plan)
    campaign.write_json(out/'execution_manifest.json',dict(forecast_jobs=forecast_jobs,policy_tasks=policy_jobs,statistical_jobs=stat_jobs,path_specs=paths,
        baseline_specs=baseline_path_specs(cfg,paths),ensemble_specs=ensemble_jobs(cfg,forecast_jobs),rollout_specs=rollout_jobs(cfg,forecast_jobs)))
    campaign.verify_registration(cfg,full=True)
    barrier=dict(registration_sha256=regsha,at=campaign.now(),fit_sha256=frozen,
        fit_file_identity={relative:campaign.file_identity(out/relative)for relative in frozen},
        comparison_plan_sha256=campaign.sha(out/'comparison_plan.json'),execution_manifest_sha256=campaign.sha(out/'execution_manifest.json'),
        input_selection_sha256=campaign.sha(out/'input_selection.json'),producer_selection_sha256=campaign.sha(out/'producer_selection.json'),
        Test_selection_allowed=False,known_historical_Test=True)
    campaign.write_json(out/'test_barrier.json',barrier)
    return barrier


def _deduplicate(tasks):
    result={}
    for task in tasks:
        if task['job']in result and result[task['job']]!=task:
            raise ValueError('conflicting duplicate task identifier')
        result[task['job']]=task
    return list(result.values())


def run(cfg):
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'completion.json').exists():
        campaign.verify_test_barrier(cfg)
        print((out/'completion.json').read_text(),flush=True)
        return
    regsha=campaign.preregister(cfg)
    first=campaign.screen_jobs(cfg)+campaign.producer_jobs(cfg)+campaign.ablation_jobs(cfg)
    campaign.write_json(out/'initial_job_manifest.json',dict(registration_sha256=regsha,jobs=first))
    parallel_stage(cfg,'forecast_screen_producers_ablations',first,_fit_worker,regsha)
    selection=select_inputs(cfg,campaign.screen_jobs(cfg))
    confirm=selected_confirmation_jobs(cfg,selection)
    existing={j['job']for j in first}
    parallel_stage(cfg,'forecast_confirmation',[j for j in confirm if j['job']not in existing],_fit_worker,regsha)
    sources=producer_sources(cfg,campaign.producer_jobs(cfg))
    probes=probe_jobs(cfg,sources)
    parallel_stage(cfg,'frozen_representation_probes',probes,_fit_worker,regsha)
    all_forecasts=_deduplicate(first+confirm+probes)
    stat_jobs=[dict(job=f'statistical_f{f:02d}',fold=f,role='statistical')for f in cfg['splits']['folds']]
    parallel_stage(cfg,'statistical_TrainVal',stat_jobs,_stat_fit_worker,regsha,classical_workers=7)
    cache_jobs=[dict(job=f'producer_context_f{f:02d}_s{s}',fold=f,seed=s,sources=src,split='train_val')for(f,s),src in sources.items()]
    parallel_stage(cfg,'producer_TrainVal_contexts',cache_jobs,_cache_worker,regsha)
    policies=campaign.policy_tasks(cfg)
    parallel_stage(cfg,'BC_and_RL_TrainVal',policies,_policy_worker,regsha,classical_workers=cfg['runtime']['policy_workers'])
    paths=path_specs(cfg,policies,sources)
    _freeze_barrier(cfg,regsha,all_forecasts,policies,stat_jobs,paths)
    parallel_stage(cfg,'producer_Test_contexts',[{**j,'job':j['job']+'_test','split':'test'}for j in cache_jobs],_cache_worker,regsha)
    parallel_stage(cfg,'forecast_Test_report_only',all_forecasts,_forecast_eval_worker,regsha)
    from .wm_tail_transfer_report import baseline_path_specs, ensemble_jobs, rollout_jobs
    ensembles=ensemble_jobs(cfg,all_forecasts)
    parallel_stage(cfg,'forecast_seed_ensemble_Test_report_only',ensembles,_ensemble_eval_worker,regsha)
    rollouts=rollout_jobs(cfg,all_forecasts)
    parallel_stage(cfg,'WM_generated_path_Test_report_only',rollouts,_rollout_eval_worker,regsha)
    parallel_stage(cfg,'statistical_Test_report_only',stat_jobs,_stat_eval_worker,regsha,classical_workers=7)
    baselines=baseline_path_specs(cfg,paths)
    parallel_stage(cfg,'execution_baselines_Test_report_only',baselines,_baseline_eval_worker,regsha,classical_workers=cfg['runtime']['policy_workers'])
    parallel_stage(cfg,'policy_Test_continuous_report_only',paths,_policy_eval_worker,regsha,classical_workers=cfg['runtime']['policy_workers'])
    from .wm_tail_transfer_report import aggregate_report
    report=aggregate_report(cfg,all_forecasts,paths,regsha)
    campaign.verify_test_barrier(cfg,full=True)
    campaign.verify_registration(cfg,full=True)
    actual_fits=[json.loads((out/'fits'/job['job']/'fit.json').read_text())for job in all_forecasts]
    completed=dict(complete=True,at=campaign.now(),registration_sha256=regsha,forecast_training_fits=sum(fit.get('available',True)for fit in actual_fits),
        forecast_registered_conditions=len(all_forecasts),forecast_unavailable_data=sum(not fit.get('available',True)for fit in actual_fits),
        policy_training_fits=len(policies),statistical_training_fits=len(stat_jobs),continuous_policy_paths=len(paths),
        fixed_ensemble_outputs=len(ensembles),generated_path_diagnostics=len(rollouts),execution_baseline_environments=len(baselines),
        Test_selection_changed=False,previous_sources_unchanged=True,known_historical_Test=True,report=report)
    campaign.write_json(out/'completion.json',completed)
    _status(cfg,'complete',len(all_forecasts)+len(policies),len(all_forecasts)+len(policies))
    print(json.dumps(completed),flush=True)
