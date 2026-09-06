"""No-fit development-only matched-support hindsight information diagnostics."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from .alpha_dd_search import digest, file_digest, metrics
from .oracle_confirmation_contract import calendar
from .oracle_conditional_planner import conditional_targets
from .oracle_derivative_crossed_decisions import _immutable_json
from .oracle_fallback_planner import fallback_targets
from .oracle_frozen_procedure_parity import (
    FOLDS, MEANS, POLICIES as CONTROLS, RULES, SOURCES as PARITY_SOURCES,
    compare_array, compare_tree, prepare as prepare_parity,
)
from .oracle_information_interventions import SWAPS, mark_hindsight_trace, substitute_information
from .oracle_matched_hindsight import matched_hindsight_targets


HALVES=("technical_half","perp_delay0_half")
HYBRIDS=tuple(m+"_oracle_"+swap for m in HALVES for swap in SWAPS)
RL_RULES=("hold","fallback_bh")
RL_IDS=tuple(f"matched_rl_beam32_{rule}_risk{p}" for rule in RL_RULES for p in (0,1))
ORACLES=tuple(m+"_"+rule for m in HYBRIDS for rule in RULES)+RL_IDS
POLICIES=CONTROLS+ORACLES
SOURCE_ROOT=Path("codex_outputs/oracle_frozen_procedure_parity_v1")
SOURCES=tuple("unidream/experiments/"+n for n in dict.fromkeys((
    "oracle_information_decomposition.py","oracle_information_interventions.py",
    "oracle_matched_hindsight.py","oracle_frontier_hindsight.py",*PARITY_SOURCES)))
FIXED={"schema":"oracle-information-decomposition-v1","development_validation_folds":list(FOLDS),
    "data_cutoff":"2023-04-16T13:45:00Z","source_root":str(SOURCE_ROOT),
    "parity_config":"configs/oracle_frozen_procedure_parity_20260906.yaml",
    "output_dir":"codex_outputs/oracle_information_decomposition_v1",
    "half_means":list(HALVES),"swaps":list(SWAPS),"rules":list(RULES),
    "control_ids":list(CONTROLS),"diagnostic_ids":list(ORACLES),
    "replacement_support":"existing_saved_score_support_only_keep_learned_elsewhere",
    "utility_risk_aversion":1,"utility_cost_multiplier":2,"horizon_bars":24,
    "rl_beam_width":32,"rl_risk_penalties":[0.,1.],"rl_missing_input_rules":list(RL_RULES),
    "new_model_fitting_permitted":False,"selection_permitted":False,"teacher_use_allowed":False,
    "test_or_additional_periods_permitted":False,"new_causal_candidate_count":0}


def validate_config(cfg):
    if (set(cfg)!=set(FIXED)|{"source_bindings","parity_config_sha256","source_manifest_bindings","preflight_sha256"}
            or any(type(cfg.get(k)) is not type(v) or cfg[k]!=v for k,v in FIXED.items())
            or set(cfg["source_bindings"])!=set(SOURCES)):
        raise ValueError("unregistered Oracle decomposition family")
    expected={str(SOURCE_ROOT/p) for p in ("registration.json","preflight.json","results.json")}
    expected|={str(SOURCE_ROOT/f"fold_{f}.json") for f in FOLDS}
    if set(cfg["source_manifest_bindings"])!=expected:raise ValueError("incomplete parent manifest inventory")


def prepare(config_path):
    cfg=yaml.safe_load(Path(config_path).read_text());validate_config(cfg)
    for p,h in {**cfg["source_bindings"],**cfg["source_manifest_bindings"],cfg["parity_config"]:cfg["parity_config_sha256"]}.items():
        if file_digest(Path(p))!=h:raise ValueError("registered source/config/manifest changed: "+p)
    source_reg=json.loads((SOURCE_ROOT/"registration.json").read_text())
    source_result=json.loads((SOURCE_ROOT/"results.json").read_text())
    if source_result["registration_sha256"]!=digest(source_reg) or source_result["parity_pass"] is not True:
        raise ValueError("parent parity not complete")
    parent=prepare_parity(Path(cfg["parity_config"]))
    pc,dc,fc,bars,groups,original,y,masks,pp,old_rows,old_scores,ancestors=parent
    if (source_reg["config"]!=pc or source_reg["config_sha256"]!=cfg["parity_config_sha256"]
            or source_reg["preflight_sha256"]!=file_digest(SOURCE_ROOT/"preflight.json")
            or pp!=json.loads((SOURCE_ROOT/"preflight.json").read_text())):
        raise ValueError("parent registration/config/preflight chain changed")
    for name,expected_sha in source_reg["source_sha256"].items():
        if file_digest(Path(__file__).with_name(name))!=expected_sha:raise ValueError("parent registered source changed")
    if fc["data_cutoff"]!=cfg["data_cutoff"]:raise ValueError("development cutoff changed")
    rows={(r["fold"],r["candidate_id"]):r for r in source_result["rows"]}
    if len(rows)!=96 or set(rows)!={(f,c) for f in FOLDS for c in CONTROLS}:raise ValueError("parent control family incomplete")
    bindings=dict(ancestors);forecasts={};support=[]
    for f in FOLDS:
        fold=json.loads((SOURCE_ROOT/f"fold_{f}.json").read_text())
        if (fold["registration_sha256"]!=digest(source_reg) or len(fold["artifact_sha256"])!=33
                or fold["rows"]!=[r for r in source_result["rows"] if r["fold"]==f]):
            raise ValueError("parent fold family inconsistent")
        for p,h in fold["artifact_sha256"].items():
            if file_digest(Path(p))!=h:raise ValueError("parent artifact changed")
            if p in bindings and bindings[p]!=h:raise ValueError("conflicting parent artifact")
            bindings[p]=h
        dates=calendar(f-1);ix=np.asarray((bars.index>=dates["evaluation_start"])&(bars.index<dates["evaluation_end"]))
        expected=y[ix].copy();expected[~masks[f]["score"][ix]]=np.nan
        for mean in MEANS:
            p=SOURCE_ROOT/"forecasts"/f"fold{f}_{mean}.npz"
            with np.load(p,allow_pickle=False) as saved:
                if set(saved.files)!={"timestamps","mu","variance","actual","score_support","inference_mask","fit_return_mean"}:
                    raise ValueError("parent forecast schema changed")
                a={k:saved[k] for k in saved.files}
            for k,v in {"timestamps":bars.index[ix].asi8,"actual":expected,
                    "score_support":masks[f]["score"][ix],"inference_mask":masks[f]["inference"][ix]}.items():
                compare_array(a[k],v,name="parent "+k,exact=True)
            forecasts[f,mean]=a
        reference=forecasts[f,HALVES[0]];infer=reference["inference_mask"];score=reference["score_support"]
        if any(rows[f,c]["regime"]!=old_rows[f,c]["regime"] for c in CONTROLS):raise ValueError("parent regime changed")
        support.append({"fold":f,"evaluation_start":dates["evaluation_start"].isoformat(),
            "evaluation_end":dates["evaluation_end"].isoformat(),"regime":rows[f,CONTROLS[0]]["regime"],
            "inference_rows":int(infer.sum()),"replacement_rows":int(score.sum()),
            "learned_remainder_rows":int((infer&~score).sum()),
            "current_open_missing_inference_rows":int((infer&~np.isfinite(bars.open.to_numpy()[ix])).sum())})
    pre={"schema":"oracle-information-decomposition-preflight-v1",
        "config_contract_sha256":digest({k:v for k,v in cfg.items() if k!="preflight_sha256"}),
        "source_bindings":cfg["source_bindings"],"source_artifact_bindings":bindings,
        "source_manifest_bindings":cfg["source_manifest_bindings"],"support":support,
        "spot_data_proof":pp["spot_data_proof"],"um_data_proof":pp["um_data_proof"],
        "scope":"No-fit, no-policy data-only preflight on original reused development8quarters",
        "new_policy_computed":False,"new_causal_candidate_count":0,"test_results_used_for_selection":False}
    return cfg,fc,bars,forecasts,rows,pre


def summarize(rows):
    index={(r["fold"],r["candidate_id"]):r for r in rows}
    if len(index)!=len(rows) or set(index)!={(f,c) for f in FOLDS for c in POLICIES}:
        raise ValueError("incomplete diagnostic family")
    keys=("alpha_ex","maxdd_delta","turnover","trades")
    def average(values):return math.fsum(v/len(values) for v in values) if values else None
    result={"policies":{},"oracle_minus_own_learned":{}}
    for cid in POLICIES:
        result["policies"][cid]={}
        for regime in ("all","bull","bear","sideways"):
            selected=[r for r in rows if r["candidate_id"]==cid and (regime=="all" or r["regime"]["trend"]==regime)]
            result["policies"][cid][regime]={"quarters":len(selected),**{
                cost:{k:average([r[cost][k] for r in selected]) for k in keys} for cost in ("base","stress_2x")}}
    for half in HALVES:
        for swap in SWAPS:
            for rule in RULES:
                cid=half+"_oracle_"+swap+"_"+rule;ref=half+"_"+rule
                result["oracle_minus_own_learned"][cid]={"reference_id":ref,"strata":{}}
                for regime in ("all","bull","bear","sideways"):
                    fs=[f for f in FOLDS if regime=="all" or index[f,cid]["regime"]["trend"]==regime]
                    result["oracle_minus_own_learned"][cid]["strata"][regime]={"quarters":len(fs),**{
                        cost:{k:average([index[f,cid][cost][k]-index[f,ref][cost][k] for f in fs]) for k in keys}
                        for cost in ("base","stress_2x")}}
    result.update(scope="Hindsight information intervention only, not predictive performance or a global upper bound",
        new_causal_candidate_count=0,selection_performed=False,teacher_use_allowed=False,
        high_probability_generalization_established=False)
    return result


def run(config_path):
    cfg,fc,bars,forecasts,controls,pre=prepare(config_path)
    output=Path(cfg["output_dir"])
    if (output/"results.json").exists():raise ValueError("immutable information diagnostic already completed")
    if (file_digest(output/"preflight.json")!=cfg["preflight_sha256"]
            or json.loads((output/"preflight.json").read_text())!=pre):raise ValueError("registered preflight changed")
    registration={"config":cfg,"config_sha256":file_digest(config_path),"preflight_sha256":cfg["preflight_sha256"],
        "source_bindings":cfg["source_bindings"],"source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "scope":"Original development8:12 fixed information hybrids+4 matched hindsight beams+12 causal controls; no fits"}
    _immutable_json(output/"registration.json",registration)
    execution=fc["execution"];stress={**execution,"one_way_cost":2*execution["one_way_cost"],"borrow_annual":2*execution["borrow_annual"]}
    all_rows=[]
    for f in FOLDS:
        fold_path=output/f"fold_{f}.json"
        dates=calendar(f-1);ix=np.asarray((bars.index>=dates["evaluation_start"])&(bars.index<dates["evaluation_end"]))
        window=bars.loc[ix];regime=controls[f,CONTROLS[0]]["regime"]
        rows,bindings=[],{}
        def save_npz(kind,name,arrays):
            p=output/kind/f"fold{f}_{name}.npz";p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                with np.load(p,allow_pickle=False) as prior:
                    if set(prior.files)!=set(arrays):raise ValueError("partial artifact schema changed")
                    for k,v in arrays.items():compare_array(v,prior[k],name=str(p)+k,exact=True)
            else:np.savez_compressed(p,**arrays)
            bindings[str(p)]=file_digest(p);return p
        def evaluate(cid,targets,diagnostic=None):
            p=save_npz("targets",cid,{"timestamps":window.index.asi8,"targets":targets})
            row={"fold":f,"candidate_id":cid,"regime":regime,"hindsight_only":cid in ORACLES,
                **{cost:metrics(window,targets,c) for cost,c in (("base",execution),("stress_2x",stress))},
                "targets_sha256":bindings[str(p)]}
            if diagnostic is not None:
                p=output/"traces"/f"fold{f}_{cid}.json";_immutable_json(p,diagnostic)
                bindings[str(p)]=file_digest(p);row["diagnostic_sha256"]=bindings[str(p)]
            rows.append(row)
        for cid in CONTROLS:
            p=SOURCE_ROOT/"targets"/f"fold{f}_{cid}.npz"
            with np.load(p,allow_pickle=False) as saved:
                compare_array(saved["timestamps"],window.index.asi8,name="control calendar",exact=True)
                evaluate(cid,saved["targets"])
            for cost in ("base","stress_2x"):compare_tree(rows[-1][cost],controls[f,cid][cost],name="unchanged causal control")
        for half in HALVES:
            original=forecasts[f,half]
            for swap in SWAPS:
                hybrid=half+"_oracle_"+swap
                intervention=substitute_information(original["mu"],original["variance"],
                    inference_mask=original["inference_mask"],score_support=original["score_support"],actual=original["actual"],swap=swap)
                save_npz("forecasts",hybrid,{**original,"mu":intervention["mu"],"variance":intervention["variance"]})
                for rule in RULES:
                    cid=hybrid+"_"+rule
                    if rule==RULES[1]:
                        target,trace=fallback_targets(window,intervention["mu"],intervention["variance"],execution,
                            inference_mask=original["inference_mask"],risk_aversion=1,cost_multiplier=2)
                    else:target,trace=conditional_targets(window,intervention["mu"],intervention["variance"],execution,risk_aversion=1,cost_multiplier=2)
                    trace=mark_hindsight_trace(trace,swap=swap,score_support=original["score_support"])
                    trace["information_intervention"]=intervention["metadata"]
                    evaluate(cid,target,trace)
        for rule in RL_RULES:
            for penalty in cfg["rl_risk_penalties"]:
                cid=f"matched_rl_beam32_{rule}_risk{penalty:g}"
                targets,trace=matched_hindsight_targets(window,execution,decision_support=forecasts[f,HALVES[0]]["inference_mask"],
                    beam_width=32,risk_penalty=penalty,missing_input_rule=rule)
                evaluate(cid,targets,trace)
        saved={"registration_sha256":digest(registration),"rows":rows,"artifact_sha256":bindings}
        # Deterministic replay compares every artifact and complete payload if
        # resuming, rather than trusting an incomplete saved manifest.
        _immutable_json(fold_path,saved);all_rows.extend(rows)
        print(json.dumps({"event":"fold_complete","fold":f,"rows":len(rows),"artifacts":len(bindings)}),flush=True)
    result={"registration_sha256":digest(registration),"rows":all_rows,"summary":summarize(all_rows),
        "new_models_fitted":0,"new_causal_candidates":0,"hindsight_only_diagnostics":len(ORACLES),
        "selection_performed":False,"test_periods_used":False,"teacher_use_allowed":False}
    _immutable_json(output/"results.json",result);return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true");args=parser.parse_args()
    if args.preflight:
        cfg,*_,pre=prepare(args.config);p=Path(cfg["output_dir"])/"preflight.json";_immutable_json(p,pre)
        print(json.dumps({"path":str(p),"sha256":file_digest(p),"new_policy_computed":False}))
    else:run(args.config)
