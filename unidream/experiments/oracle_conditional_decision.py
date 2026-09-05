"""Registered decision-value diagnostic for frozen calibrated forecasts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .alpha_dd_search import digest,file_digest,load_bars,metrics,validate_data_artifact,write_json
from .oracle_frontier import summarize
from .oracle_conditional_planner import conditional_targets


def run(config_path):
    cfg=yaml.safe_load(config_path.read_text())
    fc=yaml.safe_load(Path(cfg["source_config"]).read_text())
    root=Path(cfg["calibration_root"]);out=Path(cfg["output_dir"])
    if (out/"results.json").exists():raise ValueError("immutable results already exist")
    source=json.loads((root/"results.json").read_text())
    registration={"config":cfg,"source_config":fc,
        "source_sha256":{p.name:file_digest(p) for p in [Path(__file__),Path(conditional_targets.__code__.co_filename),
            Path(metrics.__code__.co_filename),Path(summarize.__code__.co_filename)]},
        "data_proof":validate_data_artifact(Path(fc["data_path"]),expected_symbol=fc["symbol"]),
        "calibration_registration_sha256":file_digest(root/"registration.json"),
        "calibration_results_sha256":file_digest(root/"results.json"),
        "scope":"estimated one-step conditional utility; adaptive reused validation; not exact/global oracle"}
    inputs={}
    for score in source["scores"]:
        if score["model_id"] not in cfg["forecast_models"]:continue
        p=root/"forecasts"/f"fold{score['fold']}_{score['model_id']}.npz"
        if file_digest(p)!=score["forecast_sha256"]:raise ValueError("forecast changed")
        inputs[str(p)]=score["forecast_sha256"]
    registration["forecast_bindings"]=inputs
    write_json(out/"registration.json",registration)
    bars=load_bars(Path(fc["data_path"]),cutoff=fc["data_cutoff"])
    contract=fc["execution"];stress={**contract,"one_way_cost":2*contract["one_way_cost"],"borrow_annual":2*contract["borrow_annual"]}
    score_map={(s["fold"],s["model_id"]):s for s in source["scores"]}
    rows=[r for r in source["rows"] if r["candidate_id"] in ["bh","common_robust"]+[m+"_point" for m in cfg["forecast_models"]]]
    for fold in fc["development_folds"]:
        for model in cfg["forecast_models"]:
            saved=np.load(root/"forecasts"/f"fold{fold}_{model}.npz")
            index=pd.to_datetime(saved["timestamps"],utc=True)
            if not index.isin(bars.index).all():raise ValueError("forecast timestamp outside source support")
            window=bars.loc[index]
            mu=saved["mu"].copy();variance=saved["variance"].copy()
            mu[~saved["inference_mask"]]=np.nan;variance[~saved["inference_mask"]]=np.nan
            for risk in cfg["risk_aversions"]:
                targets,diagnostics=conditional_targets(window,mu,variance,contract,risk_aversion=risk,cost_multiplier=cfg["cost_multiplier"])
                cid=f"{model}_utility_risk{risk:g}"
                path=out/"targets"/f"fold{fold}_{cid}.npz";path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,targets=targets,timestamps=window.index.asi8)
                rows.append({"fold":fold,"candidate_id":cid,"regime":score_map[fold,model]["regime"],
                    "base":metrics(window,targets,contract),"stress_2x":metrics(window,targets,stress),
                    "targets_sha256":file_digest(path),"diagnostics":diagnostics})
        print(json.dumps({"fold":fold,"completed":True}),flush=True)
    result={"registration_sha256":digest(registration),"rows":rows,"summary":summarize(rows,3),
        "scope":registration["scope"],"high_probability_generalization_established":False}
    write_json(out/"results.json",result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
