"""Fixed second-stage risk forecast ablation on reused development validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .alpha_dd_features import make_features
from .alpha_dd_search import file_digest, load_bars, metrics, write_json
from .oracle_frontier import summarize


def risk_targets(backbone, prediction_vol, trailing_vol_annual, *, strength,
                 horizon=24, ratio_floor=.5, ratio_ceiling=1.5):
    backbone = np.asarray(backbone, float)
    prediction_vol = np.asarray(prediction_vol, float)
    trailing_vol_annual = np.asarray(trailing_vol_annual, float)
    if not (backbone.shape == prediction_vol.shape == trailing_vol_annual.shape):
        raise ValueError("aligned causal inputs required")
    if not 0 < strength <= 1 or horizon < 1:
        raise ValueError("invalid fixed risk controller")
    out = backbone.copy()
    available = np.isfinite(backbone) & np.isfinite(prediction_vol) & np.isfinite(trailing_vol_annual)
    available &= trailing_vol_annual >= 0
    ratio = np.clip(trailing_vol_annual[available] * np.sqrt(horizon / 35040.) /
                    np.maximum(prediction_vol[available], .001), ratio_floor, ratio_ceiling)
    out[available] = np.clip(backbone[available] * ratio**strength, .5, 1.12)
    return out


def run(config_path: Path):
    config = yaml.safe_load(config_path.read_text())
    fc = yaml.safe_load(Path(config["frontier_config"]).read_text())
    root = Path(config["frontier_results"]).parent
    source = json.loads(Path(config["frontier_results"]).read_text())
    output = Path(config["output_dir"])
    if (output / "results.json").exists():
        raise ValueError("immutable result already exists")
    proof = {"config": config, "config_sha256": file_digest(config_path),
             "source_sha256": file_digest(Path(__file__)),
             "frontier_result_sha256": file_digest(Path(config["frontier_results"])),
             "frontier_registration_sha256": file_digest(Path(config["frontier_registration"])),
             "evaluation_source_sha256": file_digest(Path(metrics.__code__.co_filename)),
             "summary_source_sha256": file_digest(Path(summarize.__code__.co_filename)),
             "feature_source_sha256": file_digest(Path(make_features.__code__.co_filename)),
             "scope": "adaptive reused validation; not independent confirmation"}
    forecasts = {(x["fold"], x["model_id"]): x for x in source["forecast_scores"]}
    targets = {(x["fold"], x["candidate_id"]): x for x in source["rows"]}
    bindings = {}
    for fold in fc["development_folds"]:
        for group in config["forecast_groups"]:
            mid = f"{group}_hgb_h24"
            path = root / "forecasts" / f"fold{fold}_{mid}.npz"
            expected = forecasts[fold, mid]["provenance"]["predictions_sha256"]
            if file_digest(path) != expected:
                raise ValueError("source forecast changed")
            bindings[str(path)] = expected
        for base in config["base_policies"]:
            path = root / "targets" / f"fold{fold}_{base}.npz"
            expected = targets[fold, base]["targets_sha256"]
            if file_digest(path) != expected:
                raise ValueError("source backbone changed")
            bindings[str(path)] = expected
    proof["input_bindings"] = bindings
    write_json(output / "registration.json", proof)
    bars = load_bars(Path(fc["data_path"]), cutoff=fc["data_cutoff"])
    features = make_features(bars)
    execution = fc["execution"]
    stress = {**execution, "one_way_cost": execution["one_way_cost"] * 2,
              "borrow_annual": execution["borrow_annual"] * 2}
    rows = []
    for fold in fc["development_folds"]:
        for base in config["base_policies"]:
            source_row = targets[fold, base]
            rows.append(source_row)
            saved = np.load(root / "targets" / f"fold{fold}_{base}.npz")
            ix = (bars.index >= source_row["validation_start"]) & (bars.index < source_row["validation_end"])
            window = bars.loc[ix]
            if not np.array_equal(saved["timestamps"], window.index.asi8):
                raise ValueError("backbone timestamps disagree")
            for group in config["forecast_groups"]:
                pred = np.load(root / "forecasts" / f"fold{fold}_{group}_hgb_h24.npz")
                if not np.array_equal(saved["timestamps"], pred["timestamps"]):
                    raise ValueError("forecast timestamps disagree")
                for strength in config["strengths"]:
                    cid = f"{base}__risk_{group}_s{strength:g}"
                    intent = risk_targets(saved["targets"], pred["predictions"][:, 2],
                        features.vol_7.to_numpy()[ix], strength=strength,
                        horizon=config["horizon_bars"], ratio_floor=config["ratio_floor"],
                        ratio_ceiling=config["ratio_ceiling"])
                    path = output / "targets" / f"fold{fold}_{cid}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, targets=intent, timestamps=window.index.asi8)
                    rows.append({"fold": fold, "candidate_id": cid, "regime": source_row["regime"],
                        "validation_start": source_row["validation_start"],
                        "validation_end": source_row["validation_end"], "hindsight_only": False,
                        "targets_sha256": file_digest(path),
                        "base": metrics(window, intent, execution),
                        "stress_2x": metrics(window, intent, stress)})
    summaries = summarize(rows, config["minimum_quarters_per_regime"])
    ranking = sorted([c for c in summaries if "__risk_" in c],
                     key=lambda k: (-summaries[k]["worst_regime_score"], k))
    result = {"registration_sha256": file_digest(output / "registration.json"),
              "summary": summaries, "rows": rows, "ranking": ranking,
              "candidate_count": len(ranking), "scope": proof["scope"],
              "high_probability_generalization_established": False}
    write_json(output / "results.json", result)
    print(json.dumps({"candidates": len(ranking), "selected": ranking[0],
                      "direction_pass": [k for k in ranking if summaries[k]["direction_pass"]],
                      "regime_pass": [k for k in ranking if summaries[k]["exploratory_regime_direction_pass"]]}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
