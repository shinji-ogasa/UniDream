"""Development-only outcome prediction / executable hindsight frontier.

Past validation quarters are explicitly reused exploratory evidence. No test
stage exists. Hindsight diagnostics are never inputs to fitted models.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .alpha_dd_search import (aggregate, digest, file_digest, fold_spec,
                              load_bars, metrics, validate_data_artifact, write_json)
from .oracle_frontier_features import make_feature_groups
from .robust_overlay import build_targets

OUTCOMES = ("return", "adverse_excursion", "realized_volatility")


def outcome_frame(bars: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Decision t -> fill open t+1 -> marked closes t+1 through t+h.

    Full future support is required for training/scoring only. It never decides
    whether a causal policy may issue an order.
    """
    if horizon < 1:
        raise ValueError("positive horizon required")
    close = bars.close.astype(float)
    entry = bars.open.shift(-1).astype(float)
    terminal = close.shift(-horizon)
    future_min = close.rolling(horizon).min().shift(-horizon)
    # t+1 starts at its open; later increments run between marked closes.
    squared = np.log(close / close.shift(1)).pow(2)
    intermediate = squared.rolling(horizon - 1).sum().shift(-horizon) if horizon > 1 else 0.0
    total_sq = np.log(close.shift(-1) / entry).pow(2) + intermediate
    available = bars.bar_available.astype(int).rolling(horizon).sum().shift(-horizon).eq(horizon)
    y = pd.DataFrame({
        "return": np.log(terminal / entry),
        "adverse_excursion": np.maximum(-np.log(future_min / entry), 0.0),
        "realized_volatility": np.sqrt(total_sq.clip(lower=0)),
    }, index=bars.index)
    return y.where(available & entry.notna() & (entry > 0), np.nan)


def map_outcomes(prediction: np.ndarray, mapper: str) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    if prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("three outcome columns required")
    mu = prediction[:, 0].copy()
    if mapper == "downside":
        mu -= .25 * np.maximum(prediction[:, 1], 0)
    elif mapper != "return":
        raise ValueError(mapper)
    signal = np.tanh(2 * mu / np.maximum(prediction[:, 2], .001))
    target = 1 + .12 * np.maximum(signal, 0) + .50 * np.minimum(signal, 0)
    target[~np.isfinite(prediction).all(axis=1)] = np.nan
    return target


def fit_mask(index: pd.DatetimeIndex, complete: np.ndarray, *,
             start: pd.Timestamp, end: pd.Timestamp, horizon: int,
             cadence_hours: int) -> np.ndarray:
    # Bar-open t+h's close is only known at t+h+1; strict purge before boundary.
    outcome_end = index + pd.Timedelta(minutes=15 * (horizon + 1))
    schedule = (index.hour % cadence_hours == 0) & (index.minute == 0)
    return np.asarray((index >= start) & (outcome_end < end) & schedule & complete)


def forecast_scores(y: np.ndarray, pred: np.ndarray, climatology: np.ndarray) -> dict:
    valid = np.isfinite(y).all(axis=1) & np.isfinite(pred).all(axis=1)
    y, pred = y[valid], pred[valid]
    if len(y) < 16:
        return {"rows": len(y), "status": "insufficient"}
    skill = {}
    for k, name in enumerate(OUTCOMES):
        mse = float(np.mean((y[:, k] - pred[:, k]) ** 2))
        reference = float(np.mean((y[:, k] - climatology[k]) ** 2))
        skill[name] = {"mse": mse, "climatology_mse": reference,
                       "mse_skill": 1 - mse / reference if reference > 0 else None}
    ic = (float(spearmanr(y[:, 0], pred[:, 0]).statistic)
          if np.ptp(pred[:, 0]) > 0 and np.ptp(y[:, 0]) > 0 else None)
    return {"rows": len(y), "return_rank_ic": ic,
            "return_sign_accuracy": float(np.mean((y[:, 0] > 0) == (pred[:, 0] > 0))),
            "outcomes": skill}


def quarter_regime(features: pd.DataFrame, ix: np.ndarray, threshold: float) -> dict:
    selected = features.loc[ix]
    first = selected.loc[(selected.index.hour % 6 == 0) & (selected.index.minute == 0)].iloc[0]
    z = first.momentum_90 / max(first.vol_7 * np.sqrt(90 / 365), 1e-6)
    if not np.isfinite(z):
        return {"trend": "unavailable", "normalized_momentum_90": None}
    return {"trend": "bull" if z > threshold else "bear" if z < -threshold else "sideways",
            "normalized_momentum_90": float(z), "volatility_7": float(first.vol_7)}


def summarize(rows: list[dict], minimum_quarters: int) -> dict:
    summaries = {}
    for candidate_id in sorted({r["candidate_id"] for r in rows}):
        candidate_rows = [r for r in rows if r["candidate_id"] == candidate_id]
        regimes = {}
        scores = []
        for regime in ("bull", "bear", "sideways"):
            subset = [r for r in candidate_rows if r["regime"]["trend"] == regime]
            regimes[regime] = {"quarters": len(subset)}
            if subset:
                for cost in ("base", "stress_2x"):
                    a = aggregate([r[cost] for r in subset])
                    regimes[regime][cost] = a
                    scores.extend((a["alpha_ex_mean"], -a["maxdd_delta_mean"]))
        base = aggregate([r["base"] for r in candidate_rows])
        stress = aggregate([r["stress_2x"] for r in candidate_rows])
        coverage = all(regimes[r]["quarters"] >= minimum_quarters for r in regimes)
        summaries[candidate_id] = {
            "base": base, "stress_2x": stress, "regimes": regimes,
            "worst_regime_score": min(scores) if scores else None,
            "all_regime_sample_coverage": coverage,
            "direction_pass": bool(base["alpha_ex_mean"] > 0 and base["maxdd_delta_mean"] < 0
                                   and stress["alpha_ex_mean"] > 0 and stress["maxdd_delta_mean"] < 0),
            "exploratory_regime_direction_pass": bool(coverage and scores and min(scores) > 0),
            "high_probability_generalization_established": False,
        }
    return summaries


def run(config_path: Path, *, skip_hindsight: bool = False) -> dict:
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name("oracle_frontier_features.py"),
                    Path(__file__).with_name("oracle_frontier_hindsight.py"),
                    Path(__file__).with_name("alpha_dd_search.py"),
                    Path(__file__).with_name("alpha_dd_features.py"),
                    Path(__file__).with_name("robust_overlay.py")]
    registration = {"config": config, "config_sha256": file_digest(config_path),
                    "source_sha256": {p.name: file_digest(p) for p in source_paths},
                    "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "data_proof": validate_data_artifact(Path(config["data_path"]), expected_symbol=config["symbol"]),
                    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__},
                    "scope": "reused development validation only; no test selection or formal P1",
                    "hindsight_enabled": not skip_hindsight}
    registration_path = output / "registration.json"
    if registration_path.exists() and json.loads(registration_path.read_text()) != registration:
        raise ValueError("immutable registration differs; use a new output directory")
    write_json(registration_path, registration)
    registration_sha = digest(registration)
    bars = load_bars(Path(config["data_path"]), cutoff=config["data_cutoff"])
    groups = make_feature_groups(bars)
    if set(config["feature_groups"]) - set(groups):
        raise ValueError("registered feature group unavailable")
    common = np.logical_and.reduce([np.isfinite(groups[g].to_numpy(float)).all(axis=1)
                                   for g in config["feature_groups"]])
    cadence = (bars.index.hour % config["decision_cadence_hours"] == 0) & (bars.index.minute == 0)
    all_y = {h: outcome_frame(bars, h) for h in config["horizons_bars"]}
    base_contract = config["execution"]
    stress_contract = {**base_contract, "one_way_cost": 2 * base_contract["one_way_cost"],
                       "borrow_annual": 2 * base_contract["borrow_annual"]}
    all_rows, all_forecasts = [], []
    for fold_id in config["development_folds"]:
        fold_path = output / f"fold_{fold_id}.json"
        if fold_path.exists():
            saved = json.loads(fold_path.read_text())
            if saved["registration_sha256"] != registration_sha:
                raise ValueError("fold registration mismatch")
            all_rows.extend(saved["rows"])
            all_forecasts.extend(saved["forecast_scores"])
            continue
        fold = fold_spec(fold_id, config["fold_anchor"])
        start, end = fold["val_start"], fold["val_end"]
        if end > pd.Timestamp(config["data_cutoff"]):
            raise ValueError("validation exceeds cutoff")
        ix = np.asarray((bars.index >= start) & (bars.index < end))
        window = bars.loc[ix]
        expected_index = pd.date_range(start, end, freq="15min", inclusive="left")
        if not window.index.equals(expected_index):
            raise ValueError("incomplete registered validation grid; cannot shorten quarter")
        if float(window.bar_available.mean()) < config["minimum_bar_coverage"]:
            raise ValueError("registered validation coverage not met")
        regime = quarter_regime(groups["base16"], ix, config["regime"]["normalized_momentum_90_threshold"])
        inference = common & cadence & ix
        rows, forecasts = [], []
        def evaluate(cid, targets, *, oracle=False, metadata=None):
            targets = np.asarray(targets, float)
            row = {"fold": fold_id, "validation_start": str(start), "validation_end": str(end),
                   "candidate_id": cid, "regime": regime, "hindsight_only": oracle,
                   "base": metrics(window, targets, base_contract),
                   "stress_2x": metrics(window, targets, stress_contract),
                   "scheduled_decisions": int(np.isfinite(targets).sum()), "metadata": metadata or {}}
            rows.append(row)
            path = output / "targets" / f"fold{fold_id}_{cid}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, targets=targets, timestamps=window.index.asi8)
            row["targets_sha256"] = file_digest(path)
        evaluate("bh", np.full(len(window), np.nan))
        evaluate("frozen_robust_overlay", build_targets(groups["base16"])[ix])
        matched_robust = build_targets(groups["base16"])
        matched_robust[~inference] = np.nan
        evaluate("common_mask_robust_overlay", matched_robust[ix])
        for h, target_frame in all_y.items():
            y = target_frame.to_numpy(float)
            score_support = inference & np.asarray(
                bars.index + pd.Timedelta(minutes=15 * (h + 1)) <= end)
            train = fit_mask(bars.index, common & np.isfinite(y).all(axis=1),
                             start=start-pd.DateOffset(years=config["train_years"]), end=start,
                             horizon=h, cadence_hours=config["decision_cadence_hours"])
            if train.sum() < config["minimum_fit_rows"]:
                raise ValueError(f"fold{fold_id} h{h}: {train.sum()} fit rows")
            climatology = y[train].mean(axis=0)
            fit_info = {"fit_rows": int(train.sum()), "fit_start": str(bars.index[train][0]),
                        "fit_last_origin": str(bars.index[train][-1]),
                        "fit_last_outcome_end": str(bars.index[train][-1]+pd.Timedelta(minutes=15*(h+1))),
                        "fit_cutoff": str(start), "horizon_bars": h,
                        "common_predict_rows": int(inference.sum())}
            for mapper in config["mappers"]:
                pred = np.tile(climatology, (len(bars), 1))
                targets = map_outcomes(pred, mapper)
                targets[~inference] = np.nan
                evaluate(f"climatology_h{h}_{mapper}", targets[ix], metadata=fit_info)
                oracle_targets = map_outcomes(y, mapper)
                oracle_targets[~score_support] = np.nan
                evaluate(f"ml_perfect_h{h}_{mapper}", oracle_targets[ix], oracle=True,
                         metadata={"scope": "perfect outcome through fixed mapper; not global upper bound"})
            for group in config["feature_groups"]:
                x = groups[group].to_numpy(float)
                for family in config["models"]:
                    model_id = f"{group}_{family}_h{h}"
                    if family == "ridge":
                        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
                    elif family == "hgb":
                        model = MultiOutputRegressor(HistGradientBoostingRegressor(
                            max_iter=100, max_leaf_nodes=7, min_samples_leaf=64,
                            l2_regularization=10., learning_rate=.04, early_stopping=False,
                            random_state=config["seed"]))
                    else:
                        raise ValueError(family)
                    pred = np.full_like(y, np.nan)
                    with threadpool_limits(limits=2):
                        model.fit(x[train], y[train])
                        pred[inference] = model.predict(x[inference])
                    if not np.isfinite(pred[inference]).all():
                        raise ValueError("nonfinite model output")
                    model_path = output / "models" / f"fold{fold_id}_{model_id}.joblib"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, model_path)
                    prediction_path = output / "forecasts" / f"fold{fold_id}_{model_id}.npz"
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    actual = y[ix].copy()
                    actual[~score_support[ix]] = np.nan
                    np.savez_compressed(prediction_path, predictions=pred[ix], actual=actual,
                                        inference_mask=inference[ix], score_support=score_support[ix],
                                        timestamps=window.index.asi8)
                    info = {**fit_info, "features": list(groups[group].columns),
                            "model_sha256": file_digest(model_path),
                            "predictions_sha256": file_digest(prediction_path)}
                    forecasts.append({"fold": fold_id, "model_id": model_id, "regime": regime,
                                      **forecast_scores(y[score_support], pred[score_support], climatology),
                                      "provenance": info})
                    for mapper in config["mappers"]:
                        evaluate(f"{model_id}_{mapper}", map_outcomes(pred, mapper)[ix], metadata=info)
        if not skip_hindsight:
            from .oracle_frontier_hindsight import hindsight_targets
            for penalty in config["hindsight"]["risk_penalties"]:
                targets, diagnostic = hindsight_targets(window, base_contract,
                    beam_width=config["hindsight"]["beam_width"], risk_penalty=penalty)
                evaluate(f"rl_hindsight_beam32_risk{penalty:g}", targets, oracle=True, metadata=diagnostic)
        saved = {"registration_sha256": registration_sha, "rows": rows, "forecast_scores": forecasts}
        write_json(fold_path, saved)
        all_rows.extend(rows)
        all_forecasts.extend(forecasts)
        write_json(output / "progress.json", {"completed_folds": sorted({r["fold"] for r in all_rows}),
                   "planned_folds": config["development_folds"], "registration_sha256": registration_sha})
        print(json.dumps({"event": "fold_complete", "fold": fold_id, "policies": len(rows),
                          "forecast_models": len(forecasts), "regime": regime}), flush=True)
    summaries = summarize(all_rows, config["regime"]["minimum_quarters_per_regime"])
    learned = [key for key in summaries if any(key.startswith(g + "_") for g in config["feature_groups"])]
    ranked = sorted(learned, key=lambda k: (-summaries[k]["worst_regime_score"], k))
    result = {"registration_sha256": registration_sha, "scope": registration["scope"],
              "candidate_count": len(learned), "ranking": ranked, "summary": summaries,
              "rows": all_rows, "forecast_scores": all_forecasts,
              "selected_for_exploratory_review": ranked[0], "formal_p1_result": False,
              "untouched_confirmation_run": False, "high_probability_generalization_established": False}
    write_json(output / "results.json", result)
    write_json(output / "selection_lock.json", {"registration_sha256": registration_sha,
               "result_sha256": file_digest(output / "results.json"), "candidate_id": ranked[0],
               "scope": "exploratory validation only; future qualification required"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"selected": result["selected_for_exploratory_review"],
                      "candidate_count": result["candidate_count"]}), flush=True)


if __name__ == "__main__":
    main()
