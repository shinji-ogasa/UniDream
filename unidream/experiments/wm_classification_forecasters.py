"""Quarterly chronological variance forecasts for the classification study.

This auxiliary stage never changes the earlier registered experiments.
Six early quarters receive new Train/Val fits; fifteen later quarters use
the already selected WM seed7 checkpoint and variance LightGBM. Forecast
workers load input arrays only and never inspect their quarter's labels.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .wm_ablation_evaluation import variance_qlike
from .wm_ablation_model import FeatureAblationWM
from .wm_technical_ablation import (
    _arrays, _context_mask, _features, _input, _json, _load_config, _sha,
)
from .wm_model_comparison import (
    diagnostic_scores, load_config as load_comparison_config, predict_two,
    tabular_design, tabular_scaler, update_patience,
)


BAR_NS = 15 * 60 * 10**9
HORIZONS = np.array([4, 16, 64], dtype=np.int64)


def _utc(value):
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def forecaster_bounds(quarter_start):
    start = _utc(quarter_start)
    if start.day != 1 or start.month not in (1, 4, 7, 10) or start != start.normalize():
        raise ValueError("forecast quarter must start at a UTC calendar quarter")
    val_start = start - pd.DateOffset(months=3)
    return {
        "train_start": max(_utc("2020-09-01"), val_start - pd.DateOffset(years=2)),
        "train_end": val_start, "val_start": val_start, "val_end": start,
        "quarter_start": start, "quarter_end": start + pd.DateOffset(months=3),
    }


def causal_forecaster_cohorts(index_ns, raw, market, target_mask, bounds, context=256):
    """Common complete input windows; all64-bar labels strictly inside split."""
    index_ns = np.asarray(index_ns, dtype=np.int64)
    raw, market = np.asarray(raw), np.asarray(market)
    mask = np.asarray(target_mask, dtype=bool)
    valid = np.isfinite(raw).all(1) & (raw[:, :4] > 0).all(1) & (raw[:, 4] >= 0) & np.isfinite(market).all(1)
    history = _context_mask(valid, context)
    target_valid = mask.all(axis=1)
    result = {}
    for split in ("train", "val"):
        start, end = _utc(bounds[split + "_start"]).value, _utc(bounds[split + "_end"]).value
        keep = history & target_valid & (index_ns >= start) & (index_ns + 64 * BAR_NS < end)
        if split == "train":
            keep &= index_ns >= start + context * BAR_NS
        result[split] = np.flatnonzero(keep)
    return result


def assert_oof_cutoff(decision_ns, selection_end_ns):
    values = np.asarray(decision_ns, dtype=np.int64)
    if not len(values) or not np.all(values > int(selection_end_ns)):
        raise ValueError("every forecast decision must follow model selection labels")


@lru_cache(maxsize=1)
def inference_arrays(ablation_output):
    """Deliberately cannot load any future target/head arrays."""
    with np.load(Path(ablation_output) / "prepared/arrays.npz", allow_pickle=False) as z:
        return {name: z[name] for name in ("raw", "market", "index_ns")}


def _configs(task):
    parent = _load_config(task["ablation_config"])
    comparison = load_comparison_config(task["comparison_config"])
    if Path(comparison["parent_config"]).resolve() != Path(task["ablation_config"]).resolve():
        raise ValueError("forecaster parents do not match")
    return parent, comparison


def _registration(task):
    path = Path(task["registration_path"])
    if not path.is_file():
        raise ValueError("freeze the source-stage preregistration before fitting")
    registration = json.loads(path.read_text())
    expected = registration.get("source_sha256", {})
    if Path(__file__).name in expected and expected[Path(__file__).name] != _sha(__file__):
        raise ValueError("forecaster module changed after source-stage registration")
    return _sha(path)


def _early_dir(task):
    return Path(task["output_dir"]) / "forecasters/early" / f"quarter_{task['quarter_id']:02d}"


def _fold_data(path):
    with np.load(path, allow_pickle=False) as z:
        return {name: z[name] for name in ("train", "val", "selection_val", "mean", "std")}


def _tabular_rows(parent, comparison, arrays, decisions, fold_data):
    """Reuse raw cached rows, compute missing early rows, use THIS Train scaler."""
    cache = Path(comparison["output_dir"]) / "cache"
    cached_ids = np.load(cache / "decisions.npy", mmap_mode="r")
    cached = np.load(cache / "tabular_raw.npy", mmap_mode="r")
    decisions = np.asarray(decisions, dtype=np.int64)
    positions = np.searchsorted(cached_ids, decisions)
    found = positions < len(cached_ids)
    found[found] &= cached_ids[positions[found]] == decisions[found]
    context = parent["model"]["context"]
    lags = comparison["tabular"]["market_lag_positions"]
    width = context * 5 + 58 + 14 * len(lags)
    values = np.empty((len(decisions), width), dtype=np.float32)
    values[found] = cached[positions[found]]
    missing = np.flatnonzero(~found)
    batch = comparison["tabular"]["feature_batch"]
    for offset in range(0, len(missing), batch):
        rows = missing[offset:offset + batch]
        features = _features(arrays["raw"], arrays["market"], decisions[rows], context)
        values[rows] = tabular_design(features, lags)
    mean, std = tabular_scaler(fold_data, context, lags)
    values = (values - mean) / std
    if not np.isfinite(values).all():
        raise ValueError("nonfinite selected forecaster tabular input")
    return values


def prepare_forecaster_plan(output_dir, *, ablation_config, comparison_config, registration_path=None):
    """Prepare twelve early fit tasks and twenty-one label-free inference tasks."""
    out = Path(output_dir).resolve()
    stage = out / "forecasters"
    if (stage / "plan.json").exists():
        raise ValueError("an existing forecaster plan is immutable")
    parent = _load_config(ablation_config)
    comparison = load_comparison_config(comparison_config)
    arrays = _arrays(parent["output_dir"])
    old = Path(parent["output_dir"])
    folds = json.loads((old / "prepared/folds.json").read_text())
    pieces = []
    for fold in folds:
        with np.load(old / "prepared" / f"fold_{fold['fold']:02d}.npz") as z:
            pieces.extend(z[name] for name in ("train", "val", "test"))
    needed = np.unique(np.concatenate(pieces))
    needed = needed[(arrays["index_ns"][needed] >= _utc("2021-07-01").value) & (arrays["index_ns"][needed] < _utc("2026-10-01").value)]
    stage.mkdir(parents=True, exist_ok=True)
    np.save(stage / "needed_ids.npy", needed)
    common = {
        "output_dir": str(out), "ablation_config": str(Path(ablation_config).resolve()),
        "comparison_config": str(Path(comparison_config).resolve()),
        "registration_path": str(Path(registration_path).resolve() if registration_path else out / "preregistration.json"),
    }
    fit_tasks, prediction_tasks = [], []
    for qid, start in enumerate(pd.date_range("2021-07-01", "2026-10-01", freq="QS", inclusive="left", tz="UTC")):
        bounds = forecaster_bounds(start)
        ids = needed[(arrays["index_ns"][needed] >= start.value) & (arrays["index_ns"][needed] < bounds["quarter_end"].value)]
        if not len(ids):
            raise ValueError(f"empty requested forecast quarter {start}")
        ids_path = stage / f"quarter_{qid:02d}_ids.npy"
        np.save(ids_path, ids)
        reused = [fold for fold in folds if _utc(fold["test_start"]) == start]
        task = {**common, "quarter_id": qid, "quarter": str(start.tz_localize(None).to_period("Q")),
                "quarter_start": str(start), "quarter_end": str(bounds["quarter_end"]),
                "forecast_ids_path": str(ids_path), "reuse": bool(reused),
                "source_fold": reused[0]["fold"] if reused else None}
        if reused:
            task["prepared_path"] = str(old / "prepared" / f"fold_{reused[0]['fold']:02d}.npz")
        else:
            directory = _early_dir(task)
            directory.mkdir(parents=True, exist_ok=True)
            data = causal_forecaster_cohorts(arrays["index_ns"], arrays["raw"], arrays["market"], arrays["target_mask"], bounds)
            if min(map(len, data.values())) < 1000:
                raise ValueError("insufficient early Train/Val support")
            rng = np.random.default_rng(20261110 + qid)
            chosen = rng.choice(data["train"], min(1024, len(data["train"])), replace=False)
            total, square, count = np.zeros(63), np.zeros(63), 0
            for offset in range(0, len(chosen), 64):
                values = _features(arrays["raw"], arrays["market"], chosen[offset:offset + 64], 256).astype(np.float64)
                total += values.sum((0, 1)); square += np.square(values).sum((0, 1)); count += values.shape[0] * 256
            mean = total / count
            std = np.sqrt(np.maximum(square / count - mean**2, 0)); std[std < 1e-7] = 1
            train = data["train"]
            scales = {
                "return": np.maximum(np.sqrt(np.mean(arrays["returns"][train]**2, axis=0)), 1e-4).tolist(),
                "variance": np.maximum(np.mean(arrays["variance"][train], axis=0), 1e-8).tolist(),
                "downside": np.maximum(np.sqrt(np.mean(arrays["downside"][train]**2, axis=0)), 1e-4).tolist(),
            }
            selection = data["val"][np.linspace(0, len(data["val"]) - 1, 384).astype(int)]
            prepared = directory / "prepared.npz"
            np.savez(prepared, **data, selection_val=selection, mean=mean.astype(np.float32), std=std.astype(np.float32))
            cohort = {name: str(value) for name, value in bounds.items()}
            cohort.update(scales=scales, support={key: len(value) for key, value in data.items()},
                          normalization_seed=20261110 + qid,
                          selection_end_ns=int(arrays["index_ns"][selection].max() + 64 * BAR_NS),
                          prepared_sha256=_sha(prepared))
            _json(directory / "cohort.json", cohort)
            task.update(prepared_path=str(prepared), cohort_path=str(directory / "cohort.json"))
            fit_tasks.extend({**task, "model": model} for model in ("wm", "lgbm"))
        prediction_tasks.append(task)
    plan = {"fit_tasks": fit_tasks, "prediction_tasks": prediction_tasks,
            "quarters": prediction_tasks, "needed_ids_path": str(stage / "needed_ids.npy"),
            "needed_rows": len(needed), "forecaster_source_sha256": _sha(__file__)}
    _json(stage / "plan.json", plan)
    return plan


def _fit_early_wm(task, parent, comparison, data, cohort, registration_sha):
    directory = _early_dir(task)
    torch.set_num_threads(comparison["cpu_threads_per_worker"])
    torch.use_deterministic_algorithms(True)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    if parent["device"] == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS required for early WM")
    arrays = _arrays(parent["output_dir"])
    model = FeatureAblationWM(63, parent["model"], cohort["scales"]).to(parent["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=parent["training"]["learning_rate"], weight_decay=parent["training"]["weight_decay"])
    rng = np.random.default_rng(7)
    diagnostic_rng = np.random.default_rng(20261110 + task["quarter_id"])
    train_panel = np.sort(diagnostic_rng.choice(data["train"], 384, replace=False))
    settings = comparison["neural"]
    train_start, train_end = _utc(cohort["train_start"]).value, _utc(cohort["train_end"]).value
    best_score, best_step, best_state = float("inf"), 0, None
    anchor, last_significant, reason = float("inf"), 0, "max_steps"
    gradients, targets_seen, history = np.zeros(12, dtype=np.int64), np.zeros(3, dtype=np.int64), []
    for step in range(1, settings["max_steps"] + 1):
        model.train()
        decisions = rng.choice(data["train"], parent["training"]["batch_size"], replace=True)
        x, base = _input(arrays, decisions, data, "D", 256)
        positions = decisions[:, None] - 256 + np.arange(1, 257)[None, :]
        ends = arrays["index_ns"][positions, None] + HORIZONS[None, None, :] * BAR_NS
        valid = arrays["target_mask"][positions] & (ends < train_end) & (arrays["index_ns"][positions, None] >= train_start)
        targets_seen += valid.sum((0, 1))
        targets = {key: torch.from_numpy(arrays[name][positions].astype(np.float32)).to(parent["device"])
                   for key, name in (("return", "returns"), ("variance", "variance"), ("downside", "downside"), ("crash", "crash"))}
        next_return = np.log(arrays["raw"][positions, 3] / arrays["raw"][positions - 1, 3]).astype(np.float32)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.loss(torch.from_numpy(x).to(parent["device"]), torch.from_numpy(base).to(parent["device"]),
                                 targets, torch.from_numpy(valid).to(parent["device"]), torch.from_numpy(next_return).to(parent["device"]))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite early WM loss")
        loss.backward()
        gradients += (model.head[-1].weight.grad.detach().abs().sum(1).cpu().numpy() > 0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), parent["training"]["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        if step % settings["validation_every"] == 0:
            predicted = predict_two(model, arrays, data["selection_val"], data, parent)
            val = diagnostic_scores(arrays, data["selection_val"], predicted, cohort["scales"]["return"][1])
            predicted_train = predict_two(model, arrays, train_panel, data, parent)
            train = diagnostic_scores(arrays, train_panel, predicted_train, cohort["scales"]["return"][1])
            row = {"step": step, "train": train, "val": val, "batch_total_train_loss": float(loss.detach().cpu()),
                   "batch_loss_parts": {key: float(value.cpu()) for key, value in parts.items()}}
            history.append(row); print(json.dumps(row), flush=True)
            if val["score"] < best_score:
                best_score, best_step = val["score"], step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            anchor, last_significant = update_patience(val["score"], step, anchor, last_significant, settings["relative_improvement"])
            if step >= settings["min_steps"] and step - last_significant >= settings["patience_steps"]:
                reason = "validation_patience"; break
    if best_state is None or not np.all(gradients > 0) or not np.all(targets_seen > 0):
        raise RuntimeError("early WM head coverage/selection failed")
    checkpoint = directory / "wm.pt"
    torch.save({"format": "classification_forecaster_wm_v1", "model": best_state, "scales": cohort["scales"],
                "quarter_id": task["quarter_id"], "seed": 7, "best_step": best_step, "preregistration_sha256": registration_sha}, checkpoint)
    if parent["device"] == "mps":
        torch.mps.empty_cache()
    return {"model_path": str(checkpoint), "artifact_sha256": _sha(checkpoint), "scales": cohort["scales"],
            "best_step": best_step, "best_val_score": best_score, "steps": step, "stop_reason": reason,
            "right_censored": reason == "max_steps", "history": history,
            "target_counts": targets_seen.tolist(), "nonzero_gradient_steps": gradients.tolist()}


def _fit_early_lgbm(task, parent, comparison, data, cohort):
    import lightgbm as lgb
    arrays = _arrays(parent["output_dir"])
    x = _tabular_rows(parent, comparison, arrays, data["train"], data)
    xv = _tabular_rows(parent, comparison, arrays, data["selection_val"], data)
    scale = cohort["scales"]["variance"][1]
    y = np.maximum(arrays["variance"][data["train"], 1], 1e-12) / scale
    yv = np.maximum(arrays["variance"][data["selection_val"], 1], 1e-12) / scale
    settings = comparison["tabular"]
    candidates = []
    for leaves in settings["lightgbm_leaves"]:
        model = lgb.LGBMRegressor(objective="gamma", n_estimators=settings["lightgbm_rounds"],
             learning_rate=settings["lightgbm_learning_rate"], num_leaves=leaves,
             min_child_samples=settings["lightgbm_min_data_leaf"], reg_lambda=settings["lightgbm_l2"],
             random_state=settings["lightgbm_seed"], n_jobs=comparison["classical_threads"],
             verbosity=-1, deterministic=True, force_col_wise=True, subsample=1., colsample_bytree=1., max_bin=63)
        model.fit(x, y, eval_set=[(xv, yv)], eval_metric="gamma",
                  callbacks=[lgb.early_stopping(settings["lightgbm_patience"], verbose=False)])
        prediction = model.predict(xv)
        score = float(np.mean(variance_qlike(yv, prediction)))
        if not np.isfinite(score) or not np.isfinite(prediction).all() or not (prediction > 0).all():
            raise RuntimeError("invalid early Gamma predictions")
        candidates.append((score, leaves, model))
    best = min(candidates, key=lambda item: item[0])
    path = _early_dir(task) / "lgbm_variance.txt"
    best[2].booster_.save_model(str(path))
    return {"model_path": str(path), "artifact_sha256": _sha(path), "variance_scale": scale,
            "selected_leaves": best[1], "selection_score": best[0], "best_iteration": best[2].best_iteration_,
            "candidates": [{"leaves": leaves, "score": score, "best_iteration": model.best_iteration_,
                            "history": model.evals_result_} for score, leaves, model in candidates]}


def fit_forecaster_task(task):
    """One early WM/MPS or LightGBM/CPU worker; no forecast-quarter outcomes."""
    if task["reuse"] or task["model"] not in ("wm", "lgbm"):
        raise ValueError("only early WM/LGBM fit tasks may be trained")
    registration_sha = _registration(task)
    parent, comparison = _configs(task)
    directory = _early_dir(task)
    path = directory / (task["model"] + "_fit.json")
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["preregistration_sha256"] != registration_sha or _sha(previous["model_path"]) != previous["artifact_sha256"]:
            raise ValueError("stale/modified early forecaster fit")
        return previous
    cohort = json.loads(Path(task["cohort_path"]).read_text())
    if _sha(task["prepared_path"]) != cohort["prepared_sha256"]:
        raise ValueError("early cohort changed")
    data = _fold_data(task["prepared_path"])
    started = time.monotonic()
    with (directory / (task["model"] + "_training.log")).open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=comparison["classical_threads"]):
        result = (_fit_early_wm(task, parent, comparison, data, cohort, registration_sha)
                  if task["model"] == "wm" else _fit_early_lgbm(task, parent, comparison, data, cohort))
        result.update(quarter_id=task["quarter_id"], quarter=task["quarter"], model=task["model"], seed=7,
                      preregistration_sha256=registration_sha, selection_end_ns=cohort["selection_end_ns"],
                      train_start=cohort["train_start"], train_end=cohort["train_end"],
                      val_start=cohort["val_start"], val_end=cohort["val_end"],
                      prepared_sha256=cohort["prepared_sha256"], elapsed_seconds=time.monotonic() - started)
        _json(path, result)
    return result


def _source_pair(task, parent, comparison):
    arrays = inference_arrays(parent["output_dir"])
    data = _fold_data(task["prepared_path"])
    end = int(arrays["index_ns"][data["selection_val"]].max() + 64 * BAR_NS)
    if end >= _utc(task["quarter_start"]).value:
        raise ValueError("forecaster source selection crosses forecast quarter")
    if task["reuse"]:
        fold = task["source_fold"]
        source = Path(comparison["output_dir"])
        expected = _sha(source / "preregistration.json")
        wm_path = source / "neural" / f"fold_{fold:02d}_wm_extended_s7/fit.json"
        wm = json.loads(wm_path.read_text())
        classical_path = source / "classical" / f"fold_{fold:02d}/fit.json"
        classical = json.loads(classical_path.read_text())
        if wm["preregistration_sha256"] != expected or classical["preregistration_sha256"] != expected:
            raise ValueError("reused source fit registration mismatch")
        checkpoint = wm_path.parent / "research_model.pt"
        booster = Path(classical["lightgbm_variance_path"])
        if _sha(checkpoint) != wm["checkpoint_sha256"] or _sha(booster) != classical["artifact_sha256"][booster.name]:
            raise ValueError("reused source model changed")
        cohort = json.loads((Path(parent["output_dir"]) / "prepared/folds.json").read_text())[fold]
        wm_info = {"model_path": str(checkpoint), "artifact_sha256": wm["checkpoint_sha256"], "scales": cohort["scales"]}
        lgb_info = {"model_path": str(booster), "artifact_sha256": classical["artifact_sha256"][booster.name], "variance_scale": classical["variance_scale"]}
        bounds = {key: cohort[key] for key in ("train_start", "train_end", "val_start", "val_end")}
    else:
        wm_info = json.loads((_early_dir(task) / "wm_fit.json").read_text())
        lgb_info = json.loads((_early_dir(task) / "lgbm_fit.json").read_text())
        expected = _registration(task)
        for info in (wm_info, lgb_info):
            if info["preregistration_sha256"] != expected or _sha(info["model_path"]) != info["artifact_sha256"]:
                raise ValueError("early model changed or stale")
        bounds = {key: wm_info[key] for key in ("train_start", "train_end", "val_start", "val_end")}
    if _utc(bounds["val_end"]) != _utc(task["quarter_start"]):
        raise ValueError("source validation end must equal forecast quarter start")
    return data, wm_info, lgb_info, {**bounds, "model_selection_end_ns": end,
            "selection_end_definition": "conservative latest64-bar label end among immutable selection_val decisions"}


def predict_forecaster_quarter(task):
    """Input-only inference, after ALL auxiliary source selections finish."""
    registration_sha = _registration(task)
    parent, comparison = _configs(task)
    plan = json.loads((Path(task["output_dir"]) / "forecasters/plan.json").read_text())
    for source_task in plan["prediction_tasks"]:
        _source_pair(source_task, parent, comparison)
    arrays = inference_arrays(parent["output_dir"])
    ids = np.load(task["forecast_ids_path"], allow_pickle=False)
    data, wm_info, lgb_info, bounds = _source_pair(task, parent, comparison)
    assert_oof_cutoff(arrays["index_ns"][ids], bounds["model_selection_end_ns"])
    if not np.all((arrays["index_ns"][ids] >= _utc(task["quarter_start"]).value) & (arrays["index_ns"][ids] < _utc(task["quarter_end"]).value)):
        raise ValueError("forecast IDs do not belong to their quarter")
    directory = Path(task["output_dir"]) / "forecasters/oof"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"quarter_{task['quarter_id']:02d}.parquet"
    if path.with_suffix(".json").exists():
        previous = json.loads(path.with_suffix(".json").read_text())
        if previous["preregistration_sha256"] != registration_sha or _sha(path) != previous["prediction_sha256"]:
            raise ValueError("stale/modified OOF forecasts")
        return previous
    torch.set_num_threads(comparison["cpu_threads_per_worker"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(7)
    model = FeatureAblationWM(63, parent["model"], wm_info["scales"]).to(parent["device"])
    checkpoint = torch.load(wm_info["model_path"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # predict_two reads only raw/market. Its inference encoder uses probabilities.
    prediction = predict_two(model, arrays, ids, data, parent)
    wm_variance = prediction["variance"][:, 1]
    del model, checkpoint, prediction
    if parent["device"] == "mps":
        torch.mps.empty_cache()
    from lightgbm import Booster
    x = _tabular_rows(parent, comparison, arrays, ids, data)
    lgb_variance = Booster(model_file=lgb_info["model_path"]).predict(x, num_threads=comparison["classical_threads"]) * lgb_info["variance_scale"]
    if not np.isfinite(wm_variance).all() or not np.isfinite(lgb_variance).all() or not (wm_variance > 0).all() or not (lgb_variance > 0).all():
        raise RuntimeError("OOF variance predictions must be finite and positive")
    frame = pd.DataFrame({"decision_id": ids, "timestamp": pd.to_datetime(arrays["index_ns"][ids], utc=True),
                          "quarter_id": task["quarter_id"], "vhat_wm": wm_variance, "vhat_lgbm": lgb_variance})
    temporary = path.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False); temporary.replace(path)
    metadata = {"quarter_id": task["quarter_id"], "quarter": task["quarter"], "n": len(ids),
                "preregistration_sha256": registration_sha, "prediction_sha256": _sha(path),
                "forecast_first_decision_ns": int(arrays["index_ns"][ids].min()),
                "forecast_last_decision_ns": int(arrays["index_ns"][ids].max()),
                "forecast_decision_strictly_after_model_selection_end": True,
                "forecast_quarter_labels_read": False, "source_reused": task["reuse"],
                "wm_source_sha256": wm_info["artifact_sha256"], "lgbm_source_sha256": lgb_info["artifact_sha256"], **bounds}
    _json(path.with_suffix(".json"), metadata)
    return metadata


def combine_forecaster_oof(output_dir):
    stage = Path(output_dir) / "forecasters"
    plan = json.loads((stage / "plan.json").read_text())
    frames = [pd.read_parquet(stage / "oof" / f"quarter_{task['quarter_id']:02d}.parquet") for task in plan["prediction_tasks"]]
    frame = pd.concat(frames, ignore_index=True).sort_values("decision_id").reset_index(drop=True)
    needed = np.load(plan["needed_ids_path"], allow_pickle=False)
    if frame.decision_id.duplicated().any() or not np.array_equal(frame.decision_id.to_numpy(), needed):
        raise ValueError("OOF results must cover exactly the requested unique decisions")
    path = stage / "oof_variance.parquet"
    frame.to_parquet(path, index=False)
    return str(path)
