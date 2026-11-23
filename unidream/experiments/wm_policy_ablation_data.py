"""Causal, registered data for exploratory policy A/B/C/D ablations.

Forecast heads were selected on previously inspected forecast Test results.  They
are policy DEVELOPMENT features, not an independently unused holdout.  Rewards
are evaluated only on common contiguous saved-forecast segments; segment gaps
must never be presented as a live availability rule or full-calendar PnL.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import wm_full_grid_benchmark as full
from . import wm_joint_grid_benchmark as joint
from . import wm_lstm_grid_benchmark as lstm
from . import wm_seven_task_benchmark as seven
from .wm_seven_task_evaluation import TASKS, fit_target_scales
from .wm_significance_cache import _ordered, _verified_prediction
from .wm_technical_ablation import ROOT

BAR_NS = 15 * 60 * 1_000_000_000
TOKEN_COLUMNS = np.r_[np.arange(255 * 19, 256 * 19), np.arange(4864, 4908)]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("policy arrays must not contain pickle objects")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def _path(value):
    return (ROOT / str(value)).resolve()


def _load(path):
    result = yaml.safe_load(Path(path).read_text())
    result["config_path"] = str(Path(path).resolve())
    result["output_dir"] = str(_path(result["output_dir"]))
    return result


def native_head_specs(points):
    """Keep EVERY positive globally corrected native head; deduplicate aliases.

    Scalar-derived regression BA is never a native classification feature.
    Classical joint rows reuse the two separately fitted control heads, hence
    are aliases rather than extra information. Neural joint heads are distinct.
    """
    required = {"group", "model", "task", "formulation"}
    required |= {f"{prefix}_{suffix}" for prefix in ("scalar_primary", "class_BA")
                 for suffix in ("p_holm_global1320", "improvement")}
    if not required.issubset(points.columns):
        raise ValueError("complete Holm1320 native-head aggregate columns required")
    candidates = {}
    for row in points.to_dict("records"):
        formulation = row["formulation"]
        for head, prefix, eligible in (
            ("regression", "scalar_primary", formulation in ("regression", "joint")),
            ("classification", "class_BA", formulation in ("classification", "joint")),
        ):
            p, improvement = row[f"{prefix}_p_holm_global1320"], row[f"{prefix}_improvement"]
            if not eligible or not np.isfinite(p) or not np.isfinite(improvement) or p >= .05 or improvement <= 0:
                continue
            task = row["task"]
            if task not in TASKS or row["group"] not in "ABCD" or row["model"] not in ("linear", "lightgbm", "direct", "wm", "lstm"):
                raise ValueError("unknown selected native-head identity")
            source_formulation = head if row["model"] in ("linear", "lightgbm") else formulation
            key = (row["group"], row["model"], source_formulation, task, head)
            columns = (["scalar_" + task] if head == "regression" else
                       ["p_crash_1"] if task == "crash" else [f"p_{task}_{i}" for i in range(3)])
            if key not in candidates:
                candidates[key] = {"group": key[0], "model": key[1], "source_formulation": key[2],
                    "task": task, "head": head, "columns": columns,
                    "selection_p_holm_global1320": float(p), "selection_improvement": float(improvement),
                    "aggregate_aliases": []}
            candidates[key]["aggregate_aliases"].append(formulation)
    specs, offset = [], 0
    for key in sorted(candidates):
        entry = candidates[key]
        entry["head_id"] = "_".join(key)
        entry["feature_start"], entry["feature_stop"] = offset, offset + len(entry["columns"])
        entry["aggregate_aliases"] = sorted(set(entry["aggregate_aliases"]))
        specs.append(entry)
        offset = entry["feature_stop"]
    if not specs:
        raise ValueError("no positive significant native heads available")
    return specs


def policy_folds(cfg):
    """Six fixed 2y/3m/3m windows, advanced by one calendar quarter."""
    policy = cfg["policy"]
    start = pd.Timestamp(cfg["splits"]["start"], tz="UTC")
    end = pd.Timestamp(cfg["splits"]["end"], tz="UTC")
    folds, current = [], start
    while True:
        train_end = current + pd.DateOffset(years=int(policy["train_years"]))
        val_end = train_end + pd.DateOffset(months=int(policy["val_months"]))
        test_end = val_end + pd.DateOffset(months=int(policy["test_months"]))
        if test_end > end:
            break
        folds.append({"fold": len(folds), "train_start": current.isoformat(), "train_end": train_end.isoformat(),
            "val_start": train_end.isoformat(), "val_end": val_end.isoformat(),
            "test_start": val_end.isoformat(), "test_end": test_end.isoformat()})
        current += pd.DateOffset(months=int(policy["test_months"]))
    return folds


def inspect_sources(cfg):
    """Read only already-seen aggregate scores and immutable source metadata.

    This is safe before policy registration: no individual forecast rows, raw
    prices, labels, or arrays are loaded here.
    """
    configs = {name: _load(_path(cfg[name + "_config"])) for name in ("significance", "seven", "full", "joint", "lstm")}
    significance_root = Path(configs["significance"]["output_dir"])
    aggregate_path = significance_root / "final/results_420_significance.csv"
    aggregate = pd.read_csv(aggregate_path, float_precision="round_trip")
    if len(aggregate) != 420 or aggregate.duplicated(["group", "model", "task", "formulation"]).any():
        raise ValueError("frozen complete420 significance aggregate required")
    specs = native_head_specs(aggregate)
    registrations = {name: json.loads((Path(configs[name]["output_dir"]) / "preregistration.json").read_text())
                     for name in ("seven", "full", "joint", "lstm")}
    producer_folds = registrations["full"]["folds"]
    if len(producer_folds) != 15:
        raise ValueError("complete fifteen-quarter forecast producer contract required")
    previous_end = None
    for fold in producer_folds:
        train_end, val_start, val_end, test_start, test_end = [pd.Timestamp(fold[k]) for k in
            ("train_end", "val_start", "val_end", "test_start", "test_end")]
        if not (train_end <= val_start < val_end <= test_start < test_end):
            raise ValueError("forecast producer Train/Val must precede forecast Test decisions")
        if previous_end is not None and previous_end != test_start:
            raise ValueError("forecast producer Test quarters must be adjacent")
        previous_end = test_end
    if (pd.Timestamp(producer_folds[0]["test_start"]) != pd.Timestamp(cfg["splits"]["start"], tz="UTC")
            or pd.Timestamp(producer_folds[-1]["test_end"]) != pd.Timestamp(cfg["splits"]["end"], tz="UTC")):
        raise ValueError("policy development timeline must match saved producer forecast quarters")
    cache_manifest_path = significance_root / "prepared/cache_manifest.json"
    source_cache = json.loads(cache_manifest_path.read_text())
    provenance = {str(aggregate_path): _sha(aggregate_path), str(cache_manifest_path): _sha(cache_manifest_path)}
    for name, config in configs.items():
        provenance[config["config_path"]] = _sha(config["config_path"])
        root = Path(config["output_dir"])
        for filename in ("preregistration.json", "completion.json"):
            path = root / filename
            provenance[str(path)] = _sha(path)
    # Bind raw/prepared arrays by frozen metadata, without loading their values.
    parent_config = _load(_path(configs["seven"]["parent_config"]))
    parent_reg_path = Path(parent_config["output_dir"]) / "preregistration.json"
    parent_reg = json.loads(parent_reg_path.read_text())
    provenance[parent_config["config_path"]] = _sha(parent_config["config_path"])
    provenance[str(parent_reg_path)] = _sha(parent_reg_path)
    expected_arrays = {str(Path(parent_config["output_dir"]) / name): digest
                       for name, digest in parent_reg["prepared_sha256"].items()}
    expected_arrays.update({str(Path(configs["seven"]["output_dir"]) / "prepared" / name): digest
                            for name, digest in registrations["seven"]["prepared_sha256"].items()})
    producer_semantics = [{"fold": row["fold"], "label_contract": row["label_contract"], "scales": row["scales"],
        **{k: row[k] for k in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")}}
        for row in producer_folds]
    return {"head_specs": specs, "forecast_dim": sum(len(s["columns"]) for s in specs), "n_heads": len(specs),
        "folds": policy_folds(cfg), "producer_folds": producer_semantics,
        "source_artifact_sha256": provenance, "source_baseline_path": source_cache["baseline_path"],
        "source_baseline_sha256": source_cache["baseline_sha256"],
        "source_prepared_sha256": expected_arrays,
        "feature_selection_is_exploratory": True, "independently_unused_holdout": False,
        "feature_names": [spec["head_id"] + ":" + col for spec in specs for col in spec["columns"]],
        "token_columns": TOKEN_COLUMNS.tolist(), "effective_raw_history_bars": 511,
        "context_semantics": "256 causal cached last-token63 vectors; each token sees preceding256 raw bars",
        "classification_semantics": "raw native probabilities; producer-quarter Train cutpoints, RMS and priors saved as metadata only",
        "evaluation_scope": "conditional contiguous saved forecast cohort; reset at gaps and split boundaries; not full-calendar PnL"}


def require_registration(cfg, registration):
    path = Path(cfg["output_dir"]) / "preregistration.json"
    if not path.exists():
        raise ValueError("policy preregistration must exist before individual forecast, price or outcome reads")
    if json.loads(path.read_text()) != registration:
        raise ValueError("supplied policy registration differs from immutable disk registration")
    return _sha(path)


def contiguous_segments(index_ns):
    times = np.asarray(index_ns, dtype=np.int64)
    if times.ndim != 1 or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("segment timestamps must be unique and strictly increasing")
    if not len(times):
        return np.empty(0, dtype=np.int64)
    return np.cumsum(np.r_[True, np.diff(times) != BAR_NS]).astype(np.int64) - 1


def causal_context_valid(x_grid, decision_ids, context=256):
    """Validity uses X tokens ending at the current decision, never future rows."""
    good = np.isfinite(x_grid).all(axis=1)
    prefix = np.r_[0, np.cumsum(~good, dtype=np.int64)]
    ids = np.asarray(decision_ids, dtype=np.int64)
    valid = ids >= context - 1
    eligible = ids[valid]
    valid[valid] = prefix[eligible + 1] == prefix[eligible - context + 1]
    return valid


def future_labels(close, decision_ids, horizon=16):
    """Cumulative log(C[t+k]/C[t]), k=1..horizon; label-only one-bar delay."""
    close = np.asarray(close, dtype=np.float64)
    ids = np.asarray(decision_ids, dtype=np.int64)
    path = np.full((len(ids), horizon), np.nan, dtype=np.float64)
    valid = (ids >= 1) & (ids + horizon < len(close))
    rows = np.flatnonzero(valid)
    positions = ids[rows, None] + np.arange(horizon + 1)[None, :]
    prices = close[positions]
    finite = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    valid[rows] = finite
    rows = rows[finite]
    if len(rows):
        path[rows] = np.log(close[ids[rows, None] + np.arange(1, horizon + 1)] / close[ids[rows, None]])
    current = np.full(len(ids), np.nan, dtype=np.float64)
    good = (ids >= 1) & (ids < len(close))
    selected = ids[good]
    values = np.log(close[selected] / close[selected - 1])
    current[good] = np.where(np.isfinite(values), values, np.nan)
    return current, path, valid


def fit_normalizer(values, train_ids):
    selected = np.asarray(values)[np.asarray(train_ids, dtype=np.int64)].astype(np.float64)
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError("normalizer requires nonempty finite Train-only rows")
    mean, std = selected.mean(axis=0), selected.std(axis=0)
    return mean.astype(np.float64), np.where(std > 1e-8, std, 1.).astype(np.float64)


def purged_split_ids(index_ns, valid, start, end, purge_bars=64):
    times = np.asarray(index_ns, dtype=np.int64)
    lower, upper = pd.Timestamp(start).value, pd.Timestamp(end).value
    return np.flatnonzero(np.asarray(valid, bool) & (times >= lower) & (times + purge_bars * BAR_NS < upper))


def x_windows(arrays, ledger_ids, fold_data, context=256, clip=10.):
    """Return normalized X[d-context+1:d+1]; all IDs are LOCAL ledger rows."""
    ledger_ids = np.asarray(ledger_ids, dtype=np.int64)
    decisions = np.asarray(arrays["decision_id"])[ledger_ids]
    if not np.asarray(arrays["context_valid"])[ledger_ids].all() or np.any(decisions < context - 1):
        raise ValueError("requested X window has unavailable causal history")
    positions = decisions[:, None] - np.arange(context - 1, -1, -1)[None, :]
    raw = np.asarray(arrays["x_grid"])[positions].astype(np.float32)
    result = (raw - np.asarray(fold_data["x_mean"], np.float32)) / np.asarray(fold_data["x_std"], np.float32)
    if not np.isfinite(result).all():
        raise ValueError("normalized causal X history must be finite")
    return np.clip(result, -clip, clip).astype(np.float32)


def f_values(arrays, ledger_ids, fold_data, clip=10.):
    raw = np.asarray(arrays["f"])[np.asarray(ledger_ids, np.int64)]
    result = (raw - fold_data["f_mean"]) / fold_data["f_std"]
    if not np.isfinite(result).all():
        raise ValueError("normalized native forecast features must be finite")
    return np.clip(result, -clip, clip).astype(np.float32)


def _forecast_source(spec, fold, configs, registrations):
    name = "lstm" if spec["model"] == "lstm" else "joint" if spec["source_formulation"] == "joint" else "full"
    kind = "classical" if spec["model"] in ("linear", "lightgbm") else "neural"
    matches = [row for row in registrations[name][kind + "_tasks"] if row["fold"] == fold and row["group"] == spec["group"]
        and (kind == "classical" or (row["model"] == spec["model"] and row["formulation"] == spec["source_formulation"]))]
    if len(matches) != 1:
        raise ValueError("unique frozen source job required for each native forecast head")
    return Path(configs[name]["output_dir"]), kind, matches[0]["job"]


def _extract_feature(frame, spec):
    columns = spec["columns"]
    expected = (["scalar_" + spec["task"]] if spec["head"] == "regression" else
                ["p_crash_1"] if spec["task"] == "crash" else [f"p_{spec['task']}_{i}" for i in range(3)])
    if columns != expected:
        raise ValueError("only explicitly whitelisted native forecast columns may enter F")
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("selected native forecasts must be finite on the complete common cohort")
    if spec["head"] == "classification":
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("native class probabilities must lie in [0,1]")
        if values.shape[1] == 3 and not np.allclose(values.sum(axis=1), 1., atol=1e-6, rtol=0):
            raise ValueError("native class probabilities must sum to one")
    return values


def _prior_cache(path, registration_sha):
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if not manifest.get("complete") or manifest.get("registration_sha256") != registration_sha:
        raise ValueError("existing policy cache belongs to a different or incomplete registration")
    expected = dict(manifest["source_artifact_sha256"])
    for entry in list(manifest["arrays"].values()) + list(manifest["fold_arrays"].values()):
        expected[entry["path"]] = entry["sha256"]
    for name, digest in expected.items():
        if _sha(name) != digest:
            raise ValueError("policy cache or immutable forecast source changed")
    return {**manifest, "cache_manifest_path": str(path.resolve()), "cache_manifest_sha": _sha(path)}


def build_cache(cfg, registration):
    """Materialize common forecast and causal-market features AFTER registration."""
    reg_sha = require_registration(cfg, registration)
    root = Path(cfg["output_dir"]).resolve()
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    manifest_path = prepared / "cache_manifest.json"
    previous = _prior_cache(manifest_path, reg_sha)
    if previous is not None:
        return previous
    inspected = inspect_sources(cfg)
    registered_sources = registration.get("data_sources")
    if registered_sources is None or inspected != registered_sources:
        raise ValueError("forecast head inventory or source metadata changed after policy registration")
    configs = {name: _load(_path(cfg[name + "_config"])) for name in ("seven", "full", "joint", "lstm")}
    registrations = {name: json.loads((Path(config["output_dir"]) / "preregistration.json").read_text()) for name, config in configs.items()}
    provenance = dict(inspected["source_artifact_sha256"])
    for filename, expected in inspected["source_prepared_sha256"].items():
        if _sha(filename) != expected:
            raise ValueError("frozen original prepared array changed")
        provenance[filename] = expected
    baseline_path = Path(inspected["source_baseline_path"])
    if _sha(baseline_path) != inspected["source_baseline_sha256"]:
        raise ValueError("frozen shared forecast ledger changed")
    provenance[str(baseline_path)] = inspected["source_baseline_sha256"]
    with np.load(baseline_path, allow_pickle=False) as baseline:
        decisions = baseline["decision_id"].astype(np.int64)
        times = baseline["index_ns"].astype(np.int64)
        producer_fold = baseline["fold"].astype(np.int64)
        ewma_variance = baseline["filter_variance"][:, 1].astype(np.float64)
    if (len(decisions) != 124488 or len(np.unique(decisions)) != len(decisions)
            or np.any(np.diff(times) <= 0) or set(producer_fold.tolist()) != set(range(15))):
        raise ValueError("complete unique chronological124488 forecast decisions required")
    scfg = seven.load_config(configs["seven"]["config_path"])
    sreg = seven.verify(scfg, full=True)
    raw_arrays = seven.arrays_for(scfg)
    if not np.array_equal(np.asarray(raw_arrays["index_ns"])[decisions], times):
        raise ValueError("forecast ledger global decisions must match original raw grid timestamps")
    raw_n = len(raw_arrays["index_ns"])
    if np.any(np.diff(np.asarray(raw_arrays["index_ns"], np.int64)) != BAR_NS):
        raise ValueError("causal token histories require the original regular15min raw grid")
    if not np.isfinite(ewma_variance).all() or np.any(ewma_variance < 0):
        raise ValueError("frozen causal EWMA variance control must be finite and nonnegative")
    cache_root = seven.cache_directory(scfg)
    for name, expected in sreg["cache_sha256"].items():
        provenance[str((cache_root / name).resolve())] = expected
    cache_ids = np.load(cache_root / "decisions.npy", mmap_mode="r")
    tabular = np.load(cache_root / "tabular_raw.npy", mmap_mode="r")
    if tabular.shape != (len(cache_ids), 4908) or np.any(np.diff(cache_ids) <= 0):
        raise ValueError("frozen chronological4908-channel causal token cache required")
    x_grid = np.full((raw_n, 63), np.nan, dtype=np.float32)
    for start in range(0, len(cache_ids), 8192):
        selected = slice(start, start + 8192)
        x_grid[np.asarray(cache_ids[selected], np.int64)] = tabular[selected][:, TOKEN_COLUMNS]
    features = np.empty((len(decisions), inspected["forecast_dim"]), dtype=np.float32)
    for fold in range(15):
        rows = np.flatnonzero(producer_fold == fold)
        meta = inspected["producer_folds"][fold]
        if not len(rows) or np.any(times[rows] < pd.Timestamp(meta["test_start"]).value) or np.any(times[rows] >= pd.Timestamp(meta["test_end"]).value):
            raise ValueError("forecast producer chronology differs from registered Test quarter")
        frames = {}
        for spec in inspected["head_specs"]:
            source_root, kind, job = _forecast_source(spec, fold, configs, registrations)
            key = (str(source_root), kind, job)
            if key not in frames:
                path, _ = _verified_prediction(source_root, kind, job, provenance)
                frames[key] = pd.read_parquet(path)
            selected = _ordered(frames[key], {"split": "test", "fold": fold, "group": spec["group"], "seed": 7,
                "model": spec["model"], "formulation": spec["source_formulation"]}, decisions[rows], times[rows])
            features[rows, spec["feature_start"]:spec["feature_stop"]] = _extract_feature(selected, spec)
        print(json.dumps({"policy_forecast_fold": fold, "n_rows": len(rows)}), flush=True)
    context_valid = causal_context_valid(x_grid, decisions, int(cfg["policy"]["context"]))
    close = np.asarray(raw_arrays["raw"][:, 3], dtype=np.float64)
    current_return, future_path, teacher_valid = future_labels(close, decisions, int(cfg["policy"]["horizon"]))
    if not np.isfinite(current_return).all():
        raise ValueError("common reward cohort must have finite current bar returns")
    wm_targets = np.column_stack([raw_arrays[t][:, 0] for t in TASKS]).astype(np.float64)
    wm_mask = np.asarray(raw_arrays["mask"][:, 0], bool) & np.isfinite(wm_targets).all(axis=1)
    values = {"decision_id": decisions, "index_ns": times, "producer_fold": producer_fold,
        "segment_id": contiguous_segments(times), "x_grid": x_grid, "f": features, "context_valid": context_valid,
        "ewma_variance": ewma_variance,
        "bar_log_return": current_return, "future_path_log_returns": future_path, "teacher_valid": teacher_valid,
        "raw_close": close, "raw_index_ns": np.asarray(raw_arrays["index_ns"], np.int64),
        "wm_targets_grid": wm_targets, "wm_mask_grid": wm_mask,
        "wm_label_end_ns": np.asarray(raw_arrays["label_end_ns"][:, 0], np.int64)}
    array_manifest = {}
    for name, value in values.items():
        path = prepared / f"{name}.npy"
        _npy(path, value)
        array_manifest[name] = {"path": str(path), "sha256": _sha(path), "shape": list(value.shape), "dtype": str(value.dtype)}
    fold_manifest, fold_audits = {}, []
    for meta in inspected["folds"]:
        data, audit = _make_fold(values, meta, cfg)
        path = prepared / f"fold_{meta['fold']:02d}.npz"
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, **data)
        temporary.replace(path)
        fold_manifest[str(meta["fold"])] = {"path": str(path), "sha256": _sha(path)}
        fold_audits.append(audit)
    manifest = {"complete": True, "registration_sha256": reg_sha, "arrays": array_manifest, "fold_arrays": fold_manifest,
        "source_artifact_sha256": provenance, "head_specs": inspected["head_specs"], "feature_names": inspected["feature_names"],
        "forecast_dim": inspected["forecast_dim"], "n_heads": inspected["n_heads"], "n_rows": len(decisions),
        "n_context_valid": int(context_valid.sum()), "n_teacher_valid": int(teacher_valid.sum()), "folds": fold_audits,
        "producer_folds": inspected["producer_folds"], "exploratory": True, "independently_unused_holdout": False,
        "evaluation_scope": inspected["evaluation_scope"], "context_semantics": inspected["context_semantics"],
        "effective_raw_history_bars": 511, "task_order": list(TASKS),
        "future_path_semantics": "LABEL ONLY cumulative log(close[t+k]/close[t]),k=1..16; actor request[t] executes at t+1",
        "normalizer_fit": "current X tokens and F at policy Train decisions only; std floor1e-8 ->1; clip fixed YAML"}
    _json(manifest_path, manifest)
    return {**manifest, "cache_manifest_path": str(manifest_path), "cache_manifest_sha": _sha(manifest_path)}


def _make_fold(arrays, meta, cfg):
    times, valid = np.asarray(arrays["index_ns"]), np.asarray(arrays["context_valid"], bool)
    valid = valid & np.isfinite(arrays["f"]).all(axis=1) & np.isfinite(arrays["bar_log_return"])
    purge = int(cfg["policy"]["purge_bars"])
    if purge < int(cfg["policy"]["horizon"]) + 1:
        raise ValueError("purge must cover execution delay and all teacher future labels")
    ranges = {name: (meta[name + "_start"], meta[name + "_end"]) for name in ("train", "val", "test")}
    val_start = pd.Timestamp(meta["val_start"])
    for offset, name in enumerate(("earlystop", "calibration", "selection")):
        ranges[name] = ((val_start + pd.DateOffset(months=offset)).isoformat(),
                        (val_start + pd.DateOffset(months=offset + 1)).isoformat())
    data, support = {}, {}
    for name, (start, end) in ranges.items():
        ids = purged_split_ids(times, valid, start, end, purge)
        if not len(ids):
            raise ValueError(f"policy fold {meta['fold']} has no eligible {name} observations")
        data[name] = ids.astype(np.int64)
        data[name + "_segment_id"] = contiguous_segments(times[ids])
        support[name] = {"n_rows": len(ids), "n_segments": int(data[name + "_segment_id"][-1] + 1),
            "teacher_rows": int(np.asarray(arrays["teacher_valid"])[ids].sum())}
    train = data["train"]
    decisions = np.asarray(arrays["decision_id"])[train]
    data["x_mean"], data["x_std"] = fit_normalizer(np.asarray(arrays["x_grid"]), decisions)
    data["f_mean"], data["f_std"] = fit_normalizer(arrays["f"], train)
    wm_valid = np.asarray(arrays["wm_mask_grid"])[decisions] & (np.asarray(arrays["wm_label_end_ns"])[decisions] < pd.Timestamp(meta["train_end"]).value)
    train_targets = {task: np.asarray(arrays["wm_targets_grid"])[decisions, col:col+1] for col, task in enumerate(TASKS)}
    train_targets["mask"] = wm_valid[:, None]
    scales = fit_target_scales(train_targets, np.arange(len(train)))
    for task, scale in scales.items():
        data["scale_" + task] = np.asarray(scale, np.float64)
    data["wm_train"] = train[wm_valid].astype(np.int64)
    for name in ("earlystop", "calibration", "selection", "val", "test"):
        ids = data[name]
        decisions = np.asarray(arrays["decision_id"])[ids]
        end = ranges[name][1]
        good = np.asarray(arrays["wm_mask_grid"])[decisions] & (np.asarray(arrays["wm_label_end_ns"])[decisions] < pd.Timestamp(end).value)
        data["wm_" + name] = ids[good].astype(np.int64)
    return data, {**meta, "support": support, "scales": scales, "x_norm_rows": len(train), "f_norm_rows": len(train)}


def load_arrays(cfg):
    root = Path(cfg["output_dir"]) / "prepared"
    manifest = json.loads((root / "cache_manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("complete immutable policy cache required")
    return {name: np.load(entry["path"], mmap_mode="r", allow_pickle=False) for name, entry in manifest["arrays"].items()}


def fold_arrays(cfg, fold):
    path = Path(cfg["output_dir"]) / "prepared" / f"fold_{int(fold):02d}.npz"
    with np.load(path, allow_pickle=False) as result:
        return {name: result[name] for name in result.files}
