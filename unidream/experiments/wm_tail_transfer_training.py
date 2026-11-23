"""Fresh Train/Val-only forecast workers and chronological producer exports.

This isolated study uses native target-specific heads. Test inference is a
separate function guarded by the campaign's global selection barrier.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import time

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
import torch

from . import wm_tail_transfer_data as data_module
from . import wm_tail_transfer_models as models
from . import wm_tail_transfer_evaluation as evaluation


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def _json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fit_directory(cfg, task):
    return Path(cfg["output_dir"])/"fits"/task["job"]


def _ranges(cfg, task):
    from .wm_tail_transfer_benchmark import fold_definitions
    meta = next(x for x in fold_definitions(cfg) if x["fold"] == int(task["fold"]))
    if task["role"] == "producer":
        return meta, {"train": (meta["producer_train_start"], meta["producer_train_end"]),
                      "earlystop": (meta["producer_earlystop_start"], meta["producer_earlystop_end"]),
                      "calibration": (meta["producer_calibration_start"], meta["producer_calibration_end"]),
                      "selection": (meta["producer_calibration_start"], meta["producer_calibration_end"])}
    return meta, {key: (meta[key+"_start"], meta[key+"_end"]) for key in ("train", "earlystop", "calibration", "selection")}


def split_arrays(cfg, task, data, *, allow_empty=False):
    from .wm_tail_transfer_benchmark import split_ids
    meta, ranges = _ranges(cfg, task)
    ids = {key: split_ids(data, task["group"], start, end, cohort=task.get("cohort")) for key, (start, end) in ranges.items()}
    if any(not len(v) for v in ids.values()) and not allow_empty:
        raise ValueError(f"nonempty purged chronological Train/Val partitions required for {task['job']}")
    if (len(ids["train"]) and len(ids["earlystop"])
            and np.max(data["index_ns"][ids["train"]]) >= np.min(data["index_ns"][ids["earlystop"]])):
        raise ValueError("forecast Train/Val overlap")
    return meta, ranges, ids


def _windows(data, ids, group, cfg):
    cache = data.get("window_cache")
    if cache is not None:
        result = np.asarray(cache[np.asarray(ids, np.int64)], np.float32)
    else:
        result = data_module.window_inputs(data, ids, group, int(cfg["model"].get("context", 128)))
    if result.ndim != 3 or not np.isfinite(result).all():
        raise ValueError("finite canonical window-local causal inputs required")
    return result


def attach_window_cache(cfg, data, group):
    if "window_cache" in data:
        return data
    path = Path(cfg["output_dir"])/"data/window_cache/base_D03.npy"
    if path.exists():
        from .wm_tail_transfer_benchmark import WindowCache
        return {**data, "window_cache": WindowCache(cfg, data, group)}
    return data


def target_panels(data, ids, specs):
    return ({s["name"]: np.asarray(data["targets"][s["name"]][ids], np.float32) for s in specs},
            {s["name"]: np.asarray(data["masks"][s["name"]][ids], bool) for s in specs})


def normalization(data, ids, specs, group, cfg):
    """All normalizers use policy/forecast Train only, last-token inputs."""
    total = squared = None
    count = 0
    for start in range(0, len(ids), 512):
        selected = ids[start:start+512]
        part = np.asarray(data["input_"+group][selected] if "input_"+group in data else _windows(data, selected, group, cfg)[:, -1], np.float64)
        if total is None:
            total = np.zeros(part.shape[1]); squared = np.zeros_like(total)
        total += part.sum(0); squared += np.square(part).sum(0); count += len(part)
    mean = total/count
    std = np.sqrt(np.maximum(0, squared/count-mean*mean))
    std = np.where(std > 1e-8, std, 1.)
    scales = {}
    for spec in specs:
        name, kind = spec["name"], spec["kind"]
        if kind in models.CLASSIFICATION_KINDS:
            scales[name] = np.ones(spec["shape"], np.float32).tolist(); continue
        y = np.asarray(data["targets"][name][ids], np.float64)
        valid = np.asarray(data["masks"][name][ids], bool) & np.isfinite(y)
        den = valid.sum(0)
        total_y = np.where(valid, y if kind == "variance" else y*y, 0).sum(0)
        value = np.divide(total_y, den, out=np.ones_like(total_y), where=den > 0)
        if kind != "variance":
            value = np.sqrt(value)
        scales[name] = np.maximum(value, 1e-10 if kind == "variance" else 1e-6).tolist()
    close = np.asarray(data["ohlcspot"][:, 3])
    r = np.log(close[ids]/close[ids-1])
    r = r[np.isfinite(r)]
    scales["next_return"] = max(float(np.sqrt(np.mean(r*r))) if len(r) else .005, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32), scales


def tabular_inputs(windows, recipe="window_summaries"):
    if recipe in ("full_flatten", "fullflatten"):
        return np.asarray(windows, np.float32).reshape(len(windows), -1)
    if recipe not in ("window_summaries", "last_mean_std_min_max_trailing4_16_64", "window_last_mean_std_min_max_trailing4_16_64_means", "summary"):
        raise ValueError("unknown registered classical window feature recipe")
    parts = [windows[:, -1], windows.mean(1), windows.std(1), windows.min(1), windows.max(1)]
    parts += [windows[:, -min(k, windows.shape[1]):].mean(1) for k in (4, 16, 64)]
    return np.concatenate(parts, -1).astype(np.float32)


def _tabular(data, ids, group, cfg, mean, std, recipe):
    pieces = []
    for start in range(0, len(ids), 512):
        x = ((_windows(data, ids[start:start+512], group, cfg)-mean)/std).clip(-10, 10)
        pieces.append(tabular_inputs(x, recipe))
    return np.concatenate(pieces)


_LATENT_FEATURES = {}


def prepare_latent_features(cfg, task, registration_sha, *, arrays=None, split="train_val"):
    """One fixed source encoder cache shared by both classical latent probes."""
    from .wm_tail_transfer_benchmark import execution_device, fold_definitions, split_ids, verify_test_barrier
    if split not in ("train_val", "test"):
        raise ValueError("latent feature split must be train_val or report-only Test")
    if split == "test" and verify_test_barrier(cfg)["registration_sha256"] != registration_sha:
        raise ValueError("latent-probe Test cache requires the global barrier")
    source_dir = Path(cfg["output_dir"])/"fits"/task["source_job"]
    source_files = {str((source_dir/name).resolve()): _sha(source_dir/name) for name in ("selected.pt", "norm.npz", "fit.json")}
    directory = Path(cfg["output_dir"])/"latent_probe_features"/task["source_job"]/split
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/"cache.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        marker = directory/"manifest.json"
        if marker.exists():
            manifest = json.loads(marker.read_text())
            if manifest["registration_sha256"] != registration_sha or manifest["source_sha256"] != source_files:
                raise ValueError("latent probe source encoder/normalizer changed")
            if any(_sha(directory/name) != value for name, value in manifest["artifact_sha256"].items()):
                raise ValueError("frozen latent-probe features changed")
            return manifest
        obj = torch.load(source_dir/"selected.pt", map_location="cpu", weights_only=True)
        if obj["registration_sha256"] != registration_sha:
            raise ValueError("latent probe source belongs to another campaign")
        model = _construct(obj["constructor"])
        model.load_state_dict(obj["state_dict"], strict=True)
        source_fit = json.loads((source_dir/"fit.json").read_text())
        source_device = execution_device(cfg, source_fit["task"])
        model.to(source_device).eval().requires_grad_(False)
        with np.load(source_dir/"norm.npz", allow_pickle=False) as norm:
            mean, std = norm["mean"], norm["std"]
        data = attach_window_cache(cfg, data_module.load_data(cfg) if arrays is None else arrays, task["group"])
        meta = next(x for x in fold_definitions(cfg) if x["fold"] == int(task["fold"]))
        start, end = (meta["train_start"], meta["val_end"]) if split == "train_val" else (meta["test_start"], meta["test_end"])
        ids = split_ids(data, task["group"], start, end, purge=False, require_labels=False)
        output = np.lib.format.open_memmap(directory/"features.npy", mode="w+", dtype=np.float32, shape=(len(ids), int(model.representation_dim)))
        batch = int(cfg["training"].get("evaluation_batch_size", 256))
        with torch.no_grad():
            for offset in range(0, len(ids), batch):
                selected = ids[offset:offset+batch]
                x = ((_windows(data, selected, task["group"], cfg)-mean)/std).clip(-10, 10).astype(np.float32)
                output[offset:offset+len(selected)] = model.latent_last(torch.from_numpy(x).to(next(model.parameters()).device)).cpu().numpy()
        output.flush(); np.save(directory/"raw_ids.npy", ids, allow_pickle=False)
        manifest = {"complete": True, "registration_sha256": registration_sha, "split": split,
                    "source_job": task["source_job"], "source_sha256": source_files,
                    "directory": str(directory.resolve()), "rows": len(ids), "width": int(model.representation_dim),
                    "source_normalizer_inherited": True, "encoder_updated": False,
                    "source_execution_device": source_device,
                    "artifact_sha256": {name: _sha(directory/name) for name in ("features.npy", "raw_ids.npy")}}
        _json(marker, manifest)
        return manifest


def _latent_tabular(cfg, task, registration_sha, data, ids):
    from .wm_tail_transfer_benchmark import fold_definitions
    meta = next(x for x in fold_definitions(cfg) if x["fold"] == int(task["fold"]))
    is_test = np.asarray(data["index_ns"])[np.asarray(ids)].max() >= pd.Timestamp(meta["test_start"]).value
    split = "test" if is_test else "train_val"
    key = (str(cfg["output_dir"]), task["source_job"], split, registration_sha)
    if key not in _LATENT_FEATURES:
        manifest = prepare_latent_features(cfg, task, registration_sha, arrays=data, split=split)
        directory = Path(manifest["directory"])
        _LATENT_FEATURES[key] = (np.load(directory/"raw_ids.npy", mmap_mode="r"), np.load(directory/"features.npy", mmap_mode="r"))
    raw_ids, features = _LATENT_FEATURES[key]
    position = np.searchsorted(raw_ids, ids)
    if np.any(position >= len(raw_ids)) or not np.array_equal(raw_ids[position], ids):
        raise ValueError("latent probe requested unavailable source contexts")
    return np.asarray(features[position], np.float32)


def _construct(constructor):
    cls = constructor.get("class", "TailModel")
    kwargs = {k: constructor[k] for k in ("input_dim", "cfg", "specs", "scales", "variant", "objectives")}
    if cls == "HurdleTailModel":
        from .wm_tail_transfer_probes import HurdleTailModel
        return HurdleTailModel(**kwargs)
    if cls == "FrozenTailProbe":
        from .wm_tail_transfer_probes import FrozenTailProbe
        base = _construct(constructor["base_constructor"])
        return FrozenTailProbe(base, constructor["cfg"], constructor["specs"], constructor["scales"], constructor["objectives"])
    return models.TailModel(**kwargs)


def _save_neural(path, model, constructor, registration_sha, step):
    obj = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
           "constructor": constructor, "registration_sha256": registration_sha, "selected_step": int(step),
           "metadata": model.metadata()}
    temporary = Path(str(path)+".tmp")
    torch.save(obj, temporary); temporary.replace(path)


def load_fit(cfg, task, registration_sha=None):
    from .wm_tail_transfer_benchmark import execution_device
    directory = fit_directory(cfg, task)
    fit = json.loads((directory/"fit.json").read_text())
    if not fit.get("complete") or registration_sha is not None and fit["registration_sha256"] != registration_sha:
        raise ValueError("complete matching forecast registration required")
    for name in ("selected.pt", "bundle.pkl", "norm.npz", "calibrators.pkl"):
        if name in fit["artifact_sha256"] and _sha(directory/name) != fit["artifact_sha256"][name]:
            raise ValueError("selected forecast model or calibration changed")
    for path, expected in fit.get("source_artifact_sha256", {}).items():
        if _sha(path) != expected:
            raise ValueError("frozen source encoder/checkpoint/normalizer changed")
    with np.load(directory/"norm.npz", allow_pickle=False) as v:
        norm = {k: v[k] for k in v.files}
    if task["model"] in ("linear", "lightgbm", "lgbm"):
        with (directory/"bundle.pkl").open("rb") as f:
            model = pickle.load(f)
    else:
        obj = torch.load(directory/"selected.pt", map_location="cpu", weights_only=True)
        if obj["registration_sha256"] != fit["registration_sha256"]:
            raise ValueError("neural forecast checkpoint provenance differs")
        model = _construct(obj["constructor"])
        model.load_state_dict(obj["state_dict"], strict=True)
        model.to(execution_device(cfg, task)).eval()
        if hasattr(model, "backbone_digest"):
            expected = obj["metadata"]["source_backbone_sha256"]
            if model.backbone_digest() != expected:
                raise ValueError("frozen probe backbone differs from source checkpoint")
            model._source_backbone_sha256 = expected
    with (directory/"calibrators.pkl").open("rb") as f:
        calibrators = pickle.load(f)
    return model, norm, fit, calibrators


@torch.no_grad()
def _predict(model, data, ids, task, cfg, mean, std, *, latents=False, registration_sha=None):
    batch = int(cfg["training"].get("evaluation_batch_size", 256))
    pieces, latent_parts = defaultdict(list), []
    classical = isinstance(model, dict)
    if not classical:
        device = next(model.parameters()).device; model.eval()
    for start in range(0, len(ids), batch):
        selected = np.asarray(ids[start:start+batch], np.int64)
        if classical:
            from .wm_tail_transfer_classical import predict_bundle
            if task.get("architecture") == "latent_probe":
                if registration_sha is None:
                    raise ValueError("latent-probe inference requires registered source identity")
                features = _latent_tabular(cfg, task, registration_sha, data, selected)
            else:
                x = ((_windows(data, selected, task["group"], cfg)-mean)/std).clip(-10, 10).astype(np.float32)
                recipe = task.get("feature_recipe", cfg.get("classical", {}).get("feature_recipe", "window_summaries"))
                features = tabular_inputs(x, recipe)
            prediction = predict_bundle(model, features)
        else:
            x = ((_windows(data, selected, task["group"], cfg)-mean)/std).clip(-10, 10).astype(np.float32)
            tensor = torch.from_numpy(x).to(device)
            if latents and hasattr(model, "_outputs"):
                representation = model.encode(tensor)[:, -1]
                prediction = {name: value.cpu().numpy() for name, value in model._outputs(representation)[0].items()}
                latent_parts.append(representation.cpu().numpy())
            else:
                prediction = {name: value.cpu().numpy() for name, value in model.predict_last(tensor).items()}
                if latents:
                    latent_parts.append(model.latent_last(tensor).cpu().numpy())
        for name, value in prediction.items():
            pieces[name].append(value)
    result = {name: np.concatenate(part) for name, part in pieces.items()}
    return (result, np.concatenate(latent_parts)) if latents and not classical else result


def fit_calibrators(predictions, targets, masks, specs, method="parametric"):
    """Separate calibration-month fit; pooled per-head mappings are explicit."""
    if method == "none":
        return {"method": "none", "heads": {}}
    if method not in ("parametric", "isotonic"):
        raise ValueError("unknown registered calibration method")
    heads = {}
    for spec in specs:
        name, kind = spec["name"], spec["kind"]
        if kind not in models.CLASSIFICATION_KINDS:
            continue
        y, mask, prediction = np.asarray(targets[name]), np.asarray(masks[name], bool), np.asarray(predictions[name])
        if not mask.any():
            continue
        if kind == "binary":
            actual, prob = y[mask], prediction[mask]
            if len(np.unique(actual)) < 2:
                heads[name] = {"type": "identity", "reason": "one_observed_class"}; continue
            if method == "isotonic":
                from sklearn.isotonic import IsotonicRegression
                calibrator = IsotonicRegression(out_of_bounds="clip").fit(prob, actual)
                heads[name] = {"type": "isotonic", "model": calibrator}
            else:
                from sklearn.linear_model import LogisticRegression
                logits = np.log(np.clip(prob, 1e-7, 1-1e-7)/np.clip(1-prob, 1e-7, 1-1e-7))
                calibrator = LogisticRegression(C=1e6, max_iter=200).fit(logits[:, None], actual)
                heads[name] = {"type": "platt", "coefficient": float(calibrator.coef_[0, 0]), "intercept": float(calibrator.intercept_[0])}
        else:
            p, actual = prediction[mask], y[mask].astype(int)
            if method == "isotonic":
                from sklearn.isotonic import IsotonicRegression
                mappings = []
                for k in range(p.shape[-1]):
                    binary = actual == k
                    mappings.append({"constant": float(binary.mean())} if binary.all() or not binary.any() else
                                    {"model": IsotonicRegression(out_of_bounds="clip").fit(p[:, k], binary)})
                heads[name] = {"type": "ovr_isotonic", "mappings": mappings,
                               "normalization": "renormalize_class_probabilities; original_if_all_zero"}
            else:
                from scipy.optimize import minimize_scalar
                logits = np.log(np.clip(p, 1e-7, 1))
                objective = lambda log_t: float(-np.log(np.clip(softmax(logits/np.exp(log_t), axis=-1)[np.arange(len(actual)), actual], 1e-12, 1)).mean())
                fitted = minimize_scalar(objective, bounds=(-3., 3.), method="bounded")
                heads[name] = {"type": "temperature", "temperature": float(np.exp(fitted.x))}
    return {"method": method, "pooling": "one_mapping_per_native_head_over_all_observed_coordinates", "heads": heads}


def apply_calibrators(predictions, calibrators):
    output = {name: np.asarray(value) for name, value in predictions.items()}
    for name, item in calibrators["heads"].items():
        p = output[name]
        if item["type"] == "platt":
            logit = np.log(np.clip(p, 1e-7, 1-1e-7)/np.clip(1-p, 1e-7, 1-1e-7))
            output[name] = expit(item["coefficient"]*logit+item["intercept"]).astype(np.float32)
        elif item["type"] == "isotonic":
            output[name] = item["model"].predict(p.reshape(-1)).reshape(p.shape).astype(np.float32)
        elif item["type"] == "temperature":
            output[name] = softmax(np.log(np.clip(p, 1e-7, 1))/item["temperature"], axis=-1).astype(np.float32)
        elif item["type"] == "ovr_isotonic":
            flat = p.reshape(-1, p.shape[-1]); value = np.empty_like(flat)
            for k, mapping in enumerate(item["mappings"]):
                value[:, k] = mapping["constant"] if "constant" in mapping else mapping["model"].predict(flat[:, k])
            denominator = value.sum(-1, keepdims=True)
            value = np.divide(value, denominator, out=flat.copy(), where=denominator > 0)
            output[name] = value.reshape(p.shape).astype(np.float32)
    return output


def _auxiliary(data, ids, cfg, start, end, group):
    context = int(cfg["model"].get("context", 128))
    decisions = np.asarray(ids)[:, None]-context+1+np.arange(context)[None, :]
    close = np.asarray(data["ohlcspot"][:, 3])
    next_return = np.log(close[decisions]/close[decisions-1]).astype(np.float32)
    times = np.asarray(data["index_ns"])[decisions]
    valid = (times >= pd.Timestamp(start).value) & (times < pd.Timestamp(end).value) & np.isfinite(next_return)
    if group in ("D06", "D07"):
        # These sparse inputs repeat decision-current liquidation summaries;
        # older-token return targets would then see later information.
        valid[:, :-1] = False
    return next_return, valid


def _score(data, ids, prediction, specs, cfg, train_ids, task, details=False):
    explicitly_legacy = isinstance(task.get("objectives"), list) and any(str(x).startswith("legacy") for x in task["objectives"])
    selected = specs if explicitly_legacy else [s for s in specs if s["task"].startswith("T")]
    if not selected:
        selected = specs
    filters = None
    if task.get("direction") in ("up", "down"):
        direction = 1 if task["direction"] == "up" else 0
        filters = {s["name"]: [[h, direction, k] for h in range(s["shape"][0]) for k in range(s["shape"][2])]
                   for s in selected if s["name"] in ("reach_fixed", "reach_dynamic")}
    return evaluation.score_predictions(data, ids, prediction, selected, cfg, train_ids,
                                        return_details=details, coordinate_filters=filters)


def train(cfg, task, registration_sha, arrays=None):
    """Train one registered fresh model without accessing forecast Test ids."""
    from .wm_tail_transfer_benchmark import execution_device, verify_registration
    registration = verify_registration(cfg)
    if _sha(Path(cfg["output_dir"])/"preregistration.json") != registration_sha:
        raise ValueError("forecast worker belongs to another registration")
    directory = fit_directory(cfg, task); directory.mkdir(parents=True, exist_ok=True)
    marker = directory/"fit.json"
    if marker.exists():
        fit = json.loads(marker.read_text())
        if fit.get("registration_sha256") != registration_sha or fit.get("task") != task or not fit.get("complete"):
            raise ValueError("foreign or incomplete forecast fit marker")
        if any(_sha(directory/name) != expected for name, expected in fit["artifact_sha256"].items()):
            raise ValueError("immutable forecast fit changed")
        return fit
    settings = copy.deepcopy(cfg)
    settings["device"] = execution_device(cfg, task)
    settings["loss"] = {**settings.get("loss", {}), **task.get("loss", {})}
    seed = int(task["seed"])
    torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 1)))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    data = data_module.load_data(cfg) if arrays is None else arrays
    data = attach_window_cache(cfg, data, task["group"])
    meta, ranges, splits = split_arrays(cfg, task, data, allow_empty=True)
    partition_rows = {name: len(ids) for name, ids in splits.items()}
    empty_partitions = [name for name, count in partition_rows.items() if count == 0]
    if empty_partitions:
        # Sparse free archive observations may leave a whole chronological
        # month unavailable. Record this declared condition without inventing
        # a fitted model, score or normalizer. All other split errors propagate.
        fit = {"complete": True, "available": False, "status": "unavailable_data",
               "registration_sha256": registration_sha, "task": task,
               "execution_device": settings["device"],
               "cohort": task.get("cohort", "native"), "ranges": {name: list(value) for name, value in ranges.items()},
               "reason": "empty purged chronological partition(s): " + ", ".join(empty_partitions),
               "partition_rows": partition_rows, "empty_partitions": empty_partitions,
               "actual_steps": 0, "stop_reason": "unavailable_data",
               "validation_score": None, "earlystop_score": None,
               "selected_validation_score": None, "test_read": False,
               "elapsed_seconds": time.monotonic()-started, "artifact_sha256": {}}
        _json(marker, fit)
        return fit
    objectives = task.get("objectives", task.get("variant", "joint"))
    specs = models.active_specs(models.build_specs(cfg), objectives)
    mean, std, scales = normalization(data, splits["train"], specs, task["group"], settings)
    source_obj = None
    if task.get("architecture") in ("frozen_probe", "latent_probe"):
        source_dir = Path(cfg["output_dir"])/"fits"/task["source_job"]
        source_obj = torch.load(source_dir/"selected.pt", map_location="cpu", weights_only=True)
        if source_obj["registration_sha256"] != registration_sha:
            raise ValueError("frozen probe source belongs to another registration")
        with np.load(source_dir/"norm.npz", allow_pickle=False) as source_norm:
            mean, std = source_norm["mean"].copy(), source_norm["std"].copy()
    np.savez(directory/"norm.npz", mean=mean, std=std, train_ids=splits["train"])
    if settings["loss"].get("classification_loss") in ("weighted_bce", "weighted"):
        options = settings["loss"].setdefault("heads", {})
        for spec in specs:
            if spec["kind"] == "binary":
                y, m = target_panels(data, splits["train"], [spec])
                valid = m[spec["name"]]; den = valid.sum(0)
                prior = np.divide(np.where(valid, y[spec["name"]], 0).sum(0), den, out=np.full(den.shape, .5), where=den > 0)
                options[spec["name"]] = {**options.get(spec["name"], {}), "positive_prior": prior.tolist()}
    panel = splits["earlystop"]
    maximum = int(cfg["training"].get("selection_rows", 2048))
    if len(panel) > maximum:
        panel = np.sort(rng.choice(panel, maximum, replace=False))
    native = task["model"] in ("linear", "lightgbm", "lgbm")
    history, gradient_coverage = [], defaultdict(int)
    best, best_step, last_significant, anchor = math.inf, 0, 0, math.inf
    stop_reason = "max_steps"
    if native:
        from .wm_tail_transfer_classical import fit_bundle
        train_ids = splits["train"]
        maximum = int(cfg["classical"].get("max_train_rows", 16384))
        if len(train_ids) > maximum:
            train_ids = np.sort(rng.choice(train_ids, maximum, replace=False))
        recipe = task.get("feature_recipe", cfg["classical"].get("feature_recipe", "window_summaries"))
        if task.get("architecture") == "latent_probe":
            x = _latent_tabular(cfg, task, registration_sha, data, train_ids)
            xv = _latent_tabular(cfg, task, registration_sha, data, panel)
        else:
            x = _tabular(data, train_ids, task["group"], cfg, mean, std, recipe)
            xv = _tabular(data, panel, task["group"], cfg, mean, std, recipe)
        y, m = target_panels(data, train_ids, specs); yv, mv = target_panels(data, panel, specs)
        model = fit_bundle(x, y, m, xv, yv, mv, settings, specs, scales, family=task["model"], objectives=objectives, seed=seed)
        with (directory/"bundle.pkl").open("wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        pred = _predict(model, data, panel, task, settings, mean, std, registration_sha=registration_sha)
        best = _score(data, panel, pred, specs, cfg, splits["train"], task)
        constructor = None
        model_metadata = model.get("metadata", {})
    else:
        constructor = {"class": "TailModel", "input_dim": len(mean), "cfg": settings, "specs": specs,
                       "scales": scales, "variant": task["model"], "objectives": objectives}
        if task.get("architecture") == "hurdle":
            constructor["class"] = "HurdleTailModel"
        if task.get("architecture") == "frozen_probe":
            constructor.update({"class": "FrozenTailProbe", "base_constructor": source_obj["constructor"]})
            from .wm_tail_transfer_probes import FrozenTailProbe
            base = _construct(source_obj["constructor"])
            base.load_state_dict(source_obj["state_dict"], strict=True)
            model = FrozenTailProbe(base, settings, specs, scales, objectives)
        else:
            model = _construct(constructor)
        device = settings.get("device", "mps")
        model.to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(settings["training"]["learning_rate"]),
                                     weight_decay=float(settings["training"].get("weight_decay", .0001)))
        train_ids = splits["train"]
        maximum = int(settings["training"]["max_steps"])
        for step in range(1, maximum+1):
            model.train()
            ids = rng.choice(train_ids, int(settings["training"]["batch_size"]), replace=True)
            raw = _windows(data, ids, task["group"], settings)
            x = torch.from_numpy(((raw-mean)/std).clip(-10, 10).astype(np.float32)).to(device)
            y, m = target_panels(data, ids, specs)
            target = {k: torch.from_numpy(v).to(device) for k, v in y.items()}
            mask = {k: torch.from_numpy(v).to(device) for k, v in m.items()}
            if task.get("direction") in ("up", "down"):
                direction = 1 if task["direction"] == "up" else 0
                for name in ("reach_fixed", "reach_dynamic"):
                    if name in mask:
                        mask[name] = mask[name].clone(); mask[name][:, :, 1-direction] = False
            reward, reward_mask = _auxiliary(data, ids, settings, *ranges["train"], task["group"])
            optimizer.zero_grad(set_to_none=True)
            loss, parts = model.loss(x, target, mask, base_target=torch.from_numpy(raw[..., :5].copy()).to(device),
                                     next_return=torch.from_numpy(reward).to(device), next_return_mask=torch.from_numpy(reward_mask).to(device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite native tail-transfer objective")
            loss.backward()
            for name, covered in model.supervised_gradient_coverage().items():
                gradient_coverage[name] += int(covered)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(settings["training"].get("gradient_clip", 10.)), error_if_nonfinite=True)
            optimizer.step()
            if step % int(settings["training"]["validation_every"]) == 0 or step == maximum:
                prediction = _predict(model, data, panel, task, settings, mean, std)
                details = _score(data, panel, prediction, specs, settings, train_ids, task, details=True)
                score = float(details["score"])
                if not math.isfinite(score):
                    raise RuntimeError("no finite available-task validation score")
                record = {"step": step, "score": score, "by_task": _safe(details["by_task"]),
                          "training_loss": float(loss.detach().cpu()), "gradient_norm": float(norm),
                          "elapsed_seconds": time.monotonic()-started}
                history.append(record)
                if score < best:
                    best, best_step = score, step
                    _save_neural(directory/"selected.pt", model, constructor, registration_sha, step)
                if not math.isfinite(anchor) or score < anchor-float(settings["training"].get("relative_improvement", .0005))*abs(anchor):
                    anchor, last_significant = score, step
                _json(directory/"progress.json", {"stage": "TrainVal", **record, "best_step": best_step})
                if step >= int(settings["training"]["min_steps"]) and step-last_significant >= int(settings["training"]["patience_steps"]):
                    stop_reason = "validation_plateau"; break
        obj = torch.load(directory/"selected.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(obj["state_dict"], strict=True); model.eval()
        model_metadata = model.metadata()
        if hasattr(model, "assert_backbone_unchanged"):
            model.assert_backbone_unchanged()
    # Calibrator fits exclusively on the second validation month. Its results
    # never change checkpoint selection, which used the first month only.
    calibration_ids = splits["calibration"]
    calibration_predictions = _predict(model, data, calibration_ids, task, settings, mean, std, registration_sha=registration_sha)
    yc, mc = target_panels(data, calibration_ids, specs)
    method = task.get("calibration", "none")
    calibration_methods = cfg.get("ablation", {}).get("calibration", ["none", "parametric", "isotonic"])
    all_calibrators = {name: fit_calibrators(calibration_predictions, yc, mc, specs, name) for name in calibration_methods}
    calibrators = all_calibrators.get(method)
    if calibrators is None:
        calibrators = fit_calibrators(calibration_predictions, yc, mc, specs, method)
        all_calibrators[method] = calibrators
    with (directory/"calibrators.pkl").open("wb") as f:
        pickle.dump(calibrators, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (directory/"calibrators_all.pkl").open("wb") as f:
        pickle.dump(all_calibrators, f, protocol=pickle.HIGHEST_PROTOCOL)
    selection_ids = splits["selection"]
    selected_predictions = apply_calibrators(_predict(model, data, selection_ids, task, settings, mean, std, registration_sha=registration_sha), calibrators)
    details = _score(data, selection_ids, selected_predictions, specs, settings, splits["train"], task, details=True)
    np.savez_compressed(directory/"validation_predictions.npz", raw_ids=selection_ids, **selected_predictions)
    _json(directory/"history.json", history)
    artifacts = ["norm.npz", "bundle.pkl" if native else "selected.pt", "calibrators.pkl", "calibrators_all.pkl", "validation_predictions.npz", "history.json"]
    fit = {"complete": True, "available": True, "status": "fitted",
           "registration_sha256": registration_sha, "task": task, "cohort": task.get("cohort", "native"),
           "execution_device": settings["device"],
           "ranges": ranges, "selected_step": best_step, "actual_steps": 0 if native else step,
           "stop_reason": "classical_native_selection" if native else stop_reason, "earlystop_score": float(best),
           "validation_score": float(details["score"]), "validation_metrics": _safe(details),
           "selected_validation_score": float(details["score"]), "calibration_method": method,
           "calibration_methods_available": list(all_calibrators),
           "head_specs": specs, "scales": scales, "model_metadata": _safe(model_metadata),
           "gradient_coverage": dict(gradient_coverage), "uncovered_heads": [s["name"] for s in specs if not native and gradient_coverage[s["name"]] == 0],
           "train_rows": len(splits["train"]), "fit_train_rows": len(train_ids),
           "selection_rows": len(selection_ids), "normalizer_scope": "source_frozen_Train_normalizer" if source_obj is not None else "Train_last_token_only",
           "train_ids_sha256": hashlib.sha256(np.ascontiguousarray(splits["train"]).tobytes()).hexdigest(),
           "auxiliary_return_scope": "last_token_only_current_liquidation_summary" if task["group"] in ("D06", "D07") else "all_Train_tokens_masked_before_Train_end",
           "feature_recipe": "latent_probe" if task.get("architecture") == "latent_probe" else task.get("feature_recipe", cfg.get("classical", {}).get("feature_recipe", "canonical_window")),
           "source_artifact_sha256": ({str((Path(cfg["output_dir"])/"fits"/task["source_job"]/name).resolve()): _sha(Path(cfg["output_dir"])/"fits"/task["source_job"]/name)
                                      for name in ("selected.pt", "norm.npz", "fit.json")} if source_obj is not None else {}),
           "test_read": False, "elapsed_seconds": time.monotonic()-started,
           "artifact_sha256": {name: _sha(directory/name) for name in artifacts}}
    _json(marker, _safe(fit))
    return fit


def predict(cfg, task, registration_sha, raw_ids, *, split="test", arrays=None, latents=False, calibration=True):
    """Explicit report-only Test gate, also used for chronological producers."""
    if split == "test":
        from .wm_tail_transfer_benchmark import verify_test_barrier
        barrier = verify_test_barrier(cfg)
        if barrier.get("registration_sha256") != registration_sha:
            raise ValueError("report-only Test barrier belongs to another campaign")
    elif split not in ("train", "validation", "producer"):
        raise ValueError("unknown forecast inference split")
    data = attach_window_cache(cfg, data_module.load_data(cfg) if arrays is None else arrays, task["group"])
    ids = np.asarray(raw_ids, np.int64)
    if ids.ndim != 1 or not len(ids) or np.any(np.diff(ids) <= 0):
        raise ValueError("unique chronological prediction ids required")
    if split != "test":
        from .wm_tail_transfer_benchmark import fold_definitions
        meta = next(x for x in fold_definitions(cfg) if x["fold"] == int(task["fold"]))
        if np.any(np.asarray(data["index_ns"])[ids] >= pd.Timestamp(meta["test_start"]).value):
            raise ValueError("a non-Test inference alias cannot bypass the global Test barrier")
    model, norm, fit, calibrators = load_fit(cfg, task, registration_sha)
    if isinstance(calibration, str):
        path = fit_directory(cfg, task)/"calibrators_all.pkl"
        if _sha(path) != fit["artifact_sha256"]["calibrators_all.pkl"]:
            raise ValueError("frozen calibration family changed")
        with path.open("rb") as f:
            calibrators = pickle.load(f)[calibration]
    value = _predict(model, data, ids, task, cfg, norm["mean"], norm["std"], latents=latents, registration_sha=registration_sha)
    if latents and not isinstance(model, dict):
        prediction, latent = value
        return (apply_calibrators(prediction, calibrators) if calibration else prediction), latent
    return apply_calibrators(value, calibrators) if calibration else value


def flatten_predictions(predictions, specs):
    parts, metadata, offset = [], [], 0
    for spec in specs:
        if spec["name"] not in predictions:
            continue
        value = np.asarray(predictions[spec["name"]], np.float32)
        flat = value.reshape(len(value), -1)
        parts.append(flat)
        metadata.append({"name": spec["name"], "shape": list(value.shape[1:]), "start": offset, "stop": offset+flat.shape[1]})
        offset += flat.shape[1]
    if not parts:
        raise ValueError("at least one native forecast head required")
    return np.concatenate(parts, -1), metadata


def semantic_risks(predictions, cfg):
    horizons = list(cfg["targets"]["horizons"])
    hi = horizons.index(int(cfg["targets"]["primary_horizon"]))
    threshold = min(range(len(cfg["targets"]["fixed_thresholds"])), key=lambda i: abs(cfg["targets"]["fixed_thresholds"][i]-.03))
    if "rv" not in predictions or "reach_fixed" not in predictions:
        raise ValueError("policy producer requires explicit variance and reach probabilities")
    return {"risk_sigma": np.sqrt(np.maximum(np.asarray(predictions["rv"])[:, hi], 1e-12)),
            "down_probability": np.asarray(predictions["reach_fixed"])[:, hi, 0, threshold],
            "up_probability": np.asarray(predictions["reach_fixed"])[:, hi, 1, threshold]}


class IndexedRows:
    """Read-only sparse calendar view without materializing mostly-NaN Z/F."""
    def __init__(self, raw_ids, values, calendar_rows):
        self.raw_ids, self.values = np.asarray(raw_ids, np.int64), values
        self.shape, self.dtype = (int(calendar_rows), int(values.shape[1])), np.dtype(np.float32)

    def __getitem__(self, ids):
        ids = np.asarray(ids, np.int64)
        flat = ids.reshape(-1)
        position = np.searchsorted(self.raw_ids, flat)
        valid = position < len(self.raw_ids)
        valid[valid] &= self.raw_ids[position[valid]] == flat[valid]
        result = np.full((len(flat), self.shape[1]), np.nan, np.float32)
        result[valid] = self.values[position[valid]]
        result = result.reshape(*ids.shape, self.shape[1])
        return result


class OracleRows:
    """Impossible future-truth diagnostic, strictly bounded by each split."""
    names = ("return", "mae", "mfe", "rv", "peak_dd")

    def __init__(self, data, meta, horizons):
        self.data, self.meta = data, meta
        self.shape, self.dtype = (len(data["index_ns"]), len(self.names)*len(horizons)), np.dtype(np.float32)

    def __getitem__(self, ids):
        ids = np.asarray(ids, np.int64); flat = ids.reshape(-1)
        pieces = [np.asarray(self.data["targets"][name][flat], np.float32) for name in self.names]
        result = np.concatenate(pieces, -1)
        valid = np.logical_and.reduce([np.asarray(self.data["masks"][name][flat], bool).all(1) for name in self.names])
        times = np.asarray(self.data["index_ns"])[flat]
        cutoffs = np.zeros(len(flat), np.int64)
        for start, end in (("producer_calibration_start", "producer_calibration_end"),
                           ("policy_train_start", "policy_train_end"), ("val_start", "val_end"), ("test_start", "test_end")):
            lower, upper = pd.Timestamp(self.meta[start]).value, pd.Timestamp(self.meta[end]).value
            cutoffs[(times >= lower) & (times < upper)] = upper
        predictive = np.asarray(self.data["label_end_ns"])[flat].max(1)
        passive = np.asarray(self.data.get("passive_label_end_ns", self.data["label_end_ns"]))[flat].max(1)
        valid &= np.maximum(predictive, passive) < cutoffs
        result[~valid] = np.nan
        return result.reshape(*ids.shape, self.shape[1])


def _uncertainty_vector(predictions, cfg):
    hi = list(cfg["targets"]["horizons"]).index(int(cfg["targets"]["primary_horizon"]))
    risk = semantic_risks(predictions, cfg)
    return np.column_stack((np.asarray(predictions["return"])[:, hi], risk["risk_sigma"],
                            risk["up_probability"], risk["down_probability"], np.asarray(predictions["mae"])[:, hi]))


def prepare_producer_cache(cfg, fold, seed, registration_sha, sources, *, split="train_val", arrays=None):
    """Fixed producer models export semantic F/Z for later policy learning.

    Uncertainty is ensemble forecast dispersion, scaled in producer-Train
    units and ranked against the producer calibration month. It is not a
    posterior probability or a separate independent sample of the market.
    """
    from .wm_tail_transfer_benchmark import fold_definitions, split_ids, verify_registration, verify_test_barrier
    if split not in ("train_val", "test"):
        raise ValueError("producer export split must be train_val or report-only test")
    verify_registration(cfg)
    if split == "test":
        if verify_test_barrier(cfg)["registration_sha256"] != registration_sha:
            raise ValueError("producer Test export requires the final selection barrier")
    directory = Path(cfg["output_dir"])/"producer_caches"/f"fold_{int(fold):02d}_s{int(seed)}"/split
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/"manifest.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        if result.get("registration_sha256") != registration_sha or result.get("sources") != sources:
            raise ValueError("foreign producer feature cache")
        if any(_sha(directory/name) != expected for name, expected in result["artifact_sha256"].items()):
            raise ValueError("frozen producer features changed")
        return result
    data = attach_window_cache(cfg, data_module.load_data(cfg) if arrays is None else arrays, cfg["policy"].get("source_input_group", "D03"))
    meta = next(x for x in fold_definitions(cfg) if x["fold"] == int(fold))
    start, end = (meta["producer_calibration_start"], meta["val_end"]) if split == "train_val" else (meta["test_start"], meta["test_end"])
    group = cfg["policy"].get("source_input_group", "D03")
    ids = split_ids(data, group, start, end, purge=False, require_labels=False)
    inference_split = "test" if split == "test" else "producer"
    old_prediction, old_z = predict(cfg, sources["old"], registration_sha, ids, split=inference_split, arrays=data, latents=True)
    tail_prediction, new_z = predict(cfg, sources["tail"], registration_sha, ids, split=inference_split, arrays=data, latents=True)
    forecast_task = sources["forecast"]
    calibration_method = cfg["policy"].get("producer_calibration", "parametric")
    prediction = predict(cfg, forecast_task, registration_sha, ids, split=inference_split, arrays=data, calibration=calibration_method)
    _, _, forecast_fit, _ = load_fit(cfg, forecast_task, registration_sha)
    forecasts, forecast_columns = flatten_predictions(prediction, forecast_fit["head_specs"])
    risk = semantic_risks(prediction, cfg)
    members = sources.get("ensemble", [sources["tail"]])
    if len(members) < 2:
        raise ValueError("uncertainty exports require at least two predeclared producer members")
    ensemble = []
    for task in members:
        p = tail_prediction if task == sources["tail"] else predict(cfg, task, registration_sha, ids, split=inference_split, arrays=data)
        ensemble.append(_uncertainty_vector(p, cfg))
    _, _, tail_fit, _ = load_fit(cfg, sources["tail"], registration_sha)
    hi = list(cfg["targets"]["horizons"]).index(int(cfg["targets"]["primary_horizon"]))
    units = np.array([np.asarray(tail_fit["scales"]["return"])[hi], math.sqrt(np.asarray(tail_fit["scales"]["rv"])[hi]),
                      1., 1., np.asarray(tail_fit["scales"]["mae"])[hi]], np.float64).clip(1e-8)
    uncertainty = (np.std(np.stack(ensemble), axis=0, ddof=1)/units).astype(np.float32)
    raw_score = np.sqrt(np.mean(uncertainty*uncertainty, axis=1))
    if split == "train_val":
        cal = (np.asarray(data["index_ns"])[ids] >= pd.Timestamp(meta["producer_calibration_start"]).value) & (np.asarray(data["index_ns"])[ids] < pd.Timestamp(meta["producer_calibration_end"]).value)
        reference = np.sort(raw_score[cal])
        if not len(reference):
            raise ValueError("producer calibration month must support uncertainty ranking")
        np.save(directory/"uncertainty_reference.npy", reference, allow_pickle=False)
    else:
        source = directory.parent/"train_val"
        old_manifest = json.loads((source/"manifest.json").read_text())
        if old_manifest["registration_sha256"] != registration_sha or _sha(source/"uncertainty_reference.npy") != old_manifest["artifact_sha256"]["uncertainty_reference.npy"]:
            raise ValueError("Train-only uncertainty calibration reference changed")
        reference = np.load(source/"uncertainty_reference.npy", allow_pickle=False)
    score = np.searchsorted(reference, raw_score, side="right")/len(reference)
    values = {"raw_ids": ids, "old_z": old_z, "new_z": new_z, "forecasts": forecasts, "uncertainty": uncertainty,
              "risk_sigma": risk["risk_sigma"], "up_probability": risk["up_probability"], "down_probability": risk["down_probability"],
              "uncertainty_score": score.astype(np.float32)}
    for name, value in values.items():
        if not np.isfinite(value).all():
            raise ValueError("producer caches require finite causal forecasts and states")
        np.save(directory/(name+".npy"), value, allow_pickle=False)
    cal_ids = split_ids(data, group, meta["producer_calibration_start"], meta["producer_calibration_end"])
    label_end = int(max(np.asarray(data["label_end_ns"])[cal_ids].max(), np.asarray(data.get("passive_label_end_ns", data["label_end_ns"]))[cal_ids].max()))
    if label_end >= pd.Timestamp(meta["policy_train_start"]).value:
        raise ValueError("producer calibration outcomes overlap policy Train")
    files = [name+".npy" for name in values]+(["uncertainty_reference.npy"] if split == "train_val" else [])
    source_tasks = [sources["old"], sources["tail"], sources["forecast"], *members]
    result = {"complete": True, "registration_sha256": registration_sha, "fold": int(fold), "seed": int(seed),
              "split": split, "directory": str(directory.resolve()), "sources": sources, "n_rows": len(ids),
              "calendar_rows": len(data["index_ns"]), "forecast_columns": forecast_columns,
              "producer_label_end_ns": label_end, "forecast_calibration": calibration_method,
              "uncertainty": "producer-model ensemble disagreement; Train-scaled; calibration-month ECDF",
              "uncertainty_components": ["return", "sigma", "up3pct", "down3pct", "mae"],
              "source_fit_sha256": {task["job"]: _sha(fit_directory(cfg, task)/"fit.json") for task in source_tasks},
              "artifact_sha256": {name: _sha(directory/name) for name in files}, "test_read": split == "test"}
    _json(marker, result)
    return result


def load_producer_cache(cfg, manifest, seed=None, *, split="train_val", arrays=None):
    from .wm_tail_transfer_benchmark import fold_definitions
    if isinstance(manifest, (int, np.integer)):
        if seed is None:
            raise ValueError("producer cache fold lookup requires a seed")
        manifest = Path(cfg["output_dir"])/"producer_caches"/f"fold_{int(manifest):02d}_s{int(seed)}"/split/"manifest.json"
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    if not manifest.get("complete"):
        raise ValueError("complete producer feature cache required")
    directory = Path(manifest["directory"])
    for name, expected in manifest["artifact_sha256"].items():
        if _sha(directory/name) != expected:
            raise ValueError("frozen producer feature cache changed")
    data = data_module.load_data(cfg) if arrays is None else arrays
    n = len(data["index_ns"])
    ids = np.load(directory/"raw_ids.npy", mmap_mode="r")
    market = {"raw": data["input_"+cfg["policy"].get("source_input_group", "D03")]}
    for name in ("old_z", "new_z", "forecasts", "uncertainty"):
        market[name] = IndexedRows(ids, np.load(directory/(name+".npy"), mmap_mode="r"), n)
    available = np.zeros(n, bool); available[ids] = True
    output = {**data, "market": market, "available": available, "producer_label_end_ns": manifest["producer_label_end_ns"]}
    for name in ("risk_sigma", "up_probability", "down_probability", "uncertainty_score"):
        values = np.zeros(n, np.float32); values[ids] = np.load(directory/(name+".npy"), mmap_mode="r")
        output[name] = values
    meta = next(x for x in fold_definitions(cfg) if x["fold"] == manifest["fold"])
    market["oracle"] = OracleRows(data, meta, cfg["targets"]["horizons"])
    index = np.asarray(data["index_ns"])
    for key, start, end in (("train_ids", "policy_train_start", "policy_train_end"), ("val_ids", "val_start", "val_end"), ("test_ids", "test_start", "test_end")):
        output[key] = np.flatnonzero((index >= pd.Timestamp(meta[start]).value) & (index < pd.Timestamp(meta[end]).value))
    return attach_window_cache(cfg, output, cfg["policy"].get("source_input_group", "D03"))
