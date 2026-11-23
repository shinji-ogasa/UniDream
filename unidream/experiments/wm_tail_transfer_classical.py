"""Train/Val-only affine and native LightGBM tail-risk controls.

Independent scalar fits are packaged together for I/O efficiency. They never
constitute a shared-representation regression/classification joint model.
VaR/ES uses its proper joint score, a different meaning of joint. Hazard
controls pool observed risk sets with explicit time/threshold covariates.
"""
from __future__ import annotations

from collections import defaultdict
import math
import time

import numpy as np
from scipy.special import expit, softmax
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from .wm_tail_transfer_models import (
    active_specs, build_specs, output_shape, _scale_tensor, decode_head, head_loss,
)


def fz0_numpy(raw, actual, alpha):
    """FZ0 in standardized-return units, with smooth negative q/e links."""
    raw = np.asarray(raw, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    q = -(np.logaddexp(0., raw[:, 0]) + 1e-5)
    e = q - np.logaddexp(0., raw[:, 1]) - 1e-5
    score = -(actual <= q).astype(float) * (q - actual) / (float(alpha) * e) + q / e + np.log(-e) - 1.
    return score, q, e


def fz0_gradient_hessian(raw, actual, alpha):
    """Analytic exact first derivative and positive diagonal Newton surrogate.

    FZ0 is not globally convex in the linked parameters. The Newton matrix is
    explicitly a bounded positive gradient-magnitude surrogate, not claimed
    to be the true Hessian. This preserves the actual FZ0 gradient and avoids
    silently presenting a two-stage pseudo-target fit as joint FZ0 learning.
    """
    raw = np.asarray(raw, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    _, q, e = fz0_numpy(raw, y, alpha)
    event = (y <= q).astype(float)
    dq = (1. - event / alpha) / e
    de = (event * (q - y) / alpha - q + e) / np.square(e)
    gradient = np.column_stack((-(dq + de) * expit(raw[:, 0]), -de * expit(raw[:, 1])))
    hessian = np.clip(np.abs(gradient), .05, 100.)
    return gradient, hessian


def _inputs(x, name, width=None):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not len(x) or (width is not None and x.shape[1] != width) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite nonempty [rows,features]")
    return x


def _targets(values, masks, n, specs):
    result, availability = {}, {}
    for spec in specs:
        name = spec["name"]
        y = np.asarray(values[name])
        m = np.asarray(masks[name])
        if y.shape != (n, *spec["shape"]) or m.shape != y.shape or m.dtype != bool:
            raise ValueError(f"{name} has invalid target/mask shape or mask dtype")
        if not np.isfinite(y[m]).all():
            raise ValueError(f"{name} observed targets must be finite")
        if spec["kind"] in ("binary", "categorical", "hazard"):
            classes = 2 if spec["kind"] == "binary" else int(spec.get("classes", 3))
            if np.any((y[m] != np.floor(y[m])) | (y[m] < 0) | (y[m] >= classes)):
                raise ValueError(f"{name} has invalid class ids")
        result[name] = np.where(m, y, 0.).astype(np.float32)
        availability[name] = m
    return result, availability


class _AffineHeads(nn.Module):
    """Independent affine/logistic/GLM heads with no learned common encoder."""
    def __init__(self, width, specs, scales, cfg):
        super().__init__()
        self.specs, self.options = specs, cfg.get("loss", {})
        self.heads = nn.ModuleDict()
        for i, spec in enumerate(specs):
            self.heads[spec["name"]] = nn.Linear(width, math.prod(output_shape(spec)))
            nn.init.zeros_(self.heads[spec["name"]].weight)
            nn.init.zeros_(self.heads[spec["name"]].bias)
            self.register_buffer(f"scale_{i}", _scale_tensor(spec, scales))

    def outputs(self, x, ordered=False):
        raw, prediction = {}, {}
        for i, spec in enumerate(self.specs):
            name = spec["name"]
            r = self.heads[name](x).reshape(len(x), *output_shape(spec))
            scale = getattr(self, f"scale_{i}")
            raw[name] = r
            if spec["kind"] in ("quantile", "positive_quantile"):
                p = r * scale[..., :1]
                if ordered:
                    p = p.sort(-1).values
                    if spec["kind"] == "positive_quantile":
                        p = p.clamp_min(0.)
                prediction[name] = p
            else:
                prediction[name] = decode_head(r, spec, scale)
        return prediction, raw

    def loss(self, x, targets, masks, regularization=0.):
        prediction, raw = self.outputs(x)
        families = defaultdict(list)
        for i, spec in enumerate(self.specs):
            name = spec["name"]
            options = {**self.options, **self.options.get("heads", {}).get(name, {})}
            value, valid = head_loss(raw[name], prediction[name], targets[name], masks[name], spec, getattr(self, f"scale_{i}"), options)
            if bool(valid.any()):
                families[spec["task"]].append(torch.where(valid, value, 0.).sum() / valid.sum())
        if not families:
            raise ValueError("no observed classical training targets")
        value = torch.stack([torch.stack(parts).mean() for parts in families.values()]).mean()
        if regularization:
            value = value + regularization * torch.stack([head.weight.square().mean() for head in self.heads.values()]).mean()
        return value


def _fit_linear(x, y, mask, xv, yv, mv, cfg, specs, scales, seed):
    cc = cfg.get("classical", {})
    mean, std = x.mean(0, dtype=np.float64), x.std(0, dtype=np.float64)
    std[std < 1e-7] = 1.
    xt = torch.from_numpy(((x - mean) / std).astype(np.float32))
    vt = torch.from_numpy(((xv - mean) / std).astype(np.float32))
    yt = {k: torch.from_numpy(v) for k, v in y.items()}
    mt = {k: torch.from_numpy(v) for k, v in mask.items()}
    vy = {k: torch.from_numpy(v) for k, v in yv.items()}
    vm = {k: torch.from_numpy(v) for k, v in mv.items()}
    torch.manual_seed(seed)
    model = _AffineHeads(x.shape[1], specs, scales, cfg)
    # Exact Train priors initialize rare binary intercepts without a test fit.
    with torch.no_grad():
        for spec in specs:
            if spec["kind"] != "binary":
                continue
            name = spec["name"]
            count = mask[name].sum(0)
            prior = np.divide((y[name] * mask[name]).sum(0), count, out=np.full(count.shape, .5), where=count > 0)
            prior = np.clip(prior, 1e-5, 1 - 1e-5)
            model.heads[name].bias.copy_(torch.from_numpy(np.log(prior / (1 - prior)).ravel().astype(np.float32)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cc.get("linear_learning_rate", .02)))
    rng = np.random.default_rng(seed)
    maximum = int(cc.get("linear_steps", 400))
    every = int(cc.get("linear_validation_every", 40))
    patience = int(cc.get("linear_patience", 120))
    batch = min(int(cc.get("linear_batch_size", 1024)), len(x))
    best, best_state, best_step, history = np.inf, None, 0, []
    for step in range(1, maximum + 1):
        ids = rng.choice(len(x), batch, replace=False)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(xt[ids], {k: v[ids] for k, v in yt.items()}, {k: v[ids] for k, v in mt.items()}, float(cc.get("linear_alpha", 100.)) / len(x))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite affine control objective")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if step % every == 0 or step == maximum:
            with torch.no_grad():
                # Chunking keeps the hazard output memory bounded. Combining
                # chunk means is a fixed-row approximation, reported below.
                chunk_values = []
                for start in range(0, len(vt), 2048):
                    ids_v = slice(start, start + 2048)
                    score = model.loss(vt[ids_v], {k: v[ids_v] for k, v in vy.items()}, {k: v[ids_v] for k, v in vm.items()})
                    chunk_values.append((min(2048, len(vt) - start), float(score)))
                score = sum(n * v for n, v in chunk_values) / len(vt)
            history.append(dict(step=step, validation_score=score, training_loss=float(loss.detach())))
            if score < best:
                best, best_step = score, step
                best_state = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
            if step - best_step >= patience:
                break
    if best_state is None:
        raise RuntimeError("affine fit has no selected validation checkpoint")
    return dict(kind="affine", mean=mean, std=std, state=best_state, cfg=cfg,
        metadata=dict(steps=step, selected_step=best_step, validation_score=best, history=history,
            objective="native independent affine/ridge/logistic/pinball/FZ0; no learned shared representation",
            quantile_link="independent affine quantiles; deterministic prediction-time monotone rearrangement",
            variance_link="softplus generalized-linear mean", validation_reduction="row-weighted chunk task means",
            optimizer="Adam, finite fixed budget, not a convergence guarantee"))


def _native_score(prediction, y, kind, level=None):
    p = np.asarray(prediction)
    y = np.asarray(y)
    if kind == "binary":
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))
    if kind in ("categorical", "hazard"):
        return float(-np.log(np.clip(p[np.arange(len(y)), y.astype(int)], 1e-9, 1)).mean())
    if kind == "variance":
        ratio = np.maximum(y, 1e-12) / np.maximum(p, 1e-12)
        return float(np.mean(ratio - np.log(ratio) - 1))
    if kind in ("quantile", "positive_quantile"):
        error = y - p
        return float(np.mean(np.maximum(level * error, (level - 1) * error)))
    return float(np.mean(np.square(p - y)))


def _fit_booster(x, y, xv, yv, kind, cfg, level=None, seed=7):
    import lightgbm as lgb
    cc = cfg.get("classical", {})
    params = dict(learning_rate=float(cc.get("lightgbm_learning_rate", .05)),
        num_leaves=int(cc.get("lightgbm_leaves", 15)), min_data_in_leaf=int(cc.get("lightgbm_min_data_leaf", 128)),
        lambda_l2=float(cc.get("lightgbm_l2", 10.)), num_threads=int(cc.get("threads", 1)),
        seed=seed, deterministic=True, force_col_wise=True, verbosity=-1,
        feature_pre_filter=False, metric="None", max_bin=int(cc.get("lightgbm_max_bin", 63)))
    unique = np.unique(y)
    if kind in ("binary", "categorical", "hazard") and len(unique) == 1:
        classes = 2 if kind == "binary" else 3
        value = float(unique[0]) if kind == "binary" else np.eye(classes)[int(unique[0])]
        return dict(kind="constant", value=value, classes=classes), dict(single_class=True, train_class=int(unique[0]), observed_rounds=0)
    train_weights = None
    options = cfg.get("loss", {})
    custom_binary = False
    log_variance = False
    if kind == "binary":
        params["objective"] = "binary"
        if options.get("classification_loss", "bce") in ("weighted_bce", "weighted"):
            prior = np.clip(y.mean(), 1e-4, 1 - 1e-4)
            train_weights = np.where(y > .5, .5 / prior, .5 / (1 - prior))
        elif options.get("classification_loss") == "focal":
            gamma = float(options.get("focal_gamma", 2.))

            def focal_objective(raw, dataset):
                signs = 2 * dataset.get_label() - 1
                pt = np.clip(expit(signs * raw), 1e-12, 1 - 1e-12)
                gradient = signs * (1 - pt) ** gamma * (gamma * pt * np.log(pt) - (1 - pt))
                return gradient, np.clip(np.abs(gradient), .01, 10.)

            params["objective"] = focal_objective
            custom_binary = True
    elif kind in ("categorical", "hazard"):
        params.update(objective="multiclass", num_class=3)
    elif kind == "variance":
        y, yv = np.maximum(y, 1e-12), np.maximum(yv, 1e-12)
        if options.get("variance_loss", "qlike") in ("log_mse", "logvar_mse"):
            params["objective"] = "regression"
            y, yv = np.log(y), np.log(yv)
            log_variance = True
        else:
            params["objective"] = "gamma"
    elif kind in ("quantile", "positive_quantile"):
        params.update(objective="quantile", alpha=float(level))
    elif kind == "var_es":
        def objective(raw, dataset):
            return fz0_gradient_hessian(raw, dataset.get_label(), float(level))
        params.update(objective=objective, num_class=2)
    else:
        params["objective"] = "regression"
    train = lgb.Dataset(x, label=y, weight=train_weights, params=params)
    validation = lgb.Dataset(xv, label=yv, reference=train, params=params)
    history = []

    def metric(raw, dataset):
        actual = dataset.get_label()
        if kind == "var_es":
            score = float(fz0_numpy(raw, actual, level)[0].mean())
        elif custom_binary:
            score = _native_score(expit(raw), actual, kind, level)
        elif log_variance:
            score = _native_score(np.exp(raw), np.exp(actual), kind, level)
        else:
            score = _native_score(raw, actual, kind, level)
        history.append(score)
        return "native_validation", score, False

    booster = lgb.train(params, train, num_boost_round=int(cc.get("lightgbm_rounds", 200)),
        valid_sets=[validation], feval=metric,
        callbacks=[lgb.early_stopping(int(cc.get("lightgbm_patience", 30)), first_metric_only=True, verbose=False)])
    selected = int(np.argmin(history)) + 1
    # Detached text has no local Python callback closure and is safely picklable.
    text = booster.model_to_string(num_iteration=selected)
    return dict(kind="booster", model_text=text, selected_iteration=selected,
                native_kind=kind, level=level, custom_binary=custom_binary,
                log_variance=log_variance), dict(single_class=False,
                    selected_iteration=selected, observed_rounds=len(history), validation_score=min(history))


def _booster_predict(head, x):
    if head["kind"] == "constant":
        value = np.asarray(head["value"])
        return np.broadcast_to(value, (len(x), *value.shape)).copy()
    import lightgbm as lgb
    booster = lgb.Booster(model_str=head["model_text"])
    if head["native_kind"] == "var_es":
        raw = booster.predict(x, raw_score=True, num_threads=1)
        _, q, e = fz0_numpy(raw, np.zeros(len(x)), head["level"])
        return np.column_stack((q, e))
    value = booster.predict(x, raw_score=True, num_threads=1) if head.get("custom_binary") else booster.predict(x, num_threads=1)
    if head.get("custom_binary"):
        value = expit(value)
    if head.get("log_variance"):
        value = np.exp(value)
    return value


def _fit_lightgbm(x, y, mask, xv, yv, mv, cfg, specs, scales, seed):
    cc = cfg.get("classical", {})
    fitted, info = {}, []
    for spec in specs:
        name, kind = spec["name"], spec["kind"]
        started = time.monotonic()
        scale = _scale_tensor(spec, scales).numpy()
        native = []
        if kind == "hazard":
            # A pooled, time-dependent competing-risk model avoids 384 almost
            # empty independent fits. Every training row is an observed risk
            # set; trajectory right-censoring is preserved in the sample mask.
            def riskset(features, values, available, maximum=None):
                rows, at, threshold = np.nonzero(available)
                if maximum is not None and len(rows) > maximum:
                    selected = np.sort(np.random.default_rng(seed).choice(len(rows), maximum, replace=False))
                    rows, at, threshold = rows[selected], at[selected], threshold[selected]
                design = np.column_stack((features[rows], at / max(1, spec["shape"][0] - 1), threshold / max(1, spec["shape"][1] - 1))).astype(np.float32)
                return design, values[rows, at, threshold]
            hx, hy = riskset(x, y[name], mask[name], int(cc.get("max_train_rows", 16384)))
            hvx, hvy = riskset(xv, yv[name], mv[name], int(cc.get("hazard_max_val_risksets", 65536)))
            if not len(hx) or not len(hvx):
                raise ValueError("hazard has no observed Train/Val risk sets")
            head, metadata = _fit_booster(hx, hy, hvx, hvy, kind, cfg, seed=seed)
            fitted[name] = dict(kind="hazard_pool", head=head)
            info.append(dict(name=name, cells=1, elapsed_seconds=time.monotonic() - started,
                             train_risksets=len(hx), val_risksets=len(hvx), **metadata))
            continue
        for cell in np.ndindex(*spec["shape"]):
            cell_started = time.monotonic()
            mt, me = mask[name][(slice(None), *cell)], mv[name][(slice(None), *cell)]
            actual, valid_actual = y[name][(slice(None), *cell)][mt], yv[name][(slice(None), *cell)][me]
            if not len(actual) or not len(valid_actual):
                # Keep the declared cell visible, but flag unavailable fitting.
                # Evaluation must retain its original masks and report N/A.
                if kind == "var_es":
                    fallback = np.array([-1., -2.])
                elif kind == "categorical":
                    counts = np.bincount(actual.astype(int), minlength=3) if len(actual) else np.ones(3)
                    fallback = counts / counts.sum()
                elif kind in ("quantile", "positive_quantile") and len(actual):
                    fallback = np.quantile(actual / float(scale[cell]), spec["levels"][cell[-1]])
                else:
                    fallback = float(actual.mean()) / (1. if kind == "binary" else float(scale[cell])) if len(actual) else .0
                native.append(dict(cell=cell, scale=float(scale[cell]), head=dict(kind="constant", value=fallback)))
                info.append(dict(name=name, cell=list(cell), single_class=True,
                    unavailable_fit=True, train_observed=len(actual), validation_observed=len(valid_actual)))
                continue
            unit = float(scale[cell])
            native_kind = kind
            level = spec.get("levels", [None])[cell[-1]] if kind in ("quantile", "positive_quantile", "var_es") else None
            normalized = actual if kind in ("binary", "categorical") else actual / unit
            normalized_val = valid_actual if kind in ("binary", "categorical") else valid_actual / unit
            head, metadata = _fit_booster(x[mt], normalized, xv[me], normalized_val, native_kind, cfg, level, seed)
            native.append(dict(cell=cell, scale=unit, head=head))
            info.append(dict(name=name, cell=list(cell), elapsed_seconds=time.monotonic() - cell_started, **metadata))
        fitted[name] = dict(kind="cells", heads=native)
    return dict(kind="lightgbm", fitted=fitted, metadata=dict(
        native_heads=info, fitted_boosters=sum(1 for i in info if not i.get("single_class", False)),
        constant_heads=sum(1 for i in info if i.get("single_class", False)),
        hazard="pooled competing risks with explicit time/threshold; Train/Val riskset subsampling is fixed",
        var_es="proper FZ0 native two-output objective; exact gradient, bounded positive Newton surrogate",
        quantile_link="native independent quantile boosters; deterministic monotone rearrangement"))


def fit_bundle(x_train, targets_train, masks_train, x_val, targets_val, masks_val,
               cfg, specs=None, scales=None, family="linear", objectives="joint", seed=7):
    """Fit exclusively from explicit Train/Val arguments; no Test path exists."""
    started = time.monotonic()
    specs = active_specs(build_specs(cfg) if specs is None else specs, objectives)
    scales = {} if scales is None else dict(scales)
    x = _inputs(x_train, "Train")
    xv = _inputs(x_val, "Val", x.shape[1])
    y, mask = _targets(targets_train, masks_train, len(x), specs)
    yv, mv = _targets(targets_val, masks_val, len(xv), specs)
    maximum = int(cfg.get("classical", {}).get("max_train_rows", 16384))
    if len(x) > maximum:
        ids = np.sort(np.random.default_rng(seed).choice(len(x), maximum, replace=False))
        x, y, mask = x[ids], {k: v[ids] for k, v in y.items()}, {k: v[ids] for k, v in mask.items()}
    else:
        ids = np.arange(len(x))
    with threadpool_limits(limits=int(cfg.get("classical", {}).get("threads", 1))):
        if family == "linear":
            result = _fit_linear(x, y, mask, xv, yv, mv, cfg, specs, scales, seed)
        elif family in ("lightgbm", "lgbm"):
            result = _fit_lightgbm(x, y, mask, xv, yv, mv, cfg, specs, scales, seed)
        else:
            raise ValueError("classical family must be linear or lightgbm")
    return dict(family=family, input_dim=x.shape[1], specs=specs, scales=scales, model=result,
        metadata=dict(elapsed_seconds=time.monotonic() - started, original_train_rows=len(x_train),
            train_rows=len(x), validation_rows=len(xv), sampled_train_rows=ids.tolist(),
            true_joint_representation=False, no_test_used=True, native_fit=result["metadata"]))


def predict_bundle(bundle, x):
    x = _inputs(x, "Prediction", int(bundle["input_dim"]))
    model, specs = bundle["model"], bundle["specs"]
    if model["kind"] == "affine":
        network = _AffineHeads(x.shape[1], specs, bundle["scales"], model["cfg"])
        network.load_state_dict({k: torch.from_numpy(v) for k, v in model["state"].items()})
        normalized = ((x - model["mean"]) / model["std"]).astype(np.float32)
        pieces = defaultdict(list)
        with torch.no_grad():
            for start in range(0, len(x), 2048):
                result, _ = network.outputs(torch.from_numpy(normalized[start:start + 2048]), ordered=True)
                for name, value in result.items():
                    pieces[name].append(value.numpy())
        return {name: np.concatenate(values) for name, values in pieces.items()}
    result = {}
    for spec in specs:
        name = spec["name"]
        fitted = model["fitted"][name]
        output = np.empty((len(x), *output_shape(spec)), dtype=np.float64)
        if fitted["kind"] == "hazard_pool":
            at, threshold = np.indices(spec["shape"])
            for start in range(0, len(x), 256):
                features = x[start:start + 256]
                design = np.column_stack((np.repeat(features, at.size, axis=0),
                    np.tile(at.ravel() / max(1, spec["shape"][0] - 1), len(features)),
                    np.tile(threshold.ravel() / max(1, spec["shape"][1] - 1), len(features))))
                output[start:start + len(features)] = _booster_predict(fitted["head"], design).reshape(len(features), *output_shape(spec))
        else:
            for entry in fitted["heads"]:
                value = _booster_predict(entry["head"], x)
                if spec["kind"] not in ("binary", "categorical"):
                    value = value * entry["scale"]
                output[(slice(None), *entry["cell"])] = value
            if spec["kind"] in ("quantile", "positive_quantile"):
                output.sort(axis=-1)
            if spec["kind"] in ("positive", "positive_quantile", "variance"):
                output = np.maximum(output, 1e-12 if spec["kind"] == "variance" else 0.)
            if spec["kind"] == "bounded":
                output = np.clip(output, -1., 1.)
        if not np.isfinite(output).all():
            raise RuntimeError(f"nonfinite {name} prediction")
        result[name] = output
    return result
