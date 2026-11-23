"""Causal, heterogeneous tail-risk predictors for an isolated research study.

The public specs describe *target* shapes, excluding batch/time. Categorical
predictions append a class axis; VaR/ES appends (quantile, expected-shortfall).
Every head has its own availability mask. Losses average within a head, then
within a task, then over available tasks. No study artifact is opened here.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td

from unidream.world_model.encoder import ObsDecoder, OneHotDist, symlog
from unidream.world_model.transformer import TransformerWorldModel, twohot_loss
from .wm_ablation_model import _AttentionBlock


CLASSIFICATION_KINDS = frozenset(("binary", "categorical", "hazard"))


def build_specs(cfg: Mapping) -> list[dict]:
    """Return the JSON-serializable, target-major output contract."""
    t = cfg.get("targets", cfg)
    h = len(t.get("horizons", [1, 4, 16, 48, 96]))
    nf = len(t.get("fixed_thresholds", [.01, .02, .03, .05]))
    nd = len(t.get("dynamic_multipliers", [1., 2., 3.]))
    nq = t.get("return_quantiles", [.01, .05, .5, .95, .99])
    pq = t.get("positive_quantiles", [.5, .9, .95, .99])
    alphas = t.get("es_alphas", [.01, .05])
    nl = len(t.get("passive_leverages", [.5, 1., 1.5, 2., 3.]))
    ns = len(t.get("passive_loss_thresholds", [.05, .1, .2]))
    specs = []

    def add(name, kind, shape, task, levels=None, **extra):
        specs.append(dict(name=name, kind=kind, shape=list(shape), task=task,
                          levels=levels, **extra))

    add("return", "real", [h], "T01")
    add("return_class", "categorical", [h, 2], "T01", classes=3)
    add("reach_fixed", "binary", [h, 2, nf], "T02")
    add("reach_dynamic", "binary", [h, 2, nd], "T02")
    add("barrier", "categorical", [h, len(t.get("barrier_ratios", [[1, 1], [2, 1], [1, 2]])), 2], "T03", classes=3)
    add("mfe", "positive", [h], "T04")
    add("mfe_quantiles", "positive_quantile", [h, len(pq)], "T04", list(pq), scale_key="mfe")
    add("mae", "positive", [h], "T05")
    add("mae_quantiles", "positive_quantile", [h, len(pq)], "T05", list(pq), scale_key="mae")
    add("peak_dd", "positive", [h], "T06")
    add("peak_dd_quantiles", "positive_quantile", [h, len(pq)], "T06", list(pq), scale_key="peak_dd")
    add("rv", "variance", [h], "T07")
    add("up_semivariance", "variance", [h], "T08")
    add("down_semivariance", "variance", [h], "T08")
    add("return_quantiles", "quantile", [h, len(nq)], "T09", list(nq), scale_key="return")
    add("var_es", "var_es", [h, len(alphas)], "T10", list(alphas), scale_key="return")
    add("time_hazard", "hazard", [max(t.get("horizons", [1, 4, 16, 48, 96])), len(t.get("hazard_thresholds", [.01, .02, .03, .05]))], "T11", classes=3)
    add("passive_account_return", "real", [h, nl], "T12")
    add("loss_probability", "binary", [h, nl, ns], "T12")
    add("ruin", "binary", [h, nl], "T12")
    if t.get("include_perpetual", False):
        add("perp_liquidation", "binary", [h, nl], "T12")
        add("perp_passive_return", "real", [h, nl], "T12")
    if t.get("include_legacy", True):
        add("legacy_downside", "positive", [h], "legacy_downside")
        add("crash", "binary", [h], "legacy_crash")
        add("dd_excess", "positive", [h], "legacy_dd_excess")
        add("advantage", "real", [h], "legacy_advantage")
        add("recovery", "bounded", [h], "legacy_recovery")
    return specs


def output_shape(spec: Mapping) -> tuple[int, ...]:
    shape = tuple(int(v) for v in spec["shape"])
    if spec["kind"] in ("categorical", "hazard"):
        shape += (int(spec.get("classes", 3)),)
    elif spec["kind"] == "var_es":
        shape += (2,)
    return shape


def active_specs(specs, objectives="joint"):
    if objectives in ("joint", "all", None):
        return [dict(s) for s in specs]
    if objectives == "classification":
        return [dict(s) for s in specs if s["kind"] in CLASSIFICATION_KINDS]
    if objectives == "regression":
        return [dict(s) for s in specs if s["kind"] not in CLASSIFICATION_KINDS]
    wanted = {objectives} if isinstance(objectives, str) else set(objectives)
    selected = [dict(s) for s in specs if s["name"] in wanted or s["task"] in wanted]
    if not selected:
        raise ValueError("the objective selection contains no registered heads")
    return selected


def _scale_tensor(spec, scales):
    value = np.asarray(scales.get(spec["name"], scales.get(spec.get("scale_key", spec["name"]), 1.)), dtype=np.float32)
    shape = tuple(spec["shape"])
    if value.ndim == 1 and len(shape) > 1 and value.shape == (shape[0],):
        value = value.reshape((shape[0],) + (1,) * (len(shape) - 1))
    try:
        value = np.broadcast_to(value, shape).copy()
    except ValueError as error:
        raise ValueError(f"{spec['name']} scale cannot broadcast to {shape}") from error
    if not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError(f"{spec['name']} scales must be positive finite Train-only values")
    return torch.from_numpy(np.maximum(value, 1e-10 if spec["kind"] == "variance" else 1e-6))


def decode_head(raw, spec, scale):
    """Constrain native predictions without inspecting labels."""
    kind = spec["kind"]
    if kind in ("categorical", "hazard"):
        return F.softmax(raw, dim=-1)
    if kind == "binary":
        return torch.sigmoid(raw)
    if kind == "variance":
        return (F.softplus(raw) + 1e-5) * scale
    if kind == "positive":
        return F.softplus(raw) * scale
    if kind == "bounded":
        return torch.tanh(raw)  # legacy recovery is defined on [-1,1].
    if kind in ("quantile", "positive_quantile"):
        # Use a single common physical scale so ordering survives decoding.
        common_scale = scale[..., :1]
        if kind == "positive_quantile":
            return torch.cumsum(F.softplus(raw), dim=-1) * common_scale
        levels = list(spec["levels"])
        center = min(range(len(levels)), key=lambda i: abs(levels[i] - .5))
        median = raw[..., center:center + 1]
        left = -torch.flip(torch.cumsum(torch.flip(F.softplus(raw[..., :center]), [-1]), -1), [-1])
        right = torch.cumsum(F.softplus(raw[..., center + 1:]), -1)
        return torch.cat((median + left, median, median + right), -1) * common_scale
    if kind == "var_es":
        # FZ0 uses negative lower-tail VaR and ES. This constraint is reported
        # explicitly, rather than silently evaluating FZ0 outside its domain.
        quantile = -(F.softplus(raw[..., 0]) + 1e-5) * scale
        expected_shortfall = quantile - (F.softplus(raw[..., 1]) + 1e-5) * scale
        return torch.stack((quantile, expected_shortfall), -1)
    return raw * scale


def head_loss(raw, prediction, target, valid, spec, scale, options=None):
    """Native per-cell losses and masks; placeholders are cleaned first."""
    options = options or {}
    kind = spec["kind"]
    if target.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError(f"{spec['name']} target/mask shapes differ or mask is not boolean")
    if bool((valid & ~torch.isfinite(target)).any()):
        raise ValueError(f"{spec['name']} has nonfinite observed targets")
    clean = torch.where(valid, target, torch.zeros_like(target))
    if kind in ("categorical", "hazard"):
        if bool((valid & ((clean < 0) | (clean >= raw.shape[-1]) | (clean != clean.long()))).any()):
            raise ValueError("categorical targets must be in-range integer class ids")
        value = F.cross_entropy(raw.reshape(-1, raw.shape[-1]), clean.long().reshape(-1), reduction="none").reshape(clean.shape)
    elif kind == "binary":
        if bool((valid & ((clean < 0) | (clean > 1))).any()):
            raise ValueError("binary targets must be between zero and one")
        value = F.binary_cross_entropy_with_logits(raw, clean.to(raw.dtype), reduction="none")
        objective = options.get("classification_loss", "bce")
        if objective in ("weighted_bce", "weighted"):
            prior = torch.as_tensor(options.get("positive_prior", .5), dtype=raw.dtype, device=raw.device).clamp(1e-4, 1 - 1e-4)
            weight = torch.where(clean > .5, .5 / prior, .5 / (1 - prior))
            value = value * weight
        elif objective == "focal":
            pt = torch.where(clean > .5, prediction, 1 - prediction)
            value = value * (1 - pt).pow(float(options.get("focal_gamma", 2.)))
        elif objective != "bce":
            raise ValueError(f"unknown classification loss {objective}")
    elif kind == "variance":
        if bool((valid & (clean < 0)).any()):
            raise ValueError("realized variance cannot be negative")
        y, p = clean.clamp_min(1e-12), prediction.clamp_min(1e-12)
        if options.get("variance_loss", "qlike") == "qlike":
            ratio = y / p
            value = ratio - ratio.log() - 1
        elif options["variance_loss"] in ("log_mse", "logvar_mse"):
            value = (p.log() - y.log()).square()
        else:
            raise ValueError("unknown variance objective")
    elif kind in ("quantile", "positive_quantile"):
        levels = torch.as_tensor(spec["levels"], device=raw.device, dtype=raw.dtype)
        error = (clean - prediction) / scale
        value = torch.maximum(levels * error, (levels - 1) * error)
    elif kind == "var_es":
        alpha = torch.as_tensor(spec["levels"], device=raw.device, dtype=raw.dtype)
        q, e = prediction[..., 0] / scale, prediction[..., 1] / scale
        y = clean / scale
        value = -(y <= q).to(raw.dtype) * (q - y) / (alpha * e) + q / e + torch.log(-e) - 1
    else:
        error = (prediction - clean) / scale
        value = F.huber_loss(error, torch.zeros_like(error), reduction="none") if options.get("regression_loss") == "huber" else error.square()
    if kind == "hazard":
        # A right-censored competing-risk log likelihood sums observed hazard
        # terms along time; average across at-risk trajectories, not step count.
        axis = value.ndim - len(spec["shape"])
        numerator = torch.where(valid, value, torch.zeros_like(value)).sum(dim=axis)
        observed = valid.any(dim=axis)
        return numerator / int(spec["shape"][0]), observed
    return value, valid


class TailModel(nn.Module):
    def __init__(self, input_dim, cfg, specs=None, scales=None, variant="transformer", objectives="joint"):
        super().__init__()
        self.cfg = dict(cfg)
        mc = cfg.get("model", cfg)
        self.input_dim = int(input_dim)
        self.context = int(mc.get("context", 128))
        self.d_model = int(mc.get("d_model", 96))
        self.variant = variant
        self.specs = active_specs(build_specs(cfg) if specs is None else specs, objectives)
        self.objectives = objectives
        self.options = dict(cfg.get("loss", {}))
        self.scales = {} if scales is None else dict(scales)
        heads, layers = int(mc.get("n_heads", 4)), int(mc.get("n_layers", 2))
        dropout, hidden = float(mc.get("dropout", .1)), int(mc.get("encoder_hidden", self.d_model))
        if self.input_dim < 1 or self.context < 1 or self.d_model % heads:
            raise ValueError("invalid causal model dimensions")
        if variant == "wm":
            self.wm = TransformerWorldModel(obs_dim=self.input_dim, act_dim=1,
                n_categoricals=int(mc.get("n_categoricals", 16)), n_classes=int(mc.get("n_classes", 16)),
                d_model=self.d_model, n_heads=heads, n_layers=layers,
                d_ff=int(mc.get("d_ff", 3 * self.d_model)), dropout=dropout,
                max_seq_len=self.context, encoder_hidden=hidden, encoder_layers=2)
            for block in self.wm.blocks:
                block.__class__ = _AttentionBlock
            self.wm.decoder = ObsDecoder(self.wm.z_dim, self.d_model, 5, hidden)
            self.representation_dim = self.wm.z_dim + self.d_model
            self.student_head = nn.Linear(self.representation_dim, 3)
            self.register_buffer("one_return_scale", torch.tensor(float(self.scales.get("next_return", .005))))
        elif variant in ("transformer", "lstm"):
            self.encoder = nn.Sequential(nn.Linear(self.input_dim, hidden), nn.ELU(), nn.Linear(hidden, self.d_model))
            self.drop = nn.Dropout(dropout)
            if variant == "lstm":
                recurrent_hidden = int(mc.get("lstm_hidden", self.d_model))
                self.recurrent = nn.LSTM(self.d_model, recurrent_hidden,
                    num_layers=int(mc.get("lstm_layers", layers)), batch_first=True,
                    dropout=dropout if int(mc.get("lstm_layers", layers)) > 1 else 0.)
                self.representation_dim = recurrent_hidden
            else:
                self.position = nn.Parameter(torch.empty(1, self.context, self.d_model))
                nn.init.normal_(self.position, std=.02)
                self.blocks = nn.ModuleList([_AttentionBlock(self.d_model, heads, int(mc.get("d_ff", 3 * self.d_model)), dropout) for _ in range(layers)])
                self.representation_dim = self.d_model
            self.final_norm = nn.LayerNorm(self.representation_dim)
        else:
            raise ValueError("variant must be transformer, lstm, or wm")
        self.shared_head = nn.Sequential(nn.Linear(self.representation_dim, self.d_model), nn.ELU())
        self.heads = nn.ModuleDict()
        for i, spec in enumerate(self.specs):
            self.heads[spec["name"]] = nn.Linear(self.d_model, math.prod(output_shape(spec)))
            self.register_buffer(f"_scale_{i}", _scale_tensor(spec, self.scales))

    def _represent(self, x):
        if x.ndim != 3 or x.shape[-1] != self.input_dim or not 0 < x.shape[1] <= self.context:
            raise ValueError("x must be [batch,1..context,input_dim]")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("model inputs must be finite")
        if self.variant == "wm":
            z, posterior = self.wm.encode(x)
            dynamics = self.wm(z, x.new_ones((*x.shape[:2], 1)))
            return torch.cat((z, dynamics["h"]), -1), dict(z=z, posterior=posterior, **dynamics)
        hidden = self.encoder(x)
        if self.variant == "lstm":
            hidden, _ = self.recurrent(self.drop(hidden))
        else:
            hidden = self.drop(hidden + self.position[:, :x.shape[1]])
            mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), 1)
            for block in self.blocks:
                hidden = block(hidden, attn_mask=mask)
        return self.final_norm(hidden), {}

    def encode(self, x):
        return self._represent(x)[0]

    @torch.no_grad()
    def latent_last(self, x):
        return self.encode(x)[:, -1]

    def _outputs(self, representation):
        hidden = self.shared_head(representation)
        raw, predictions = {}, {}
        for i, spec in enumerate(self.specs):
            name = spec["name"]
            raw[name] = self.heads[name](hidden).reshape(*hidden.shape[:-1], *output_shape(spec))
            predictions[name] = decode_head(raw[name], spec, getattr(self, f"_scale_{i}"))
        return predictions, raw

    def forward(self, x):
        return self._outputs(self.encode(x))[0]

    @torch.no_grad()
    def predict_last(self, x):
        return self._outputs(self.encode(x)[:, -1])[0]

    def return_distribution(self, x):
        if self.variant != "wm":
            raise ValueError("stochastic return dynamics are a WM-only auxiliary")
        return self._student_parameters(self.encode(x))

    def _student_parameters(self, representation):
        raw = self.student_head(representation)
        return torch.stack((raw[..., 0] * self.one_return_scale,
                            (F.softplus(raw[..., 1]) + 1e-4) * self.one_return_scale,
                            F.softplus(raw[..., 2]) + 2.01), -1)

    def loss(self, x, targets, masks, base_target=None, next_return=None, next_return_mask=None):
        representation, state = self._represent(x)
        first = self.specs[0]
        shape_rank = len(first["shape"])
        last_only = targets[first["name"]].ndim == 1 + shape_rank
        supervised_rep = representation[:, -1] if last_only else representation
        predictions, raw = self._outputs(supervised_rep)
        task_losses, parts = defaultdict(list), {}
        for i, spec in enumerate(self.specs):
            name = spec["name"]
            expected = (*supervised_rep.shape[:-1], *spec["shape"])
            if targets[name].shape != expected or masks[name].shape != expected:
                raise ValueError(f"{name} targets/masks must have shape {expected}")
            opts = {**self.options, **self.options.get("heads", {}).get(name, {})}
            value, valid = head_loss(raw[name], predictions[name], targets[name], masks[name], spec, getattr(self, f"_scale_{i}"), opts)
            observed = valid.sum()
            if bool(observed > 0):
                mean = torch.where(valid, value, torch.zeros_like(value)).sum() / observed
                task_losses[spec["task"]].append(mean)
                parts[name] = mean.detach()
        if not task_losses:
            raise ValueError("no observed supervision for any selected task")
        means = {task: torch.stack(values).mean() for task, values in task_losses.items()}
        supervised = torch.stack(list(means.values())).mean()
        parts.update({"task_" + task: value.detach() for task, value in means.items()})
        parts["supervised"] = supervised.detach()
        total = supervised
        if self.variant == "wm":
            if base_target is None:
                # Reconstruction only uses observed input coordinates, never a
                # future target. Coordinator supplies causal OHLCV if present.
                base_target = x[..., :5]
            if base_target.shape != (*x.shape[:2], 5) or not bool(torch.isfinite(base_target).all()):
                raise ValueError("WM reconstruction needs finite causal [B,T,5] inputs")
            reconstruction = F.mse_loss(self.wm.decoder(state["z"], state["h"]), symlog(base_target))
            if x.shape[1] > 1:
                posterior, prior = state["posterior"][:, 1:], state["prior_logits"][:, :-1]
                dyn = td.kl_divergence(self.wm.encoder.get_dist(posterior.detach()), self.wm.encoder.get_dist(prior)).clamp_min(1.).mean()
                rep = td.kl_divergence(self.wm.encoder.get_dist(posterior), self.wm.encoder.get_dist(prior.detach())).clamp_min(1.).mean()
            else:
                dyn = rep = representation.sum() * 0
            reward, student = representation.sum() * 0, representation.sum() * 0
            if next_return is not None:
                if next_return.shape != x.shape[:2]:
                    raise ValueError("next_return must be [B,T]")
                valid = torch.isfinite(next_return) if next_return_mask is None else next_return_mask & torch.isfinite(next_return)
                if bool(valid.any()):
                    reward = twohot_loss(state["reward_logits"][valid], next_return[valid], self.wm.bins)
                    params = self._student_parameters(representation)[valid]
                    # Standardized NLL removes irrelevant log(scale) constants.
                    dist = td.StudentT(params[:, 2], params[:, 0] / self.one_return_scale, params[:, 1] / self.one_return_scale)
                    nll = -dist.log_prob(next_return[valid] / self.one_return_scale)
                    tail_weight = float(self.options.get("student_tail_weight", 0.))
                    weights = 1 + tail_weight * (next_return[valid].abs() / self.one_return_scale >= 2).to(nll.dtype)
                    student = (nll * weights).mean()
            auxiliary = .3 * reconstruction + .2 * dyn + .05 * rep + .3 * reward + .1 * student
            total = total + auxiliary
            parts.update(reconstruction=reconstruction.detach(), kl_dyn=dyn.detach(), kl_rep=rep.detach(),
                         next_return=reward.detach(), student_nll=student.detach(), auxiliary=auxiliary.detach())
        return total, parts

    def supervised_gradient_coverage(self):
        names, observed_names, gradients, edges = [], [], [], [0]
        for spec in self.specs:
            name = spec["name"]
            names.append(name)
            gradient = self.heads[name].weight.grad
            if gradient is not None:
                observed_names.append(name)
                gradients.append(gradient)
                edges.append(edges[-1] + gradient.shape[0])
        observed = {}
        if gradients:
            # All native heads have the same shared-feature width. Fusing
            # their rows preserves the finite/every-row-nonzero predicate,
            # including noncontiguous gradients, without per-head GPU kernels.
            gradient = torch.cat(gradients, dim=0)
            good = torch.isfinite(gradient).all(1) & (gradient.abs().sum(1) > 0)
            invalid = (~good).to(torch.int32).cumsum(0, dtype=torch.int32)
            prefix = torch.cat((invalid.new_zeros(1), invalid))
            edge = torch.tensor(edges, dtype=torch.int64, device=gradient.device)
            flags = (prefix[edge[1:]] - prefix[edge[:-1]]) == 0
            observed = dict(zip(observed_names, flags.detach().cpu().tolist()))
        return {name: observed.get(name, False) for name in names}

    def metadata(self):
        return dict(variant=self.variant, input_dim=self.input_dim, context=self.context,
            representation_dim=self.representation_dim, specs=self.specs, objectives=self.objectives,
            parameters=sum(p.numel() for p in self.parameters()), causal=True,
            persistent_recurrent_state=False, loss_reduction="head mean; task mean; available-task equal mean",
            var_es_domain="negative lower-tail quantile and ES <= quantile; proper FZ0 score",
            hazard="competing-risk discrete survival likelihood; right-censored at last observed step",
            wm_auxiliary_evaluations_per_batch=int(self.variant == "wm"),
            student_tail_reweighting_changes_density_objective=bool(self.options.get("student_tail_weight", 0.)))

    @torch.no_grad()
    def sample_rollout(self, x, steps, n_samples=32):
        """Sample genuine learned categorical transitions and Student-t returns.

        There is no market-input refresh after initialization, no hindsight
        path selection, and no price clipping. Consumers must report any
        numerical/path validity failures. Eval mode is restored on return.
        """
        if self.variant != "wm" or steps < 1 or n_samples < 1:
            raise ValueError("positive rollout sizes and a WM are required")
        training = self.training
        self.eval()
        try:
            z, _ = self.wm.encode(x)
            batch = len(x)
            z = z.repeat_interleave(n_samples, dim=0)
            paths = []
            for _ in range(int(steps)):
                dynamics = self.wm(z[:, -self.context:], z.new_ones((len(z), min(z.shape[1], self.context), 1)))
                state = torch.cat((z[:, -1], dynamics["h"][:, -1]), -1)
                params = self._student_parameters(state)
                # CPU Student-t sampling avoids unsupported MPS gamma kernels.
                p = params.cpu()
                sampled = td.StudentT(p[:, 2], p[:, 0], p[:, 1]).sample().to(x.device)
                paths.append(sampled.reshape(batch, n_samples))
                prior = OneHotDist(logits=dynamics["prior_logits"][:, -1], unimix_ratio=self.wm.unimix_ratio)
                z_next = prior.sample().flatten(-2)
                z = torch.cat((z[:, -self.context + 1:], z_next[:, None]), 1) if self.context > 1 else z_next[:, None]
            return torch.stack(paths, -1)
        finally:
            self.train(training)
