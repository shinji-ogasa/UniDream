"""Isolated causal market-WM, BC policy and real-history PPO components.

This file never reads a dataset or opens a Test split. All four actor arms
have identical parameter geometry. Inputs F and Z are hard-masked according
to the registered arm, while the ordinary history encoder is always used.
PPO acts in Gaussian latent-action space; tanh, position limits and execution
lag are deterministic environment transformations. No hindsight target is
ever an RL observation, and PPO does not optimize a BC teacher loss.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

from .wm_full_grid_models import WMFullGrid


ARMS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}
INVENTORY_NAMES = ("requested", "executed", "previous_delta", "log_wealth", "drawdown", "max_drawdown")


def _accounting(cfg):
    return cfg.get("accounting", cfg)


def _rl(cfg):
    supplied = cfg.get("rl_training", cfg.get("rl", cfg.get("ppo", cfg)))
    result = dict(supplied)
    aliases = {"ppo_epochs": "epochs", "clip_ratio": "clip", "value_coefficient": "critic_coef",
        "bc_prior_coefficient": "bc_prior_coef", "entropy_coefficient": "entropy_coef"}
    for canonical, internal in aliases.items():
        if canonical in supplied:
            if internal in supplied and supplied[internal] != supplied[canonical]:
                raise ValueError(f"conflicting RL configuration values: {canonical}/{internal}")
            result[internal] = supplied[canonical]
    return result


class PolicyAblationWM(WMFullGrid):
    """Fresh X-only market WM with actual categorical z and causal dynamics."""
    def __init__(self, input_dim, model_cfg, scales):
        self._policy_frozen = False
        super().__init__(input_dim, model_cfg, scales, "regression")

    @property
    def policy_latent_dim(self):
        return self.wm.z_dim + self.wm.d_model

    def train(self, mode=True):
        if mode and self._policy_frozen:
            raise RuntimeError("policy WM is frozen and must remain in evaluation mode")
        return super().train(mode)

    def freeze_for_policy(self):
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        self._policy_frozen = True
        super().train(False)
        return self

    @torch.no_grad()
    def encode_policy_state(self, x):
        if not self._policy_frozen or self.training:
            raise RuntimeError("freeze the selected Train-only WM before encoding policy state")
        return self.latent_last(x).detach()


class PolicyActorCritic(nn.Module):
    """Same-sized X-history/F/Z/inventory actor-critic for arms A through D."""
    def __init__(self, x_dim, forecast_dim, latent_dim, cfg, arm):
        super().__init__()
        if arm not in ARMS or min(x_dim, forecast_dim, latent_dim) < 1:
            raise ValueError("known arm and positive X/F/Z dimensions required")
        actor = cfg.get("actor", cfg)
        accounting = _accounting(cfg)
        self.arm = arm
        self.use_forecasts, self.use_latents = ARMS[arm]
        self.x_dim, self.forecast_dim, self.latent_dim = int(x_dim), int(forecast_dim), int(latent_dim)
        self.context = int(actor.get("context", 256))
        self.hidden_size = int(actor.get("lstm_hidden_size", 64))
        self.layers = int(actor.get("lstm_layers", actor.get("layers", 1)))
        self.projection_size = int(actor.get("projection_dim", actor.get("projection_size", 64)))
        self.head_hidden = int(actor.get("head_hidden", actor.get("hidden_size", 128)))
        if min(self.context, self.hidden_size, self.layers, self.projection_size, self.head_hidden) < 1:
            raise ValueError("positive actor dimensions required")
        if float(actor.get("dropout", 0.)) != 0.:
            raise ValueError("actor dropout must be zero for the registered PPO likelihood")
        self.position_min = float(accounting.get("position_min", .50))
        self.position_max = float(accounting.get("position_max", 1.12))
        self.max_position_step = float(accounting.get("max_position_step", .08))
        if not 0 <= self.position_min < self.position_max or self.max_position_step <= 0:
            raise ValueError("invalid position bounds or maximum request change")
        self.x_encoder = nn.Sequential(nn.Linear(self.x_dim, self.hidden_size), nn.ELU())
        self.history_encoder = nn.LSTM(self.hidden_size, self.hidden_size, self.layers,
            batch_first=True, bidirectional=False, dropout=0.)
        self.history_norm = nn.LayerNorm(self.hidden_size)
        self.forecast_encoder = nn.Sequential(nn.Linear(self.forecast_dim, self.projection_size), nn.ELU())
        self.latent_encoder = nn.Sequential(nn.Linear(self.latent_dim, self.projection_size), nn.ELU())
        width = self.hidden_size + 2 * self.projection_size + len(INVENTORY_NAMES)
        self.policy_head = nn.Sequential(nn.Linear(width, self.head_hidden), nn.ELU(), nn.Linear(self.head_hidden, 1))
        self.critic_head = nn.Sequential(nn.Linear(width, self.head_hidden), nn.ELU(), nn.Linear(self.head_hidden, 1))
        self.log_std = nn.Parameter(torch.tensor(float(actor.get("initial_log_std", -1.))))
        self.min_log_std = float(actor.get("min_log_std", -4.))
        self.max_log_std = float(actor.get("max_log_std", 1.))
        with torch.no_grad():
            self.policy_head[-1].weight.mul_(.01)
            self.policy_head[-1].bias.zero_()

    def encode_context(self, x, forecasts, latents):
        if (x.ndim != 3 or x.shape[-1] != self.x_dim or not 0 < x.shape[1] <= self.context
                or forecasts.shape != (len(x), self.forecast_dim) or latents.shape != (len(x), self.latent_dim)):
            raise ValueError("matched causal X windows and current F/Z vectors required")
        hidden, _ = self.history_encoder(self.x_encoder(x))
        history = self.history_norm(hidden[:, -1])
        # Inactive inputs are not read, so even NaN placeholders cannot leak
        # through a multiply-by-zero operation or a learned projection bias.
        f = self.forecast_encoder(forecasts) if self.use_forecasts else history.new_zeros((len(x), self.projection_size))
        z = self.latent_encoder(latents) if self.use_latents else history.new_zeros((len(x), self.projection_size))
        return torch.cat((history, f, z), -1)

    def heads(self, context, inventory):
        if inventory.shape != (len(context), len(INVENTORY_NAMES)):
            raise ValueError("inventory must contain the six registered own-account features")
        features = torch.cat((context, inventory), -1)
        mean = self.policy_head(features).squeeze(-1)
        std = self.log_std.clamp(self.min_log_std, self.max_log_std).exp().expand_as(mean)
        return Normal(mean, std, validate_args=False), self.critic_head(features).squeeze(-1)

    def forward(self, x, forecasts, latents, inventory):
        distribution, value = self.heads(self.encode_context(x, forecasts, latents), inventory)
        request = self.request_from_action(distribution.mean, inventory[:, 0])
        return {"mean": distribution.mean, "std": distribution.stddev, "value": value, "mean_request": request}

    def request_from_action(self, raw_action, previous_request):
        return (previous_request + self.max_position_step * torch.tanh(raw_action)).clamp(self.position_min, self.position_max)

    def act_from_context(self, context, inventory, deterministic=False):
        distribution, value = self.heads(context, inventory)
        raw_action = distribution.mean if deterministic else distribution.sample()
        return {"request": self.request_from_action(raw_action, inventory[:, 0]),
            "raw_action": raw_action, "log_prob": distribution.log_prob(raw_action), "value": value,
            "mean": distribution.mean, "std": distribution.stddev}

    def act(self, x, forecasts, latents, inventory, deterministic=False):
        return self.act_from_context(self.encode_context(x, forecasts, latents), inventory, deterministic)

    def bc_loss(self, x, forecasts, latents, inventory, teacher_request):
        result = self(x, forecasts, latents, inventory)
        if teacher_request.shape != result["mean_request"].shape:
            raise ValueError("teacher request must match the policy batch")
        loss = ((result["mean_request"] - teacher_request) / self.max_position_step).square().mean()
        return loss, {"bc_request_mse": loss.detach()}

    def state_gradient_coverage(self):
        def present(module):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            return bool(gradients and all(torch.isfinite(g).all() for g in gradients)
                        and sum(float(g.detach().abs().sum().cpu()) for g in gradients) > 0)
        return {"history": present(self.history_encoder), "forecast": present(self.forecast_encoder),
            "latent": present(self.latent_encoder), "policy": present(self.policy_head), "critic": present(self.critic_head)}

    def checkpoint_metadata(self):
        return {"arm": self.arm, "context": self.context, "hidden_size": self.hidden_size, "lstm_layers": self.layers,
            "x_dim": self.x_dim, "forecast_dim": self.forecast_dim, "latent_dim": self.latent_dim,
            "use_forecasts": self.use_forecasts, "use_latents": self.use_latents,
            "inventory_names": list(INVENTORY_NAMES), "position_min": self.position_min,
            "position_max": self.position_max, "max_position_step": self.max_position_step,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "same_parameter_geometry_all_arms": True, "bidirectional": False, "persistent_recurrent_state": False,
            "policy_density": "Gaussian latent-action likelihood; deterministic tanh/rate/bounds/execution transformations",
            "rl_algorithm": "real-history on-policy clipped PPO; no imagination or BC teacher loss"}


def freeze_bc_prior(model):
    return copy.deepcopy(model).eval().requires_grad_(False)


def initial_portfolio(batch, device, cfg, dtype=torch.float32):
    accounting = _accounting(cfg)
    initial = float(accounting.get("initial_position", 1.))
    if not float(accounting.get("position_min", .5)) <= initial <= float(accounting.get("position_max", 1.12)):
        raise ValueError("initial inventory must lie within the registered bounds")
    zero = torch.zeros(batch, device=device, dtype=dtype)
    return {"requested": zero + initial, "executed": zero + initial, "previous_delta": zero.clone(),
        "log_wealth": zero.clone(), "peak_log_wealth": zero.clone(), "drawdown": zero.clone(), "max_drawdown": zero.clone()}


def inventory_features(state):
    return torch.stack([state[name] for name in INVENTORY_NAMES], -1)


@torch.no_grad()
def advance_portfolio(state, new_request, bar_log_return, cfg):
    """One exact self-financing transition with one-bar delayed execution."""
    a = _accounting(cfg)
    executed = state["requested"]
    turnover = (executed - state["executed"]).abs()
    cost = float(a.get("cost_rate", .00055)) * turnover
    borrow = (executed - 1.).clamp_min(0) * float(a.get("borrow_apr", .10)) / float(a.get("annual_bars", 35040))
    simple = executed * torch.expm1(bar_log_return) - borrow
    gross = 1. + simple
    if not bool((gross > 0).all()) or not bool((cost < 1).all()):
        raise ValueError("historical interval is insolvent under the registered accounting")
    if (not bool(torch.isfinite(new_request).all()) or not bool(torch.isfinite(bar_log_return).all())
            or bool((new_request < float(a.get("position_min", .5)) - 1e-6).any())
            or bool((new_request > float(a.get("position_max", 1.12)) + 1e-6).any())
            or bool(((new_request - state["requested"]).abs() > float(a.get("max_position_step", .08)) + 1e-6).any())):
        raise ValueError("nonfinite inputs or infeasible historical policy request")
    net_log_return = torch.log1p(-cost) + torch.log1p(simple)
    log_wealth = state["log_wealth"] + net_log_return
    peak = torch.maximum(state["peak_log_wealth"], log_wealth)
    drawdown = -torch.expm1(log_wealth - peak)
    max_drawdown = torch.maximum(state["max_drawdown"], drawdown)
    max_dd_increment = max_drawdown - state["max_drawdown"]
    unscaled_reward = net_log_return - bar_log_return - float(a.get("drawdown_penalty", .25)) * max_dd_increment
    reward_scale = float(_rl(cfg).get("reward_scale", 100.))
    updated = {"requested": new_request.detach(), "executed": executed.detach(),
        "previous_delta": (new_request - state["requested"]).detach(), "log_wealth": log_wealth,
        "peak_log_wealth": peak, "drawdown": drawdown, "max_drawdown": max_drawdown}
    return updated, {"reward": reward_scale * unscaled_reward, "unscaled_reward": unscaled_reward,
        "net_log_return": net_log_return, "excess_log_return": net_log_return - bar_log_return,
        "executed": executed, "turnover": turnover, "borrow": borrow, "cost": cost,
        "drawdown": drawdown, "max_drawdown": max_drawdown, "max_dd_increment": max_dd_increment}


@torch.no_grad()
def compute_gae(rewards, values, terminals, bootstrap_value, gamma=.99, gae_lambda=.95):
    """Time-major [T,B] GAE with genuine next-state bootstrapping."""
    if (rewards.ndim != 2 or values.shape != rewards.shape or terminals.shape != rewards.shape
            or terminals.dtype != torch.bool or bootstrap_value.shape != rewards.shape[1:]):
        raise ValueError("GAE requires matched [T,B] rewards/values/terminals and final [B] values")
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("GAE discount and lambda must lie in[0,1]")
    advantages = torch.zeros_like(rewards)
    following_advantage = torch.zeros_like(bootstrap_value)
    following_value = bootstrap_value
    for step in range(len(rewards) - 1, -1, -1):
        continuation = (~terminals[step]).to(rewards.dtype)
        delta = rewards[step] + gamma * following_value * continuation - values[step]
        following_advantage = delta + gamma * gae_lambda * continuation * following_advantage
        advantages[step] = following_advantage
        following_value = values[step]
    return advantages, advantages + values


@torch.no_grad()
def collect_history_rollout(model, x, forecasts, latents, bar_log_returns, cfg, terminals=None, initial_state=None):
    """Collect on-policy inventory trajectories; all market inputs stay causal.

    X/F/Z are batch-major [B,T+1,...]; the final observation is used only to
    bootstrap a truncated nonterminal rollout. Market encodings are cached
    under no_grad for collection; PPO recomputes them WITH gradients later.
    """
    if x.ndim != 4 or bar_log_returns.shape != (x.shape[0], x.shape[1] - 1):
        raise ValueError("rollout needs [B,T+1,context,X] and [B,T] realized returns")
    batch, length = bar_log_returns.shape
    if forecasts.shape[:2] != (batch, length + 1) or latents.shape[:2] != (batch, length + 1):
        raise ValueError("market features must include the final bootstrap observation")
    if terminals is None:
        terminals = torch.zeros_like(bar_log_returns, dtype=torch.bool)
    if terminals.shape != bar_log_returns.shape or terminals.dtype != torch.bool:
        raise ValueError("terminal flags must be boolean [B,T]")
    model.eval()
    flat_x = x.flatten(0, 1)
    flat_f, flat_z = forecasts.flatten(0, 1), latents.flatten(0, 1)
    chunk = int(_rl(cfg).get("market_batch_size", 64))
    contexts = torch.cat([model.encode_context(flat_x[start:start + chunk], flat_f[start:start + chunk], flat_z[start:start + chunk])
        for start in range(0, len(flat_x), chunk)]).reshape(batch, length + 1, -1)
    state = initial_portfolio(batch, x.device, cfg, x.dtype) if initial_state is None else {k: v.detach().clone() for k, v in initial_state.items()}
    saved = {key: [] for key in ("inventory", "raw_action", "old_log_prob", "old_value", "request", "reward",
        "net_log_return", "excess_log_return", "executed", "turnover", "max_dd_increment")}
    for step in range(length):
        inventory = inventory_features(state)
        action = model.act_from_context(contexts[:, step], inventory)
        saved["inventory"].append(inventory)
        for output, field in (("raw_action", "raw_action"), ("old_log_prob", "log_prob"), ("old_value", "value"), ("request", "request")):
            saved[output].append(action[field])
        state, accounting = advance_portfolio(state, action["request"], bar_log_returns[:, step], cfg)
        for field in ("reward", "net_log_return", "excess_log_return", "executed", "turnover", "max_dd_increment"):
            saved[field].append(accounting[field])
        if bool(terminals[:, step].any()):
            reset = initial_portfolio(batch, x.device, cfg, x.dtype)
            state = {key: torch.where(terminals[:, step], reset[key], value) for key, value in state.items()}
    bootstrap = model.heads(contexts[:, -1], inventory_features(state))[1]
    saved = {key: torch.stack(value, 1) for key, value in saved.items()}
    settings = _rl(cfg)
    advantages, returns = compute_gae(saved["reward"].T, saved["old_value"].T, terminals.T, bootstrap,
        float(settings.get("gamma", .99)), float(settings.get("gae_lambda", .95)))
    flat = {key: value.flatten(0, 1).detach() for key, value in saved.items() if key not in
        ("reward", "net_log_return", "excess_log_return", "executed", "turnover", "max_dd_increment", "request")}
    flat.update(x=x[:, :-1].flatten(0, 1).detach(), forecasts=forecasts[:, :-1].flatten(0, 1).detach(),
        latents=latents[:, :-1].flatten(0, 1).detach(), advantages=advantages.T.flatten().detach(), returns=returns.T.flatten().detach())
    return {"batch": flat, "ledger": saved, "final_state": state, "bootstrap_value": bootstrap.detach()}


def ppo_update(model, prior, optimizer, rollout, cfg):
    """True policy/critic updates from detached, on-policy historical rewards."""
    settings = _rl(cfg)
    batch = rollout["batch"] if "batch" in rollout else rollout
    n = len(batch["raw_action"])
    if n < 2:
        raise ValueError("PPO needs at least two on-policy transitions")
    if prior is None or any(p.requires_grad for p in prior.parameters()):
        raise ValueError("PPO requires the immutable selected BC prior")
    advantages = batch["advantages"].detach()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    epochs, minibatch = int(settings.get("epochs", 2)), int(settings.get("minibatch_size", 64))
    clip = float(settings.get("clip", .2))
    if min(epochs, minibatch) < 1 or not 0 < clip < 1:
        raise ValueError("positive PPO epochs/minibatch and clipping parameter in(0,1) required")
    rows = []
    model.train()
    prior.eval()
    # The immutable prior sees the same stored causal states in both PPO
    # epochs. Evaluate it once; the trainable policy is still recomputed with
    # gradients in EVERY minibatch.
    prior_means, prior_stds = [], []
    with torch.no_grad():
        for start in range(0, n, minibatch):
            values = [batch[key][start:start + minibatch] for key in ("x", "forecasts", "latents", "inventory")]
            reference, _ = prior.heads(prior.encode_context(*values[:3]), values[3])
            prior_means.append(reference.mean)
            prior_stds.append(reference.stddev)
    prior_mean, prior_std = torch.cat(prior_means), torch.cat(prior_stds)
    for _ in range(epochs):
        order = torch.randperm(n, device=advantages.device)
        for start in range(0, n, minibatch):
            ids = order[start:start + minibatch]
            x, forecasts, latents, inventory = (batch[key][ids] for key in ("x", "forecasts", "latents", "inventory"))
            current, value = model.heads(model.encode_context(x, forecasts, latents), inventory)
            reference = Normal(prior_mean[ids], prior_std[ids], validate_args=False)
            log_prob = current.log_prob(batch["raw_action"][ids].detach())
            ratio = torch.exp(log_prob - batch["old_log_prob"][ids].detach())
            weighted = ratio * advantages[ids]
            clipped = ratio.clamp(1 - clip, 1 + clip) * advantages[ids]
            policy_loss = -torch.minimum(weighted, clipped).mean()
            critic_loss = F.mse_loss(value, batch["returns"][ids].detach())
            prior_kl = kl_divergence(current, reference).mean()
            entropy = current.entropy().mean()
            loss = (policy_loss + float(settings.get("critic_coef", .5)) * critic_loss
                + float(settings.get("bc_prior_coef", .01)) * prior_kl - float(settings.get("entropy_coef", .001)) * entropy)
            policy_parameters = [*model.policy_head.parameters(), model.log_std]
            policy_gradients = torch.autograd.grad(policy_loss, policy_parameters, retain_graph=True, allow_unused=True)
            pg_norm = torch.sqrt(sum(g.square().sum() for g in policy_gradients if g is not None))
            before = model.policy_head[-1].weight.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings.get("gradient_clip", 10.)), error_if_nonfinite=True)
            optimizer.step()
            names = ("policy_loss", "critic_loss", "prior_kl", "latent_entropy", "policy_gradient_norm",
                     "total_gradient_norm", "policy_parameter_delta", "clip_fraction")
            diagnostic = torch.stack((policy_loss.detach(), critic_loss.detach(), prior_kl.detach(), entropy.detach(),
                pg_norm.detach(), norm.detach(), (model.policy_head[-1].weight.detach() - before).norm(),
                ((ratio - 1).abs() > clip).float().mean().detach())).cpu().tolist()
            rows.append(dict(zip(names, map(float, diagnostic))))
    result = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    result.update(optimizer_steps=len(rows), transitions=n, reward_mean=float(rollout["ledger"]["reward"].mean().cpu()) if "ledger" in rollout else None,
        advantage_std=float(batch["advantages"].std(unbiased=False).cpu()),
        real_policy_gradient_steps=sum(row["policy_gradient_norm"] > 0 for row in rows),
        changed_policy_steps=sum(row["policy_parameter_delta"] > 0 for row in rows),
        ppo_epochs=epochs, clip_ratio=clip, value_coefficient=float(settings.get("critic_coef", .5)),
        entropy_coefficient=float(settings.get("entropy_coef", .001)), bc_prior_coefficient=float(settings.get("bc_prior_coef", .01)))
    return result
