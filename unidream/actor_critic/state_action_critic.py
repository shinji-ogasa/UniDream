from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateQNet(nn.Module):
    """Small state-action critic for fixed position candidates."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ELU()]
        for _ in range(max(int(n_layers) - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError("state must be (B, D)")
        if action_features.ndim == 2:
            x = torch.cat([state, action_features], dim=-1)
            return self.net(x).squeeze(-1)
        if action_features.ndim != 3:
            raise ValueError("action_features must be (B, A, F) or (B, F)")
        bsz, n_actions, feat_dim = action_features.shape
        state_exp = state.unsqueeze(1).expand(-1, n_actions, -1)
        x = torch.cat([state_exp, action_features], dim=-1).reshape(bsz * n_actions, state.shape[-1] + feat_dim)
        return self.net(x).reshape(bsz, n_actions)


@dataclass
class CandidateQTrainConfig:
    hidden_dim: int = 256
    n_layers: int = 2
    steps: int = 800
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    cql_lite_coef: float = 0.0
    cql_temperature: float = 1.0
    rank_ce_coef: float = 0.0
    rank_tau: float = 0.25
    rank_target_mode: str = "softmax"
    rank_margin: float = 0.0
    anchor_mse_coef: float = 0.0
    ensemble_size: int = 1
    seed: int = 7
    log_interval: int = 200


def build_state_matrix(
    *,
    z: np.ndarray,
    h: np.ndarray,
    inventory: np.ndarray,
    regime: np.ndarray | None = None,
    advantage: np.ndarray | None = None,
) -> np.ndarray:
    parts = [np.asarray(z, dtype=np.float32), np.asarray(h, dtype=np.float32), np.asarray(inventory, dtype=np.float32)]
    if regime is not None:
        parts.append(np.asarray(regime, dtype=np.float32))
    if advantage is not None:
        adv = np.asarray(advantage, dtype=np.float32)
        if adv.ndim == 1:
            adv = adv[:, None]
        parts.append(adv)
    n = min(len(p) for p in parts)
    return np.concatenate([p[:n] for p in parts], axis=1).astype(np.float32)


def build_action_feature_matrix(
    *,
    candidate_actions: np.ndarray,
    current_positions: np.ndarray,
    benchmark_position: float,
) -> np.ndarray:
    actions = np.asarray(candidate_actions, dtype=np.float32)
    current = np.asarray(current_positions, dtype=np.float32)
    if actions.ndim == 1:
        action_matrix = np.broadcast_to(actions[None, :], (len(current), len(actions))).astype(np.float32)
    elif actions.ndim == 2:
        action_matrix = actions.astype(np.float32)
        if len(action_matrix) != len(current):
            n = min(len(action_matrix), len(current))
            action_matrix = action_matrix[:n]
            current = current[:n]
    else:
        raise ValueError("candidate_actions must be a 1D or 2D array")
    overlay = action_matrix - np.float32(benchmark_position)
    delta = action_matrix - current[:, None]
    active_eps = np.float32(0.05)
    short_flag = (overlay < -active_eps).astype(np.float32)
    flat_flag = (np.abs(overlay) <= active_eps).astype(np.float32)
    long_flag = (overlay > active_eps).astype(np.float32)
    features = np.stack(
        [
            action_matrix,
            overlay,
            delta,
            np.abs(delta),
            short_flag,
            flat_flag,
            long_flag,
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def nearest_candidate_indices(positions: np.ndarray, candidate_actions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float32)
    actions = np.asarray(candidate_actions, dtype=np.float32)
    if actions.ndim == 1:
        action_matrix = actions[None, :]
    elif actions.ndim == 2:
        action_matrix = actions
        pos = pos[: len(action_matrix)]
    else:
        raise ValueError("candidate_actions must be a 1D or 2D array")
    return np.argmin(np.abs(pos[:, None] - action_matrix), axis=1).astype(np.int64)


def _rankdata_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(a * b) / denom)


def mean_row_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rows = []
    for p, t in zip(pred, target):
        mask = np.isfinite(p) & np.isfinite(t)
        if mask.sum() < 3:
            continue
        rows.append(_corr(_rankdata_1d(p[mask]), _rankdata_1d(t[mask])))
    if not rows:
        return float("nan")
    return float(np.nanmean(rows))


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _normalize_targets(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    vals = np.asarray(values, dtype=np.float32)
    valid = vals[np.asarray(mask, dtype=bool)]
    mean = float(valid.mean()) if valid.size else 0.0
    std = float(valid.std()) if valid.size else 1.0
    std = max(std, 1e-6)
    return ((vals - mean) / std).astype(np.float32), mean, std


def train_candidate_q_ensemble(
    *,
    train_state: np.ndarray,
    train_action_features: np.ndarray,
    train_values: np.ndarray,
    train_anchor_idx: np.ndarray,
    cfg: CandidateQTrainConfig,
    device: str = "cpu",
) -> tuple[list[CandidateQNet], dict]:
    dev = torch.device(device)
    state = np.asarray(train_state, dtype=np.float32)
    action_features = np.asarray(train_action_features, dtype=np.float32)
    values = np.asarray(train_values, dtype=np.float32)
    mask_np = np.isfinite(values)
    row_mask = mask_np.sum(axis=1) >= 2
    state = state[row_mask]
    action_features = action_features[row_mask]
    values = values[row_mask]
    mask_np = mask_np[row_mask]
    anchor_idx = np.asarray(train_anchor_idx, dtype=np.int64)[row_mask]
    values_norm, target_mean, target_std = _normalize_targets(values, mask_np)

    state_t = torch.tensor(state, dtype=torch.float32, device=dev)
    action_t = torch.tensor(action_features, dtype=torch.float32, device=dev)
    values_t = torch.tensor(np.nan_to_num(values_norm, nan=0.0), dtype=torch.float32, device=dev)
    mask_t = torch.tensor(mask_np, dtype=torch.float32, device=dev)
    anchor_t = torch.tensor(anchor_idx, dtype=torch.long, device=dev)

    models: list[CandidateQNet] = []
    histories = []
    n = len(state)
    if n == 0:
        raise ValueError("No valid rows for candidate Q training")
    for member in range(max(int(cfg.ensemble_size), 1)):
        torch.manual_seed(int(cfg.seed) + member)
        model = CandidateQNet(
            state_dim=state.shape[1],
            action_dim=action_features.shape[-1],
            hidden_dim=int(cfg.hidden_dim),
            n_layers=int(cfg.n_layers),
        ).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
        last = {}
        for step in range(max(int(cfg.steps), 1)):
            idx = torch.randint(0, n, (min(int(cfg.batch_size), n),), device=dev)
            pred = model(state_t[idx], action_t[idx])
            target = values_t[idx]
            mask = mask_t[idx]
            mse = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
            loss = mse
            rank_ce = torch.zeros((), dtype=torch.float32, device=dev)
            if cfg.rank_ce_coef > 0.0:
                tau = max(float(cfg.rank_tau), 1e-6)
                pred_masked = pred.masked_fill(mask <= 0.0, -1e9)
                target_masked = target.masked_fill(mask <= 0.0, -1e9)
                if str(cfg.rank_target_mode).lower() == "margin_best":
                    best_target, best_idx = target_masked.max(dim=1)
                    anchor_label = anchor_t[idx].clamp(min=0, max=target.shape[1] - 1)
                    anchor_target = target.gather(1, anchor_label.unsqueeze(1)).squeeze(1)
                    use_best = (best_target - anchor_target) > float(cfg.rank_margin)
                    labels = torch.where(use_best, best_idx, anchor_label)
                    rank_ce = F.cross_entropy(pred_masked / tau, labels)
                else:
                    target_probs = torch.softmax(target_masked / tau, dim=1)
                    pred_log_probs = torch.log_softmax(pred_masked / tau, dim=1)
                    rank_ce = -(target_probs * pred_log_probs).sum(dim=1).mean()
                loss = loss + float(cfg.rank_ce_coef) * rank_ce
            cql = torch.zeros((), dtype=torch.float32, device=dev)
            if cfg.cql_lite_coef > 0.0:
                temp = max(float(cfg.cql_temperature), 1e-6)
                logsum = torch.logsumexp(pred / temp, dim=1) * temp
                anchor_q = pred.gather(1, anchor_t[idx].unsqueeze(1)).squeeze(1)
                cql = (logsum - anchor_q).mean()
                loss = loss + float(cfg.cql_lite_coef) * cql
            anchor_mse = torch.zeros((), dtype=torch.float32, device=dev)
            if cfg.anchor_mse_coef > 0.0:
                anchor_q = pred.gather(1, anchor_t[idx].unsqueeze(1)).squeeze(1)
                anchor_target = target.gather(1, anchor_t[idx].unsqueeze(1)).squeeze(1)
                anchor_mse = F.mse_loss(anchor_q, anchor_target)
                loss = loss + float(cfg.anchor_mse_coef) * anchor_mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()
            last = {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "mse": float(mse.detach().cpu()),
                "rank_ce": float(rank_ce.detach().cpu()),
                "cql": float(cql.detach().cpu()),
                "anchor_mse": float(anchor_mse.detach().cpu()),
            }
        models.append(model.eval())
        histories.append(last)
    meta = {
        "target_mean": target_mean,
        "target_std": target_std,
        "train_rows": int(n),
        "train_valid_fraction": float(mask_np.mean()),
        "members": histories,
    }
    return models, meta


@torch.no_grad()
def predict_candidate_q(
    models: Iterable[CandidateQNet],
    *,
    state: np.ndarray,
    action_features: np.ndarray,
    target_mean: float,
    target_std: float,
    device: str = "cpu",
    reduce: str = "mean",
    batch_size: int = 4096,
) -> np.ndarray:
    dev = torch.device(device)
    state_np = np.asarray(state, dtype=np.float32)
    action_np = np.asarray(action_features, dtype=np.float32)
    preds_by_model = []
    for model in models:
        model = model.to(dev).eval()
        chunks = []
        for start in range(0, len(state_np), batch_size):
            end = min(start + batch_size, len(state_np))
            s = torch.tensor(state_np[start:end], dtype=torch.float32, device=dev)
            a = torch.tensor(action_np[start:end], dtype=torch.float32, device=dev)
            chunks.append(model(s, a).detach().cpu().numpy())
        preds_by_model.append(np.concatenate(chunks, axis=0))
    stack = np.stack(preds_by_model, axis=0)
    if reduce == "min":
        pred = np.min(stack, axis=0)
    elif reduce == "median":
        pred = np.median(stack, axis=0)
    else:
        pred = np.mean(stack, axis=0)
    return (pred * float(target_std) + float(target_mean)).astype(np.float32)


def evaluate_candidate_q(
    *,
    q_pred: np.ndarray,
    values: np.ndarray,
    candidate_actions: np.ndarray,
    anchor_idx: np.ndarray,
    current_positions: np.ndarray,
    benchmark_position: float,
) -> dict:
    q = np.asarray(q_pred, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    actions = np.asarray(candidate_actions, dtype=np.float64)
    anchor = np.asarray(anchor_idx, dtype=np.int64)
    current = np.asarray(current_positions, dtype=np.float64)
    n = min(len(q), len(vals), len(anchor), len(current))
    q = q[:n]
    vals = vals[:n]
    anchor = anchor[:n]
    current = current[:n]
    if actions.ndim == 1:
        action_matrix = np.broadcast_to(actions[None, :], q.shape).astype(np.float64)
    elif actions.ndim == 2:
        action_matrix = actions[:n].astype(np.float64)
    else:
        raise ValueError("candidate_actions must be a 1D or 2D array")
    valid_row = np.isfinite(vals).sum(axis=1) >= 2
    q_work = np.where(np.isfinite(vals), q, -np.inf)
    selected = np.argmax(q_work, axis=1)
    best = np.nanargmax(np.where(np.isfinite(vals), vals, -np.inf), axis=1)
    row = np.arange(n)
    selected_value = vals[row, selected]
    anchor_value = vals[row, anchor]
    best_value = vals[row, best]
    selected_adv_anchor = selected_value - anchor_value
    best_adv_anchor = best_value - anchor_value
    pred_margin_anchor = q[row, selected] - q[row, anchor]
    active_eps = 0.05
    selected_actions = action_matrix[row, selected]
    selected_overlay = selected_actions - float(benchmark_position)
    selected_delta = np.abs(selected_actions - current)
    rates = {
        "short": float((selected_overlay < -active_eps).mean()) if n else 0.0,
        "flat": float((np.abs(selected_overlay) <= active_eps).mean()) if n else 0.0,
        "long": float((selected_overlay > active_eps).mean()) if n else 0.0,
    }
    action_rows = []
    best_action_rows = []
    n_actions = q.shape[1]
    for idx in range(n_actions):
        mask = selected == idx
        selected_action_values = action_matrix[mask, idx] if mask.any() else np.asarray([], dtype=np.float64)
        action_rows.append(
            {
                "action": float(idx),
                "mean_action": _safe_mean(selected_action_values) if mask.any() else float("nan"),
                "selected_rate": float(mask.mean()) if n else 0.0,
                "mean_realized_value": _safe_mean(vals[mask, idx]) if mask.any() else float("nan"),
                "mean_q": _safe_mean(q[mask, idx]) if mask.any() else float("nan"),
            }
        )
        best_mask = best == idx
        best_action_values = action_matrix[best_mask, idx] if best_mask.any() else np.asarray([], dtype=np.float64)
        best_action_rows.append(
            {
                "action": float(idx),
                "mean_action": _safe_mean(best_action_values) if best_mask.any() else float("nan"),
                "best_rate": float(best_mask.mean()) if n else 0.0,
                "mean_best_adv_vs_anchor": _safe_mean(best_adv_anchor[best_mask]) if best_mask.any() else float("nan"),
            }
        )
    top_mask = np.zeros(n, dtype=bool)
    if n:
        finite_margin = pred_margin_anchor[np.isfinite(pred_margin_anchor)]
        if finite_margin.size:
            threshold = np.quantile(finite_margin, 0.90)
            top_mask = pred_margin_anchor >= threshold
    return {
        "n": int(n),
        "valid_rows": int(valid_row.sum()),
        "rmse": float(np.sqrt(np.nanmean((q[valid_row] - vals[valid_row]) ** 2))) if valid_row.any() else float("nan"),
        "flat_pearson": _corr(q[valid_row].reshape(-1), vals[valid_row].reshape(-1)) if valid_row.any() else float("nan"),
        "row_spearman": mean_row_spearman(q[valid_row], vals[valid_row]) if valid_row.any() else float("nan"),
        "top1_best_match": float((selected[valid_row] == best[valid_row]).mean()) if valid_row.any() else 0.0,
        "selected_realized_adv_vs_anchor": _safe_mean(selected_adv_anchor[valid_row]),
        "best_possible_adv_vs_anchor": _safe_mean(best_adv_anchor[valid_row]),
        "top_decile_selected_realized_adv_vs_anchor": _safe_mean(selected_adv_anchor[top_mask & valid_row]),
        "selected_mean_abs_delta": _safe_mean(selected_delta[valid_row]),
        "selected_short_rate": rates["short"],
        "selected_flat_rate": rates["flat"],
        "selected_long_rate": rates["long"],
        "selected_action_rows": action_rows,
        "best_action_rows": best_action_rows,
    }
