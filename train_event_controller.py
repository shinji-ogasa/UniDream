from __future__ import annotations

import argparse
import glob
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from train import _benchmark_position_value, _format_m2_scorecard, _m2_scorecard
from unidream.actor_critic.imagination_ac import _action_stats, _fmt_action_stats
from unidream.data.download import fetch_binance_ohlcv
from unidream.data.features import compute_features, get_raw_returns, augment_with_rebound_features
from unidream.data.dataset import get_wfo_splits, WFODataset
from unidream.eval.backtest import Backtest


_CACHE_STALE_DAYS = 7


def _cache_is_fresh(path: str, stale_days: int = _CACHE_STALE_DAYS) -> bool:
    if not os.path.exists(path):
        return False
    import time
    return (time.time() - os.path.getmtime(path)) / 86400 < stale_days


def _resolve_cache_pair(cache_dir: str, cache_tag: str) -> tuple[str, str]:
    features_cache = os.path.join(cache_dir, f"{cache_tag}_features.parquet")
    returns_cache = os.path.join(cache_dir, f"{cache_tag}_returns.parquet")
    if os.path.exists(features_cache) and os.path.exists(returns_cache):
        return features_cache, returns_cache

    feature_candidates = sorted(glob.glob(os.path.join(cache_dir, f"{cache_tag}*_features.parquet")))
    return_candidates = sorted(glob.glob(os.path.join(cache_dir, f"{cache_tag}*_returns.parquet")))
    if feature_candidates and return_candidates:
        return feature_candidates[0], return_candidates[0]
    return features_cache, returns_cache


def _read_optional_parquet(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df.index, pd.DatetimeIndex) and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False)
        df = df.set_index("time")
    return df.sort_index()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_costs(cfg: dict, cost_profile: str | None = None) -> tuple[dict, str]:
    resolved_cfg = dict(cfg)
    profile_name = cost_profile or cfg.get("cost_profile") or "default"
    profiles = cfg.get("cost_profiles")

    if profiles:
        if profile_name == "default":
            profile_name = "base" if "base" in profiles else next(iter(profiles))
        if profile_name not in profiles:
            available = ", ".join(profiles.keys())
            raise KeyError(f"Unknown cost profile '{profile_name}'. Available: {available}")
        resolved_cfg["costs"] = dict(profiles[profile_name])
        resolved_cfg["cost_profile"] = profile_name
    else:
        resolved_cfg["costs"] = dict(cfg.get("costs", {}))
        resolved_cfg["cost_profile"] = profile_name

    return resolved_cfg, resolved_cfg["cost_profile"]


def _future_window_stats(returns: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(returns)
    future_sum = np.zeros(T, dtype=np.float32)
    neg_frac = np.zeros(T, dtype=np.float32)
    pos_frac = np.zeros(T, dtype=np.float32)
    for i in range(T):
        end = min(T, i + 1 + horizon)
        window = returns[i + 1:end]
        if window.size == 0:
            continue
        future_sum[i] = float(window.sum())
        neg_frac[i] = float((window < 0).mean())
        pos_frac[i] = float((window > 0).mean())
    return future_sum, neg_frac, pos_frac


def _event_targets(
    returns: np.ndarray,
    horizon: int,
    downside_quantile: float,
    upside_quantile: float,
    min_neg_frac: float,
    min_pos_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    future_sum, neg_frac, pos_frac = _future_window_stats(returns, horizon)
    valid_len = max(len(returns) - horizon, 0)
    valid = np.zeros(len(returns), dtype=bool)
    valid[:valid_len] = True
    if valid_len <= 0:
        return np.zeros(len(returns), dtype=np.float32), np.zeros(len(returns), dtype=np.float32), valid

    valid_sum = future_sum[:valid_len]
    down_cut = float(np.quantile(valid_sum, downside_quantile))
    up_cut = float(np.quantile(valid_sum, upside_quantile))
    downside = (future_sum <= down_cut) & (neg_frac >= min_neg_frac) & valid
    upside = (future_sum >= up_cut) & (pos_frac >= min_pos_frac) & valid
    overlap = downside & upside
    downside[overlap] = False
    upside[overlap] = False
    return downside.astype(np.float32), upside.astype(np.float32), valid


def _triple_barrier_targets(
    returns: np.ndarray,
    horizon: int,
    vol_lookback: int,
    barrier_mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(returns)
    downside = np.zeros(T, dtype=np.float32)
    upside = np.zeros(T, dtype=np.float32)
    valid = np.zeros(T, dtype=bool)
    if T <= horizon:
        return downside, upside, valid

    for i in range(T - horizon):
        hist_start = max(0, i - vol_lookback + 1)
        hist = returns[hist_start:i + 1]
        if hist.size < max(8, vol_lookback // 4):
            continue
        sigma = float(np.std(hist))
        if sigma < 1e-8:
            continue
        valid[i] = True
        barrier = barrier_mult * sigma * np.sqrt(horizon)
        future_path = np.cumsum(returns[i + 1:i + 1 + horizon])
        hit_up = np.where(future_path >= barrier)[0]
        hit_down = np.where(future_path <= -barrier)[0]
        first_up = int(hit_up[0]) if hit_up.size > 0 else None
        first_down = int(hit_down[0]) if hit_down.size > 0 else None
        if first_up is None and first_down is None:
            continue
        if first_down is None or (first_up is not None and first_up < first_down):
            upside[i] = 1.0
        elif first_up is None or first_down < first_up:
            downside[i] = 1.0
    return downside, upside, valid


class EventMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Linear(obs_dim, 2)
        else:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_event_model(
    x_train: np.ndarray,
    y_down: np.ndarray,
    y_up: np.ndarray,
    valid: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
) -> EventMLP:
    x = torch.tensor(x_train[valid], dtype=torch.float32, device=device)
    y = torch.tensor(np.stack([y_down[valid], y_up[valid]], axis=1), dtype=torch.float32, device=device)

    model = EventMLP(x.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = ((1.0 - y.mean(dim=0)) / y.mean(dim=0).clamp(min=1e-4)).clamp(min=1.0, max=20.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        count = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(x[idx])
            loss = loss_fn(logits, y[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            count += 1
        print(f"[EVENT] Epoch {epoch + 1}/{epochs} | Loss: {total / max(count, 1):.4f}")
    return model


@torch.no_grad()
def _predict_event_probs(model: EventMLP, features: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(features, dtype=torch.float32, device=device)
    logits = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs[:, 0], probs[:, 1]


def _build_positions(
    p_down: np.ndarray,
    p_up: np.ndarray,
    benchmark_position: float,
    down_position: float,
    up_position: float,
    entry_threshold: float,
    exit_threshold: float,
    margin: float,
    min_hold_bars: int,
) -> np.ndarray:
    positions = np.full(len(p_down), benchmark_position, dtype=np.float32)
    state = 0
    hold = 0
    for t in range(len(positions)):
        down_score = float(p_down[t])
        up_score = float(p_up[t])
        if state == 0:
            if down_score >= entry_threshold and (down_score - up_score) >= margin:
                state = -1
                hold = 0
            elif up_score >= entry_threshold and (up_score - down_score) >= margin:
                state = 1
                hold = 0
        else:
            hold += 1
            if hold >= min_hold_bars:
                if state == -1 and up_score >= entry_threshold and (up_score - down_score) >= margin:
                    state = 1
                    hold = 0
                elif state == 1 and down_score >= entry_threshold and (down_score - up_score) >= margin:
                    state = -1
                    hold = 0
                else:
                    keep_score = down_score if state == -1 else up_score
                    if keep_score < exit_threshold:
                        state = 0
                        hold = 0

        if state == -1:
            positions[t] = down_position
        elif state == 1:
            positions[t] = up_position
    return positions


def _policy_score(metrics, stats: dict) -> float:
    alpha_excess = 100.0 * float(metrics.alpha_excess or 0.0)
    sharpe_delta = float(metrics.sharpe_delta or 0.0)
    maxdd_delta_pt = 100.0 * float(metrics.maxdd_delta or 0.0)
    win_rate = float(metrics.win_rate_vs_bh or 0.0)
    score = 2.0 * alpha_excess + 5.0 * sharpe_delta
    score += 50.0 * max(0.0, -maxdd_delta_pt)
    score += 20.0 * max(0.0, win_rate - 0.5)
    if alpha_excess < 0.0:
        score -= 100.0 + 0.5 * abs(alpha_excess)
    if max(stats["long"], stats["short"]) >= 0.90:
        score -= 100.0
    if stats["turnover"] > 8.0:
        score -= 5.0 * (stats["turnover"] - 8.0)
    return float(score)


@dataclass
class CandidateResult:
    params: dict
    metrics: object
    stats: dict
    scorecard: dict
    score: float
    positions: np.ndarray


def _evaluate_candidates(
    returns: np.ndarray,
    p_down: np.ndarray,
    p_up: np.ndarray,
    cfg: dict,
) -> CandidateResult:
    costs_cfg = cfg.get("costs", {})
    benchmark_position = _benchmark_position_value(cfg)
    ecfg = cfg.get("event_controller", {})
    best: CandidateResult | None = None
    interval = cfg.get("data", {}).get("interval", "15m")

    for down_position in ecfg.get("down_positions", [0.90]):
        for up_position in ecfg.get("up_positions", [1.05]):
            for entry_threshold in ecfg.get("entry_thresholds", [0.60]):
                for exit_threshold in ecfg.get("exit_thresholds", [0.50]):
                    if exit_threshold > entry_threshold:
                        continue
                    for margin in ecfg.get("margins", [0.05]):
                        for min_hold_bars in ecfg.get("min_hold_bars_grid", [8]):
                            positions = _build_positions(
                                p_down,
                                p_up,
                                benchmark_position=benchmark_position,
                                down_position=float(down_position),
                                up_position=float(up_position),
                                entry_threshold=float(entry_threshold),
                                exit_threshold=float(exit_threshold),
                                margin=float(margin),
                                min_hold_bars=int(min_hold_bars),
                            )
                            metrics = Backtest(
                                returns,
                                positions,
                                spread_bps=costs_cfg.get("spread_bps", 0.0),
                                fee_rate=costs_cfg.get("fee_rate", 0.0),
                                slippage_bps=costs_cfg.get("slippage_bps", 0.0),
                                interval=interval,
                                benchmark_positions=np.full(len(positions), benchmark_position, dtype=np.float64),
                            ).run()
                            stats = _action_stats(positions, benchmark_position=benchmark_position)
                            scorecard = _m2_scorecard(metrics, stats, cfg)
                            score = _policy_score(metrics, stats)
                            result = CandidateResult(
                                params={
                                    "down_position": float(down_position),
                                    "up_position": float(up_position),
                                    "entry_threshold": float(entry_threshold),
                                    "exit_threshold": float(exit_threshold),
                                    "margin": float(margin),
                                    "min_hold_bars": int(min_hold_bars),
                                },
                                metrics=metrics,
                                stats=stats,
                                scorecard=scorecard,
                                score=score,
                                positions=positions,
                            )
                            if best is None or result.score > best.score:
                                best = result
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Event target controller probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints/event_controller")
    parser.add_argument("--folds", default=None)
    parser.add_argument("--cost-profile", default=None)
    parser.add_argument("--data_cache_dir", default="checkpoints/data_cache")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg, active_cost_profile = resolve_costs(cfg, args.cost_profile)
    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    symbol = args.symbol or cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")
    data_cfg = cfg.get("data", {})
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}"
    features_cache, returns_cache = _resolve_cache_pair(args.data_cache_dir, cache_tag)
    ohlcv_cache = os.path.join(args.data_cache_dir, f"{cache_tag}_ohlcv.parquet")
    funding_cache = os.path.join(args.data_cache_dir, f"{cache_tag}_funding.parquet")
    oi_cache = os.path.join(args.data_cache_dir, f"{cache_tag}_oi.parquet")
    mark_cache = os.path.join(args.data_cache_dir, f"{cache_tag}_mark.parquet")

    if _cache_is_fresh(features_cache) and _cache_is_fresh(returns_cache):
        print("[Data] Loading cached features...")
        features_df = pd.read_parquet(features_cache)
        raw_returns = pd.read_parquet(returns_cache).squeeze()
    else:
        ohlcv = _read_optional_parquet(ohlcv_cache)
        if ohlcv is None:
            print("[Data] Computing features from OHLCV...")
            ohlcv = fetch_binance_ohlcv(symbol=symbol, interval=interval, start=args.start, end=args.end)
        else:
            print("[Data] Rebuilding features from cached OHLCV...")
        funding_df = _read_optional_parquet(funding_cache)
        oi_df = _read_optional_parquet(oi_cache)
        mark_price_df = _read_optional_parquet(mark_cache)
        features_df = compute_features(
            ohlcv,
            zscore_window_days=zscore_window,
            interval=interval,
            funding_df=funding_df,
            oi_df=oi_df,
            mark_price_df=mark_price_df,
        )
        raw_returns = get_raw_returns(ohlcv)

    feature_extras_cfg = cfg.get("feature_extras", {})
    if feature_extras_cfg.get("rebound_v1", False):
        features_df = augment_with_rebound_features(
            features_df,
            raw_returns,
            zscore_window_days=zscore_window,
            windows_hours=feature_extras_cfg.get("rebound_windows_hours", [24, 72]),
            interval=interval,
        )
        print(f"[Data] Rebound features enabled -> obs_dim={features_df.shape[1]}")

    raw_returns = raw_returns.reindex(features_df.index)

    splits = get_wfo_splits(
        features_df,
        train_years=data_cfg.get("train_years", 2),
        val_months=data_cfg.get("val_months", 3),
        test_months=data_cfg.get("test_months", 3),
    )
    selected_folds = None
    if args.folds:
        selected_folds = {int(x.strip()) for x in args.folds.split(",") if x.strip()}
        splits = [split for split in splits if split.fold_idx in selected_folds]
    if not splits:
        raise RuntimeError("No WFO folds selected")

    print(f"[EVENT] Device={args.device} cost_profile={active_cost_profile} folds={[s.fold_idx for s in splits]}")

    ecfg = cfg.get("event_controller", {})
    horizon = int(ecfg.get("horizon", 16))
    label_mode = str(ecfg.get("label_mode", "quantile"))

    summaries: list[dict] = []
    for split in splits:
        print(
            f"[EVENT] Fold {split.fold_idx} "
            f"train {split.train_start.date()}->{split.train_end.date()} "
            f"val {split.val_start.date()}->{split.val_end.date()} "
            f"test {split.test_start.date()}->{split.test_end.date()}"
        )
        wfo_ds = WFODataset(features_df, raw_returns, split, seq_len=data_cfg.get("seq_len", 64))
        train_x = wfo_ds.train_features.astype(np.float32)
        val_x = wfo_ds.val_features.astype(np.float32)
        test_x = wfo_ds.test_features.astype(np.float32)
        train_ret = wfo_ds.train_returns.astype(np.float32)
        val_ret = wfo_ds.val_returns.astype(np.float32)
        test_ret = wfo_ds.test_returns.astype(np.float32)

        if label_mode == "triple_barrier":
            y_down, y_up, valid = _triple_barrier_targets(
                train_ret,
                horizon=horizon,
                vol_lookback=int(ecfg.get("vol_lookback", 96)),
                barrier_mult=float(ecfg.get("barrier_mult", 1.5)),
            )
        else:
            y_down, y_up, valid = _event_targets(
                train_ret,
                horizon=horizon,
                downside_quantile=float(ecfg.get("downside_quantile", 0.25)),
                upside_quantile=float(ecfg.get("upside_quantile", 0.75)),
                min_neg_frac=float(ecfg.get("min_neg_frac", 0.60)),
                min_pos_frac=float(ecfg.get("min_pos_frac", 0.60)),
            )
        print(
            f"[EVENT] Train labels ({label_mode}): down={float(y_down[valid].mean()):.1%} "
            f"up={float(y_up[valid].mean()):.1%} valid={int(valid.sum())}"
        )

        model = _fit_event_model(
            train_x,
            y_down,
            y_up,
            valid,
            device=args.device,
            epochs=int(ecfg.get("epochs", 20)),
            batch_size=int(ecfg.get("batch_size", 256)),
            lr=float(ecfg.get("lr", 3e-4)),
            weight_decay=float(ecfg.get("weight_decay", 1e-4)),
            hidden_dim=int(ecfg.get("hidden_dim", 128)),
            dropout=float(ecfg.get("dropout", 0.05)),
        )

        val_down, val_up = _predict_event_probs(model, val_x, device=args.device)
        best = _evaluate_candidates(val_ret, val_down, val_up, cfg)
        print(
            f"[EVENT] Fold {split.fold_idx} best val params={best.params} "
            f"score={best.score:.3f} {_format_m2_scorecard(best.scorecard)} {_fmt_action_stats(best.stats)}"
        )

        test_down, test_up = _predict_event_probs(model, test_x, device=args.device)
        test_positions = _build_positions(
            test_down,
            test_up,
            benchmark_position=_benchmark_position_value(cfg),
            **best.params,
        )
        test_metrics = Backtest(
            test_ret,
            test_positions,
            spread_bps=cfg.get("costs", {}).get("spread_bps", 0.0),
            fee_rate=cfg.get("costs", {}).get("fee_rate", 0.0),
            slippage_bps=cfg.get("costs", {}).get("slippage_bps", 0.0),
            interval=interval,
            benchmark_positions=np.full(len(test_positions), _benchmark_position_value(cfg), dtype=np.float64),
        ).run()
        test_stats = _action_stats(test_positions, benchmark_position=_benchmark_position_value(cfg))
        test_scorecard = _m2_scorecard(test_metrics, test_stats, cfg)
        print(
            f"[EVENT] Fold {split.fold_idx} test "
            f"{_format_m2_scorecard(test_scorecard)} {_fmt_action_stats(test_stats)}"
        )

        summaries.append(
            {
                "fold": split.fold_idx,
                "val_score": best.score,
                "val_alpha_pt": 100.0 * float(best.metrics.alpha_excess or 0.0),
                "test_alpha_pt": 100.0 * float(test_metrics.alpha_excess or 0.0),
                "test_sharpe_delta": float(test_metrics.sharpe_delta or 0.0),
                "test_maxdd_delta_pt": 100.0 * float(test_metrics.maxdd_delta or 0.0),
                "test_win_rate_vs_bh": float(test_metrics.win_rate_vs_bh or 0.0),
                "test_m2_pass": bool(test_scorecard["m2_pass"]),
                "params": best.params,
            }
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.checkpoint_dir, "event_controller_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("[EVENT] Summary:")
    print(summary_df.to_string(index=False))
    print(f"[EVENT] Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
