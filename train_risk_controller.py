from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from unidream.actor_critic.imagination_ac import _action_stats, _fmt_action_stats
from unidream.data.dataset import get_wfo_splits, WFODataset
from unidream.eval.backtest import Backtest
from unidream.experiments.m2 import (
    benchmark_position_value as _benchmark_position_value,
    format_m2_scorecard as _format_m2_scorecard,
    m2_scorecard as _m2_scorecard,
)
from unidream.experiments.probe_common import (
    apply_feature_extras,
    load_probe_features,
    resolve_costs,
    set_seed,
)


def _future_extrema_targets(returns: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(returns)
    future_down = np.zeros(T, dtype=np.float32)
    future_up = np.zeros(T, dtype=np.float32)
    valid = np.zeros(T, dtype=bool)
    if T <= horizon:
        return future_down, future_up, valid
    for i in range(T - horizon):
        future_path = np.cumsum(returns[i + 1:i + 1 + horizon])
        future_down[i] = float(np.min(future_path))
        future_up[i] = float(np.max(future_path))
        valid[i] = True
    return future_down, future_up, valid


class RiskMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128, dropout: float = 0.05):
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


def _fit_model(
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
) -> RiskMLP:
    x = torch.tensor(x_train[valid], dtype=torch.float32, device=device)
    y = torch.tensor(np.stack([y_down[valid], y_up[valid]], axis=1), dtype=torch.float32, device=device)
    model = RiskMLP(x.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.01)

    n = x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        count = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(x[idx])
            loss = loss_fn(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            count += 1
        print(f"[RISK] Epoch {epoch + 1}/{epochs} | Loss: {total / max(count, 1):.5f}")
    return model


@torch.no_grad()
def _predict(model: RiskMLP, features: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    x = torch.tensor(features, dtype=torch.float32, device=device)
    pred = model(x).cpu().numpy()
    return pred[:, 0], pred[:, 1]


def _build_positions(
    pred_down: np.ndarray,
    pred_up: np.ndarray,
    benchmark_position: float,
    down_position: float,
    up_position: float,
    down_entry: float,
    up_entry: float,
    exit_band: float,
    margin: float,
    min_hold_bars: int,
) -> np.ndarray:
    positions = np.full(len(pred_down), benchmark_position, dtype=np.float32)
    state = 0
    hold = 0
    for t in range(len(positions)):
        downside = max(0.0, -float(pred_down[t]))
        upside = max(0.0, float(pred_up[t]))
        if state == 0:
            if downside >= down_entry and downside >= upside + margin:
                state = -1
                hold = 0
            elif upside >= up_entry and upside >= downside + margin:
                state = 1
                hold = 0
        else:
            hold += 1
            if hold >= min_hold_bars:
                if state == -1 and upside >= up_entry and upside >= downside + margin:
                    state = 1
                    hold = 0
                elif state == 1 and downside >= down_entry and downside >= upside + margin:
                    state = -1
                    hold = 0
                elif max(downside, upside) < exit_band:
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


def _evaluate_candidates(returns: np.ndarray, pred_down: np.ndarray, pred_up: np.ndarray, cfg: dict):
    rcfg = cfg.get("risk_controller", {})
    benchmark_position = _benchmark_position_value(cfg)
    costs_cfg = cfg.get("costs", {})
    interval = cfg.get("data", {}).get("interval", "15m")
    best = None
    for down_position in rcfg.get("down_positions", [0.90]):
        for up_position in rcfg.get("up_positions", [1.05]):
            for down_entry in rcfg.get("down_entry_grid", [0.01]):
                for up_entry in rcfg.get("up_entry_grid", [0.01]):
                    for exit_band in rcfg.get("exit_band_grid", [0.005]):
                        for margin in rcfg.get("margin_grid", [0.0]):
                            for min_hold_bars in rcfg.get("min_hold_bars_grid", [8]):
                                positions = _build_positions(
                                    pred_down,
                                    pred_up,
                                    benchmark_position=benchmark_position,
                                    down_position=float(down_position),
                                    up_position=float(up_position),
                                    down_entry=float(down_entry),
                                    up_entry=float(up_entry),
                                    exit_band=float(exit_band),
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
                                if best is None or score > best["score"]:
                                    best = {
                                        "params": {
                                            "down_position": float(down_position),
                                            "up_position": float(up_position),
                                            "down_entry": float(down_entry),
                                            "up_entry": float(up_entry),
                                            "exit_band": float(exit_band),
                                            "margin": float(margin),
                                            "min_hold_bars": int(min_hold_bars),
                                        },
                                        "positions": positions,
                                        "metrics": metrics,
                                        "stats": stats,
                                        "scorecard": scorecard,
                                        "score": score,
                                    }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk regression controller probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints/risk_controller")
    parser.add_argument("--folds", default=None)
    parser.add_argument("--cost-profile", default=None)
    parser.add_argument("--data_cache_dir", default="checkpoints/data_cache")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg, active_cost_profile = resolve_costs(cfg, args.cost_profile)
    set_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    symbol = cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    features_df, raw_returns, _ = load_probe_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        data_cache_dir=args.data_cache_dir,
    )
    features_df, raw_returns = apply_feature_extras(
        features_df,
        raw_returns,
        feature_extras_cfg=cfg.get("feature_extras", {}),
        zscore_window=zscore_window,
        interval=interval,
    )

    splits = get_wfo_splits(
        features_df,
        train_years=cfg.get("data", {}).get("train_years", 1),
        val_months=cfg.get("data", {}).get("val_months", 2),
        test_months=cfg.get("data", {}).get("test_months", 2),
    )
    if args.folds:
        wanted = {int(x.strip()) for x in args.folds.split(",") if x.strip()}
        splits = [split for split in splits if split.fold_idx in wanted]
    print(f"[RISK] Device={args.device} cost_profile={active_cost_profile} folds={[s.fold_idx for s in splits]}")

    rcfg = cfg.get("risk_controller", {})
    horizon = int(rcfg.get("horizon", 16))
    summaries = []
    for split in splits:
        print(f"[RISK] Fold {split.fold_idx}")
        wfo_ds = WFODataset(features_df, raw_returns, split, seq_len=cfg.get("data", {}).get("seq_len", 64))
        train_x = wfo_ds.train_features.astype(np.float32)
        val_x = wfo_ds.val_features.astype(np.float32)
        test_x = wfo_ds.test_features.astype(np.float32)
        feature_subset = rcfg.get("feature_subset")
        if feature_subset:
            name_to_idx = {name: idx for idx, name in enumerate(wfo_ds.feature_columns)}
            missing = [name for name in feature_subset if name not in name_to_idx]
            keep = [name_to_idx[name] for name in feature_subset if name in name_to_idx]
            if missing:
                print(f"[RISK] Missing feature subset entries: {missing}")
                if not rcfg.get("feature_subset_allow_missing", False):
                    raise RuntimeError("Configured feature_subset is missing required feature columns")
            if not keep:
                raise RuntimeError("Configured feature_subset did not match any feature columns")
            train_x = train_x[:, keep]
            val_x = val_x[:, keep]
            test_x = test_x[:, keep]
            kept_names = [wfo_ds.feature_columns[idx] for idx in keep]
            print(f"[RISK] Feature subset ({len(keep)}): {kept_names}")
        train_ret = wfo_ds.train_returns.astype(np.float32)
        val_ret = wfo_ds.val_returns.astype(np.float32)
        test_ret = wfo_ds.test_returns.astype(np.float32)

        y_down, y_up, valid = _future_extrema_targets(train_ret, horizon=horizon)
        print(
            f"[RISK] Train targets: "
            f"mean_down={float(y_down[valid].mean()):+.5f} mean_up={float(y_up[valid].mean()):+.5f} valid={int(valid.sum())}"
        )
        model = _fit_model(
            train_x,
            y_down,
            y_up,
            valid,
            device=args.device,
            epochs=int(rcfg.get("epochs", 20)),
            batch_size=int(rcfg.get("batch_size", 256)),
            lr=float(rcfg.get("lr", 3e-4)),
            weight_decay=float(rcfg.get("weight_decay", 1e-4)),
            hidden_dim=int(rcfg.get("hidden_dim", 128)),
            dropout=float(rcfg.get("dropout", 0.05)),
        )
        val_down, val_up = _predict(model, val_x, device=args.device)
        best = _evaluate_candidates(val_ret, val_down, val_up, cfg)
        print(
            f"[RISK] Fold {split.fold_idx} best val params={best['params']} "
            f"score={best['score']:.3f} {_format_m2_scorecard(best['scorecard'])} {_fmt_action_stats(best['stats'])}"
        )

        test_down, test_up = _predict(model, test_x, device=args.device)
        test_positions = _build_positions(
            test_down,
            test_up,
            benchmark_position=_benchmark_position_value(cfg),
            **best["params"],
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
            f"[RISK] Fold {split.fold_idx} test "
            f"{_format_m2_scorecard(test_scorecard)} {_fmt_action_stats(test_stats)}"
        )
        summaries.append(
            {
                "fold": split.fold_idx,
                "val_score": best["score"],
                "val_alpha_pt": 100.0 * float(best["metrics"].alpha_excess or 0.0),
                "test_alpha_pt": 100.0 * float(test_metrics.alpha_excess or 0.0),
                "test_sharpe_delta": float(test_metrics.sharpe_delta or 0.0),
                "test_maxdd_delta_pt": 100.0 * float(test_metrics.maxdd_delta or 0.0),
                "test_win_rate_vs_bh": float(test_metrics.win_rate_vs_bh or 0.0),
                "test_m2_pass": bool(test_scorecard["m2_pass"]),
                "params": best["params"],
            }
        )
    summary = pd.DataFrame(summaries)
    summary_path = os.path.join(args.checkpoint_dir, "risk_controller_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("[RISK] Summary:")
    print(summary.to_string(index=False))
    print(f"[RISK] Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
