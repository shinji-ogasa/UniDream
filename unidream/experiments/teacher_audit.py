from __future__ import annotations

import copy
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from unidream.actor_critic.imagination_ac import _action_stats
from unidream.data.download import (
    fetch_binance_ohlcv,
    fetch_funding_rate,
    fetch_mark_price_klines,
    fetch_open_interest_hist,
)
from unidream.data.features import compute_features, get_raw_returns
from unidream.data.dataset import WFODataset
from unidream.data.oracle import _forward_window_stats

from .fold_inputs import prepare_fold_inputs
from .runtime import read_extra_series_caches, read_optional_parquet
from .wfo_runtime import build_wfo_splits, select_wfo_splits


def _find_optional_cache(source_dirs: list[str], cache_tag: str, suffix: str) -> str | None:
    for cache_dir in source_dirs:
        direct = os.path.join(cache_dir, f"{cache_tag}_{suffix}.parquet")
        if os.path.exists(direct):
            return direct
        candidates = sorted(glob.glob(os.path.join(cache_dir, f"{cache_tag}*_{suffix}.parquet")))
        if candidates:
            return candidates[0]
    return None


def load_audit_features(
    *,
    symbol: str,
    interval: str,
    start: str,
    end: str,
    zscore_window: int,
    cache_dir: str,
    cache_tag: str,
    raw_cache_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    features_cache = os.path.join(cache_dir, f"{cache_tag}_features.parquet")
    returns_cache = os.path.join(cache_dir, f"{cache_tag}_returns.parquet")
    if os.path.exists(features_cache) and os.path.exists(returns_cache):
        features_df = pd.read_parquet(features_cache)
        raw_returns = pd.read_parquet(returns_cache).squeeze()
        return features_df, raw_returns

    source_dirs: list[str] = [cache_dir]
    if raw_cache_dir and os.path.abspath(raw_cache_dir) != os.path.abspath(cache_dir):
        source_dirs.append(raw_cache_dir)

    ohlcv_path = _find_optional_cache(source_dirs, cache_tag, "ohlcv")
    funding_path = _find_optional_cache(source_dirs, cache_tag, "funding")
    oi_path = _find_optional_cache(source_dirs, cache_tag, "oi")
    mark_path = _find_optional_cache(source_dirs, cache_tag, "mark")

    df = read_optional_parquet(ohlcv_path) if ohlcv_path else None
    funding_df = read_optional_parquet(funding_path) if funding_path else None
    oi_df = read_optional_parquet(oi_path) if oi_path else None
    mark_price_df = read_optional_parquet(mark_path) if mark_path else None

    extra_series: dict[str, pd.Series] = {}
    for source_dir in source_dirs:
        extra_series.update(read_extra_series_caches(source_dir, cache_tag))

    if df is None:
        df = fetch_binance_ohlcv(symbol, interval, start, end)
    if funding_df is None:
        try:
            funding_df = fetch_funding_rate(symbol, start, end)
        except Exception:
            funding_df = None
    if oi_df is None:
        try:
            oi_df = fetch_open_interest_hist(symbol, interval, start, end)
        except Exception:
            oi_df = None
    if mark_price_df is None:
        try:
            mark_price_df = fetch_mark_price_klines(symbol, interval, start, end)
        except Exception:
            mark_price_df = None

    features_df = compute_features(
        df,
        zscore_window_days=zscore_window,
        interval=interval,
        funding_df=funding_df,
        oi_df=oi_df,
        mark_price_df=mark_price_df,
        extra_series=extra_series,
    )
    raw_returns = get_raw_returns(df)
    common_idx = features_df.index.intersection(raw_returns.index)
    return features_df.loc[common_idx], raw_returns.loc[common_idx]


def _overall_and_regime_rows(
    *,
    config_name: str,
    fold_idx: int,
    split_name: str,
    positions: np.ndarray,
    returns: np.ndarray,
    regime_probs: np.ndarray | None,
    benchmark_position: float,
    min_hold: int,
    regime_expected_returns: np.ndarray | None = None,
) -> list[dict]:
    rows: list[dict] = []

    def _row_for_subset(regime_label: str, pos_subset: np.ndarray, ret_subset: np.ndarray) -> dict:
        stats = _action_stats(pos_subset, benchmark_position=benchmark_position)
        return {
            "config": config_name,
            "fold": fold_idx,
            "split": split_name,
            "regime": regime_label,
            "n_bars": int(len(pos_subset)),
            "fraction": float(len(pos_subset) / max(len(positions), 1)),
            "long_ratio": float(stats["long"]),
            "short_ratio": float(stats["short"]),
            "flat_ratio": float(stats["flat"]),
            "mean_overlay": float(stats["mean"]),
            "turnover": float(stats["turnover"]),
            "switches": int(stats["switches"]),
            "avg_hold": float(stats["avg_hold"]),
            "mean_return": float(np.mean(ret_subset)) if len(ret_subset) else 0.0,
            "mean_abs_return": float(np.mean(np.abs(ret_subset))) if len(ret_subset) else 0.0,
            "oracle_min_hold": int(min_hold),
        }

    rows.append(_row_for_subset("all", positions, returns))
    if regime_probs is None:
        return rows

    regime_ids = np.argmax(regime_probs, axis=1)
    for regime_id in range(regime_probs.shape[1]):
        mask = regime_ids == regime_id
        if not np.any(mask):
            continue
        row = _row_for_subset(f"regime_{regime_id}", positions[mask], returns[mask])
        if regime_expected_returns is not None and regime_id < len(regime_expected_returns):
            row["regime_expected_return"] = float(regime_expected_returns[regime_id])
        rows.append(row)
    return rows


def run_teacher_audit(
    *,
    cfg: dict,
    config_name: str,
    features_df: pd.DataFrame,
    raw_returns: pd.Series,
    min_hold_values: list[int],
    folds_arg: str | None = None,
    checkpoint_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    data_cfg = cfg.get("data", {})
    splits = build_wfo_splits(features_df, data_cfg)
    selected_splits, _ = select_wfo_splits(splits, folds_arg)
    seq_len = data_cfg.get("seq_len", 64)

    summary_rows: list[dict] = []
    hold_rows: list[dict] = []

    for min_hold in min_hold_values:
        audit_cfg = copy.deepcopy(cfg)
        audit_cfg.setdefault("oracle", {})["min_hold"] = int(min_hold)
        cost_cfg = dict(audit_cfg.get("costs", {}))
        reward_cfg = audit_cfg.get("reward", {})
        ac_cfg = audit_cfg.get("ac", {})
        bc_cfg = audit_cfg.get("bc", {})
        hmm_n_states = int(audit_cfg.get("eval", {}).get("hmm_n_states", 3))

        for split in selected_splits:
            wfo_dataset = WFODataset(features_df, raw_returns, split, seq_len=seq_len)
            fold_inputs = prepare_fold_inputs(
                wfo_dataset=wfo_dataset,
                cfg=audit_cfg,
                costs_cfg=cost_cfg,
                ac_cfg=ac_cfg,
                bc_cfg=bc_cfg,
                reward_cfg=reward_cfg,
                action_stats_fn=_action_stats,
                format_action_stats_fn=lambda stats: (
                    f"long={stats['long']:.0%} short={stats['short']:.0%} flat={stats['flat']:.0%} "
                    f"mean={stats['mean']:+.3f} switches={stats['switches']} avg_hold={stats['avg_hold']:.1f}b "
                    f"turnover={stats['turnover']:.2f}"
                ),
                benchmark_position=benchmark_position,
                forward_window_stats_fn=_forward_window_stats,
                log_ts=lambda: "audit",
            )
            detector = fold_inputs["hmm_det"]
            regime_expected = detector.expected_returns() if detector is not None else np.full(hmm_n_states, np.nan)

            train_rows = _overall_and_regime_rows(
                config_name=config_name,
                fold_idx=split.fold_idx,
                split_name="train",
                positions=fold_inputs["oracle_positions"],
                returns=wfo_dataset.train_returns,
                regime_probs=fold_inputs["train_regime_probs"],
                benchmark_position=benchmark_position,
                min_hold=min_hold,
                regime_expected_returns=regime_expected,
            )
            val_rows = _overall_and_regime_rows(
                config_name=config_name,
                fold_idx=split.fold_idx,
                split_name="val",
                positions=fold_inputs["val_oracle_positions"],
                returns=wfo_dataset.val_returns,
                regime_probs=fold_inputs["val_regime_probs"],
                benchmark_position=benchmark_position,
                min_hold=min_hold,
                regime_expected_returns=regime_expected,
            )
            hold_rows.extend(train_rows)
            hold_rows.extend(val_rows)

    detail_df = pd.DataFrame(hold_rows).sort_values(["oracle_min_hold", "fold", "split", "regime"]).reset_index(drop=True)
    if detail_df.empty:
        return detail_df, detail_df

    summary_df = (
        detail_df.groupby(["config", "oracle_min_hold", "split", "regime"], dropna=False)
        .agg(
            n_bars=("n_bars", "sum"),
            fraction=("fraction", "mean"),
            long_ratio=("long_ratio", "mean"),
            short_ratio=("short_ratio", "mean"),
            flat_ratio=("flat_ratio", "mean"),
            mean_overlay=("mean_overlay", "mean"),
            turnover=("turnover", "mean"),
            switches=("switches", "mean"),
            avg_hold=("avg_hold", "mean"),
            mean_return=("mean_return", "mean"),
            mean_abs_return=("mean_abs_return", "mean"),
            regime_expected_return=("regime_expected_return", "mean"),
        )
        .reset_index()
        .sort_values(["oracle_min_hold", "split", "regime"])
    )

    if checkpoint_dir:
        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_dir / f"{config_name}_teacher_audit_summary.csv", index=False)
        detail_df.to_csv(out_dir / f"{config_name}_teacher_audit_detail.csv", index=False)
    return summary_df, detail_df
