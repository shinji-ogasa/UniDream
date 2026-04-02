"""UniDream メイン学習スクリプト.

SPEC.md の実装順序に従って以下を実行する:
  1. データ取得・特徴量計算
  2. WFO 分割
  3. train 期間で Hindsight Oracle 計算
  4. BC 初期化（Actor を oracle に模倣させる）
  5. Transformer 世界モデル学習
  6. Imagination AC fine-tune
  7. test 期間バックテスト
  8. PBO / Deflated Sharpe による過学習検出
  9. HMM レジーム別メトリクス

Usage:
    python train.py [--config configs/trading.yaml] [--symbol BTCUSDT]
                    [--start 2018-01-01] [--end 2024-01-01]
                    [--device cuda] [--seed 42] [--resume]
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

from unidream.data.download import fetch_binance_ohlcv, fetch_funding_rate, fetch_open_interest_hist
from unidream.data.features import compute_features, get_raw_returns
from unidream.data.oracle import hindsight_oracle_dp, oracle_to_dataset, ACTIONS as _ACTIONS
from unidream.data.dataset import get_wfo_splits, WFODataset, SequenceDataset
from unidream.world_model.train_wm import WorldModelTrainer, build_ensemble
from unidream.actor_critic.actor import Actor
from unidream.actor_critic.critic import Critic
from unidream.actor_critic.bc_pretrain import BCPretrainer
from unidream.actor_critic.imagination_ac import ImagACTrainer, _action_stats, _fmt_action_stats, _ac_alerts
from unidream.eval.backtest import Backtest, pnl_attribution
from unidream.eval.wfo import aggregate_wfo_results
from unidream.eval.pbo import compute_pbo, deflated_sharpe
from unidream.eval.regime import RegimeDetector, regime_metrics, print_regime_report


_CACHE_STALE_DAYS = 7  # キャッシュがこの日数以上古ければ再取得


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _cache_is_fresh(path: str, stale_days: int = _CACHE_STALE_DAYS) -> bool:
    """キャッシュファイルが存在し、stale_days 以内に更新されていれば True."""
    if not os.path.exists(path):
        return False
    import time
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    return age_days < stale_days


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_PIPELINE_STAGES = ("wm", "bc", "ac", "test")
_STAGE_TO_INDEX = {stage: idx for idx, stage in enumerate(_PIPELINE_STAGES)}


def _stage_idx(stage: str) -> int:
    return _STAGE_TO_INDEX[stage]


def _benchmark_positions(length: int, cfg: dict) -> np.ndarray:
    benchmark_pos = cfg.get("reward", {}).get("benchmark_position", 1.0)
    return np.full(length, benchmark_pos, dtype=np.float64)


def resolve_costs(cfg: dict, cost_profile: str | None = None) -> tuple[dict, str]:
    """Resolve trading costs from either legacy `costs` or named `cost_profiles`."""
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


def run_fold(
    fold_idx: int,
    wfo_dataset: WFODataset,
    cfg: dict,
    device: str,
    checkpoint_dir: str,
    resume: bool = False,
    start_from: str = "wm",
    stop_after: str = "test",
) -> dict:
    """1 WFO fold の学習・評価を実行する.

    resume=True の場合、保存済みチェックポイントから再開する:
      - ac.pt が存在 → fold 全体をスキップ（バックテストのみ再実行）
      - bc_actor.pt が存在 → WM・BC をロードして AC から再開
      - world_model.pt が存在 → WM をロードして BC から再開

    Returns:
        {"fold": fold_idx, "metrics": BacktestMetrics, "positions": np.ndarray}
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx}: train {wfo_dataset.split.train_start.date()} → "
          f"{wfo_dataset.split.train_end.date()} | "
          f"test {wfo_dataset.split.test_start.date()} → "
          f"{wfo_dataset.split.test_end.date()}")
    print(f"{'='*60}")

    ac_cfg = cfg.get("ac", {})
    bc_cfg = cfg.get("bc", {})
    wm_cfg = cfg.get("world_model", {})
    costs_cfg = cfg.get("costs", {})

    obs_dim = wfo_dataset.obs_dim
    seq_len = cfg.get("data", {}).get("seq_len", 64)

    fold_ckpt_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    wm_path = os.path.join(fold_ckpt_dir, "world_model.pt")
    bc_path = os.path.join(fold_ckpt_dir, "bc_actor.pt")
    ac_path = os.path.join(fold_ckpt_dir, "ac.pt")

    start_idx = _stage_idx(start_from)
    stop_idx = _stage_idx(stop_after)
    has_wm_ckpt = os.path.exists(wm_path)
    has_bc_ckpt = os.path.exists(bc_path)
    has_ac_ckpt = os.path.exists(ac_path)
    has_wm = has_wm_ckpt and (resume or start_idx > _stage_idx("wm"))
    has_bc = has_bc_ckpt and (resume or start_idx > _stage_idx("bc"))
    has_ac = has_ac_ckpt and (resume or start_idx > _stage_idx("ac"))

    if start_idx > _stage_idx("wm") and not has_wm_ckpt:
        raise FileNotFoundError(
            f"Fold {fold_idx}: missing WM checkpoint for --start-from {start_from}: {wm_path}"
        )
    if start_from == "ac" and not has_bc_ckpt:
        raise FileNotFoundError(
            f"Fold {fold_idx}: missing BC checkpoint for --start-from ac: {bc_path}"
        )
    if start_from == "test" and not (has_bc_ckpt or has_ac_ckpt):
        raise FileNotFoundError(
            f"Fold {fold_idx}: --start-from test requires {bc_path} or {ac_path}"
        )

    # --------- Step 1: Hindsight Oracle ---------
    print(f"\n[{_ts()}] [Step 1] Hindsight Oracle DP...")
    train_returns = wfo_dataset.train_returns
    oracle_cfg = cfg.get("oracle", {})
    reward_cfg = cfg.get("reward", {})
    oracle_action_values = np.asarray(
        oracle_cfg.get("action_values", cfg.get("actions", {}).get("values", _ACTIONS)),
        dtype=np.float32,
    )
    oracle_min_hold = oracle_cfg.get("min_hold", 0)
    oracle_soft_temp = oracle_cfg.get("soft_label_temp", 0.0)
    oracle_reward_mode = reward_cfg.get("mode", "absolute")
    oracle_benchmark_position = reward_cfg.get("benchmark_position", 1.0)
    oracle_actions, oracle_values, oracle_soft_labels = hindsight_oracle_dp(
        train_returns,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        discount=oracle_cfg.get("discount", 1.0),
        min_hold=oracle_min_hold,
        soft_label_temp=oracle_soft_temp,
        reward_mode=oracle_reward_mode,
        benchmark_position=oracle_benchmark_position,
        action_values=oracle_action_values,
    )
    print(f"  Oracle computed: {len(oracle_actions)} steps, "
          f"mean value={oracle_values.mean():.4f}")
    print(f"  Oracle objective: {oracle_reward_mode} (benchmark={oracle_benchmark_position:+.2f})")
    _oracle_pos = oracle_action_values[oracle_actions]
    _oracle_s = _action_stats(_oracle_pos)
    print(f"  Oracle dist: {_fmt_action_stats(_oracle_s)}")

    # Val oracle actions（分布比較・WM 学習に使用）
    val_oracle_actions, _, _ = hindsight_oracle_dp(
        wfo_dataset.val_returns,
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        discount=oracle_cfg.get("discount", 1.0),
        min_hold=oracle_min_hold,
        reward_mode=oracle_reward_mode,
        benchmark_position=oracle_benchmark_position,
        action_values=oracle_action_values,
    )
    oracle_positions = oracle_action_values[oracle_actions].astype(np.float32)
    val_oracle_positions = oracle_action_values[val_oracle_actions].astype(np.float32)

    # --------- HMM レジーム事後確率（Actor 入力用）---------
    # Actor 生成前に計算して regime_dim を確定する
    n_states = cfg.get("eval", {}).get("hmm_n_states", 3)
    hmm_det = None
    regime_dim = 0
    train_regime_probs = None
    val_regime_probs = None
    test_regime_probs = None
    try:
        from unidream.eval.regime import RegimeDetector
        hmm_det = RegimeDetector(n_states=n_states)
        hmm_det.fit(wfo_dataset.train_returns)  # 内部で平均リターン昇順にソート
        train_regime_probs = hmm_det.predict_proba(wfo_dataset.train_returns).astype(np.float32)
        val_regime_probs = hmm_det.predict_proba(wfo_dataset.val_returns).astype(np.float32)
        test_regime_probs = hmm_det.predict_proba(wfo_dataset.test_returns).astype(np.float32)
        regime_dim = n_states
        print(f"[Regime] HMM fitted, regime_dim={regime_dim}")
    except Exception as e:
        print(f"[Regime] HMM skipped: {e}")

    # --------- Step 2: 世界モデル学習 ---------
    ensemble = build_ensemble(obs_dim, cfg)
    wm_trainer = WorldModelTrainer(ensemble, cfg, device=device)

    if has_wm:
        print(f"\n[{_ts()}] [Step 2] World Model — loading checkpoint: {wm_path}")
        wm_trainer.load(wm_path)
    else:
        print(f"\n[{_ts()}] [Step 2] World Model Training...")
        train_ds_with_actions = SequenceDataset(
            wfo_dataset.train_features,
            seq_len=seq_len,
            actions=oracle_positions[:len(wfo_dataset.train_features)],
            returns=train_returns,
        )
        val_ds = SequenceDataset(
            wfo_dataset.val_features,
            seq_len=seq_len,
            actions=val_oracle_positions[:len(wfo_dataset.val_features)],
            returns=wfo_dataset.val_returns,
        )
        wm_trainer.train_on_dataset(
            train_ds_with_actions,
            val_dataset=val_ds,
            checkpoint_path=wm_path,
        )

    if stop_after == "wm":
        print(f"\n[{_ts()}] [Stop] Requested stop after WM")
        return {"fold": fold_idx, "completed_stage": "wm"}

    # train 期間の全シーケンスをエンコード
    # BC 学習・評価の一貫性のため no-action（flat）context で encode する。
    # oracle action context で h を作ると「未来情報リーク入り h」になり、
    # val/test（no-action h）との分布ギャップが大きすぎて BC が崩壊する。
    # z はエンコーダ出力で action-independent なため、どちらの encode でも同一。
    encoded = wm_trainer.encode_sequence(
        wfo_dataset.train_features,
        actions=None,   # no oracle context → BC/val/test で h 分布を一致させる
        seq_len=seq_len,
    )
    z_train = encoded["z"]
    h_train = encoded["h"]

    # --------- Step 3: BC 初期化 ---------
    actor = Actor(
        z_dim=ensemble.get_z_dim(),
        h_dim=ensemble.get_d_model(),
        act_dim=int(len(oracle_action_values)),
        hidden_dim=ac_cfg.get("actor_hidden", 256),
        n_layers=ac_cfg.get("ac_layers", 2),
        regime_dim=regime_dim,
        dropout_p=ac_cfg.get("actor_dropout", 0.0),
    )
    actor.target_values = oracle_action_values.astype(np.float32)
    actor.benchmark_position = reward_cfg.get("benchmark_position", 1.0)
    actor.abs_min_position = ac_cfg.get("abs_min_position", -1.0)
    actor.abs_max_position = ac_cfg.get("abs_max_position", 1.0)
    actor.infer_temperature = ac_cfg.get("infer_temperature", 1.0)
    actor.infer_trade_threshold = ac_cfg.get("infer_trade_threshold", 0.5)
    actor.max_position_step = ac_cfg.get("max_position_step", 10.0)
    actor.min_band = ac_cfg.get("min_band", 0.02)
    actor.max_band = ac_cfg.get("max_band", 0.20)
    actor.min_target_std = ac_cfg.get("min_target_std", 0.05)
    actor.max_target_std = ac_cfg.get("max_target_std", 0.35)

    if has_bc:
        print(f"\n[{_ts()}] [Step 3] BC — loading checkpoint: {bc_path}")
        bc_trainer = BCPretrainer(
            actor=actor,
            z_dim=ensemble.get_z_dim(),
            h_dim=ensemble.get_d_model(),
            target_aux_coef=bc_cfg.get("target_aux_coef", 1.0),
            trade_aux_coef=bc_cfg.get("trade_aux_coef", 0.5),
            band_aux_coef=bc_cfg.get("band_aux_coef", 0.25),
            inventory_noise_std=bc_cfg.get("inventory_noise_std", 0.0),
            device=device,
        )
        bc_trainer.load(bc_path)
    else:
        if start_idx <= _stage_idx("bc"):
            print(f"\n[{_ts()}] [Step 3] BC Pre-training...")
            bc_trainer = BCPretrainer(
                actor=actor,
                z_dim=ensemble.get_z_dim(),
                h_dim=ensemble.get_d_model(),
                lr=bc_cfg.get("lr", 3e-4),
                batch_size=bc_cfg.get("batch_size", 256),
                n_epochs=bc_cfg.get("n_epochs", 5),
                sirl_hidden=bc_cfg.get("sirl_hidden", 128),
                label_smoothing=bc_cfg.get("label_smoothing", 0.0),
                entropy_coef=bc_cfg.get("entropy_coef", 0.0),
                chunk_size=bc_cfg.get("chunk_size", 1),
                class_balanced=bc_cfg.get("class_balanced", False),
                target_aux_coef=bc_cfg.get("target_aux_coef", 1.0),
                trade_aux_coef=bc_cfg.get("trade_aux_coef", 0.5),
                band_aux_coef=bc_cfg.get("band_aux_coef", 0.25),
                inventory_noise_std=bc_cfg.get("inventory_noise_std", 0.0),
                device=device,
            )
            T_enc = min(len(z_train), len(oracle_positions))
            bc_trainer.train(
                z=z_train[:T_enc],
                h=h_train[:T_enc],
                oracle_positions=oracle_positions[:T_enc],
                regime_probs=train_regime_probs[:T_enc] if train_regime_probs is not None else None,
                soft_labels=None,
            )
            bc_trainer.save(bc_path)
        else:
            print(f"\n[{_ts()}] [Step 3] BC — skipped (AC checkpoint will provide actor weights)")

    if stop_after == "bc":
        print(f"\n[{_ts()}] [Stop] Requested stop after BC")
        return {"fold": fold_idx, "completed_stage": "bc"}

    # --------- Step 4: Imagination AC Fine-tune ---------
    # no-action h（z_train/h_train）を AC でもそのまま使う。
    # BC-action re-encode は廃止: test/val も no-action h のため、
    # re-encode すると AC 開始状態が no-action h から外れ、
    # BC 正則化（_oracle_z/_oracle_h = no-action h）と AC gradient が
    # 異なる h 分布上で計算されて矛盾した gradient が生じる。
    # no-action h に統一することで BC train/AC train/val/test が一貫する。

    ac_requested = stop_idx >= _stage_idx("ac") or has_ac
    if ac_requested:
        critic = Critic(
            z_dim=ensemble.get_z_dim(),
            h_dim=ensemble.get_d_model(),
            hidden_dim=ac_cfg.get("critic_hidden", 256),
            n_layers=ac_cfg.get("ac_layers", 2),
            n_bins=wm_cfg.get("n_bins", 255),
            ema_decay=ac_cfg.get("ema_decay", 0.98),
        )
        ac_trainer = ImagACTrainer(
            actor=actor,
            critic=critic,
            ensemble=ensemble,
            cfg=cfg,
            device=device,
        )
        if has_ac:
            ac_trainer.load(ac_path)

        if start_idx <= _stage_idx("ac") or has_ac:
            # oracle データは resume 時も必要（BC 損失計算用）
            # z/h は oracle エンコード（z は obs のみなので同一、h は BC エンコードとは別途保持）
            T_enc = min(len(z_train), len(oracle_positions))
            ac_trainer.set_oracle_data(
                z=z_train[:T_enc],
                h=h_train[:T_enc],
                oracle_positions=oracle_positions[:T_enc],
                regime_probs=train_regime_probs[:T_enc] if train_regime_probs is not None else None,
            )

            encoded_list = [{
                "z": z_train,
                "h": h_train,
                "regime": train_regime_probs if train_regime_probs is not None else None,
            }]

            # Val encoding を no-action（flat）で一度だけ固定する。
            # test と同じ single-pass encoding に統一することで val/test の比較を整合させる。
            # 自己参照ループ（AC の自身の予測を context に使う → val Sharpe 過大）は
            # ここで固定した z_val_fixed/h_val_fixed を AC 学習中ずっと使い回すことで回避する。
            val_features_arr = wfo_dataset.val_features
            val_returns_arr = wfo_dataset.val_returns
            if len(val_features_arr) > 0:
                _enc_val_fixed = wm_trainer.encode_sequence(val_features_arr, seq_len=seq_len)
                z_val_fixed = _enc_val_fixed["z"]
                h_val_fixed = _enc_val_fixed["h"]
            else:
                z_val_fixed = h_val_fixed = None

            # Val backtest function — used for AC checkpoint selection
            def _val_eval() -> tuple[float, str]:
                if z_val_fixed is None:
                    return -float("inf"), "raw=-inf score=-inf"
                pos = actor.predict_positions(
                    z_val_fixed, h_val_fixed, regime_np=val_regime_probs, device=device
                )
                T_min = min(len(val_returns_arr), len(pos))
                metrics = Backtest(
                    val_returns_arr[:T_min], pos[:T_min],
                    spread_bps=costs_cfg.get("spread_bps", 5.0),
                    fee_rate=costs_cfg.get("fee_rate", 0.0004),
                    slippage_bps=costs_cfg.get("slippage_bps", 2.0),
                    interval=cfg.get("data", {}).get("interval", "15m"),
                    benchmark_positions=_benchmark_positions(T_min, cfg),
                ).run()
                stats = _action_stats(pos[:T_min])

                alpha_excess = 100.0 * (metrics.alpha_excess or 0.0)
                sharpe_delta = metrics.sharpe_delta or 0.0
                score = 2.0 * alpha_excess + 5.0 * sharpe_delta
                penalty = 0.0
                if alpha_excess <= 0.0:
                    penalty += 100.0 + 0.5 * abs(alpha_excess)
                if stats["flat"] >= 0.50:
                    penalty += 30.0
                if stats["flat"] >= 0.80:
                    penalty += 100.0
                if stats["long"] >= 0.85:
                    penalty += 120.0
                if stats["short"] >= 0.85:
                    penalty += 120.0
                if max(stats["long"], stats["short"], stats["flat"]) >= 0.80:
                    penalty += 200.0
                if stats["avg_hold"] < 2.0:
                    penalty += 10.0
                if stats["switches"] == 0:
                    penalty += 25.0

                score -= penalty
                label = (
                    f"alpha={alpha_excess:+.2f}pt sharpeΔ={sharpe_delta:+.3f} score={score:.3f} "
                    f"long={stats['long']:.0%} short={stats['short']:.0%} flat={stats['flat']:.0%}"
                )
                return score, label

            ac_max_steps = ac_cfg.get("max_steps", 200_000)
            if ac_trainer.global_step >= ac_max_steps:
                print(f"\n[{_ts()}] [Step 4] AC — already complete (step={ac_trainer.global_step})")
            else:
                bc_val_sharpe = -float("inf")
                if has_ac:
                    print(f"\n[{_ts()}] [Step 4] AC — resuming from step {ac_trainer.global_step}/{ac_max_steps}")
                else:
                    print(f"\n[{_ts()}] [Step 4] Imagination AC Fine-tuning...")

                    # BC-only val score (checkpoint selection のベースライン)
                    bc_val_sharpe, bc_val_label = _val_eval()
                    print(f"[AC] BC-only val score: {bc_val_label}")
                    if z_val_fixed is not None:
                        _bc_pos = actor.predict_positions(
                            z_val_fixed, h_val_fixed, regime_np=val_regime_probs, device=device
                        )
                        _bc_T = min(len(val_returns_arr), len(_bc_pos))
                        _bc_m = Backtest(
                            val_returns_arr[:_bc_T], _bc_pos[:_bc_T],
                            spread_bps=costs_cfg.get("spread_bps", 5.0),
                            fee_rate=costs_cfg.get("fee_rate", 0.0004),
                            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
                            interval=cfg.get("data", {}).get("interval", "15m"),
                            benchmark_positions=_benchmark_positions(_bc_T, cfg),
                        ).run()
                        _bc_attr = pnl_attribution(
                            val_returns_arr[:_bc_T], _bc_pos[:_bc_T],
                            spread_bps=costs_cfg.get("spread_bps", 5.0),
                            fee_rate=costs_cfg.get("fee_rate", 0.0004),
                            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
                        )
                        _bc_s = _action_stats(_bc_pos[:_bc_T])
                        print(f"  BC val dist: {_fmt_action_stats(_bc_s)}")
                        print(f"  BC val: TotalRet={_bc_m.total_return:.3f}  "
                              f"AlphaExcess={100.0 * (_bc_m.alpha_excess or 0.0):+.2f}pt  "
                              f"long={_bc_attr['long_gross']:+.4f}  "
                              f"short={_bc_attr['short_gross']:+.4f}  "
                              f"cost={_bc_attr['cost_total']:.4f}")
                        _oracle_val_pos = val_oracle_positions[:_bc_T]
                        _oracle_val_s = _action_stats(_oracle_val_pos)
                        print(f"  Oracle val dist: {_fmt_action_stats(_oracle_val_s)}")
                        _ac_alerts("BC-val", _bc_s)

                    critic_pretrain_steps = ac_cfg.get("critic_pretrain_steps", 0)
                    if critic_pretrain_steps > 0:
                        ac_trainer.pretrain_critic(
                            encoded_sequences=encoded_list,
                            n_steps=critic_pretrain_steps,
                            batch_size=ac_cfg.get("batch_size", 32),
                        )

                interval = cfg.get("data", {}).get("interval", "15m")
                bars_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}.get(interval, 96)
                online_wm_window = ac_cfg.get("online_wm_window_days", 30) * bars_per_day
                online_wm_steps_val = ac_cfg.get("online_wm_steps", 0)

                def _online_wm_cb(step: int) -> None:
                    if online_wm_steps_val <= 0:
                        return
                    T_train = len(wfo_dataset.train_features)
                    window_start = max(0, T_train - online_wm_window)
                    recent_feat = wfo_dataset.train_features[window_start:]
                    recent_returns = wfo_dataset.train_returns[window_start:]
                    recent_regime = (
                        train_regime_probs[window_start:T_train]
                        if train_regime_probs is not None else None
                    )
                    enc_recent = wm_trainer.encode_sequence(recent_feat, seq_len=seq_len)
                    recent_pos = actor.predict_positions(
                        enc_recent["z"], enc_recent["h"],
                        regime_np=recent_regime, device=device,
                    )
                    recent_ds = SequenceDataset(
                        recent_feat, seq_len=seq_len,
                        actions=recent_pos[:len(recent_feat)],
                        returns=recent_returns[:len(recent_feat)],
                    )
                    if len(recent_ds) < 2:
                        return
                    wm_trainer.ensemble.train()
                    wm_trainer.train_on_dataset(recent_ds, max_steps=online_wm_steps_val, checkpoint_path=None)
                    wm_trainer.ensemble.eval()

                online_wm_callback = _online_wm_cb if online_wm_steps_val > 0 else None

                ac_trainer.train(
                    encoded_sequences=encoded_list,
                    batch_size=ac_cfg.get("batch_size", 32),
                    checkpoint_path=ac_path,
                    val_eval_fn=_val_eval,
                    val_baseline_sharpe=bc_val_sharpe,
                    online_wm_callback=online_wm_callback,
                )
                ac_trainer.save(ac_path)
        else:
            print(f"\n[{_ts()}] [Step 4] AC — skipped (BC actor only for test)")

    if stop_after == "ac":
        print(f"\n[{_ts()}] [Stop] Requested stop after AC")
        return {"fold": fold_idx, "completed_stage": "ac"}

    # --------- Step 5: Test バックテスト ---------
    print(f"\n[{_ts()}] [Step 5] Test Backtest...")
    test_features = wfo_dataset.test_dataset().features.numpy()
    test_returns = wfo_dataset.test_returns

    # Single-pass encoding: no-action context（flat）で encode して actor を適用する。
    # 2-pass encoding はノイジーな BC 予測を context に混入させ switching を増幅するため、
    # val と同じ no-action single-pass に統一して安定性を確保する。
    enc_test = wm_trainer.encode_sequence(test_features, seq_len=seq_len)
    positions = actor.predict_positions(
        enc_test["z"], enc_test["h"], regime_np=test_regime_probs, device=device
    )

    T_min = min(len(test_returns), len(positions))
    bt = Backtest(
        test_returns[:T_min],
        positions[:T_min],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
        interval=cfg.get("data", {}).get("interval", "15m"),
        benchmark_positions=_benchmark_positions(T_min, cfg),
    )
    metrics = bt.run()

    _test_attr = pnl_attribution(
        test_returns[:T_min], positions[:T_min],
        spread_bps=costs_cfg.get("spread_bps", 5.0),
        fee_rate=costs_cfg.get("fee_rate", 0.0004),
        slippage_bps=costs_cfg.get("slippage_bps", 2.0),
    )
    _test_s = _action_stats(positions[:T_min])
    print(f"  Sharpe:   {metrics.sharpe:.3f}")
    print(f"  Sortino:  {metrics.sortino:.3f}")
    print(f"  MaxDD:    {metrics.max_drawdown:.3f}")
    print(f"  Calmar:   {metrics.calmar:.3f}")
    print(f"  TotalRet: {metrics.total_return:.4f}")
    if metrics.alpha_excess is not None:
        print(f"  AlphaEx:  {100.0 * metrics.alpha_excess:+.2f} pt/yr")
        print(f"  SharpeΔ:  {(metrics.sharpe_delta or 0.0):+.3f}")
    print(f"  PnL attr: long={_test_attr['long_gross']:+.4f}  "
          f"short={_test_attr['short_gross']:+.4f}  "
          f"cost={_test_attr['cost_total']:.4f}  "
          f"net={_test_attr['net_total']:+.4f}")
    print(f"  Test dist: {_fmt_action_stats(_test_s)}")
    _ac_alerts("test", _test_s)

    return {
        "fold": fold_idx,
        "metrics": metrics,
        "positions": positions[:T_min],
        "test_returns": test_returns[:T_min],
        "completed_stage": "test",
    }


def main():
    parser = argparse.ArgumentParser(description="UniDream Training Script")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--symbol", default=None, help="Binance symbol (overrides config)")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="保存済みチェックポイントから再開する")
    parser.add_argument(
        "--start-from",
        default="wm",
        choices=_PIPELINE_STAGES,
        help="Start pipeline from this stage",
    )
    parser.add_argument(
        "--stop-after",
        default="test",
        choices=_PIPELINE_STAGES,
        help="Stop pipeline after this stage",
    )
    parser.add_argument(
        "--cost-profile",
        default=None,
        help="Named cost profile from config cost_profiles (for example: base, stress)",
    )
    args = parser.parse_args()

    if _stage_idx(args.start_from) > _stage_idx(args.stop_after):
        parser.error("--start-from must be earlier than or equal to --stop-after")

    cfg = load_config(args.config)
    cfg, active_cost_profile = resolve_costs(cfg, args.cost_profile)
    set_seed(args.seed)

    symbol = args.symbol or cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")

    print(f"UniDream Training | {symbol} {interval} | {args.start} → {args.end}")
    print(f"Device: {args.device} | Seed: {args.seed} | Resume: {args.resume}")
    print(f"Stages: {args.start_from} -> {args.stop_after}")
    costs_cfg = cfg.get("costs", {})
    total_cost_bps = (
        costs_cfg.get("spread_bps", 0.0) / 2
        + costs_cfg.get("fee_rate", 0.0) * 10000
        + costs_cfg.get("slippage_bps", 0.0)
    )
    print(
        "Costs: "
        f"profile={active_cost_profile} | "
        f"spread={costs_cfg.get('spread_bps', 0.0):.2f}bps "
        f"fee={costs_cfg.get('fee_rate', 0.0) * 10000:.2f}bps "
        f"slip={costs_cfg.get('slippage_bps', 0.0):.2f}bps "
        f"=> one-way Δpos=1 cost={total_cost_bps:.2f}bps"
    )

    # --------- データ取得・特徴量計算（キャッシュ対応）---------
    cache_dir = os.path.join(args.checkpoint_dir, "data_cache")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_cache = os.path.join(cache_dir, f"{cache_tag}_features.parquet")
    returns_cache = os.path.join(cache_dir, f"{cache_tag}_returns.parquet")

    if _cache_is_fresh(features_cache) and _cache_is_fresh(returns_cache):
        print("\n[Data] Loading cached features...")
        features_df = pd.read_parquet(features_cache)
        raw_returns = pd.read_parquet(returns_cache).squeeze()
        print(f"  Cached: {features_df.shape} | obs_dim={features_df.shape[1]}")
    else:
        print("\n[Data] Fetching OHLCV...")
        df = fetch_binance_ohlcv(symbol, interval, args.start, args.end)
        print(f"  Raw data: {len(df)} bars ({df.index[0]} → {df.index[-1]})")

        # Futures 追加データ（funding rate, OI）
        funding_df = None
        oi_df = None
        try:
            print("[Data] Fetching funding rate...")
            funding_df = fetch_funding_rate(symbol, args.start, args.end)
            print(f"  Funding rate: {len(funding_df)} records")
        except Exception as e:
            print(f"  Funding rate skipped: {e}")
        try:
            print("[Data] Fetching open interest...")
            oi_df = fetch_open_interest_hist(symbol, interval, args.start, args.end)
            print(f"  Open interest: {len(oi_df)} records")
        except Exception as e:
            print(f"  Open interest skipped: {e}")

        print("[Data] Computing features...")
        features_df = compute_features(
            df,
            zscore_window_days=cfg.get("normalization", {}).get("zscore_window_days", 60),
            interval=interval,
            funding_df=funding_df,
            oi_df=oi_df,
        )
        raw_returns = get_raw_returns(df)
        common_idx = features_df.index.intersection(raw_returns.index)
        features_df = features_df.loc[common_idx]
        raw_returns = raw_returns.loc[common_idx]
        print(f"  Features: {features_df.shape} | obs_dim={features_df.shape[1]}")

        # キャッシュ保存
        os.makedirs(cache_dir, exist_ok=True)
        features_df.to_parquet(features_cache)
        raw_returns.to_frame().to_parquet(returns_cache)
        print(f"  Cached to {cache_dir}")

    # --------- WFO 分割 ---------
    print("[Data] WFO splits...")
    data_cfg = cfg.get("data", {})
    splits = get_wfo_splits(
        features_df,
        train_years=data_cfg.get("train_years", 2),
        val_months=data_cfg.get("val_months", 3),
        test_months=data_cfg.get("test_months", 3),
    )
    print(f"  {len(splits)} folds")

    if len(splits) == 0:
        print("ERROR: WFO splits が空です。データ期間が短すぎます。")
        return

    # --------- 各 Fold の学習・評価 ---------
    fold_results = {}
    for split in splits:
        wfo_ds = WFODataset(
            features_df,
            raw_returns,
            split,
            seq_len=data_cfg.get("seq_len", 64),
        )
        result = run_fold(
            fold_idx=split.fold_idx,
            wfo_dataset=wfo_ds,
            cfg=cfg,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            start_from=args.start_from,
            stop_after=args.stop_after,
        )
        fold_results[split.fold_idx] = result

    if args.stop_after != "test":
        print("\n" + "="*60)
        print("Stage Summary")
        print("="*60)
        for fold_idx, r in fold_results.items():
            print(f"  Fold {fold_idx}: completed_stage={r.get('completed_stage', args.stop_after)}")
        return

    # --------- PBO / Deflated Sharpe ---------
    # 注意: PBO は「fold を戦略候補扱いした IS/OOS 分割の簡略版」。
    #       標準 CSCV-PBO（Bailey & Lopez de Prado 2014）ではない。
    #       複数モデル構成を比較する際は CSCV 版に差し替えること。
    # 注意: DSR は n_trials=1（ハイパーパラメータ探索なし）のため、
    #       多重比較補正なしの「best fold Sharpe の t 統計量」として機能する。
    print("\n[Eval] Overfitting Diagnostics (simplified)...")
    pnl_list = [r["metrics"].pnl_series for r in fold_results.values()]
    eval_cfg = cfg.get("eval", {})
    pbo = compute_pbo(
        pnl_list,
        n_combinations=eval_cfg.get("pbo_n_trials"),
    )
    print(f"  PBO (simplified): {pbo:.4f} (< 0.5 が望ましい; fold IS/OOS split 版)")

    all_sharpes = [r["metrics"].sharpe for r in fold_results.values()]
    best_sharpe = max(all_sharpes)
    # n_trials=1: ハイパーパラメータ探索を行っていないため多重比較補正なし
    n_trials = 1
    T_avg = int(np.mean([len(r["metrics"].pnl_series) for r in fold_results.values()]))
    dsr = deflated_sharpe(best_sharpe, n_trials=n_trials, T=T_avg)
    dsr_str = f"{dsr:.4f}" if np.isfinite(dsr) else f"N/A ({dsr}, fold 数不足)"
    print(f"  Sharpe t-stat (DSR, n_trials=1): {dsr_str} (> 0 が望ましい)")

    # --------- レジーム別メトリクス ---------
    print("\n[Eval] Regime Analysis...")
    all_test_returns = np.concatenate([r["test_returns"] for r in fold_results.values()])
    all_positions = np.concatenate([r["positions"] for r in fold_results.values()])

    try:
        detector = RegimeDetector(n_states=cfg.get("eval", {}).get("hmm_n_states", 3))
        regimes = detector.fit_predict(all_test_returns)
        regime_results = regime_metrics(
            all_test_returns,
            all_positions,
            regimes,
            n_states=detector.n_states,
            interval=interval,
            spread_bps=cfg.get("costs", {}).get("spread_bps", 5.0),
            fee_rate=cfg.get("costs", {}).get("fee_rate", 0.0004),
            slippage_bps=cfg.get("costs", {}).get("slippage_bps", 2.0),
        )
        print_regime_report(regime_results, detector)
    except Exception as e:
        print(f"  Regime analysis skipped: {e}")

    # --------- サマリー ---------
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for fold_idx, r in fold_results.items():
        m = r["metrics"]
        print(f"  Fold {fold_idx}: Sharpe={m.sharpe:.3f}, MaxDD={m.max_drawdown:.3f}, "
              f"Calmar={m.calmar:.3f}, TotalRet={m.total_return:.4f}")
    print(f"  Mean Sharpe: {np.mean(all_sharpes):.3f}")
    dsr_summary = f"{dsr:.4f}" if np.isfinite(dsr) else "N/A"
    print(f"  PBO (simplified): {pbo:.4f} | Sharpe t-stat: {dsr_summary}")


if __name__ == "__main__":
    main()
