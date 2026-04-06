from __future__ import annotations

from unidream.actor_critic.critic import Critic
from unidream.actor_critic.imagination_ac import ImagACTrainer


def build_ac_trainer(
    *,
    actor,
    ensemble,
    cfg: dict,
    ac_cfg: dict,
    wm_cfg: dict,
    device: str,
    has_ac: bool,
    ac_path: str,
) -> ImagACTrainer:
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
    return ac_trainer


def run_ac_stage(
    *,
    actor,
    ensemble,
    cfg: dict,
    ac_cfg: dict,
    wm_cfg: dict,
    costs_cfg: dict,
    device: str,
    has_ac: bool,
    ac_path: str,
    z_train,
    h_train,
    oracle_positions,
    train_regime_probs,
    wfo_dataset,
    wm_trainer,
    seq_len: int,
    val_regime_probs,
    val_oracle_positions,
    start_idx: int,
    ac_stage_idx: int,
    ac_max_steps_cfg: int,
    log_ts,
    backtest_cls,
    pnl_attribution_fn,
    action_stats_fn,
    format_action_stats_fn,
    ac_alerts_fn,
    benchmark_positions_fn,
    benchmark_position: float,
    policy_score_fn,
    sequence_dataset_cls,
):
    ac_requested = ((start_idx <= ac_stage_idx) and ac_max_steps_cfg > 0) or has_ac
    if not ac_requested:
        print(f"\n[{log_ts()}] [Step 4] AC - skipped (BC actor only for test)")
        return None

    ac_trainer = build_ac_trainer(
        actor=actor,
        ensemble=ensemble,
        cfg=cfg,
        ac_cfg=ac_cfg,
        wm_cfg=wm_cfg,
        device=device,
        has_ac=has_ac,
        ac_path=ac_path,
    )

    if not (start_idx <= ac_stage_idx or has_ac):
        print(f"\n[{log_ts()}] [Step 4] AC - skipped (BC actor only for test)")
        return ac_trainer

    t_enc = min(len(z_train), len(oracle_positions))
    ac_trainer.set_oracle_data(
        z=z_train[:t_enc],
        h=h_train[:t_enc],
        oracle_positions=oracle_positions[:t_enc],
        regime_probs=train_regime_probs[:t_enc] if train_regime_probs is not None else None,
    )
    encoded_list = [{
        "z": z_train,
        "h": h_train,
        "regime": train_regime_probs if train_regime_probs is not None else None,
    }]

    val_features_arr = wfo_dataset.val_features
    val_returns_arr = wfo_dataset.val_returns
    if len(val_features_arr) > 0:
        enc_val_fixed = wm_trainer.encode_sequence(val_features_arr, seq_len=seq_len)
        z_val_fixed = enc_val_fixed["z"]
        h_val_fixed = enc_val_fixed["h"]
    else:
        z_val_fixed = h_val_fixed = None

    def _val_eval():
        if z_val_fixed is None:
            return -float("inf"), "raw=-inf score=-inf"
        pos = actor.predict_positions(
            z_val_fixed, h_val_fixed, regime_np=val_regime_probs, device=device
        )
        t_min = min(len(val_returns_arr), len(pos))
        metrics = backtest_cls(
            val_returns_arr[:t_min],
            pos[:t_min],
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            fee_rate=costs_cfg.get("fee_rate", 0.0004),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            interval=cfg.get("data", {}).get("interval", "15m"),
            benchmark_positions=benchmark_positions_fn(t_min),
        ).run()
        stats = action_stats_fn(pos[:t_min], benchmark_position=benchmark_position)
        return policy_score_fn(metrics, stats, benchmark_position=benchmark_position)

    ac_max_steps = ac_cfg.get("max_steps", 200_000)
    if ac_trainer.global_step >= ac_max_steps:
        print(f"\n[{log_ts()}] [Step 4] AC - already complete (step={ac_trainer.global_step})")
        return ac_trainer

    bc_val_sharpe = -float("inf")
    if has_ac:
        print(f"\n[{log_ts()}] [Step 4] AC - resuming from step {ac_trainer.global_step}/{ac_max_steps}")
    else:
        print(f"\n[{log_ts()}] [Step 4] Imagination AC Fine-tuning...")
        bc_val_sharpe, bc_val_label = _val_eval()
        print(f"[AC] BC-only val score: {bc_val_label}")
        if z_val_fixed is not None:
            bc_pos = actor.predict_positions(
                z_val_fixed, h_val_fixed, regime_np=val_regime_probs, device=device
            )
            bc_t = min(len(val_returns_arr), len(bc_pos))
            bc_metrics = backtest_cls(
                val_returns_arr[:bc_t],
                bc_pos[:bc_t],
                spread_bps=costs_cfg.get("spread_bps", 5.0),
                fee_rate=costs_cfg.get("fee_rate", 0.0004),
                slippage_bps=costs_cfg.get("slippage_bps", 2.0),
                interval=cfg.get("data", {}).get("interval", "15m"),
                benchmark_positions=benchmark_positions_fn(bc_t),
            ).run()
            bc_attr = pnl_attribution_fn(
                val_returns_arr[:bc_t],
                bc_pos[:bc_t],
                spread_bps=costs_cfg.get("spread_bps", 5.0),
                fee_rate=costs_cfg.get("fee_rate", 0.0004),
                slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            )
            bc_stats = action_stats_fn(bc_pos[:bc_t], benchmark_position=benchmark_position)
            print(f"  BC val dist: {format_action_stats_fn(bc_stats)}")
            print(
                f"  BC val: TotalRet={bc_metrics.total_return:.3f}  "
                f"AlphaExcess={100.0 * (bc_metrics.alpha_excess or 0.0):+.2f}pt  "
                f"long={bc_attr['long_gross']:+.4f}  "
                f"short={bc_attr['short_gross']:+.4f}  "
                f"cost={bc_attr['cost_total']:.4f}"
            )
            oracle_val_pos = val_oracle_positions[:bc_t]
            oracle_val_stats = action_stats_fn(oracle_val_pos, benchmark_position=benchmark_position)
            print(f"  Oracle val dist: {format_action_stats_fn(oracle_val_stats)}")
            ac_alerts_fn("BC-val", bc_stats)

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
        t_train = len(wfo_dataset.train_features)
        window_start = max(0, t_train - online_wm_window)
        recent_feat = wfo_dataset.train_features[window_start:]
        recent_returns = wfo_dataset.train_returns[window_start:]
        recent_regime = train_regime_probs[window_start:t_train] if train_regime_probs is not None else None
        enc_recent = wm_trainer.encode_sequence(recent_feat, seq_len=seq_len)
        recent_pos = actor.predict_positions(
            enc_recent["z"], enc_recent["h"], regime_np=recent_regime, device=device
        )
        recent_ds = sequence_dataset_cls(
            recent_feat,
            seq_len=seq_len,
            actions=recent_pos[: len(recent_feat)],
            returns=recent_returns[: len(recent_feat)],
        )
        if len(recent_ds) < 2:
            return
        wm_trainer.ensemble.train()
        wm_trainer.train_on_dataset(recent_ds, max_steps=online_wm_steps_val, checkpoint_path=None)
        wm_trainer.ensemble.eval()

    ac_trainer.train(
        encoded_sequences=encoded_list,
        batch_size=ac_cfg.get("batch_size", 32),
        checkpoint_path=ac_path,
        val_eval_fn=_val_eval,
        val_baseline_sharpe=bc_val_sharpe,
        online_wm_callback=_online_wm_cb if online_wm_steps_val > 0 else None,
    )
    ac_trainer.save(ac_path)
    return ac_trainer
