from __future__ import annotations

import os
import shutil

import torch

from unidream.actor_critic.critic import Critic
from unidream.actor_critic.imagination_ac import ImagACTrainer
from unidream.eval.action_execution import (
    contract_pnl_attribution,
    configured_action_execution_contract,
    replay_contract_absolute_path,
    run_contract_backtest,
)
from unidream.experiments.overlay_teacher import (
    apply_benchmark_overlay_teacher,
    benchmark_overlay_teacher_enabled,
    describe_benchmark_overlay_teacher,
)
from unidream.experiments.policy_fire import evaluate_fire_metrics, format_fire_metrics


def _run_stage_backtest(
    *,
    backtest_cls,
    returns,
    positions,
    benchmark_positions,
    contract,
    **kwargs,
):
    if contract is not None:
        return run_contract_backtest(
            backtest_cls,
            returns,
            positions,
            benchmark_positions=benchmark_positions,
            contract=contract,
            **kwargs,
        ).run()
    return backtest_cls(
        returns,
        positions,
        benchmark_positions=benchmark_positions,
        **kwargs,
    ).run()


def _optimizer_lr(optimizer, default: float) -> float:
    if not optimizer.param_groups:
        return default
    return float(optimizer.param_groups[0].get("lr", default))


def _rebuild_actor_optimizer(ac_trainer: ImagACTrainer, lr: float) -> None:
    ac_trainer._apply_actor_trainable_mask()
    actor_params = [p for p in ac_trainer.actor.parameters() if p.requires_grad]
    if not actor_params:
        actor_params = list(ac_trainer.actor.parameters())
    ac_trainer.actor_optimizer = torch.optim.Adam(actor_params, lr=lr)


def _cleanup_auxiliary_checkpoints(
    ac_path: str | None,
    *,
    step_checkpoint_prefix: str = "ac_step",
) -> None:
    """Remove internal selection/step artifacts after final ``ac.pt`` is saved."""
    if not ac_path:
        return
    base, extension = os.path.splitext(ac_path)
    candidates = [f"{base}_best{extension}", f"{base}_fire_best{extension}"]
    directory = os.path.dirname(ac_path) or "."
    prefix = os.path.basename(base)
    step_prefix = os.path.basename(str(step_checkpoint_prefix))
    for filename in os.listdir(directory):
        step_prefixes = {f"{prefix}_step"}
        if step_prefix:
            step_prefixes.add(step_prefix)
        is_step_checkpoint = any(
            filename.startswith(candidate_prefix)
            and filename.endswith(extension)
            and filename[len(candidate_prefix) : -len(extension)].isdigit()
            for candidate_prefix in step_prefixes
        )
        if is_step_checkpoint:
            candidates.append(os.path.join(directory, filename))
        if filename.startswith(f"{prefix}_stage_") and filename.endswith(extension):
            candidates.append(os.path.join(directory, filename))
    for candidate in candidates:
        if os.path.exists(candidate):
            os.remove(candidate)


def _apply_curriculum_stage(
    ac_trainer: ImagACTrainer,
    actor,
    stage_cfg: dict,
    base_ac_cfg: dict,
) -> None:
    actor_overrides = dict(stage_cfg.get("actor") or {})
    trainer_overrides = dict(stage_cfg.get("trainer") or stage_cfg.get("ac") or {})
    for key in (
        "actor_lr",
        "critic_only",
        "trainable_actor_prefixes",
        "td3bc_alpha",
        "alpha_init",
        "alpha_final",
        "alpha_decay_steps",
        "prior_kl_coef",
        "prior_trade_coef",
        "prior_band_coef",
        "prior_flow_coef",
        "turnover_coef",
        "flow_change_coef",
        "entropy_scale",
    ):
        if key in stage_cfg:
            trainer_overrides[key] = stage_cfg[key]

    for key, value in actor_overrides.items():
        if not hasattr(actor, key):
            raise KeyError(f"Unknown AC curriculum actor override: {key}")
        if key not in ac_trainer.actor_runtime_defaults:
            ac_trainer.actor_runtime_defaults[key] = getattr(actor, key)
        setattr(actor, key, value)
        ac_trainer.actor_runtime_overrides[key] = value
        if key in {"abs_min_position", "abs_max_position"}:
            setattr(ac_trainer, key, float(value))

    lr = float(trainer_overrides.pop(
        "actor_lr",
        _optimizer_lr(ac_trainer.actor_optimizer, float(base_ac_cfg.get("actor_lr", 3e-5))),
    ))
    rebuild_actor_optimizer = False

    if "critic_only" in trainer_overrides:
        ac_trainer.critic_only = bool(trainer_overrides.pop("critic_only"))
        rebuild_actor_optimizer = True
    if "trainable_actor_prefixes" in trainer_overrides:
        raw_prefixes = trainer_overrides.pop("trainable_actor_prefixes") or []
        ac_trainer.trainable_actor_prefixes = tuple(str(prefix) for prefix in raw_prefixes)
        rebuild_actor_optimizer = True

    for key, value in trainer_overrides.items():
        if not hasattr(ac_trainer, key):
            raise KeyError(f"Unknown AC curriculum trainer override: {key}")
        setattr(ac_trainer, key, value)

    if bool(stage_cfg.get("reset_alpha_schedule", True)):
        ac_trainer.begin_alpha_stage()

    if rebuild_actor_optimizer or abs(lr - _optimizer_lr(ac_trainer.actor_optimizer, lr)) > 0.0:
        _rebuild_actor_optimizer(ac_trainer, lr)


def build_ac_trainer(
    *,
    actor,
    ensemble,
    cfg: dict,
    ac_cfg: dict,
    wm_cfg: dict,
    device: str,
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
    ac_path: str,
    z_train,
    h_train,
    oracle_positions,
    train_regime_probs,
    train_advantage_values,
    wfo_dataset,
    wm_trainer,
    seq_len: int,
    val_regime_probs,
    val_advantage_values,
    val_oracle_positions,
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
    checkpoint_metadata: dict | None = None,
):
    action_contract = configured_action_execution_contract(cfg)
    if ac_max_steps_cfg <= 0:
        print(f"\n[{log_ts()}] [Step 4] AC - skipped (BC actor only for test)")
        return None

    ac_trainer = build_ac_trainer(
        actor=actor,
        ensemble=ensemble,
        cfg=cfg,
        ac_cfg=ac_cfg,
        wm_cfg=wm_cfg,
        device=device,
    )
    ac_trainer.checkpoint_metadata = dict(checkpoint_metadata or {})

    t_enc = min(len(z_train), len(oracle_positions))
    ac_oracle_positions = oracle_positions[:t_enc]
    if benchmark_overlay_teacher_enabled(cfg.get("bc", {}), ac_cfg):
        ac_oracle_positions = apply_benchmark_overlay_teacher(
            ac_oracle_positions,
            bc_cfg=cfg.get("bc", {}),
            ac_cfg=ac_cfg,
            reward_cfg=cfg.get("reward", {}),
            advantage_values=train_advantage_values[:t_enc] if train_advantage_values is not None else None,
            returns=wfo_dataset.train_returns[:t_enc] if wfo_dataset is not None else None,
        )
        print(
            f"[AC] {describe_benchmark_overlay_teacher(oracle_positions[:t_enc], ac_oracle_positions, reward_cfg=cfg.get('reward', {}))}"
        )
    ac_trainer.set_oracle_data(
        z=z_train[:t_enc],
        h=h_train[:t_enc],
        oracle_positions=ac_oracle_positions,
        regime_probs=train_regime_probs[:t_enc] if train_regime_probs is not None else None,
        advantage_values=train_advantage_values[:t_enc] if train_advantage_values is not None else None,
    )
    encoded_list = [{
        "z": z_train,
        "h": h_train,
        "regime": train_regime_probs if train_regime_probs is not None else None,
        "advantage": train_advantage_values if train_advantage_values is not None else None,
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
            z_val_fixed,
            h_val_fixed,
            regime_np=val_regime_probs,
            advantage_np=val_advantage_values,
            device=device,
        )
        t_min = min(len(val_returns_arr), len(pos))
        metrics = _run_stage_backtest(
            backtest_cls=backtest_cls,
            returns=val_returns_arr[:t_min],
            positions=pos[:t_min],
            benchmark_positions=benchmark_positions_fn(t_min),
            contract=action_contract,
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            fee_rate=costs_cfg.get("fee_rate", 0.0004),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            interval=cfg.get("data", {}).get("interval", "15m"),
        )
        if action_contract is not None:
            stats = action_stats_fn(
                replay_contract_absolute_path(
                    val_returns_arr[:t_min], pos[:t_min], action_contract
                ).scored_positions,
                benchmark_position=benchmark_position,
            )
        else:
            stats = action_stats_fn(pos[:t_min], benchmark_position=benchmark_position)
        return policy_score_fn(metrics, stats, benchmark_position=benchmark_position)

    fire_selector_cfg = dict(ac_cfg.get("fire_checkpoint_selector") or {})
    fire_selector_enabled = bool(fire_selector_cfg.get("enabled", False))
    fire_selector_bc_baseline = {"alpha_excess_pt": None, "sharpe_delta": None}

    def _fire_checkpoint_eval():
        if z_val_fixed is None:
            return {"score": -float("inf"), "label": "unavailable", "accepted": False}
        if action_contract is not None:
            # policy_fire is a historical every-bar evaluator and cannot prove
            # the P0-C delayed/committed trajectory.  Do not let it silently
            # select an AC checkpoint under legacy defaults.
            return {
                "score": -float("inf"),
                "label": "disabled: action execution contract requires shared evaluator",
                "accepted": False,
            }
        selector_cfg = dict(fire_selector_cfg)
        if bool(selector_cfg.get("use_bc_val_baseline", False)):
            alpha_base = fire_selector_bc_baseline["alpha_excess_pt"]
            sharpe_base = fire_selector_bc_baseline["sharpe_delta"]
            if alpha_base is not None:
                selector_cfg["alpha_floor_pt"] = max(
                    float(selector_cfg.get("alpha_floor_pt", -float("inf"))),
                    float(alpha_base) + float(selector_cfg.get("min_alpha_improvement_pt", 0.0)),
                )
            if sharpe_base is not None:
                selector_cfg["sharpe_floor"] = max(
                    float(selector_cfg.get("sharpe_floor", -float("inf"))),
                    float(sharpe_base) + float(selector_cfg.get("min_sharpe_improvement", 0.0)),
                )
        fire_metrics = evaluate_fire_metrics(
            actor=actor,
            z=z_val_fixed,
            h=h_val_fixed,
            returns=val_returns_arr,
            regime_np=val_regime_probs,
            advantage_np=val_advantage_values,
            device=device,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_positions_fn=benchmark_positions_fn,
            benchmark_position=benchmark_position,
            backtest_cls=backtest_cls,
            action_stats_fn=action_stats_fn,
            selector_cfg=selector_cfg,
        )
        return {
            "score": fire_metrics.score,
            "accepted": fire_metrics.accepted,
            "label": format_fire_metrics(fire_metrics),
        }

    ac_max_steps = ac_max_steps_cfg
    bc_val_sharpe = -float("inf")
    print(f"\n[{log_ts()}] [Step 4] Imagination AC Fine-tuning...")
    bc_val_sharpe, bc_val_label = _val_eval()
    print(f"[AC] BC-only val score: {bc_val_label}")
    if z_val_fixed is not None:
        bc_pos = actor.predict_positions(
            z_val_fixed,
            h_val_fixed,
            regime_np=val_regime_probs,
            advantage_np=val_advantage_values,
            device=device,
        )
        bc_t = min(len(val_returns_arr), len(bc_pos))
        bc_metrics = _run_stage_backtest(
            backtest_cls=backtest_cls,
            returns=val_returns_arr[:bc_t],
            positions=bc_pos[:bc_t],
            benchmark_positions=benchmark_positions_fn(bc_t),
            contract=action_contract,
            spread_bps=costs_cfg.get("spread_bps", 5.0),
            fee_rate=costs_cfg.get("fee_rate", 0.0004),
            slippage_bps=costs_cfg.get("slippage_bps", 2.0),
            interval=cfg.get("data", {}).get("interval", "15m"),
        )
        fire_selector_bc_baseline["alpha_excess_pt"] = 100.0 * float(bc_metrics.alpha_excess or 0.0)
        fire_selector_bc_baseline["sharpe_delta"] = float(bc_metrics.sharpe_delta or 0.0)
        if action_contract is not None:
            bc_attr = contract_pnl_attribution(
                val_returns_arr[:bc_t], bc_pos[:bc_t], action_contract
            )
            bc_trajectory = replay_contract_absolute_path(
                val_returns_arr[:bc_t], bc_pos[:bc_t], action_contract
            )
            bc_stats = action_stats_fn(
                bc_trajectory.scored_positions,
                benchmark_position=benchmark_position,
            )
        else:
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
    if ac_cfg.get("critic_only", False) and not ac_cfg.get("curriculum"):
        if ac_path:
            ac_trainer.save(ac_path)
            _cleanup_auxiliary_checkpoints(
                ac_path,
                step_checkpoint_prefix=ac_trainer.step_checkpoint_prefix,
            )
        print(f"[{log_ts()}] [Step 4] AC critic-only requested; actor update skipped")
        return ac_trainer

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
            enc_recent["z"],
            enc_recent["h"],
            regime_np=recent_regime,
            advantage_np=(
                None if train_advantage_values is None else train_advantage_values[window_start:t_train]
            ),
            device=device,
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

    curriculum = ac_cfg.get("curriculum") or []
    if curriculum:
        print(f"[{log_ts()}] [Step 4] AC curriculum enabled ({len(curriculum)} stages)")
        select_curriculum_stage = bool(ac_cfg.get("select_curriculum_stage", False))
        stage_candidates: list[dict] = []
        if select_curriculum_stage:
            bc_stage_path = ac_path.replace(".pt", "_stage_00_bc.pt")
            ac_trainer.save(bc_stage_path)
            stage_candidates.append({
                "name": "bc",
                "score": float(bc_val_sharpe),
                "label": str(bc_val_label),
                "path": bc_stage_path,
            })
            print(f"[{log_ts()}] [AC curriculum] saved candidate bc: {bc_val_label}")
        for idx, stage_cfg in enumerate(curriculum, start=1):
            until_step = int(stage_cfg.get("until_step", stage_cfg.get("max_steps", 0)))
            if until_step <= 0:
                raise ValueError(f"AC curriculum stage {idx} requires until_step/max_steps > 0")
            if ac_trainer.global_step >= until_step:
                print(
                    f"[{log_ts()}] [AC curriculum] "
                    f"{stage_cfg.get('name', f'stage{idx}')} already complete "
                    f"(step={ac_trainer.global_step}/{until_step})"
                )
                continue

            _apply_curriculum_stage(ac_trainer, actor, stage_cfg, ac_cfg)
            stage_name = stage_cfg.get("name", f"stage{idx}")
            print(
                f"[{log_ts()}] [AC curriculum] {stage_name}: "
                f"step {ac_trainer.global_step}->{until_step}"
            )
            batch_size = int(stage_cfg.get("batch_size", ac_cfg.get("batch_size", 32)))
            if ac_trainer.critic_only:
                ac_trainer.pretrain_critic(
                    encoded_sequences=encoded_list,
                    n_steps=until_step - ac_trainer.global_step,
                    batch_size=batch_size,
                )
                ac_trainer.global_step = until_step
                if ac_path:
                    ac_trainer.save(ac_path)
            else:
                ac_trainer.train(
                    encoded_sequences=encoded_list,
                    max_steps=until_step,
                    batch_size=batch_size,
                    checkpoint_path=ac_path,
                    val_eval_fn=_val_eval,
                    checkpoint_eval_fn=_fire_checkpoint_eval if fire_selector_enabled else None,
                    val_baseline_sharpe=bc_val_sharpe,
                    online_wm_callback=_online_wm_cb if online_wm_steps_val > 0 else None,
                )
            if ac_path:
                ac_trainer.save(ac_path)
            if select_curriculum_stage:
                safe_stage_name = "".join(
                    char if char.isalnum() or char in {"-", "_"} else "_"
                    for char in str(stage_name)
                )
                train_best = ac_trainer.last_train_best_candidate
                if train_best is not None:
                    best_stage_path = ac_path.replace(
                        ".pt",
                        f"_stage_{idx:02d}_{safe_stage_name}_best.pt",
                    )
                    shutil.copy2(str(train_best["path"]), best_stage_path)
                    stage_candidates.append({
                        "name": f"{stage_name}_best",
                        "score": float(train_best["score"]),
                        "label": str(train_best["label"]),
                        "path": best_stage_path,
                    })
                    print(
                        f"[{log_ts()}] [AC curriculum] saved candidate "
                        f"{stage_name}_best: {train_best['label']}"
                    )
                stage_score, stage_label = _val_eval()
                stage_path = ac_path.replace(
                    ".pt",
                    f"_stage_{idx:02d}_{safe_stage_name}.pt",
                )
                ac_trainer.save(stage_path)
                stage_candidates.append({
                    "name": str(stage_name),
                    "score": float(stage_score),
                    "label": str(stage_label),
                    "path": stage_path,
                })
                print(
                    f"[{log_ts()}] [AC curriculum] saved candidate "
                    f"{stage_name}: {stage_label}"
                )

        if select_curriculum_stage:
            selected = max(stage_candidates, key=lambda candidate: candidate["score"])
            ac_trainer.load(selected["path"])
            ac_trainer.checkpoint_metadata["curriculum_stage_selection"] = {
                "selected": selected["name"],
                "score": selected["score"],
                "label": selected["label"],
                "candidates": [
                    {
                        "name": candidate["name"],
                        "score": candidate["score"],
                        "label": candidate["label"],
                    }
                    for candidate in stage_candidates
                ],
            }
            print(
                f"[{log_ts()}] [AC curriculum] selected stage "
                f"{selected['name']}: {selected['label']}"
            )
    else:
        ac_trainer.train(
            encoded_sequences=encoded_list,
            batch_size=ac_cfg.get("batch_size", 32),
            checkpoint_path=ac_path,
            val_eval_fn=_val_eval,
            checkpoint_eval_fn=_fire_checkpoint_eval if fire_selector_enabled else None,
            val_baseline_sharpe=bc_val_sharpe,
            online_wm_callback=_online_wm_cb if online_wm_steps_val > 0 else None,
        )
    ac_trainer.save(ac_path)
    _cleanup_auxiliary_checkpoints(
        ac_path,
        step_checkpoint_prefix=ac_trainer.step_checkpoint_prefix,
    )
    return ac_trainer
