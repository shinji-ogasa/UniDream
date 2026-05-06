from __future__ import annotations


def run_val_selector_stage(
    *,
    actor,
    wm_trainer,
    wfo_dataset,
    seq_len: int,
    val_regime_probs,
    val_advantage_values,
    device: str,
    cfg: dict,
    ac_cfg: dict,
    costs_cfg: dict,
    backtest_cls,
    action_stats_fn,
    selector_cfg_fn,
    selector_candidate_fn,
    select_policy_candidate_fn,
    candidate_to_text_fn,
    benchmark_positions_fn,
    benchmark_position: float,
):
    source = str(ac_cfg.get("test_policy_source", ac_cfg.get("policy_source", "actor"))).lower()
    if source in {"hierarchy_bundle", "external_hierarchy_bundle"}:
        print("  [ValAdj] skipped: external hierarchy policy source")
        return
    adjust_scale_grid = ac_cfg.get("val_adjust_rate_scale_grid", [])
    adv_level_grid = ac_cfg.get("val_advantage_level_grid", [actor.infer_advantage_level])
    if len(adjust_scale_grid) == 0:
        return

    val_features_arr = wfo_dataset.val_features
    val_returns_arr = wfo_dataset.val_returns
    if len(val_features_arr) == 0:
        return

    enc_val = wm_trainer.encode_sequence(val_features_arr, seq_len=seq_len)
    original_scale = float(getattr(actor, "infer_adjust_rate_scale", 1.0))
    original_adv = float(getattr(actor, "infer_advantage_level", 0.0))
    selector_cfg = selector_cfg_fn(ac_cfg)
    selector_candidates = []
    for adv_level in adv_level_grid:
        actor.infer_advantage_level = float(adv_level)
        for candidate in adjust_scale_grid:
            actor.infer_adjust_rate_scale = float(candidate)
            pos = actor.predict_positions(
                enc_val["z"],
                enc_val["h"],
                regime_np=val_regime_probs,
                advantage_np=val_advantage_values,
                device=device,
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
            candidate_key = {"scale": float(candidate), "adv": float(adv_level)}
            candidate_rec = selector_candidate_fn(
                candidate_key,
                metrics,
                stats,
                benchmark_position=benchmark_position,
                selector_cfg=selector_cfg,
                cfg=cfg,
            )
            selector_candidates.append(candidate_rec)
            print(f"  [ValAdj] {candidate_to_text_fn(candidate_key)} {candidate_rec['label']}")

    chosen = select_policy_candidate_fn(selector_candidates, selector_cfg)
    chosen_candidate = chosen["candidate"] if isinstance(chosen["candidate"], dict) else {
        "scale": float(chosen["candidate"]),
        "adv": original_adv,
    }
    actor.infer_adjust_rate_scale = float(chosen_candidate["scale"])
    actor.infer_advantage_level = float(chosen_candidate.get("adv", original_adv))
    print(
        f"  [ValAdj] selected {candidate_to_text_fn(chosen_candidate)} "
        f"(default=scale={original_scale:.3f} adv={original_adv:.2f}) {chosen['label']}"
    )
