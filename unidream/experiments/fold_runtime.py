from __future__ import annotations

import os


PIPELINE_STAGES = ("wm", "bc", "ac", "test")
_STAGE_TO_INDEX = {stage: idx for idx, stage in enumerate(PIPELINE_STAGES)}


def stage_idx(stage: str) -> int:
    return _STAGE_TO_INDEX[stage]


def resolve_ac_max_steps(ac_cfg: dict) -> int:
    max_steps = int(ac_cfg.get("max_steps", 200_000))
    curriculum = ac_cfg.get("curriculum") or []
    if not curriculum:
        return max_steps
    stage_steps = [
        int(stage.get("until_step", stage.get("max_steps", 0)))
        for stage in curriculum
    ]
    return max(max_steps, max(stage_steps, default=0))


def prepare_fold_runtime(
    *,
    fold_idx: int,
    checkpoint_dir: str,
    ac_cfg: dict,
    resume: bool,
    start_from: str,
    stop_after: str,
) -> dict:
    fold_ckpt_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    wm_path = os.path.join(fold_ckpt_dir, "world_model.pt")
    bc_path = os.path.join(fold_ckpt_dir, "bc_actor.pt")
    ac_path = os.path.join(fold_ckpt_dir, "ac.pt")
    ac_max_steps_cfg = resolve_ac_max_steps(ac_cfg)
    ignore_ac_ckpt = bool(ac_cfg.get("ignore_ac_checkpoint", False)) or ac_max_steps_cfg <= 0

    start_index = stage_idx(start_from)
    stop_index = stage_idx(stop_after)
    has_wm_ckpt = os.path.exists(wm_path)
    has_bc_ckpt = os.path.exists(bc_path)
    has_ac_ckpt = os.path.exists(ac_path) and not ignore_ac_ckpt

    return {
        "fold_ckpt_dir": fold_ckpt_dir,
        "wm_path": wm_path,
        "bc_path": bc_path,
        "ac_path": ac_path,
        "ac_max_steps_cfg": ac_max_steps_cfg,
        "ignore_ac_ckpt": ignore_ac_ckpt,
        "start_idx": start_index,
        "stop_idx": stop_index,
        "has_wm_ckpt": has_wm_ckpt,
        "has_bc_ckpt": has_bc_ckpt,
        "has_ac_ckpt": has_ac_ckpt,
        "has_wm": has_wm_ckpt and (resume or start_index > stage_idx("wm")),
        "has_bc": has_bc_ckpt and (resume or start_index > stage_idx("bc")),
        "has_ac": has_ac_ckpt and (resume or start_index > stage_idx("ac")),
    }
