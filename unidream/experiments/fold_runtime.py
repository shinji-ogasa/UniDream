"""Fold-level helpers for the strict from-scratch training pipeline."""
from __future__ import annotations


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
