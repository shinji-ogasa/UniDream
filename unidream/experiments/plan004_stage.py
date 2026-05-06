from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.research.plan004_residual_bc_ac import run_plan004_fold_policy


def _save_plan004_policy_npz(path: str, result: dict[str, Any]) -> None:
    """Persist the fold-local Plan004 policy result.

    The complete replay/evaluation positions are saved here. The neural WM/BC/AC
    checkpoints remain in world_model.pt, bc_actor.pt and ac.pt in the same fold
    directory.
    """
    selected = result.get("selected_row", {})
    best = result.get("best_candidate") or {}
    rec = best.get("record") or {}
    model = rec.get("model")
    arrays: dict[str, Any] = {
        "positions": np.asarray(result["positions"], dtype=np.float32),
        "benchmark_position": np.asarray([float(result.get("benchmark_position", 1.0))], dtype=np.float64),
        "fold": np.asarray([int(result.get("fold", -1))], dtype=np.int64),
        "source": np.asarray([str(selected.get("source", ""))], dtype=object),
        "spec": np.asarray([str(selected.get("spec", ""))], dtype=object),
        "status": np.asarray([str(result.get("status", ""))], dtype=object),
    }
    if model is not None:
        arrays.update(
            {
                "residual_model_mean": np.asarray(model.mean, dtype=np.float64),
                "residual_model_std": np.asarray(model.std, dtype=np.float64),
                "residual_model_coef": np.asarray(model.coef, dtype=np.float64),
            }
        )
    np.savez_compressed(path, **arrays)


def run_plan004_stage(
    *,
    fold_idx: int,
    wfo_dataset,
    cfg: dict[str, Any],
    costs_cfg: dict[str, Any],
    fold_ckpt_dir: str,
    log_ts,
) -> dict[str, Any] | None:
    plan_cfg = cfg.get("plan004_residual_bc_ac") or cfg.get("plan004") or {}
    if not bool(plan_cfg.get("enabled", False)):
        return None

    print(f"\n[{log_ts()}] [Step 5a] Plan004 residual BC/AC extraction...")
    result = run_plan004_fold_policy(
        ds=wfo_dataset,
        cfg=cfg,
        costs_cfg=costs_cfg,
        fold_idx=fold_idx,
        seed=int(plan_cfg.get("seed", 7)),
        ridge_l2=float(plan_cfg.get("ridge_l2", 1.0)),
        max_train_samples=int(plan_cfg.get("max_train_samples", 50000)),
        source_selection_mode=str(plan_cfg.get("source_selection_mode", "multi_source_val")),
        teacher_selection_mode=str(plan_cfg.get("teacher_selection_mode", "val_only")),
        selection_stress_mode=str(plan_cfg.get("selection_stress_mode", "primary")),
    )
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    policy_path = os.path.join(fold_ckpt_dir, "plan004_policy.npz")
    summary_path = os.path.join(fold_ckpt_dir, "plan004_summary.json")
    _save_plan004_policy_npz(policy_path, result)

    selected = result.get("selected_row", {})
    summary = {
        "fold": int(fold_idx),
        "status": result.get("status"),
        "selected": selected,
        "config": result.get("config", {}),
        "artifacts": {
            "plan004_policy": "plan004_policy.npz",
            "world_model": "world_model.pt",
            "bc_actor": "bc_actor.pt",
            "ac": "ac.pt",
        },
    }
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(summary), f, ensure_ascii=False, indent=2)

    stress = selected.get("stress", {}).get("cost_x1", {})
    print(
        "  Plan004 selected: "
        f"source={selected.get('source')} spec={selected.get('spec')} "
        f"alpha={float(stress.get('alpha_excess_pt', 0.0)):+.2f}pt "
        f"maxddD={float(stress.get('maxdd_delta_pt', 0.0)):+.2f}pt "
        f"turnover={float(stress.get('turnover', 0.0)):.2f}"
    )
    print(f"  Plan004 artifacts: {policy_path}, {summary_path}")
    return result
