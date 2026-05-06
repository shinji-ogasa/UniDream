from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.exploration_board_probe import _rolling_past_vol
from unidream.research.plan004_residual_bc_ac import SPECS, run_plan004_fold_policy


def _finite_or_marker(value: Any) -> float:
    if value is None:
        return float("nan")
    if value == "inf":
        return float("inf")
    return float(value)


def _binary_pipeline_arrays(model: Any) -> dict[str, np.ndarray]:
    if model is None or not hasattr(model, "named_steps"):
        return {}
    scaler = model.named_steps.get("standardscaler")
    classifier = model.named_steps.get("logisticregression")
    if scaler is None or classifier is None:
        return {}
    return {
        "base_danger_scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "base_danger_scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "base_danger_coef": np.asarray(classifier.coef_, dtype=np.float64),
        "base_danger_intercept": np.asarray(classifier.intercept_, dtype=np.float64),
    }


def _train_vol_quantiles(train_returns: np.ndarray) -> tuple[float, float]:
    vol = _rolling_past_vol(np.asarray(train_returns, dtype=np.float64), 64)
    finite = vol[np.isfinite(vol)]
    if len(finite) < 20:
        return float("nan"), float("nan")
    q1, q2 = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    return float(q1), float(q2)


def _save_plan004_policy_npz(path: str, result: dict[str, Any], *, train_returns: np.ndarray, cfg: dict[str, Any]) -> None:
    """Persist the fold-local Plan004 policy result.

    The complete replay/evaluation positions are saved here. The neural WM/BC/AC
    checkpoints remain in world_model.pt, bc_actor.pt and ac.pt in the same fold
    directory.
    """
    selected = result.get("selected_row", {})
    best = result.get("best_candidate") or {}
    rec = best.get("record") or {}
    model = rec.get("model")
    spec = next((s for s in SPECS if s.name == str(selected.get("spec", ""))), None)
    selection = selected.get("selection") or {}
    threshold = _finite_or_marker(selection.get("threshold", float("inf")))
    base_source = str(selected.get("source", ""))
    source_candidate = best.get("source_candidate") or {}
    base_model = source_candidate.get("_selector_model")
    base_spec = source_candidate.get("_selector_spec")
    base_selection = (source_candidate.get("meta") or {}).get("selection") or {}
    base_kind = "d_risk_selector" if base_source.startswith("D_risk_sensitive") and base_model is not None else "benchmark"
    q1, q2 = _train_vol_quantiles(train_returns)
    ac_cfg = cfg.get("ac", {})
    arrays: dict[str, Any] = {
        "positions": np.asarray(result["positions"], dtype=np.float32),
        "benchmark_position": np.asarray([float(result.get("benchmark_position", 1.0))], dtype=np.float64),
        "fold": np.asarray([int(result.get("fold", -1))], dtype=np.int64),
        "source": np.asarray([str(selected.get("source", ""))], dtype=object),
        "spec": np.asarray([str(selected.get("spec", ""))], dtype=object),
        "status": np.asarray([str(result.get("status", ""))], dtype=object),
        "base_kind": np.asarray([base_kind], dtype=object),
        "threshold": np.asarray([threshold], dtype=np.float64),
        "hold_bars": np.asarray([int(selection.get("hold_bars", 1))], dtype=np.int64),
        "cooldown_bars": np.asarray([int(selection.get("cooldown_bars", 0))], dtype=np.int64),
        "max_total_turnover": np.asarray([float(spec.max_turnover) if spec is not None else float("inf")], dtype=np.float64),
        "min_position": np.asarray([float(ac_cfg.get("abs_min_position", 0.0))], dtype=np.float64),
        "max_position": np.asarray([float(ac_cfg.get("abs_max_position", 1.25))], dtype=np.float64),
        "deltas": np.asarray(spec.deltas if spec is not None else (0.0,), dtype=np.float64),
        "base_regime_q1": np.asarray([q1], dtype=np.float64),
        "base_regime_q2": np.asarray([q2], dtype=np.float64),
    }
    if model is not None:
        arrays.update(
            {
                "model_mean": np.asarray(model.mean, dtype=np.float64),
                "model_std": np.asarray(model.std, dtype=np.float64),
                "model_coef": np.asarray(model.coef, dtype=np.float64),
            }
        )
    if base_kind == "d_risk_selector" and base_spec is not None:
        arrays.update(
            {
                "base_model_mean": np.asarray(base_model.mean, dtype=np.float64),
                "base_model_std": np.asarray(base_model.std, dtype=np.float64),
                "base_model_coef": np.asarray(base_model.coef, dtype=np.float64),
                "base_candidates": np.asarray(base_spec.candidates, dtype=np.float64),
                "base_threshold": np.asarray([_finite_or_marker(base_selection.get("threshold", float("inf")))], dtype=np.float64),
                "base_cooldown_bars": np.asarray([int(base_selection.get("cooldown_bars", getattr(base_spec, "cooldown_bars", 0)))], dtype=np.int64),
                "base_hold_bars": np.asarray([int(getattr(base_spec, "hold_bars", 1))], dtype=np.int64),
                "base_horizon": np.asarray([int(getattr(base_spec, "horizon", 1))], dtype=np.int64),
                "base_mode": np.asarray([str(getattr(base_spec, "mode", ""))], dtype=object),
                "base_regime": np.asarray([str(base_selection.get("regime", "all"))], dtype=object),
                "base_danger_cap": np.asarray([_finite_or_marker(base_selection.get("danger_cap"))], dtype=np.float64),
            }
        )
        arrays.update(_binary_pipeline_arrays(source_candidate.get("_danger_model")))
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
    _save_plan004_policy_npz(policy_path, result, train_returns=wfo_dataset.train_returns, cfg=cfg)

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
