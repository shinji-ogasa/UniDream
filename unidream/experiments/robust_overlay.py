"""Fixed multiscale, low-turnover B&H-relative overlay experiment.

The policy is intentionally small and causal.  A slow 90-day trend provides
the backbone exposure, while a bounded agreement score over 7/30/90-day
momentum decides whether the tactical component is allowed to move away from
the benchmark.  When the horizons disagree, the tactical component is exactly
the B&H target.  Decisions are emitted every six hours to avoid turning noisy
15-minute updates into a high-cost policy.

This module is a separate research family.  It does not amend Plan011/P1 or
overwrite any prior Alpha/DD artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .alpha_dd_features import make_features
from .alpha_dd_search import (
    aggregate,
    file_digest,
    fold_spec,
    load_bars,
    metrics,
    validate_data_artifact,
    write_json,
)


HORIZONS_DAYS = (7, 30, 90)
DECISION_CADENCE_HOURS = 6
LOWER_EXPOSURE = 0.50
UPPER_EXPOSURE = 1.12
SLOW_WEIGHT = 0.50
CONSENSUS_SCALE = 0.50
CONSENSUS_THRESHOLD = 0.15


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _finite_clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    finite = np.isfinite(values)
    values[finite] = np.clip(values[finite], low, high)
    return values


def build_targets(features: pd.DataFrame, *, cadence_hours: int = DECISION_CADENCE_HOURS) -> np.ndarray:
    """Build the fixed causal target path used by the research evaluator.

    Feature rows are already shifted by the registered feature builder.  The
    function only reads those rows and therefore never accesses an outcome or
    a future bar.  Non-decision bars are NaN so the evaluator cannot silently
    turn a six-hour policy into an hourly policy.
    """
    if cadence_hours <= 0 or 24 % cadence_hours != 0:
        raise ValueError("cadence_hours must be a positive divisor of 24")
    required = ["momentum_7", "momentum_30", "momentum_90", "vol_7"]
    missing = [name for name in required if name not in features.columns]
    if missing:
        raise ValueError(f"missing robust-overlay features: {missing}")
    index = pd.DatetimeIndex(features.index)
    if index.tz is None:
        raise ValueError("robust-overlay decisions require a timezone-aware index")

    vol = features["vol_7"].to_numpy(dtype=float)
    scores = []
    for days in HORIZONS_DAYS:
        momentum = features[f"momentum_{days}"].to_numpy(dtype=float)
        denominator = vol * np.sqrt(days / 365.0) + 1e-6
        scores.append(np.tanh((momentum / denominator) / CONSENSUS_SCALE))
    score_matrix = np.vstack(scores)
    # Divide by the fixed number of registered horizons.  A missing horizon
    # must not be silently averaged away; consensus_finite below makes that
    # row fail closed while avoiding nanmean's all-NaN warning.
    consensus = np.sum(score_matrix, axis=0) / float(len(HORIZONS_DAYS))
    consensus_finite = np.isfinite(score_matrix).all(axis=0)

    slow_momentum = features["momentum_90"].to_numpy(dtype=float)
    slow_finite = np.isfinite(slow_momentum)
    slow_target = np.where(slow_momentum >= 0.0, UPPER_EXPOSURE, LOWER_EXPOSURE)
    tactical_target = np.where(
        consensus >= CONSENSUS_THRESHOLD,
        UPPER_EXPOSURE,
        np.where(consensus <= -CONSENSUS_THRESHOLD, LOWER_EXPOSURE, 1.0),
    )
    targets = SLOW_WEIGHT * slow_target + (1.0 - SLOW_WEIGHT) * tactical_target
    targets[~(slow_finite & consensus_finite)] = np.nan

    decision = (index.hour % cadence_hours == 0) & (index.minute == 0)
    targets[~decision] = np.nan
    return _finite_clip(targets, LOWER_EXPOSURE, UPPER_EXPOSURE)


def _stage_config(config: dict[str, Any], stage: str) -> dict[str, Any]:
    try:
        value = config["stages"][stage]
    except KeyError as exc:
        raise ValueError(f"missing stage configuration: {stage}") from exc
    if not value.get("folds"):
        raise ValueError(f"stage {stage} has no folds")
    return value


def _run_stage(config: dict[str, Any], stage: str, output: Path, registration_sha: str) -> dict:
    stage_cfg = _stage_config(config, stage)
    data_path = Path(config["data_path"])
    bars = load_bars(data_path, cutoff=stage_cfg["data_cutoff"])
    features = make_features(bars)
    if not features.index.equals(bars.index):
        raise ValueError("feature/index alignment changed")
    targets = build_targets(features)
    base_contract = dict(config["execution"])
    stress_contract = {
        **base_contract,
        "one_way_cost": float(base_contract["one_way_cost"]) * 2.0,
        "borrow_annual": float(base_contract["borrow_annual"]) * 2.0,
    }
    rows: list[dict] = []
    for fold_id in stage_cfg["folds"]:
        fold = fold_spec(fold_id, config["fold_anchor"])
        if fold["test_end"] > pd.Timestamp(stage_cfg["data_cutoff"]):
            raise ValueError(f"fold {fold_id} exceeds its stage cutoff")
        ix = (bars.index >= fold["test_start"]) & (bars.index < fold["test_end"])
        window = bars.loc[ix]
        if len(window) == 0 or float(window["bar_available"].mean()) < float(config["minimum_bar_coverage"]):
            raise ValueError(f"fold {fold_id}: insufficient observed price coverage")
        base = metrics(window, targets[ix], base_contract)
        stress = metrics(window, targets[ix], stress_contract)
        rows.append({
            "fold": fold_id,
            "start": str(fold["test_start"]),
            "end": str(fold["test_end"]),
            "policy": {
                "decision_cadence_hours": DECISION_CADENCE_HOURS,
                "scheduled_decisions": int(np.isfinite(targets[ix]).sum()),
            },
            "base": base,
            "stress_2x": stress,
        })
        write_json(output / f"{stage}_progress.json", {
            "registration_sha256": registration_sha,
            "stage": stage,
            "folds_completed": len(rows),
            "folds": [row["fold"] for row in rows],
        })
        print(json.dumps({"event": "fold_complete", "stage": stage, "fold": fold_id}), flush=True)

    base_summary = aggregate([row["base"] for row in rows])
    stress_summary = aggregate([row["stress_2x"] for row in rows])
    result = {
        "schema": "robust-overlay-v1",
        "stage": stage,
        "registration_sha256": registration_sha,
        "data_file_sha256": file_digest(data_path),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "policy": {
            "horizons_days": list(HORIZONS_DAYS),
            "lower_exposure": LOWER_EXPOSURE,
            "upper_exposure": UPPER_EXPOSURE,
            "slow_weight": SLOW_WEIGHT,
            "consensus_scale": CONSENSUS_SCALE,
            "consensus_threshold": CONSENSUS_THRESHOLD,
            "decision_cadence_hours": DECISION_CADENCE_HOURS,
            "missing_feature_policy": "hold_previous_position",
        },
        "summary": base_summary,
        "stress_2x": stress_summary,
        "rows": rows,
        "confirmation_is_report_only": stage != "development",
        "formal_p1_result": False,
        "orders_submitted": 0,
    }
    write_json(output / f"{stage}.json", result)
    return result


def run(config_path: Path, stage: str) -> dict:
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(config["data_path"])
    data_proof = validate_data_artifact(data_path, expected_symbol=config.get("symbol"))
    registration = {
        "schema": "robust-overlay-v1",
        "config": config,
        "data_proof": data_proof,
        "policy": {
            "horizons_days": list(HORIZONS_DAYS),
            "lower_exposure": LOWER_EXPOSURE,
            "upper_exposure": UPPER_EXPOSURE,
            "slow_weight": SLOW_WEIGHT,
            "consensus_scale": CONSENSUS_SCALE,
            "consensus_threshold": CONSENSUS_THRESHOLD,
            "decision_cadence_hours": DECISION_CADENCE_HOURS,
        },
        "feature_source_sha256": file_digest(Path(make_features.__code__.co_filename)),
        "policy_source_sha256": file_digest(Path(__file__)),
    }
    registration_sha = _digest(registration)
    registration_path = output / "registration.json"
    if registration_path.exists():
        if json.loads(registration_path.read_text()) != registration:
            raise ValueError("registration changed; use a new output directory")
    elif stage != "development":
        raise ValueError("development must create the registration first")
    else:
        write_json(registration_path, registration)

    stage_path = output / f"{stage}.json"
    if stage_path.exists():
        raise ValueError(f"immutable stage already exists: {stage_path}")
    if stage != "development":
        lock_path = output / "selection_lock.json"
        if not lock_path.exists():
            raise ValueError("development selection lock is required before confirmation")
        lock = json.loads(lock_path.read_text())
        if lock.get("registration_sha256") != registration_sha:
            raise ValueError("selection lock does not match registration")
        development_path = output / "development.json"
        if file_digest(development_path) != lock.get("development_file_sha256"):
            raise ValueError("development artifact changed after lock")

    result = _run_stage(config, stage, output, registration_sha)
    if stage == "development":
        write_json(output / "selection_lock.json", {
            "registration_sha256": registration_sha,
            "development_file_sha256": file_digest(stage_path),
            "policy_id": "robust_90d_backbone_7_30_90_consensus_6h",
            "selected_before_confirmation": True,
        })
    if stage == "fresh":
        historical_path = output / "historical.json"
        historical = json.loads(historical_path.read_text())
        all_rows = historical["rows"] + result["rows"]
        expected = _stage_config(config, "historical")["folds"] + stage_cfg_folds(config, "fresh")
        complete = [row["fold"] for row in all_rows] == expected
        qualification = {
            "schema": "robust-overlay-v1-qualification",
            "registration_sha256": registration_sha,
            "historical_file_sha256": file_digest(historical_path),
            "fresh_file_sha256": file_digest(stage_path),
            "complete": complete,
            "formal_p1_result": False,
        }
        if complete:
            combined_base = aggregate([row["base"] for row in all_rows])
            combined_stress = aggregate([row["stress_2x"] for row in all_rows])
            qualification.update({
                "combined": combined_base,
                "stress_2x": combined_stress,
                "historical": historical["summary"],
                "fresh": result["summary"],
                "minimum_target_pass": bool(
                    combined_base["minimum_target_pass"] and combined_stress["minimum_target_pass"]
                ),
                "preferred_target_pass": bool(
                    combined_base["preferred_target_pass"] and combined_stress["preferred_target_pass"]
                ),
            })
        write_json(output / "qualification.json", qualification)
    return result


def stage_cfg_folds(config: dict[str, Any], stage: str) -> list[int]:
    return list(_stage_config(config, stage)["folds"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "historical", "fresh"), required=True)
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
