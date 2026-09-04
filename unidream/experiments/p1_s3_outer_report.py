"""Run the fixed S3 outer report as a terminal, report-only calculation.

The preregistered validation runner intentionally keeps ``execute_outer_report``
blocked so an ordinary validation call cannot consume the reserved outer
period.  This module is the explicit terminal operation for that period.  It
loads the immutable manifest and authenticated v4 body, fits each fixed
continuous model exactly once at the registered S3 outer origin, and writes a
descriptive report.  It never edits the manifest, tunes a model, changes a
threshold, or emits a promotable forecast/action artifact.

The report is therefore evidence about the fixed natural-BTC period, not a
replacement for the preregistered validation artifacts or a live-trading
claim.  Action replay uses the same canonical delay/fill/outcome mask graph as
the production validation path; outcome gaps can remove scoring but cannot
change a previously selected/fillable action state.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import warnings
from typing import Any

import numpy as np

from unidream.eval.action_execution import (
    ActionExecutionContract,
    complete_decision_starts,
    replay_action_path,
    select_block_decisions,
)

from .p1_recovery_runner import (
    S3_OUTER_END,
    S3_TRAIN_START,
    S3_VALIDATION_END,
    S3_VALIDATION_ORIGIN,
    FORECAST_HORIZONS,
    build_s3_arm_dataset,
    fit_model_at_origin,
    load_runner_manifest,
    load_s3_validation_body,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "codex_outputs" / "p1_s3_outer_report_20260904"
DEFAULT_PROVENANCE_AUDIT = ROOT / "codex_outputs" / "p1_s3_provenance_audit_20260904.json"
OUTER_HORIZON = 4
OUTER_MODELS = ("zero_return", "persistence_last_observed", "ridge")
OUTER_ARMS = ("zero_injection_control", "injected")


class S3OuterReportError(RuntimeError):
    """Raised when the fixed outer report cannot be safely calculated."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise S3OuterReportError("outer report contains a non-finite scalar")
        return float(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise S3OuterReportError(f"unsupported outer report value: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded + b"\n")
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return revision if len(revision) == 40 else "unknown"


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_outer_contract() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load and bind the exact fixed S3 outer ranges from the manifest."""

    manifest = load_runner_manifest()
    if manifest.get("results_observed") is not False:
        raise S3OuterReportError("outer report requires the immutable results_observed=false manifest")
    scenarios = manifest.get("scenarios")
    if not isinstance(scenarios, Mapping) or not isinstance(scenarios.get("S3"), Mapping):
        raise S3OuterReportError("fixed manifest is missing the S3 scenario")
    scenario = scenarios["S3"]
    expected = {
        "outer_report_origin_raw_index": S3_VALIDATION_END,
        "outer_report_fit_raw_range": (S3_TRAIN_START, S3_VALIDATION_END),
        "outer_report_prediction_raw_range": (S3_VALIDATION_END, S3_OUTER_END),
        "outer_report_refit_origins": (),
        "outer_test_is_report_only": True,
    }
    for field, expected_value in expected.items():
        actual = scenario.get(field)
        if field.endswith("_range") or field.endswith("_origins"):
            actual = tuple(actual) if isinstance(actual, (list, tuple)) else actual
        if actual != expected_value:
            raise S3OuterReportError(
                f"S3 outer manifest binding mismatch for {field}: {actual!r} != {expected_value!r}"
            )
    if tuple(scenario.get("outer_report_prediction_raw_range", ())) != (
        S3_VALIDATION_END,
        S3_OUTER_END,
    ):
        raise S3OuterReportError("S3 outer prediction range is not the fixed right-exclusive range")
    return manifest, scenario


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _forecast_metrics(
    predictions: np.ndarray,
    target: np.ndarray,
    score_mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(score_mask, dtype=bool) & np.isfinite(predictions) & np.isfinite(target)
    if not np.any(mask):
        raise S3OuterReportError("outer forecast has no finite score rows")
    predicted = predictions[mask]
    observed = target[mask]
    error = predicted - observed
    return {
        "score_rows": int(np.count_nonzero(mask)),
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
        "ic_pearson": _safe_correlation(predicted, observed),
        "sign_accuracy": float(np.mean((predicted > 0.0) == (observed > 0.0))),
        "prediction_mean": float(np.mean(predicted)),
        "target_mean": float(np.mean(observed)),
        "prediction_std": float(np.std(predicted)),
        "target_std": float(np.std(observed)),
    }


def _max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    equity = np.cumsum(np.asarray(pnl, dtype=np.float64))
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:]
    return float(np.max(peak - equity))


def _action_metrics(
    dataset: Any,
    fit: Any,
    contract: ActionExecutionContract,
) -> dict[str, Any]:
    n_rows = len(dataset.returns)
    starts = complete_decision_starts(n_rows, contract)
    common_mask = np.ones(len(starts), dtype=bool)
    bar_available = np.asarray(dataset.availability["spot_bar_observed"], dtype=bool)
    prediction_mask = np.asarray(fit.prediction_mask, dtype=bool)
    predictions = np.asarray(fit.predictions, dtype=np.float64)
    if prediction_mask.shape != (n_rows,) or predictions.shape != (n_rows,):
        raise S3OuterReportError("outer action forecast is not full-grid aligned")
    deltas = select_block_decisions(
        predictions,
        contract,
        decision_eligible=prediction_mask,
        bar_available=bar_available,
    )
    trajectory = replay_action_path(
        dataset.returns,
        deltas,
        contract,
        decision_eligible=prediction_mask,
        bar_available=bar_available,
        forecast_finite_mask=np.isfinite(predictions),
        common_mask=common_mask,
    )
    hold = replay_action_path(
        dataset.returns,
        np.zeros(n_rows, dtype=np.float64),
        contract,
        decision_eligible=prediction_mask,
        bar_available=bar_available,
        forecast_finite_mask=np.isfinite(predictions),
        common_mask=common_mask,
    )
    scored = trajectory.scored_mask
    hold_scored = hold.scored_mask
    if not np.array_equal(scored, hold_scored):
        raise S3OuterReportError("outer action and hold score masks diverged")
    action_net = trajectory.net_pnl[scored]
    hold_net = hold.net_pnl[hold_scored]
    before_perturb = 150_000
    if before_perturb >= n_rows:
        raise S3OuterReportError("future perturbation boundary is outside the outer body")
    changed_returns = np.array(dataset.returns, dtype=np.float64, copy=True)
    changed_tail = changed_returns[before_perturb:]
    finite = np.isfinite(changed_tail)
    changed_tail[finite] += 0.000123456789
    changed = replay_action_path(
        changed_returns,
        deltas,
        contract,
        decision_eligible=prediction_mask,
        bar_available=bar_available,
        forecast_finite_mask=np.isfinite(predictions),
        common_mask=common_mask,
    )
    for name in ("intent_deltas", "decision_deltas", "fill_mask", "effective_positions"):
        if not np.array_equal(getattr(trajectory, name), getattr(changed, name), equal_nan=True):
            raise S3OuterReportError(f"future return perturbation changed action state: {name}")
    mask_hashes = trajectory.block_masks.mask_hash_registry
    return {
        "forecast_model_id": fit.model_id,
        "contract_hash": contract.contract_hash,
        "mask_hashes": dict(mask_hashes),
        "decision_delta_sha256": _sha256_array(deltas),
        "forecast_finite_rows": int(np.count_nonzero(prediction_mask)),
        "counts": trajectory.eligibility_counts,
        "action_metric_blocks": int(np.count_nonzero(trajectory.block_masks.action_metric_mask)),
        "utility_metric_blocks": int(np.count_nonzero(trajectory.block_masks.utility_metric_mask)),
        "filled_blocks": int(trajectory.n_filled_blocks),
        "turnover": float(np.abs(trajectory.decision_deltas).sum()),
        "gross_total": float(trajectory.gross_pnl[scored].sum()),
        "cost_total": float(trajectory.transition_costs[scored].sum()),
        "net_total": float(action_net.sum()),
        "hold_net_total": float(hold_net.sum()),
        "alpha_ex_vs_hold": float(action_net.sum() - hold_net.sum()),
        "max_drawdown": _max_drawdown(action_net),
        "hold_max_drawdown": _max_drawdown(hold_net),
        "future_return_perturbation_state_invariant": True,
        "future_return_perturbation_start": before_perturb,
    }


def _run_arm(
    body: Any,
    arm: str,
    model_ids: Sequence[str],
    contract: ActionExecutionContract,
) -> dict[str, Any]:
    dataset = build_s3_arm_dataset(body, arm)  # type: ignore[arg-type]
    try:
        horizon_column = tuple(FORECAST_HORIZONS).index(OUTER_HORIZON)
    except ValueError as exc:  # pragma: no cover - fixed runner contract
        raise S3OuterReportError("runner does not expose the fixed h4 horizon") from exc
    target = np.asarray(dataset.targets[:, horizon_column], dtype=np.float64)
    results: dict[str, Any] = {}
    for model_id in model_ids:
        task = "continuous"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_model_at_origin(
                dataset,
                model_id,
                S3_VALIDATION_END,
                OUTER_HORIZON,
                task=task,
                prediction_range=(S3_VALIDATION_END, S3_OUTER_END),
                train_start=S3_TRAIN_START,
            )
        if fit.status != "ok":
            raise S3OuterReportError(f"outer {arm}/{model_id} fit returned {fit.status}: {fit.reason}")
        forecast_mask = np.asarray(fit.eligible_mask, dtype=bool)
        entry: dict[str, Any] = {
            "fit": {
                "status": fit.status,
                "model_id": fit.model_id,
                "horizon": fit.horizon,
                "origin": fit.origin,
                "train_start": fit.train_start,
                "train_rows": int(np.count_nonzero(fit.train_mask)),
                "inference_rows": int(np.count_nonzero(fit.prediction_mask)),
                "score_rows": int(np.count_nonzero(forecast_mask)),
                "prediction_range": [S3_VALIDATION_END, S3_OUTER_END],
                "warnings": [
                    {"category": type(item.message).__name__, "message": str(item.message)}
                    for item in caught
                ],
            },
            "forecast": _forecast_metrics(fit.predictions, target, forecast_mask),
            "action": _action_metrics(dataset, fit, contract),
        }
        results[model_id] = entry
    return {
        "arm": arm,
        "beta": float(dataset.beta),
        "source_body_sha256": dataset.source_body_sha256,
        "results": results,
    }


def run_s3_outer_report(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    provenance_audit: str | Path = DEFAULT_PROVENANCE_AUDIT,
    model_ids: Sequence[str] = OUTER_MODELS,
    arms: Sequence[str] = OUTER_ARMS,
) -> Mapping[str, Any]:
    """Execute the fixed S3 report-only outer calculation exactly once."""

    model_ids = tuple(model_ids)
    arms = tuple(arms)
    if not model_ids or any(model not in OUTER_MODELS for model in model_ids):
        raise S3OuterReportError("outer model_ids must be a non-empty subset of the fixed continuous models")
    if not arms or any(arm not in OUTER_ARMS for arm in arms):
        raise S3OuterReportError("outer arms must be a non-empty subset of the fixed S3 arms")
    manifest, scenario = _manifest_outer_contract()
    audit_path = Path(provenance_audit)
    if not audit_path.exists():
        raise S3OuterReportError(f"S3 provenance audit is missing: {audit_path}")
    try:
        audit = json.loads(audit_path.read_text())
    except (OSError, ValueError) as exc:
        raise S3OuterReportError("S3 provenance audit is not valid JSON") from exc
    if audit.get("disposition", {}).get("status") != "pass_body_source_provenance_only_difference":
        raise S3OuterReportError("S3 provenance audit does not have the approved body-only disposition")
    if audit.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise S3OuterReportError("S3 provenance audit is bound to a different manifest")

    body = load_s3_validation_body(root=ROOT)
    runtime = body.runtime
    if runtime.get("v4_runtime_validation_status") != "passed":
        raise S3OuterReportError("authenticated v4 runtime status is not passed")
    if runtime.get("v4_runtime_body_match") is not True or runtime.get("v4_runtime_loaded_body_match") is not True:
        raise S3OuterReportError("authenticated v4 body/content match is not true")
    if body.body_sha256 != audit.get("body_sha256"):
        raise S3OuterReportError("S3 provenance audit body digest does not match the authenticated body")
    contract = ActionExecutionContract.canonical()
    arm_results = {
        arm: _run_arm(body, arm, model_ids, contract)
        for arm in arms
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_id": "p1-s3-outer-report-20260904",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_revision": _git_revision(),
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_results_observed": manifest.get("results_observed"),
        "scenario": "S3",
        "outer_report_executed": True,
        "outer_results_observed": True,
        "promotion_allowed": False,
        "selection_allowed": False,
        "threshold_revision_allowed": False,
        "report_only": True,
        "fit_rule": scenario.get("outer_report_fit_rule"),
        "origin": S3_VALIDATION_END,
        "fit_prefix_range": [S3_TRAIN_START, S3_VALIDATION_END],
        "prediction_range": [S3_VALIDATION_END, S3_OUTER_END],
        "refit_origins": [],
        "horizon": OUTER_HORIZON,
        "models": list(model_ids),
        "arms": list(arms),
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash,
        "provenance_audit_path": str(audit_path),
        "provenance_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "body_sha256": body.body_sha256,
        "runtime_provenance": _plain(runtime),
        "results": arm_results,
        "natural_control_arm": "zero_injection_control",
        "natural_control_summary": arm_results.get("zero_injection_control"),
        "notes": [
            "One fit per fixed model at the registered S3 outer origin; no refit, selection, or threshold tuning.",
            "This terminal report is separate from validation artifacts and cannot be loaded as a promotable production artifact.",
            "The source body is authenticated and content-matched; only the local source-probe metadata revision differs from frozen provenance.",
        ],
    }
    destination = Path(output)
    # Hash the canonical report object before adding its own digest field; the
    # on-disk newline and the self-field therefore cannot make the identity
    # circular or platform-dependent.
    report_content_sha = hashlib.sha256(_json_bytes(report)).hexdigest()
    report["report_content_sha256"] = report_content_sha
    # Rewrite once with its own digest omitted from the hashed payload to keep
    # the identity non-circular; callers can hash the final file separately.
    _write_json(destination / "outer_report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provenance-audit", type=Path, default=DEFAULT_PROVENANCE_AUDIT)
    args = parser.parse_args(argv)
    report = run_s3_outer_report(args.output, provenance_audit=args.provenance_audit)
    print(json.dumps({
        "report": str(args.output / "outer_report.json"),
        "report_content_sha256": report.get("report_content_sha256"),
        "manifest_sha256": report.get("manifest_sha256"),
        "body_sha256": report.get("body_sha256"),
        "outer_results_observed": report.get("outer_results_observed"),
        "promotion_allowed": report.get("promotion_allowed"),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["S3OuterReportError", "run_s3_outer_report", "main"]
