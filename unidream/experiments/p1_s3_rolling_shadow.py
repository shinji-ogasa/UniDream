"""Report-only rolling BTC diagnostics and an orderless paper shadow.

The preregistered S3 outer operation is a single fixed-origin calculation.
This module is deliberately separate from that operation: it runs a fixed,
pre-declared set of expanding-origin windows on the authenticated cached S3
control body, without model/threshold selection or manifest mutation.  The
last window is converted into an offline paper-shadow record.  No exchange
connection, order submission, or live-money state is touched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import argparse
import hashlib
import json
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
    FORECAST_HORIZONS,
    S3_OUTER_END,
    S3_TRAIN_START,
    build_s3_arm_dataset,
    fit_model_at_origin,
    load_runner_manifest,
    load_s3_validation_body,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "codex_outputs" / "p1_s3_rolling_shadow_20260904"
OUTER_HORIZON = 4
MODELS = ("zero_return", "persistence_last_observed", "ridge")

# These raw-index windows are fixed before execution.  They are deliberately
# simple, contiguous and aligned to the four-bar commitment grid.  They are a
# post-hoc diagnostic, not a new preregistered test or selector.
ROLLING_WINDOWS: tuple[tuple[int, int], ...] = (
    (104_528, 113_000),
    (121_000, 130_000),
    (139_568, 148_000),
    (156_000, 164_000),
    (164_000, S3_OUTER_END),
)


class RollingShadowError(RuntimeError):
    """Raised when a report-only rolling calculation is unsafe."""


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
            raise RollingShadowError("report contains a non-finite scalar")
        return float(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise RollingShadowError(f"unsupported report value: {type(value).__name__}")


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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_json_bytes(value) + b"\n")
    temporary.replace(path)
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
        raise RollingShadowError("rolling window has no finite forecast score rows")
    predicted = np.asarray(predictions, dtype=np.float64)[mask]
    observed = np.asarray(target, dtype=np.float64)[mask]
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


def _window_bar_available(dataset: Any, start: int, end: int) -> np.ndarray:
    """Keep only the declared window available for report-only replay."""

    n_rows = len(dataset.returns)
    if not (0 <= start < end <= n_rows):
        raise RollingShadowError(f"invalid rolling window {(start, end)} for {n_rows} rows")
    source = np.asarray(dataset.availability["spot_bar_observed"], dtype=np.bool_)
    if source.shape != (n_rows,):
        raise RollingShadowError("Spot availability is not full-grid aligned")
    result = np.array(source, copy=True)
    result[:start] = False
    result[end:] = False
    result.setflags(write=False)
    return result


def _action_metrics(
    dataset: Any,
    fit: Any,
    contract: ActionExecutionContract,
    *,
    start: int,
    end: int,
) -> dict[str, Any]:
    n_rows = len(dataset.returns)
    prediction_mask = np.asarray(fit.prediction_mask, dtype=np.bool_)
    predictions = np.asarray(fit.predictions, dtype=np.float64)
    if prediction_mask.shape != (n_rows,) or predictions.shape != (n_rows,):
        raise RollingShadowError("rolling action forecast is not full-grid aligned")
    bar_available = _window_bar_available(dataset, start, end)
    common_mask = np.ones(len(complete_decision_starts(n_rows, contract)), dtype=np.bool_)
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
        raise RollingShadowError("rolling action and hold score masks diverged")
    action_pnl = trajectory.net_pnl[scored]
    hold_pnl = hold.net_pnl[hold_scored]

    # A tail return perturbation is not used to fit a model.  It verifies that
    # replay state/intent remains causal while outcome metrics may change.
    perturb_start = start + max((end - start) // 2, 1)
    changed_returns = np.array(dataset.returns, dtype=np.float64, copy=True)
    changed_tail = changed_returns[perturb_start:end]
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
        if not np.array_equal(
            getattr(trajectory, name), getattr(changed, name), equal_nan=True
        ):
            raise RollingShadowError(f"future return perturbation changed {name}")

    return {
        "mask_hashes": dict(trajectory.block_masks.mask_hash_registry),
        "decision_delta_sha256": hashlib.sha256(
            np.ascontiguousarray(deltas).tobytes(order="C")
        ).hexdigest(),
        "forecast_finite_rows": int(np.count_nonzero(prediction_mask)),
        "counts": trajectory.eligibility_counts,
        "action_metric_blocks": int(np.count_nonzero(trajectory.block_masks.action_metric_mask)),
        "utility_metric_blocks": int(np.count_nonzero(trajectory.block_masks.utility_metric_mask)),
        "filled_blocks": int(trajectory.n_filled_blocks),
        "turnover": float(np.abs(trajectory.decision_deltas).sum()),
        "gross_total": float(trajectory.gross_pnl[scored].sum()),
        "cost_total": float(trajectory.transition_costs[scored].sum()),
        "net_total": float(action_pnl.sum()),
        "hold_net_total": float(hold_pnl.sum()),
        "alpha_ex_vs_hold": float(action_pnl.sum() - hold_pnl.sum()),
        "max_drawdown": _max_drawdown(action_pnl),
        "hold_max_drawdown": _max_drawdown(hold_pnl),
        "future_return_perturbation_state_invariant": True,
        "future_return_perturbation_start": perturb_start,
    }


def _validate_windows(n_rows: int) -> None:
    previous_end = None
    for start, end in ROLLING_WINDOWS:
        if start % 4 or end % 4 and end != n_rows:
            raise RollingShadowError(f"window is not commitment-grid aligned: {(start, end)}")
        if start < S3_TRAIN_START or end > n_rows or end <= start:
            raise RollingShadowError(f"window is outside the authenticated body: {(start, end)}")
        if previous_end is not None and start < previous_end:
            raise RollingShadowError("rolling windows overlap or are out of order")
        previous_end = end


def _window_record(
    dataset: Any,
    contract: ActionExecutionContract,
    *,
    start: int,
    end: int,
    model_ids: Sequence[str],
) -> dict[str, Any]:
    try:
        horizon_column = tuple(FORECAST_HORIZONS).index(OUTER_HORIZON)
    except ValueError as exc:  # pragma: no cover - fixed runner contract
        raise RollingShadowError("runner does not expose h4") from exc
    target = np.asarray(dataset.targets[:, horizon_column], dtype=np.float64)
    results: dict[str, Any] = {}
    for model_id in model_ids:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_model_at_origin(
                dataset,
                model_id,
                start,
                OUTER_HORIZON,
                task="continuous",
                prediction_range=(start, end),
                train_start=S3_TRAIN_START,
            )
        if fit.status != "ok":
            raise RollingShadowError(
                f"rolling {start}:{end}/{model_id} fit returned {fit.status}: {fit.reason}"
            )
        score_mask = np.asarray(fit.eligible_mask, dtype=np.bool_)
        results[model_id] = {
            "fit": {
                "status": fit.status,
                "model_id": fit.model_id,
                "horizon": fit.horizon,
                "origin": fit.origin,
                "train_start": fit.train_start,
                "train_rows": int(np.count_nonzero(fit.train_mask)),
                "inference_rows": int(np.count_nonzero(fit.prediction_mask)),
                "score_rows": int(np.count_nonzero(score_mask)),
                "prediction_range": [start, end],
                "warnings": [
                    {"category": type(item.message).__name__, "message": str(item.message)}
                    for item in caught
                ],
            },
            "forecast": _forecast_metrics(fit.predictions, target, score_mask),
            "action": _action_metrics(dataset, fit, contract, start=start, end=end),
        }
    return {
        "window": {
            "raw_range": [start, end],
            "start_timestamp": str(dataset.timestamps[start]),
            "end_timestamp_exclusive": str(dataset.timestamps[end]) if end < len(dataset.timestamps) else None,
        },
        "results": results,
    }


def _aggregate(records: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for model_id in model_ids:
        rows = [record["results"][model_id] for record in records]
        numeric_fields = (
            ("forecast", "mse"),
            ("forecast", "mae"),
            ("forecast", "sign_accuracy"),
            ("action", "net_total"),
            ("action", "hold_net_total"),
            ("action", "alpha_ex_vs_hold"),
            ("action", "cost_total"),
            ("action", "filled_blocks"),
        )
        values: dict[str, float] = {}
        for section, field in numeric_fields:
            key = f"{section}.{field}"
            values[key] = float(np.mean([float(row[section][field]) for row in rows]))
        ics = [row["forecast"]["ic_pearson"] for row in rows]
        values["forecast.ic_pearson_mean_finite"] = float(
            np.mean([float(value) for value in ics if value is not None])
        ) if any(value is not None for value in ics) else None
        aggregate[model_id] = values
    return aggregate


def run_s3_rolling_shadow(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    model_ids: Sequence[str] = MODELS,
) -> Mapping[str, Any]:
    """Run fixed-window natural BTC diagnostics and an offline shadow."""

    model_ids = tuple(model_ids)
    if not model_ids or any(model not in MODELS for model in model_ids):
        raise RollingShadowError("model_ids must be a non-empty subset of fixed models")
    manifest = load_runner_manifest()
    if manifest.get("results_observed") is not False:
        raise RollingShadowError("rolling diagnostic requires immutable results_observed=false manifest")
    body = load_s3_validation_body(root=ROOT)
    dataset = build_s3_arm_dataset(body, "zero_injection_control")
    _validate_windows(len(dataset.returns))
    contract = ActionExecutionContract.canonical()
    records = [
        _window_record(
            dataset,
            contract,
            start=start,
            end=end,
            model_ids=model_ids,
        )
        for start, end in ROLLING_WINDOWS
    ]
    last = records[-1]
    shadow_model = "ridge" if "ridge" in model_ids else model_ids[0]
    shadow_row = last["results"][shadow_model]
    shadow = {
        "mode": "offline_orderless_paper_shadow",
        "source": "authenticated_cached_s3_zero_injection_control",
        "model_id": shadow_model,
        "window": last["window"],
        "orders_submitted": 0,
        "external_fills": 0,
        "simulated_filled_blocks": shadow_row["action"]["filled_blocks"],
        "simulated_net_total": shadow_row["action"]["net_total"],
        "simulated_alpha_ex_vs_hold": shadow_row["action"]["alpha_ex_vs_hold"],
        "data_freshness": "cached_historical_body; no live feed",
        "promotion_allowed": False,
        "live_money": False,
        "notes": [
            "This is a deterministic replay shadow, not exchange connectivity or execution evidence.",
            "No order, account, or external trading state was written.",
        ],
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "report_id": "p1-s3-rolling-shadow-20260904",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "code_revision": _git_revision(),
        "manifest_id": manifest.get("manifest_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_results_observed": manifest.get("results_observed"),
        "body_sha256": body.body_sha256,
        "arm": "zero_injection_control",
        "horizon": OUTER_HORIZON,
        "train_start": S3_TRAIN_START,
        "windows": records,
        "aggregate_mean": _aggregate(records, model_ids),
        "contract": contract.to_dict(),
        "contract_hash": contract.contract_hash,
        "selection_allowed": False,
        "threshold_revision_allowed": False,
        "promotion_allowed": False,
        "report_only": True,
        "outer_results_observed": True,
        "offline_paper_shadow": shadow,
        "notes": [
            "Fixed expanding-origin diagnostic windows were declared in source before execution.",
            "This post-hoc diagnostic does not amend or replace the preregistered single S3 outer operation.",
            "The Ridge matmul runtime warning is retained in each fit record; finite outputs are not a numerical sign-off.",
        ],
    }
    destination = Path(output)
    content_digest = hashlib.sha256(_json_bytes(report)).hexdigest()
    report["report_content_sha256"] = content_digest
    _write_json(destination / "rolling_shadow.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run_s3_rolling_shadow(args.output)
    print(
        json.dumps(
            {
                "report": str(args.output / "rolling_shadow.json"),
                "report_content_sha256": report["report_content_sha256"],
                "body_sha256": report["body_sha256"],
                "promotion_allowed": report["promotion_allowed"],
                "orders_submitted": report["offline_paper_shadow"]["orders_submitted"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "MODELS",
    "ROLLING_WINDOWS",
    "RollingShadowError",
    "_forecast_metrics",
    "_window_bar_available",
    "run_s3_rolling_shadow",
    "main",
]
