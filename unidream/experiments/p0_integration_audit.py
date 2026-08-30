"""Small, reproducible integration audit for the P0 contracts.

This module deliberately sits below the expensive WM/BC/AC pipeline.  It
loads a schema-v4 cache, expands the P0-A sidecar to the complete timestamp
grid (never compacting missing body rows), and feeds the resulting causal
decision/score masks to the P0-C replay.  The same function is used by the
fixture tests and by the optional real-cache audit command.

The causal teacher in this audit is only a geometry probe.  Its score is the
decision-bar ``return[t]``/close observation available before the fill at
``t+1``; it is not a promoted forecast and its PnL must not be read as an
accuracy result.  U0 is replayed separately from realized returns as an
upper-bound diagnostic.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unidream.data.availability_contract import validate_availability
from unidream.data.cache_v4 import (
    MODEL_FEATURE_COLUMNS,
    cache_v4_paths,
    load_cache_v4,
)
from unidream.data.window_quality import valid_sequence_starts
from unidream.data.oracle import (
    conditional_oracle_teacher_path,
    hindsight_upper_bound_path,
)
from unidream.eval.action_execution import (
    ActionExecutionContract,
    ActionExecutionTrajectory,
    complete_decision_starts,
    replay_action_path,
    replay_contract_absolute_path,
    select_block_decisions,
)


DEFAULT_CACHE_DIR = "/tmp/unidream-v4-p0-20260830"
DEFAULT_CACHE_TAG = "BTCUSDT_15m_2018-01-01_2024-01-01_z60_v4_official"
_REQUIRED_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS


@dataclass(frozen=True, slots=True)
class V4ContractInputs:
    """Cache values and full-grid masks consumed by the P0-C replay."""

    features: pd.DataFrame
    returns: pd.Series
    availability: pd.DataFrame
    metadata: dict[str, Any]
    metadata_path: Path
    metadata_file_sha256: str
    grid_index: pd.DatetimeIndex
    context_bars: int
    body_offsets: np.ndarray
    returns_grid: np.ndarray
    decision_eligible: np.ndarray
    score_eligible: np.ndarray
    body_row_eligible: np.ndarray


def load_v4_contract_inputs(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    cache_tag: str = DEFAULT_CACHE_TAG,
    *,
    context_bars: int = 64,
    metadata_path: str | Path | None = None,
) -> V4ContractInputs:
    """Load v4 and map P0-A eligibility onto the complete sidecar grid.

    ``load_cache_v4`` validates the feature body/returns/sidecar digests and
    rejects malformed metadata.  The body is intentionally sparse relative to
    the complete sidecar grid.  We therefore use NaN placeholders for missing
    body returns and false eligibility at those timestamps rather than
    reindexing and compacting the arrays away.
    """
    cache_paths = cache_v4_paths(cache_dir, cache_tag)
    selected_metadata_path = (
        Path(metadata_path) if metadata_path is not None else cache_paths["metadata"]
    )
    if metadata_path is None:
        features, returns, availability, metadata = load_cache_v4(cache_dir, cache_tag)
    else:
        features, returns, availability, metadata = load_cache_v4(
            feature_path=cache_paths["features"],
            returns_path=cache_paths["returns"],
            availability_path=cache_paths["availability"],
            metadata_path=selected_metadata_path,
        )
        if metadata.get("cache_tag") != cache_tag:
            raise ValueError(
                "explicit v4 metadata cache_tag does not match the requested cache"
            )
    metadata_file_sha256 = hashlib.sha256(
        selected_metadata_path.read_bytes()
    ).hexdigest()
    if tuple(str(column) for column in features.columns) != _REQUIRED_FEATURE_COLUMNS:
        raise ValueError("v4 feature body is not the canonical 17-column schema")
    if not isinstance(availability.index, pd.DatetimeIndex):
        raise ValueError("v4 availability index must be a DatetimeIndex")

    if isinstance(context_bars, (bool, np.bool_)) or not isinstance(
        context_bars, (int, np.integer)
    ) or int(context_bars) < 1:
        raise ValueError("context_bars must be a positive integer")
    context_bars = int(context_bars)
    selected = validate_availability(
        availability,
        features.index,
        include_funding=True,
        include_mark=True,
    )
    grid_index = availability.index
    body_offsets = grid_index.get_indexer(features.index)
    if np.any(body_offsets < 0):  # load_cache_v4 already checks this; keep the boundary explicit.
        raise ValueError("v4 feature body contains a timestamp outside the sidecar grid")
    if len(np.unique(body_offsets)) != len(body_offsets):
        raise ValueError("v4 body-to-sidecar mapping is not one-to-one")

    feature_values = features.to_numpy(dtype=np.float64)
    return_values = returns.to_numpy(dtype=np.float64)
    feature_finite = np.isfinite(feature_values).all(axis=1)
    return_finite = np.isfinite(return_values)
    body_row_eligible = np.asarray(selected.row_eligible, dtype=bool) & feature_finite

    # Both masks are full sidecar-grid vectors.  Decision eligibility is the
    # end row t of a canonical context window [t-context_bars+1, t].  This
    # uses the same P0-A gap/source-window checker as SequenceDataset and also
    # requires every model feature in the window to be finite.  A missing
    # body row therefore remains a gap instead of becoming a compacted row.
    context_starts = valid_sequence_starts(
        grid_index,
        context_bars,
        interval="15m",
        availability=availability,
        include_funding=True,
        include_mark=True,
    )
    feature_finite_grid = np.zeros(len(grid_index), dtype=bool)
    feature_finite_grid[body_offsets] = feature_finite
    feature_prefix = np.concatenate(([0], np.cumsum(feature_finite_grid.astype(np.int64))))
    decision_eligible = np.zeros(len(grid_index), dtype=bool)
    for context_start in context_starts:
        end = int(context_start) + context_bars
        if int(feature_prefix[end] - feature_prefix[int(context_start)]) == context_bars:
            decision_eligible[end - 1] = True
    score_eligible = np.zeros(len(grid_index), dtype=bool)
    # A score/outcome row only needs an observed Spot bar and a finite return.
    # Funding/mark availability affects the decision/context mask, not whether
    # an observed Spot return can be scored in the delayed P0-C block.
    spot_observed = availability["spot_bar_observed"].to_numpy(dtype=bool)
    interval_delta = pd.Timedelta(minutes=15)
    contiguous_row = np.ones(len(grid_index), dtype=bool)
    if len(grid_index) > 1:
        contiguous_edges = np.asarray(
            np.diff(grid_index.to_numpy()) == interval_delta,
            dtype=bool,
        )
        contiguous_row[1:] &= contiguous_edges
        contiguous_row[:-1] &= contiguous_edges
    score_eligible[body_offsets] = (
        spot_observed[body_offsets]
        & contiguous_row[body_offsets]
        & return_finite
    )
    returns_grid = np.full(len(grid_index), np.nan, dtype=np.float64)
    returns_grid[body_offsets] = return_values

    return V4ContractInputs(
        features=features,
        returns=returns,
        availability=availability,
        metadata=dict(metadata),
        metadata_path=selected_metadata_path,
        metadata_file_sha256=metadata_file_sha256,
        grid_index=grid_index,
        context_bars=context_bars,
        body_offsets=np.asarray(body_offsets, dtype=np.int64),
        returns_grid=returns_grid,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
        body_row_eligible=body_row_eligible,
    )


def _causal_probe_scores(
    returns_grid: np.ndarray,
    score_eligible: np.ndarray,
    contract: ActionExecutionContract,
) -> np.ndarray:
    """Create a deterministic score using only information available before t."""
    scores = np.zeros(len(returns_grid), dtype=np.float64)
    for start in complete_decision_starts(len(returns_grid), contract):
        # close/return[t] is observed by the end of decision bar t and is
        # therefore causal for the fill at t+1.  It is not the delayed future
        # outcome used for t+1..t+4 scoring.
        if bool(score_eligible[start]) and np.isfinite(returns_grid[start]):
            scores[start] = float(returns_grid[start])
    return scores


def _expected_scored_mask(
    n_bars: int,
    contract: ActionExecutionContract,
    score_eligible: np.ndarray,
) -> np.ndarray:
    expected = np.zeros(n_bars, dtype=bool)
    for start in complete_decision_starts(n_bars, contract):
        fill = start + contract.execution_delay_bars
        end = fill + contract.commitment_bars
        if bool(score_eligible[fill:end].all()):
            expected[fill:end] = True
    return expected


def _trajectory_pnl(trajectory: ActionExecutionTrajectory) -> float:
    return float(np.sum(trajectory.scored_pnl, dtype=np.float64))


def _path_summary(trajectory: ActionExecutionTrajectory) -> dict[str, Any]:
    return {
        "contract_hash": trajectory.contract_hash,
        "eligibility_mask_hash": trajectory.eligibility_mask_hash,
        "scored_bars": int(trajectory.n_scored_bars),
        "scorable_blocks": int(trajectory.n_scorable_blocks),
        "filled_blocks": int(trajectory.n_filled_blocks),
        "execution_skipped_blocks": int(trajectory.n_execution_skipped_blocks),
        "excluded_blocks": int(trajectory.n_excluded_blocks),
        "net_pnl": _trajectory_pnl(trajectory),
    }


def audit_contract_paths(
    inputs: V4ContractInputs,
    *,
    contract: ActionExecutionContract | None = None,
    run_u0: bool = True,
) -> dict[str, Any]:
    """Replay strategy, benchmark, U0 and causal teacher under one contract."""
    contract = contract or ActionExecutionContract.canonical()
    decision_eligible = inputs.decision_eligible
    score_eligible = inputs.score_eligible
    scores = _causal_probe_scores(inputs.returns_grid, score_eligible, contract)

    causal_teacher = conditional_oracle_teacher_path(
        scores,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    # ``ActionExecutionTrajectory.decision_deltas`` records the *effective*
    # clipped delta (for example -0.02 when a requested -0.08 reaches the
    # 0.50 floor).  The replay API accepts the registered request grid, so
    # retain the selector output for the actual-return strategy replay.
    selected_deltas = select_block_decisions(
        scores,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    strategy = replay_action_path(
        inputs.returns_grid,
        selected_deltas,
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    benchmark = replay_contract_absolute_path(
        inputs.returns_grid,
        np.full(len(inputs.grid_index), contract.p_start, dtype=np.float64),
        contract,
        decision_eligible=decision_eligible,
        score_eligible=score_eligible,
    )
    u0 = (
        hindsight_upper_bound_path(
            inputs.returns_grid,
            contract,
            decision_eligible=decision_eligible,
            score_eligible=score_eligible,
        )
        if run_u0
        else None
    )

    trajectories: dict[str, ActionExecutionTrajectory] = {
        "strategy": strategy,
        "benchmark": benchmark,
        "causal_teacher": causal_teacher,
    }
    if u0 is not None:
        trajectories["u0"] = u0
    contract_hashes = {name: path.contract_hash for name, path in trajectories.items()}
    mask_hashes = {name: path.eligibility_mask_hash for name, path in trajectories.items()}
    scored_masks = {name: path.scored_mask for name, path in trajectories.items()}
    reference_mask = strategy.scored_mask
    same_scored_mask = all(np.array_equal(reference_mask, value) for value in scored_masks.values())
    same_contract_hash = len(set(contract_hashes.values())) == 1
    same_mask_hash = len(set(mask_hashes.values())) == 1

    expected_mask = _expected_scored_mask(
        len(inputs.grid_index), contract, score_eligible
    )
    timeline_ok = bool(np.array_equal(reference_mask, expected_mask))
    timeline_sample = []
    starts = complete_decision_starts(len(inputs.grid_index), contract)
    sample_starts = tuple(dict.fromkeys(starts[:3] + starts[-3:]))
    for start in sample_starts:
        fill = start + contract.execution_delay_bars
        end = fill + contract.commitment_bars
        timeline_sample.append(
            {
                "decision_t": int(start),
                "fill_t_plus_1": int(fill),
                "returns_start": int(fill),
                "returns_end_exclusive": int(end),
                "score_block_eligible": bool(score_eligible[fill:end].all()),
                "scored_bars": int(reference_mask[fill:end].sum()),
            }
        )

    result: dict[str, Any] = {
        "contract_hash": contract.contract_hash,
        "contract": contract.to_dict(),
        "grid_rows": int(len(inputs.grid_index)),
        "feature_body_rows": int(len(inputs.features)),
        "body_row_eligible": int(inputs.body_row_eligible.sum()),
        "decision_context_bars": int(inputs.context_bars),
        "decision_eligible_rows": int(decision_eligible.sum()),
        "score_eligible_rows": int(score_eligible.sum()),
        "scheduled_decisions": int(len(starts)),
        "contract_path_counts": strategy.eligibility_counts,
        "contract_path_same_scored_mask": same_scored_mask,
        "contract_path_same_contract_hash": same_contract_hash,
        "contract_path_same_eligibility_mask_hash": same_mask_hash,
        "contract_path_contract_hashes": contract_hashes,
        "contract_path_eligibility_mask_hashes": mask_hashes,
        "timeline_ok": timeline_ok,
        "timeline_sample": timeline_sample,
        "u0_run": u0 is not None,
        "paths": {
            name: _path_summary(path) for name, path in trajectories.items()
        },
        "cache": {
            "cache_tag": inputs.metadata.get("cache_tag"),
            "metadata_path": str(inputs.metadata_path),
            "metadata_file_sha256": inputs.metadata_file_sha256,
            "schema_version": inputs.metadata.get("schema_version"),
            "schema_digest": inputs.metadata.get("schema_digest"),
            "source_provenance_digest": inputs.metadata.get("source_provenance_digest"),
            "content_digests": dict(inputs.metadata.get("content_digests", {})),
            "gap_count": int(len(inputs.metadata.get("gap_list", []))),
            "first_feature_timestamp": str(inputs.features.index[0]),
            "last_feature_timestamp": str(inputs.features.index[-1]),
            "first_grid_timestamp": str(inputs.grid_index[0]),
            "last_grid_timestamp": str(inputs.grid_index[-1]),
        },
        "causal_probe": {
            "score_source": "decision_time_spot_return[t]_close_observation",
            "uses_future_returns": False,
            "uses_u0": False,
        },
        "p0_status": {
            "p0_a": "passed",
            "p0_b": "partial",
            "p0_c": "passed" if same_scored_mask and same_contract_hash and same_mask_hash and timeline_ok else "failed",
        },
    }
    return result


def audit_v4_cache(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    cache_tag: str = DEFAULT_CACHE_TAG,
    *,
    context_bars: int = 64,
    run_u0: bool = True,
    frozen_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full-grid audit against one validated v4 cache."""
    inputs = load_v4_contract_inputs(
        cache_dir,
        cache_tag,
        context_bars=context_bars,
    )
    result = audit_contract_paths(inputs, run_u0=run_u0)
    result["cache"]["cache_dir"] = str(cache_dir)
    if frozen_metadata_path is not None:
        frozen_inputs = load_v4_contract_inputs(
            cache_dir,
            cache_tag,
            context_bars=context_bars,
            metadata_path=frozen_metadata_path,
        )
        local_digests = dict(inputs.metadata.get("content_digests", {}))
        frozen_digests = dict(frozen_inputs.metadata.get("content_digests", {}))
        result["frozen_metadata_validation"] = {
            "status": "v4_verified",
            "metadata_path": str(frozen_inputs.metadata_path),
            "metadata_file_sha256": frozen_inputs.metadata_file_sha256,
            "source_provenance_digest": frozen_inputs.metadata.get(
                "source_provenance_digest"
            ),
            "content_digests_equal_to_cache_local": local_digests == frozen_digests,
            "source_provenance_digest_equal_to_cache_local": inputs.metadata.get(
                "source_provenance_digest"
            )
            == frozen_inputs.metadata.get("source_provenance_digest"),
            "metadata_file_sha256_equal_to_cache_local": inputs.metadata_file_sha256
            == frozen_inputs.metadata_file_sha256,
            "body_and_sidecar_rows_equal_to_cache_local": (
                len(inputs.features) == len(frozen_inputs.features)
                and len(inputs.availability) == len(frozen_inputs.availability)
                and inputs.features.index.equals(frozen_inputs.features.index)
                and inputs.availability.index.equals(frozen_inputs.availability.index)
            ),
        }
    return result


def markdown_report(result: dict[str, Any]) -> str:
    """Render a compact human-readable artifact from an audit result."""
    cache = result["cache"]
    counts = result["contract_path_counts"]
    lines = [
        "# P0 integration audit",
        "",
        "This is a contract/data-integrity audit, not a market-accuracy result.",
        "",
        f"- P0-A: **{result['p0_status']['p0_a']}**",
        f"- P0-B: **{result['p0_status']['p0_b']}** (legacy full-WM conditional OOF runner remains disconnected)",
        f"- P0-C: **{result['p0_status']['p0_c']}**",
        "",
        "## Source and masks",
        "",
        f"- cache: `{cache.get('cache_dir')}` / `{cache.get('cache_tag')}`",
        f"- schema/content: v{cache.get('schema_version')}, schema `{cache.get('schema_digest')}`",
        f"- rows: feature body `{result['feature_body_rows']}`, complete grid `{result['grid_rows']}`",
        f"- P0-A body eligible `{result['body_row_eligible']}`; decision context `{result['decision_context_bars']}` bars / mask `{result['decision_eligible_rows']}`; Spot score mask `{result['score_eligible_rows']}`",
        f"- sidecar gaps: `{cache.get('gap_count')}`",
        f"- features digest: `{cache.get('content_digests', {}).get('features')}`",
        f"- returns digest: `{cache.get('content_digests', {}).get('returns')}`",
        f"- availability digest: `{cache.get('content_digests', {}).get('availability')}`",
        "",
        "## P0-C parity",
        "",
        f"- contract hash: `{result['contract_hash']}`",
        f"- scheduled/scorable/filled/skipped/excluded/scored bars: `{counts['scheduled_decisions']}` / `{counts['scorable_blocks']}` / `{counts['filled_blocks']}` / `{counts['execution_skipped_blocks']}` / `{counts['excluded_blocks']}` / `{counts['scored_bars']}`",
        f"- same scored mask: `{result['contract_path_same_scored_mask']}`; same contract hash: `{result['contract_path_same_contract_hash']}`; same mask hash: `{result['contract_path_same_eligibility_mask_hash']}`",
        f"- timeline `decision t -> fill t+1 -> returns t+1..t+4`: `{result['timeline_ok']}`",
        f"- U0 replayed: `{result['u0_run']}`; causal probe uses future returns: `{result['causal_probe']['uses_future_returns']}`",
        "",
        "## P0-B boundary",
        "",
        "The integrated fixture/OOF guard verifies same-row perturbation invariance and rejects hindsight inventory provenance. This does not promote the legacy full-WM path: chronological per-fold WM retraining, OOF normalizer/calibrator and student replay remain an explicit remediation gap.",
        "",
        "## Provenance",
        "",
        f"- source provenance digest: `{cache.get('source_provenance_digest')}`",
        f"- first/last feature timestamp: `{cache.get('first_feature_timestamp')}` / `{cache.get('last_feature_timestamp')}`",
        f"- first/last grid timestamp: `{cache.get('first_grid_timestamp')}` / `{cache.get('last_grid_timestamp')}`",
    ]
    frozen = result.get("frozen_metadata_validation")
    if frozen is not None:
        lines.extend(
            [
                "",
                "## Tracked frozen metadata cross-check",
                "",
                f"- frozen metadata: `{frozen['metadata_path']}`",
                f"- cache-local/frozen metadata SHA-256: `{cache.get('metadata_file_sha256')}` / `{frozen['metadata_file_sha256']}`",
                f"- content digests equal (body/returns/sidecar): `{frozen['content_digests_equal_to_cache_local']}`",
                f"- body/sidecar rows and indexes equal: `{frozen['body_and_sidecar_rows_equal_to_cache_local']}`",
                f"- source provenance digest equal: `{frozen['source_provenance_digest_equal_to_cache_local']}`",
                "The cache-local metadata and tracked frozen metadata validate the same data body and sidecar, but are not byte-identical and carry different source-provenance digests. Future P1 runs must pin explicit body paths and the tracked metadata path.",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--cache-tag", default=DEFAULT_CACHE_TAG)
    parser.add_argument(
        "--frozen-metadata-path",
        type=Path,
        help="optionally validate the same body/sidecar with a tracked frozen metadata file",
    )
    parser.add_argument(
        "--context-bars",
        type=int,
        default=64,
        help="canonical causal context length; use a smaller value only for fixtures",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument(
        "--skip-u0",
        action="store_true",
        help="skip the realized-future U0 replay for a faster mask-only audit",
    )
    args = parser.parse_args(argv)
    result = audit_v4_cache(
        args.cache_dir,
        args.cache_tag,
        context_bars=args.context_bars,
        run_u0=not args.skip_u0,
        frozen_metadata_path=args.frozen_metadata_path,
    )
    if args.output_json is not None:
        _write_json(args.output_json, result)
    report = markdown_report(result)
    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report, encoding="utf-8")
    if args.output_json is None and args.output_report is None:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI smoke command.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CACHE_TAG",
    "V4ContractInputs",
    "audit_contract_paths",
    "audit_v4_cache",
    "load_v4_contract_inputs",
    "main",
    "markdown_report",
]
