"""Export real BNB history-to-target parity cases from locked research results."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .alpha_dd_search import (Candidate, file_digest, load_bars, make_features,
                              rule_targets, validate_data_artifact, write_json)


def export(data_path: Path, result_dir: Path, output_dir: Path) -> dict:
    proof = validate_data_artifact(data_path, expected_symbol="BNBUSDT")
    qualification = json.loads((result_dir / "qualification.json").read_text())
    candidate = Candidate("trend", 90, 0.0, 1.12, 0.0)
    if qualification["selected_id"] != candidate.id or not qualification["minimum_target_pass"]:
        raise ValueError("only the locked qualified BNB candidate may be exported")
    rows = []
    for stage in ("historical", "fresh"):
        path = result_dir / f"{stage}.json"
        if file_digest(path) != qualification[f"{stage}_file_sha256"]:
            raise ValueError("confirmation result digest changed")
        rows += json.loads(path.read_text())["rows"][candidate.id]
    decisions = pd.DatetimeIndex(sorted({decision for row in rows for decision in
        (pd.Timestamp(row["start"]).ceil("h"),
         (pd.Timestamp(row["end"]) - pd.Timedelta(hours=1)).floor("h"))}))
    cutoff = max(pd.Timestamp(row["end"]) for row in rows)
    bars = load_bars(data_path, cutoff=str(cutoff))
    features = make_features(bars)
    signal = features.loc[decisions, "momentum_90"].to_numpy(float)
    target = rule_targets(candidate, features)[bars.index.get_indexer(decisions)]
    if not np.isfinite(signal).all() or len(set(target)) != 2:
        raise ValueError("real fixture must cover finite positive and negative regimes")
    earliest = decisions[0] - pd.Timedelta(minutes=15 * 8641)
    history = bars.loc[(bars.index >= earliest) & (bars.index < decisions[-1])]
    indices = history.index.get_indexer(decisions - pd.Timedelta(minutes=15)) + 1
    if (indices < 8641).any():
        raise ValueError("fixture history does not contain exact lagged endpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "research_parity.npz"
    if artifact.exists():
        raise ValueError("existing published parity fixture is immutable")
    np.savez_compressed(artifact, timestamp_ns=history.index.asi8,
                        close=history.close.to_numpy(float), decision_indices=indices,
                        expected_signal=signal, expected_target=target)
    metadata = {"kind": "alpha_trend_real_research_parity_v1", "candidate_id": candidate.id,
                "cases": len(decisions), "history_bars": len(history),
                "npz_sha256": file_digest(artifact), "source_data": proof,
                "qualification_sha256": file_digest(result_dir / "qualification.json"),
                "evaluator_sha256": file_digest(Path(__file__).with_name("alpha_dd_search.py")),
                "builder_sha256": file_digest(Path(__file__)),
                "decisions": [str(x) for x in decisions],
                "expected_target_values": sorted(set(target)),
                "scope": "real archived history -> locked rule target parity, not live-data or performance proof"}
    write_json(output_dir / "research_parity.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.data, args.results, args.output), indent=2))


if __name__ == "__main__":
    main()
