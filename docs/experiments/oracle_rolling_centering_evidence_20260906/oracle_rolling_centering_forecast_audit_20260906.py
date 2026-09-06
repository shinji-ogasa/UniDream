"""Independent Stage14 arithmetic audit. Never execute before root authorizes.

No repository helper imports, fitting, calibration, or policy rollouts. The
original same-fold delay raw streams and explicitly cutoff-filtered Spot bars
independently reconstruct H, raw averages, anchor and fixed-weight forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GROUPS = ("technical", "perp_delay0")
FOLDS = tuple(range(5, 13))
NEW_MEANS = ("rolling_anchor", "technical_rolling", "perp_delay0_rolling")
SPOT = Path("/Users/sophie/Documents/UniDream/.worktrees/alpha-dd-goal/checkpoints/alpha_dd_data/spot_15m.parquet")
SPOT_SHA = "5e20e81e86f76b95d1301be7a8a366aa9ad78134f954ec8c9dbf83c0db1acf69"
CUTOFF = pd.Timestamp("2023-04-16T13:45:00Z")
DELAY = Path("codex_outputs/oracle_derivative_delay_v1")
PARENT = Path("codex_outputs/oracle_mean_reliability_decisions_v1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as stream:
        return {k: stream[k].copy() for k in stream.files}


def equal(left, right, label):
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        okay = a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    else:
        okay = a.shape == b.shape and np.array_equal(a, b)
    if not okay:
        raise AssertionError("exact mismatch: " + label)


def close(left, right, label, maxima):
    a, b = float(left), float(right)
    if not math.isfinite(a) or not math.isfinite(b):
        raise AssertionError("nonfinite selected " + label)
    delta = abs(a - b)
    maxima[label] = max(maxima.get(label, 0.), delta)
    if delta > 1e-14:
        raise AssertionError(f"numeric mismatch {label}: {a!r} != {b!r}")


def strict_index(values):
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    step = pd.Timedelta(minutes=15).value
    assert len(index) and not index.hasnans
    assert np.all(index.asi8 % step == 0) and np.all(np.diff(index.asi8) == step)
    return index


def audit(config_path, output):
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(cfg["output_dir"])
    reg, results, pre = read(out / "registration.json"), read(out / "results.json"), read(out / "preflight.json")
    assert reg["config"] == cfg and reg["config_sha256"] == sha(config_path)
    assert reg["preflight_sha256"] == cfg["preflight_sha256"] == sha(out / "preflight.json")
    assert results["registration_sha256"] == digest(reg)
    assert cfg["development_folds"] == list(FOLDS) and cfg["minimum_history_pairs"] == 64
    assert cfg["history_calendar_months"] == 3 and cfg["maturity_minutes"] == 375
    assert cfg["data_cutoff"] == "2023-04-16T13:45:00Z"
    bound = {}
    for mapping in (cfg["source_bindings"], pre["direct_source_bindings"], pre["source_artifact_bindings"]):
        for path, expected in mapping.items():
            assert path not in bound or bound[path] == expected
            bound[path] = expected
    for path, expected in bound.items():
        assert sha(path) == expected, path
    assert sha(SPOT) == SPOT_SHA
    # Predicate is applied by the parquet reader, before any later rows decode.
    spot = pd.read_parquet(SPOT, filters=[("bar_open_ts", "<", CUTOFF)])
    assert spot.index.tz is not None and spot.index.max() < CUTOFF
    spot = spot.reindex(pd.date_range(spot.index.min(), CUTOFF - pd.Timedelta(minutes=15), freq="15min"))
    observed = spot[["open", "high", "low", "close"]].notna().all(axis=1).to_numpy()
    opening, closing = spot.open.to_numpy(float), spot.close.to_numpy(float)
    # Independent canonical return and full next-24-bar observed support.
    label = np.full(len(spot), np.nan)
    for i in range(len(spot) - 24):
        if observed[i + 1:i + 25].all() and math.isfinite(opening[i + 1]) and opening[i + 1] > 0:
            with np.errstate(all="raise"):
                label[i] = np.log(closing[i + 24] / opening[i + 1])
    source_proofs, fold_checks, maxima = {}, [], {}
    total_histories, total_inference, total_score, endpoint_arrays = 0, 0, 0, 0
    parent_results = read(PARENT / "results.json")
    delay_results = read(DELAY / "results.json")
    for f in FOLDS:
        manifest_path = out / f"fold_{f}.json"
        manifest = read(manifest_path)
        assert manifest["registration_sha256"] == digest(reg)
        assert len(manifest["artifact_sha256"]) == 32
        for path, expected in manifest["artifact_sha256"].items():
            assert sha(path) == expected, path
        saved = {m: arrays(out / "forecasts" / f"fold{f}_{m}.npz") for m in NEW_MEANS}
        ref = arrays(PARENT / "forecasts" / f"fold{f}_technical_reliability.npz")
        current = strict_index(ref["timestamps"])
        infer, score = ref["inference_mask"], ref["score_support"]
        assert infer.dtype == score.dtype == np.dtype(bool)
        assert not np.any(score & ~infer)
        for m, stream in saved.items():
            assert set(stream) == set(ref)
            for key in ref:
                if key != "mu":
                    equal(stream[key], ref[key], m + " " + key)
            equal(np.isfinite(stream["mu"]), infer, m + " I")
        raw_pair, weights, history = {}, {}, None
        for g in GROUPS:
            cp, ep = DELAY / "calibration" / f"fold{f}_{g}.npz", DELAY / "forecasts" / f"fold{f}_{g}_raw.npz"
            for p in (cp, ep):
                assert str(p) in bound and sha(p) == bound[str(p)], str(p)
                source_proofs[str(p)] = sha(p)
            ca, ev = arrays(cp), arrays(ep)
            times = np.concatenate((ca["timestamps"], ev["timestamps"]))
            candidate_history = strict_index(times)
            if history is not None:
                equal(candidate_history.asi8, history.asi8, "same raw history")
            history = candidate_history
            equal(ev["timestamps"], current.asi8, "eval raw calendar")
            equal(ev["inference_mask"], infer, "original I")
            equal(ev["score_support"], score, "original O")
            raw_pair[g] = np.concatenate((ca["mu"], ev["mu"]))
            weight_path = PARENT / "weights" / f"fold{f}_{g}.json"
            weight_record = read(weight_path)
            assert weight_record == next(x for x in parent_results["fits"] if x["fold"] == f and x["group"] == g)
            assert str(weight_path) in bound and sha(weight_path) == bound[str(weight_path)]
            weights[g] = weight_record["fit"]["weight"]
            assert type(weights[g]) in (int, float) and 0 <= weights[g] <= 1
            record = next(x for x in delay_results["scores"] if x["fold"] == f and x["model_id"] == g + "_raw")
            proof = record["provenance"]
            assert proof["calibration"]["path"] == str(cp) and proof["calibration"]["sha256"] == sha(cp)
            assert record["forecast_sha256"] == sha(ep)
            model = proof["models"]["return"]
            assert sha(model["path"]) == model["sha256"]
            source_proofs[model["path"]] = model["sha256"]
        origin = np.isfinite(raw_pair[GROUPS[0]])
        equal(origin, np.isfinite(raw_pair[GROUPS[1]]), "shared origins")
        hloc = spot.index.get_indexer(history)
        assert np.all(hloc >= 0)
        y = label[hloc]
        origin_positions = np.flatnonzero(origin)
        locations = history.get_indexer(current)
        assert np.all(locations >= 0)
        trace = read(out / "rolling_traces" / f"fold{f}_shared_history.json")
        decisions = trace["decisions"]
        assert len(decisions) == int(infer.sum())
        support = next(s for s in pre["support"] if s["fold"] == f)
        counts, membership = [], []
        for row, i in zip(decisions, np.flatnonzero(infer)):
            t = current[i]
            lower = t - pd.DateOffset(months=3)
            # Select temporal candidates before reading their labels.
            candidates = [j for j in origin_positions if lower <= history[j] < t
                          and history[j] + pd.Timedelta(minutes=375) <= t]
            selected = [j for j in candidates if math.isfinite(float(y[j]))]
            n = len(selected)
            assert n >= 64 and row["history_count"] == n and row["reason"] == "available"
            timestamps = history[np.asarray(selected)]
            hsha = hashlib.sha256(timestamps.asi8.tobytes()).hexdigest()
            assert row["history_timestamp_sha256"] == hsha
            assert row["window_start"] == lower.isoformat() and row["window_end_exclusive"] == t.isoformat()
            assert row["decision_at"] == t.isoformat()
            assert row["maturity_limit_origin"] == (t - pd.Timedelta(minutes=375)).isoformat()
            assert row["oldest_origin"] == timestamps[0].isoformat()
            assert row["latest_origin"] == timestamps[-1].isoformat()
            assert row["latest_maturity"] == (timestamps[-1] + pd.Timedelta(minutes=375)).isoformat()
            assert row["forecast_history_count"] == len(candidates)
            assert row["mature_label_missing_count"] == len(candidates) - n
            counts.append(n); membership.append({"decision_at": t.isoformat(), "history_timestamp_sha256": hsha})
            anchor = math.fsum(float(y[j]) / n for j in selected)
            close(anchor, row["rolling_anchor"], "anchor_trace", maxima)
            close(anchor, saved["rolling_anchor"]["mu"][i], "anchor_npz", maxima)
            for g in GROUPS:
                raw_mean = math.fsum(float(raw_pair[g][j]) / n for j in selected)
                current_raw = float(raw_pair[g][locations[i]])
                assert origin[locations[i]] and row["weights"][g] == weights[g]
                forecast = anchor if weights[g] == 0 else anchor + weights[g] * (current_raw - raw_mean)
                close(raw_mean, row["raw_means"][g], "raw_mean", maxima)
                close(current_raw, row["current_raw"][g], "current_raw", maxima)
                close(forecast, row["forecasts"][g + "_rolling"], "forecast_trace", maxima)
                close(forecast, saved[g + "_rolling"]["mu"][i], "forecast_npz", maxima)
        assert digest(counts) == support["history_counts_sha256"]
        assert membership == support["history_membership"] and digest(membership) == support["history_membership_sha256"]
        for g in GROUPS:
            if weights[g] == 0:
                equal(saved[g + "_rolling"]["mu"], saved["rolling_anchor"]["mu"], "w0 endpoint")
                endpoint_arrays += 1
        total_histories += len(counts); total_inference += int(infer.sum()); total_score += int(score.sum())
        fold_checks.append({"fold": f, "histories": len(counts), "score_rows": int(score.sum()),
            "minimum_pairs": min(counts), "maximum_pairs": max(counts),
            "membership_sha256": digest(membership), "forecast_files": 3})
    assert total_histories == total_inference == 2586 and total_score == 2574
    report = {"status": "passed", "schema": "independent-rolling-centering-arithmetic-audit-v1",
        "config_sha256": sha(config_path), "registration_sha256": sha(out / "registration.json"),
        "results_sha256": sha(out / "results.json"), "preflight_sha256": sha(out / "preflight.json"),
        "audit_script_sha256": sha(__file__), "source_bindings_verified": len(bound),
        "source_raw_and_return_model_bindings": source_proofs,
        "spot_sha256": SPOT_SHA, "spot_predicate_cutoff_exclusive": CUTOFF.isoformat(),
        "spot_decoded_maximum": spot.index[-1].isoformat(), "histories_checked": total_histories,
        "inference_rows": total_inference, "score_rows": total_score, "new_forecast_npz_checked": 24,
        "zero_weight_full_array_identities": endpoint_arrays, "absolute_tolerance": 1e-14,
        "max_absolute_difference": maxima, "folds": fold_checks,
        "new_models_or_weights_fitted": False, "new_policy_rollouts": False,
        "interpretation": "Arithmetic and source/support consistency only; no new performance or generalization claim."}
    output = Path(output)
    assert not output.exists(), "immutable audit output already exists"
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "path": str(output), "sha256": sha(output),
                      "histories": total_histories, "max_absolute_difference": maxima}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorized-run", action="store_true", help="Use only after root's explicit go")
    parser.add_argument("--worktree", default="/Users/sophie/Documents/UniDream/.worktrees/oracle-feature-frontier-20260905")
    parser.add_argument("--config", default="configs/oracle_rolling_centering_decisions_20260906.yaml")
    parser.add_argument("--output", default="/tmp/oracle_rolling_centering_forecast_audit_20260906.json")
    args = parser.parse_args()
    if not args.authorized_run:
        parser.error("No real arithmetic allowed before root explicitly authorizes --authorized-run")
    os.chdir(args.worktree)
    audit(Path(args.config), Path(args.output))
