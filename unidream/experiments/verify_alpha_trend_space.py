"""Verify a pinned public HF revision, both inference routes, and real Spot data."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import requests


SPACE = "https://shinjiaa-unidream-space.hf.space"
HUB = "https://huggingface.co/api/spaces/ShinjiAA/unidream-space"
SPOT = "https://data-api.binance.vision/api/v3/klines"


def verify(revision: str, bundle: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("smoke evidence is immutable; use a new output for a new attempt")
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    checks = []
    session = requests.Session()

    def call(method, path, *, expected=200, payload=None):
        started = perf_counter()
        response = session.request(method, SPACE + path, json=payload, timeout=(10, 60))
        body = response.json()
        record = {"method": method, "path": path, "status": response.status_code,
                  "elapsed_seconds": perf_counter() - started,
                  "response_sha256": hashlib.sha256(response.content).hexdigest(),
                  "body": body}
        if payload is not None:
            record["request_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        checks.append(record)
        print(json.dumps({"event": "http_check", "path": path, "status": response.status_code}), flush=True)
        if response.status_code != expected:
            raise ValueError(f"{path}: HTTP {response.status_code}, expected {expected}")
        return body

    initial = session.get(HUB, timeout=20).json()
    if initial.get("sha") != revision or initial.get("runtime", {}).get("sha") != revision:
        raise ValueError("HF repo and running revision must both match the pinned deployment")
    if initial["runtime"]["stage"] != "RUNNING":
        raise ValueError("HF revision is not running yet")
    health = call("GET", "/v2/health")
    assert health["ok"] and health["model_id"] == "bnb-trend90-20260905"
    assert health["schema"]["bundle_digest"] == manifest_digest
    assert health["research_evidence"]["minimum_target_pass"] is True
    assert health["live_performance"] is False
    parity = call("GET", "/v2/sample/verify")
    assert parity["strict_ok"] and parity["real_research_parity"]["n_cases"] == 20
    assert parity["real_research_parity"]["max_target_abs_diff"] == 0

    with np.load(bundle / "research_parity.npz", allow_pickle=False) as z:
        for target in (0.0, 1.12):
            case = int(np.flatnonzero(z["expected_target"] == target)[0])
            end = int(z["decision_indices"][case])
            stamps = pd.to_datetime(z["timestamp_ns"][end-8641:end], utc=True)
            closes = z["close"][end-8641:end]
            payload = {"symbol": "BNBUSDT", "interval": "15m",
                       "decision_ts": (stamps[-1] + pd.Timedelta(minutes=15)).isoformat(),
                       "candles": [{"timestamp": t.isoformat(), "close": float(c) if np.isfinite(c) else None}
                                   for t, c in zip(stamps, closes)]}
            result = call("POST", "/v2/predict", payload=payload)
            assert result["desired_target"] == target and not result["execution_intent"]["filled"]
            assert result["orders_submitted"] == 0 and not result["execution_ready"]
    call("POST", "/v2/predict", payload={**payload, "symbol": "BTCUSDT"}, expected=422)
    call("POST", "/v2/predict", payload={**payload, "decision_ts": "2100-01-01T00:00:00Z"}, expected=400)

    latest = call("GET", "/v2/predict/latest")
    proof = latest["data_provenance"]
    assert latest["symbol"] == "BNBUSDT" and proof["closed_bar_only"]
    assert proof["requires_derivatives"] is False and latest["orders_submitted"] == 0
    decision = pd.Timestamp(proof["decision_ts"])
    if datetime.now(timezone.utc) - decision.to_pydatetime() > pd.Timedelta(minutes=20):
        raise ValueError("live signal is stale")
    endpoint_closes = []
    for stamp in (decision - pd.Timedelta(minutes=15 * 8641), decision - pd.Timedelta(minutes=15)):
        ms = int(stamp.timestamp() * 1000)
        response = session.get(SPOT, params={"symbol": "BNBUSDT", "interval": "15m",
                               "startTime": ms, "endTime": ms, "limit": 1}, timeout=20)
        response.raise_for_status()
        row = response.json()[0]
        assert row[0] == ms
        endpoint_closes.append(float(row[4]))
    live_signal = float(np.log(endpoint_closes[1]) - np.log(endpoint_closes[0]))
    assert latest["desired_target"] == (1.12 if live_signal >= 0 else 0.0)
    assert abs(latest["meta"]["feature_value"] - live_signal) <= 1e-12
    legacy_health = call("GET", "/health")
    assert legacy_health["ok"]
    legacy = call("GET", "/sample/verify")
    assert legacy["strict_ok"] and legacy["live_default_strict_ok"]
    # Existing trusted local bundle, not an untrusted external pickle.
    with np.load(bundle.parent / "current" / "sample_input.npz", allow_pickle=True) as sample:
        old_payload = {"symbol": "BTCUSDT", "interval": "15m", "tail": 1,
                       "features": sample["features"][:64].astype(float).tolist()}
    old = call("POST", "/predict", payload=old_payload)
    assert len(old["positions"]) == 64 and np.isfinite(old["positions"]).all()
    final = session.get(HUB, timeout=20).json()
    assert final.get("sha") == revision and final.get("runtime", {}).get("sha") == revision
    evidence = {"complete": True, "checked_at": datetime.now(timezone.utc).isoformat(),
                "space_url": SPACE, "deployment_revision": revision, "bundle_digest": manifest_digest,
                "runtime_before": initial["runtime"], "runtime_after": final["runtime"],
                "independent_live_endpoint_closes": endpoint_closes,
                "independent_live_momentum": live_signal, "checks": checks,
                "orders_submitted": 0,
                "scope": "public inference and parity only; no new economic test or live trading"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "output": str(output), "checks": len(checks)}), flush=True)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    verify(args.revision, args.bundle, args.output)


if __name__ == "__main__":
    main()
