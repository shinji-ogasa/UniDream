"""Data-only, immutable monthly UM-kline acquisition for a later ablation.

Uses the existing strict D1 parser and official checksum verifier.  Retains raw
ZIPs, parsed monthly checkpoints, a source ledger and a full bar-open grid.
No features, labels, predictions or financial performance are computed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from unidream.data.d1_signed_flow import download_d1_kline_month


RAW_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume",
              "n_trades", "taker_buy_base", "taker_buy_quote")
INTERVAL = pd.Timedelta(minutes=15)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(payload)


def _parquet_once(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not pd.read_parquet(path).equals(frame):
            raise ValueError(f"immutable parsed artifact differs: {path}")
        return
    frame.to_parquet(path)


def _append_once(path: Path, record: dict) -> None:
    previous = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    matches = [row for row in previous if row["month"] == record["month"]]
    if matches:
        if matches != [record]:
            raise ValueError("source ledger revision differs; use a new acquisition")
        return
    with path.open("ab") as stream:
        stream.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())


def _check_frame(frame: pd.DataFrame, month: pd.Timestamp) -> None:
    if not set(RAW_FIELDS + ("bar_close_ts",)).issubset(frame.columns):
        raise ValueError("UM parser omitted required raw fields")
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None or
            frame.index.has_duplicates or not frame.index.is_monotonic_increasing):
        raise ValueError("UM raw bar-open timestamps must be unique, ordered and timezone-aware")
    end = month + pd.offsets.MonthBegin(1)
    if ((frame.index < month) | (frame.index >= end)).any():
        raise ValueError("parsed bar outside requested archive month")
    if np.any(frame.index.asi8 % INTERVAL.value):
        raise ValueError("off-grid UM bar cannot be remapped")
    expected_close = frame.index + INTERVAL - pd.Timedelta(milliseconds=1)
    if not pd.DatetimeIndex(frame.bar_close_ts).equals(expected_close):
        raise ValueError("UM inclusive close timestamp differs from the 15-minute contract")
    values = frame.loc[:, RAW_FIELDS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid raw numeric field")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("nonpositive raw price")


def run(config_path: Path, *, downloader=None) -> dict:
    """Acquire the configured fixed interval; valid prior months are reused."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if (config["source"] != "um_klines" or config["symbol"] != "BTCUSDT" or
            config["interval"] != "15m"):
        raise ValueError("this acquisition is restricted to UM BTCUSDT 15m")
    first = pd.Timestamp(config["start_month"] + "-01", tz="UTC")
    last = pd.Timestamp(config["end_month"] + "-01", tz="UTC")
    cutoff = pd.Timestamp(config["feature_decision_cutoff"])
    if last < first or cutoff.tzinfo is None or not first < cutoff <= last + pd.offsets.MonthBegin(1):
        raise ValueError("invalid registered month range/cutoff")
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).parents[1] / "data/d1_signed_flow.py",
               Path(__file__).parents[1] / "data/official_v4_sources.py"]
    identity = {"config": config, "config_sha256": sha256(config_path),
                "source_sha256": {p.name: sha256(p) for p in sources}}
    registration_path = out / "registration.json"
    if registration_path.exists():
        registration = json.loads(registration_path.read_text())
        if any(registration[key] != value for key, value in identity.items()):
            raise ValueError("immutable acquisition registration differs")
    else:
        registration = {**identity,
            "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "scope": "raw data acquisition only; no features, labels, model or performance",
            "live_causal_eligible": False}
        _write_once(registration_path, _json_bytes(registration))
    binding = sha256(registration_path)
    fetch = downloader or download_d1_kline_month
    frames, records = [], []
    ledger_path = out / "source_ledger.jsonl"
    for month in pd.date_range(first, last, freq="MS"):
        name = month.strftime("%Y-%m")
        checkpoint = out / "monthly" / f"{name}.parquet"
        record_path = out / "monthly" / f"{name}.json"
        raw_path = out / "raw" / "um_klines" / f"BTCUSDT-15m-{name}.zip"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["registration_sha256"] != binding:
                raise ValueError("monthly registration mismatch")
            if record["status"] == "available":
                if (sha256(raw_path) != record["response_sha256"] or
                        sha256(checkpoint) != record["monthly_parquet_sha256"]):
                    raise ValueError("retained monthly raw/parsed digest mismatch")
                frame = pd.read_parquet(checkpoint)
                _check_frame(frame, month)
                frames.append(frame)
            elif record["status"] != "unavailable_month":
                raise ValueError("failed source contract remains blocked")
        else:
            if raw_path.exists() or checkpoint.exists():
                raise ValueError("unregistered existing monthly payload; refusing overwrite")
            frame, source = fetch("um_klines", symbol="BTCUSDT", interval="15m", month=name,
                                  raw_dir=out / "raw", timeout=float(config["timeout_seconds"]),
                                  timestamp_unit="ms", quarantine_invalid_rows=False)
            record = {**source, "registration_sha256": binding}
            if source.get("http_status") == 404:
                if not frame.empty:
                    raise ValueError("404 archive unexpectedly returned data")
                record["status"] = "unavailable_month"
            else:
                if (source.get("http_status") != 200 or not source.get("checksum_verified") or
                        source.get("error") or frame.empty):
                    record["status"] = "failed_source_contract"
                    _write_once(record_path, _json_bytes(record))
                    _append_once(ledger_path, record)
                    raise ValueError(f"month {name} failed source/checksum/parser contract")
                _check_frame(frame, month)
                if (not raw_path.exists() or sha256(raw_path) != source.get("response_sha256") or
                        source.get("checksum_expected_sha256") != source.get("response_sha256")):
                    raise ValueError("retained ZIP does not match official checksum identity")
                _parquet_once(checkpoint, frame)
                record.update(status="available", raw_path=str(raw_path),
                              monthly_parquet_path=str(checkpoint),
                              monthly_parquet_sha256=sha256(checkpoint))
                frames.append(frame)
            _write_once(record_path, _json_bytes(record))
        _append_once(ledger_path, record)
        records.append(record)
        print(json.dumps({"event": "month_complete", "month": name, "status": record["status"],
                          "rows": record.get("parsed_rows", 0)}), flush=True)
    grid = pd.date_range(first, last + pd.offsets.MonthBegin(1), freq="15min",
                         inclusive="left", name="bar_open_ts")
    if frames:
        raw = pd.concat(frames).loc[:, list(RAW_FIELDS) + ["bar_close_ts"]].reindex(grid)
    else:
        raw = pd.DataFrame(np.nan, index=grid, columns=RAW_FIELDS)
        raw["bar_close_ts"] = pd.Series(pd.NaT, index=grid, dtype="datetime64[ns, UTC]")
    raw["decision_ts"] = grid + INTERVAL
    observed = raw.loc[:, RAW_FIELDS].notna().all(axis=1)
    before_cutoff = np.asarray(raw.decision_ts < cutoff)
    availability = pd.DataFrame({"um_bar_observed": observed,
        "feature_decision_before_cutoff": before_cutoff,
        "feature_eligible_observed_bar": observed & before_cutoff}, index=grid)
    data_path, availability_path = out / "um_15m.parquet", out / "um_15m_availability.parquet"
    _parquet_once(data_path, raw)
    _parquet_once(availability_path, availability)
    missing_months = [r["month"] for r in records if r["status"] == "unavailable_month"]
    sidecar = {"schema": "oracle-derivative-raw-v1", "symbol": "BTCUSDT", "source": "um_klines",
        "status": "complete" if observed.all() else "complete_with_gaps",
        "data_path": str(data_path), "data_sha256": sha256(data_path),
        "availability_path": str(availability_path), "availability_sha256": sha256(availability_path),
        "source_ledger_path": str(ledger_path), "source_ledger_sha256": sha256(ledger_path),
        "registration_path": str(registration_path), "registration_sha256": binding,
        "raw_fields": list(RAW_FIELDS), "rows": len(raw), "observed_rows": int(observed.sum()),
        "missing_rows": int((~observed).sum()), "requested_months": len(records),
        "available_months": len(records) - len(missing_months), "unavailable_months": missing_months,
        "bar_open_start": str(grid[0]), "bar_open_end_exclusive": str(grid[-1] + INTERVAL),
        "first_observed_bar_open": str(grid[observed][0]) if observed.any() else None,
        "last_observed_bar_open": str(grid[observed][-1]) if observed.any() else None,
        "feature_decision_cutoff_exclusive": str(cutoff),
        "feature_eligible_observed_rows": int((observed & before_cutoff).sum()),
        "retained_post_cutoff_observed_rows": int((observed & ~before_cutoff).sum()),
        "timestamp_semantics": {"index": "bar OPEN time, UTC",
            "bar_close_ts": "observed inclusive exchange close = bar open + 15 minutes - 1 millisecond",
            "decision_ts": "bar open + 15 minutes; clock retained even when observation absent",
            "feature_cutoff": "decision_ts strictly before registered cutoff; April raw tail retained only"},
        "missing_policy": "full grid; NaN raw observations and false observation mask; no interpolation",
        "live_causal_eligible": False, "model_fitting_performed": False,
        "raw_zip_payloads_retained": True}
    _write_once(out / "um_15m.sha256.json", _json_bytes(sidecar))
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    result = run(parser.parse_args().config)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
