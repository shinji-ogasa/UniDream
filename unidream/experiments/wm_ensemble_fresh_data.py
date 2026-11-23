"""Acquire a genuinely new calendar period without changing frozen data.

The original conservative source-time lag and expiry rules are preserved.
Fresh labels are never used here for model fitting, selection or calibration.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from .wm_ablation_data import ArchiveJob, acquire_archive, parse_archive, causal_asof
from .wm_ablation_evaluation import build_future_targets
from .wm_technical_ablation import _context_mask, _json, _sha, market_features


def acquire(cfg: dict) -> dict:
    output = Path(cfg["output_dir"]) / "fresh_data"
    if (output / "manifest.json").exists():
        previous = json.loads((output / "manifest.json").read_text())
        if previous.get("status") == "complete":
            for filename, digest in previous["sha256"].items():
                if _sha(output / filename) != digest:
                    raise ValueError("fresh input changed")
            return previous
    output.mkdir(parents=True, exist_ok=True)
    start, end = (pd.Timestamp(cfg[k], tz="UTC") for k in ("fresh_start", "fresh_end_exclusive"))
    if start < pd.Timestamp(cfg["previously_evaluated_end_exclusive"], tz="UTC") or end <= start:
        raise ValueError("fresh period overlaps previously evaluated dates")
    if end > pd.Timestamp.now(tz="UTC").floor("D"):
        raise ValueError("only completed UTC days may be acquired")
    jobs = []
    endpoints = {"spot": "spot/daily/klines", "perp": "futures/um/daily/klines",
        "mark": "futures/um/daily/markPriceKlines", "index": "futures/um/daily/indexPriceKlines",
        "premium": "futures/um/daily/premiumIndexKlines"}
    for day in pd.date_range(start, end, freq="D", inclusive="left"):
        period = day.strftime("%Y-%m-%d")
        for source, endpoint in endpoints.items():
            jobs.append(ArchiveJob(source, period, f"https://data.binance.vision/data/{endpoint}/BTCUSDT/15m/BTCUSDT-15m-{period}.zip"))
        jobs.append(ArchiveJob("metrics", period, f"https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{period}.zip"))
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(lambda job: acquire_archive(job, output), jobs))
    _json(output / "acquisition_ledger.json", records)
    if any(not r.get("checksum_verified") for r in records):
        raise ValueError("fresh daily archives missing or checksum verification failed")
    by_source = {source: [] for source in (*endpoints, "metrics")}
    parsing = []
    for job, record in zip(jobs, records, strict=True):
        frame, metadata, conflicts = parse_archive(job, Path(record["archive_path"]).read_bytes())
        lo = pd.Timestamp(job.period, tz="UTC")
        if not conflicts.empty or ((frame.index < lo) | (frame.index >= lo + pd.Timedelta(days=1))).any():
            raise ValueError("conflicting rows or wrong daily interval")
        by_source[job.source].append(frame)
        parsing.append({"source": job.source, "period": job.period, **metadata})
    _json(output / "parse_ledger.json", parsing)
    original = Path(cfg["ablation_dir"]) / "data/base"
    old = pd.read_parquet(original / "raw_market.parquet")
    if old.index.max() >= start:
        raise ValueError("fresh period already exists in historical source data")
    grid = pd.date_range(start, end, freq="15min", inclusive="left", name="bar_open")
    fresh = pd.DataFrame(index=grid)
    availability = pd.DataFrame(index=grid)
    for source in endpoints:
        frame = pd.concat(by_source[source]).sort_index()
        if frame.index.has_duplicates:
            raise ValueError("duplicate fresh candle timestamps")
        fresh = fresh.join(frame.reindex(grid))
    metric_events = pd.concat([pd.read_parquet(original / "metrics_events.parquet"), *by_source["metrics"]]).sort_index()
    if metric_events.index.has_duplicates:
        raise ValueError("duplicate metrics timestamps")
    metrics, times = causal_asof(metric_events, grid, pd.Timedelta(minutes=30))
    fresh = fresh.join(metrics)
    availability["metrics_source_ts"] = times
    # Public funding REST endpoint; retain the exact response and request range.
    params = {"symbol": "BTCUSDT", "startTime": int((start - pd.Timedelta(days=1)).timestamp() * 1000),
              "endTime": int(end.timestamp() * 1000) - 1, "limit": 1000}
    response = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params=params, timeout=(10, 30))
    response.raise_for_status()
    (output / "funding_response.json").write_bytes(response.content)
    rows = response.json()
    if not isinstance(rows, list) or not rows or len(rows) >= 1000:
        raise ValueError("unexpected funding response or pagination required")
    events = pd.DataFrame(rows)
    if not events.symbol.eq("BTCUSDT").all():
        raise ValueError("wrong funding instrument")
    index = pd.to_datetime(events.fundingTime, unit="ms", utc=True)
    funding_events = pd.DataFrame({"funding_rate": pd.to_numeric(events.fundingRate).to_numpy()}, index=pd.DatetimeIndex(index, name="bar_open"))
    if funding_events.index.has_duplicates or funding_events.index.min() < start - pd.Timedelta(days=1) or funding_events.index.max() >= end:
        raise ValueError("invalid funding timestamps")
    funding, times = causal_asof(funding_events, grid, pd.Timedelta(hours=24))
    fresh = fresh.join(funding).reindex(columns=old.columns)
    availability["funding_source_ts"] = times
    if not ((availability.metrics_source_ts.dropna() <= availability.metrics_source_ts.dropna().index).all()
            and (availability.funding_source_ts.dropna() <= availability.funding_source_ts.dropna().index).all()):
        raise ValueError("noncausal as-of join")
    full_index = pd.date_range(old.index.min(), end, freq="15min", inclusive="left", name="bar_open")
    combined = pd.concat([old, fresh]).reindex(full_index)
    if not combined.loc[old.index].equals(old):
        raise ValueError("historical rows changed while extending inputs")
    raw = combined[[f"spot_{name}" for name in ("open", "high", "low", "close", "volume")]].to_numpy(dtype=float)
    market = market_features(combined)
    valid = np.isfinite(raw).all(1) & (raw[:, :4] > 0).all(1) & (raw[:, 4] >= 0) & np.isfinite(market).all(1)
    context_valid = _context_mask(valid, 256)
    targets = build_future_targets(combined.index, raw[:, 3], crash_threshold=.012)
    # Match the conservative 16h boundary purge of the original cohort.
    eligible = ((combined.index >= start) & (combined.index + pd.Timedelta(hours=cfg["purge_hours"]) < end)
                & context_valid & targets.mask.all(1))
    ids = np.flatnonzero(eligible)
    if len(ids) < 96:
        raise ValueError("fewer than one day of common fresh decision rows")
    np.savez(output / "arrays.npz", raw=raw, market=market, index_ns=combined.index.asi8,
             returns=targets.returns, variance=targets.variance, downside=targets.downside,
             crash=targets.crash, target_mask=targets.mask, ids=ids)
    fresh.to_parquet(output / "raw_market_new.parquet")
    availability.to_parquet(output / "availability_new.parquet")
    # Outcome distributions are deliberately not summarized before selection.
    manifest = {"status": "complete", "start": str(start), "end_exclusive": str(end), "n_eligible": len(ids),
        "first_decision": str(combined.index[ids[0]]), "last_decision": str(combined.index[ids[-1]]),
        "source_time_delay": "metrics/funding <= candle open; decision after candle close",
        "historical_publication_timestamps": "unknown; same conservative source-time assumptions as original experiment",
        "funding_request": {"url": response.url, "status": response.status_code, "params": params},
        "historical_raw_sha256": _sha(original / "raw_market.parquet"),
        "sha256": {name: _sha(output / name) for name in ("arrays.npz", "raw_market_new.parquet", "availability_new.parquet", "funding_response.json", "acquisition_ledger.json", "parse_ledger.json")}}
    _json(output / "manifest.json", manifest)
    print(json.dumps({"fresh_data": "complete", "n_eligible": len(ids), "first": manifest["first_decision"], "last": manifest["last_decision"]}), flush=True)
    return manifest


if __name__ == "__main__":
    config = yaml.safe_load(Path("configs/experiments/wm_classification_ensemble_20261110.yaml").read_text())
    acquire(config)
