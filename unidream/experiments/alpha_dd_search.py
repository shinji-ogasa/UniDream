"""Fixed, causal policy-family research for joint B&H alpha / drawdown targets.

This is a new experiment, not a change to the preregistered P1 experiment.
Configuration fixes periods, candidate universe, costs, and selection before any
historical confirmation result can be calculated. Only development selects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda fn: fn


BARS_DAY = 96
BARS_YEAR = BARS_DAY * 365
FEATURE_NAMES = tuple(
    [f"momentum_{d}" for d in (1, 7, 30, 90)]
    + [f"vol_{d}" for d in (1, 7, 30)]
    + [f"drawdown_{d}" for d in (7, 30, 90)]
    + ["vol_ratio", "flow_1", "flow_7"]
)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@dataclass(frozen=True)
class Candidate:
    family: str
    lookback: int = 30
    floor: float = 0.5
    ceiling: float = 1.12
    parameter: float = 0.0

    @property
    def id(self) -> str:
        return f"{self.family}_d{self.lookback}_lo{self.floor:g}_hi{self.ceiling:g}_p{self.parameter:g}"


def candidate_universe() -> list[Candidate]:
    items = [Candidate("hold", floor=1.0, ceiling=1.0)]
    for days in (7, 30, 90):
        for low in (0.0, 0.5, 0.8):
            for high in (1.0, 1.12):
                items.append(Candidate("trend", days, low, high))
    for target in (0.4, 0.6, 0.8):
        for low in (0.5, 0.8):
            for high in (1.0, 1.12):
                items.append(Candidate("volatility", 7, low, high, target))
    for days in (30, 90):
        for target in (0.6, 0.8):
            items.append(Candidate("trend_volatility", days, 0.5, 1.12, target))
        for threshold in (0.05, 0.10, 0.15):
            for low in (0.5, 0.8):
                for high in (1.0, 1.12):
                    items.append(Candidate("drawdown", days, low, high, threshold))
    for family in ("ridge", "hgb", "logistic"):
        for days in (7, 30):
            for low in (0.5, 0.8):
                for high in (1.0, 1.12):
                    items.append(Candidate(family, days, low, high, 2.0))
    assert len({x.id for x in items}) == len(items)
    return items


def load_bars(path: Path, *, cutoff: str) -> pd.DataFrame:
    bars = pd.read_parquet(path)
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("bar-open DatetimeIndex required")
    bars.index = pd.to_datetime(bars.index, utc=True)
    if bars.index.has_duplicates or not bars.index.is_monotonic_increasing:
        raise ValueError("duplicate or nonmonotone input bars")
    bars = bars.loc[bars.index < pd.Timestamp(cutoff)]
    if bars.empty:
        raise ValueError("no input bars in registered support")
    grid = pd.date_range(bars.index[0], bars.index[-1], freq="15min", tz="UTC")
    bars = bars.reindex(grid)
    required = ["open", "high", "low", "close", "quote_volume", "taker_buy_quote"]
    if not set(required).issubset(bars.columns):
        raise ValueError("raw official OHLC and signed quote-volume fields required")
    for col in required:
        if np.isinf(bars[col].to_numpy(dtype=float)).any():
            raise ValueError(f"infinite {col}")
    prices = bars[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError("nonpositive observed price")
    bars["bar_available"] = prices.notna().all(axis=1)
    return bars


def make_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Feature row t uses bars no later than t-1; no backfill or future masks."""
    log_price = np.log(bars["close"])
    returns = log_price.diff()
    out = pd.DataFrame(index=bars.index)
    for days in (1, 7, 30, 90):
        out[f"momentum_{days}"] = log_price.diff(days * BARS_DAY)
    for days in (1, 7, 30):
        # Require the whole volatility window; missing returns are not zero.
        out[f"vol_{days}"] = returns.rolling(days * BARS_DAY).std() * np.sqrt(BARS_YEAR)
    for days in (7, 30, 90):
        out[f"drawdown_{days}"] = np.expm1(log_price - log_price.rolling(days * BARS_DAY).max())
    out["vol_ratio"] = out["vol_1"] / out["vol_30"].clip(lower=1e-6)
    q = bars["quote_volume"].where(bars["quote_volume"] > 0)
    flow = (2 * bars["taker_buy_quote"] / q - 1).where(q.notna())
    out["flow_1"] = flow.rolling(BARS_DAY).mean()
    out["flow_7"] = flow.rolling(7 * BARS_DAY).mean()
    return out.shift(1)


def rule_targets(candidate: Candidate, features: pd.DataFrame) -> np.ndarray:
    c = candidate
    if c.family == "hold":
        return np.ones(len(features))
    if c.family in ("trend", "trend_volatility"):
        signal = features[f"momentum_{c.lookback}"].to_numpy()
        target = np.where(signal >= 0, c.ceiling, c.floor)
        target[~np.isfinite(signal)] = np.nan
        if c.family == "trend_volatility":
            vol = features.vol_7.to_numpy()
            target = np.minimum(target, np.clip(c.parameter / np.maximum(vol, 1e-6), c.floor, c.ceiling))
    elif c.family == "volatility":
        target = np.clip(c.parameter / np.maximum(features.vol_7.to_numpy(), 1e-6), c.floor, c.ceiling)
    elif c.family == "drawdown":
        dd = -features[f"drawdown_{c.lookback}"].to_numpy()
        # Smooth protection; fully reduced once the registered threshold is reached.
        target = c.ceiling - (c.ceiling - c.floor) * np.clip(dd / c.parameter, 0, 1)
    else:
        raise ValueError(f"not a rule candidate: {c.id}")
    return target


def forecast_targets(candidate: Candidate, predictions: np.ndarray,
                     features: pd.DataFrame) -> np.ndarray:
    if candidate.family == "logistic":
        signal = np.clip((predictions - 0.5) * candidate.parameter, -1, 1)
    else:
        risk = features.vol_7.to_numpy() * math.sqrt(candidate.lookback / 365)
        signal = np.tanh(candidate.parameter * predictions / np.maximum(risk, 0.01))
    return (1 + np.maximum(signal, 0) * (candidate.ceiling - 1)
            + np.minimum(signal, 0) * (1 - candidate.floor))


@njit(cache=True)
def _simulate(open_price, close_price, targets, schedule, cost,
              borrow_annual, max_step, deadband):
    """Self-financing cash/units account, delayed fills, and no free rebalancing."""
    n = len(open_price)
    equity = np.full(n, np.nan)
    exposure = np.full(n, np.nan)
    cash = 0.0
    units = 1.0 / open_price[0]  # Same pre-existing B&H inventory for both arms.
    turnover = 0.0
    fees = 0.0
    borrow = 0.0
    trades = 0
    for i in range(n):
        # Fill availability is observable open only. Unknown high/low/close must
        # never suppress an order that already filled at this bar's open.
        fill_available = np.isfinite(open_price[i])
        nav = cash + units * open_price[i] if fill_available else np.nan
        if fill_available and nav <= 0:
            return equity, exposure, turnover, fees, borrow, trades
        if fill_available and i > 0 and schedule[i - 1] and np.isfinite(targets[i - 1]):
            old_exposure = units * open_price[i] / nav
            desired = min(max(targets[i - 1], 0.0), 1.12)
            change = min(max(desired - old_exposure, -max_step), max_step)
            if abs(change) >= deadband:
                desired = old_exposure + change
                # Trade value x solves (old_value + x)/(nav - fee*abs(x)) = desired.
                x = (desired * nav - units * open_price[i]) / (1 + cost * desired * (1 if change > 0 else -1))
                fee = cost * abs(x)
                cash -= x + fee
                units += x / open_price[i]
                turnover += abs(x) / nav
                fees += fee
                trades += 1
        if cash < 0:
            charge = -cash * (math.exp(borrow_annual / BARS_YEAR) - 1)
            cash -= charge
            borrow += charge
        if not np.isfinite(close_price[i]):
            continue  # Carry filled units/cash; later observed prices mark the gap.
        equity[i] = cash + units * close_price[i]
        if equity[i] <= 0:
            return equity, exposure, turnover, fees, borrow, trades
        exposure[i] = units * close_price[i] / equity[i]
    return equity, exposure, turnover, fees, borrow, trades


def metrics(bars: pd.DataFrame, targets: np.ndarray, contract: dict) -> dict:
    if not len(bars) or not bars.bar_available.iloc[0] or not bars.bar_available.iloc[-1]:
        raise ValueError("evaluation boundary bars must be present; no boundary shifting")
    if len(targets) != len(bars):
        raise ValueError("target row alignment mismatch")
    finite = np.isfinite(targets)
    if ((targets[finite] < 0) | (targets[finite] > 1.12)).any():
        raise ValueError("target outside registered exposure bounds")
    # Fixed UTC hourly decisions; targets use only the previous completed bar.
    schedule = (bars.index.minute == 0).astype(bool)
    equity, positions, turnover, fees, borrow, trades = _simulate(
        bars.open.to_numpy(float), bars.close.to_numpy(float),
        np.asarray(targets, dtype=float), np.asarray(schedule),
        contract["one_way_cost"], contract["borrow_annual"], contract["max_step"], contract["deadband"],
    )
    if not np.isfinite(equity[-1]) or equity[-1] <= 0:
        raise ValueError("insolvency or missing terminal equity")
    observed = np.isfinite(bars.close.to_numpy(float))
    values = np.r_[1.0, equity[observed]]
    benchmark = np.r_[1.0, bars.close.to_numpy(float)[observed] / float(bars.open.iloc[0])]
    dd = float(np.max(1 - values / np.maximum.accumulate(values)))
    bh_dd = float(np.max(1 - benchmark / np.maximum.accumulate(benchmark)))
    return {
        "alpha_ex": float(values[-1] - benchmark[-1]), "maxdd_delta": dd - bh_dd,
        "total_return": float(values[-1] - 1), "bh_total_return": float(benchmark[-1] - 1),
        "maxdd": dd, "bh_maxdd": bh_dd, "turnover": float(turnover), "trades": int(trades),
        "fees_initial_equity_units": float(fees), "borrow_initial_equity_units": float(borrow),
        "mean_exposure": float(np.nanmean(positions)),
        "bar_coverage": float(bars.bar_available.mean()), "close_coverage": float(observed.mean()),
        "intent_coverage": float(finite.mean()), "rows": len(bars),
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("empty fold metrics cannot pass")
    alpha = np.array([x["alpha_ex"] for x in rows])
    dd = np.array([x["maxdd_delta"] for x in rows])
    a, d = float(alpha.mean()), float(dd.mean())
    rng = np.random.default_rng(7381)
    draws = rng.integers(0, len(rows), (5000, len(rows)))
    a_ci = np.quantile(alpha[draws].mean(axis=1), [0.025, 0.975]).tolist()
    d_ci = np.quantile(dd[draws].mean(axis=1), [0.025, 0.975]).tolist()
    return {
        "folds": len(rows), "alpha_ex_mean": a, "maxdd_delta_mean": d,
        "alpha_ex_median": float(np.median(alpha)), "maxdd_delta_median": float(np.median(dd)),
        "alpha_ex_worst": float(alpha.min()), "alpha_positive_folds": int((alpha > 0).sum()),
        "dd_improved_folds": int((dd < 0).sum()), "selection_score": min(a, -d),
        "minimum_target_pass": bool(a >= 0.01 and d <= -0.01),
        "preferred_target_pass": bool(a >= 0.03 and d <= -0.03),
        "descriptive_fold_bootstrap_alpha_ci": a_ci,
        "descriptive_fold_bootstrap_dd_ci": d_ci,
        "ci_scope": "descriptive iid fold bootstrap; not selection-adjusted proof",
    }


def fold_spec(fold: int, anchor: str) -> dict:
    start = pd.Timestamp(anchor) + pd.DateOffset(months=3 * fold)
    return {"fold": fold, "train_start": start - pd.DateOffset(years=2, months=3),
            "train_end": start - pd.DateOffset(months=3),
            "val_start": start - pd.DateOffset(months=3), "val_end": start,
            "test_start": start, "test_end": start + pd.DateOffset(months=3)}


def fit_predictions(candidate: Candidate, features: pd.DataFrame, bars: pd.DataFrame,
                    fold: dict, output: Path) -> tuple[np.ndarray, dict]:
    x = features.loc[:, FEATURE_NAMES].to_numpy(float)
    horizon = candidate.lookback * BARS_DAY
    # Decision t -> fill open[t+1] -> horizon realized at close[t+h].
    y = (np.log(bars.close.shift(-horizon)) - np.log(bars.open.shift(-1))).to_numpy()
    full_target = bars.bar_available.astype(int).rolling(horizon).sum().shift(-horizon).to_numpy() == horizon
    target_end = bars.index + pd.Timedelta(minutes=15 * (horizon + 1))
    train = ((bars.index >= fold["train_start"]) & (target_end < fold["train_end"]))
    train &= np.isfinite(x).all(axis=1) & np.isfinite(y) & full_target
    train &= np.arange(len(bars)) % 16 == 0
    indices = np.flatnonzero(train)
    if len(indices) < 256:
        raise ValueError(f"insufficient causal fit rows {candidate.id}: {len(indices)}")
    predict = ((bars.index >= fold["val_start"]) & (bars.index < fold["test_end"]))
    predict &= np.isfinite(x).all(axis=1)
    pred = np.full(len(bars), np.nan)
    if candidate.family == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        labels = y[train]
    elif candidate.family == "hgb":
        model = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=7,
            min_samples_leaf=64, l2_regularization=10.0, learning_rate=0.04,
            early_stopping=False, random_state=7)
        labels = y[train]
    elif candidate.family == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=7))
        labels = y[train] > 0
    else:
        raise ValueError(candidate.family)
    with threadpool_limits(limits=2):
        model.fit(x[train], labels)
        pred[predict] = (model.predict_proba(x[predict])[:, 1] if candidate.family == "logistic"
                         else model.predict(x[predict]))
    # Trusted local serialized fit, generated here, for reproducible export work.
    import joblib
    model_path = output / "models" / f"fold{fold['fold']}_{candidate.family}_h{candidate.lookback}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return pred, {"fit_rows": int(train.sum()), "predict_rows": int(predict.sum()),
                  "fit_last_feature_ts": str(bars.index[indices[-1]]),
                  "fit_last_target_end_exclusive": str(target_end[indices[-1]]),
                  "fit_cutoff": str(fold["train_end"]), "model_file": str(model_path),
                  "model_sha256": file_digest(model_path)}


def select_development(summary: dict[str, dict]) -> str:
    # Selection never reads historical/fresh confirmation files or results.
    return sorted(summary, key=lambda k: (-summary[k]["selection_score"],
                   -summary[k]["alpha_ex_mean"], summary[k]["maxdd_delta_mean"], k))[0]


def run(config_path: Path, stage: str) -> dict:
    config = yaml.safe_load(config_path.read_text())
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    universe = candidate_universe()
    registration = {"schema": "alpha-dd-research-v1", "config": config,
                    "candidates": [{"id": c.id, **asdict(c)} for c in universe],
                    "source_sha256": file_digest(Path(__file__))}
    reg_sha = digest(registration)
    registry_path = out / "registration.json"
    if registry_path.exists():
        if json.loads(registry_path.read_text()) != registration:
            raise ValueError("registration changed; use a new registered experiment output")
    else:
        if stage != "development":
            raise ValueError("development registration must precede confirmation")
        write_json(registry_path, registration)
    result_path = out / f"{stage}.json"
    if result_path.exists():
        raise ValueError("completed stage is immutable; inspect existing result")
    active = universe
    locked = None
    if stage != "development":
        lock_path = out / "selection_lock.json"
        locked = json.loads(lock_path.read_text())
        if locked["registration_sha256"] != reg_sha:
            raise ValueError("selection registration mismatch")
        dev_file = out / "development.json"
        if file_digest(dev_file) != locked["development_file_sha256"]:
            raise ValueError("development artifact modified after selection")
        active = [c for c in universe if c.id in (locked["selected_id"], universe[0].id)]
    # Development features are built from a physically truncated view.
    support = config["stages"][stage]
    path = Path(config["data_path"])
    bars = load_bars(path, cutoff=support["data_cutoff"])
    features = make_features(bars)
    rows: dict[str, list[dict]] = {c.id: [] for c in active}
    provenance = []
    for fold_id in support["folds"]:
        fold = fold_spec(fold_id, config["fold_anchor"])
        if fold["test_end"] > pd.Timestamp(support["data_cutoff"]):
            raise ValueError("fold exceeds stage data boundary")
        if fold["test_end"] - pd.Timedelta(minutes=15) > bars.index[-1]:
            raise ValueError("incomplete registered fold; cannot shorten test to make it pass")
        ix = np.asarray((bars.index >= fold["test_start"]) & (bars.index < fold["test_end"]))
        window = bars.loc[ix]
        if not len(window) or window.bar_available.mean() < config["minimum_bar_coverage"]:
            raise ValueError(f"fold {fold_id}: inadequate raw price coverage")
        predictions = {}
        for c in active:
            if c.family in ("ridge", "hgb", "logistic"):
                key = (c.family, c.lookback)
                if key not in predictions:
                    pred, proof = fit_predictions(c, features, bars, fold, out)
                    predictions[key] = pred
                    provenance.append({"fold": fold_id, "family": c.family, "horizon_days": c.lookback, **proof})
                target = forecast_targets(c, predictions[key], features)
            else:
                target = rule_targets(c, features)
            record = {"fold": fold_id, "start": str(fold["test_start"]), "end": str(fold["test_end"]),
                      **metrics(window, target[ix], config["execution"])}
            # Validation is an audit only in this fixed-candidate first experiment.
            # Candidate family and all sizing choices are selected on development folds.
            if stage != "development":
                stress = {**config["execution"], "one_way_cost": config["execution"]["one_way_cost"] * 2,
                          "borrow_annual": config["execution"]["borrow_annual"] * 2}
                record["stress_2x"] = metrics(window, target[ix], stress)
                np.savez_compressed(out / f"{stage}_fold{fold_id}_{c.id}.npz",
                                    timestamp_ns=window.index.asi8, targets=target[ix])
            rows[c.id].append(record)
        print(json.dumps({"event": "fold_complete", "stage": stage, "fold": fold_id,
                          "candidates": len(active)}), flush=True)
        write_json(out / f"{stage}_progress.json", {"registration_sha256": reg_sha, "folds_completed": len(next(iter(rows.values()))), "rows": rows})
    summary = {key: aggregate(values) for key, values in rows.items()}
    selected_id = select_development(summary) if stage == "development" else locked["selected_id"]
    result = {"stage": stage, "registration_sha256": reg_sha, "data_file_sha256": file_digest(path),
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": registration["source_sha256"], "selected_id": selected_id,
              "summary": summary, "rows": rows, "model_provenance": provenance,
              "formal_p1_result": False, "orders_submitted": 0,
              "candidate_count_development": len(universe),
              "confirmation_is_report_only": stage != "development"}
    write_json(result_path, result)
    if stage == "development":
        write_json(out / "selection_lock.json", {"registration_sha256": reg_sha,
                   "development_file_sha256": file_digest(result_path), "selected_id": selected_id,
                   "rule": "maximum min(mean AlphaEx, -mean MaxDDDelta); deterministic ties",
                   "selected_before_historical_or_fresh_results": True})
    print(json.dumps({"stage": stage, "selected_id": selected_id, "summary": summary[selected_id]}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "historical", "fresh"), required=True)
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
