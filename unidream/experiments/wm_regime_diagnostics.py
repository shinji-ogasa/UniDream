"""Posthoc regime, error, overfitting and short-window diagnostics.

Read-only with respect to all prior models, selectors, thresholds and input
artifacts. Correlations and historical-window ranks are descriptive; neither
are causal effects or calibrated significance tests. Realized future variance
describes a market period, while past variance alone defines causal regimes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.special import softmax
import yaml

from .wm_ablation_features import BASE_COLUMNS, TA_COLUMNS
from .wm_classification_complementarity import MODELS, file_hash
from .wm_classification_evaluation import score_classification, paired_classification_block_bootstrap
from .wm_technical_ablation import MARKET_COLUMNS, _features, _json
from .wm_volatility_classification import get_design, tabular_design, tabular_scaler


DEFAULT_CONFIG = Path("configs/experiments/wm_regime_diagnostics_20261110.yaml")


def load_config(path=DEFAULT_CONFIG):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg["analysis_role"] != "posthoc_descriptive" or cfg["test_selection_allowed"] is not False
            or cfg["model_refitting"] is not False or cfg["condition"] != "fixed"
            or tuple(cfg["models"]) != MODELS or cfg["latest_fold"] != 14):
        raise ValueError("read-only fixed-threshold diagnostic contract required")
    cfg["config_path"] = str(Path(path).resolve())
    return cfg


def past_variance(close):
    """Variance known at decision t: squared returns through candle t-1."""
    close = np.asarray(close, dtype=float)
    log_return = np.r_[np.nan, np.diff(np.log(np.where(close > 0, close, np.nan)))]
    return pd.Series(log_return ** 2).rolling(16, min_periods=16).sum().shift(1).to_numpy()


def decision_metrics(y, pred):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    if y.ndim != 1 or pred.shape != y.shape or not len(y) or not np.isin(y, (0, 1, 2)).all() or not np.isin(pred, (0, 1, 2)).all():
        raise ValueError("valid three-class decision vectors required")
    cm = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.full(3, np.nan), where=support > 0)
    return {"balanced_accuracy": float(recall.mean()) if (support > 0).all() else None,
            "accuracy": float(np.trace(cm) / len(y)), "n": len(y),
            **{f"recall_{name}": float(recall[c]) if support[c] else None for c, name in enumerate(("down", "flat", "up"))},
            **{f"actual_{name}": float(support[c] / len(y)) for c, name in enumerate(("down", "flat", "up"))},
            **{f"predicted_{name}": float(cm.sum(0)[c] / len(y)) for c, name in enumerate(("down", "flat", "up"))},
            **{f"support_{name}": int(support[c]) for c, name in enumerate(("down", "flat", "up"))}}


def model_scores(panel, models=MODELS):
    """Pool matched seed panels exactly as in the preceding experiments."""
    result = []
    y = panel.actual.to_numpy(dtype=int)
    for model in models:
        p = panel[[f"p{c}_{model}" for c in range(3)]].to_numpy(dtype=float)
        p /= p.sum(1, keepdims=True)
        priors = panel[[f"train_prior_{c}" for c in range(3)]].to_numpy()
        pred, raw = np.argmax(p / priors, axis=1), p.argmax(1)
        score = decision_metrics(y, pred)
        raw_score = decision_metrics(y, raw)
        result.append({"model": model, **score, "n_unique_timestamps": int(panel.timestamp.nunique()),
            "n_seeds": int(panel.seed.nunique()), "raw_balanced_accuracy": raw_score["balanced_accuracy"],
            "raw_accuracy": raw_score["accuracy"], "log_loss": float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()),
            "brier": float(np.square(p - np.eye(3)[y]).sum(1).mean()),
            "mean_primary_margin": float(np.sort(p / priors, axis=1)[:, -1:].mean() - np.sort(p / priors, axis=1)[:, -2:-1].mean())})
    return result


def volatility_summary(arrays, ids, past):
    ids = np.asarray(ids, dtype=int)
    if not len(ids) or not np.isfinite(arrays["variance"][ids, 1]).all() or not np.isfinite(past[ids]).all():
        raise ValueError("finite common volatility cohort required")
    return {"n_unique_timestamps": len(ids),
            "realized_4h_vol_rms": float(np.sqrt(np.mean(arrays["variance"][ids, 1]))),
            "past_4h_vol_rms": float(np.sqrt(np.mean(past[ids]))),
            "net_4h_return_rms": float(np.sqrt(np.mean(arrays["returns"][ids, 1] ** 2))),
            "realized_4h_vol_median": float(np.median(np.sqrt(arrays["variance"][ids, 1]))),
            "past_4h_vol_median": float(np.median(np.sqrt(past[ids])))}


def contiguous_windows(timestamps, length, stride):
    """Return row slices; never compress gaps or silently cross a segment."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    if length < 1 or stride < 1 or timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("positive window sizes and unique chronological timestamps required")
    breaks = np.r_[0, np.flatnonzero(np.diff(timestamps.asi8) != pd.Timedelta(minutes=15).value) + 1, len(timestamps)]
    return [slice(start, start + length) for lo, hi in zip(breaks[:-1], breaks[1:]) for start in range(lo, hi - length + 1, stride)]


def correlation_summary(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n_folds": len(x), "spearman": None, "pearson": None, "leave_one_out_min": None, "leave_one_out_max": None}
    deleted = [float(spearmanr(np.delete(x, i), np.delete(y, i)).statistic) for i in range(len(x))]
    return {"n_folds": len(x), "spearman": float(spearmanr(x, y).statistic), "pearson": float(pearsonr(x, y).statistic),
            "leave_one_out_min": min(deleted), "leave_one_out_max": max(deleted)}


def feature_name(index):
    if not 0 <= index < 4909:
        raise ValueError("unexpected tabular feature index")
    if index < 4864:
        position, column = divmod(index, 19)
        name = (*BASE_COLUMNS, *MARKET_COLUMNS)[column]
        return name, 256 - position, "OHLCV" if column < 5 else "market"
    if index < 4908:
        return TA_COLUMNS[index - 4864], 1, "technical"
    return "threshold_side", 0, "threshold"


def distribution_shift(train, observed, bins=10):
    train, observed = np.asarray(train, float), np.asarray(observed, float)
    if not len(train) or not len(observed) or not np.isfinite(train).all() or not np.isfinite(observed).all():
        raise ValueError("finite nonempty feature samples required")
    edges = np.r_[-np.inf, np.unique(np.quantile(train, np.arange(1, bins) / bins)), np.inf]
    a, b = np.histogram(train, edges)[0], np.histogram(observed, edges)[0]
    a, b = (a + .5) / (len(train) + .5 * len(a)), (b + .5) / (len(observed) + .5 * len(b))
    q01, q99 = np.quantile(train, [.01, .99])
    return {"psi": float(np.sum((b - a) * np.log(b / a))),
        "outside_train_range": float(np.mean((observed < train.min()) | (observed > train.max()))),
        "outside_train_01_99": float(np.mean((observed < q01) | (observed > q99))),
        "median": float(np.median(observed)), "train_median": float(np.median(train))}


def train_probes(cfg, folds, old_arrays, fresh_arrays, panels):
    """Inference only, on predetermined evenly-spaced original Train rows."""
    from lightgbm import Booster
    source = Path(cfg["classification_dir"])
    rows, selections, probes = [], [], {}
    cfg_design = {"output_dir": str(source)}
    for fold in folds:
        f = fold["fold"]
        data = dict(np.load(source / "prepared" / f"fold_{f:02d}.npz"))
        selected = data["train"][np.linspace(0, len(data["train"]) - 1, min(cfg["classical_train_probe_rows"], len(data["train"])), dtype=int)]
        design = get_design(cfg_design, selected, data)
        theta = data["theta_fixed"][selected]
        x = np.column_stack([design, np.log(theta / fold["return_rms"]).astype(np.float32)])
        y, prior = data["label_fixed"][selected], np.array(fold["priors"]["fixed"])
        directory = source / "classical" / f"fold_{f:02d}_fixed"
        fit = json.loads((directory / "fit.json").read_text())
        for name, digest in fit["artifact_sha256"].items():
            if file_hash(directory / name) != digest:
                raise ValueError("frozen classical artifact changed")
        booster = Booster(model_file=str(directory / "lightgbm.txt"))
        logistic = pickle.loads((directory / "logistic.pkl").read_bytes())
        logits = np.stack([(x * row).sum(1) + bias for row, bias in zip(logistic.coef_, logistic.intercept_)], axis=1)
        for name, p in (("logistic", softmax(logits, axis=1)), ("lightgbm", booster.predict(x, num_threads=cfg["cpu_threads"]))):
            s = score_classification(y, p, prior)
            rows.append({"fold": f, "model": name, "split": "train_probe_in_sample", "n": len(selected),
                "balanced_accuracy": s["balanced_accuracy"], "log_loss": s["log_loss"], "accuracy": s["accuracy"]})
        choice = fit["lightgbm"]["selected"]
        selections.append({"fold": f, "num_leaves": choice["num_leaves"], "selected_iteration": choice["selected_iteration"],
            "selector_panel_ba": choice["balanced_accuracy"], "selector_panel_log_loss": choice["log_loss"]})
        if f == cfg["latest_fold"]:
            gain = booster.feature_importance(importance_type="gain")
            indices = np.argsort(-gain)[:cfg["feature_shift_top_gain_features"]]
            probes = {"train": x[:, indices], "indices": indices, "gain": gain[indices], "total_gain": gain.sum(),
                      "gain_all": gain}
            mean, std = tabular_scaler(data)
            for split, arrays in (("val", old_arrays), ("test", old_arrays), ("fresh", fresh_arrays)):
                panel = panels[f, split].loc[lambda v: v.seed == cfg["seeds"][0]].sort_values("timestamp")
                ids = panel.decision_id.to_numpy(dtype=int)
                if split == "fresh":
                    z = (tabular_design(_features(arrays["raw"], arrays["market"], ids, 256)) - mean) / std
                    z = np.column_stack([z, np.full(len(ids), np.log(fold["calibration"]["fixed"]["fixed_theta"] / fold["return_rms"]), dtype=np.float32)])
                else:
                    z = get_design(cfg_design, ids, data)
                    z = np.column_stack([z, np.log(data["theta_fixed"][ids] / fold["return_rms"]).astype(np.float32)])
                probes[split] = z[:, indices]
        print(f"Train inference probe fold {f:02d} complete", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(selections), probes


def make_report(out, tables, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"logistic": "Logistic", "lightgbm": "LightGBM", "direct": "Classifier Transformer", "wm": "WM"}
    scores, vol = tables["fold_model_diagnostics"], tables["fold_volatility"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS):
        data = scores.query("split == 'test' and model == @model").merge(vol.query("split == 'test'"), on="fold", suffixes=("", "_market"))
        x, y = data.relative_past_vol, 100 * data.balanced_accuracy
        axis.scatter(x, y, color="steelblue")
        for _, row in data.iterrows():
            axis.annotate(str(int(row.fold)), (row.relative_past_vol, 100 * row.balanced_accuracy), fontsize=8, xytext=(3, 3), textcoords="offset points")
        fresh_v = vol.query("split == 'fresh'").iloc[0]
        fresh_s = scores.query("split == 'fresh' and model == @model").iloc[0]
        axis.scatter([fresh_v.relative_past_vol], [100 * fresh_s.balanced_accuracy], marker="*", s=160, color="darkorange", label="November (short)")
        axis.set_title(names[model] + f" / Spearman r={spearmanr(x,y).statistic:.2f}")
        axis.set_xlabel("Past 4h volatility / Train RMS volatility")
        axis.set_ylabel("Balanced accuracy (%)")
        axis.legend(fontsize=8)
    fig.suptitle("15 walk-forward Test quarters / fixed threshold\nLabels are fold IDs; November excluded from correlations")
    fig.savefig(out / "fold_volatility_vs_ba.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    window = tables["windows"].query("window_kind == 'daily_rolling' and fold == 14")
    for axis, model in zip(axes.flat, MODELS):
        values = 100 * window.query("model == @model").balanced_accuracy.dropna()
        actual = 100 * scores.query("split == 'fresh' and model == @model").iloc[0].balanced_accuracy
        axis.hist(values, bins=15, color="steelblue", alpha=.8)
        axis.axvline(actual, color="darkorange", linewidth=2, label=f"November {actual:.2f}")
        axis.set_title(names[model])
        axis.set_xlabel("BA on 736-row historical windows (%)")
        axis.legend()
    fig.suptitle("Same fitted checkpoints: July-September windows vs November\nRolling windows overlap; histogram is descriptive, not a P-value")
    fig.savefig(out / "short_window_reference.png", dpi=180)
    plt.close(fig)
    html = ['<!doctype html><meta charset="utf-8"><title>Regime diagnostics</title>',
        '<style>body{font:16px sans-serif;max-width:1250px;margin:30px auto;padding:0 16px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:6px}img{max-width:100%}</style>',
        '<h1>LightGBMのBA低下：15fold・レジーム・短期変動の診断</h1>',
        '<p>事後的な記述分析。モデル・重み・閾値・採用設定は変更していない。相関、入力分布差、Trainとの乖離だけで原因を断定しない。</p>',
        '<p>主レジーム量は予測時点までの過去16本のリターン二乗和の平方根。Trainの同じ量のRMSで規格化する。0.75以下/0.75超1.25以下/1.25超をquiet/normal/highとする。未来16本の実現分散も期間の説明用に保存し、予測に使える過去指標とは分ける。</p>',
        '<p>4h実現ボラの期間代表値はsqrt(mean(sum(future 16 returns^2)))。4h累積リターンのRMSとは別。相関は15個のTest quarterだけから計算し、11月を混ぜない。15点での因果や有意性は主張せず、1fold除外時の感度も記録。</p>',
        '<img src="fold_volatility_vs_ba.png">',
        '<h2>同じ最終foldのモデル：Val→Q3→11月</h2>',
        scores.query("fold == 14").to_html(index=False),
        '<h2>15fold TestのボラティリティとBA</h2>', tables["fold_overview"].round(5).to_html(index=False),
        '<h2>ボラとBAの相関（記述的）</h2>', tables["correlations"].round(5).to_html(index=False),
        '<h2>短い期間の参照分布</h2>', '<img src="short_window_reference.png">',
        '<p>736行、1日ずつずらす窓は互いに重複する。非重複decision窓には4h教師ラベル用の16行gapを加える。どちらの順位も「偶然である確率」ではない。全期間ではfoldごとに学習モデルが異なるため、同じ最終fold内の結果も併記。</p>',
        tables["window_reference"].round(5).to_html(index=False),
        '<h2>Train内プローブと汎化差</h2>', tables["generalization_gaps"].round(5).to_html(index=False),
        '<p>Trainは元モデルが学習したデータの4096等間隔行。再学習ではない。TrainとVal/Testの差には過学習だけでなく市場分布の違いも含む。</p>',
        '<h2>重要特徴の分布変化（最終fold）</h2>', tables["feature_shift_summary"].round(5).to_html(index=False),
        '<p>上位64特徴は学習済みLightGBMのgainだけで選ぶ。PSIのbinはTrainプローブだけで作り、各binに0.5の疑似件数を加える。分布差やgainは誤差の原因を証明しない。特徴量の全一覧はCSV。</p>',
        '<p><a href="fold_overview.csv">15fold一覧</a> / <a href="fold_model_diagnostics.csv">クラス別・raw判定を含む指標</a> / <a href="windows.csv">全短期窓</a> / <a href="feature_shift.csv">重要特徴の分布差</a> / <a href="fresh_pairwise_ci.csv">11月のモデル差CI</a></p>']
    (out / "report.html").write_text("\n".join(html))


def run(path):
    cfg = load_config(path)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    source, ensemble = Path(cfg["classification_dir"]), Path(cfg["ensemble_dir"])
    paths = [Path(cfg["config_path"]), Path(__file__), Path(cfg["aligned_predictions"]),
        source / "prepared/folds.json", ensemble / "fresh_base_predictions.parquet", ensemble / "fresh_predictions.parquet",
        Path(cfg["ablation_dir"]) / "prepared/arrays.npz", ensemble / "fresh_data/arrays.npz"]
    paths += sorted((source / "prepared").glob("fold_*.npz"))
    paths += [p for p in (source / "classical").glob("fold_*_fixed/*") if p.suffix in (".json", ".pkl", ".txt")]
    hashes = {str(p): file_hash(p) for p in paths}
    _json(out / "manifest.json", {"status": "running", "analysis_role": "posthoc_descriptive", "config": cfg,
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes})
    old_arrays = dict(np.load(Path(cfg["ablation_dir"]) / "prepared/arrays.npz"))
    fresh_arrays = dict(np.load(ensemble / "fresh_data/arrays.npz"))
    past = past_variance(old_arrays["raw"][:, 3])
    fresh_past = past_variance(fresh_arrays["raw"][:, 3])
    assert np.allclose(past, old_arrays["baseline_variance"][:, 1], equal_nan=True, atol=1e-12)
    wide = pd.read_parquet(cfg["aligned_predictions"])
    fresh = pd.read_parquet(ensemble / "fresh_base_predictions.parquet")
    folds = json.loads((source / "prepared/folds.json").read_text())
    panels, score_rows, vol_rows, window_rows, regime_rows = {}, [], [], [], []
    length = int(fresh.timestamp.nunique())
    train_vol = {}
    for fold in folds:
        f = fold["fold"]
        with np.load(source / "prepared" / f"fold_{f:02d}.npz") as data:
            train_vol[f] = volatility_summary(old_arrays, data["train"], past)
        vol_rows.append({"fold": f, "split": "train", **train_vol[f], "relative_past_vol": 1., "relative_realized_vol": 1., "relative_net_return_rms": 1.})
        splits = ("val", "test", "fresh") if f == cfg["latest_fold"] else ("val", "test")
        for split in splits:
            frame = fresh.copy() if split == "fresh" else wide.loc[(wide.fold == f) & (wide.split == split)].copy()
            frame = frame.sort_values(["timestamp", "seed"])
            if frame.duplicated(["timestamp", "seed"]).any() or set(frame.seed) != set(cfg["seeds"]):
                raise ValueError("cohort/seed alignment failure")
            panels[f, split] = frame
            base = frame.loc[frame.seed == cfg["seeds"][0]].sort_values("timestamp")
            ids = base.decision_id.to_numpy(dtype=int)
            arrays, pv = (fresh_arrays, fresh_past) if split == "fresh" else (old_arrays, past)
            market = volatility_summary(arrays, ids, pv)
            market.update(relative_past_vol=market["past_4h_vol_rms"] / train_vol[f]["past_4h_vol_rms"],
                          relative_realized_vol=market["realized_4h_vol_rms"] / train_vol[f]["realized_4h_vol_rms"],
                          relative_net_return_rms=market["net_4h_return_rms"] / train_vol[f]["net_4h_return_rms"])
            vol_rows.append({"fold": f, "split": split, **market})
            score_rows.extend({"fold": f, "split": split, **score} for score in model_scores(frame))
            ratio = np.sqrt(pv[frame.decision_id.to_numpy(dtype=int)]) / train_vol[f]["past_4h_vol_rms"]
            regimes = np.digitize(ratio, cfg["volatility_regime_boundaries"], right=True)
            for i, name in enumerate(("quiet", "normal", "high")):
                part = frame.loc[regimes == i]
                if len(part):
                    regime_rows.extend({"fold": f, "split": split, "regime": name, **score} for score in model_scores(part))
            if split != "test":
                continue
            for kind, stride in (("daily_rolling", cfg["rolling_stride_bars"]), ("disjoint", length + cfg["disjoint_label_gap_bars"])):
                for section in contiguous_windows(base.timestamp, length, stride):
                    cohort = base.iloc[section]
                    part = frame.loc[(frame.timestamp >= cohort.timestamp.iloc[0]) & (frame.timestamp <= cohort.timestamp.iloc[-1])]
                    window_vol = volatility_summary(arrays, cohort.decision_id.to_numpy(dtype=int), pv)
                    for score in model_scores(part):
                        window_rows.append({"fold": f, "window_kind": kind, "first": str(cohort.timestamp.iloc[0]), "last": str(cohort.timestamp.iloc[-1]),
                            "relative_past_vol": window_vol["past_4h_vol_rms"] / train_vol[f]["past_4h_vol_rms"],
                            "realized_4h_vol_rms": window_vol["realized_4h_vol_rms"], **score})
    scores, vol, windows = pd.DataFrame(score_rows), pd.DataFrame(vol_rows), pd.DataFrame(window_rows)
    overview = scores.query("split == 'test'").pivot(index="fold", columns="model", values="balanced_accuracy").reset_index()
    overview["quarter"] = [str(pd.Timestamp(f["test_start"]).tz_localize(None).to_period("Q")) for f in folds]
    overview = overview.merge(vol.query("split == 'test'").drop(columns="split"), on="fold")
    correlations = []
    for model in MODELS:
        for variable in ("realized_4h_vol_rms", "past_4h_vol_rms", "relative_past_vol", "relative_realized_vol", "relative_net_return_rms", "fold"):
            correlations.append({"model": model, "variable": variable, **correlation_summary(overview[variable], overview[model])})
    reference = []
    fresh_vol = vol.query("split == 'fresh'").iloc[0].relative_past_vol
    for kind in ("daily_rolling", "disjoint"):
        for scope in ("all_folds", "same_final_fold", "same_final_fold_similar_vol"):
            data = windows.loc[windows.window_kind == kind]
            if scope != "all_folds":
                data = data.loc[data.fold == cfg["latest_fold"]]
            if scope.endswith("similar_vol"):
                data = data.loc[np.abs(data.relative_past_vol / fresh_vol - 1) <= cfg["matched_volatility_relative_tolerance"]]
            pivot = data.pivot(index=["fold", "first"], columns="model", values="balanced_accuracy")
            for name in (*MODELS, "wm-lightgbm", "logistic-lightgbm"):
                if "-" in name:
                    b, a = name.split("-")
                    values = pivot[b] - pivot[a] if len(pivot) else pd.Series(dtype=float)
                    fresh_score = float(scores.query("split == 'fresh' and model == @b").iloc[0].balanced_accuracy - scores.query("split == 'fresh' and model == @a").iloc[0].balanced_accuracy)
                else:
                    values = pivot[name] if len(pivot) else pd.Series(dtype=float)
                    fresh_score = float(scores.query("split == 'fresh' and model == @name").iloc[0].balanced_accuracy)
                values = values.dropna()
                reference.append({"window_kind": kind, "scope": scope, "model_or_difference": name, "n_windows": len(values), "fresh_score": fresh_score,
                    "historical_q05": float(values.quantile(.05)) if len(values) else None, "historical_median": float(values.median()) if len(values) else None,
                    "historical_q95": float(values.quantile(.95)) if len(values) else None,
                    "fraction_historical_at_or_below_fresh": float((values <= fresh_score).mean()) if len(values) else None})
    train, selections, probes = train_probes(cfg, folds, old_arrays, fresh_arrays, panels)
    gaps = train.merge(scores.query("split == 'val'")[["fold", "model", "balanced_accuracy"]], on=["fold", "model"], suffixes=("_train_probe", "_val"))
    gaps = gaps.merge(scores.query("split == 'test'")[["fold", "model", "balanced_accuracy"]].rename(columns={"balanced_accuracy": "balanced_accuracy_test"}), on=["fold", "model"])
    gaps["train_minus_val"] = gaps.balanced_accuracy_train_probe - gaps.balanced_accuracy_val
    gaps["val_minus_test"] = gaps.balanced_accuracy_val - gaps.balanced_accuracy_test
    gaps = gaps.merge(selections, on="fold")
    for model in ("logistic", "lightgbm"):
        g = gaps.loc[gaps.model == model]
        correlations.append({"model": model, "variable": "train_val_gap_vs_val_test_drop", **correlation_summary(g.train_minus_val, g.val_minus_test)})
    shifts, importance = [], []
    for index, gain in enumerate(probes["gain_all"]):
        if gain:
            name, lag, group = feature_name(index)
            importance.append({"feature_index": index, "feature": name, "lag_bars": lag, "group": group, "gain": float(gain), "gain_share": float(gain / probes["total_gain"])})
    for j, (index, gain) in enumerate(zip(probes["indices"], probes["gain"])):
        name, lag, group = feature_name(int(index))
        for split in ("val", "test", "fresh"):
            shifts.append({"feature_index": int(index), "feature": name, "lag_bars": lag, "group": group, "split": split,
                "gain_share": float(gain / probes["total_gain"]), **distribution_shift(probes["train"][:, j], probes[split][:, j], cfg["feature_histogram_bins"])})
    shifts = pd.DataFrame(shifts)
    shift_summary = []
    for split, part in shifts.groupby("split"):
        weights = part.gain_share.to_numpy()
        shift_summary.append({"split": split, "top_feature_gain_coverage": float(weights.sum()),
            **{f"gain_weighted_{k}": float(np.average(part[k], weights=weights)) for k in ("psi", "outside_train_range", "outside_train_01_99")}})
    original_fresh = pd.read_parquet(ensemble / "fresh_predictions.parquet")
    ci = []
    for model in ("logistic", "direct", "wm"):
        for row in paired_classification_block_bootstrap(original_fresh, "lightgbm", model, condition="fixed", block_days=cfg["block_days"], replicates=cfg["bootstrap_replicates"]):
            ci.append({**{k: v for k, v in row.items() if k != "metrics"}, **row["metrics"]["balanced_accuracy_difference"], "interpretation": "posthoc; short-period descriptive interval"})
    tables = {"fold_model_diagnostics": scores, "fold_volatility": vol, "fold_overview": overview,
        "correlations": pd.DataFrame(correlations), "windows": windows, "window_reference": pd.DataFrame(reference),
        "regime_scores": pd.DataFrame(regime_rows), "classical_train_probes": train, "generalization_gaps": gaps,
        "classical_selection": selections, "feature_shift": shifts, "feature_importance": pd.DataFrame(importance),
        "feature_shift_summary": pd.DataFrame(shift_summary), "fresh_pairwise_ci": pd.DataFrame(ci)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    summary = {"status": "complete", "fresh_window_length": length, "n_folds": len(folds),
        "analysis_role": "posthoc_descriptive", "test_selection_allowed": False, "model_refitting": False}
    make_report(out, tables, summary)
    if hashes != {str(p): file_hash(p) for p in paths}:
        raise ValueError("a frozen source artifact changed")
    summary.update(source_artifacts_unchanged=True, completed_utc=datetime.now(timezone.utc).isoformat(), config=cfg,
                   source_sha256=hashes, output_sha256={p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"})
    _json(out / "manifest.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "fresh_window_length", "n_folds", "source_artifacts_unchanged")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run(parser.parse_args().config)
