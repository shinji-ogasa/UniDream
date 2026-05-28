from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from unidream.cli.exploration_board_probe import _apply_event_throttle, _shift_for_execution, _state_features
from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.plan003_policy_blend_probe import _stress_metrics
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.research.plan004_residual_bc_ac import run_plan004_fold_policy
from unidream.research.plan005_meta_guard_bc_ac import (
    _build_past_features,
    guard_positions_from_features,
    select_meta_guard_mode,
)


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _mask_for(index, start, end, *, inclusive_end: bool) -> np.ndarray:
    if inclusive_end:
        return np.asarray((index >= start) & (index <= end), dtype=bool)
    return np.asarray((index >= start) & (index < end), dtype=bool)


def _segment_start_idx(mask: np.ndarray) -> int:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(idx) == 0:
        raise ValueError("empty segment mask")
    return int(idx[0])


def _fit_guard_student(x: np.ndarray, y: np.ndarray, *, seed: int, max_train_samples: int) -> Any | None:
    yy = np.asarray(y, dtype=np.int64)
    mask = np.all(np.isfinite(x), axis=1) & np.isfinite(yy)
    idx = np.flatnonzero(mask)
    if len(idx) < 500 or len(np.unique(yy[idx])) < 2:
        return None
    if len(idx) > int(max_train_samples):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=int(max_train_samples), replace=False))
    model = make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=120,
            l2_regularization=0.05,
            random_state=seed,
        ),
    )
    model.fit(x[idx], yy[idx])
    return model


def _score(model: Any | None, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(x), np.nan, dtype=np.float64)
    return np.asarray(model.predict_proba(x)[:, -1], dtype=np.float64)


def _thresholds(score: np.ndarray) -> list[float]:
    vals = np.asarray(score, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return [float("inf")]
    qs = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995)
    out = [float(np.quantile(vals, q)) for q in qs]
    out.append(float("inf"))
    return sorted(set(out))


def _score_val(metrics: dict[str, Any], *, turnover_cap: float, target_dd: float) -> float:
    alpha = float(metrics.get("alpha_excess_pt", 0.0))
    maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
    sharpe = float(metrics.get("sharpe_delta", 0.0))
    turnover = float(metrics.get("turnover", 999.0))
    if turnover > float(turnover_cap):
        return -1_000_000.0 - 1000.0 * (turnover - float(turnover_cap))
    if turnover <= 1e-12:
        return -1000.0
    return alpha + 8.0 * sharpe + 4.0 * max(0.0, -maxdd) - 10.0 * max(0.0, maxdd - float(target_dd)) - 0.15 * turnover


def _student_guard_positions(
    *,
    score: np.ndarray,
    teacher_guard: np.ndarray,
    threshold: float,
    benchmark_position: float,
    hold_bars: int,
    cooldown_bars: int,
) -> np.ndarray:
    active = np.isfinite(score) & (score > float(threshold))
    selected = np.full(len(score), float(benchmark_position), dtype=np.float64)
    selected[active] = np.asarray(teacher_guard, dtype=np.float64)[: len(score)][active]
    throttled = _apply_event_throttle(
        selected,
        benchmark_position=float(benchmark_position),
        hold_bars=int(hold_bars),
        cooldown_bars=int(cooldown_bars),
    )
    return _shift_for_execution(throttled, float(benchmark_position))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == group]
        m = [r["stress"]["cost_x1"] for r in rs]
        alpha = np.asarray([float(x["alpha_excess_pt"]) for x in m], dtype=np.float64)
        dd = np.asarray([float(x["maxdd_delta_pt"]) for x in m], dtype=np.float64)
        turnover = np.asarray([float(x["turnover"]) for x in m], dtype=np.float64)
        out[group] = {
            "folds": int(len(rs)),
            "pass_alpha_ge3_dd_le_neg3": int(np.sum((alpha >= 3.0) & (dd <= -3.0))),
            "pass_alpha_ge10_dd_le_neg5": int(np.sum((alpha >= 10.0) & (dd <= -5.0))),
            "alpha_mean": float(np.mean(alpha)) if len(alpha) else float("nan"),
            "alpha_median": float(np.median(alpha)) if len(alpha) else float("nan"),
            "alpha_worst": float(np.min(alpha)) if len(alpha) else float("nan"),
            "maxdd_mean": float(np.mean(dd)) if len(dd) else float("nan"),
            "maxdd_worst": float(np.max(dd)) if len(dd) else float("nan"),
            "turnover_mean": float(np.mean(turnover)) if len(turnover) else float("nan"),
            "turnover_max": float(np.max(turnover)) if len(turnover) else float("nan"),
        }
    return out


def _stress_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stresses = sorted({stress for row in rows for stress in row["stress"].keys()})
    out: dict[str, Any] = {}
    for stress in stresses:
        out[stress] = {}
        for group in sorted({r["group"] for r in rows}):
            rs = [r for r in rows if r["group"] == group]
            metrics = [r["stress"][stress] for r in rs]
            alpha = np.asarray([float(x["alpha_excess_pt"]) for x in metrics], dtype=np.float64)
            dd = np.asarray([float(x["maxdd_delta_pt"]) for x in metrics], dtype=np.float64)
            turnover = np.asarray([float(x["turnover"]) for x in metrics], dtype=np.float64)
            out[stress][group] = {
                "folds": int(len(metrics)),
                "pass_alpha_ge3_dd_le_neg3": int(np.sum((alpha >= 3.0) & (dd <= -3.0))),
                "alpha_median": float(np.median(alpha)) if len(alpha) else float("nan"),
                "alpha_worst": float(np.min(alpha)) if len(alpha) else float("nan"),
                "maxdd_worst": float(np.max(dd)) if len(dd) else float("nan"),
                "turnover_max": float(np.max(turnover)) if len(turnover) else float("nan"),
            }
    return out


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan009 Guard Student Probe",
        "",
        "Small HGB student imitates the Plan005 past-only guard, then validation selects threshold/hold/cooldown.",
        "",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Seed: `{payload['seed']}`",
        "",
        "## Aggregate: cost_x1",
        "",
        "| group | folds | pass +3/-3 | pass +10/-5 | Alpha mean | Alpha median | Alpha worst | MaxDD mean | MaxDD worst | TO mean | TO max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, row in payload["aggregate"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    group,
                    str(row["folds"]),
                    str(row["pass_alpha_ge3_dd_le_neg3"]),
                    str(row["pass_alpha_ge10_dd_le_neg5"]),
                    _fmt(row["alpha_mean"]),
                    _fmt(row["alpha_median"]),
                    _fmt(row["alpha_worst"]),
                    _fmt(row["maxdd_mean"]),
                    _fmt(row["maxdd_worst"]),
                    _fmt(row["turnover_mean"]),
                    _fmt(row["turnover_max"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Stress Aggregate", ""])
    for stress, groups in payload["stress_aggregate"].items():
        lines.extend(
            [
                f"### {stress}",
                "",
                "| group | folds | pass +3/-3 | Alpha median | Alpha worst | MaxDD worst | TO max |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group, row in groups.items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        group,
                        str(row["folds"]),
                        str(row["pass_alpha_ge3_dd_le_neg3"]),
                        _fmt(row["alpha_median"]),
                        _fmt(row["alpha_worst"]),
                        _fmt(row["maxdd_worst"]),
                        _fmt(row["turnover_max"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(["## Fold Detail: cost_x1", ""])
    lines.append("| fold | group | mode | AlphaEx | MaxDDDelta | TO | val Alpha | val MaxDD | val TO | selection |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["rows"]:
        m = row["stress"]["cost_x1"]
        d = row.get("diag", {})
        v = d.get("val", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["group"],
                    str(d.get("mode", "")),
                    _fmt(m["alpha_excess_pt"]),
                    _fmt(m["maxdd_delta_pt"]),
                    _fmt(m["turnover"]),
                    _fmt(v.get("alpha_excess_pt")),
                    _fmt(v.get("maxdd_delta_pt")),
                    _fmt(v.get("turnover")),
                    f"thr={d.get('threshold')} hold={d.get('hold_bars')} cd={d.get('cooldown_bars')}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Leak Discipline",
            "",
            "- Teacher guard uses shifted past-only rolling return/drawdown features.",
            "- Guard mode is selected from shifted features at the test segment start, matching Plan005 availability.",
            "- Student trains on train segment only; threshold/hold/cooldown are selected on validation only.",
            "- Test metrics are report-only; fold0-12 is still a development set.",
        ]
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _load_base_cache(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan009_guard_student_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--base-positions-cache", default="")
    parser.add_argument("--turnover-cap", type=float, default=45.0)
    parser.add_argument("--target-dd", type=float, default=-3.0)
    parser.add_argument("--hold-grid", default="1,64,128,256")
    parser.add_argument("--cooldown-grid", default="0,64,128,256")
    parser.add_argument("--output-json", default="docs_local/20260527_plan009_guard_student.json")
    parser.add_argument("--output-md", default="docs_local/20260527_plan009_guard_student.md")
    args = parser.parse_args()
    set_seed(int(args.seed))

    cfg = load_config(args.config)
    cfg, _profile = resolve_costs(cfg, None)
    data_cfg = cfg.get("data", {})
    symbol = data_cfg.get("symbol", "BTCUSDT")
    interval = data_cfg.get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir="checkpoints/data_cache",
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits, selected_folds = select_wfo_splits(build_wfo_splits(features_df, data_cfg), args.folds)
    costs_cfg = cfg.get("costs", {})
    benchmark_position = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
    full_returns = np.asarray(raw_returns, dtype=np.float64)
    past_features = _build_past_features(full_returns)
    base_cache = _load_base_cache(args.base_positions_cache)
    hold_grid = tuple(int(x) for x in str(args.hold_grid).split(",") if x.strip())
    cooldown_grid = tuple(int(x) for x in str(args.cooldown_grid).split(",") if x.strip())

    rows: list[dict[str, Any]] = []
    for cache_idx, split in enumerate(splits):
        fid = int(split.fold_idx)
        print(f"[Plan009Student] fold={fid} start")
        ds = WFODataset(features_df, raw_returns, split, seq_len=data_cfg.get("seq_len", 64))
        train_mask = _mask_for(features_df.index, split.train_start, split.train_end, inclusive_end=True)
        val_mask = _mask_for(features_df.index, split.val_start, split.val_end, inclusive_end=False)
        test_mask = _mask_for(features_df.index, split.test_start, split.test_end, inclusive_end=True)

        mode, mode_diag = select_meta_guard_mode(past_features, _segment_start_idx(test_mask))
        teacher_full = guard_positions_from_features(past_features, mode=mode)
        teacher_train = teacher_full[train_mask][: len(ds.train_returns)]
        teacher_val = teacher_full[val_mask][: len(ds.val_returns)]
        teacher_test = teacher_full[test_mask][: len(ds.test_returns)]
        y_train = (teacher_train < benchmark_position - 1e-9).astype(np.int64)

        x_train = _state_features(ds.train_features, ds.train_returns)
        x_val = _state_features(ds.val_features, ds.val_returns)
        x_test = _state_features(ds.test_features, ds.test_returns)
        model = _fit_guard_student(
            x_train,
            y_train,
            seed=int(args.seed) + fid * 31,
            max_train_samples=int(args.max_train_samples),
        )
        score_val = _score(model, x_val)
        score_test = _score(model, x_test)

        if base_cache is not None:
            base_pos = np.asarray(base_cache["base_positions"][cache_idx], dtype=np.float64)[: len(ds.test_returns)]
            base_source = str(base_cache["selected"][cache_idx]) if "selected" in base_cache else "cache"
        else:
            base_rec = run_plan004_fold_policy(
                ds=ds,
                cfg=cfg,
                costs_cfg=costs_cfg,
                fold_idx=fid,
                seed=args.seed,
                ridge_l2=args.ridge_l2,
                max_train_samples=args.max_train_samples,
            )
            base_pos = np.asarray(base_rec["positions"], dtype=np.float64)[: len(ds.test_returns)]
            base_source = f"{base_rec['selected_row'].get('source')}:{base_rec['selected_row'].get('spec')}"
        base_stress = _stress_metrics(
            returns=ds.test_returns,
            positions=base_pos,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        rows.append(
            {
                "fold": fid,
                "group": "plan004_base",
                "source": base_source,
                "stress": base_stress,
                "diag": {"mode": "plan004_base", "val": {}, **mode_diag},
            }
        )

        best: dict[str, Any] | None = None
        for threshold in _thresholds(score_val):
            for hold in hold_grid:
                for cooldown in cooldown_grid:
                    guard_val = _student_guard_positions(
                        score=score_val,
                        teacher_guard=teacher_val,
                        threshold=threshold,
                        benchmark_position=benchmark_position,
                        hold_bars=hold,
                        cooldown_bars=cooldown,
                    )
                    val_metrics = _stress_metrics(
                        returns=ds.val_returns,
                        positions=guard_val,
                        cfg=cfg,
                        costs_cfg=costs_cfg,
                        benchmark_position=benchmark_position,
                    )["cost_x1"]
                    score = _score_val(
                        val_metrics,
                        turnover_cap=float(args.turnover_cap),
                        target_dd=float(args.target_dd),
                    )
                    cand = {
                        "score": float(score),
                        "threshold": float(threshold),
                        "hold_bars": int(hold),
                        "cooldown_bars": int(cooldown),
                        "val": val_metrics,
                    }
                    if best is None or cand["score"] > best["score"]:
                        best = cand
        assert best is not None
        guard_test = _student_guard_positions(
            score=score_test,
            teacher_guard=teacher_test,
            threshold=float(best["threshold"]),
            benchmark_position=benchmark_position,
            hold_bars=int(best["hold_bars"]),
            cooldown_bars=int(best["cooldown_bars"]),
        )
        positions = np.minimum(base_pos[: len(guard_test)], guard_test)
        stress = _stress_metrics(
            returns=ds.test_returns[: len(positions)],
            positions=positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        print(
            f"[Plan009Student] fold={fid} mode={mode} "
            f"alpha={stress['cost_x1']['alpha_excess_pt']:+.2f} "
            f"dd={stress['cost_x1']['maxdd_delta_pt']:+.2f} "
            f"to={stress['cost_x1']['turnover']:.2f}"
        )
        rows.append(
            {
                "fold": fid,
                "group": "plan009_guard_student",
                "source": base_source,
                "stress": stress,
                "diag": {
                    "mode": mode,
                    "teacher_underweight_rate_train": float(np.mean(y_train)),
                    "score": best["score"],
                    "threshold": best["threshold"],
                    "hold_bars": best["hold_bars"],
                    "cooldown_bars": best["cooldown_bars"],
                    "val": best["val"],
                    **mode_diag,
                },
            }
        )

    payload = {
        "experiment": "plan009_guard_student_probe",
        "seed": int(args.seed),
        "config": args.config,
        "folds": selected_folds,
        "base_positions_cache": args.base_positions_cache,
        "turnover_cap": float(args.turnover_cap),
        "target_dd": float(args.target_dd),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "stress_aggregate": _stress_aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[Plan009Student] wrote {args.output_json}")
    print(f"[Plan009Student] wrote {args.output_md}")


if __name__ == "__main__":
    main()
