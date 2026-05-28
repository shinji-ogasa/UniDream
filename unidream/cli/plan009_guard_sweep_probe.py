from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from unidream.cli.plan003_bc_student_probe import _json_sanitize
from unidream.cli.plan003_policy_blend_probe import _stress_metrics
from unidream.data.dataset import WFODataset
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits
from unidream.research.plan004_residual_bc_ac import run_plan004_fold_policy
from unidream.research.plan005_meta_guard_bc_ac import _build_past_features


@dataclass(frozen=True)
class GuardSpec:
    name: str
    mom_key: str | None
    mom_threshold: float | None
    dd_key: str | None
    dd_threshold: float | None
    low_position: float
    hold_bars: int
    cooldown_bars: int


def _fmt(v: Any, digits: int = 3) -> str:
    try:
        x = float(v)
    except Exception:
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def _signal(features: dict[str, np.ndarray], spec: GuardSpec) -> np.ndarray:
    out: np.ndarray | None = None
    if spec.mom_key and spec.mom_threshold is not None:
        out = features[spec.mom_key] < float(spec.mom_threshold)
    if spec.dd_key and spec.dd_threshold is not None:
        dd = features[spec.dd_key] < float(spec.dd_threshold)
        out = dd if out is None else (out | dd)
    if out is None:
        raise ValueError(f"empty guard signal: {spec}")
    return np.asarray(out, dtype=bool)


def _signal_positions(signal: np.ndarray, *, low: float, high: float, hold_bars: int, cooldown_bars: int) -> np.ndarray:
    sig = np.asarray(signal, dtype=bool)
    out_signal = np.zeros(len(sig), dtype=bool)
    hold_until = -1
    cooldown_until = -1
    hold = max(int(hold_bars), 1)
    cooldown = max(int(cooldown_bars), 0)
    for i, active in enumerate(sig):
        if i < hold_until:
            out_signal[i] = True
            continue
        if bool(active) and i >= cooldown_until:
            hold_until = i + hold
            cooldown_until = hold_until + cooldown
            out_signal[i] = True
    return np.where(out_signal, float(low), float(high)).astype(np.float64)


def _guard_full_positions(features: dict[str, np.ndarray], spec: GuardSpec, benchmark_position: float) -> np.ndarray:
    return _signal_positions(
        _signal(features, spec),
        low=float(spec.low_position),
        high=float(benchmark_position),
        hold_bars=int(spec.hold_bars),
        cooldown_bars=int(spec.cooldown_bars),
    )


def _build_specs(*, max_specs: int) -> list[GuardSpec]:
    specs: list[GuardSpec] = []
    for hold, cooldown in ((128, 128), (256, 128), (256, 256), (512, 256), (768, 512), (1024, 512)):
        for low in (0.0, 0.25, 0.50, 0.75):
            for mom_key, mom_thr in (("mom768", -0.02), ("mom768", -0.04), ("mom1536", -0.06), (None, None)):
                for dd_key, dd_thr in (
                    ("dd1536", -0.12),
                    ("dd1536", -0.18),
                    ("dd6144", -0.18),
                    ("dd12288", -0.25),
                    ("dd12288", -0.35),
                ):
                    parts = []
                    if mom_key:
                        parts.append(f"{mom_key}lt{str(mom_thr).replace('-', 'm').replace('.', 'p')}")
                    parts.append(f"{dd_key}lt{str(dd_thr).replace('-', 'm').replace('.', 'p')}")
                    specs.append(
                        GuardSpec(
                            name=f"{'_or_'.join(parts)}_low{str(low).replace('.', 'p')}_h{hold}_cd{cooldown}",
                            mom_key=mom_key,
                            mom_threshold=mom_thr,
                            dd_key=dd_key,
                            dd_threshold=dd_thr,
                            low_position=low,
                            hold_bars=hold,
                            cooldown_bars=cooldown,
                        )
                    )
    return specs[: int(max_specs)] if int(max_specs) > 0 else specs


def _score_val(metrics: dict[str, Any], *, turnover_cap: float, alpha_floor: float, dd_target: float) -> float:
    alpha = float(metrics.get("alpha_excess_pt", 0.0))
    maxdd = float(metrics.get("maxdd_delta_pt", 0.0))
    sharpe = float(metrics.get("sharpe_delta", 0.0))
    turnover = float(metrics.get("turnover", 999.0))
    if turnover > float(turnover_cap):
        return -1_000_000.0 - 1000.0 * (turnover - float(turnover_cap))
    score = alpha + 8.0 * sharpe + 3.0 * max(0.0, -maxdd) - 15.0 * max(0.0, maxdd)
    score -= 0.25 * turnover
    score -= 2.0 * max(0.0, float(alpha_floor) - alpha)
    score -= 4.0 * max(0.0, maxdd - float(dd_target))
    return float(score)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == group]
        metrics = [r["stress"]["cost_x1"] for r in rs]
        alphas = np.asarray([float(m["alpha_excess_pt"]) for m in metrics], dtype=np.float64)
        dds = np.asarray([float(m["maxdd_delta_pt"]) for m in metrics], dtype=np.float64)
        tos = np.asarray([float(m["turnover"]) for m in metrics], dtype=np.float64)
        out[group] = {
            "folds": int(len(rs)),
            "pass_alpha_ge3_dd_le_neg3": int(np.sum((alphas >= 3.0) & (dds <= -3.0))),
            "pass_alpha_ge10_dd_le_neg5": int(np.sum((alphas >= 10.0) & (dds <= -5.0))),
            "alpha_mean": float(np.mean(alphas)) if len(alphas) else float("nan"),
            "alpha_median": float(np.median(alphas)) if len(alphas) else float("nan"),
            "alpha_worst": float(np.min(alphas)) if len(alphas) else float("nan"),
            "maxdd_mean": float(np.mean(dds)) if len(dds) else float("nan"),
            "maxdd_worst": float(np.max(dds)) if len(dds) else float("nan"),
            "turnover_mean": float(np.mean(tos)) if len(tos) else float("nan"),
            "turnover_max": float(np.max(tos)) if len(tos) else float("nan"),
        }
    return out


def _stress_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stresses = sorted({s for r in rows for s in r["stress"].keys()})
    out: dict[str, Any] = {}
    for stress in stresses:
        out[stress] = {}
        for group in sorted({r["group"] for r in rows}):
            rs = [r for r in rows if r["group"] == group]
            metrics = [r["stress"][stress] for r in rs if stress in r["stress"]]
            alphas = np.asarray([float(m["alpha_excess_pt"]) for m in metrics], dtype=np.float64)
            dds = np.asarray([float(m["maxdd_delta_pt"]) for m in metrics], dtype=np.float64)
            tos = np.asarray([float(m["turnover"]) for m in metrics], dtype=np.float64)
            out[stress][group] = {
                "folds": int(len(metrics)),
                "pass_alpha_ge3_dd_le_neg3": int(np.sum((alphas >= 3.0) & (dds <= -3.0))),
                "alpha_median": float(np.median(alphas)) if len(alphas) else float("nan"),
                "alpha_worst": float(np.min(alphas)) if len(alphas) else float("nan"),
                "maxdd_worst": float(np.max(dds)) if len(dds) else float("nan"),
                "turnover_max": float(np.max(tos)) if len(tos) else float("nan"),
            }
    return out


def _load_base_cache(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _write_md(path: str, payload: dict[str, Any]) -> None:
    lines = [
        "# Plan009 Guard Sweep Probe",
        "",
        "Past-only guard sweep over Plan004 fold-local base positions.",
        "",
        f"Folds: `{', '.join(map(str, payload['folds']))}`",
        f"Seed: `{payload['seed']}`",
        f"Turnover cap: `{payload['turnover_cap']}`",
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
    lines.append("| fold | group | AlphaEx | MaxDDDelta | TO | selected | val Alpha | val MaxDD | val TO |")
    lines.append("|---:|---|---:|---:|---:|---|---:|---:|---:|")
    for row in payload["rows"]:
        m = row["stress"]["cost_x1"]
        d = row.get("diag", {})
        val = d.get("val", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["fold"]),
                    row["group"],
                    _fmt(m["alpha_excess_pt"]),
                    _fmt(m["maxdd_delta_pt"]),
                    _fmt(m["turnover"]),
                    str(d.get("spec", "")),
                    _fmt(val.get("alpha_excess_pt")),
                    _fmt(val.get("maxdd_delta_pt")),
                    _fmt(val.get("turnover")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Leak Discipline",
            "",
            "- Base positions are Plan004 fold-local policies.",
            "- Guard features are shifted rolling returns/drawdowns; the current bar never uses its own return.",
            "- Guard spec is selected on validation only, then applied once to test.",
            "- This is a fold0-12 development probe; it is not a fold13 or future-period claim.",
        ]
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m unidream.cli.plan009_guard_sweep_probe")
    parser.add_argument("--config", default="configs/trading.yaml")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--folds", default="0,1,2,3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--base-positions-cache", default="")
    parser.add_argument("--turnover-cap", type=float, default=12.0)
    parser.add_argument("--val-alpha-floor", type=float, default=-5.0)
    parser.add_argument("--val-dd-target", type=float, default=-3.0)
    parser.add_argument("--max-specs", type=int, default=0)
    parser.add_argument("--output-json", default="docs_local/20260527_plan009_guard_sweep.json")
    parser.add_argument("--output-md", default="docs_local/20260527_plan009_guard_sweep.md")
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
    features = _build_past_features(full_returns)
    specs = _build_specs(max_specs=int(args.max_specs))
    base_cache = _load_base_cache(args.base_positions_cache)

    rows: list[dict[str, Any]] = []
    for cache_idx, split in enumerate(splits):
        fid = int(split.fold_idx)
        print(f"[Plan009Guard] fold={fid} start")
        ds = WFODataset(features_df, raw_returns, split, seq_len=data_cfg.get("seq_len", 64))
        test_mask = np.asarray((features_df.index >= split.test_start) & (features_df.index <= split.test_end), dtype=bool)
        val_mask = np.asarray((features_df.index >= split.val_start) & (features_df.index < split.val_end), dtype=bool)
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
                "diag": {"spec": "plan004_base", "val": {}},
            }
        )

        best: dict[str, Any] | None = None
        best_guard: np.ndarray | None = None
        val_returns = full_returns[val_mask]
        for spec in specs:
            guard_full = _guard_full_positions(features, spec, benchmark_position)
            val_pos = guard_full[val_mask]
            if len(val_pos) == 0:
                continue
            val_stress = _stress_metrics(
                returns=val_returns,
                positions=val_pos,
                cfg=cfg,
                costs_cfg=costs_cfg,
                benchmark_position=benchmark_position,
            )
            val_metrics = val_stress["cost_x1"]
            score = _score_val(
                val_metrics,
                turnover_cap=float(args.turnover_cap),
                alpha_floor=float(args.val_alpha_floor),
                dd_target=float(args.val_dd_target),
            )
            if best is None or float(score) > float(best["score"]):
                best = {"spec": spec, "score": float(score), "val": val_metrics}
                best_guard = guard_full
        if best is None or best_guard is None:
            raise RuntimeError(f"fold {fid} has no selected guard")
        spec = best["spec"]
        guard_test = np.asarray(best_guard[test_mask], dtype=np.float64)[: len(base_pos)]
        positions = np.minimum(base_pos[: len(guard_test)], guard_test)
        stress = _stress_metrics(
            returns=ds.test_returns[: len(positions)],
            positions=positions,
            cfg=cfg,
            costs_cfg=costs_cfg,
            benchmark_position=benchmark_position,
        )
        print(
            f"[Plan009Guard] fold={fid} spec={spec.name} "
            f"alpha={stress['cost_x1']['alpha_excess_pt']:+.2f} "
            f"dd={stress['cost_x1']['maxdd_delta_pt']:+.2f} "
            f"to={stress['cost_x1']['turnover']:.2f}"
        )
        rows.append(
            {
                "fold": fid,
                "group": "plan009_val_guard_sweep",
                "source": base_source,
                "stress": stress,
                "diag": {
                    "spec": spec.name,
                    "spec_params": asdict(spec),
                    "score": best["score"],
                    "val": best["val"],
                },
            }
        )

    payload = {
        "experiment": "plan009_guard_sweep_probe",
        "seed": int(args.seed),
        "config": args.config,
        "folds": selected_folds,
        "base_positions_cache": args.base_positions_cache,
        "turnover_cap": float(args.turnover_cap),
        "val_alpha_floor": float(args.val_alpha_floor),
        "val_dd_target": float(args.val_dd_target),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "stress_aggregate": _stress_aggregate(rows),
        "spec_count": len(specs),
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_json_sanitize(payload), f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, payload)
    print(f"[Plan009Guard] wrote {args.output_json}")
    print(f"[Plan009Guard] wrote {args.output_md}")


if __name__ == "__main__":
    main()
