from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from unidream.actor_critic.imagination_ac import _action_stats, _fmt_action_stats
from unidream.data.dataset import WFODataset
from unidream.data.oracle import _forward_window_stats
from unidream.device import add_device_argument, resolve_device
from unidream.experiments.fold_inputs import prepare_fold_inputs
from unidream.experiments.m2 import benchmark_position_value
from unidream.experiments.runtime import load_config, load_training_features, resolve_costs, set_seed
from unidream.experiments.train_app import resolve_cache_dir
from unidream.experiments.transition_advantage import (
    compute_transition_advantage,
    config_from_dict,
    current_positions_from_path,
    recovery_latency,
    summarize_transition_advantage,
)
from unidream.experiments.wfo_runtime import build_wfo_splits, select_wfo_splits


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _parse_actions(text: str | None) -> list[float] | None:
    if not text:
        return None
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def _fmt_float(value: float, digits: int = 6) -> str:
    if value != value:
        return "nan"
    return f"{value:.{digits}f}"


def _summary_to_markdown(results: list[dict], sources: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Transition Advantage Probe")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Fold Summary")
    lines.append("")
    lines.append("| fold | train period | target short | target benchmark | target overweight | mean best adv | recovery rate | teacher recovery median |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["summary"]
        rec = r["teacher_recovery_latency"]
        lines.append(
            "| "
            f"{r['fold']} | {r['train_start']} -> {r['train_end']} | "
            f"{s['target_short_rate']:.1%} | {s['target_benchmark_rate']:.1%} | "
            f"{s['target_overweight_rate']:.1%} | {_fmt_float(s['mean_best_advantage'])} | "
            f"{s['recovery_rate_from_underweight']:.1%} | {_fmt_float(rec['median'], 1)} |"
        )
    lines.append("")

    for r in results:
        s = r["summary"]
        lines.append(f"## Fold {r['fold']}")
        lines.append("")
        lines.append(f"- Train: `{r['train_start']} -> {r['train_end']}`")
        lines.append(f"- Teacher dist: `{r['teacher_action_stats']}`")
        lines.append(f"- Candidate actions: `{', '.join(str(x) for x in r['candidate_actions'])}`")
        lines.append(f"- Horizons: `{', '.join(str(x) for x in r['horizons'])}`")
        lines.append("")
        lines.append("### Best Transition Class")
        lines.append("")
        lines.append("| class | count | rate | mean_adv | median_adv | top_decile | bottom_decile | positive_rate |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in s["classes"]:
            lines.append(
                "| "
                f"{row['class']} | {row['count']} | {row['rate']:.1%} | "
                f"{_fmt_float(row['mean_adv'])} | {_fmt_float(row['median_adv'])} | "
                f"{_fmt_float(row['top_decile_mean'])} | {_fmt_float(row['bottom_decile_mean'])} | "
                f"{row['positive_rate']:.1%} |"
            )
        lines.append("")
        lines.append("### Candidate Action Value")
        lines.append("")
        lines.append("| action | mean | median | top_decile | bottom_decile | positive_rate |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in s["actions"]:
            lines.append(
                "| "
                f"{row['action']:.2f} | {_fmt_float(row['mean'])} | {_fmt_float(row['median'])} | "
                f"{_fmt_float(row['top_decile_mean'])} | {_fmt_float(row['bottom_decile_mean'])} | "
                f"{row['positive_rate']:.1%} |"
            )
        lines.append("")
        lines.append("### Transition Matrix")
        lines.append("")
        lines.append("| current -> target | underweight | benchmark | overweight |")
        lines.append("|---|---:|---:|---:|")
        for cur, cols in s["transition_matrix"].items():
            lines.append(
                f"| {cur} | {cols['underweight']} | {cols['benchmark']} | {cols['overweight']} |"
            )
        lines.append("")
    if sources:
        lines.append("## Sources")
        lines.append("")
        for src in sources:
            lines.append(f"- {src}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m unidream.cli.transition_advantage_probe",
        description="Probe cost-adjusted transition advantages without training.",
    )
    parser.add_argument("--config", default="configs/bc_multitask_aux_nopred_fairwm_s007.yaml")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", default="4")
    parser.add_argument("--cost-profile", default=None)
    parser.add_argument("--candidate-actions", default="0.0,0.5,1.0,1.25")
    parser.add_argument("--output-json", default="documents/20260425_transition_advantage_probe.json")
    parser.add_argument("--output-md", default="documents/20260425_transition_advantage_probe.md")
    add_device_argument(parser)
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg, active_cost_profile = resolve_costs(cfg, args.cost_profile)
    cfg["cost_profile"] = active_cost_profile
    if args.candidate_actions:
        cfg.setdefault("bc", {})["transition_candidate_actions"] = _parse_actions(args.candidate_actions)
    checkpoint_dir = cfg.get("logging", {}).get("checkpoint_dir", "checkpoints/transition_probe")
    symbol = args.symbol or cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")
    zscore_window = cfg.get("normalization", {}).get("zscore_window_days", 60)
    cache_dir = resolve_cache_dir(checkpoint_dir, cfg)
    cache_tag = f"{symbol}_{interval}_{args.start}_{args.end}_z{zscore_window}_v2"
    data_cfg = cfg.get("data", {})

    print(f"Transition Advantage Probe | {symbol} {interval} | {args.start} -> {args.end}")
    features_df, raw_returns = load_training_features(
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        zscore_window=zscore_window,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        extra_series_mode=data_cfg.get("extra_series_mode", "derived"),
        extra_series_include=data_cfg.get("extra_series_include"),
        include_funding=bool(data_cfg.get("include_funding", True)),
        include_oi=bool(data_cfg.get("include_oi", True)),
        include_mark=bool(data_cfg.get("include_mark", True)),
    )
    splits = build_wfo_splits(features_df, data_cfg)
    splits, selected = select_wfo_splits(splits, args.folds)
    if selected is not None:
        print(f"Running selected folds only: {selected}")

    results: list[dict] = []
    for split in splits:
        wfo_dataset = WFODataset(
            features_df,
            raw_returns,
            split,
            seq_len=data_cfg.get("seq_len", 64),
        )
        fold_inputs = prepare_fold_inputs(
            wfo_dataset=wfo_dataset,
            cfg=cfg,
            costs_cfg=cfg.get("costs", {}),
            ac_cfg=cfg.get("ac", {}),
            bc_cfg=cfg.get("bc", {}),
            reward_cfg=cfg.get("reward", {}),
            action_stats_fn=_action_stats,
            format_action_stats_fn=_fmt_action_stats,
            benchmark_position=benchmark_position_value(cfg),
            forward_window_stats_fn=_forward_window_stats,
            log_ts=_ts,
        )
        oracle_positions = fold_inputs["oracle_positions"]
        benchmark = float(cfg.get("reward", {}).get("benchmark_position", 1.0))
        current = current_positions_from_path(oracle_positions, benchmark)
        ta_cfg = config_from_dict(
            cfg.get("bc", {}),
            costs_cfg=cfg.get("costs", {}),
            benchmark_position=benchmark,
            default_actions=cfg.get("bc", {}).get("transition_candidate_actions", [0.0, 0.5, 1.0, 1.25]),
        )
        bundle = compute_transition_advantage(wfo_dataset.train_returns, current, ta_cfg)
        summary = summarize_transition_advantage(bundle, current, benchmark)
        result = {
            "fold": int(split.fold_idx),
            "train_start": str(split.train_start.date()),
            "train_end": str(split.train_end.date()),
            "candidate_actions": [float(x) for x in bundle["actions"]],
            "horizons": [int(x) for x in bundle["horizons"]],
            "teacher_action_stats": _fmt_action_stats(_action_stats(oracle_positions, benchmark_position=benchmark)),
            "teacher_recovery_latency": recovery_latency(oracle_positions, benchmark),
            "summary": summary,
        }
        print(
            f"[Fold {split.fold_idx}] target short={summary['target_short_rate']:.1%} "
            f"bench={summary['target_benchmark_rate']:.1%} ow={summary['target_overweight_rate']:.1%} "
            f"mean_adv={summary['mean_best_advantage']:.6f}"
        )
        results.append(result)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    sources = [
        "IQL: https://arxiv.org/abs/2110.06169",
        "AWAC: https://arxiv.org/abs/2006.09359",
        "TD3+BC: https://arxiv.org/abs/2106.06860",
        "CQL: https://arxiv.org/abs/2006.04779",
    ]
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_summary_to_markdown(results, sources), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
