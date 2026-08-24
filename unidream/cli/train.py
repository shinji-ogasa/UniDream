"""Strict CLI entrypoint for the current WM -> BC -> AC -> Test pipeline."""
from __future__ import annotations

import argparse

from unidream.device import DEVICE_HELP, resolve_device
from unidream.experiments.fold_training import run_fold
from unidream.experiments.m2 import format_m2_scorecard
from unidream.experiments.run_config import configure_determinism, load_training_run_config
from unidream.experiments.runtime import load_config, resolve_costs, set_seed
from unidream.experiments.train_app import run_training_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m unidream.cli.train",
        description="Strict reproducible UniDream WM -> BC -> AC -> Test pipeline",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, help="Self-contained training YAML")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True, help=DEVICE_HELP)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)

    cfg = load_config(args.config)
    cfg, active_cost_profile = resolve_costs(cfg)
    try:
        run_cfg = load_training_run_config(cfg)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    configure_determinism(args.seed)
    set_seed(args.seed)
    run_training_app(
        config_path=args.config,
        cfg=cfg,
        run_cfg=run_cfg,
        seed=args.seed,
        device=args.device,
        active_cost_profile=active_cost_profile,
        run_fold_fn=run_fold,
        format_m2_scorecard_fn=format_m2_scorecard,
    )


if __name__ == "__main__":
    main()
