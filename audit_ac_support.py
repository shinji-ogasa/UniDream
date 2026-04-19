from __future__ import annotations

import argparse
from pathlib import Path

from unidream.device import add_device_argument, resolve_device
from unidream.experiments.ac_support_audit import run_ac_support_audit
from unidream.experiments.runtime import load_config, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit BC->AC support drift by regime")
    p.add_argument("--config", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--symbol", default=None)
    p.add_argument("--cache-dir", default="checkpoints/data_cache")
    p.add_argument("--raw-cache-dir", default=None)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--folds", default=None)
    add_device_argument(p)
    p.add_argument("--checkpoint-name", default="ac_best.pt")
    p.add_argument("--splits", default="train,val")
    p.add_argument("--max-bars", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.device = resolve_device(args.device)
    cfg = load_config(args.config)
    set_seed(args.seed)

    symbol = args.symbol or cfg.get("data", {}).get("symbol", "BTCUSDT")
    interval = cfg.get("data", {}).get("interval", "15m")
    config_name = Path(args.config).stem

    summary_df, _ = run_ac_support_audit(
        cfg=cfg,
        config_name=config_name,
        symbol=symbol,
        interval=interval,
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        raw_cache_dir=args.raw_cache_dir,
        checkpoint_dir=args.checkpoint_dir,
        folds_arg=args.folds,
        device=args.device,
        checkpoint_name=args.checkpoint_name,
        split_filter=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        max_bars=args.max_bars,
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
