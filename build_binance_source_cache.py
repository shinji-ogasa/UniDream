from __future__ import annotations

import argparse
import os

from unidream.data.download import (
    fetch_binance_ohlcv,
    fetch_funding_rate,
    fetch_open_interest_hist,
    fetch_mark_price_klines,
    fetch_taker_buy_sell_volume,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Binance raw source cache files")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--skip-funding", action="store_true")
    parser.add_argument("--skip-oi", action="store_true")
    parser.add_argument("--skip-mark", action="store_true")
    parser.add_argument("--skip-taker-flow", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print("[SRC] Fetching spot OHLCV...")
    ohlcv = fetch_binance_ohlcv(args.symbol, args.interval, args.start, args.end)
    ohlcv_path = os.path.join(args.cache_dir, f"{args.cache_tag}_ohlcv.parquet")
    ohlcv.to_parquet(ohlcv_path)
    print(f"[SRC] Wrote {ohlcv_path} ({len(ohlcv)} rows)")

    if not args.skip-funding:
        try:
            print("[SRC] Fetching funding rate...")
            funding = fetch_funding_rate(args.symbol, args.start, args.end)
            funding_path = os.path.join(args.cache_dir, f"{args.cache_tag}_funding.parquet")
            funding.to_parquet(funding_path)
            print(f"[SRC] Wrote {funding_path} ({len(funding)} rows)")
        except Exception as e:
            print(f"[SRC] Funding skipped: {e}")

    if not args.skip-oi:
        try:
            print("[SRC] Fetching open interest...")
            oi = fetch_open_interest_hist(args.symbol, args.interval, args.start, args.end)
            oi_path = os.path.join(args.cache_dir, f"{args.cache_tag}_oi.parquet")
            oi.to_parquet(oi_path)
            print(f"[SRC] Wrote {oi_path} ({len(oi)} rows)")
        except Exception as e:
            print(f"[SRC] OI skipped: {e}")

    if not args.skip-mark:
        try:
            print("[SRC] Fetching mark price...")
            mark = fetch_mark_price_klines(args.symbol, args.interval, args.start, args.end)
            mark_path = os.path.join(args.cache_dir, f"{args.cache_tag}_mark.parquet")
            mark.to_parquet(mark_path)
            print(f"[SRC] Wrote {mark_path} ({len(mark)} rows)")
        except Exception as e:
            print(f"[SRC] Mark skipped: {e}")

    if not args.skip_taker_flow:
        try:
            print("[SRC] Fetching taker buy/sell flow...")
            taker = fetch_taker_buy_sell_volume(args.symbol, args.interval, args.start, args.end)
            for col in ["signed_order_flow", "taker_imbalance", "buy_sell_ratio"]:
                if col not in taker.columns:
                    continue
                out_path = os.path.join(args.cache_dir, f"{args.cache_tag}_series_{col}.parquet")
                taker[[col]].to_parquet(out_path)
                print(f"[SRC] Wrote {out_path} ({len(taker)} rows)")
        except Exception as e:
            print(f"[SRC] Taker flow skipped: {e}")


if __name__ == "__main__":
    main()
