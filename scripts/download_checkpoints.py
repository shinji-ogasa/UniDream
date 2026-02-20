#!/usr/bin/env python3
"""
Download pretrained dreamer4 checkpoints from HuggingFace.

Usage:
    python scripts/download_checkpoints.py [--outdir checkpoints/]

These are DMControl (robot control, 30 tasks) checkpoints.
Purpose: debug / shape-check only. Minecraft training starts from scratch.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "tokenizer_ckpt.pt": (
        "https://huggingface.co/nicklashansen/dreamer4/resolve/main/tokenizer_ckpt.pt",
        None,  # md5 – fill in after first download if you want integrity checks
    ),
    "dynamics_ckpt.pt": (
        "https://huggingface.co/nicklashansen/dreamer4/resolve/main/dynamics_ckpt.pt",
        None,
    ),
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    print(f"  -> {url}")
    print(f"     saving to {dest}")

    def _progress(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(count * block_size / total_size * 100, 100)
        bar = "#" * int(pct / 2)
        sys.stdout.write(f"\r     [{bar:<50s}] {pct:5.1f}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        print()  # newline after progress bar
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="checkpoints", help="destination directory")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)

    for name, (url, expected_md5) in CHECKPOINTS.items():
        dest = outdir / name
        if args.skip_existing and dest.exists():
            print(f"[skip] {name} already exists")
            continue
        print(f"[download] {name}")
        _download(url, dest)

        if expected_md5:
            got = hashlib.md5(dest.read_bytes()).hexdigest()
            if got != expected_md5:
                print(f"[WARN] MD5 mismatch for {name}: got {got}, expected {expected_md5}")
            else:
                print(f"[ok] MD5 verified: {name}")

    print("\nDone. Files in", outdir)
    for p in sorted(outdir.glob("*.pt")):
        mb = p.stat().st_size / 1e6
        print(f"  {p.name}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
