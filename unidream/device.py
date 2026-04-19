from __future__ import annotations

import torch

DEFAULT_DEVICE = "auto"
DEVICE_HELP = "auto, mps, cuda, or cpu"


def mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def preferred_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def resolve_device(device: str | None = None) -> str:
    requested = (device or DEFAULT_DEVICE).strip().lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device '{device}'. Use auto, cpu, cuda, or mps.")
    if requested == "auto":
        return preferred_device()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine. Use --device auto, mps, or cpu.")
    if requested == "mps" and not mps_available():
        raise RuntimeError("MPS is not available on this machine. Use --device auto or cpu.")
    return requested


def add_device_argument(parser, *, default: str = DEFAULT_DEVICE):
    parser.add_argument("--device", default=default, help=DEVICE_HELP)
