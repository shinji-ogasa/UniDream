"""Small logging helpers shared by experiment stages."""
from __future__ import annotations

from datetime import datetime


def log_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")
