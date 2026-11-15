"""Free-space guard. Every stage calls this first; a full disk has killed sessions on this Mac before."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

RESERVE_GB = 6.0
ROOT = Path(__file__).resolve().parents[1]


def free_gb(path: Path | str = ROOT) -> float:
    return shutil.disk_usage(path).free / 1e9


def guard(need_gb: float = 0.0, reserve_gb: float = RESERVE_GB, what: str = "") -> None:
    """Exit if free space would drop under the reserve after writing need_gb."""
    f = free_gb()
    if f - need_gb < reserve_gb:
        msg = f"disk guard: {f:.1f} GB free, need {need_gb:.1f} GB + {reserve_gb:.1f} GB reserve"
        if what:
            msg += f" for {what}"
        sys.exit(msg)


def ok(need_gb: float = 0.0, reserve_gb: float = RESERVE_GB) -> bool:
    return free_gb() - need_gb >= reserve_gb


if __name__ == "__main__":
    need = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    guard(need, what=sys.argv[2] if len(sys.argv) > 2 else "")
    print(f"disk ok: {free_gb():.1f} GB free")
