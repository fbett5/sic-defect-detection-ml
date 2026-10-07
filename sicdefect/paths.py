"""Where the data lives. Change WM811K_DEFAULT here, or set the WM811K_PATH
environment variable, to point at your copy of LSWMD.pkl (file or folder)."""
from __future__ import annotations

import os
from pathlib import Path

# Festus's Windows machine
WM811K_DEFAULT = r"C:\Users\festu\Downloads\Sofia University\ML Projects\WafferMap-811"


def wm811k_default() -> str:
    return os.environ.get("WM811K_PATH", WM811K_DEFAULT)


def resolve_pkl(path: str | Path) -> Path:
    """Accept either the .pkl file itself or a folder that contains it."""
    p = Path(path).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        found = sorted(p.rglob("LSWMD*.pkl")) or sorted(p.rglob("*.pkl"))
        if found:
            return found[0]
        raise FileNotFoundError(
            f"No .pkl file in {p}. If you downloaded the Kaggle zip, extract it first "
            "so LSWMD.pkl sits inside this folder."
        )
    raise FileNotFoundError(
        f"{p} does not exist. Pass --pkl <file or folder> or set WM811K_PATH."
    )
