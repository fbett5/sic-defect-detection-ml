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
        # Top level first (fast), then subfolders, skipping virtual envs and
        # this repo's own outputs so an unrelated .pkl is never picked up.
        skip = {".venv", "venv", ".venv-anomaly", "site-packages", ".git",
                "outputs", "mlartifacts", "mlruns", "__pycache__"}

        def ok(f: Path) -> bool:
            return not (set(f.relative_to(p).parts[:-1]) & skip)

        for pattern in ("LSWMD*.pkl", "*.pkl"):
            top = sorted(p.glob(pattern))
            if top:
                return top[0]
            deep = sorted(f for f in p.rglob(pattern) if ok(f))
            if deep:
                return deep[0]
        raise FileNotFoundError(
            f"No .pkl file in {p}. If you downloaded the Kaggle zip, extract it first "
            "so LSWMD.pkl sits inside this folder."
        )
    raise FileNotFoundError(
        f"{p} does not exist. Pass --pkl <file or folder> or set WM811K_PATH."
    )
