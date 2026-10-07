"""Apply an exported model bundle to new wafer-map data.

This is the counterpart to :mod:`sicdefect.export`: it reads a bundle
directory (or ``.zip``) and scores data that the model has never seen, in
whatever form it arrives.

Accepted inputs
---------------
``.npy``
    One wafer map, or a stack of them, with values in {0, 1, 2}.
``.npz``
    A processed file from ``prepare_wm811k.py`` (key ``maps``, optionally
    filtered to one split with ``--split``), or any archive holding a ``maps``
    array.
``.png`` / ``.jpg``
    A wafer map saved as an image, decoded back through the {0, 127, 255}
    convention that ``sicdefect.wm811k.to_png`` writes.
a directory
    Every ``.npy``, ``.png`` and ``.jpg`` inside it, sorted by name.

Maps of any size are accepted and resized to the bundle's input geometry with
nearest-neighbour interpolation, implemented here in NumPy so that the ONNX
path needs neither OpenCV nor PyTorch.

Usage::

    python -m sicdefect.infer --bundle exported/resnet18-weighted_sampler \\
        --input new_wafers/ --out scored.csv --root-causes
    python -m sicdefect.infer --bundle exported/run.zip --input w.npy --backend onnx
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ARRAY_SUFFIXES = {".npy"}


# --------------------------------------------------------------------------
# Bundle loading
# --------------------------------------------------------------------------
class Bundle:
    """An exported model plus the contract describing how to feed it."""

    def __init__(self, root: Path, spec: dict, backend: str):
        self.root = root
        self.spec = spec
        self.backend = backend
        self.classes: list[str] = list(spec["classes"])
        self.input_size: int = int(spec["input_size"])
        self.channels: int = int(spec["preprocessing"]["channels"])
        self.divisor: float = float(spec["preprocessing"]["scale_divisor"])
        self._session = None
        self._module = None

    @property
    def name(self) -> str:
        return str(self.spec.get("name", self.root.name))

    def _load_onnx(self):
        import onnxruntime as ort

        path = self.root / self.spec["files"]["onnx"]
        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def _load_torchscript(self):
        import torch

        path = self.root / self.spec["files"]["torchscript"]
        self._module = torch.jit.load(str(path), map_location="cpu").eval()

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Run the network on a prepared ``(N, C, H, W)`` float32 batch."""
        if self.backend == "onnx":
            if self._session is None:
                self._load_onnx()
            return self._session.run(["logits"], {"wafer": x})[0]
        import torch

        if self._module is None:
            self._load_torchscript()
        with torch.no_grad():
            return self._module(torch.from_numpy(x)).numpy()


def load_bundle(path: str | Path, backend: str = "auto") -> Bundle:
    """Open a bundle directory or ``.zip`` and pick a usable backend.

    A zip is extracted to a temporary directory that lives as long as the
    process, so the returned bundle stays valid for the whole run.
    """
    path = Path(path)
    if path.is_file() and path.suffix == ".zip":
        tmp = tempfile.mkdtemp(prefix="sicdefect_bundle_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        root = Path(tmp)
    elif path.is_dir():
        root = path
    else:
        raise FileNotFoundError(f"{path} is not a bundle directory or .zip")

    spec_path = root / "bundle.json"
    if not spec_path.exists():
        # A zip made from a parent folder nests everything one level down.
        nested = next((p for p in root.glob("*/bundle.json")), None)
        if nested is None:
            raise FileNotFoundError(f"no bundle.json inside {path}")
        root, spec_path = nested.parent, nested
    spec = json.loads(spec_path.read_text())
    if spec.get("format_version") != 1:
        raise ValueError(
            f"unsupported bundle format_version {spec.get('format_version')!r}; "
            "this build reads version 1"
        )

    chosen = backend
    if backend == "auto":
        chosen = "onnx" if _have("onnxruntime") else "torch"
    if chosen == "onnx" and not _have("onnxruntime"):
        raise RuntimeError("--backend onnx needs onnxruntime: pip install onnxruntime")
    if chosen == "torch" and not _have("torch"):
        raise RuntimeError("--backend torch needs PyTorch installed")
    return Bundle(root, spec, chosen)


def _have(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------
_RESIZE_FALLBACK_WARNED = False


def nearest_resize(m: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize preserving the {0, 1, 2} die-value domain.

    Training resizes with ``cv2.resize(..., INTER_NEAREST)``, so inference uses
    OpenCV too whenever it is installed: the preprocessing a model is fed must
    be the preprocessing it was fitted under.

    Without OpenCV -- the dependency-free path, where a bundle runs on nothing
    but NumPy and onnxruntime -- a NumPy equivalent is used instead.  The two
    agree except at destination indices where the source coordinate is an exact
    integer: OpenCV's ``floor(dx * (src/dst))`` can land a fraction below that
    integer and take the previous column, where exact arithmetic does not.
    This affects a handful of columns at some size ratios and never arises in
    the normal flow, because ``prepare_wm811k.py`` has already resized the maps
    with OpenCV and this function then returns them untouched.
    """
    global _RESIZE_FALLBACK_WARNED

    m = np.asarray(m)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D wafer map, got shape {m.shape}")
    if m.shape == (size, size):
        return m

    try:
        import cv2

        return cv2.resize(
            np.ascontiguousarray(m, dtype=np.uint8), (size, size),
            interpolation=cv2.INTER_NEAREST,
        )
    except ImportError:
        if not _RESIZE_FALLBACK_WARNED:
            print(
                f"note: OpenCV not installed; resizing {m.shape} -> "
                f"({size}, {size}) with the NumPy fallback, which can differ "
                "from the training-time resize at exact index boundaries.",
                file=sys.stderr,
            )
            _RESIZE_FALLBACK_WARNED = True

    h, w = m.shape
    rows = np.minimum((np.arange(size) * h) // size, h - 1)
    cols = np.minimum((np.arange(size) * w) // size, w - 1)
    return m[rows][:, cols]


def decode_png(arr: np.ndarray) -> np.ndarray:
    """Invert ``to_png``: {0, 127, 255} -> {0, 1, 2}."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[2] <= 4 else a.mean(axis=2)
    return np.clip(np.rint(a.astype(np.float64) / 127.0), 0, 2).astype(np.uint8)


def _read_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        return decode_png(np.array(Image.open(path)))
    except ImportError:
        pass
    try:
        import cv2

        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(f"could not read {path}")
        return decode_png(a)
    except ImportError as exc:
        raise RuntimeError(
            f"reading {path.suffix} files needs Pillow or OpenCV installed"
        ) from exc


def read_input(path: str | Path, split: str | None = None) -> tuple[list[np.ndarray], list[str]]:
    """Collect wafer maps and a human-readable id for each."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        files = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES | ARRAY_SUFFIXES
        )
        if not files:
            raise ValueError(
                f"no .npy/.png/.jpg files in {path}"
            )
        maps, ids = [], []
        for f in files:
            for j, m in enumerate(_read_one(f)):
                maps.append(m)
                ids.append(f.name if len(_shape_hint(f)) <= 1 else f"{f.name}[{j}]")
        return maps, ids

    if path.suffix.lower() == ".npz":
        d = np.load(path, allow_pickle=True)
        if "maps" not in d.files:
            raise ValueError(f"{path} has no 'maps' array (found {d.files})")
        maps = d["maps"]
        idx = np.arange(len(maps))
        if split is not None:
            if "split" not in d.files:
                raise ValueError(f"--split given but {path} has no 'split' array")
            idx = np.flatnonzero(d["split"] == split)
            if not len(idx):
                raise ValueError(f"no rows with split={split!r} in {path}")
        return [maps[i] for i in idx], [f"{path.name}[{i}]" for i in idx]

    stack = _read_one(path)
    ids = [path.name] if len(stack) == 1 else [f"{path.name}[{i}]" for i in range(len(stack))]
    return list(stack), ids


def _read_one(path: Path) -> list[np.ndarray]:
    if path.suffix.lower() in ARRAY_SUFFIXES:
        a = np.load(path, allow_pickle=False)
        return [a] if a.ndim == 2 else list(a)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return [_read_image(path)]
    raise ValueError(f"unsupported input type: {path.suffix} ({path})")


def _shape_hint(path: Path) -> tuple:
    if path.suffix.lower() in ARRAY_SUFFIXES:
        try:
            return np.load(path, allow_pickle=False, mmap_mode="r").shape
        except Exception:  # noqa: BLE001
            return ()
    return ()


def prepare(maps: list[np.ndarray], bundle: Bundle) -> np.ndarray:
    """Apply the bundle's preprocessing contract to raw wafer maps."""
    resized = np.stack([nearest_resize(m, bundle.input_size) for m in maps])
    bad = np.setdiff1d(np.unique(resized), np.array([0, 1, 2]))
    if bad.size:
        raise ValueError(
            f"wafer maps must contain only 0/1/2 (off-wafer/pass/fail); found {bad[:5]}. "
            "If these came from images, they were not written with the {0,127,255} convention."
        )
    x = (resized.astype(np.float32) / bundle.divisor)[:, None]
    return np.repeat(x, bundle.channels, axis=1)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score(
    bundle: Bundle,
    input_path: str | Path,
    split: str | None = None,
    batch_size: int = 256,
):
    """Score every wafer map in ``input_path``; returns a pandas DataFrame."""
    import pandas as pd

    maps, ids = read_input(input_path, split=split)
    probs = []
    for start in range(0, len(maps), batch_size):
        chunk = maps[start : start + batch_size]
        probs.append(softmax(bundle.logits(prepare(chunk, bundle))))
    p = np.concatenate(probs)

    top = p.argmax(axis=1)
    out = pd.DataFrame(
        {
            "id": ids,
            "predicted": [bundle.classes[i] for i in top],
            "confidence": p[np.arange(len(p)), top],
        }
    )
    none_index = bundle.spec.get("none_index")
    if none_index is not None:
        out.insert(3, "defect_probability", 1.0 - p[:, none_index])
    for j, c in enumerate(bundle.classes):
        out[f"p_{c}"] = p[:, j]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--bundle", required=True, help="bundle directory or .zip")
    ap.add_argument("--input", required=True, help=".npy, .npz, .png, or a directory")
    ap.add_argument("--split", default=None, help="for .npz: only this split (train/val/test)")
    ap.add_argument("--out", default=None, help="write a CSV here (default: print a summary)")
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n", type=int, default=0, help="only show the first N rows")
    ap.add_argument("--root-causes", action="store_true", help="print likely process causes")
    a = ap.parse_args(argv)

    bundle = load_bundle(a.bundle, backend=a.backend)
    df = score(bundle, a.input, split=a.split, batch_size=a.batch_size)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"scored {len(df)} wafer maps with {bundle.name} "
              f"({bundle.backend} backend) -> {a.out}", file=sys.stderr)

    shown = df.head(a.n) if a.n else df
    cols = ["id", "predicted", "confidence"] + (
        ["defect_probability"] if "defect_probability" in df else []
    )
    if not a.out or a.n:
        print(shown[cols].to_string(index=False))

    counts = df["predicted"].value_counts()
    print(f"\n{len(df)} wafer maps | predicted pattern counts:", file=sys.stderr)
    for cls, n in counts.items():
        print(f"  {cls:<12} {n}", file=sys.stderr)

    if a.root_causes:
        from .rootcause import likely_causes

        print("\nLikely process causes for the patterns seen:", file=sys.stderr)
        for cls in counts.index:
            if cls == "none":
                continue
            print(f"\n  {cls}:", file=sys.stderr)
            for c in likely_causes(cls):
                print(f"    - {c}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
