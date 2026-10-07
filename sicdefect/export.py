"""Package a training checkpoint into a portable, self-describing model bundle.

``train_cnn.py`` writes ``best.pt``, which holds a state dict plus the
architecture name.  Re-loading it requires this repository, the right
torchvision version, and knowledge of how wafer maps were encoded -- so the
checkpoint alone cannot be handed to anyone else.

A bundle fixes that.  It contains:

``model.onnx``
    The network as a framework-independent graph.  Scoring it needs only
    ``numpy`` and ``onnxruntime`` -- no PyTorch, no torchvision, none of this
    repository's code, in Python or any other ONNX-capable language.
``model.ts``
    A TorchScript copy for callers who already run PyTorch.
``bundle.json``
    The class list, the input geometry, and the *exact* preprocessing contract
    the weights were trained under, so new data can be prepared identically.
``model_card.md``
    What the model is, how it scored, what it was trained on, and where it
    should not be trusted.

Usage::

    python -m sicdefect.export --ckpt outputs/resnet18-weighted_sampler/best.pt
    python -m sicdefect.export --ckpt outputs/.../best.pt --out exported/my_model --zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .models import build_model
from .wm811k import CLASSES

BUNDLE_FORMAT_VERSION = 1

#: The preprocessing the weights were trained under.  Written into every bundle
#: so a consumer can reproduce it without reading this repository.
PREPROCESSING = {
    "description": (
        "Wafer map values are 0 = off-wafer, 1 = passing die, 2 = failing die. "
        "Resize to input_size with NEAREST interpolation (never bilinear: it "
        "invents die values that cannot exist), divide by 2.0 to land in "
        "{0.0, 0.5, 1.0}, then repeat the single channel three times because "
        "the backbone is an ImageNet model."
    ),
    "value_domain": [0, 1, 2],
    "resize_interpolation": "nearest",
    "scale_divisor": 2.0,
    "channels": 3,
    "channel_mode": "repeat_grayscale",
    "layout": "NCHW",
    "dtype": "float32",
    "imagenet_normalisation": False,
}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _library_versions() -> dict:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    try:
        import onnx

        versions["onnx"] = onnx.__version__
    except ImportError:
        pass
    return versions


def load_checkpoint(ckpt_path: str | Path) -> dict:
    """Read a ``best.pt`` written by ``train_cnn.py`` and validate its shape."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = {"model", "arch", "classes"} - set(ck)
    if missing:
        raise ValueError(
            f"{ckpt_path} is missing {sorted(missing)}; expected a checkpoint "
            "written by train_cnn.py"
        )
    return ck


def export(
    ckpt_path: str | Path,
    out_dir: str | Path | None = None,
    input_size: int = 64,
    make_zip: bool = False,
    metrics_path: str | Path | None = None,
    notes: str | None = None,
) -> Path:
    """Write a bundle next to the checkpoint (or at ``out_dir``) and return its path."""
    ckpt_path = Path(ckpt_path)
    ck = load_checkpoint(ckpt_path)
    classes = list(ck["classes"])
    arch = str(ck["arch"])

    out_dir = Path(out_dir) if out_dir else Path("exported") / ckpt_path.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(arch, len(classes), pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    example = torch.zeros(1, PREPROCESSING["channels"], input_size, input_size)

    # TorchScript: trace rather than script, since torchvision backbones are
    # plain feed-forward graphs with no data-dependent control flow.
    ts_path = out_dir / "model.ts"
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.save(str(ts_path))

    onnx_path = out_dir / "model.onnx"
    # external_data=False keeps the weights inside the single .onnx file.  The
    # default splits them into a sidecar, and a bundle whose model silently
    # stops working when one file is left behind is not portable.
    onnx_kwargs = dict(
        input_names=["wafer"],
        output_names=["logits"],
        dynamic_axes={"wafer": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    try:
        torch.onnx.export(model, example, str(onnx_path), external_data=False, **onnx_kwargs)
    except TypeError:
        # Older torch exporters have no external_data switch and always inline.
        torch.onnx.export(model, example, str(onnx_path), **onnx_kwargs)
    sidecar = onnx_path.with_suffix(".onnx.data")
    if sidecar.exists():
        raise RuntimeError(
            f"ONNX weights were written to {sidecar.name} instead of being inlined; "
            "the bundle would not be self-contained"
        )

    # Verify the two exports agree with the source model before claiming either
    # is usable; a silently wrong export is worse than no export.
    rng = np.random.default_rng(0)
    probe = torch.from_numpy(
        (rng.integers(0, 3, (4, 1, input_size, input_size)) / 2.0).astype(np.float32)
    ).repeat(1, PREPROCESSING["channels"], 1, 1)
    with torch.no_grad():
        reference = model(probe).numpy()
        ts_out = torch.jit.load(str(ts_path))(probe).numpy()
    max_ts_diff = float(np.abs(reference - ts_out).max())

    max_onnx_diff = None
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(["logits"], {"wafer": probe.numpy()})[0]
        max_onnx_diff = float(np.abs(reference - onnx_out).max())
    except ImportError:
        pass

    tolerance = 1e-4
    if max_ts_diff > tolerance:
        raise RuntimeError(f"TorchScript export disagrees with the model by {max_ts_diff:.2e}")
    if max_onnx_diff is not None and max_onnx_diff > tolerance:
        raise RuntimeError(f"ONNX export disagrees with the model by {max_onnx_diff:.2e}")

    metrics = {}
    candidate = Path(metrics_path) if metrics_path else ckpt_path.parent / "test_metrics.json"
    if candidate.exists():
        metrics = json.loads(Path(candidate).read_text())

    none_index = classes.index("none") if "none" in classes else None
    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "name": out_dir.name,
        "task": "wafer-map defect pattern classification",
        "architecture": arch,
        "classes": classes,
        "none_index": none_index,
        "defect_classes": [c for c in classes if c != "none"],
        "input_size": input_size,
        "input_shape": [None, PREPROCESSING["channels"], input_size, input_size],
        "output": "logits; apply softmax over the class axis for probabilities",
        "preprocessing": PREPROCESSING,
        "files": {
            "onnx": "model.onnx",
            "torchscript": "model.ts",
        },
        "training": {
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": sha256(ckpt_path),
            "epoch": ck.get("epoch"),
            "val_macro_f1": ck.get("val_macro_f1"),
            "test_metrics": metrics,
        },
        "export": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": _library_versions(),
            "max_abs_logit_diff_torchscript": max_ts_diff,
            "max_abs_logit_diff_onnx": max_onnx_diff,
            "onnx_opset": 17,
        },
        "notes": notes,
    }
    (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    (out_dir / "model_card.md").write_text(_model_card(bundle))

    if make_zip:
        archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
        print(f"wrote {archive}")
    return out_dir


def _model_card(b: dict) -> str:
    tm = b["training"]["test_metrics"]
    headline = (
        "\n".join(
            f"| {k} | {v:.4f} |"
            for k, v in tm.items()
            if k in ("macro_f1", "balanced_acc", "accuracy", "defect_recall", "defect_precision")
        )
        or "| (no test_metrics.json found next to the checkpoint) | |"
    )
    per_class = "\n".join(
        f"| {k[3:]} | {v:.4f} |" for k, v in tm.items() if k.startswith("f1_")
    ) or "| (none recorded) | |"
    return f"""# Model card — `{b["name"]}`

**Task.** {b["task"]} over {len(b["classes"])} classes:
{", ".join(f"`{c}`" for c in b["classes"])}.

**Architecture.** `{b["architecture"]}`, classifier head resized to
{len(b["classes"])} outputs. Exported at ONNX opset {b["export"]["onnx_opset"]}.

## Input contract

Wafer maps use the WM-811K convention: `0` off-wafer, `1` passing die,
`2` failing die.

{b["preprocessing"]["description"]}

The network takes `float32` `{tuple(x if x else "batch" for x in b["input_shape"])}`
(NCHW) and returns **logits**; apply a softmax over the class axis for
probabilities.

## Scores

| metric | value |
|---|---|
{headline}

Per-class F1:

| class | F1 |
|---|---|
{per_class}

Accuracy is not the headline metric. Roughly 85% of labelled WM-811K wafers are
`none`, so always predicting `none` already scores about 85%. Judge this model
on macro-F1 and balanced accuracy.

## Using it without this repository

```python
import json, numpy as np, onnxruntime as ort

bundle = json.load(open("bundle.json"))
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

m = my_wafer_map                                  # 2-D array of 0/1/2
m = nearest_resize(m, bundle["input_size"])       # NEAREST only
x = (m / bundle["preprocessing"]["scale_divisor"]).astype("float32")
x = np.repeat(x[None, None], bundle["preprocessing"]["channels"], axis=1)

logits = sess.run(["logits"], {{"wafer": x}})[0]
p = np.exp(logits - logits.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
print(bundle["classes"][int(p.argmax())], float(p.max()))
```

`sicdefect.infer` does all of this, including the resize, for `.npy`, `.npz`,
PNG files and whole directories.

## Limitations

- **Resize with NEAREST interpolation only.** Bilinear or bicubic resampling
  invents die values between 0, 1 and 2 that cannot occur on a real wafer, and
  the model was never trained on them.
- **Domain.** Trained on WM-811K electrical-test wafer maps. SiC optical or
  X-ray topography images are a different modality; treat transfer to them as
  an open question, not a given.
- **Provenance of the score above.** If these metrics came from a run on
  synthetic data (`tools/make_synthetic_wm811k.py`), they only demonstrate that
  the pipeline executes. The synthetic patterns are far cleaner than real
  wafer maps and near-perfect scores there mean nothing about real performance.
- **Root-cause hints** from `sicdefect/rootcause.py` are hypothesis generators
  drawn from common failure signatures, not verdicts. Confirm against tool
  history and inline metrology before acting.
- **Class coverage.** Patterns outside the {len(b["classes"])} trained classes
  are forced into one of them. The model cannot say "something else"; use the
  anomaly-detection track for that.

Exported {b["export"]["created_utc"]} with
torch {b["export"]["versions"].get("torch")}.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="outputs/<run>/best.pt")
    ap.add_argument("--out", default=None, help="bundle directory (default: exported/<run>)")
    ap.add_argument("--input-size", type=int, default=64)
    ap.add_argument("--metrics", default=None, help="test_metrics.json (default: next to --ckpt)")
    ap.add_argument("--notes", default=None, help="free-text note stored in the bundle")
    ap.add_argument("--zip", action="store_true", dest="make_zip", help="also write <out>.zip")
    a = ap.parse_args(argv)
    out = export(
        a.ckpt, a.out, input_size=a.input_size, make_zip=a.make_zip,
        metrics_path=a.metrics, notes=a.notes,
    )
    print(f"wrote bundle to {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
