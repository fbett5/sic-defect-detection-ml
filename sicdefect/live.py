"""Live training dashboard for notebooks (Colab / Jupyter).

Training scripts write a small ``live.json`` (train_cnn.py) or Ultralytics'
``results.csv`` (train_yolo.py) as they go. ``watch()`` starts the script in the
background and redraws a dashboard from those files every few seconds:

  - progress bar for the current epoch, elapsed time, GPU use
  - train/val loss and val macro-F1 / balanced accuracy per epoch
  - latest validation confusion matrix (row-normalised = per-class recall)
  - sample validation wafer maps with true vs predicted class
  - the last lines of the script's console output

Usage in a notebook cell:

    from sicdefect.live import watch
    watch("python train_cnn.py --model resnet18 --epochs 20")

Interrupting the cell (stop button) stops the training process too.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def write_live(path: Path, state: dict) -> None:
    """Atomic JSON write, so the dashboard never reads half a file."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


# ---------------------------------------------------------------- reading state
def _arg(argv: list[str], name: str, default: str | None = None) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def _guess_live_source(argv: list[str]) -> tuple[str, Path]:
    """Work out where the script will write its progress, from its arguments."""
    script = Path(next((a for a in argv if a.endswith(".py")), "")).name
    out = Path(_arg(argv, "--out", "outputs"))
    if not out.is_absolute():
        out = ROOT / out
    if script == "train_yolo.py":
        name = _arg(argv, "--run-name") or Path(_arg(argv, "--model", "yolov8n-cls.pt")).stem
        return "yolo", out / "yolo" / name / "results.csv"
    run = _arg(argv, "--run-name") or f"{_arg(argv, '--model', 'resnet18')}-{_arg(argv, '--imbalance', 'weighted_sampler')}"
    return "cnn", out / run / "live.json"


def _read_cnn(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _read_yolo(path: Path) -> dict | None:
    """Turn Ultralytics results.csv into the same shape as live.json."""
    try:
        with open(path) as f:
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    except OSError:
        return None
    if not rows:
        return None

    def col(*names):
        for n in names:
            if n in rows[0]:
                return [float(r[n]) for r in rows]
        return []

    return {
        "history": {
            "epoch": col("epoch"),
            "train_loss": col("train/loss"),
            "val_loss": col("val/loss"),
            "val_top1": col("metrics/accuracy_top1"),
            "val_top5": col("metrics/accuracy_top5"),
        },
        "epoch": int(col("epoch")[-1]),
    }


def _gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        util, used, total = [s.strip() for s in out.strip().split(",")]
        return f"GPU {util}% busy, {int(used) / 1024:.1f}/{int(total) / 1024:.1f} GB"
    except Exception:
        return "GPU: n/a"


# ---------------------------------------------------------------- drawing
def _bar(frac: float, width: int = 30) -> str:
    frac = min(max(frac, 0.0), 1.0)
    full = int(round(frac * width))
    return "█" * full + "░" * (width - full) + f" {frac * 100:5.1f}%"


def _draw(kind: str, st: dict | None, log: deque, elapsed: float, title: str, done: bool):
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display

    clear_output(wait=True)
    status = "finished" if done else "running"
    print(f"{title}\n{status} · {elapsed / 60:.1f} min · {_gpu()}")

    if st is None:
        print("\nWaiting for the first progress update (loading data / building model)...")
        print("\n".join(list(log)[-12:]))
        return

    h = st.get("history", {})
    if kind == "cnn":
        p = st.get("progress", {})
        if p:
            ep_frac = p.get("batch", 0) / max(p.get("batches", 1), 1)
            print(f"\nepoch {p.get('epoch', 0)}/{p.get('epochs', '?')}  {_bar(ep_frac)}  "
                  f"running loss {p.get('loss', float('nan')):.4f}  · {p.get('phase', '')}")
            tot = p.get("epochs", 1)
            print(f"overall           {_bar(((p.get('epoch', 1) - 1) + ep_frac) / max(tot, 1))}")
        if h.get("val_macro_f1"):
            best = int(np.argmax(h["val_macro_f1"]))
            print(f"best so far: epoch {int(h['epoch'][best])}  val macro-F1 {h['val_macro_f1'][best]:.4f}")
    else:
        print(f"\nepoch {st.get('epoch', 0)} complete")

    cm = np.array(st["val_confusion"]) if st.get("val_confusion") else None
    samples = st.get("samples")
    extra = (cm is not None) or (samples is not None)
    fig = plt.figure(figsize=(13, 10.5 if extra else 4.5))
    gs = fig.add_gridspec(2 if extra else 1, 2, hspace=0.45, wspace=0.25)
    from matplotlib.ticker import MaxNLocator

    ax = fig.add_subplot(gs[0, 0])
    if h.get("train_loss"):
        ax.plot(h["epoch"], h["train_loss"], "o-", label="train loss")
    if h.get("val_loss"):
        ax.plot(h["epoch"], h["val_loss"], "s-", label="val loss")
    ax.set_title("Loss (lower is better)")
    ax.set_xlabel("epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    for key, label in (("val_macro_f1", "val macro-F1"), ("val_balanced_acc", "val balanced acc"),
                       ("val_defect_recall", "val defect recall"), ("val_top1", "val top-1 acc")):
        if h.get(key):
            ax.plot(h["epoch"], h[key], "o-", label=label)
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation scores (higher is better)")
    ax.set_xlabel("epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    if cm is not None:
        ax = fig.add_subplot(gs[1, 0])
        names = st.get("classes", [str(k) for k in range(len(cm))])
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(names)), names, fontsize=9)
        for r in range(len(names)):
            for c in range(len(names)):
                if norm[r, c] >= 0.05:
                    ax.text(c, r, f"{norm[r, c]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if norm[r, c] > 0.6 else "black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Val confusion (share of each true class), epoch {st.get('val_epoch', '?')}")

    if samples is not None:
        maps = np.array(samples["maps"], dtype=np.uint8)
        cols = 4
        rows = int(np.ceil(len(maps) / cols))
        sub = gs[1, 1].subgridspec(rows, cols, hspace=0.55, wspace=0.1)
        for j in range(len(maps)):
            a = fig.add_subplot(sub[j // cols, j % cols])
            a.imshow(maps[j], cmap="viridis", vmin=0, vmax=2, interpolation="nearest")
            ok = samples["true"][j] == samples["pred"][j]
            a.set_title(f"{samples['true'][j]}\n→ {samples['pred'][j]}", fontsize=8,
                        color="green" if ok else "red")
            a.set_xticks([])
            a.set_yticks([])
            if j == 1:
                a.text(1.05, 1.75, "Val samples: true → predicted (red = wrong)", transform=a.transAxes,
                       ha="center", fontsize=12)

    display(fig)
    plt.close(fig)

    tail = [ln for ln in log if ln.strip()][-6:]
    if tail:
        print("console:\n  " + "\n  ".join(tail))


# ---------------------------------------------------------------- main entry
def watch(cmd: str, refresh: float = 10.0, live_path: str | Path | None = None) -> int:
    """Run ``cmd`` (a training command) in the background and show a live dashboard.

    Returns the process exit code; raises RuntimeError if it failed.
    """
    argv = shlex.split(cmd)
    if argv and argv[0] in ("python", "python3"):
        argv[0] = sys.executable
    kind, path = _guess_live_source(argv)
    if live_path:
        path = Path(live_path)
    if path.exists():
        path.unlink()  # don't show the previous run's curves

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log_file = path.parent / "console.log" if kind == "cnn" else ROOT / "outputs" / "console.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log: deque = deque(maxlen=200)
    t0 = time.time()

    proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _pump():  # copy console output to the log file without blocking the dashboard
        with open(log_file, "w") as lf:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                for ln in ANSI.sub("", line).replace("\r", "\n").split("\n"):
                    if ln.strip():
                        log.append(ln)

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    try:
        last_draw = 0.0
        while True:
            running = proc.poll() is None
            if not running:
                reader.join(timeout=5)
            if not running or time.time() - last_draw >= refresh:
                st = _read_cnn(path) if kind == "cnn" else _read_yolo(path)
                _draw(kind, st, log, time.time() - t0, cmd, done=not running)
                last_draw = time.time()
            if not running:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait(timeout=30)
        print("\nStopped by you. Partial results are in", path.parent)
        return -1

    if proc.returncode != 0:
        print("\n".join(list(log)[-40:]))
        raise RuntimeError(f"Training exited with code {proc.returncode}. Full log: {log_file}")
    final = [ln for ln in log if ln.startswith(("test ", "Checkpoint", "Weights", "Report"))]
    if final:
        print("\n".join(final))
    return 0
