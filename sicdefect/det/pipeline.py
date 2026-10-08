"""The end-to-end inference pipeline.

    image (+ reference) -> preprocess -> YOLO boxes -> crops -> CNN -> fused decision -> review status

Fusion (``mode``):
    yolo        YOLO's own class and confidence (single stage)
    two_stage   CNN decides the class among defect types and can veto the box:
                    class = argmax over defect classes of the CNN
                    score = sqrt(yolo_conf * p_cnn(class))
                A box the CNN calls background gets a low score, so it drops out
                at the operating threshold instead of reaching the engineer.

Review status per box (the human-in-the-loop part):
    auto        score >= accept_thr and YOLO and CNN agree -> pre-accepted, engineer can still reject
    review      everything else above show_thr: engineer must decide
    hidden      below show_thr: not shown (but kept in the JSON for audit)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .crops import cut, to_tensor
from .preprocess import make_input, read_gray


@dataclass
class Thresholds:
    show: float = 0.25      # minimum score to show a box to the engineer
    accept: float = 0.70    # at or above (and models agree) -> pre-accepted
    det_floor: float = 0.001  # YOLO confidence floor (same as Ultralytics val, so mAP is comparable)
    cnn_floor: float = 0.01   # only boxes above this go to the CNN (keeps CPU cost bounded)
    dedupe_iou: float = 0.6   # two-stage: same final class + IoU above this -> keep the best only


@dataclass
class Pipeline:
    det_weights: str
    cnn_weights: str | None = None
    view: str = "refdiff"
    imgsz: int = 640
    device: str = "auto"
    thresholds: Thresholds = field(default_factory=Thresholds)

    def __post_init__(self):
        from ultralytics import YOLO

        from ..sem import build
        from ..utils import get_device

        dev = get_device(self.device)
        self._dev = dev
        self._ydev = 0 if dev.type == "cuda" else "cpu"
        self.det = YOLO(self.det_weights)
        self.classes = [self.det.names[i] for i in sorted(self.det.names)]
        self.cnn = None
        if self.cnn_weights:
            ck = torch.load(self.cnn_weights, map_location=dev, weights_only=False)
            if ck["view"] != self.view:
                raise ValueError(f"CNN was trained on view {ck['view']!r}, pipeline uses {self.view!r}")
            if ck["classes"][: ck["n_defect_classes"]] != self.classes:
                raise ValueError(f"Class mismatch: YOLO {self.classes} vs CNN {ck['classes']}")
            self.cnn = build(ck["arch"], len(ck["classes"]), pretrained=False)
            self.cnn.load_state_dict(ck["model"])
            self.cnn.to(dev).eval()
            self.crop, self.pad = ck["crop"], ck["pad"]
        self.timing = {"preprocess": [], "detect": [], "classify": [], "total": []}

    # ------------------------------------------------------------------
    def prepare(self, image_path, reference_path=None) -> np.ndarray:
        img = read_gray(image_path)
        ref = read_gray(reference_path) if reference_path else None
        return make_input(img, ref, self.view)

    def run(self, image_path=None, reference_path=None, x: np.ndarray | None = None) -> list[dict]:
        """All candidate boxes for one image (sorted by score), each with both models' opinions."""
        t0 = time.perf_counter()
        if x is None:
            x = self.prepare(image_path, reference_path)
        t1 = time.perf_counter()
        r = self.det.predict(x, conf=self.thresholds.det_floor, imgsz=self.imgsz, device=self._ydev,
                             verbose=False, max_det=300)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        ycls = r.boxes.cls.cpu().numpy().astype(int)
        yconf = r.boxes.conf.cpu().numpy()
        t2 = time.perf_counter()
        probs = None
        sel = np.flatnonzero(yconf >= self.thresholds.cnn_floor)
        if self.cnn is not None and len(sel):
            crops = np.stack([cut(x, boxes[i], self.crop, self.pad) for i in sel])
            with torch.no_grad():
                pr = torch.softmax(self.cnn(to_tensor(crops).to(self._dev)), 1).cpu().numpy()
            probs = {int(i): pr[k] for k, i in enumerate(sel)}
        t3 = time.perf_counter()

        n = len(self.classes)
        out = []
        for i, b in enumerate(boxes):
            d = {"box": [round(float(v), 1) for v in b], "yolo_cls": int(ycls[i]), "yolo_conf": float(yconf[i])}
            if probs is not None and i in probs:
                p = probs[i]
                c = int(np.argmax(p[:n]))
                d.update(cnn_cls=c, cnn_conf=float(p[c]), p_background=float(p[n]),
                         cnn_probs=[round(float(v), 4) for v in p])
            out.append(d)
        for d in out:
            for mode in ("yolo", "two_stage"):
                if mode == "two_stage" and "cnn_cls" not in d:
                    continue
                c, s = self.fuse(d, mode)
                d[f"{mode}_final_cls"], d[f"{mode}_score"] = c, s
        self._dedupe(out)
        t4 = time.perf_counter()
        for k, v in (("preprocess", t1 - t0), ("detect", t2 - t1), ("classify", t3 - t2), ("total", t4 - t0)):
            self.timing[k].append(v)
        return sorted(out, key=lambda d: -d.get("two_stage_score", d["yolo_score"]))

    def _dedupe(self, out: list[dict]) -> None:
        """YOLO's NMS is per class, so one defect can carry two boxes with different YOLO
        classes. After the CNN relabels them they may agree: keep only the best-scoring one."""
        from .boxes import iou_matrix

        ts = [d for d in out if "two_stage_score" in d]
        ts.sort(key=lambda d: -d["two_stage_score"])
        kept = []
        for d in ts:
            if any(k["two_stage_final_cls"] == d["two_stage_final_cls"]
                   and iou_matrix([d["box"]], [k["box"]])[0, 0] > self.thresholds.dedupe_iou for k in kept):
                d["two_stage_duplicate"] = True
            else:
                kept.append(d)

    @staticmethod
    def fuse(d: dict, mode: str) -> tuple[int, float]:
        if mode == "yolo":
            return d["yolo_cls"], d["yolo_conf"]
        return d["cnn_cls"], float(np.sqrt(d["yolo_conf"] * d["cnn_conf"]))

    def status(self, d: dict, mode: str) -> str:
        if f"{mode}_score" not in d or d.get(f"{mode}_duplicate"):
            return "hidden"
        s = d[f"{mode}_score"]
        if s < self.thresholds.show:
            return "hidden"
        agree = mode == "yolo" or d["yolo_cls"] == d["cnn_cls"]
        return "auto" if s >= self.thresholds.accept and agree else "review"

    def timing_summary(self) -> dict:
        return {k: {"mean_ms": 1000 * float(np.mean(v)), "p95_ms": 1000 * float(np.percentile(v, 95))}
                for k, v in self.timing.items() if v}


def as_eval_preds(results: dict[str, list[dict]], mode: str) -> dict:
    """Pipeline output -> evaluate.py format {img: [(cls, box, score)]}."""
    return {k: [(d[f"{mode}_final_cls"], tuple(d["box"]), d[f"{mode}_score"]) for d in v
                if f"{mode}_score" in d and not d.get(f"{mode}_duplicate")] for k, v in results.items()}


def find_reference(image_path: Path, ref_dir: Path | None) -> Path | None:
    """Reference image with the same stem in ref_dir, or <stem>_ref / <stem>_temp next to the image."""
    p = Path(image_path)
    cands = []
    if ref_dir:
        cands += [Path(ref_dir) / p.name] + [Path(ref_dir) / f"{p.stem}{e}" for e in (".png", ".jpg", ".tif")]
    cands += [p.with_name(f"{p.stem}{s}{p.suffix}") for s in ("_ref", "_temp")]
    return next((c for c in cands if c.exists()), None)
