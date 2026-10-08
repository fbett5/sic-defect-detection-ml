"""Detection metrics.

Inputs
    gt     {image_id: [(class_id, (x1, y1, x2, y2)), ...]}
    preds  {image_id: [(class_id, (x1, y1, x2, y2), confidence), ...]}

Two kinds of numbers:

1. Threshold-free ranking quality (COCO style): AP per class at IoU 0.50 and averaged
   over IoU 0.50:0.95, plus class-agnostic AP ("did we find a defect here at all?").
2. One operating point (a confidence threshold): what an engineer actually sees.
   Each image is matched class-agnostically at IoU >= 0.5, highest confidence first:
       TP      right place, right class
       MISCLS  right place, wrong class  (counts as FP for the predicted class
               and FN for the true class in per-class precision/recall)
       FP      a box where there is no defect (false alarm)
       FN      a defect with no box (miss)
   From these: per-class precision/recall/F1, a confusion matrix that includes
   "background" (false alarms and misses), false alarms per image, the miss rate,
   the share of defect-free images that raise any alarm, and recall by defect size.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .boxes import greedy_match, iou_matrix

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


# ------------------------------------------------------------------ AP
def _ap_101(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-conf, kind="stable")
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]  # precision envelope
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def _tp_conf(gt, preds, cls, iou_thr):
    """Per-prediction TP flags (sorted by confidence) and the number of GT boxes."""
    tps, confs, n_gt = [], [], 0
    for img in set(gt) | set(preds):
        g = [b for c, b in gt.get(img, []) if cls is None or c == cls]
        p = [(b, s) for c, b, s in preds.get(img, []) if cls is None or c == cls]
        n_gt += len(g)
        if not p:
            continue
        sc = np.array([s for _, s in p])
        m = greedy_match(iou_matrix([b for b, _ in p], g), sc, iou_thr) if g else {}
        tps += [1.0 if i in m else 0.0 for i in range(len(p))]
        confs += list(sc)
    tp, conf = np.array(tps), np.array(confs)
    order = np.argsort(-conf, kind="stable")
    return tp[order], conf[order], n_gt


def pr_curve(gt, preds, cls: int | None, iou_thr: float = 0.5):
    """(recall, precision, confidence, n_gt) for one class (None = class-agnostic)."""
    tp, conf, n_gt = _tp_conf(gt, preds, cls, iou_thr)
    ctp = np.cumsum(tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.arange(1, len(tp) + 1) if len(tp) else np.array([])
    return rec, prec, conf, n_gt


def average_precision(gt, preds, n_classes: int) -> dict:
    out = {"per_class": {}, "agnostic": {}}
    for c in list(range(n_classes)) + [None]:
        aps = []
        for t in IOU_THRESHOLDS:
            aps.append(_ap_101(*_tp_conf(gt, preds, c, t)))
        d = {"AP50": aps[0], "AP50_95": float(np.nanmean(aps)), "AP75": aps[5]}
        if c is None:
            out["agnostic"] = d
        else:
            out["per_class"][c] = d
    pc = list(out["per_class"].values())
    out["mAP50"] = float(np.nanmean([d["AP50"] for d in pc]))
    out["mAP50_95"] = float(np.nanmean([d["AP50_95"] for d in pc]))
    return out


# ------------------------------------------------------------------ operating point
def match_image(gts, preds, iou_thr: float = 0.5) -> list[dict]:
    """Outcome per box for one image (see module docstring)."""
    pb = [b for _, b, _ in preds]
    sc = np.array([s for _, _, s in preds]) if preds else np.zeros(0)
    m = greedy_match(iou_matrix(pb, [b for _, b in gts]), sc, iou_thr) if preds and gts else {}
    out = []
    for i, (c, b, s) in enumerate(preds):
        if i in m:
            gc, gb = gts[m[i]]
            kind = "TP" if gc == c else "MISCLS"
            out.append({"kind": kind, "box": b, "cls": c, "conf": float(s), "gt_cls": gc, "gt_box": gb})
        else:
            out.append({"kind": "FP", "box": b, "cls": c, "conf": float(s), "gt_cls": None})
    hit = set(m.values())
    for j, (gc, gb) in enumerate(gts):
        if j not in hit:
            out.append({"kind": "FN", "box": gb, "cls": gc, "conf": None, "gt_cls": gc})
    return out


def filter_conf(preds, thr: float):
    return {k: [p for p in v if p[2] >= thr] for k, v in preds.items()}


def operating_point(gt, preds, classes: list[str], conf_thr: float, iou_thr: float = 0.5,
                    clean_ids: set[str] | None = None, size_edges: tuple[float, float] | None = None) -> dict:
    n = len(classes)
    P = filter_conf(preds, conf_thr)
    tp, fp, fn = np.zeros(n), np.zeros(n), np.zeros(n)
    cm = np.zeros((n + 1, n + 1), dtype=int)  # rows true (last = background), cols predicted (last = missed)
    per_image, size_hits = {}, defaultdict(lambda: [0, 0])
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for v in gt.values() for _, b in v]
    if size_edges is None and areas:
        size_edges = tuple(np.percentile(areas, [33.3, 66.7]))
    for img in sorted(set(gt) | set(P)):
        res = match_image(gt.get(img, []), P.get(img, []), iou_thr)
        per_image[img] = res
        for r in res:
            k = r["kind"]
            if k == "TP":
                tp[r["cls"]] += 1
                cm[r["cls"], r["cls"]] += 1
            elif k == "MISCLS":
                fp[r["cls"]] += 1
                fn[r["gt_cls"]] += 1
                cm[r["gt_cls"], r["cls"]] += 1
            elif k == "FP":
                fp[r["cls"]] += 1
                cm[n, r["cls"]] += 1
            else:
                fn[r["cls"]] += 1
                cm[r["cls"], n] += 1
            if k in ("TP", "MISCLS", "FN"):
                b = r["gt_box"] if k != "FN" else r["box"]
                a = (b[2] - b[0]) * (b[3] - b[1])
                bucket = "small" if a < size_edges[0] else "medium" if a < size_edges[1] else "large"
                size_hits[bucket][0] += k != "FN"   # localised (any class)
                size_hits[bucket][1] += 1

    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / np.maximum(tp + fn, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    n_gt = TP + FN
    n_fa = sum(r["kind"] == "FP" for v in per_image.values() for r in v)
    n_miss = sum(r["kind"] == "FN" for v in per_image.values() for r in v)
    n_mis = sum(r["kind"] == "MISCLS" for v in per_image.values() for r in v)
    out = {
        "conf_threshold": conf_thr, "iou_threshold": iou_thr,
        "per_class": {classes[c]: {"precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c]),
                                   "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c])} for c in range(n)},
        "micro": {"precision": float(TP / max(TP + FP, 1e-9)), "recall": float(TP / max(n_gt, 1e-9)),
                  "f1": float(2 * TP / max(2 * TP + FP + FN, 1e-9))},
        "macro_f1": float(f1.mean()),
        "counts": {"gt_defects": int(n_gt), "correct": int(TP), "wrong_class": int(n_mis),
                   "false_alarms": int(n_fa), "missed": int(n_miss), "images": len(per_image)},
        "rates": {
            "miss_rate": float(n_miss / max(n_gt, 1)),                    # defects with no box at all
            "localisation_recall": float((n_gt - n_miss) / max(n_gt, 1)),  # found, any class
            "wrong_class_rate": float(n_mis / max(n_gt - n_miss, 1)),     # of found defects
            "false_alarms_per_image": float(n_fa / max(len(per_image), 1)),
            "false_discovery_rate": float(FP / max(TP + FP, 1e-9)),
        },
        "recall_by_size": {k: {"localised": v[0], "total": v[1], "recall": v[0] / max(v[1], 1)}
                           for k, v in sorted(size_hits.items())},
        "size_edges_px2": [float(x) for x in size_edges] if size_edges else None,
        "confusion": cm.tolist(),
        "confusion_labels": classes + ["background"],
    }
    if clean_ids:
        alarms = [img for img in clean_ids if any(r["kind"] == "FP" for r in per_image.get(img, []))]
        out["clean_images"] = {"n": len(clean_ids), "with_false_alarm": len(alarms),
                               "false_alarm_image_rate": len(alarms) / max(len(clean_ids), 1)}
    out["_per_image"] = per_image
    return out


def best_f1_threshold(gt, preds, classes, iou_thr=0.5, grid=None) -> tuple[float, float]:
    """Confidence threshold that maximises micro-F1 (pick it on VAL, then freeze it for test)."""
    best = (0.25, -1.0)
    for t in grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.05), 2):
        f = operating_point(gt, preds, classes, float(t), iou_thr)["micro"]["f1"]
        if f > best[1]:
            best = (float(t), f)
    return best
