"""Track 3, step 4: benchmark the detection pipeline and write the automated report.

Examples:
  python evaluate_det.py --det outputs/det/yolov8n-refdiff/weights/best.pt \
      --cnn outputs/cnn/cnn-resnet18-refdiff-hardneg/model.pt --view refdiff \
      --compare outputs/det/yolov8n-gray/weights/best.pt:gray --name deeppcb

What it does
  1. runs the pipeline on every val and test image (predictions cached as JSON)
  2. for each approach (YOLO only, YOLO + CNN) picks the confidence threshold with the
     best F1 on VAL and freezes it, then scores TEST once at that threshold
  3. computes mAP@0.5, mAP@0.5:0.95, per-class precision/recall/F1, confusion matrix with
     background, false alarms per image, miss rate, false alarms on defect-free boards,
     recall by defect size, and inference time per stage
  4. compares the approaches box by box: which false alarms the CNN removed and which
     real defects it wrongly rejected
  5. writes outputs/eval/<name>/report.html, metrics.json, cases.csv and all figures
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from sicdefect.det import report as R
from sicdefect.det.crops import model_input
from sicdefect.det.data import ground_truth, load_classes, load_manifest
from sicdefect.det.evaluate import average_precision, filter_conf, operating_point
from sicdefect.det.pipeline import Pipeline, as_eval_preds
from sicdefect.utils import ROOT, get_device


def run_split(pipe: Pipeline, ds: Path, split: str, view: str, cache: Path, reuse: bool) -> dict:
    if reuse and cache.exists():
        return json.loads(cache.read_text())
    ids = [r["id"] for r in load_manifest(ds) if r["split"] == split]
    res = {}
    for k, i in enumerate(ids):
        res[i] = pipe.run(x=model_input(ds, split, i, view))
        if k % 100 == 0:
            print(f"  {split}: {k}/{len(ids)}")
    cache.write_text(json.dumps(res))
    return res


def sweep(gt, preds, classes, clean):
    out = []
    for t in np.round(np.arange(0.05, 0.96, 0.025), 3):
        op = operating_point(gt, preds, classes, float(t), clean_ids=clean)
        out.append({"t": float(t), **op["micro"], "fa_per_img": op["rates"]["false_alarms_per_image"]})
    return out


def compare_modes(gt, res, classes, thr_y, thr_t):
    """Box-level effect of adding the CNN, at each approach's own threshold."""
    py = filter_conf(as_eval_preds(res, "yolo"), thr_y)
    pt = filter_conf(as_eval_preds(res, "two_stage"), thr_t)
    oy = operating_point(gt, py, classes, 0.0)["_per_image"]
    ot = operating_point(gt, pt, classes, 0.0)["_per_image"]
    removed_fa = added_fa = lost = gained = fixed_cls = broke_cls = 0
    lost_items, removed_items = [], []
    for img in gt:
        def gt_state(outs):
            st = {}
            for r in outs:
                if r["kind"] in ("TP", "MISCLS"):
                    st[tuple(r["gt_box"])] = r["kind"]
            return st
        sy, st_ = gt_state(oy.get(img, [])), gt_state(ot.get(img, []))
        for c, b in gt[img]:
            a, z = sy.get(tuple(b)), st_.get(tuple(b))
            if a and not z:
                lost += 1
                lost_items.append((img, {"kind": "FN", "box": b, "cls": c, "conf": None, "gt_cls": c}))
            if z and not a:
                gained += 1
            if a == "MISCLS" and z == "TP":
                fixed_cls += 1
            if a == "TP" and z == "MISCLS":
                broke_cls += 1
        fy = [r for r in oy.get(img, []) if r["kind"] == "FP"]
        ft = [r for r in ot.get(img, []) if r["kind"] == "FP"]
        removed_fa += max(len(fy) - len(ft), 0)
        added_fa += max(len(ft) - len(fy), 0)
        tb = [r["box"] for r in ft]
        for r in fy:
            if r["box"] not in tb:
                removed_items.append((img, r))
    return {"false_alarms_removed": removed_fa, "false_alarms_added": added_fa, "defects_lost": lost,
            "defects_gained": gained, "class_fixed": fixed_cls, "class_broken": broke_cls,
            "_lost": lost_items, "_removed": removed_items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/det/deeppcb")
    ap.add_argument("--det", required=True, help="YOLO weights")
    ap.add_argument("--cnn", default=None, help="crop CNN (train_crop_cnn.py); enables the two-stage approach")
    ap.add_argument("--view", default="refdiff")
    ap.add_argument("--compare", nargs="*", default=[], help="extra detectors as weights:view, e.g. gray ablation")
    ap.add_argument("--name", default="deeppcb")
    ap.add_argument("--split", default="test")
    ap.add_argument("--reuse", action="store_true", help="reuse cached predictions")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-cases", type=int, default=24)
    args = ap.parse_args()

    ds = (ROOT / args.dataset).resolve()
    out = ROOT / "outputs" / "eval" / args.name
    for d in ("figures", "cases", "failures"):
        (out / d).mkdir(parents=True, exist_ok=True)
    classes = load_classes(ds)
    manifest = load_manifest(ds)
    clean = {r["id"] for r in manifest if r["split"] == args.split and r["group"] == "clean"}
    clean_val = {r["id"] for r in manifest if r["split"] == "val" and r["group"] == "clean"}
    gt = {"val": ground_truth(ds, "val"), args.split: ground_truth(ds, args.split)}
    dev = str(get_device(args.device))

    pipe = Pipeline(args.det, args.cnn, view=args.view, device=args.device)
    res = {s: run_split(pipe, ds, s, args.view, out / f"predictions_{s}.json", args.reuse)
           for s in ("val", args.split)}
    timing = pipe.timing_summary()
    if not timing and (out / "metrics.json").exists():  # --reuse: keep the timing measured last time
        timing = json.loads((out / "metrics.json").read_text()).get("timing", {})
    modes = ["yolo"] + (["two_stage"] if args.cnn else [])

    chosen, sweeps, aps, ops = {}, {}, {}, {}
    for m in modes:
        pv = as_eval_preds(res["val"], m)
        sweeps[m] = sweep(gt["val"], pv, classes, clean_val)
        chosen[m] = max(sweeps[m], key=lambda s: s["f1"])["t"]
        pt = as_eval_preds(res[args.split], m)
        aps[m] = average_precision(gt[args.split], pt, len(classes))
        ops[m] = operating_point(gt[args.split], pt, classes, chosen[m], clean_ids=clean)
        print(f"{m:10s} thr={chosen[m]:.3f}  mAP50={aps[m]['mAP50']:.4f}  mAP50-95={aps[m]['mAP50_95']:.4f}  "
              f"P={ops[m]['micro']['precision']:.4f} R={ops[m]['micro']['recall']:.4f} F1={ops[m]['micro']['f1']:.4f}")
    best = modes[-1]

    # ---------------- preprocessing ablation (other detectors, YOLO only)
    extra_tables, ablation = [], {}
    for spec in args.compare:
        w, v = spec.rsplit(":", 1)
        p2 = Pipeline(w, None, view=v, device=args.device)
        r2 = {s: run_split(p2, ds, s, v, out / f"predictions_{s}_{v}.json", args.reuse) for s in ("val", args.split)}
        pv = as_eval_preds(r2["val"], "yolo")
        t2 = max(sweep(gt["val"], pv, classes, clean_val), key=lambda s: s["f1"])["t"]
        pt = as_eval_preds(r2[args.split], "yolo")
        a2 = average_precision(gt[args.split], pt, len(classes))
        o2 = operating_point(gt[args.split], pt, classes, t2, clean_ids=clean)
        ablation[v] = {"weights": w, "threshold": t2, "ap": a2, "op": {k: o2[k] for k in o2 if k != "_per_image"}}
    if ablation:
        rows = [[f"{args.view} (main)", f"{aps['yolo']['mAP50']:.3f}", f"{aps['yolo']['mAP50_95']:.3f}",
                 f"{ops['yolo']['micro']['f1']:.3f}", R.pct(ops["yolo"]["rates"]["miss_rate"]),
                 f"{ops['yolo']['rates']['false_alarms_per_image']:.2f}",
                 R.pct(ops["yolo"].get("clean_images", {}).get("false_alarm_image_rate", 0))]]
        for v, a in ablation.items():
            rows.append([v, f"{a['ap']['mAP50']:.3f}", f"{a['ap']['mAP50_95']:.3f}", f"{a['op']['micro']['f1']:.3f}",
                         R.pct(a["op"]["rates"]["miss_rate"]), f"{a['op']['rates']['false_alarms_per_image']:.2f}",
                         R.pct(a["op"].get("clean_images", {}).get("false_alarm_image_rate", 0))])
        extra_tables.append({"title": "Preprocessing ablation (YOLO only, same training recipe)",
                             "header": ["input view", "mAP50", "mAP50-95", "F1", "missed", "false alarms / image",
                                        "good boards alarmed"], "rows": rows,
                             "note": "refdiff = tested image + defect-free reference + their difference; "
                                     "gray = tested image alone."})

    cmp_ = None
    if "two_stage" in modes:
        cmp_ = compare_modes(gt[args.split], res[args.split], classes, chosen["yolo"], chosen["two_stage"])
        extra_tables.append({"title": "What the CNN stage changed (test, box by box)",
                             "header": ["effect", "count"],
                             "rows": [["false alarms removed", cmp_["false_alarms_removed"]],
                                      ["false alarms added", cmp_["false_alarms_added"]],
                                      ["real defects lost (CNN rejected or score fell below threshold)",
                                       cmp_["defects_lost"]],
                                      ["real defects gained", cmp_["defects_gained"]],
                                      ["wrong class fixed", cmp_["class_fixed"]],
                                      ["right class broken", cmp_["class_broken"]]]})
    cnn_metrics = None
    if args.cnn and (Path(args.cnn).parent / "test_metrics.json").exists():
        cnn_metrics = json.loads((Path(args.cnn).parent / "test_metrics.json").read_text())

    # ---------------- figures
    F = {}
    pt_by_mode = {m: as_eval_preds(res[args.split], m) for m in modes}
    R.fig_pr(gt[args.split], pt_by_mode, classes, out / "figures" / "pr_curves.png")
    F["pr"] = "figures/pr_curves.png"
    R.fig_threshold(sweeps, out / "figures" / "threshold.png", chosen)
    F["threshold"] = "figures/threshold.png"
    for m in modes:
        R.fig_confusion(ops[m], out / "figures" / f"confusion_{m}.png",
                        f"{R.MODE_LABEL[m]}: test boxes at threshold {chosen[m]:.2f}")
        F[f"confusion_{m}"] = f"figures/confusion_{m}.png"
    R.fig_per_class(ops, classes, out / "figures" / "per_class.png")
    F["per_class"] = "figures/per_class.png"
    if timing:
        R.fig_timing(timing, out / "figures" / "timing.png")
        F["timing"] = "figures/timing.png"

    cache = {}

    def load(img_id):
        if img_id not in cache:
            cache[img_id] = model_input(ds, args.split, img_id, args.view)[..., 0]  # tested-image channel
            cache[img_id] = cv2.cvtColor(cache[img_id], cv2.COLOR_GRAY2BGR)
        return cache[img_id]

    per = ops[best]["_per_image"]
    allr = [(i, r) for i, v in per.items() for r in v]
    fps = sorted([x for x in allr if x[1]["kind"] == "FP"], key=lambda x: -x[1]["conf"])
    fns = [x for x in allr if x[1]["kind"] == "FN"]
    random.Random(0).shuffle(fns)
    mis = sorted([x for x in allr if x[1]["kind"] == "MISCLS"], key=lambda x: -x[1]["conf"])
    cfp = [x for x in fps if x[0] in clean]
    fails = []
    for items, fname, cap, tf in (
        (fps, "false_alarms.png", "Highest-confidence false alarms",
         lambda i, r: f"{classes[r['cls']]} {r['conf']:.2f}"),
        (fns, "missed.png", "Missed defects (sample)", lambda i, r: f"missed {classes[r['cls']]}"),
        (mis, "wrong_class.png", "Right place, wrong class",
         lambda i, r: f"{classes[r['cls']]} (is {classes[r['gt_cls']]})"),
        (cfp, "clean_board_alarms.png", "False alarms on defect-free boards",
         lambda i, r: f"{classes[r['cls']]} {r['conf']:.2f}"),
        ((cmp_ or {}).get("_removed", []), "cnn_removed.png", "False alarms the CNN removed",
         lambda i, r: f"YOLO said {classes[r['cls']]} {r['conf']:.2f}"),
        ((cmp_ or {}).get("_lost", []), "cnn_lost.png", "Real defects the CNN stage lost",
         lambda i, r: f"lost {classes[r['cls']]}"),
    ):
        if R.gallery(items, load, classes, out / "failures" / fname, tf):
            fails.append((f"failures/{fname}", f"{cap} ({len(items)} total)"))

    # ---------------- cases
    cases, rows_csv = [], []
    for img_id, outs in per.items():
        c = R.outcome_counts(outs)
        rows_csv.append({"image": img_id, "defects": sum(1 for r in outs if r["kind"] in ("TP", "MISCLS", "FN")),
                         "correct": c["tp"], "wrong_class": c["mis"], "false_alarms": c["fp"], "missed": c["fn"],
                         "clean_board": int(img_id in clean),
                         "detail": "; ".join(f"{r['kind']} {classes[r['cls']]}" + (f" {r['conf']:.2f}" if r["conf"] else "")
                                             for r in outs if r["kind"] != "TP")})
    with open(out / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)
    errs = sorted(rows_csv, key=lambda r: -(r["wrong_class"] + r["false_alarms"] + r["missed"]))
    pick = errs[: args.n_cases * 2 // 3] + random.Random(1).sample(
        [r for r in rows_csv if r["wrong_class"] + r["false_alarms"] + r["missed"] == 0] or rows_csv,
        k=min(args.n_cases // 3, len(rows_csv)))
    for r in pick:
        p = out / "cases" / f"{r['image']}.jpg"
        R.case_image(load(r["image"]), per[r["image"]], classes, p)
        cases.append({"id": r["image"], "img": f"cases/{p.name}", "tp": r["correct"], "fp": r["false_alarms"],
                      "fn": r["missed"], "mis": r["wrong_class"], "errors": r["false_alarms"] + r["missed"] + r["wrong_class"],
                      "detail": r["detail"][:160] or "all defects found, correct class"})

    # ---------------- data section
    st = json.loads((ds / "stats.json").read_text())
    for f in ("stats_class_distribution.png", "stats_samples.png"):
        if (ds / f).exists():
            shutil.copy(ds / f, out / "figures" / f)
    src = json.loads((ds / "classes.json").read_text())["source"]
    data_html = (f"<p>{R.html.escape(src)}. {st['n_images']} images; "
                 + ", ".join(f"{s}: {d['images']} images / {d['boxes']} defects" for s, d in st["splits"].items())
                 + f". The test split includes {len(clean)} defect-free boards to measure false alarms. "
                 "Full documentation: <code>DATASET_CARD.md</code> in the dataset folder.</p>"
                 "<img src='figures/stats_class_distribution.png' alt='class distribution'>"
                 "<img src='figures/stats_samples.png' alt='samples'>")
    def rel(p):
        try:
            return str(Path(p).resolve().relative_to(ROOT))
        except ValueError:
            return str(p)

    det_cfg = Path(args.det).parent.parent / "config_used.yaml"
    models = {"detector": f"{rel(args.det)} (input view: {args.view})"
              + (f"; config {rel(det_cfg)}" if det_cfg.exists() else "")}
    if args.cnn:
        models["classifier"] = rel(args.cnn) + (f"; macro-F1 {cnn_metrics['macro_f1']:.3f} on ground-truth test crops"
                                           if cnn_metrics else "")
    if cnn_metrics:
        extra_tables.append({
            "title": "CNN on perfect boxes (ground-truth test crops)",
            "header": ["metric", "value"],
            "rows": [["macro-F1 (defect classes + background)", f"{cnn_metrics['macro_f1']:.3f}"],
                     ["accuracy", R.pct(cnn_metrics["accuracy"])]]
                    + [[f"F1 {k[3:]}", f"{v:.3f}"] for k, v in cnn_metrics.items() if k.startswith("f1_")]})

    # ---------------- auto-written analysis
    o = ops[best]
    pc = o["per_class"]
    worst = min(pc, key=lambda c: pc[c]["f1"])
    bestc = max(pc, key=lambda c: pc[c]["f1"])
    cm = np.array(o["confusion"])
    off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j]
    top_conf = max(off) if off else (0, "", "")
    fa_cls = Counter(classes[r["cls"]] for _, r in fps)
    miss_cls = Counter(classes[r["cls"]] for _, r in fns)
    rs = o.get("recall_by_size", {})
    lines = [f"Best class: <b>{bestc}</b> (F1 {pc[bestc]['f1']:.3f}); weakest: <b>{worst}</b> (F1 {pc[worst]['f1']:.3f}).",
             f"Most frequent confusion: true <b>{top_conf[1]}</b> predicted as <b>{top_conf[2]}</b> ({top_conf[0]} boxes)."
             if top_conf[0] else "No class confusions at this threshold.",
             f"False alarms by predicted class: {dict(fa_cls.most_common())}." if fa_cls else "No false alarms.",
             f"Misses by class: {dict(miss_cls.most_common())}." if miss_cls else "No missed defects."]
    if rs:
        lines.append("Localisation recall by size: " + ", ".join(f"{k} {R.pct(v['recall'])}" for k, v in rs.items()) + ".")
    if "clean_images" in o:
        ci = o["clean_images"]
        lines.append(f"{ci['with_false_alarm']} of {ci['n']} defect-free boards raised at least one alarm.")
    if cmp_:
        lines.append(f"The CNN stage removed {cmp_['false_alarms_removed']} false alarms and added "
                     f"{cmp_['false_alarms_added']}, lost {cmp_['defects_lost']} real defects and gained "
                     f"{cmp_['defects_gained']}, fixed {cmp_['class_fixed']} wrong classes and broke {cmp_['class_broken']}.")
    failure_text = "<br>".join(lines)
    limits_html = (Path(ROOT / "docs" / "limitations_snippet.html").read_text()
                   if (ROOT / "docs" / "limitations_snippet.html").exists() else
                   "<p>See TECHNICAL_ANALYSIS.md in the repository.</p>")

    metrics = {"split": args.split, "modes": modes, "thresholds": chosen, "device": dev, "timing": timing,
               "ap": aps, "ops": {m: {k: v for k, v in ops[m].items() if k != "_per_image"} for m in modes},
               "ablation": ablation, "cnn_effect": {k: v for k, v in (cmp_ or {}).items() if not k.startswith("_")},
               "cnn_gt_crops": cnn_metrics, "val_sweeps": sweeps,
               "weights": {"det": args.det, "cnn": args.cnn}}
    R.save_json(out / "metrics.json", metrics)
    rp = R.write_report(out, {
        "title": f"Defect detection benchmark: {args.name}",
        "subtitle": f"{R.MODE_LABEL[best]} pipeline, {args.split} split, generated automatically",
        "dataset": args.dataset, "classes": classes, "models": models, "modes": modes, "ap": aps, "ops": ops,
        "chosen": chosen, "timing": timing, "device": dev, "figures": F, "failure_figs": fails,
        "failure_text": failure_text, "cases": cases, "split": args.split, "data_html": data_html,
        "cases_note": f"{len(cases)} test images: the ones with the most errors, plus a random sample of perfect ones. "
                      "All images are in cases.csv.", "extra_tables": extra_tables, "limits_html": limits_html})
    print(f"Report: {rp}")


if __name__ == "__main__":
    main()
