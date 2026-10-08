"""Run the defect pipeline on new images and prepare them for engineer review.

    image (+ reference) -> preprocess -> YOLO -> crops -> CNN -> review page + batch report

Examples:
  # a folder of new images, references with the same file names in another folder
  python run_pipeline.py --det outputs/det/yolov8n-refdiff/weights/best.pt \
      --cnn outputs/cnn/cnn-resnet18-refdiff-hardneg/model.pt --view refdiff \
      --input new_boards/ --refs golden/ --out outputs/runs/lot42

  # demo on 20 test images from the curated dataset
  python run_pipeline.py --det ... --cnn ... --view refdiff --from-dataset data/det/deeppcb --n 20 \
      --out outputs/runs/demo

Output folder:
  review/index.html    open in a browser: accept / reject / relabel boxes, draw missed defects,
                       export decisions_<batch>.json  (then: python ingest_review.py ...)
  batch_report.html    what was found in each image, with overlays
  results.json         every candidate box with YOLO and CNN opinions, score and review status
  summary.csv          one row per image
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from sicdefect.det.data import load_manifest
from sicdefect.det.pipeline import Pipeline, Thresholds, find_reference
from sicdefect.det.preprocess import read_gray
from sicdefect.det.viz import draw_boxes

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STATUS_BGR = {"auto": (87, 139, 46), "review": (0, 138, 217)}


def collect(args) -> list[tuple[str, Path, Path | None]]:
    if args.from_dataset:
        ds = Path(args.from_dataset)
        rows = [r for r in load_manifest(ds) if r["split"] == args.split]
        rng = np.random.default_rng(args.seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), size=min(args.n, len(rows)), replace=False))]
        out = []
        for r in rows:
            ref = ds / "references" / args.split / f"{r['id']}.png"
            out.append((r["id"], ds / "images" / args.split / f"{r['id']}.png", ref if ref.exists() else None))
        return out
    p = Path(args.input)
    files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS
                                           and not f.stem.endswith(("_ref", "_temp")))
    return [(f.stem, f, find_reference(f, Path(args.refs) if args.refs else None)) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--cnn", default=None)
    ap.add_argument("--view", default="refdiff")
    ap.add_argument("--input", help="image file or folder")
    ap.add_argument("--refs", help="folder of defect-free reference images (same file names)")
    ap.add_argument("--from-dataset", help="take images from a curated dataset instead (demo)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show", type=float, default=None, help="min score to show (default: from metrics.json or 0.25)")
    ap.add_argument("--accept", type=float, default=0.70, help="score to pre-accept when models agree")
    ap.add_argument("--metrics", default=None, help="metrics.json from evaluate_det.py: use its val-tuned threshold")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if not (args.input or args.from_dataset):
        raise SystemExit("Pass --input or --from-dataset")

    mode = "two_stage" if args.cnn else "yolo"
    show = args.show
    if show is None and args.metrics and Path(args.metrics).exists():
        show = json.loads(Path(args.metrics).read_text())["thresholds"].get(mode)
        print(f"Using the val-tuned threshold {show:.3f} from {args.metrics}")
    th = Thresholds(show=show if show is not None else 0.25, accept=args.accept)
    pipe = Pipeline(args.det, args.cnn, view=args.view, device=args.device, thresholds=th)
    if args.view == "refdiff" and not args.refs and not args.from_dataset:
        print("NOTE: refdiff needs a reference per image (--refs, or <name>_ref.png next to it)")

    out = Path(args.out)
    batch = out.name
    (out / "review" / "img").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(exist_ok=True)
    classes = pipe.classes
    images, results, summary = [], {}, []
    for img_id, path, ref in collect(args):
        if args.view == "refdiff" and ref is None:
            print(f"skip {img_id}: no reference image")
            continue
        x = pipe.prepare(path, ref)
        boxes = pipe.run(x=x)
        for b in boxes:
            b["status"] = pipe.status(b, mode)
            b["cls"], b["score"] = b.get(f"{mode}_final_cls", b["yolo_cls"]), b.get(f"{mode}_score", b["yolo_conf"])
        boxes = [b for b in boxes if b["status"] != "hidden" or b["score"] >= th.show / 3]  # keep near-misses
        results[img_id] = {"file": str(path), "reference": str(ref) if ref else None, "boxes": boxes}

        tested = read_gray(path)
        h, w = tested.shape
        rd = out / "review" / "img"
        cv2.imwrite(str(rd / f"{img_id}.jpg"), tested, [cv2.IMWRITE_JPEG_QUALITY, 90])
        entry = {"id": img_id, "file": str(path), "width": w, "height": h, "image": f"img/{img_id}.jpg",
                 "boxes": [{k: b.get(k) for k in ("box", "cls", "score", "status", "yolo_cls", "yolo_conf",
                                                  "cnn_cls", "cnn_conf", "p_background")} for b in boxes]}
        if ref is not None:
            r = read_gray(ref)
            r = cv2.resize(r, (w, h)) if r.shape != tested.shape else r
            cv2.imwrite(str(rd / f"{img_id}_ref.jpg"), r, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(rd / f"{img_id}_diff.jpg"), 255 - cv2.absdiff(tested, r), [cv2.IMWRITE_JPEG_QUALITY, 90])
            entry.update(reference=f"img/{img_id}_ref.jpg", diff=f"img/{img_id}_diff.jpg")
        images.append(entry)

        shown = [b for b in boxes if b["status"] != "hidden"]
        over = cv2.cvtColor(tested, cv2.COLOR_GRAY2BGR)
        for st_ in ("review", "auto"):
            over = draw_boxes(over, [(b["box"], b["cls"], b["score"]) for b in shown if b["status"] == st_],
                              classes, color=STATUS_BGR[st_])
        cv2.imwrite(str(out / "overlays" / f"{img_id}.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cnt = Counter(classes[b["cls"]] for b in shown)
        summary.append({"image": img_id, "suspected_defects": len(shown),
                        "pre_accepted": sum(b["status"] == "auto" for b in shown),
                        "needs_review": sum(b["status"] == "review" for b in shown),
                        "classes": "; ".join(f"{k} x{v}" for k, v in cnt.most_common()),
                        "max_score": round(max((b["score"] for b in shown), default=0), 3)})
        print(f"{img_id:20s} {len(shown):3d} suspected ({summary[-1]['needs_review']} to review)  {summary[-1]['classes']}")

    (out / "results.json").write_text(json.dumps({"mode": mode, "classes": classes, "thresholds": vars(th),
                                                  "timing": pipe.timing_summary(), "images": results}, indent=1))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    models = f"{Path(args.det).name}" + (f" + {Path(args.cnn).name}" if args.cnn else "") + f", view {args.view}"
    data = {"batch": batch, "classes": classes, "models": models, "source": args.input or args.from_dataset,
            "images": images}
    tpl = (Path(__file__).parent / "sicdefect" / "det" / "review_page.html").read_text()
    (out / "review" / "index.html").write_text(tpl.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    batch_report(out, batch, summary, classes, pipe.timing_summary(), models, th)
    print(f"\nReview page: {out / 'review' / 'index.html'}\nBatch report: {out / 'batch_report.html'}")


def batch_report(out, batch, summary, classes, timing, models, th):
    from sicdefect.det.report import CSS, FONTS, table

    E = html.escape
    tot = Counter()
    for s in summary:
        for part in filter(None, s["classes"].split("; ")):
            k, v = part.rsplit(" x", 1)
            tot[k] += int(v)
    n_rev = sum(s["needs_review"] for s in summary)
    n_auto = sum(s["pre_accepted"] for s in summary)
    flagged = sum(s["suspected_defects"] > 0 for s in summary)
    p = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
         f"initial-scale=1'><title>Batch {E(batch)}</title>{FONTS}<style>{CSS}</style></head><body><main>",
         f"<h1>Inspection batch: {E(batch)}</h1><p class=mut>{E(models)}. Boxes shown at score ≥ {th.show:.2f}; "
         f"pre-accepted at ≥ {th.accept:.2f} when YOLO and CNN agree.</p><div class=cards>"]
    for v, k in ((len(summary), "images"), (flagged, "images with suspected defects"),
                 (n_auto + n_rev, "suspected defects"), (n_auto, "pre-accepted"), (n_rev, "need engineer review"),
                 (f"{timing.get('total', {}).get('mean_ms', 0):.0f} ms", "per image")):
        p.append(f"<div class=card><b>{v}</b><span>{E(str(k))}</span></div>")
    p.append("</div><p><a href='review/index.html'><b>Open the review page →</b></a></p>")
    p.append("<h2>By defect type</h2>" + table(["class", "suspected"], [[E(k), v] for k, v in tot.most_common()]))
    p.append("<h2>Images</h2><p class=mut>Amber = needs review, green = pre-accepted. Highest-risk images first.</p>"
             "<div class=cases>")
    for s in sorted(summary, key=lambda s: (-s["needs_review"], -s["suspected_defects"])):
        p.append(f"<div class=case><a href='overlays/{E(s['image'])}.jpg'><img src='overlays/{E(s['image'])}.jpg' "
                 f"loading=lazy alt='{E(s['image'])}'></a><b>{E(s['image'])}</b><br>{s['suspected_defects']} suspected, "
                 f"<span class={'bad' if s['needs_review'] else 'ok'}>{s['needs_review']} to review</span><br>"
                 f"<span class=mut>{E(s['classes']) or 'nothing found'}</span></div>")
    p.append("</div><p class=mut>Generated by run_pipeline.py. Data: results.json, summary.csv.</p></main></body></html>")
    (out / "batch_report.html").write_text("\n".join(p))


if __name__ == "__main__":
    main()
