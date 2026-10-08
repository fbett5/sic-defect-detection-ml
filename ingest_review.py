"""Turn an engineer's exported review decisions into labels and an audit summary.

  python ingest_review.py --decisions decisions_lot42.json --out outputs/runs/lot42/reviewed

Writes
  labels/<image>.txt   YOLO labels of the engineer-confirmed defects (accepted, relabelled, added)
  images/<image>.png   copies of the reviewed images, so the folder is a ready YOLO export:
                         python prepare_detection.py --dataset yolo --src <out> --out data/det/<name>
                       (merge with the existing data to retrain: the active-learning loop)
  images.txt           the original image paths
  review_summary.json  how much the engineer had to correct:
                         accepted as-is, relabelled, rejected (false alarms), added (misses),
                         and the model's precision/recall as judged by the engineer
  review_log.csv       one row per box and decision, for traceability

Only images the engineer marked as reviewed are written (an unreviewed image would
otherwise become a wrong "no defects" label). Boxes left "undecided" are counted separately.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

from sicdefect.det.data import write_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="also write images not marked reviewed (their undecided boxes are dropped, so their "
                         "labels may be incomplete: do not train on them)")
    args = ap.parse_args()

    d = json.loads(Path(args.decisions).read_text())
    classes = d["classes"]
    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    c = dict(accepted=0, relabelled=0, rejected=0, added=0, undecided=0, images=0, images_skipped=0)
    per_class = {k: dict(confirmed=0, rejected=0, relabelled_from=0, added=0) for k in classes}
    log, files = [], []
    for im in d["images"]:
        if not args.include_unreviewed and not im["reviewed"]:
            c["images_skipped"] += 1
            continue
        c["images"] += 1
        keep = []
        for b in im["boxes"]:
            dec = b["decision"]
            if dec == "not_shown":
                continue
            row = {"image": im["id"], "box": " ".join(f"{v:.1f}" for v in b["box"]), "model_class": classes[b["cls"]],
                   "final_class": classes[b["final_cls"]], "score": round(b["score"], 4), "decision": dec}
            if dec == "accepted":
                keep.append((b["final_cls"], tuple(b["box"])))
                if b["final_cls"] != b["cls"]:
                    c["relabelled"] += 1
                    per_class[classes[b["cls"]]]["relabelled_from"] += 1
                    row["decision"] = "relabelled"
                else:
                    c["accepted"] += 1
                    per_class[classes[b["cls"]]]["confirmed"] += 1
            elif dec == "rejected":
                c["rejected"] += 1
                per_class[classes[b["cls"]]]["rejected"] += 1
            else:
                c["undecided"] += 1
            log.append(row)
        for a in im.get("added", []):
            keep.append((a["cls"], tuple(a["box"])))
            c["added"] += 1
            per_class[classes[a["cls"]]]["added"] += 1
            log.append({"image": im["id"], "box": " ".join(f"{v:.1f}" for v in a["box"]), "model_class": "",
                        "final_class": classes[a["cls"]], "score": "", "decision": "added (missed by model)"})
        write_labels(out / "labels" / f"{im['id']}.txt", keep, im["width"], im["height"])
        files.append(im["file"])
        src = Path(im["file"])
        if src.exists():
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(out / "images" / f"{im['id']}.png"), img)

    shown = c["accepted"] + c["relabelled"] + c["rejected"]
    real = c["accepted"] + c["relabelled"] + c["added"]
    summary = {"batch": d["batch"], "reviewer": d.get("reviewer"), "exported": d.get("exported"), **c,
               "model_precision_as_judged": (c["accepted"] + c["relabelled"]) / max(shown, 1),
               "model_class_accuracy_as_judged": c["accepted"] / max(c["accepted"] + c["relabelled"], 1),
               "model_recall_as_judged": (c["accepted"] + c["relabelled"]) / max(real, 1),
               "per_class": per_class}
    (out / "review_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "images.txt").write_text("\n".join(files) + "\n")
    (out / "classes.txt").write_text("\n".join(classes) + "\n")
    if log:
        with open(out / "review_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader()
            w.writerows(log)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, indent=1))
    print(f"Labels: {out / 'labels'}")


if __name__ == "__main__":
    main()
