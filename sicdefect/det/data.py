"""Curated detection datasets in one canonical layout.

    data/det/<name>/
        images/<split>/<id>.png        tested images (8-bit grayscale)
        references/<split>/<id>.png    defect-free reference, when the dataset has one
        labels/<split>/<id>.txt        YOLO boxes: class cx cy w h (normalised 0-1)
        manifest.csv                   one row per image: id, split, group, size, boxes, source
        classes.json                   class names in id order
        stats.json + stats_*.png       baseline statistics (written by dataset_stats)
        DATASET_CARD.md                generated documentation
        views/<mode>/                  model-input images per preprocessing mode (build_view)

Converters:
    convert_deeppcb   DeepPCB (PCB AOI, 6 defect types, template images)
    convert_mvtec     MVTec AD pixel masks -> one box per connected defect region
    convert_yolo      any YOLO-format export (CVAT, Label Studio, Roboflow), e.g. your own SEM images
"""
from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .preprocess import make_input, read_gray

SPLITS = ("train", "val", "test")
DEEPPCB_CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ------------------------------------------------------------------ helpers
def xyxy_to_yolo(b, w, h):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def yolo_to_xyxy(b, w, h):
    cx, cy, bw, bh = b
    return (cx * w - bw * w / 2, cy * h - bh * h / 2, cx * w + bw * w / 2, cy * h + bh * h / 2)


def write_labels(path: Path, boxes: list[tuple[int, tuple]], w: int, h: int) -> None:
    lines = [f"{c} " + " ".join(f"{v:.6f}" for v in xyxy_to_yolo(b, w, h)) for c, b in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_labels(path: Path, w: int, h: int) -> list[tuple[int, tuple]]:
    """YOLO label file -> [(class_id, (x1, y1, x2, y2) in pixels)]."""
    out = []
    if not Path(path).exists():
        return out
    for ln in Path(path).read_text().split("\n"):
        p = ln.split()
        if len(p) >= 5:
            out.append((int(p[0]), yolo_to_xyxy(tuple(map(float, p[1:5])), w, h)))
    return out


def _reset(out: Path) -> None:
    for sub in ("images", "references", "labels", "views"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)


def _split_by_group(ids: list[str], groups: list[str], val: float, test: float, seed: int) -> dict[str, str]:
    """Random split within each group, so every group is represented in each split."""
    rng = np.random.default_rng(seed)
    by_g = defaultdict(list)
    for i, g in zip(ids, groups):
        by_g[g].append(i)
    split = {}
    for g in sorted(by_g):
        members = rng.permutation(sorted(by_g[g])).tolist()
        n_te, n_va = round(len(members) * test), round(len(members) * val)
        for k, i in enumerate(members):
            split[i] = "test" if k < n_te else "val" if k < n_te + n_va else "train"
    return split


def _holdout_by_group(ids: list[str], groups: list[str], val: float, test: float, seed: int) -> dict[str, str]:
    """Whole groups go to one split (no sample appears in two splits): for near-duplicate
    shots of the same die / particle / lot."""
    rng = np.random.default_rng(seed)
    by_g = defaultdict(list)
    for i, g in zip(ids, groups):
        by_g[g].append(i)
    order = rng.permutation(sorted(by_g)).tolist()
    n, split, done = len(ids), {}, {"test": 0, "val": 0}
    for g in order:
        target = "test" if done["test"] < test * n else "val" if done["val"] < val * n else "train"
        for i in by_g[g]:
            split[i] = target
        if target != "train":
            done[target] += len(by_g[g])
    return split


def _finish(out: Path, rows: list[dict], classes: list[str], source: str) -> Path:
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "classes.json").write_text(json.dumps({"classes": classes, "source": source}, indent=2))
    return out


# ------------------------------------------------------------------ DeepPCB
def convert_deeppcb(src: str | Path, out: str | Path, val: float = 0.15, seed: int = 42,
                    clean_test: int = 100) -> Path:
    """DeepPCB -> canonical layout.

    Keeps the official 1000/500 trainval/test split (so results compare with the
    paper) and carves a validation set out of trainval, stratified by board group.

    Every DeepPCB image contains defects, so on its own the test set cannot show how
    often the model raises alarms on a good board. ``clean_test`` defect-free template
    images (from test-split boards) are added to the test split as clean controls
    (id suffix ``_clean``, group ``clean``, no boxes) to measure that.
    """
    src, out = Path(src), Path(out)
    root = src / "PCBData" if (src / "PCBData").exists() else src
    _reset(out)
    (out / "references").mkdir(exist_ok=True)
    for s in SPLITS:
        (out / "references" / s).mkdir(parents=True, exist_ok=True)

    entries = []
    for list_file, official in (("trainval.txt", "trainval"), ("test.txt", "test")):
        for ln in (root / list_file).read_text().split("\n"):
            if not ln.strip():
                continue
            img_rel, ann_rel = ln.split()
            stem = Path(img_rel).stem
            group = img_rel.split("/")[0].replace("group", "")
            entries.append((stem, group, official, root / Path(img_rel).parent, root / ann_rel))

    tv = [e for e in entries if e[2] == "trainval"]
    val_split = _split_by_group([e[0] for e in tv], [e[1] for e in tv], val, 0.0, seed)

    rows = []
    for stem, group, official, img_dir, ann in entries:
        split = "test" if official == "test" else val_split[stem]
        test_img = read_gray(img_dir / f"{stem}_test.jpg")
        ref_img = read_gray(img_dir / f"{stem}_temp.jpg")
        h, w = test_img.shape
        boxes = []
        for a in ann.read_text().split("\n"):
            p = a.replace(",", " ").split()
            if len(p) >= 5:
                x1, y1, x2, y2, t = map(int, map(float, p[:5]))
                if t >= 1:
                    x1, x2 = sorted((max(0, x1), min(w, x2)))
                    y1, y2 = sorted((max(0, y1), min(h, y2)))
                    if x2 - x1 >= 2 and y2 - y1 >= 2:
                        boxes.append((t - 1, (x1, y1, x2, y2)))
        cv2.imwrite(str(out / "images" / split / f"{stem}.png"), test_img)
        cv2.imwrite(str(out / "references" / split / f"{stem}.png"), ref_img)
        write_labels(out / "labels" / split / f"{stem}.txt", boxes, w, h)
        rows.append({"id": stem, "split": split, "group": group, "width": w, "height": h,
                     "n_boxes": len(boxes), "has_reference": 1,
                     "source": str((img_dir / f"{stem}_test.jpg").relative_to(src))})

    test_entries = [e for e in entries if e[2] == "test"]
    pick = np.random.default_rng(seed).permutation(len(test_entries))[:clean_test]
    for k in sorted(pick):
        stem, group, _, img_dir, _ = test_entries[k]
        ref_img = read_gray(img_dir / f"{stem}_temp.jpg")
        h, w = ref_img.shape
        cid = f"{stem}_clean"
        cv2.imwrite(str(out / "images" / "test" / f"{cid}.png"), ref_img)
        cv2.imwrite(str(out / "references" / "test" / f"{cid}.png"), ref_img)
        write_labels(out / "labels" / "test" / f"{cid}.txt", [], w, h)
        rows.append({"id": cid, "split": "test", "group": "clean", "width": w, "height": h, "n_boxes": 0,
                     "has_reference": 1, "source": str((img_dir / f"{stem}_temp.jpg").relative_to(src))})
    return _finish(out, rows, DEEPPCB_CLASSES, "DeepPCB (github.com/tangsanli5201/DeepPCB, MIT, research use)")


# ------------------------------------------------------------------ MVTec AD
def mask_to_boxes(mask: np.ndarray, min_area: int = 16) -> list[tuple[int, int, int, int]]:
    """Binary defect mask -> one box per connected region."""
    n, _, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return [(int(x), int(y), int(x + bw), int(y + bh)) for x, y, bw, bh, a in st[1:] if a >= min_area]


def convert_mvtec(src: str | Path, out: str | Path, categories: list[str] | None = None,
                  val: float = 0.2, test: float = 0.3, seed: int = 42, size: int = 640,
                  good_per_category: int = 20) -> Path:
    """MVTec AD -> detection dataset. Class = '<category>_<defect>' (e.g. 'grid_broken').

    MVTec's own split is for anomaly detection (train = good only), so for supervised
    detection the *test* images are re-split here, grouped by category. A few 'good'
    images per category are kept as background (no boxes) to measure false positives.
    """
    src, out = Path(src), Path(out)
    cats = categories or sorted(d.name for d in src.iterdir() if (d / "ground_truth").exists())
    _reset(out)
    classes, items = [], []
    for cat in cats:
        for ddir in sorted((src / cat / "test").iterdir()):
            if ddir.name == "good":
                goods = sorted(ddir.glob("*.png"))[:good_per_category]
                items += [(cat, None, f, None) for f in goods]
                continue
            cname = f"{cat}_{ddir.name}"
            classes.append(cname)
            for f in sorted(ddir.glob("*.png")):
                items.append((cat, cname, f, src / cat / "ground_truth" / ddir.name / f"{f.stem}_mask.png"))
    cid = {c: i for i, c in enumerate(classes)}
    ids = [f"{it[0]}_{it[1] or 'good'}_{it[2].stem}" for it in items]
    split = _split_by_group(ids, [it[1] or f"{it[0]}_good" for it in items], val, test, seed)
    rows = []
    for (cat, cname, f, mpath), iid in zip(items, ids):
        img = cv2.resize(read_gray(f), (size, size), interpolation=cv2.INTER_AREA)
        boxes = []
        if cname and mpath.exists():
            m = cv2.resize(cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE), (size, size), interpolation=cv2.INTER_NEAREST)
            boxes = [(cid[cname], b) for b in mask_to_boxes(m)]
        s = split[iid]
        cv2.imwrite(str(out / "images" / s / f"{iid}.png"), img)
        write_labels(out / "labels" / s / f"{iid}.txt", boxes, size, size)
        rows.append({"id": iid, "split": s, "group": cat, "width": size, "height": size,
                     "n_boxes": len(boxes), "has_reference": 0, "source": str(f.relative_to(src))})
    return _finish(out, rows, classes, "MVTec AD (CC BY-NC-SA 4.0), masks converted to boxes")


# ------------------------------------------------------------------ generic YOLO export
def convert_yolo(src: str | Path, out: str | Path, val: float = 0.15, test: float = 0.15,
                 seed: int = 42, classes: list[str] | None = None) -> Path:
    """Any YOLO export: <src>/images/*.png + <src>/labels/*.txt (+ classes.txt or data.yaml).

    Use this for your own labelled images (e.g. SEM artifacts boxed in CVAT or Label Studio).
    Optional ``<src>/references/`` with the same file names (a good-die image of the same
    location) enables the ``refdiff`` view.
    Existing train/val/test sub-folders are respected; otherwise a random split is made,
    grouped by the filename prefix before the first '_' (e.g. sample or lot ID) so
    near-duplicate shots of the same sample stay in the same split.
    """
    src, out = Path(src), Path(out)
    if classes is None:
        if (src / "classes.txt").exists():
            classes = [c.strip() for c in (src / "classes.txt").read_text().split("\n") if c.strip()]
        elif (src / "data.yaml").exists():
            import yaml

            names = yaml.safe_load((src / "data.yaml").read_text())["names"]
            classes = list(names.values()) if isinstance(names, dict) else list(names)
        else:
            raise SystemExit("Need classes.txt or data.yaml in the export folder (or pass classes=)")
    imgs = sorted(f for f in (src / "images").rglob("*") if f.suffix.lower() in IMG_EXTS)
    preset = {f: next((s for s in SPLITS if s in f.relative_to(src / "images").parts), None) for f in imgs}
    ids = [f.stem for f in imgs]
    auto = _holdout_by_group(ids, [i.split("_")[0] for i in ids], val, test, seed)
    _reset(out)
    has_refs = (src / "references").is_dir()
    if has_refs:
        for sp in SPLITS:
            (out / "references" / sp).mkdir(parents=True, exist_ok=True)
    rows = []
    for f, iid in zip(imgs, ids):
        s = preset[f] or auto[iid]
        img = read_gray(f)
        h, w = img.shape
        ref = None
        if has_refs:
            rel = f.relative_to(src / "images")
            ref = next((p for p in [src / "references" / rel] +
                        [src / "references" / rel.parent / f"{f.stem}{e}" for e in IMG_EXTS] if p.exists()), None)
            if ref is not None:
                r = read_gray(ref)
                cv2.imwrite(str(out / "references" / s / f"{iid}.png"),
                            r if r.shape == img.shape else cv2.resize(r, (w, h)))
        lab = next((p for p in [src / "labels" / f.relative_to(src / "images").with_suffix(".txt"),
                                src / "labels" / f"{f.stem}.txt"] if p.exists()), None)
        cv2.imwrite(str(out / "images" / s / f"{iid}.png"), img)
        if lab:
            shutil.copy(lab, out / "labels" / s / f"{iid}.txt")
        else:
            (out / "labels" / s / f"{iid}.txt").write_text("")
        n = len(read_labels(out / "labels" / s / f"{iid}.txt", w, h))
        rows.append({"id": iid, "split": s, "group": iid.split("_")[0], "width": w, "height": h,
                     "n_boxes": n, "has_reference": int(ref is not None), "source": str(f.relative_to(src))})
    return _finish(out, rows, classes, f"YOLO export from {src.name}")


# ------------------------------------------------------------------ loading
def load_manifest(ds: str | Path) -> list[dict]:
    with open(Path(ds) / "manifest.csv") as f:
        return list(csv.DictReader(f))


def load_classes(ds: str | Path) -> list[str]:
    return json.loads((Path(ds) / "classes.json").read_text())["classes"]


def ground_truth(ds: str | Path, split: str) -> dict[str, list[tuple[int, tuple]]]:
    ds = Path(ds)
    return {r["id"]: read_labels(ds / "labels" / split / f"{r['id']}.txt", int(r["width"]), int(r["height"]))
            for r in load_manifest(ds) if r["split"] == split}


def build_view(ds: str | Path, mode: str = "gray") -> Path:
    """Write model-input images for a preprocessing mode + an Ultralytics data.yaml.

    Labels are linked, not copied, so they always match the curated dataset.
    """
    ds = Path(ds).resolve()
    view = ds / "views" / mode
    classes = load_classes(ds)
    for s in SPLITS:
        (view / "images" / s).mkdir(parents=True, exist_ok=True)
        lab = view / "labels" / s
        if lab.is_symlink() or lab.exists():
            if lab.is_symlink():
                lab.unlink()
            else:
                shutil.rmtree(lab)
        lab.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(ds / "labels" / s, lab, target_is_directory=True)
    for r in load_manifest(ds):
        dst = view / "images" / r["split"] / f"{r['id']}.png"
        if dst.exists():
            continue
        img = read_gray(ds / "images" / r["split"] / f"{r['id']}.png")
        ref_p = ds / "references" / r["split"] / f"{r['id']}.png"
        ref = read_gray(ref_p) if ref_p.exists() else None
        cv2.imwrite(str(dst), make_input(img, ref, mode))
    import yaml

    (view / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(view), "train": "images/train", "val": "images/val", "test": "images/test",
         "names": {i: c for i, c in enumerate(classes)}}, sort_keys=False))
    return view / "data.yaml"


# ------------------------------------------------------------------ statistics + card
def dataset_stats(ds: str | Path) -> dict:
    ds = Path(ds)
    classes, rows = load_classes(ds), load_manifest(ds)
    st = {"n_images": len(rows), "classes": classes, "splits": {}, "image_sizes": {}, "box_px": {}}
    sizes = Counter(f"{r['width']}x{r['height']}" for r in rows)
    st["image_sizes"] = dict(sizes)
    per_class_wh = defaultdict(list)
    for s in SPLITS:
        gt = ground_truth(ds, s)
        cnt = Counter(c for bs in gt.values() for c, _ in bs)
        nb = [len(b) for b in gt.values()]
        st["splits"][s] = {
            "images": len(gt), "boxes": int(sum(nb)),
            "images_without_defects": int(sum(n == 0 for n in nb)),
            "boxes_per_image": {"mean": float(np.mean(nb)) if nb else 0, "min": int(min(nb, default=0)),
                                "max": int(max(nb, default=0))},
            "per_class": {classes[k]: int(cnt.get(k, 0)) for k in range(len(classes))},
        }
        for bs in gt.values():
            for c, (x1, y1, x2, y2) in bs:
                per_class_wh[c].append((x2 - x1, y2 - y1))
    for c, wh in per_class_wh.items():
        a = np.array(wh)
        area = a[:, 0] * a[:, 1]
        st["box_px"][classes[c]] = {"w_median": float(np.median(a[:, 0])), "h_median": float(np.median(a[:, 1])),
                                    "area_p5": float(np.percentile(area, 5)), "area_median": float(np.median(area)),
                                    "area_p95": float(np.percentile(area, 95)),
                                    "aspect_median": float(np.median(a[:, 0] / np.maximum(a[:, 1], 1)))}
    (ds / "stats.json").write_text(json.dumps(st, indent=2))
    _plot_stats(ds, st, per_class_wh)
    return st


def _plot_stats(ds: Path, st: dict, per_class_wh) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz import class_colors

    classes = st["classes"]
    cols = class_colors(len(classes), rgb01=True)
    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.9), 4))
    xs = np.arange(len(classes))
    for k, s in enumerate(SPLITS):
        v = [st["splits"][s]["per_class"][c] for c in classes]
        ax.bar(xs + (k - 1) * 0.27, v, 0.27, label=f"{s} ({st['splits'][s]['images']} images)")
    ax.set_xticks(xs, classes, rotation=30 if len(classes) > 8 else 0, ha="right" if len(classes) > 8 else "center")
    ax.set_ylabel("boxes")
    ax.set_title("Defect instances per class and split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ds / "stats_class_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c, wh in sorted(per_class_wh.items()):
        a = np.array(wh)
        ax.scatter(a[:, 0], a[:, 1], s=6, alpha=0.4, color=cols[c], label=classes[c])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("box width (px)")
    ax.set_ylabel("box height (px)")
    ax.set_title("Defect box sizes")
    ax.legend(markerscale=3, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(ds / "stats_box_sizes.png", dpi=120)
    plt.close(fig)

    from .viz import draw_boxes

    rows = [r for r in load_manifest(ds) if r["split"] == "train" and int(r["n_boxes"]) > 0][:8]
    tiles = []
    for r in rows:
        img = cv2.imread(str(ds / "images" / "train" / f"{r['id']}.png"))
        gt = read_labels(ds / "labels" / "train" / f"{r['id']}.txt", int(r["width"]), int(r["height"]))
        img = draw_boxes(img, [(b, c, None) for c, b in gt], classes)
        tiles.append(cv2.resize(img, (320, 320)))
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        cv2.imwrite(str(ds / "stats_samples.png"), grid)


def write_card(ds: str | Path, st: dict | None = None) -> Path:
    ds = Path(ds)
    st = st or json.loads((ds / "stats.json").read_text())
    src = json.loads((ds / "classes.json").read_text())["source"]
    cl = st["classes"]
    L = [f"# Dataset card: {ds.name}", "", f"Source: {src}", "",
         f"Generated by `prepare_detection.py`. Images: {st['n_images']}. "
         f"Image sizes: {', '.join(f'{k} ({v})' for k, v in st['image_sizes'].items())}.", "",
         "## Splits", "", "| split | images | boxes | images without defects | boxes/image (mean, min-max) |",
         "|---|---|---|---|---|"]
    for s in SPLITS:
        d = st["splits"][s]
        b = d["boxes_per_image"]
        L.append(f"| {s} | {d['images']} | {d['boxes']} | {d['images_without_defects']} | "
                 f"{b['mean']:.1f} ({b['min']}-{b['max']}) |")
    L += ["", "## Class distribution (boxes)", "", "| class | " + " | ".join(SPLITS) + " | total | share |",
          "|---|" + "---|" * (len(SPLITS) + 2)]
    tot_all = sum(st["splits"][s]["boxes"] for s in SPLITS) or 1
    for c in cl:
        v = [st["splits"][s]["per_class"][c] for s in SPLITS]
        L.append(f"| {c} | " + " | ".join(map(str, v)) + f" | {sum(v)} | {sum(v) / tot_all:.1%} |")
    L += ["", "## Box sizes (pixels)", "", "| class | median w x h | area p5 / median / p95 | median aspect w/h |",
          "|---|---|---|---|"]
    for c in cl:
        if c in st["box_px"]:
            b = st["box_px"][c]
            L.append(f"| {c} | {b['w_median']:.0f} x {b['h_median']:.0f} | {b['area_p5']:.0f} / "
                     f"{b['area_median']:.0f} / {b['area_p95']:.0f} | {b['aspect_median']:.2f} |")
    L += ["", "![class distribution](stats_class_distribution.png)", "", "![box sizes](stats_box_sizes.png)",
          "", "![samples](stats_samples.png)", ""]
    p = ds / "DATASET_CARD.md"
    p.write_text("\n".join(L))
    return p
