"""Figures and the automated HTML report for a detection evaluation."""
from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import numpy as np

from .evaluate import pr_curve
from .viz import OUTCOME_BGR, class_colors, crop_with_context, draw_outcomes, grid, legend_strip

MODE_LABEL = {"yolo": "YOLO only", "two_stage": "YOLO + CNN"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
    return plt


# ------------------------------------------------------------------ figures
def fig_pr(gt, preds_by_mode, classes, path):
    plt = _plt()
    cols = class_colors(len(classes), rgb01=True)
    modes = list(preds_by_mode)
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        for c, name in enumerate(classes):
            rec, prec, _, _ = pr_curve(gt, preds_by_mode[mode], c)
            ax.plot(rec, prec, color=cols[c], lw=1.6, label=name)
        rec, prec, _, _ = pr_curve(gt, preds_by_mode[mode], None)
        ax.plot(rec, prec, color="black", lw=2.2, ls="--", label="any defect (class ignored)")
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(f"Precision-recall at IoU 0.5: {MODE_LABEL[mode]}")
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(op, path, title):
    plt = _plt()
    cm = np.array(op["confusion"])
    labels = op["confusion_labels"]
    xl = labels[:-1] + ["missed"]
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.grid(False)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(xl)), xl, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), [lbl if lbl != "background" else "no defect\n(false alarm)" for lbl in labels])
    for i in range(len(labels)):
        for j in range(len(xl)):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_per_class(ops_by_mode, classes, path):
    plt = _plt()
    modes = list(ops_by_mode)
    x = np.arange(len(classes))
    w = 0.8 / len(modes)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key in zip(axes, ("precision", "recall", "f1")):
        for k, m in enumerate(modes):
            v = [ops_by_mode[m]["per_class"][c][key] for c in classes]
            ax.bar(x + (k - (len(modes) - 1) / 2) * w, v, w, label=MODE_LABEL[m],
                   color=["#4878a8", "#e08a2c"][k % 2])
        ax.set_xticks(x, classes, rotation=25)
        ax.set_ylim(0, 1.05)
        ax.set_title(key.upper() if key == "f1" else key.capitalize())
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Per-class results at the chosen operating threshold (test)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_threshold(sweeps_by_mode, path, chosen):
    plt = _plt()
    fig, axes = plt.subplots(1, len(sweeps_by_mode), figsize=(5.2 * len(sweeps_by_mode), 3.6), squeeze=False)
    for ax, (mode, sw) in zip(axes[0], sweeps_by_mode.items()):
        t = [s["t"] for s in sw]
        for key, col in (("precision", "#4878a8"), ("recall", "#e08a2c"), ("f1", "#2e8b57")):
            ax.plot(t, [s[key] for s in sw], label=key, color=col, lw=1.8)
        ax2 = ax.twinx()
        ax2.plot(t, [s["fa_per_img"] for s in sw], color="#c0392b", ls=":", lw=1.6, label="false alarms / image")
        ax2.set_ylabel("false alarms per image", color="#c0392b")
        ax2.grid(False)
        ax.axvline(chosen[mode], color="gray", ls="--", lw=1)
        ax.set_xlabel("confidence threshold")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Threshold trade-off on val: {MODE_LABEL[mode]}")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_timing(timing, path):
    plt = _plt()
    stages = [s for s in ("preprocess", "detect", "classify") if s in timing]
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    left = 0
    cols = ["#9aa5b1", "#4878a8", "#e08a2c"]
    for s, c in zip(stages, cols):
        v = timing[s]["mean_ms"]
        ax.barh([0], [v], left=left, color=c, label=f"{s} {v:.0f} ms")
        left += v
    ax.set_yticks([])
    ax.set_xlabel("ms per image (mean)")
    ax.set_title("Where inference time goes")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ galleries
def gallery(items, img_loader, classes, path, title_fn, n=24, cols=6, tile=180):
    """items: [(image_id, outcome dict)] -> tiles cropped around the box with context."""
    tiles, caps = [], []
    for img_id, r in items[:n]:
        img = img_loader(img_id)
        over = draw_outcomes(img, [r], classes)
        crop = crop_with_context(over, r["box"], pad=2.0, min_side=60)
        tiles.append(crop)
        caps.append(title_fn(img_id, r))
    if not tiles:
        return False
    g = grid(tiles, cols=cols, tile=tile, captions=caps)
    cv2.imwrite(str(path), np.vstack([legend_strip(g.shape[1]), g]))
    return True


def case_image(img, outcomes, classes, path, max_side=640):
    over = draw_outcomes(img, outcomes, classes)
    h, w = over.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1:
        over = cv2.resize(over, (int(w * f), int(h * f)))
    cv2.imwrite(str(path), over, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ------------------------------------------------------------------ HTML
CSS = """
:root{--bg:#fbfbfa;--fg:#1d2329;--mut:#5f6b76;--line:#dfe3e7;--card:#fff;--accent:#2f6fa3;--bad:#c0392b;--ok:#2e8b57}
@media (prefers-color-scheme:dark){:root{--bg:#15191d;--fg:#e6e9ec;--mut:#9aa5b1;--line:#2c333a;--card:#1c2127;--accent:#79b0dd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:28px 16px 80px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 6px}.mut{color:var(--mut)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}.card b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.card span{font-size:12px;color:var(--mut)}table{border-collapse:collapse;width:100%;margin:8px 0 14px;font-size:13px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600}td.l{text-align:left;white-space:normal;word-break:break-all}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}.cases{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.case{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;font-size:12px}.case b{font-size:13px}.bad{color:var(--bad)}.ok{color:var(--ok)}
.note{background:var(--card);border-left:3px solid var(--accent);padding:8px 12px;margin:10px 0;border-radius:4px}nav a{margin-right:12px;color:var(--accent)}
"""


def table(header, rows):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def pct(v):
    return f"{100 * v:.1f}%"


def write_report(out: Path, ctx: dict) -> Path:
    """ctx keys: title, dataset, classes, models, modes, ap, ops, chosen, timing, figures, cases,
    failures (text), split, n_images."""
    E = html.escape
    modes = ctx["modes"]
    best = modes[-1]
    op, ap = ctx["ops"][best], ctx["ap"][best]
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8>"
             f"<meta name=viewport content='width=device-width,initial-scale=1'><title>{E(ctx['title'])}</title>"
             f"<style>{CSS}</style></head><body><main>",
             f"<h1>{E(ctx['title'])}</h1><p class=mut>{E(ctx['subtitle'])}</p>",
             "<nav><a href=#summary>Summary</a><a href=#data>Data</a><a href=#metrics>Metrics</a>"
             "<a href=#classes>Per class</a><a href=#failures>Failures</a><a href=#cases>Cases</a>"
             "<a href=#limits>Limitations</a></nav>",
             f"<h2 id=summary>Summary ({E(MODE_LABEL[best])}, {E(ctx['split'])} split)</h2><div class=cards>"]
    cards = [(f"{ap['mAP50']:.3f}", "mAP@0.5"), (f"{ap['mAP50_95']:.3f}", "mAP@0.5:0.95"),
             (pct(op["micro"]["precision"]), "precision"), (pct(op["micro"]["recall"]), "recall"),
             (f"{op['micro']['f1']:.3f}", "F1"), (pct(op["rates"]["miss_rate"]), "defects missed"),
             (f"{op['rates']['false_alarms_per_image']:.2f}", "false alarms / image")]
    if "clean_images" in op:
        cards.append((pct(op["clean_images"]["false_alarm_image_rate"]), "good boards with an alarm"))
    if ctx.get("timing"):
        cards.append((f"{ctx['timing']['total']['mean_ms']:.0f} ms", f"per image ({E(ctx['device'])})"))
    parts += [f"<div class=card><b>{v}</b><span>{E(k)}</span></div>" for v, k in cards] + ["</div>"]
    parts.append(f"<p>Operating threshold {op['conf_threshold']:.2f} (chosen on the validation split for best F1, "
                 f"then frozen). IoU 0.5 for a match. {op['counts']['gt_defects']} labelled defects in "
                 f"{op['counts']['images']} images.</p>")

    parts.append("<h2 id=data>Data and models</h2>" + ctx["data_html"])
    parts.append("<table><tbody>" + "".join(f"<tr><td>{E(k)}</td><td class=l>{E(v)}</td></tr>"
                                            for k, v in ctx["models"].items()) + "</tbody></table>")

    rows = []
    for m in modes:
        o, a = ctx["ops"][m], ctx["ap"][m]
        rows.append([MODE_LABEL[m], f"{a['mAP50']:.3f}", f"{a['mAP50_95']:.3f}", f"{a['agnostic']['AP50']:.3f}",
                     f"{o['conf_threshold']:.2f}", pct(o["micro"]["precision"]), pct(o["micro"]["recall"]),
                     f"{o['micro']['f1']:.3f}", o["counts"]["correct"], o["counts"]["wrong_class"],
                     o["counts"]["false_alarms"], o["counts"]["missed"],
                     pct(o["clean_images"]["false_alarm_image_rate"]) if "clean_images" in o else "-"])
    parts.append("<h2 id=metrics>Benchmark</h2>" + table(
        ["approach", "mAP50", "mAP50-95", "AP50 any-defect", "threshold", "precision", "recall", "F1", "correct",
         "wrong class", "false alarms", "missed", "good boards alarmed"], rows))
    for extra in ctx.get("extra_tables", []):
        parts.append(f"<h3>{E(extra['title'])}</h3>" + table(extra["header"], extra["rows"]))
        if extra.get("note"):
            parts.append(f"<p class=note>{extra['note']}</p>")
    for key, cap in (("pr", "Precision-recall curves"), ("threshold", "Choosing the operating threshold"),
                     ("timing", "Inference time")):
        if key in ctx["figures"]:
            parts.append(f"<h3>{cap}</h3><img src='{ctx['figures'][key]}' alt='{cap}'>")

    parts.append("<h2 id=classes>Per defect type</h2>")
    cls_rows = []
    for c in ctx["classes"]:
        r = [E(c)]
        for m in modes:
            pc = ctx["ops"][m]["per_class"][c]
            ci = ctx["classes"].index(c)
            r += [f"{ctx['ap'][m]['per_class'][ci]['AP50']:.3f}", pct(pc["precision"]), pct(pc["recall"]),
                  f"{pc['f1']:.3f}"]
        cls_rows.append(r)
    head = ["class"] + [f"{MODE_LABEL[m]} {k}" for m in modes for k in ("AP50", "P", "R", "F1")]
    parts.append(table(head, cls_rows))
    for key in ("per_class", "confusion_" + best, "confusion_yolo"):
        if key in ctx["figures"] and not (key == "confusion_yolo" and best == "yolo"):
            parts.append(f"<img src='{ctx['figures'][key]}' alt='{key}'><br>")
    rs = op.get("recall_by_size", {})
    if rs:
        parts.append("<h3>Localisation recall by defect size</h3>" + table(
            ["size (by box area tertile)", "found", "total", "recall"],
            [[k, v["localised"], v["total"], pct(v["recall"])] for k, v in rs.items()]))

    parts.append("<h2 id=failures>Failure cases</h2><p class=mut>Green = correct, red = false alarm, orange dashed = "
                 "missed, purple = right place but wrong class. Tiles are cropped around the box.</p>")
    for key, cap in ctx["failure_figs"]:
        parts.append(f"<h3>{E(cap)}</h3><img src='{key}' alt='{E(cap)}'>")
    if ctx.get("failure_text"):
        parts.append("<div class=note>" + ctx["failure_text"] + "</div>")

    parts.append(f"<h2 id=cases>Individual cases</h2><p class=mut>{E(ctx['cases_note'])}</p><div class=cases>")
    for c in ctx["cases"]:
        st = "ok" if c["errors"] == 0 else "bad"
        parts.append(f"<div class=case><a href='{c['img']}'><img src='{c['img']}' loading=lazy alt='{E(c['id'])}'></a>"
                     f"<b>{E(c['id'])}</b><br><span class={st}>{c['tp']} correct, {c['fp']} false alarm, "
                     f"{c['fn']} missed, {c['mis']} wrong class</span><br><span class=mut>{E(c['detail'])}</span></div>")
    parts.append("</div>")
    parts.append("<h2 id=limits>What this can and cannot do</h2>" + ctx["limits_html"])
    parts.append(f"<p class=mut>Generated by evaluate_det.py. Raw numbers: metrics.json.</p></main></body></html>")
    p = out / "report.html"
    p.write_text("\n".join(parts))
    return p


def outcome_counts(res):
    k = [r["kind"] for r in res]
    return {"tp": k.count("TP"), "fp": k.count("FP"), "fn": k.count("FN"), "mis": k.count("MISCLS")}


def save_json(path, obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))

    Path(path).write_text(json.dumps(obj, indent=1, default=conv))


__all__ = ["fig_pr", "fig_confusion", "fig_per_class", "fig_threshold", "fig_timing", "gallery", "case_image",
           "write_report", "outcome_counts", "save_json", "OUTCOME_BGR"]
