"""Report figures from the notebook's saved results (work/results/examples.pkl, overlap_stats.npz).

Usage: python report/make_figures.py <work_dir>
Writes report/figures/fig_examples.pdf and report/figures/fig_diagnostics.pdf.
"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
GT_COLOR, PRED_COLOR, GREY = "#f2c14e", "#2f7bd8", "#8a8a8a"
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def examples(work, out):
    ex = pickle.load(open(os.path.join(work, "results", "examples.pkl"), "rb"))
    names = ["median image", "upper-quartile image", "lower-quartile image"]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.42), gridspec_kw=dict(wspace=0.04))
    for ax, (stem, d), name in zip(axs, ex.items(), names):
        img = d["img"]
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5), interpolation="nearest")
        for lab, col, lw in ((d["gt"], GT_COLOR, 0.9), (d["pred"], PRED_COLOR, 0.7)):
            for v in np.unique(lab[lab > 0]):
                ax.contour((lab == v).astype(float), levels=[0.5], colors=[col], linewidths=lw)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name} (PQ {d['pq']:.2f})", loc="left")
    fig.legend(handles=[Line2D([], [], color=GT_COLOR, lw=1.2, label="first annotator"),
                        Line2D([], [], color=PRED_COLOR, lw=1.2, label="prediction (out-of-fold)")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.savefig(os.path.join(out, "fig_examples.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "fig_examples.png"), dpi=200, bbox_inches="tight")
    return {s: (d["pq"], d["r0"], d["c0"]) for s, d in ex.items()}


def diagnostics(work, out):
    st = np.load(os.path.join(work, "results", "overlap_stats.npz"))
    fig = plt.figure(figsize=(3.35, 2.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.75, wspace=0.08)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for ax, key, name in ((a, "iou", "IoU"), (b, "dice", "Dice")):
        w = np.full(len(st[key]), 100.0 / len(st[key]))
        ax.hist(st[key], bins=40, range=(0, 1), weights=w, color=PRED_COLOR)
        ax.axvline(st[key].mean(), color="k", lw=0.8)
        ax.set_xlabel(f"{name} of overlapping pairs")
        ax.set_title(f"mean {name} {st[key].mean():.3f}", loc="left")
    a.axvline(0.5, color=GREY, lw=0.8, ls="--")
    a.set_ylabel("% of pairs")
    top = 1.1 * max(a.get_ylim()[1], b.get_ylim()[1])
    a.set_ylim(0, top)
    b.set_ylim(0, top)
    b.tick_params(labelleft=False)
    deg_gt = np.bincount(np.minimum(st["gt_degree"], 3), minlength=4)
    deg_pr = np.bincount(np.minimum(st["pred_degree"], 3), minlength=4)
    x = np.arange(4)
    c.bar(x - 0.2, 100 * deg_gt / deg_gt.sum(), 0.4, color=GT_COLOR, label=f"annotated (n = {deg_gt.sum():,})")
    c.bar(x + 0.2, 100 * deg_pr / deg_pr.sum(), 0.4, color=PRED_COLOR, label=f"predicted (n = {deg_pr.sum():,})")
    c.set_xticks(x)
    c.set_xticklabels(["0", "1", "2", "3+"])
    c.set_xlabel("number of overlapping partners (per annotator)")
    c.set_ylabel("%")
    c.legend(frameon=False, loc="center right")
    c.set_title("one-to-many and many-to-one", loc="left")
    fig.savefig(os.path.join(out, "fig_diagnostics.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "fig_diagnostics.png"), dpi=200, bbox_inches="tight")
    return dict(iou_mean=float(st["iou"].mean()), dice_mean=float(st["dice"].mean()), share_iou_gt_05=float((st["iou"] > 0.5).mean()),
                gt_deg=deg_gt.tolist(), pred_deg=deg_pr.tolist())


if __name__ == "__main__":
    work = sys.argv[1]
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    print(examples(work, out))
    print(diagnostics(work, out))
