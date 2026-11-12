"""Diagnostics behind the out-of-fold evaluation: error anatomy and input-shift checks.

Everything here starts from the raw files (JPEG image, COCO polygons, probability PNGs):
the component tree is rebuilt per image and the selection is recomputed with the scorer
that did not see the image's fold.

anatomy_image  -> one row per ground-truth filament (per annotator) and one row per
                  selected prediction (per annotator), with the IoUs needed to say why
                  a filament was missed.
nodes_hist     -> node table with scorer output and selection for any image (labels not
                  needed), plus the 256-bin histogram of the probability map; used to
                  compare out-of-fold inputs with test inputs.
"""
from __future__ import annotations

from multiprocessing import get_context

import numpy as np
import pandas as pd
from scipy import ndimage as ndi


def _select(df: pd.DataFrame, scorer_path: str, tau: float):
    from .predict import _scorer
    from .scorer import add_features, predict
    from .select import tree_dp

    f = add_features(df).reset_index(drop=True)
    u = predict(_scorer(scorer_path), f)
    return f, u, tree_dp(f, u - tau)


def anatomy_image(task):
    """task = (stem, prob_paths, image_path, [(record_id, [polygons per instance]), ...], scorer_path, tau)."""
    from .candidates import build_nodes, disk_geometry, gt_label_maps
    from .data import rasterise
    from .extract import _read_gray, read_prob

    stem, prob_paths, img_path, annotators, scorer_path, tau = task
    img = _read_gray(img_path)
    cy, cx, rad = disk_geometry(img)
    df, masks = build_nodes(read_prob(prob_paths), img, None, keep_masks=True)
    if len(df):
        df.insert(0, "stem", stem)
        f, u, sel = _select(df, scorer_path, tau)
        nodes = f["node"].to_numpy()
        areas = f["area"].to_numpy().astype(np.float64)
        levels = f["level"].to_numpy()
    else:
        nodes, areas, levels = np.zeros(0, int), np.zeros(0), np.zeros(0, int)
        u, sel = np.zeros(0), np.zeros(0, bool)
    gt_rows, pred_rows = [], []
    for rec, polys in annotators:
        glab, gareas = gt_label_maps(polys, rasterise)
        n_gt = len(gareas) - 1
        iou = np.zeros((len(nodes), n_gt + 1))
        for i, n in enumerate(nodes):
            sl, crop = masks[n]
            cnt = np.bincount(glab[sl][crop], minlength=n_gt + 1).astype(np.float64)
            cnt[0] = 0.0
            iou[i] = cnt / np.maximum(areas[i] + gareas - cnt, 1.0)
        iou = iou[:, 1:]
        com = ndi.center_of_mass(glab > 0, glab, np.arange(1, n_gt + 1)) if n_gt else []
        isel = np.nonzero(sel)[0]
        for j in range(n_gt):
            col = iou[:, j]
            row = dict(stem=stem, rec=rec, inst=j + 1, area=int(gareas[j + 1]),
                       r_norm=float(np.hypot(com[j][0] - cy, com[j][1] - cx) / rad) if np.isfinite(com[j][0]) else np.nan,
                       best_any=0.0, best_any_area=0, best_any_uhat=np.nan, best_any_level=-1, best_any_selected=False,
                       n_nodes_hit=0, best_sel=0.0, best_sel_area=0, n_sel_overlap=0)
            if len(col):
                b = int(col.argmax())
                row.update(best_any=float(col[b]), best_any_area=int(areas[b]), best_any_uhat=float(u[b]),
                           best_any_level=int(levels[b]), best_any_selected=bool(sel[b]), n_nodes_hit=int((col > 0.5).sum()))
            if len(isel):
                cs = col[isel]
                bs = int(cs.argmax())
                row.update(best_sel=float(cs[bs]), best_sel_area=int(areas[isel[bs]]) if cs[bs] > 0 else 0,
                           n_sel_overlap=int((cs > 0).sum()))
            gt_rows.append(row)
        for i in isel:
            r = iou[i] if n_gt else np.zeros(0)
            pred_rows.append(dict(stem=stem, rec=rec, node=int(nodes[i]), level=int(levels[i]), area=int(areas[i]),
                                  u_hat=float(u[i]), r_norm=float(f["r_norm"].iat[i]),
                                  best_iou=float(r.max()) if len(r) else 0.0, n_gt_overlap=int((r > 0).sum())))
    return gt_rows, pred_rows


def nodes_hist(task):
    """task = (stem, prob_paths, image_path, scorer_path, tau) -> (node table with u_hat and sel, histogram)."""
    from .candidates import build_nodes
    from .extract import _read_gray, read_prob

    stem, prob_paths, img_path, scorer_path, tau = task
    prob = read_prob(prob_paths)
    hist = np.bincount(prob.ravel(), minlength=256)
    df, _ = build_nodes(prob, _read_gray(img_path), None)
    if len(df) == 0:
        return None, hist
    df.insert(0, "stem", stem)
    f, u, sel = _select(df, scorer_path, tau)
    f["u_hat"] = u
    f["sel"] = sel
    return f, hist


def run(worker, tasks: list, workers: int = 4) -> list:
    if workers <= 1:
        return [worker(t) for t in tasks]
    with get_context("spawn").Pool(workers, maxtasksperchild=25) as pool:
        return pool.map(worker, tasks, chunksize=2)


def classify_gt(gt: pd.DataFrame) -> pd.Series:
    """Outcome of every ground-truth filament (per annotator).

    matched        a selected instance has IoU > 0.5
    no_overlap     no candidate in the tree touches it (the probability map misses it)
    poor_candidate candidates touch it but none reaches IoU > 0.5
    too_big        a matching candidate exists, but the selected instance on it is larger than the filament
    too_small      a matching candidate exists, but the selected instance on it is smaller than the filament
    rejected       a matching candidate exists and nothing was selected on the filament
    """
    out = pd.Series("matched", index=gt.index)
    miss = gt["best_sel"] <= 0.5
    out[miss & (gt["best_any"] <= 0)] = "no_overlap"
    out[miss & (gt["best_any"] > 0) & (gt["best_any"] <= 0.5)] = "poor_candidate"
    cand = miss & (gt["best_any"] > 0.5)
    out[cand & (gt["n_sel_overlap"] == 0)] = "rejected"
    out[cand & (gt["n_sel_overlap"] > 0) & (gt["best_sel_area"] > gt["area"])] = "too_big"
    out[cand & (gt["n_sel_overlap"] > 0) & (gt["best_sel_area"] <= gt["area"])] = "too_small"
    return out
