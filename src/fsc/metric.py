"""Panoptic Quality exactly as in the organisers' self-evaluation notebook.

Official definition (https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook):
  * ground-truth ``filament_id`` = ``<annotator_id>-<image_id>_<n>``; every
    (annotator, image) pair is scored against the same set of predictions
    whose ``filament_id`` = ``<image_id>_<n>``;
  * a (GT, prediction) pair is a hit when IoU > 0.5; every hit contributes its
    IoU, a prediction with no hit is a FP, a GT with no hit is a FN;
  * PQ = sum(hit IoUs) / (|hits| + 0.5 |FP| + 0.5 |FN|), pooled over all pairs;
  * (annotator, image) pairs that do not appear in the GT are not scored.

``pq_official`` is a line-by-line numpy port (dense masks, used only to test).
``pq_fast`` computes the same quantities from compressed RLEs with
pycocotools.mask.iou, which never decodes full 2048x2048 masks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

H = W = 2048
IOU_THR = 0.5


# ----------------------------------------------------------------------------- RLE helpers
def rle_from_binary(m: np.ndarray) -> str:
    """Binary (H, W) mask -> compressed COCO RLE counts string."""
    r = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
    return r["counts"].decode("ascii")


def rle_obj(counts: str, h: int = H, w: int = W) -> dict:
    return {"size": [h, w], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def rle_area(counts: str) -> int:
    return int(mask_util.area(rle_obj(counts)))


def polygons_to_rle(polys: list, h: int = H, w: int = W) -> str:
    """COCO polygon list of one annotation -> merged compressed RLE string."""
    polys = [p for p in polys if len(p) >= 6]
    rles = mask_util.frPyObjects(polys, h, w)
    merged = mask_util.merge(rles) if isinstance(rles, list) else rles
    c = merged["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def gt_frame_from_coco(coco: dict) -> pd.DataFrame:
    """Build the official-format GT table: filament_id=<record_id>_<n>, segmentation_rle."""
    rows, counter = [], {}
    for a in coco["annotations"]:
        rec = a["image_id"]
        counter[rec] = counter.get(rec, 0) + 1
        rows.append((f"{rec}_{counter[rec]}", polygons_to_rle(a["segmentation"])))
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


# ----------------------------------------------------------------------------- official port
def _decode_stack(rles: list[str]) -> np.ndarray:
    if not rles:
        return np.zeros((0, H, W), np.float32)
    m = mask_util.decode([rle_obj(r) for r in rles])
    return m.transpose(2, 0, 1).astype(np.float32)


def pq_official(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Numpy port of the organisers' get_overlap_df + get_pq_score (dense; slow)."""
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0]
    pr_img = pred_df["filament_id"].str.split("_", n=1).str[0]
    tp_ious, fp, fn = [], 0, 0
    for ai in gt_ai.unique():
        image_id = ai.split("-", 1)[1]
        g = _decode_stack(gt_df.loc[gt_ai == ai, "segmentation_rle"].tolist())
        p = _decode_stack(pred_df.loc[pr_img == image_id, "segmentation_rle"].tolist())
        n_gt, n_pr = len(g), len(p)
        if n_gt == 0:
            fp += n_pr; continue
        if n_pr == 0:
            fn += n_gt; continue
        gf, pf = g.reshape(n_gt, -1), p.reshape(n_pr, -1)
        inter = gf @ pf.T
        union = gf.sum(1)[:, None] + pf.sum(1)[None, :] - inter
        iou = np.where(union == 0, 0.0, inter / np.maximum(union, 1e-9))
        hit = iou > IOU_THR
        tp_ious.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    den = len(tp_ious) + 0.5 * fp + 0.5 * fn
    return float(sum(tp_ious) / den) if den > 0 else 0.0


# ----------------------------------------------------------------------------- fast version
@dataclass
class PQResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    sum_iou: float
    per_pair: pd.DataFrame  # one row per (annotator, image): tp, fp, fn, sum_iou, n_gt, n_pred

    def __repr__(self) -> str:
        return (f"PQ={self.pq:.4f} (SQ={self.sq:.4f}, RQ={self.rq:.4f}) "
                f"TP={self.tp} FP={self.fp} FN={self.fn} pairs={len(self.per_pair)}")


def pq_fast(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> PQResult:
    """Same scoring rule as pq_official, computed on compressed RLEs."""
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0].to_numpy()
    pr_img = pred_df["filament_id"].str.split("_", n=1).str[0].to_numpy()
    pr_groups = pd.Series(np.arange(len(pred_df))).groupby(pr_img).apply(list).to_dict()
    gt_rles = gt_df["segmentation_rle"].to_numpy()
    pr_rles = pred_df["segmentation_rle"].to_numpy()
    rows = []
    for ai, idx in pd.Series(np.arange(len(gt_df))).groupby(gt_ai, sort=False):
        image_id = ai.split("-", 1)[1]
        g = [rle_obj(r) for r in gt_rles[idx.to_numpy()]]
        pidx = pr_groups.get(image_id, [])
        p = [rle_obj(r) for r in pr_rles[pidx]]
        n_gt, n_pr = len(g), len(p)
        if n_pr == 0:
            rows.append((ai, 0, 0, n_gt, 0.0, n_gt, 0)); continue
        iou = np.asarray(mask_util.iou(p, g, [0] * n_gt)).reshape(n_pr, n_gt).T  # (n_gt, n_pred)
        hit = iou > IOU_THR
        rows.append((ai, int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum()),
                     float(iou[hit].sum()), n_gt, n_pr))
    per = pd.DataFrame(rows, columns=["annotator_image", "tp", "fp", "fn", "sum_iou", "n_gt", "n_pred"])
    tp, fp, fn, s = int(per.tp.sum()), int(per.fp.sum()), int(per.fn.sum()), float(per.sum_iou.sum())
    den = tp + 0.5 * fp + 0.5 * fn
    pq = s / den if den > 0 else 0.0
    sq = s / tp if tp else 0.0
    rq = tp / den if den > 0 else 0.0
    return PQResult(pq, sq, rq, tp, fp, fn, s, per)


def overlap_stats(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict:
    """The organisers' diagnostics: IoU and Dice of every overlapping (GT, prediction) pair, and how many
    partners each GT filament and each prediction overlaps (one-to-many / many-to-one), per annotator."""
    gt_ai = gt_df["filament_id"].str.split("_", n=1).str[0].to_numpy()
    pr_img = pred_df["filament_id"].str.split("_", n=1).str[0].to_numpy()
    pr_groups = pd.Series(np.arange(len(pred_df))).groupby(pr_img).apply(list).to_dict()
    gt_rles, pr_rles = gt_df["segmentation_rle"].to_numpy(), pred_df["segmentation_rle"].to_numpy()
    ious, gt_deg, pr_deg = [], [], []
    for ai, idx in pd.Series(np.arange(len(gt_df))).groupby(gt_ai, sort=False):
        g = [rle_obj(r) for r in gt_rles[idx.to_numpy()]]
        pidx = pr_groups.get(ai.split("-", 1)[1], [])
        if not pidx:
            gt_deg.extend([0] * len(g))
            continue
        p = [rle_obj(r) for r in pr_rles[pidx]]
        iou = np.asarray(mask_util.iou(p, g, [0] * len(g))).reshape(len(p), len(g)).T
        ious.append(iou[iou > 0])
        gt_deg.extend((iou > 0).sum(1).tolist())
        pr_deg.extend((iou > 0).sum(0).tolist())
    iou = np.concatenate(ious) if ious else np.zeros(0)
    return dict(iou=iou, dice=2 * iou / (1 + iou), gt_degree=np.asarray(gt_deg), pred_degree=np.asarray(pr_deg))


def check_submission(sub: pd.DataFrame, h: int = H, w: int = W) -> dict:
    """Kaggle format checks: columns, unique ids, non-empty masks, no pixel shared within an image."""
    assert list(sub.columns) == ["filament_id", "segmentation_rle"], sub.columns.tolist()
    assert sub["filament_id"].is_unique, "duplicate filament_id"
    img = sub["filament_id"].str.rsplit("_", n=1).str[0]
    n_empty, n_overlap_images = 0, 0
    for _, grp in sub.groupby(img):
        rles = [rle_obj(r, h, w) for r in grp["segmentation_rle"]]
        areas = mask_util.area(rles)
        n_empty += int((areas == 0).sum())
        if len(rles) > 1 and mask_util.area(mask_util.merge(rles)) != int(areas.sum()):
            n_overlap_images += 1
    return dict(rows=len(sub), images=int(img.nunique()), empty=n_empty, images_with_overlap=n_overlap_images)
