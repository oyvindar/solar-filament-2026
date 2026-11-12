"""Candidate instances from a probability map: a multi-threshold component tree.

For thresholds t_0 < t_1 < ... the super-level sets {p > t_i} are nested, so
every connected component at level i+1 lies inside exactly one component at
level i. These components form a forest; any antichain of it (no node together
with its ancestor) is a set of pixel-disjoint instances, i.e. a valid
submission. Each node gets
  * features (shape, probability statistics, contrast, position, tree context),
  * labels against every annotator of the image (training images only):
    best IoU, and the matched IoU contribution IoU * [IoU > 0.5].

Because the pooled PQ denominator equals 0.5 * (R * #predictions + #GT) for
disjoint predictions, the PQ of any antichain follows from the node labels
alone (see select.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .metric import H, W

LEVELS = np.round(np.arange(0.10, 0.801, 0.05) * 255).astype(int)  # 15 thresholds on the uint8 map
MIN_AREA = 16
STRUCT8 = np.ones((3, 3), bool)


def disk_geometry(img: np.ndarray, thr: int = 10) -> tuple[float, float, float]:
    """Centre (row, col) and equivalent radius of the solar disk in a full-disk image."""
    m = img > thr
    rr, cc = np.nonzero(m[::4, ::4])
    return 4 * rr.mean(), 4 * cc.mean(), float(np.sqrt(m.sum() / np.pi))


def flat_field(img: np.ndarray, sigma: float = 24.0) -> np.ndarray:
    f = img.astype(np.float32)
    blur = ndi.gaussian_filter(f, sigma)
    ff = np.log1p(f) - np.log1p(blur)
    ff[f <= 10] = 0.0
    return ff


def gt_label_maps(polys_per_instance: list[list[list]], rasterise) -> tuple[np.ndarray, np.ndarray]:
    """One annotator: int16 instance-label map (0 = background) and instance areas (index = label)."""
    lab = np.zeros((H, W), np.int16)
    areas = [0]
    for k, polys in enumerate(polys_per_instance, start=1):
        m = rasterise(polys)
        lab[m > 0] = k
        areas.append(int(m.sum()))
    return lab, np.asarray(areas, np.int64)


def _shape_stats(rr: np.ndarray, cc: np.ndarray) -> tuple[float, float, float]:
    """Major/minor axis length (4*sqrt(eig) of the coordinate covariance) and |orientation|."""
    if len(rr) < 3:
        return 1.0, 1.0, 0.0
    cov = np.cov(np.vstack([rr, cc]).astype(np.float64)) + np.eye(2) / 12.0
    ev, evec = np.linalg.eigh(cov)
    major, minor = 4 * np.sqrt(max(ev[1], 1e-9)), 4 * np.sqrt(max(ev[0], 1e-9))
    ang = abs(np.arctan2(evec[0, 1], evec[1, 1]))
    return float(major), float(minor), float(ang)


def build_nodes(prob: np.ndarray, img: np.ndarray | None = None, gts: list | None = None,
                levels: np.ndarray = LEVELS, min_area: int = MIN_AREA, keep_masks: bool = False):
    """Extract the component tree of a uint8 probability map.

    gts: list of (label_map, areas) per annotator, or None for test images.
    Returns (DataFrame of nodes, masks) where masks[node_id] = (row_slice, col_slice, bool crop) if keep_masks.
    """
    geo = disk_geometry(img) if img is not None else (H / 2, W / 2, 0.45 * H)
    ff = flat_field(img) if img is not None else None
    rows, masks = [], {}
    prev_lab, prev_ids = None, None
    nid = 0
    for li, t in enumerate(levels):
        lab, n = ndi.label(prob > t, structure=STRUCT8)
        objs = ndi.find_objects(lab)
        ids = np.full(n + 1, -1, np.int64)
        for k, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            crop = lab[sl] == k
            area = int(crop.sum())
            if area < min_area:
                continue
            rr, cc = np.nonzero(crop)
            r0, c0 = sl[0].start, sl[1].start
            pv = prob[sl][crop].astype(np.float32) / 255.0
            parent = -1
            if prev_lab is not None:
                parent = int(prev_ids[prev_lab[r0 + rr[0], c0 + cc[0]]])
            major, minor, ang = _shape_stats(rr, cc)
            cy, cx = r0 + rr.mean(), c0 + cc.mean()
            rec = dict(node=nid, level=li, thr=t / 255.0, parent=parent, area=area,
                       r0=r0, r1=sl[0].stop, c0=c0, c1=sl[1].stop,
                       bbox_h=sl[0].stop - r0, bbox_w=sl[1].stop - c0, fill=area / ((sl[0].stop - r0) * (sl[1].stop - c0)),
                       p_mean=float(pv.mean()), p_max=float(pv.max()), p_q25=float(np.quantile(pv, 0.25)),
                       p_q75=float(np.quantile(pv, 0.75)), p_mass=float(pv.sum()),
                       major=major, minor=minor, elong=major / max(minor, 1e-6), thick=area / max(major, 1.0), orient=ang,
                       r_norm=float(np.hypot(cy - geo[0], cx - geo[1]) / geo[2]))
            # ring around the node (4 px) inside a padded crop: boundary sharpness and contrast
            pad = 6
            R0, R1 = max(r0 - pad, 0), min(sl[0].stop + pad, H)
            C0, C1 = max(c0 - pad, 0), min(sl[1].stop + pad, W)
            big = np.zeros((R1 - R0, C1 - C0), bool)
            big[r0 - R0:r0 - R0 + crop.shape[0], c0 - C0:c0 - C0 + crop.shape[1]] = crop
            ring = ndi.binary_dilation(big, STRUCT8, iterations=4) & ~big
            pr = prob[R0:R1, C0:C1][ring].astype(np.float32) / 255.0
            rec["p_ring"] = float(pr.mean()) if pr.size else 0.0
            if ff is not None:
                fin = ff[R0:R1, C0:C1][big]
                fr = ff[R0:R1, C0:C1][ring]
                rec["ff_in"] = float(fin.mean())
                rec["ff_ring"] = float(fr.mean()) if fr.size else 0.0
                rec["ff_contrast"] = rec["ff_in"] - rec["ff_ring"]
            if gts is not None:
                best = []
                for glab, gareas in gts:
                    g = glab[sl][crop]
                    cnt = np.bincount(g, minlength=len(gareas))
                    cnt[0] = 0
                    inter = cnt.astype(np.float64)
                    iou = inter / np.maximum(area + gareas - inter, 1)
                    iou[0] = 0.0
                    best.append(float(iou.max()))
                best = np.asarray(best)
                rec["R"] = len(best)
                rec["iou_mean"] = float(best.mean())
                rec["hits"] = int((best > 0.5).sum())
                rec["u_sum"] = float(best[best > 0.5].sum())
            rows.append(rec)
            if keep_masks:
                masks[nid] = (sl, crop)
            ids[k] = nid
            nid += 1
        prev_lab, prev_ids = lab, ids
    df = pd.DataFrame(rows)
    if len(df):
        n_child = df.groupby("parent").size()
        df["n_child"] = df["node"].map(n_child).fillna(0).astype(int)
        area_of = df.set_index("node")["area"]
        df["parent_area_ratio"] = np.where(df.parent >= 0, df.area / df.parent.map(area_of).fillna(1).clip(lower=1), 1.0)
        child_area = df[df.parent >= 0].groupby("parent")["area"].sum()
        df["child_area_ratio"] = df["node"].map(child_area).fillna(0) / df.area
        df["level_count"] = df.groupby("level")["node"].transform("size")
        # persistence: number of consecutive higher levels through which this node continues as an only child
        only = df[(df.parent >= 0)].groupby("parent").size() == 1
        child_of = df[df.parent >= 0].set_index("parent")["node"]
        pers = np.zeros(len(df), int)
        node_idx = {n: i for i, n in enumerate(df.node)}
        for i in range(len(df) - 1, -1, -1):
            n = df.node.iat[i]
            if only.get(n, False):
                c = child_of.loc[n]
                c = c.iloc[0] if isinstance(c, pd.Series) else c
                pers[i] = 1 + pers[node_idx[c]]
        df["persist"] = pers
    return df, masks
