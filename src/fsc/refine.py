"""Stage 3 post-processing: refined probability crops -> pixel-disjoint instances -> RLE rows.

The refiner (kaggle/refiner_template.py) returns, for every selected instance, a
CROP x CROP probability crop and the square window (r0, c0, S) it was cut from. Here each
crop is resized back to S x S, pasted into the image, thresholded at t, and every pixel
claimed by more than one instance goes to the instance with the highest refined
probability. Instances that end up smaller than min_area pixels are dropped (the refiner
acts as a verifier for them).
"""
from __future__ import annotations

from multiprocessing import get_context

import numpy as np
import pandas as pd
from PIL import Image

from .metric import H, W, rle_from_binary


def upsample(p8: np.ndarray, S: int) -> np.ndarray:
    if p8.shape[0] == S:
        return p8.astype(np.float32) / 255.0
    return np.asarray(Image.fromarray(p8).resize((S, S), Image.BILINEAR), dtype=np.float32) / 255.0


def _clip(r0: int, c0: int, S: int):
    return max(r0, 0), max(c0, 0), min(r0 + S, H), min(c0 + S, W)


def image_rows(task) -> list[tuple[str, str]]:
    """task = (meta rows of one image as list of dicts, probs (n, CROP, CROP) uint8, thresholds, min_area).

    Returns {t: [(filament_id, rle), ...]} for every threshold t.
    """
    rows, probs, thresholds, min_area = task
    ups = [upsample(p8, r["S"]) for r, p8 in zip(rows, probs)]
    out = {}
    for t in thresholds:
        best = np.zeros((H, W), np.float32)
        owner = np.zeros((H, W), np.int16)
        for i, (r, p) in enumerate(zip(rows, ups), start=1):
            R0, C0, R1, C1 = _clip(r["r0"], r["c0"], r["S"])
            sub = p[R0 - r["r0"]:R1 - r["r0"], C0 - r["c0"]:C1 - r["c0"]]
            b, o = best[R0:R1, C0:C1], owner[R0:R1, C0:C1]
            win = (sub > t) & (sub > b)
            b[win] = sub[win]
            o[win] = i
        res = []
        for i, r in enumerate(rows, start=1):
            R0, C0, R1, C1 = _clip(r["r0"], r["c0"], r["S"])
            local = owner[R0:R1, C0:C1] == i
            if local.sum() < min_area:
                continue
            m = np.zeros((H, W), np.uint8)
            m[R0:R1, C0:C1] = local
            res.append(m)
        stem = rows[0]["stem"]
        out[t] = [(f"{stem}_{j}", rle_from_binary(m)) for j, m in enumerate(res, start=1)]
    return out


def apply(meta: pd.DataFrame, probs: np.ndarray, thresholds, min_area: int = 16, workers: int = 4) -> dict:
    """Returns {t: DataFrame(filament_id, segmentation_rle)} over all images in meta."""
    tasks = []
    for stem, idx in meta.groupby("stem", sort=True).indices.items():
        rows = meta.iloc[idx][["stem", "r0", "c0", "S"]].to_dict("records")
        tasks.append((rows, probs[idx], list(thresholds), min_area))
    if workers <= 1:
        results = [image_rows(t) for t in tasks]
    else:
        with get_context("spawn").Pool(workers, maxtasksperchild=40) as pool:
            results = pool.map(image_rows, tasks, chunksize=4)
    return {t: pd.DataFrame([row for r in results for row in r[t]], columns=["filament_id", "segmentation_rle"])
            for t in thresholds}
