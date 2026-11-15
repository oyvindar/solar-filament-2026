"""Figures used by the notebook and the report."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .metric import H, W, rle_obj

GT_COLOR, PRED_COLOR = "#f2c14e", "#2f7bd8"


def decode(rles) -> list[np.ndarray]:
    return [mask_util.decode(rle_obj(r)).astype(bool) for r in rles]


def busiest_window(masks: list[np.ndarray], size: int = 768, step: int = 128) -> tuple[int, int]:
    """Top-left corner of the size x size window that contains the most annotated pixels."""
    acc = np.zeros((H, W), np.int32)
    for m in masks:
        acc += m
    ii = np.pad(acc.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, arg = -1, (0, 0)
    for r in range(0, H - size + 1, step):
        for c in range(0, W - size + 1, step):
            s = ii[r + size, c + size] - ii[r, c + size] - ii[r + size, c] + ii[r, c]
            if s > best:
                best, arg = s, (r, c)
    return arg


def overlay(ax, img: np.ndarray, gt_masks, pred_masks, window=None, size: int = 768):
    """H-alpha crop with annotated filaments (yellow outlines) and predictions (blue outlines)."""
    r0, c0 = window if window is not None else busiest_window(gt_masks or pred_masks, size)
    sl = (slice(r0, r0 + size), slice(c0, c0 + size))
    ax.imshow(img[sl], cmap="gray", vmin=np.percentile(img[sl], 1), vmax=np.percentile(img[sl], 99.5))
    for ms, col, lw in ((gt_masks, GT_COLOR, 1.1), (pred_masks, PRED_COLOR, 0.9)):
        for m in ms:
            crop = m[sl]
            if crop.any():
                ax.contour(crop.astype(float), levels=[0.5], colors=[col], linewidths=lw)
    ax.set_xticks([])
    ax.set_yticks([])
    return r0, c0


def per_image_pq(per_pair: pd.DataFrame) -> pd.Series:
    """PQ per image from pq_fast(...).per_pair (pooled over that image's annotators)."""
    d = per_pair.assign(stem=per_pair["annotator_image"].str.split("-", n=1).str[1]).groupby("stem")[["tp", "fp", "fn", "sum_iou"]].sum()
    return d["sum_iou"] / (d["tp"] + 0.5 * d["fp"] + 0.5 * d["fn"]).clip(lower=1e-9)
