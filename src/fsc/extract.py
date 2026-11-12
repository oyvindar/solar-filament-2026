"""Batch extraction of candidate-node tables from probability maps (parallel, low memory).

Each worker receives only the stem, file paths and (for training images) the
polygon lists of every annotator, so the 46 MB annotation file is parsed once.
"""
from __future__ import annotations

import os
from multiprocessing import get_context

import numpy as np
import pandas as pd
from PIL import Image


def _read_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def read_prob(paths) -> np.ndarray:
    """Average one or more uint8 probability PNGs (e.g. one per fold model) into one uint8 map."""
    if isinstance(paths, str):
        return _read_gray(paths)
    acc = np.zeros((2048, 2048), np.float32)
    for p in paths:
        acc += _read_gray(p)
    return np.round(acc / len(paths)).astype(np.uint8)


def _work(task):
    from .candidates import build_nodes, gt_label_maps
    from .data import rasterise
    stem, prob_paths, img_path, annotators = task
    prob = read_prob(prob_paths)
    img = _read_gray(img_path)
    gts = None if annotators is None else [gt_label_maps(polys, rasterise) for polys in annotators]
    df, _ = build_nodes(prob, img, gts)
    df.insert(0, "stem", stem)
    return df


def extract(tasks: list, workers: int = 3) -> pd.DataFrame:
    """tasks: list of (stem, prob_path_or_list, image_path, annotators_or_None)."""
    if workers <= 1:
        frames = [_work(t) for t in tasks]
    else:
        with get_context("spawn").Pool(workers, maxtasksperchild=20) as pool:
            frames = pool.map(_work, tasks, chunksize=2)
    return pd.concat(frames, ignore_index=True)


def train_tasks(coco, stems, prob_root_of) -> list:
    """prob_root_of(stem) -> path or list of paths of that image's OOF probability map(s)."""
    return [(s, prob_root_of(s), os.path.join(coco.train_dir, s + ".jpeg"),
             [coco.inst_by_rec.get(r, []) for r in coco.recs_by_stem[s]]) for s in stems]


def test_tasks(coco, stems, prob_paths_of) -> list:
    return [(s, prob_paths_of(s), os.path.join(coco.test_dir, s + ".jpeg"), None) for s in stems]
