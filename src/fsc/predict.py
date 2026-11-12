"""Per-image prediction: probability map(s) -> component tree -> scorer -> antichain -> RLE rows.

Used both for the submission (scorer trained on all out-of-fold nodes) and for the
honest out-of-fold evaluation on real masks (fold k scored by the scorer trained on
the other folds). Runs in parallel worker processes; each worker loads the pickled
scorer once.
"""
from __future__ import annotations

import os
import pickle
from multiprocessing import get_context

import numpy as np

_SCORERS: dict = {}


def _scorer(path: str):
    if path not in _SCORERS:
        with open(path, "rb") as fh:
            _SCORERS[path] = pickle.load(fh)
    return _SCORERS[path]


def predict_image(task) -> list[tuple[str, str]]:
    """task = (stem, prob_paths, image_path, scorer_path, tau) -> [(filament_id, rle), ...]."""
    from .candidates import build_nodes
    from .extract import _read_gray, read_prob
    from .metric import rle_from_binary
    from .scorer import add_features, predict
    from .select import tree_dp

    stem, prob_paths, img_path, scorer_path, tau = task
    df, masks = build_nodes(read_prob(prob_paths), _read_gray(img_path), None, keep_masks=True)
    if len(df) == 0:
        return []
    df.insert(0, "stem", stem)
    f = add_features(df).reset_index(drop=True)
    sel = tree_dp(f, predict(_scorer(scorer_path), f) - tau)
    rows = []
    for i, n in enumerate(f["node"].to_numpy()[sel], start=1):
        sl, crop = masks[n]
        m = np.zeros((2048, 2048), np.uint8)
        m[sl] = crop
        rows.append((f"{stem}_{i}", rle_from_binary(m)))
    return rows


def predict_many(tasks: list, workers: int = 4) -> list[tuple[str, str]]:
    if workers <= 1:
        out = [predict_image(t) for t in tasks]
    else:
        with get_context("spawn").Pool(workers, maxtasksperchild=25) as pool:
            out = pool.map(predict_image, tasks, chunksize=2)
    return [r for rows in out for r in rows]
