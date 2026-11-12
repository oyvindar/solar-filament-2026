"""Competition data access: image paths, COCO indexing, rasterisation, folds."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_util

from .metric import H, W

# Competition data folder (the directory that contains train/ and test/). Set FSC_DATA_ROOT or pass root=.
DEFAULT_ROOT = os.environ.get("FSC_DATA_ROOT", os.path.join("data", "MAGFiLO_1.0_Kaggle_2026"))
ANN_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


class Coco:
    """Index of the training annotations: records (annotator x image) and polygon lists per instance."""

    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root
        self.train_dir = os.path.join(root, "train", "train_images")
        self.test_dir = os.path.join(root, "test", "test_images")
        self.raw = json.load(open(os.path.join(root, "train", ANN_NAME), encoding="utf-8"))
        self.recs_by_stem: dict[str, list[str]] = {}
        for im in self.raw["images"]:
            self.recs_by_stem.setdefault(im["file_name"].rsplit(".", 1)[0], []).append(im["id"])
        self.inst_by_rec: dict[str, list[list]] = {}
        for a in self.raw["annotations"]:
            self.inst_by_rec.setdefault(a["image_id"], []).append(a["segmentation"])
        self.stems = sorted(self.recs_by_stem)
        self.test_stems = sorted(f[:-5] for f in os.listdir(self.test_dir) if f.endswith(".jpeg"))

    def image(self, stem: str, test: bool = False) -> np.ndarray:
        d = self.test_dir if test else self.train_dir
        return np.asarray(Image.open(os.path.join(d, stem + ".jpeg")).convert("L"), dtype=np.uint8)

    def n_gt(self, stems) -> int:
        """Total number of GT instances over all (annotator, image) pairs of the given images."""
        return int(sum(len(self.inst_by_rec.get(r, [])) for s in stems for r in self.recs_by_stem[s]))

    def gt_maps(self, stem: str):
        """Per annotator: (int16 instance-label map, instance areas)."""
        from .candidates import gt_label_maps
        return [gt_label_maps(self.inst_by_rec.get(r, []), rasterise) for r in self.recs_by_stem[stem]]

    def soft_target(self, stem: str) -> np.ndarray:
        acc = np.zeros((H, W), np.float32)
        for r in self.recs_by_stem[stem]:
            m = np.zeros((H, W), np.uint8)
            for polys in self.inst_by_rec.get(r, []):
                m |= rasterise(polys)
            acc += m
        return acc / len(self.recs_by_stem[stem])


def rasterise(polys: list, h: int = H, w: int = W) -> np.ndarray:
    polys = [p for p in polys if len(p) >= 6]
    if not polys:
        return np.zeros((h, w), np.uint8)
    return mask_util.decode(mask_util.merge(mask_util.frPyObjects(polys, h, w))).astype(np.uint8)


def load_folds(repo: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(repo, "data", "folds.csv"), dtype={"date": str})
