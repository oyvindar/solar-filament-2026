"""Raw-file extraction for the deep out-of-fold evaluation (steps 3 and 5).

Rebuilds, from the JPEGs, the COCO polygons and the probability PNGs:
  nodes_<set>.parquet / hist_<set>.npy for three input sets
      oof2_train  180 random training images, 2-model out-of-fold maps, fold-held-out scorer
      test2       180 test images, 2-model maps (one fold's ResNet34 + ResNet50), all-data scorer
      test10      180 test images, 10-model maps (as submitted), all-data scorer
  anatomy_gt.parquet    one row per ground-truth filament per annotator (all 707 training images)
  anatomy_pred.parquet  one row per selected prediction per annotator

Usage: python scripts/deep_eval_extract.py <data_root> <kaggle_out> <work> [tau] [workers]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
from fsc import data as fdata  # noqa: E402
from fsc import diagnostics as dg  # noqa: E402


def main(root, outr, work, tau=0.32, workers=3):
    t0 = time.time()
    coco = fdata.Coco(root)
    folds = fdata.load_folds(REPO)
    fold_of = dict(zip(folds["stem"], folds["fold"]))
    out = os.path.join(work, "deep_eval")
    os.makedirs(out, exist_ok=True)

    def maps(s, k, split, models=("r34", "r50")):
        return [os.path.join(outr, f"fsc26-unet-{m}-f{k}", split, s + ".png") for m in models]

    cv_scorer = lambda s: os.path.join(work, f"scorer_ens_hurdle_cv_f{fold_of[s]}.pkl")  # noqa: E731
    all_scorer = os.path.join(work, "scorer_ens_hurdle_all.pkl")
    train_img = lambda s: os.path.join(coco.train_dir, s + ".jpeg")  # noqa: E731
    test_img = lambda s: os.path.join(coco.test_dir, s + ".jpeg")  # noqa: E731

    rng = np.random.default_rng(2026)
    sample = sorted(rng.choice(folds["stem"].to_numpy(), 180, replace=False))
    sets = {
        "oof2_train": [(s, maps(s, fold_of[s], "oof"), train_img(s), cv_scorer(s), tau) for s in sample],
        "test2": [(s, maps(s, i % 5, "test"), test_img(s), all_scorer, tau) for i, s in enumerate(coco.test_stems)],
        "test10": [(s, [p for k in range(5) for p in maps(s, k, "test")], test_img(s), all_scorer, tau)
                   for s in coco.test_stems],
    }
    for name, tasks in sets.items():
        res = dg.run(dg.nodes_hist, tasks, workers)
        nodes = pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True)
        nodes.to_parquet(os.path.join(out, f"nodes_{name}.parquet"), index=False)
        np.save(os.path.join(out, f"hist_{name}.npy"), np.sum([r[1] for r in res], axis=0))
        print(f"[{time.time() - t0:6.0f}s] {name}: {len(tasks)} images, {len(nodes)} nodes, "
              f"{int(nodes['sel'].sum())} selected", flush=True)

    tasks = [(s, maps(s, fold_of[s], "oof"), train_img(s),
              [(r, coco.inst_by_rec.get(r, [])) for r in coco.recs_by_stem[s]], cv_scorer(s), tau)
             for s in folds["stem"]]
    res = dg.run(dg.anatomy_image, tasks, workers)
    gt = pd.DataFrame([r for g, _ in res for r in g])
    pred = pd.DataFrame([r for _, p in res for r in p])
    gt.to_parquet(os.path.join(out, "anatomy_gt.parquet"), index=False)
    pred.to_parquet(os.path.join(out, "anatomy_pred.parquet"), index=False)
    print(f"[{time.time() - t0:6.0f}s] anatomy: {len(gt)} GT rows, {len(pred)} prediction rows. DONE", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], float(a[3]) if len(a) > 3 else 0.32, int(a[4]) if len(a) > 4 else 3)
