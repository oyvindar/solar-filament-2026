"""Build a Kaggle submission CSV from the fold models' test probability maps.

Per test image: average the available fold maps -> component tree -> node
features -> expected matched IoU from the scorer (trained on all out-of-fold
nodes) -> max-weight antichain with gain u_hat - tau -> compressed COCO RLE.
With --baseline THR,MIN_AREA it instead keeps every component of the fixed
threshold level that is at least MIN_AREA pixels (classic post-processing).

Example:
  python scripts/make_submission.py --maps work/kaggle_out/fsc26-unet-r34-f0 ... \
         --scorer work/scorer_xent.pkl --tau 0.18 --out work/sub_tree.csv
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fsc.candidates import build_nodes  # noqa: E402
from fsc.data import Coco  # noqa: E402
from fsc.extract import read_prob  # noqa: E402
from fsc.metric import check_submission, rle_from_binary  # noqa: E402
from fsc.scorer import add_features, predict  # noqa: E402
from fsc.select import tree_dp  # noqa: E402


def select_image(df, args, model):
    if args.baseline:
        thr, amin = args.baseline.split(",")
        return np.isclose(df["thr"].to_numpy(), float(thr), atol=0.01) & (df["area"].to_numpy() >= int(amin))
    f = add_features(df).reset_index(drop=True)
    return tree_dp(f, predict(model, f) - args.tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True, help="fold output dirs containing test/<stem>.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scorer", help="pickled scorer from fsc.scorer.fit")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--baseline", help="THR,MIN_AREA fixed-threshold post-processing")
    ap.add_argument("--root", default=None, help="competition data root (MAGFiLO_1.0_Kaggle_2026)")
    args = ap.parse_args()
    coco = Coco(args.root) if args.root else Coco()
    model = pickle.load(open(args.scorer, "rb")) if args.scorer else None
    assert args.baseline or model is not None, "need --scorer or --baseline"
    rows, n_nodes, n_models = [], 0, []
    for s in coco.test_stems:
        paths = [p for p in (os.path.join(d, "test", s + ".png") for d in args.maps) if os.path.exists(p)]
        assert paths, f"no probability map for {s}"
        n_models.append(len(paths))
        df, masks = build_nodes(read_prob(paths), coco.image(s, test=True), None, keep_masks=True)
        if len(df) == 0:
            continue
        df.insert(0, "stem", s)
        n_nodes += len(df)
        sel = np.asarray(select_image(df, args, model))
        for i, n in enumerate(df["node"].to_numpy()[sel], start=1):
            sl, crop = masks[n]
            m = np.zeros((2048, 2048), np.uint8)
            m[sl] = crop
            rows.append((f"{s}_{i}", rle_from_binary(m)))
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    print("models per image:", dict(pd.Series(n_models).value_counts()), "| nodes", n_nodes,
          "| checks", check_submission(sub))
    sub.to_csv(args.out, index=False)
    print("wrote", args.out, len(sub), "rows")


if __name__ == "__main__":
    main()
