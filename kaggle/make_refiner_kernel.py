"""Build the Kaggle kernel folder for the stage-3 refiner and the seeds dataset folder.

Usage:
  python kaggle/make_refiner_kernel.py seeds <oof_pred.parquet> <test_submission.csv> <dataset_dir>
  python kaggle/make_refiner_kernel.py kernel [smoke]
Push:
  kaggle datasets create -p <dataset_dir>            (first time; private)
  kaggle kernels push -p kaggle/kernels/fsc26-refiner --accelerator NvidiaTeslaT4
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
USER = "saudsaudesen"
COMPETITION = "filament-segmentation-2026"
SEEDS_DATASET = f"{USER}/fsc26-refiner-seeds"
STAGE1 = [f"{USER}/fsc26-unet-{m}-f{k}" for m in ("r34", "r50") for k in range(5)]


def seeds(oof_parquet: str, test_csv: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    oof = pd.read_parquet(oof_parquet)[["filament_id", "segmentation_rle"]]
    test = pd.read_csv(test_csv)[["filament_id", "segmentation_rle"]]
    oof.to_parquet(os.path.join(out_dir, "oof_seeds.parquet"), index=False)
    test.to_parquet(os.path.join(out_dir, "test_seeds.parquet"), index=False)
    meta = {"title": "fsc26-refiner-seeds", "id": SEEDS_DATASET, "licenses": [{"name": "CC0-1.0"}],
            "description": "Selected instances (stage 2 output) used as seeds by the stage-3 refiner: "
                           "out-of-fold predictions on the training images and the submitted test predictions. "
                           "Model predictions only, no annotations."}
    with open(os.path.join(out_dir, "dataset-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(f"seeds: oof {len(oof)} rows, test {len(test)} rows -> {out_dir}")
    return out_dir


def kernel(smoke: bool) -> str:
    slug = "fsc26-refiner-smoke" if smoke else "fsc26-refiner"
    folds = pd.read_csv(os.path.join(REPO, "data", "folds.csv"))
    table = ";".join(f"{s}:{f}" for s, f in zip(folds["stem"], folds["fold"]))
    src = open(os.path.join(HERE, "refiner_template.py"), encoding="utf-8").read()
    src = src.replace("__FOLDS__", table).replace("__SMOKE__", str(bool(smoke)))
    out_dir = os.path.join(HERE, "kernels", slug)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{slug}.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [SEEDS_DATASET], "competition_sources": [COMPETITION],
            "kernel_sources": STAGE1, "model_sources": []}
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print(out_dir)
    return out_dir


if __name__ == "__main__":
    if sys.argv[1] == "seeds":
        seeds(*sys.argv[2:5])
    else:
        kernel(len(sys.argv) > 2 and sys.argv[2] == "smoke")
