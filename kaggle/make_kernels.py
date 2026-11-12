"""Generate Kaggle kernel folders (script + kernel-metadata.json) from train_unet_template.py.

Usage:  python kaggle/make_kernels.py smoke        -> kaggle/kernels/fsc26-unet-smoke
        python kaggle/make_kernels.py 0 1 2 3 4    -> kaggle/kernels/fsc26-unet-r34-f<k>
Push:   kaggle kernels push -p kaggle/kernels/<slug> --accelerator NvidiaTeslaT4
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
USER = "saudsaudesen"
COMPETITION = "filament-segmentation-2026"


def build(fold: int, smoke: bool, slug: str, encoder: str = "resnet34", epochs: int = 18) -> str:
    folds = pd.read_csv(os.path.join(REPO, "data", "folds.csv"))
    table = ";".join(f"{s}:{f}" for s, f in zip(folds["stem"], folds["fold"]))
    src = open(os.path.join(HERE, "train_unet_template.py"), encoding="utf-8").read()
    src = src.replace("__FOLD__", str(int(fold))).replace("__SMOKE__", str(bool(smoke))).replace("__FOLDS__", table)
    # the template's defaults are resnet34 / 18 epochs; other models only change these two literals
    src = src.replace('encoder="resnet34"', f'encoder="{encoder}"').replace("epochs=18,", f"epochs={int(epochs)},")
    out_dir = os.path.join(HERE, "kernels", slug)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{slug}.py"), "w", encoding="utf-8") as fh:
        fh.write(src)
    meta = {
        "id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [COMPETITION],
        "kernel_sources": [], "model_sources": [],
    }
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return out_dir


VARIANTS = {  # tag -> (timm encoder, epochs)
    "r34": ("resnet34", 18),
    "eb3": ("efficientnet_b3", 24),  # fails on Kaggle T4 (torch 2.11/cu130): CUDA misaligned address in fp16 depthwise conv
    "r50": ("resnet50", 22),
}

if __name__ == "__main__":
    args = sys.argv[1:]
    tag = "r34"
    if args and args[0].startswith("--variant="):
        tag = args.pop(0).split("=", 1)[1]
    enc, ep = VARIANTS[tag]
    for arg in args:
        if arg == "smoke":
            print(build(0, True, "fsc26-unet-smoke"))
        else:
            print(build(int(arg), False, f"fsc26-unet-{tag}-f{int(arg)}", enc, ep))
