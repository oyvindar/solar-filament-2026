"""Solar Filament Segmentation Challenge 2026 -- stage 3: instance refiner (crop re-segmentation).

Every instance selected by stage 2 (component tree + expected-IoU scorer + exact selection)
is re-segmented by a small U-Net in a square crop around it. Input channels (5):
disk-standardised intensity, log flat-field, stage-1 probability, the instance itself
("seed") and the other selected instances of the image. Target: for each annotator, the
annotated filament that overlaps the seed best (IoU >= 0.1, otherwise nothing), averaged
over annotators -- the probability that an annotator paints the pixel as part of this
filament.

Training seeds: the real out-of-fold selections (what the refiner sees at test time) plus
perturbed annotated filaments (dilated, eroded, shifted, or re-thresholded from the
probability map). Fold-honest: the refiner that refines fold k never saw fold k; test
instances are refined by the average of the five fold refiners.

Inputs: competition data; the ten stage-1 kernels (kernel_sources: OOF and test probability
maps); dataset with oof_seeds.parquet / test_seeds.parquet (filament_id, segmentation_rle).
Outputs (/kaggle/working): refined_{oof,test}.npz (uint8 probability crops) with
refined_{oof,test}_meta.parquet (crop window per instance), refiner_f<k>.pt, log.json.

Generated from kaggle/refiner_template.py by kaggle/make_refiner_kernel.py.
"""
FOLDS = "__FOLDS__"  # "stem:fold;..." -- the same date-grouped split as stage 1
SMOKE = __SMOKE__

CFG = dict(encoder="resnet34", in_ch=5, crop=192, batch_per_gpu=32, epochs=14, lr=4e-4, wd=1e-4,
           warmup=100, ema=0.995, dice_w=0.5, min_target_iou=0.1, gt_seeds=True, seed=2026)
if SMOKE:
    CFG.update(epochs=1, n_train_images=40, n_test_images=8)

import copy
import glob
import json
import math
import os
import random
import sys
import time

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)


import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mu

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
RNG = np.random.default_rng(CFG["seed"])
OUT = os.environ.get("FSC_OUT", "/kaggle/working")
INPUT = os.environ.get("FSC_INPUT", "/kaggle/input")
STAGE1_ROOT = os.environ.get("FSC_STAGE1", INPUT)  # folder that contains the ten stage-1 output folders
SEEDS_ROOT = os.environ.get("FSC_SEEDS", INPUT)    # folder with oof_seeds.parquet / test_seeds.parquet
H = W = 2048
CROP = CFG["crop"]
LOG = dict(cfg=CFG, smoke=SMOKE, folds={}, notes=[])

# ----------------------------------------------------------------------------- inputs
ann_json = glob.glob(os.path.join(INPUT, "**", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), recursive=True)[0]
ROOT = os.path.dirname(os.path.dirname(ann_json))
TRAIN_DIR = os.path.join(ROOT, "train", "train_images")
TEST_DIR = os.path.join(ROOT, "test", "test_images")
MAPS = {}
for lj in glob.glob(os.path.join(STAGE1_ROOT, "**", "log.json"), recursive=True):
    try:
        L = json.load(open(lj))
    except Exception:
        continue
    if L.get("smoke") or "cfg" not in L or "fold" not in L:
        continue
    MAPS[(L["cfg"]["encoder"], int(L["fold"]))] = os.path.dirname(lj)
log("stage-1 outputs:", sorted(MAPS))
assert len(MAPS) == 10, f"expected 10 stage-1 kernels, found {sorted(MAPS)}"
ENCODERS = ("resnet34", "resnet50")
seeds_oof = pd.read_parquet(glob.glob(os.path.join(SEEDS_ROOT, "**", "oof_seeds.parquet"), recursive=True)[0])
seeds_test = pd.read_parquet(glob.glob(os.path.join(SEEDS_ROOT, "**", "test_seeds.parquet"), recursive=True)[0])
for d in (seeds_oof, seeds_test):
    d["stem"] = d["filament_id"].str.rsplit("_", n=1).str[0]
SEEDS = {"train": seeds_oof.groupby("stem")[["filament_id", "segmentation_rle"]].apply(lambda g: list(zip(g.filament_id, g.segmentation_rle))).to_dict(),
         "test": seeds_test.groupby("stem")[["filament_id", "segmentation_rle"]].apply(lambda g: list(zip(g.filament_id, g.segmentation_rle))).to_dict()}
log("seeds: oof", len(seeds_oof), "on", seeds_oof.stem.nunique(), "images | test", len(seeds_test), "on", seeds_test.stem.nunique())

fold_of = {k: int(v) for k, v in (s.split(":") for s in FOLDS.split(";"))}
J = json.load(open(ann_json))
inst_by_rec, recs_by_stem = {}, {}
for a in J["annotations"]:
    inst_by_rec.setdefault(a["image_id"], []).append(a["segmentation"])
for im in J["images"]:
    recs_by_stem.setdefault(im["file_name"].rsplit(".", 1)[0], []).append(im["id"])
train_stems = sorted(recs_by_stem)
test_stems = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{TEST_DIR}/*.jpeg"))
if SMOKE:
    train_stems, test_stems = train_stems[:CFG["n_train_images"]], test_stems[:CFG["n_test_images"]]


# ----------------------------------------------------------------------------- image helpers
def rasterise(polys):
    polys = [p for p in polys if len(p) >= 6]
    if not polys:
        return np.zeros((H, W), np.uint8)
    return mu.decode(mu.merge(mu.frPyObjects(polys, H, W)))


def decode(rle):
    return mu.decode({"size": [H, W], "counts": rle.encode("ascii")}).astype(bool)


def channels(img):
    """uint8 image -> (z, flat-field z) as uint8 with z in [-4, 4] mapped to [0, 255]."""
    f = img.astype(np.float32)
    disk = f > 10
    blur = cv2.GaussianBlur(f, (0, 0), 24)
    ff = np.where(disk, np.log1p(f) - np.log1p(blur), 0.0).astype(np.float32)
    z = (f - f[disk].mean()) / (f[disk].std() + 1e-6)
    zf = ff / (ff[disk].std() + 1e-6)
    to8 = lambda x: np.clip(np.round((x + 4.0) * 32.0), 0, 255).astype(np.uint8)  # noqa: E731
    return to8(z), to8(zf)


def prob_map(stem, test):
    if test:
        paths = [os.path.join(MAPS[(e, k)], "test", stem + ".png") for e in ENCODERS for k in range(5)]
    else:
        paths = [os.path.join(MAPS[(e, fold_of[stem])], "oof", stem + ".png") for e in ENCODERS]
    acc = np.zeros((H, W), np.float32)
    for p in paths:
        acc += cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return np.round(acc / len(paths)).astype(np.uint8)


def window(r0, r1, c0, c1):
    h, w = r1 - r0, c1 - c0
    S = int(max(64, 1.4 * max(h, w) + 24))
    return int(round((r0 + r1) / 2.0 - S / 2.0)), int(round((c0 + c1) / 2.0 - S / 2.0)), S


def crop(a, r0, c0, S):
    out = np.zeros((S, S), a.dtype)
    R0, C0, R1, C1 = max(r0, 0), max(c0, 0), min(r0 + S, H), min(c0 + S, W)
    if R1 > R0 and C1 > C0:
        out[R0 - r0:R1 - r0, C0 - c0:C1 - c0] = a[R0:R1, C0:C1]
    return out


def rs(a):
    return cv2.resize(a, (CROP, CROP), interpolation=cv2.INTER_AREA if a.shape[0] > CROP else cv2.INTER_LINEAR)


def bbox(m):
    rows, cols = np.any(m, 1), np.any(m, 0)
    r = np.nonzero(rows)[0]
    c = np.nonzero(cols)[0]
    return r[0], r[-1] + 1, c[0], c[-1] + 1


def target_for(seed_c, win, labs):
    r0, c0, S = win
    acc = np.zeros((S, S), np.float32)
    a_seed = float(seed_c.sum())
    for L, areas in labs:
        Lc = crop(L, r0, c0, S)
        cnt = np.bincount(Lc[seed_c], minlength=len(areas)).astype(np.float64)
        cnt[0] = 0.0
        iou = cnt / np.maximum(a_seed + areas - cnt, 1.0)
        iou[0] = 0.0
        j = int(iou.argmax())
        if iou[j] >= CFG["min_target_iou"]:
            acc += (Lc == j)
    return acc / max(len(labs), 1)


def make_sample(stem, fid, seed, seed_label, S_lab, z8, f8, prob, labs, kind):
    win = window(*bbox(seed))
    r0, c0, S = win
    sc = crop(seed, r0, c0, S)
    lab_c = crop(S_lab, r0, c0, S)
    if seed_label is not None:
        oth = (lab_c > 0) & (lab_c != seed_label)
    else:  # perturbed annotation: hide the selected instances it overlaps, as the real seed would be hidden
        ov = np.unique(lab_c[sc])
        oth = (lab_c > 0) & ~np.isin(lab_c, ov[ov > 0])
    x = np.stack([rs(crop(z8, r0, c0, S)), rs(crop(f8, r0, c0, S)), rs(crop(prob, r0, c0, S)),
                  rs(sc.astype(np.uint8) * 255), rs(oth.astype(np.uint8) * 255)])
    y = None if labs is None else np.round(np.clip(rs(target_for(sc, win, labs)), 0, 1) * 255).astype(np.uint8)
    return dict(stem=stem, fid=fid, kind=kind, r0=r0, c0=c0, S=S, x=x, y=y)


K3 = np.ones((3, 3), np.uint8)


def perturb(m, prob):
    r0, r1, c0, c1 = bbox(m)
    pad = 12
    R0, R1, C0, C1 = max(r0 - pad, 0), min(r1 + pad, H), max(c0 - pad, 0), min(c1 + pad, W)
    sub = m[R0:R1, C0:C1].astype(np.uint8)
    op = int(RNG.integers(5))
    if op == 1:
        sub = cv2.dilate(sub, K3, iterations=int(RNG.integers(1, 5)))
    elif op == 2:
        e = cv2.erode(sub, K3, iterations=int(RNG.integers(1, 3)))
        sub = e if e.sum() >= 16 else sub
    elif op == 3:
        dy, dx = RNG.integers(-4, 5, size=2)
        sub = np.roll(np.roll(sub, int(dy), 0), int(dx), 1)
    elif op == 4:
        region = cv2.dilate(sub, K3, iterations=5)
        s2 = ((prob[R0:R1, C0:C1] > RNG.uniform(0.2, 0.7) * 255) & (region > 0)).astype(np.uint8)
        sub = s2 if s2.sum() >= 16 else sub
    out = np.zeros((H, W), bool)
    out[R0:R1, C0:C1] = sub > 0
    return out if out.sum() >= 9 else None


def process(stem, split):
    test = split == "test"
    img = cv2.imread(os.path.join(TEST_DIR if test else TRAIN_DIR, stem + ".jpeg"), cv2.IMREAD_GRAYSCALE)
    z8, f8 = channels(img)
    prob = prob_map(stem, test)
    labs = None
    if not test:
        labs = []
        for rid in recs_by_stem[stem]:
            L = np.zeros((H, W), np.int16)
            areas = [0]
            for j, polys in enumerate(inst_by_rec.get(rid, []), start=1):
                m = rasterise(polys)
                L[m > 0] = j
                areas.append(int(m.sum()))
            labs.append((L, np.asarray(areas, np.int64)))
    seeds = SEEDS[split].get(stem, [])
    S_lab = np.zeros((H, W), np.int16)
    masks = []
    for i, (fid, rle) in enumerate(seeds, start=1):
        m = decode(rle)
        S_lab[m] = i
        masks.append(m)
    out = [make_sample(stem, fid, m, i, S_lab, z8, f8, prob, labs, 0)
           for i, ((fid, _), m) in enumerate(zip(seeds, masks), start=1) if m.any()]
    if not test and CFG["gt_seeds"] and labs:
        L, areas = labs[int(RNG.integers(len(labs)))]
        for j in range(1, len(areas)):
            if areas[j] < 9:
                continue
            s = perturb(L == j, prob)
            if s is not None:
                out.append(make_sample(stem, None, s, None, S_lab, z8, f8, prob, labs, 1))
    return out


def build(stems, split):
    samples = []
    for i, s in enumerate(stems):
        samples.extend(process(s, split))
        if i % 100 == 0:
            log(f"{split}: {i + 1}/{len(stems)} images, {len(samples)} samples")
    X = np.stack([d["x"] for d in samples])
    Y = None if split == "test" else np.stack([d["y"] for d in samples])
    meta = pd.DataFrame([{k: d[k] for k in ("stem", "fid", "kind", "r0", "c0", "S")} for d in samples])
    meta["fold"] = meta["stem"].map(fold_of).fillna(-1).astype(int) if split != "test" else -1
    return X, Y, meta


Xtr, Ytr, Mtr = build(train_stems, "train")
log("train samples", len(Mtr), "| real seeds", int((Mtr.kind == 0).sum()), "| perturbed annotations", int((Mtr.kind == 1).sum()),
    "| mean target", float(Ytr.mean() / 255))
Xte, _, Mte = build(test_stems, "test")
log("test samples", len(Mte))
LOG["n_train_samples"], LOG["n_test_samples"] = int(len(Mtr)), int(len(Mte))

# ----------------------------------------------------------------------------- model
import timm

LOG["versions"] = dict(torch=torch.__version__, timm=timm.__version__, cv2=cv2.__version__, python=sys.version.split()[0])


class DecBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_c + skip_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.c2(self.c1(x))


class UNet(nn.Module):
    def __init__(self, encoder, in_ch, pretrained):
        super().__init__()
        self.enc = timm.create_model(encoder, features_only=True, pretrained=pretrained, in_chans=in_ch)
        chs = self.enc.feature_info.channels()
        skips = chs[:-1][::-1] + [0]
        dec = [256, 128, 64, 32, 16][:len(skips)]
        self.blocks = nn.ModuleList()
        prev = chs[-1]
        for sc, oc in zip(skips, dec):
            self.blocks.append(DecBlock(prev, sc, oc))
            prev = oc
        self.head = nn.Conv2d(prev, 1, 3, padding=1)

    def forward(self, x):
        feats = self.enc(x)
        y = feats[-1]
        for blk, s in zip(self.blocks, feats[:-1][::-1] + [None]):
            y = blk(y, s)
        return self.head(y)


dev = torch.device("cuda")
n_gpu = torch.cuda.device_count()
LOG["gpus"] = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
B = CFG["batch_per_gpu"] * max(1, n_gpu)


def norm(x8):
    x = x8.float()
    x[:, 0:2] = (x[:, 0:2] / 32.0 - 4.0) / 2.0
    x[:, 2:5] = x[:, 2:5] / 255.0
    return x


def augment(x, y):
    k = random.randint(0, 3)
    if k:
        x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))
    if random.random() < 0.5:
        x, y = torch.flip(x, (3,)), torch.flip(y, (3,))
    a = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    b = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(-0.15, 0.15)
    x[:, 0:2] = x[:, 0:2] * a + b
    u = random.random()
    if u < 0.3:
        x[:, 3:4] = F.max_pool2d(x[:, 3:4], 3, 1, 1)
    elif u < 0.45:
        x[:, 3:4] = -F.max_pool2d(-x[:, 3:4], 3, 1, 1)
    return x, y


def soft_dice_loss(logits, y):
    p = torch.sigmoid(logits)
    return 1 - (2 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)


@torch.no_grad()
def predict(m, X, idx):
    m.eval()
    out = np.zeros((len(idx), CROP, CROP), np.float32)
    for s in range(0, len(idx), 64):
        xb = norm(torch.from_numpy(X[idx[s:s + 64]]).to(dev))
        acc = torch.zeros((xb.shape[0], 1, CROP, CROP), device=dev)
        for k in range(4):
            for fl in (False, True):
                xt = torch.rot90(xb, k, (2, 3))
                if fl:
                    xt = torch.flip(xt, (3,))
                with torch.autocast("cuda", dtype=torch.float16):
                    yt = torch.sigmoid(m(xt.contiguous(memory_format=torch.channels_last)).float())
                if fl:
                    yt = torch.flip(yt, (3,))
                acc += torch.rot90(yt, -k, (2, 3))
        out[s:s + xb.shape[0]] = (acc / 8.0)[:, 0].cpu().numpy()
    return out


def train_fold(k):
    tr = np.nonzero(Mtr["fold"].to_numpy() != k)[0]
    va = np.nonzero((Mtr["fold"].to_numpy() == k) & (Mtr["kind"].to_numpy() == 0))[0]
    try:
        model = UNet(CFG["encoder"], CFG["in_ch"], pretrained=True)
        pre = True
    except Exception as e:  # noqa: BLE001
        log("WARNING pretrained weights unavailable:", repr(e)[:200])
        model, pre = UNet(CFG["encoder"], CFG["in_ch"], pretrained=False), False
    model = model.to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    net = nn.DataParallel(model) if n_gpu > 1 else model
    spe = max(1, len(tr) // B)
    total = spe * CFG["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    warm = min(CFG["warmup"], max(1, total // 5))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    scaler = torch.amp.GradScaler("cuda")
    step, rec, tf = 0, [], time.time()
    for epoch in range(CFG["epochs"]):
        model.train()
        perm = np.random.permutation(tr)
        run = 0.0
        for s in range(0, spe * B, B):
            idx = np.sort(perm[s:s + B])
            x = norm(torch.from_numpy(Xtr[idx]).to(dev, non_blocking=True))
            y = torch.from_numpy(Ytr[idx]).to(dev, non_blocking=True).float().unsqueeze(1) / 255.0
            x, y = augment(x, y)
            with torch.autocast("cuda", dtype=torch.float16):
                lg = net(x.contiguous(memory_format=torch.channels_last))
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg, y) + CFG["dice_w"] * soft_dice_loss(lg, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                d = min(CFG["ema"], (1 + step) / (10 + step))
                msd = model.state_dict()
                for kk, v in ema.state_dict().items():
                    if v.dtype.is_floating_point:
                        v.mul_(d).add_(msd[kk].detach(), alpha=1 - d)
                    else:
                        v.copy_(msd[kk])
            run += loss.item()
            step += 1
        rec.append(dict(epoch=epoch, train_loss=run / spe))
        log(f"fold {k} epoch {epoch}: loss {run / spe:.4f}")
    # validation loss on the fold's real seeds (EMA model)
    vb = vd = 0.0
    if len(va):
        p = predict(ema, Xtr, va)
        yv = Ytr[va].astype(np.float32) / 255.0
        pc = np.clip(p, 1e-4, 1 - 1e-4)
        vb = float(-(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)).mean())
        vd = float(1 - (2 * (p * yv).sum() + 1) / (p.sum() + yv.sum() + 1))
    LOG["folds"][str(k)] = dict(epochs=rec, val_bce=vb, val_dice_loss=vd, n_train=int(len(tr)), n_val=int(len(va)),
                                sec=time.time() - tf, pretrained=pre)
    torch.save({kk: (v.half() if v.dtype.is_floating_point else v) for kk, v in ema.state_dict().items()}, f"{OUT}/refiner_f{k}.pt")
    return ema


folds_to_run = sorted(set(Mtr["fold"])) if not SMOKE else sorted(set(Mtr["fold"]))[:1]
oof_idx = np.nonzero(Mtr["kind"].to_numpy() == 0)[0]
oof_prob = np.zeros((len(oof_idx), CROP, CROP), np.uint8)
oof_fold = Mtr["fold"].to_numpy()[oof_idx]
test_acc = np.zeros((len(Mte), CROP, CROP), np.float32)
for k in folds_to_run:
    m = train_fold(k)
    sel = np.nonzero(oof_fold == k)[0]
    if len(sel):
        oof_prob[sel] = np.round(predict(m, Xtr, oof_idx[sel]) * 255).astype(np.uint8)
    test_acc += predict(m, Xte, np.arange(len(Mte)))
    del m
    torch.cuda.empty_cache()
    log(f"fold {k} done: val BCE {LOG['folds'][str(k)]['val_bce']:.4f}, val dice loss {LOG['folds'][str(k)]['val_dice_loss']:.4f}")
test_prob = np.round(test_acc / len(folds_to_run) * 255).astype(np.uint8)

keep = np.isin(oof_fold, folds_to_run)
np.savez_compressed(f"{OUT}/refined_oof.npz", probs=oof_prob[keep])
Mtr.iloc[oof_idx[keep]].rename(columns={"fid": "filament_id"}).drop(columns="kind").reset_index(drop=True).to_parquet(f"{OUT}/refined_oof_meta.parquet")
np.savez_compressed(f"{OUT}/refined_test.npz", probs=test_prob)
Mte.rename(columns={"fid": "filament_id"}).drop(columns="kind").reset_index(drop=True).to_parquet(f"{OUT}/refined_test_meta.parquet")
LOG["folds_run"] = [int(k) for k in folds_to_run]
LOG["total_sec"] = time.time() - t0
json.dump(LOG, open(f"{OUT}/log.json", "w"), indent=1, default=str)
log("done", f"{LOG['total_sec']:.0f}s")
