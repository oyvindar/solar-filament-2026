"""Solar Filament Segmentation Challenge 2026 -- soft-label U-Net, one CV fold.

Stage 1 of the pipeline. Trains a U-Net (timm ResNet34 encoder, two input
channels: disk-standardised intensity and a log flat-field ratio) on native
resolution 1024x1024 crops. The target is the fraction of annotators that
marked each pixel (soft label), so the output approximates
p(x) = P(a random annotator paints pixel x).

Outputs (in /kaggle/working):
  oof/<stem>.png    probability maps (uint8, p*255) for this fold's validation images
  test/<stem>.png   probability maps for the 180 test images
  fold<k>.pt        EMA weights (fp16 state dict)
  log.json          configuration, versions, losses, timings

Generated from kaggle/train_unet_template.py by kaggle/make_kernels.py.
"""
FOLD = __FOLD__
SMOKE = __SMOKE__
FOLDS = "__FOLDS__"  # "stem:fold;stem:fold;..." -- date-grouped, year-stratified 5-fold split

CFG = dict(encoder="resnet34", in_ch=2, crop=1024, batch_per_gpu=4, epochs=18, crops_per_img=2,
           lr=3e-4, wd=1e-4, warmup=150, ema=0.998, dice_w=0.5, seed=2026 + FOLD, val_every=3)
if SMOKE:
    CFG.update(epochs=1, max_steps=25, n_pred=2, val_every=1)

import os, sys, glob, json, time, math, random, subprocess, copy

t0 = time.time()
def log(*a):
    print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
OUT = os.environ.get("FSC_OUT", "/kaggle/working")  # override to run several models in one session
INPUT = os.environ.get("FSC_INPUT", "/kaggle/input")
os.makedirs(f"{OUT}/oof", exist_ok=True); os.makedirs(f"{OUT}/test", exist_ok=True)
LOG = dict(cfg=CFG, fold=FOLD, smoke=SMOKE, epochs=[], notes=[])

# ----------------------------------------------------------------------------- data
ann_json = glob.glob(os.path.join(INPUT, "**", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), recursive=True)
assert ann_json, "annotation JSON not found under /kaggle/input"
ROOT = os.path.dirname(os.path.dirname(ann_json[0]))
TRAIN_DIR = os.path.join(ROOT, "train", "train_images")
TEST_DIR = os.path.join(ROOT, "test", "test_images")
if not os.path.isdir(TRAIN_DIR) or len(glob.glob(f"{TRAIN_DIR}/*.jpeg")) < 700:
    z = glob.glob(os.path.join(INPUT, "**", "train_images.7z"), recursive=True)
    assert z, "train images not found (neither folder nor 7z)"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py7zr"], check=True)
    import py7zr
    with py7zr.SevenZipFile(z[0]) as zf:
        zf.extractall("/kaggle/temp")
    TRAIN_DIR = os.path.dirname(glob.glob("/kaggle/temp/**/*.jpeg", recursive=True)[0])
    LOG["notes"].append("train images extracted from 7z")
log("ROOT", ROOT, "| train", len(glob.glob(f"{TRAIN_DIR}/*.jpeg")), "| test", len(glob.glob(f"{TEST_DIR}/*.jpeg")))

try:
    from pycocotools import mask as mu
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=False)
    try:
        from pycocotools import mask as mu
    except ImportError:
        mu = None
LOG["rasteriser"] = "pycocotools" if mu is not None else "cv2.fillPoly"
log("rasteriser:", LOG["rasteriser"])

def rasterise(polys, h=2048, w=2048):
    polys = [p for p in polys if len(p) >= 6]
    if not polys:
        return np.zeros((h, w), np.uint8)
    if mu is not None:
        return mu.decode(mu.merge(mu.frPyObjects(polys, h, w))).astype(np.uint8)
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p, np.float64).reshape(-1, 2)).astype(np.int32)], 1)
    return m

J = json.load(open(ann_json[0]))
polys_by_rec = {}
for a in J["annotations"]:
    polys_by_rec.setdefault(a["image_id"], []).extend(a["segmentation"])
recs_by_stem = {}
for im in J["images"]:
    recs_by_stem.setdefault(im["file_name"].rsplit(".", 1)[0], []).append(im["id"])
fold_of = dict(s.split(":") for s in FOLDS.split(";"))
fold_of = {k: int(v) for k, v in fold_of.items()}
stems = sorted(recs_by_stem)
assert set(stems) == set(fold_of), "fold table does not match annotation file"
tr_stems = [s for s in stems if fold_of[s] != FOLD]
va_stems = [s for s in stems if fold_of[s] == FOLD]
te_stems = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{TEST_DIR}/*.jpeg"))
if SMOKE:
    tr_stems = tr_stems[:24]
log("train", len(tr_stems), "| val", len(va_stems), "| test", len(te_stems))

def read_gray(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert im is not None and im.shape == (2048, 2048), path
    return im

def make_channels(img):
    """uint8 image -> (2, H, W) float16 standardised channels."""
    f = img.astype(np.float32)
    disk = f > 10
    blur = cv2.GaussianBlur(f, (0, 0), 24)
    ff = np.where(disk, np.log1p(f) - np.log1p(blur), 0.0).astype(np.float32)
    m0, s0 = f[disk].mean(), f[disk].std() + 1e-6
    s1 = ff[disk].std() + 1e-6
    x = np.stack([(f - m0) / s0, ff / s1]).astype(np.float16)
    return x

def soft_target(stem):
    acc = np.zeros((2048, 2048), np.float32)
    for rid in recs_by_stem[stem]:
        acc += rasterise(polys_by_rec.get(rid, []))
    return np.round(255 * acc / len(recs_by_stem[stem])).astype(np.uint8)

def load_set(stem_list, dirname, with_target):
    X, Y = [], []
    for s in stem_list:
        X.append(make_channels(read_gray(os.path.join(dirname, s + ".jpeg"))))
        if with_target:
            Y.append(soft_target(s))
    return X, Y

Xtr, Ytr = load_set(tr_stems, TRAIN_DIR, True)
log("loaded train", len(Xtr), "| pos frac %.4f" % float(np.mean([(y > 0).mean() for y in Ytr[:50]])))
Xva, Yva = load_set(va_stems, TRAIN_DIR, True)
log("loaded val", len(Xva))

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
            self.blocks.append(DecBlock(prev, sc, oc)); prev = oc
        self.head = nn.Conv2d(prev, 1, 3, padding=1)
    def forward(self, x):
        feats = self.enc(x)
        y = feats[-1]
        for blk, s in zip(self.blocks, feats[:-1][::-1] + [None]):
            y = blk(y, s)
        return self.head(y)

try:
    model = UNet(CFG["encoder"], CFG["in_ch"], pretrained=True); LOG["pretrained"] = True
except Exception as e:
    log("WARNING pretrained weights unavailable:", repr(e)[:200])
    model = UNet(CFG["encoder"], CFG["in_ch"], pretrained=False); LOG["pretrained"] = False
log("pretrained:", LOG["pretrained"])
dev = torch.device("cuda")
model = model.to(dev).to(memory_format=torch.channels_last)
ema = copy.deepcopy(model).eval()
for p in ema.parameters():
    p.requires_grad_(False)
n_gpu = torch.cuda.device_count()
LOG["gpus"] = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
log("gpus:", LOG["gpus"])
net = nn.DataParallel(model) if n_gpu > 1 else model
B = CFG["batch_per_gpu"] * max(1, n_gpu)

steps_per_epoch = (len(tr_stems) * CFG["crops_per_img"]) // B
total_steps = steps_per_epoch * CFG["epochs"]
if SMOKE:
    total_steps = min(total_steps, CFG["max_steps"])
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
def lr_lambda(step):
    if step < CFG["warmup"]:
        return (step + 1) / CFG["warmup"]
    t = (step - CFG["warmup"]) / max(1, total_steps - CFG["warmup"])
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
scaler = torch.amp.GradScaler("cuda")

def soft_dice_loss(logits, y):
    p = torch.sigmoid(logits)
    inter = (p * y).sum()
    return 1 - (2 * inter + 1.0) / (p.sum() + y.sum() + 1.0)

def augment(x, y):
    k = random.randint(0, 3)
    if k:
        x = torch.rot90(x, k, (2, 3)); y = torch.rot90(y, k, (2, 3))
    if random.random() < 0.5:
        x = torch.flip(x, (3,)); y = torch.flip(y, (3,))
    a = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
    b = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(-0.15, 0.15)
    x = x * a + b + 0.03 * torch.randn_like(x)
    return x, y

C = CFG["crop"]
def make_batch(idx):
    xs, ys = [], []
    for i in idx:
        r, c = np.random.randint(0, 2048 - C + 1, size=2)
        xs.append(Xtr[i][:, r:r + C, c:c + C]); ys.append(Ytr[i][None, r:r + C, c:c + C])
    x = torch.from_numpy(np.stack(xs)).to(dev, non_blocking=True).float()
    y = torch.from_numpy(np.stack(ys)).to(dev, non_blocking=True).float() / 255.0
    return augment(x, y)

@torch.no_grad()
def ema_update(decay):
    msd = model.state_dict()
    for k, v in ema.state_dict().items():
        if v.dtype.is_floating_point:
            v.mul_(decay).add_(msd[k].detach(), alpha=1 - decay)
        else:
            v.copy_(msd[k])

@torch.no_grad()
def val_loss(m):
    m.eval(); tot_bce, tot_dice, n = 0.0, 0.0, 0
    for x, y in zip(Xva[: (8 if SMOKE else len(Xva))], Yva):
        xb = torch.from_numpy(x[None, :, 512:1536, 512:1536].astype(np.float32)).to(dev)
        yb = torch.from_numpy(y[None, None, 512:1536, 512:1536].astype(np.float32) / 255.0).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = m(xb).float()
        tot_bce += F.binary_cross_entropy_with_logits(lg, yb).item(); tot_dice += soft_dice_loss(lg, yb).item(); n += 1
    return tot_bce / n, tot_dice / n

# ----------------------------------------------------------------------------- train
log(f"training: {total_steps} steps, batch {B}, {steps_per_epoch} steps/epoch")
step, t_train = 0, time.time()
for epoch in range(CFG["epochs"]):
    net.train(); model.train()
    order = np.random.permutation(np.repeat(np.arange(len(Xtr)), CFG["crops_per_img"]))
    run = 0.0; nb = 0; te = time.time()
    for s in range(0, len(order) - B + 1, B):
        if step >= total_steps:
            break
        x, y = make_batch(order[s:s + B])
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = net(x)
        lg = lg.float()
        loss = F.binary_cross_entropy_with_logits(lg, y) + CFG["dice_w"] * soft_dice_loss(lg, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        ema_update(min(CFG["ema"], (1 + step) / (10 + step)))
        run += loss.item(); nb += 1; step += 1
    rec = dict(epoch=epoch, train_loss=run / max(1, nb), sec=time.time() - te, lr=sched.get_last_lr()[0])
    if (epoch + 1) % CFG["val_every"] == 0 or epoch == CFG["epochs"] - 1:
        rec["val_bce"], rec["val_dice_loss"] = val_loss(ema)
    LOG["epochs"].append(rec); log(rec)
    if step >= total_steps:
        break
LOG["train_sec"] = time.time() - t_train
torch.save({k: (v.half() if v.dtype.is_floating_point else v) for k, v in ema.state_dict().items()}, f"{OUT}/fold{FOLD}.pt")

# ----------------------------------------------------------------------------- predict
@torch.no_grad()
def predict_full(m, x):
    x = torch.from_numpy(x[None].astype(np.float32)).to(dev)
    acc = torch.zeros((1, 1, 2048, 2048), device=dev)
    for k in range(4):
        xr = torch.rot90(x, k, (2, 3))
        xb = torch.cat([xr, torch.flip(xr, (3,))], 0).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            yb = torch.sigmoid(m(xb).float())
        y = yb[0:1] + torch.flip(yb[1:2], (3,))
        acc += torch.rot90(y, -k, (2, 3))
    return (acc / 8)[0, 0].cpu().numpy()

def write_prob(path, p):
    cv2.imwrite(path, np.round(p * 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 6])

ema.eval(); tp = time.time()
n_pred = CFG.get("n_pred", None)
for s, x in list(zip(va_stems, Xva))[:n_pred]:
    write_prob(f"{OUT}/oof/{s}.png", predict_full(ema, x))
log("oof written", len(os.listdir(f"{OUT}/oof")), "%.1fs" % (time.time() - tp))
del Xtr, Ytr
for s in te_stems[:n_pred]:
    write_prob(f"{OUT}/test/{s}.png", predict_full(ema, make_channels(read_gray(os.path.join(TEST_DIR, s + ".jpeg")))))
LOG["predict_sec"] = time.time() - tp
LOG["total_sec"] = time.time() - t0
log("test written", len(os.listdir(f"{OUT}/test")), "| total %.0fs" % LOG["total_sec"])
json.dump(LOG, open(f"{OUT}/log.json", "w"), indent=1, default=str)
