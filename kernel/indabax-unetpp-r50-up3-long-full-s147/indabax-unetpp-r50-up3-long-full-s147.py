KERNEL_ARGS = '''--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --tta 1 --full --lr 5e-4 --aug 2 --dropout 0.2 --arch UnetPlusPlus --encoder resnet50 --epochs 9 --bs 16 --upscale 3 --seed 147 --amp 1 --eval_bs 16'''.split()
import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch'], check=False)
import numpy as np
from scipy.ndimage import uniform_filter

THR = 1.0
WIN = 3


def fractions(binary, win=WIN):
    # binary: (N, H, W) -> neighbourhood fraction of rainy pixels, zero padded at edges
    return uniform_filter(binary.astype(np.float32), size=(1, win, win), mode="constant", cval=0.0)


def fss_components(pred, obs, thr=THR, win=WIN):
    pf = fractions(pred > thr, win)
    po = fractions(obs > thr, win)
    num = ((pf - po) ** 2).sum(axis=(1, 2))
    den = (pf ** 2).sum(axis=(1, 2)) + (po ** 2).sum(axis=(1, 2))
    return num, den  # per sample


def fss_pooled(pred, obs, thr=THR, win=WIN):
    """pred, obs: (N, 3, H, W). FSS per lead time pooled over samples, then mean of 3."""
    scores = []
    for k in range(obs.shape[1]):
        num, den = fss_components(pred[:, k], obs[:, k], thr, win)
        scores.append(1.0 - num.sum() / max(den.sum(), 1e-12))
    return float(np.mean(scores)), scores


def fss_per_sample(pred, obs, thr=THR, win=WIN, empty=1.0):
    """Mean over samples of per-sample FSS; samples with no rain in either field score `empty`."""
    scores = []
    for k in range(obs.shape[1]):
        num, den = fss_components(pred[:, k], obs[:, k], thr, win)
        s = np.where(den > 0, 1.0 - num / np.maximum(den, 1e-12), empty)
        scores.append(float(s.mean()))
    return float(np.mean(scores)), scores


def report(name, pred, obs):
    p, pk = fss_pooled(pred, obs)
    s, sk = fss_per_sample(pred, obs)
    print(f"{name:28s} pooled={p:.4f} [{pk[0]:.3f} {pk[1]:.3f} {pk[2]:.3f}]  "
          f"per-sample={s:.4f} [{sk[0]:.3f} {sk[1]:.3f} {sk[2]:.3f}]", flush=True)
    return p

import sys, numpy as np, cv2
from scipy.ndimage import map_coordinates


def flow_between(a, b):
    a8 = np.clip(np.log1p(np.maximum(a, 0)) * 40, 0, 255).astype(np.uint8)
    b8 = np.clip(np.log1p(np.maximum(b, 0)) * 40, 0, 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 3, 9, 3, 5, 1.1, 0)  # (H,W,2) dx,dy


def advect(field, flow, steps):
    H, W = field.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = []
    for s in range(1, steps + 1):
        # semi-Lagrangian backward: value at x comes from x - s*flow
        src_x = xx - s * flow[..., 0]; src_y = yy - s * flow[..., 1]
        out.append(map_coordinates(field, [src_y, src_x], order=1, mode="constant", cval=0.0))
    return np.stack(out)


def advect_batch(X, steps=3):
    """X: (N,3,H,W) -> (N,steps,H,W) advected forecasts of the last frame."""
    out = np.zeros((X.shape[0], steps) + X.shape[2:], np.float32)
    for i in range(X.shape[0]):
        fl = flow_between(X[i, 1], X[i, 2])
        out[i] = advect(X[i, 2], fl, steps)
    return out



import argparse, time, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", default="outputs")
p.add_argument("--encoder", default="resnet18")
p.add_argument("--arch", default="Unet", choices=["Unet", "UnetPlusPlus", "FPN", "DeepLabV3Plus"], help="segmentation-models-pytorch decoder")
p.add_argument("--attention", default="", choices=["", "scse"], help="scse: squeeze-and-excitation attention blocks in the Unet / UnetPlusPlus decoder")
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--fss_w", type=float, default=1.0)
p.add_argument("--csi_w", type=float, default=0.0, help="weight of batch-pooled soft CSI (IoU) loss")
p.add_argument("--ms_fss", type=int, default=0, help="1: add 5x5 and 9x9 soft FSS terms (weighted more for later leads)")
p.add_argument("--pos_w", type=float, default=8.0, help="BCE positive class weight")
p.add_argument("--mixup", type=float, default=0.0, help="prob of temporal mixup with the next hourly sample")
p.add_argument("--wet_sampling", type=int, default=0, help="1: sample training rows with prob proportional to q_min + rain fraction")
p.add_argument("--full", action="store_true", help="train on all data (no validation), for final submission")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--limit", type=int, default=0, help="debug: use only first N train samples")
p.add_argument("--threads", type=int, default=12)
p.add_argument("--aug", type=int, default=0, help="1: flips + 90-degree rotations, 2: random translations")
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--stride", type=int, default=1, help="use every k-th hourly train sample")
p.add_argument("--save_each", action="store_true", help="save test/val probs at every epoch")
p.add_argument("--upscale", type=int, default=1, help="2: upsample inputs 2x before the encoder, downsample logits back")
p.add_argument("--aux_reg", type=float, default=0.0, help="weight of an auxiliary head regressing log1p(rain)/2 at each lead (multi-task)")
p.add_argument("--tta", type=int, default=0, help="1: average predictions over 8-pixel shifted copies (up, down, left, right)")
p.add_argument("--eval_bs", type=int, default=64, help="batch size for validation and test predictions (lower it if they run out of GPU memory)")
p.add_argument("--amp", type=int, default=0, help="1: mixed precision (float16 forward pass with loss scaling) on GPU, faster and lighter")
p.add_argument("--advect", type=int, default=0, help="1: add optical-flow advected maps (+1h,+2h,+3h) as input channels")
a = p.parse_args(KERNEL_ARGS)
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
AMP = bool(a.amp) and dev == "cuda"
def autocast(): return torch.autocast(device_type="cuda" if AMP else "cpu", dtype=torch.float16, enabled=AMP)

import os, glob
if not os.path.exists(f"{a.data}/train.npz"):  # Kaggle sometimes mounts competition data under another path
    found = glob.glob("/kaggle/input/**/train.npz", recursive=True)
    print("data not at", a.data, "found", found, flush=True)
    if found: a.data = os.path.dirname(found[0])
tr = np.load(f"{a.data}/train.npz"); te = np.load(f"{a.data}/test.npz")
X, y, times = tr["X"], tr["y"], tr["times"]; Xt, tt = te["X"], te["times"]
lats, lons = tr["lats"], tr["lons"]
if a.limit: X, y, times = X[:a.limit], y[:a.limit], times[:a.limit]
ts = pd.to_datetime(times); tst = pd.to_datetime(tt)
val = np.asarray((ts >= "2022-12-01") & (ts < "2023-04-01")) if not a.full else np.zeros(len(ts), bool)

ADV, ADVt = (advect_batch(X), advect_batch(Xt)) if a.advect else (None, None)
if a.advect: print("advected inputs ready", ADV.shape, flush=True)
N_IN = 8 + (3 if a.advect else 0)
N_OUT = 6 if a.aux_reg > 0 else 3
lat_c = ((lats - lats.mean()) / lats.std()).astype(np.float32)
lon_c = ((lons - lons.mean()) / lons.std()).astype(np.float32)

def time_feats(t):
    h = t.hour.values + t.minute.values / 60.0
    doy = t.dayofyear.values.astype(np.float32)
    sd = np.where(doy > 300, doy - 335, doy + 30) / 120.0  # Dec 1 -> 0 ... Mar 31 -> ~1
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), sd], 1).astype(np.float32)

class Rows:
    """Read-only view of arr[idx] that avoids copying (saves about 3 GB of RAM, needed on Colab)."""
    def __init__(self, arr, idx): self.arr, self.idx = arr, idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): return self.arr[self.idx[i]]

class DS(torch.utils.data.Dataset):
    def __init__(self, X, y, tf, aug=0, adv=None, mixup=0.0): self.X, self.y, self.tf, self.aug, self.adv, self.mixup = X, y, tf, aug, adv, mixup
    def __len__(self): return len(self.X)
    def target(self, i):
        """3 binary rain maps (> 1 mm/h); with --aux_reg also 3 maps of log1p(rain)/2."""
        b = (self.y[i] > 1.0).astype(np.float32)
        if a.aux_reg > 0: b = np.concatenate([b, (np.log1p(np.maximum(self.y[i], 0)) / 2.0).astype(np.float32)], 0)
        return b
    def __getitem__(self, i):
        x = np.log1p(np.maximum(self.X[i], 0)) / 2.0  # (3,H,W)
        H, W = x.shape[1:]
        tfe = np.broadcast_to(self.tf[i][:, None, None], (3, H, W))
        parts = [x, lat_c[None], lon_c[None], tfe]
        if self.adv is not None: parts.append(np.log1p(np.maximum(self.adv[i], 0)) / 2.0)
        inp = np.concatenate(parts, 0).astype(np.float32)
        tgt = self.target(i) if self.y is not None else np.zeros((N_OUT, H, W), np.float32)
        if self.mixup > 0 and self.y is not None and i + 1 < len(self.X) and np.random.rand() < self.mixup:
            lam = np.random.rand()
            x2 = np.log1p(np.maximum(self.X[i + 1], 0)) / 2.0
            parts2 = [x2, lat_c[None], lon_c[None], np.broadcast_to(self.tf[i + 1][:, None, None], (3, H, W))]
            if self.adv is not None: parts2.append(np.log1p(np.maximum(self.adv[i + 1], 0)) / 2.0)
            inp = lam * inp + (1 - lam) * np.concatenate(parts2, 0).astype(np.float32)
            tgt = lam * tgt + (1 - lam) * self.target(i + 1)
        if self.aug == 1:  # rotations and flips: break the geographic motion prior, kept for reference
            k = np.random.randint(4)
            inp, tgt = np.rot90(inp, k, axes=(1, 2)), np.rot90(tgt, k, axes=(1, 2))
            if np.random.rand() < 0.5: inp, tgt = inp[:, :, ::-1], tgt[:, :, ::-1]
            inp, tgt = np.ascontiguousarray(inp), np.ascontiguousarray(tgt)
        elif self.aug == 2:  # random translation up to 16 px with zero fill: keeps motion direction
            dy, dx = np.random.randint(-16, 17, size=2)
            def shift(arr):
                out = np.zeros_like(arr)
                ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
                xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
                out[:, yd, xd] = arr[:, ys, xs]
                return out
            inp, tgt = shift(inp), shift(tgt)
        return torch.from_numpy(inp), torch.from_numpy(tgt)

tf_tr, tf_te = time_feats(ts), time_feats(tst)
tr_idx = np.where(~val)[0][::a.stride]
sub = lambda arr, idx: None if arr is None else arr[idx]
ds_tr = DS(Rows(X, tr_idx), Rows(y, tr_idx), tf_tr[tr_idx], a.aug, sub(ADV, tr_idx), a.mixup)
if a.wet_sampling:
    wet = np.array([(y[j] > 1.0).mean() for j in tr_idx])
    wts = torch.tensor(0.02 + wet / max(wet.mean(), 1e-6), dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(wts, num_samples=len(tr_idx), replacement=True)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.bs, sampler=sampler, num_workers=2, drop_last=True)
else:
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.bs, shuffle=True, num_workers=2, drop_last=True)
va_idx = np.where(val)[0]
dl_va = torch.utils.data.DataLoader(DS(Rows(X, va_idx), Rows(y, va_idx), tf_tr[val], 0, sub(ADV, val)), batch_size=a.eval_bs, num_workers=2) if val.any() else None
dl_te = torch.utils.data.DataLoader(DS(Xt, None, tf_te, 0, ADVt), batch_size=a.eval_bs, num_workers=2)

extra = {"decoder_attention_type": a.attention} if a.attention else {}
model = getattr(smp, a.arch)(a.encoder, encoder_weights="imagenet", in_channels=N_IN, classes=N_OUT, **extra).to(dev)
if a.dropout > 0:
    model.segmentation_head = nn.Sequential(nn.Dropout2d(a.dropout), model.segmentation_head)
if a.upscale > 1:
    class Upscaled(nn.Module):
        def __init__(self, m, f): super().__init__(); self.m, self.f = m, f
        def forward(self, x):
            y = self.m(F.interpolate(x, scale_factor=self.f, mode="bilinear", align_corners=False))
            return F.avg_pool2d(y, self.f)
    model = Upscaled(model, a.upscale).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
steps = a.epochs * len(dl_tr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
pos_w = torch.tensor([a.pos_w], device=dev)
scaler = torch.cuda.amp.GradScaler(enabled=AMP)  # rescales the loss so float16 gradients do not underflow

LEAD_W = torch.tensor([1.0, 1.0, 1.0], device=dev)
def soft_fss_loss(logits, tgt, k=3, lead_w=LEAD_W):
    # batch-pooled, like the competition metric (pooled over all test samples)
    pr = torch.sigmoid(logits)
    pf = F.avg_pool2d(pr, k, stride=1, padding=k // 2)
    po = F.avg_pool2d(tgt, k, stride=1, padding=k // 2)
    num = ((pf - po) ** 2).sum(dim=(0, 2, 3))
    den = (pf ** 2).sum(dim=(0, 2, 3)) + (po ** 2).sum(dim=(0, 2, 3)) + 1e-6
    return ((num / den) * lead_w).sum() / lead_w.sum()
def soft_csi_loss(logits, tgt):
    pr = torch.sigmoid(logits)
    inter = (pr * tgt).sum(dim=(0, 2, 3)); union = pr.sum(dim=(0, 2, 3)) + tgt.sum(dim=(0, 2, 3)) - inter
    return (1 - inter / (union + 1e-6)).mean()
def total_loss(logits, yb):
    if a.aux_reg > 0:
        reg = F.mse_loss(logits[:, 3:], yb[:, 3:]); logits, yb = logits[:, :3], yb[:, :3]
    else:
        reg = 0.0
    l = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_w) + a.fss_w * soft_fss_loss(logits, yb)
    if a.ms_fss:
        l = l + 0.5 * soft_fss_loss(logits, yb, 5, torch.tensor([0.5, 1.0, 1.5], device=dev)) + 0.5 * soft_fss_loss(logits, yb, 9, torch.tensor([0.25, 0.75, 2.0], device=dev))
    if a.csi_w > 0: l = l + a.csi_w * soft_csi_loss(logits, yb)
    return l + a.aux_reg * reg

def tshift(t, dy, dx):
    """Shift (N,C,H,W) by dy, dx with zero fill, like the training translation augmentation."""
    out = torch.zeros_like(t); H, W = t.shape[2:]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[:, :, yd, xd] = t[:, :, ys, xs]
    return out
VIEWS = [(0, 0), (8, 0), (-8, 0), (0, 8), (0, -8)] if a.tta else [(0, 0)]
def predict(dl):
    """Probabilities; with --tta, the average over shifted copies, each shifted back (pixels a view lost are skipped)."""
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(dev); acc = 0; cnt = 0
            for dy, dx in VIEWS:
                with autocast(): logits = model(tshift(xb, dy, dx))
                acc = acc + tshift(torch.sigmoid(logits[:, :3].float()), -dy, -dx)
                cnt = cnt + tshift(tshift(torch.ones_like(xb[:, :1]), dy, dx), -dy, -dx)
            out.append((acc / cnt.clamp(min=1)).cpu().numpy())
    return np.concatenate(out)

def tune_thresholds(prob, obs):
    best = []
    for k in range(3):
        cands = np.arange(0.15, 0.71, 0.025)
        sc = [fss_pooled((prob[:, k:k+1] > t) * 10.0, obs[:, k:k+1])[0] for t in cands]
        best.append(float(cands[int(np.argmax(sc))]))
        print(f"  lead {k+1}: best thr {best[-1]:.3f} fss {max(sc):.4f}", flush=True)
    return best

import os; os.makedirs(a.out, exist_ok=True)
tag = ("" if a.arch == "Unet" else a.arch.lower() + "_") + (a.attention + "_" if a.attention else "") + f"{a.encoder}_s{a.seed}" + ("_full" if a.full else "")
def masks(prob, thr):
    """Rain maps (0 or 10 mm/h) from one threshold per lead, in float32 to keep memory low."""
    return np.stack([prob[:, j] > thr[j] for j in range(3)], 1).astype(np.float32) * np.float32(10.0)

# per-epoch validation probabilities go to disk (VAL_EPOCHS holds file names): keeping nine of them in
# RAM, plus the data, does not fit in Colab's 12.7 GB
best_score, hist, VAL_EPOCHS = -1, [], []
yv = y[val] if dl_va is not None else None
for ep in range(a.epochs):
    model.train(); t0 = time.time(); tl = 0
    for i, (xb, yb) in enumerate(dl_tr):
        xb, yb = xb.to(dev), yb.to(dev)
        with autocast(): logits = model(xb)
        loss = total_loss(logits.float(), yb)  # loss in float32 even with --amp
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        if i % 50 == 0: print(f"ep{ep} it{i}/{len(dl_tr)} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    msg = f"epoch {ep} train_loss {tl/len(dl_tr):.4f} time {time.time()-t0:.0f}s"
    if dl_va is not None:
        pv = predict(dl_va)
        s05 = fss_pooled(masks(pv, [0.5] * 3), yv)[0]; s03 = fss_pooled(masks(pv, [0.3] * 3), yv)[0]
        msg += f" val_fss@0.5 {s05:.4f} @0.3 {s03:.4f}"
        score = max(s05, s03)
        if score > best_score:
            best_score = score; torch.save(model.state_dict(), f"{a.out}/{tag}.pt"); np.save(f"{a.out}/{tag}_valprob.npy", pv.astype(np.float16))
    else:
        torch.save(model.state_dict(), f"{a.out}/{tag}.pt")
    print(msg, flush=True); hist.append(msg)
    if a.save_each:
        np.save(f"{a.out}/{tag}_ep{ep}_testprob.npy", predict(dl_te).astype(np.float16))
        if dl_va is not None:
            VAL_EPOCHS.append(f"{a.out}/{tag}_ep{ep}_valprob.npy"); np.save(VAL_EPOCHS[-1], pv.astype(np.float16))
    if dl_va is not None: del pv

if dl_va is not None:
    # free the training arrays (about 4 GB) before the epoch blends: Colab has 12.7 GB of RAM
    del dl_tr, ds_tr, dl_va, X, y; import gc; gc.collect()
    model.load_state_dict(torch.load(f"{a.out}/{tag}.pt"))
    pv = np.load(f"{a.out}/{tag}_valprob.npy").astype(np.float32)
    print("tuning thresholds on validation", flush=True)
    thr = tune_thresholds(pv, yv)
    report("unet tuned", masks(pv, thr), yv); del pv
    epoch_blends = {}
    if a.save_each and len(VAL_EPOCHS) > 1:
        # blends of the last k epochs: tells which epochs to average for the test prediction
        acc = np.load(VAL_EPOCHS[-1]).astype(np.float32)
        for k in range(2, len(VAL_EPOCHS) + 1):
            acc += np.load(VAL_EPOCHS[-k])  # running sum of the last k epochs, one epoch in memory at a time
            pb = acc / k
            print(f"blend of last {k} epochs", flush=True)
            tk = tune_thresholds(pb, yv)
            sc = report(f"  last{k} tuned", masks(pb, tk), yv)
            epoch_blends[f"last{k}"] = {"thr": tk, "val_fss": sc, "epochs": list(range(len(VAL_EPOCHS) - k, len(VAL_EPOCHS)))}
            np.save(f"{a.out}/{tag}_last{k}_valprob.npy", pb.astype(np.float16))
    json.dump({"thr": thr, "best_val": best_score, "hist": hist, "epoch_blends": epoch_blends}, open(f"{a.out}/{tag}.json", "w"), indent=1)
    for f in VAL_EPOCHS: os.remove(f)  # the last-k files now hold the same information
pt = predict(dl_te); np.save(f"{a.out}/{tag}_testprob.npy", pt.astype(np.float16))
print("saved test probs", pt.shape, flush=True)
