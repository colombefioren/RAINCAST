"""U-Net rain nowcasting trained to maximise FSS(1 mm/h, 3x3).
Inputs: 3 rain maps (log1p), lat, lon, hour sin/cos, season-day. Outputs: P(rain > 1 mm/h) at +1h,+2h,+3h.
Loss: BCE + soft FSS. Validation: last season (Dec 2022 - Mar 2023). Thresholds tuned on validation."""
import argparse, time, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from fss import fss_pooled, fss_per_sample, report
from advect import advect_batch

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", default="outputs")
p.add_argument("--encoder", default="resnet18")
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
p.add_argument("--advect", type=int, default=0, help="1: add optical-flow advected maps (+1h,+2h,+3h) as input channels")
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, flush=True)

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
lat_c = ((lats - lats.mean()) / lats.std()).astype(np.float32)
lon_c = ((lons - lons.mean()) / lons.std()).astype(np.float32)

def time_feats(t):
    h = t.hour.values + t.minute.values / 60.0
    doy = t.dayofyear.values.astype(np.float32)
    sd = np.where(doy > 300, doy - 335, doy + 30) / 120.0  # Dec 1 -> 0 ... Mar 31 -> ~1
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), sd], 1).astype(np.float32)

class DS(torch.utils.data.Dataset):
    def __init__(self, X, y, tf, aug=0, adv=None, mixup=0.0): self.X, self.y, self.tf, self.aug, self.adv, self.mixup = X, y, tf, aug, adv, mixup
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = np.log1p(np.maximum(self.X[i], 0)) / 2.0  # (3,H,W)
        H, W = x.shape[1:]
        tfe = np.broadcast_to(self.tf[i][:, None, None], (3, H, W))
        parts = [x, lat_c[None], lon_c[None], tfe]
        if self.adv is not None: parts.append(np.log1p(np.maximum(self.adv[i], 0)) / 2.0)
        inp = np.concatenate(parts, 0).astype(np.float32)
        tgt = (self.y[i] > 1.0).astype(np.float32) if self.y is not None else np.zeros((3, H, W), np.float32)
        if self.mixup > 0 and self.y is not None and i + 1 < len(self.X) and np.random.rand() < self.mixup:
            lam = np.random.rand()
            x2 = np.log1p(np.maximum(self.X[i + 1], 0)) / 2.0
            parts2 = [x2, lat_c[None], lon_c[None], np.broadcast_to(self.tf[i + 1][:, None, None], (3, H, W))]
            if self.adv is not None: parts2.append(np.log1p(np.maximum(self.adv[i + 1], 0)) / 2.0)
            inp = lam * inp + (1 - lam) * np.concatenate(parts2, 0).astype(np.float32)
            tgt = lam * tgt + (1 - lam) * (self.y[i + 1] > 1.0).astype(np.float32)
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
ds_tr = DS(X[tr_idx], y[tr_idx], tf_tr[tr_idx], a.aug, sub(ADV, tr_idx), a.mixup)
if a.wet_sampling:
    wet = (y[tr_idx] > 1.0).mean(axis=(1, 2, 3))
    wts = torch.tensor(0.02 + wet / max(wet.mean(), 1e-6), dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(wts, num_samples=len(tr_idx), replacement=True)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.bs, sampler=sampler, num_workers=2, drop_last=True)
else:
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.bs, shuffle=True, num_workers=2, drop_last=True)
dl_va = torch.utils.data.DataLoader(DS(X[val], y[val], tf_tr[val], 0, sub(ADV, val)), batch_size=64, num_workers=2) if val.any() else None
dl_te = torch.utils.data.DataLoader(DS(Xt, None, tf_te, 0, ADVt), batch_size=64, num_workers=2)

model = smp.Unet(a.encoder, encoder_weights="imagenet", in_channels=N_IN, classes=3).to(dev)
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
    l = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_w) + a.fss_w * soft_fss_loss(logits, yb)
    if a.ms_fss:
        l = l + 0.5 * soft_fss_loss(logits, yb, 5, torch.tensor([0.5, 1.0, 1.5], device=dev)) + 0.5 * soft_fss_loss(logits, yb, 9, torch.tensor([0.25, 0.75, 2.0], device=dev))
    if a.csi_w > 0: l = l + a.csi_w * soft_csi_loss(logits, yb)
    return l

def predict(dl):
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in dl:
            out.append(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
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
tag = f"{a.encoder}_s{a.seed}" + ("_full" if a.full else "")
best_score, hist, VAL_EPOCHS = -1, [], []
for ep in range(a.epochs):
    model.train(); t0 = time.time(); tl = 0
    for i, (xb, yb) in enumerate(dl_tr):
        xb, yb = xb.to(dev), yb.to(dev)
        logits = model(xb)
        loss = total_loss(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        if i % 50 == 0: print(f"ep{ep} it{i}/{len(dl_tr)} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    msg = f"epoch {ep} train_loss {tl/len(dl_tr):.4f} time {time.time()-t0:.0f}s"
    if dl_va is not None:
        pv = predict(dl_va); yv = y[val]
        s05 = fss_pooled((pv > 0.5) * 10.0, yv)[0]; s03 = fss_pooled((pv > 0.3) * 10.0, yv)[0]
        msg += f" val_fss@0.5 {s05:.4f} @0.3 {s03:.4f}"
        score = max(s05, s03)
        if score > best_score:
            best_score = score; torch.save(model.state_dict(), f"{a.out}/{tag}.pt"); np.save(f"{a.out}/{tag}_valprob.npy", pv.astype(np.float16))
    else:
        torch.save(model.state_dict(), f"{a.out}/{tag}.pt")
    print(msg, flush=True); hist.append(msg)
    if a.save_each:
        np.save(f"{a.out}/{tag}_ep{ep}_testprob.npy", predict(dl_te).astype(np.float16))
        if dl_va is not None: VAL_EPOCHS.append(pv.astype(np.float16))

if dl_va is not None:
    model.load_state_dict(torch.load(f"{a.out}/{tag}.pt"))
    pv = np.load(f"{a.out}/{tag}_valprob.npy").astype(np.float32); yv = y[val]
    print("tuning thresholds on validation", flush=True)
    thr = tune_thresholds(pv, yv)
    pred = np.stack([(pv[:, k] > thr[k]) * 10.0 for k in range(3)], 1)
    report("unet tuned", pred, yv)
    epoch_blends = {}
    if a.save_each and len(VAL_EPOCHS) > 1:
        # blends of the last k epochs: tells which epochs to average for the test prediction
        for k in range(2, len(VAL_EPOCHS) + 1):
            pb = np.mean([v.astype(np.float32) for v in VAL_EPOCHS[-k:]], axis=0)
            print(f"blend of last {k} epochs", flush=True)
            tk = tune_thresholds(pb, yv)
            sc = report(f"  last{k} tuned", np.stack([(pb[:, j] > tk[j]) * 10.0 for j in range(3)], 1), yv)
            epoch_blends[f"last{k}"] = {"thr": tk, "val_fss": sc, "epochs": list(range(len(VAL_EPOCHS) - k, len(VAL_EPOCHS)))}
            np.save(f"{a.out}/{tag}_last{k}_valprob.npy", pb.astype(np.float16))
    json.dump({"thr": thr, "best_val": best_score, "hist": hist, "epoch_blends": epoch_blends}, open(f"{a.out}/{tag}.json", "w"), indent=1)
pt = predict(dl_te); np.save(f"{a.out}/{tag}_testprob.npy", pt.astype(np.float16))
print("saved test probs", pt.shape, flush=True)
