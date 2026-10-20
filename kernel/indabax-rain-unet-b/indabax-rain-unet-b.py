KERNEL_ARGS = '''--data /kaggle/input/competitions/hackathon-IndabaX-MDG-2026 --out /kaggle/working --threads 4 --save_each --encoder resnet18 --epochs 6 --lr 5e-4 --aug 1 --dropout 0.2 --stride 2 --seed 7'''.split()
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

import argparse, time, json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", default="outputs")
p.add_argument("--encoder", default="resnet18")
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--fss_w", type=float, default=1.0)
p.add_argument("--full", action="store_true", help="train on all data (no validation), for final submission")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--limit", type=int, default=0, help="debug: use only first N train samples")
p.add_argument("--threads", type=int, default=12)
p.add_argument("--aug", type=int, default=0, help="1: random flips + 90-degree rotations on train")
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--stride", type=int, default=1, help="use every k-th hourly train sample")
p.add_argument("--save_each", action="store_true", help="save test/val probs at every epoch")
a = p.parse_args(KERNEL_ARGS)
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(a.threads)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, flush=True)

tr = np.load(f"{a.data}/train.npz"); te = np.load(f"{a.data}/test.npz")
X, y, times = tr["X"], tr["y"], tr["times"]; Xt, tt = te["X"], te["times"]
lats, lons = tr["lats"], tr["lons"]
if a.limit: X, y, times = X[:a.limit], y[:a.limit], times[:a.limit]
ts = pd.to_datetime(times); tst = pd.to_datetime(tt)
val = np.asarray((ts >= "2022-12-01") & (ts < "2023-04-01")) if not a.full else np.zeros(len(ts), bool)

lat_c = ((lats - lats.mean()) / lats.std()).astype(np.float32)
lon_c = ((lons - lons.mean()) / lons.std()).astype(np.float32)

def time_feats(t):
    h = t.hour.values + t.minute.values / 60.0
    doy = t.dayofyear.values.astype(np.float32)
    sd = np.where(doy > 300, doy - 335, doy + 30) / 120.0  # Dec 1 -> 0 ... Mar 31 -> ~1
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), sd], 1).astype(np.float32)

class DS(torch.utils.data.Dataset):
    def __init__(self, X, y, tf, aug=0): self.X, self.y, self.tf, self.aug = X, y, tf, aug
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = np.log1p(np.maximum(self.X[i], 0)) / 2.0  # (3,H,W)
        H, W = x.shape[1:]
        tfe = np.broadcast_to(self.tf[i][:, None, None], (3, H, W))
        inp = np.concatenate([x, lat_c[None], lon_c[None], tfe], 0).astype(np.float32)
        tgt = (self.y[i] > 1.0).astype(np.float32) if self.y is not None else np.zeros((3, H, W), np.float32)
        if self.aug:
            k = np.random.randint(4)
            inp, tgt = np.rot90(inp, k, axes=(1, 2)), np.rot90(tgt, k, axes=(1, 2))
            if np.random.rand() < 0.5: inp, tgt = inp[:, :, ::-1], tgt[:, :, ::-1]
            inp, tgt = np.ascontiguousarray(inp), np.ascontiguousarray(tgt)
        return torch.from_numpy(inp), torch.from_numpy(tgt)

tf_tr, tf_te = time_feats(ts), time_feats(tst)
tr_idx = np.where(~val)[0][::a.stride]
dl_tr = torch.utils.data.DataLoader(DS(X[tr_idx], y[tr_idx], tf_tr[tr_idx], a.aug), batch_size=a.bs, shuffle=True, num_workers=2, drop_last=True)
dl_va = torch.utils.data.DataLoader(DS(X[val], y[val], tf_tr[val]), batch_size=64, num_workers=2) if val.any() else None
dl_te = torch.utils.data.DataLoader(DS(Xt, None, tf_te), batch_size=64, num_workers=2)

model = smp.Unet(a.encoder, encoder_weights="imagenet", in_channels=8, classes=3).to(dev)
if a.dropout > 0:
    model.segmentation_head = nn.Sequential(nn.Dropout2d(a.dropout), model.segmentation_head)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
steps = a.epochs * len(dl_tr)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
pos_w = torch.tensor([8.0], device=dev)

def soft_fss_loss(logits, tgt):
    pr = torch.sigmoid(logits)
    pf = F.avg_pool2d(pr, 3, stride=1, padding=1)
    po = F.avg_pool2d(tgt, 3, stride=1, padding=1)
    num = ((pf - po) ** 2).sum(dim=(0, 2, 3))
    den = (pf ** 2).sum(dim=(0, 2, 3)) + (po ** 2).sum(dim=(0, 2, 3)) + 1e-6
    return (num / den).mean()

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
best_score, hist = -1, []
for ep in range(a.epochs):
    model.train(); t0 = time.time(); tl = 0
    for i, (xb, yb) in enumerate(dl_tr):
        xb, yb = xb.to(dev), yb.to(dev)
        logits = model(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_w) + a.fss_w * soft_fss_loss(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
        if i % 50 == 0: print(f"ep{ep} it{i}/{len(dl_tr)} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    msg = f"epoch {ep} train_loss {tl/len(dl_tr):.4f} time {time.time()-t0:.0f}s"
    if dl_va is not None:
        pv = predict(dl_va); yv = y[val]
        s05 = fss_pooled((pv > 0.5) * 10.0, yv)[0]; s03 = fss_pooled((pv > 0.3) * 10.0, yv)[0]
        msg += f" val_fss@0.5 {s05:.4f} @0.3 {s03:.4f}"
        score = max(s05, s03)
        if score > best_score:
            best_score = score; torch.save(model.state_dict(), f"{a.out}/{tag}.pt"); np.save(f"{a.out}/{tag}_valprob.npy", pv)
    else:
        torch.save(model.state_dict(), f"{a.out}/{tag}.pt")
    print(msg, flush=True); hist.append(msg)
    if a.save_each:
        np.save(f"{a.out}/{tag}_ep{ep}_testprob.npy", predict(dl_te))
        if dl_va is not None: np.save(f"{a.out}/{tag}_ep{ep}_valprob.npy", pv)

if dl_va is not None:
    model.load_state_dict(torch.load(f"{a.out}/{tag}.pt"))
    pv = np.load(f"{a.out}/{tag}_valprob.npy"); yv = y[val]
    print("tuning thresholds on validation", flush=True)
    thr = tune_thresholds(pv, yv)
    pred = np.stack([(pv[:, k] > thr[k]) * 10.0 for k in range(3)], 1)
    report("unet tuned", pred, yv)
    json.dump({"thr": thr, "best_val": best_score, "hist": hist}, open(f"{a.out}/{tag}.json", "w"), indent=1)
pt = predict(dl_te); np.save(f"{a.out}/{tag}_testprob.npy", pt)
print("saved test probs", pt.shape, flush=True)
