"""Measure test-time augmentation (TTA) by translation on a trained checkpoint.

The model was trained with random translations (aug=2), so it should give a
similar answer when the whole input stack is shifted by a few pixels. TTA
predicts on several shifted copies, shifts each prediction back, and averages
the probabilities. Pixels that fall outside the map for a view are left out of
that view's average.

Usage: python src/eval_tta.py --data data --ckpt outputs/kup2/resnet34_s31.pt \
           --upscale 2 --dropout 0.2 --every 4 --shift 8
"""
import argparse, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from fss import fss_pooled

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--ckpt", required=True)
p.add_argument("--encoder", default="resnet34")
p.add_argument("--upscale", type=int, default=1)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--every", type=int, default=4, help="use every k-th validation sample (CPU time)")
p.add_argument("--shift", type=int, nargs="+", default=[8], help="shift sizes in pixels")
p.add_argument("--threads", type=int, default=12)
a = p.parse_args()
torch.set_num_threads(a.threads)

tr = np.load(f"{a.data}/train.npz")
ts = pd.to_datetime(tr["times"])
val = np.where(np.asarray((ts >= "2022-12-01") & (ts < "2023-04-01")))[0][::a.every]
X, y = tr["X"][val], tr["y"][val]
lats, lons = tr["lats"], tr["lons"]
lat_c = ((lats - lats.mean()) / lats.std()).astype(np.float32)
lon_c = ((lons - lons.mean()) / lons.std()).astype(np.float32)
t = ts[val]
h = t.hour.values + t.minute.values / 60.0
doy = t.dayofyear.values.astype(np.float32)
sd = np.where(doy > 300, doy - 335, doy + 30) / 120.0
tf = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), sd], 1).astype(np.float32)
N, _, H, W = X.shape
inp = np.concatenate([np.log1p(np.maximum(X, 0)) / 2.0,
                      np.broadcast_to(lat_c, (N, 1, H, W)), np.broadcast_to(lon_c, (N, 1, H, W)),
                      np.broadcast_to(tf[:, :, None, None], (N, 3, H, W))], 1).astype(np.float32)

model = smp.Unet(a.encoder, encoder_weights=None, in_channels=8, classes=3)
if a.dropout > 0: model.segmentation_head = nn.Sequential(nn.Dropout2d(a.dropout), model.segmentation_head)
if a.upscale > 1:
    class Upscaled(nn.Module):
        def __init__(self, m, f): super().__init__(); self.m, self.f = m, f
        def forward(self, x): return F.avg_pool2d(self.m(F.interpolate(x, scale_factor=self.f, mode="bilinear", align_corners=False)), self.f)
    model = Upscaled(model, a.upscale)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu")); model.eval()

def shift(arr, dy, dx):
    """Shift (N,C,H,W) by dy, dx with zero fill, same as the training augmentation."""
    out = torch.zeros_like(arr)
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[:, :, yd, xd] = arr[:, :, ys, xs]
    return out

views = [(0, 0)] + [v for s in a.shift for v in ((s, 0), (-s, 0), (0, s), (0, -s))]
probs = {}
t0 = time.time()
with torch.no_grad():
    for dy, dx in views:
        acc = []
        for i in range(0, N, 32):
            xb = shift(torch.from_numpy(inp[i:i + 32]), dy, dx)
            pr = torch.sigmoid(model(xb))
            acc.append(shift(pr, -dy, -dx).numpy())
        probs[(dy, dx)] = np.concatenate(acc)
        print(f"view {dy:+d},{dx:+d} done {time.time() - t0:.0f}s", flush=True)

def tuned(pr):
    best = []
    for k in range(3):
        sc = [fss_pooled((pr[:, k:k + 1] > c) * 10.0, y[:, k:k + 1])[0] for c in np.arange(0.3, 0.71, 0.025)]
        best.append(max(sc))
    return np.mean(best), np.round(best, 4)

ones = torch.ones(1, 1, H, W)
def tta(vs):
    num = sum(probs[v] for v in vs)
    cnt = sum(shift(shift(ones, *v), -v[0], -v[1]).numpy() for v in vs)  # 1 where the view covers the pixel
    num = sum(probs[v] for v in vs)
    return num / np.maximum(cnt, 1)

print("no TTA     ", *tuned(probs[(0, 0)]))
for s in a.shift:
    vs = [(0, 0), (s, 0), (-s, 0), (0, s), (0, -s)]
    print(f"TTA shift {s:2d}", *tuned(tta(vs)))
if len(a.shift) > 1:
    print("TTA all    ", *tuned(tta(views)))
