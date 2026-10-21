"""Context-aware recalibration of blended probabilities before decoding (second stage, CPU only).

The decoder needs calibrated probabilities: "0.4" must mean rain 40 % of the time. A single isotonic
table per lead corrects only global distortions. This script fits a small LightGBM model per lead that
predicts rain at a pixel from the blended probability and its context: neighbourhood averages and
maxima of the probabilities, the three input rain maps, latitude, longitude, hour and season day.

Honest check: the validation season is split into alternate weeks. The model is fitted on one half
(random pixel sample) and the decoded FSS is measured on the other half, against isotonic calibration
fitted the same way. Only each sample's own maps are used (rule 2.4).

Usage (from src/): python stack_calibrate.py --mu 0.425,0.375,0.35 ../outputs/a_valprob.npy ...
"""
import argparse, time, numpy as np, pandas as pd, torch, torch.nn.functional as F, lightgbm as lgb
from fss import fss_pooled
from fss_decode import fit_calibration, calibrate, decode

p = argparse.ArgumentParser()
p.add_argument("--data", default="../data")
p.add_argument("--mu", default="0.425,0.375,0.35")
p.add_argument("--px", type=int, default=600, help="random pixels per sample used for fitting")
p.add_argument("--leads", default="0,1,2")
p.add_argument("probs", nargs="+")
a = p.parse_args()
torch.set_num_threads(12)
MU = [float(m) for m in a.mu.split(",")]

PV = np.mean([np.load(f).astype(np.float32) for f in a.probs], axis=0)  # (N,3,H,W)
yv = np.load(f"{a.data}/yval.npy")
tr = np.load(f"{a.data}/train.npz")
ts_all = pd.to_datetime(tr["times"]); val = np.asarray((ts_all >= "2022-12-01") & (ts_all < "2023-04-01"))
X = tr["X"][val]; ts = ts_all[val]
lats, lons = tr["lats"], tr["lons"]
N, _, H, W = PV.shape
week = ((ts - ts[0]).days // 7).values
halves = [week % 2 == 0, week % 2 == 1]
h = ts.hour.values + ts.minute.values / 60.0
doy = ts.dayofyear.values.astype(np.float32)
sd = (np.where(doy > 300, doy - 335, doy + 30) / 120.0).astype(np.float32)

def pool(t, k, kind="avg"):
    t = torch.from_numpy(t)[:, None]
    out = F.avg_pool2d(t, k, 1, k // 2) if kind == "avg" else F.max_pool2d(t, k, 1, k // 2)
    return out[:, 0].numpy()

def features(idx, k):
    """Feature maps (n, F, H, W) for samples idx and lead k."""
    pk = PV[idx, k]
    rain = np.log1p(np.maximum(X[idx], 0)).astype(np.float32)  # (n,3,H,W)
    n = len(idx)
    cols = [pk, pool(pk, 3), pool(pk, 7), pool(pk, 15), pool(pk, 5, "max"),
            PV[idx, 0], PV[idx, 2],
            rain[:, 2], pool(rain[:, 2], 7), rain[:, 0],
            np.broadcast_to(lats, (n, H, W)), np.broadcast_to(lons, (n, H, W)),
            np.broadcast_to(np.sin(2 * np.pi * h[idx] / 24)[:, None, None], (n, H, W)),
            np.broadcast_to(np.cos(2 * np.pi * h[idx] / 24)[:, None, None], (n, H, W)),
            np.broadcast_to(sd[idx][:, None, None], (n, H, W)),
            np.broadcast_to(pk.mean(axis=(1, 2))[:, None, None], (n, H, W))]
    return np.stack(cols, 1).astype(np.float32)

NAMES = ["p", "p_box3", "p_box7", "p_box15", "p_max5", "p_lead1", "p_lead3", "rain_t", "rain_t_box7",
         "rain_t-2h", "lat", "lon", "hour_sin", "hour_cos", "season_day", "p_sample_mean"]
rng = np.random.default_rng(0)
res = {"iso": [], "gbm": []}
for k in [int(x) for x in a.leads.split(",")]:
    for hv in range(2):
        A, B = np.where(halves[hv])[0], np.where(halves[1 - hv])[0]
        t0 = time.time()
        # fit on a random pixel sample of half A
        rows, ys = [], []
        for c in range(0, len(A), 256):
            ia = A[c:c + 256]; fa = features(ia, k)
            yy = rng.integers(0, H, (len(ia), a.px)); xx = rng.integers(0, W, (len(ia), a.px))
            ii = np.repeat(np.arange(len(ia)), a.px)
            rows.append(fa[ii, :, yy.ravel(), xx.ravel()]); ys.append(yv[ia, k][ii, yy.ravel(), xx.ravel()] > 1.0)
        Xa, ya = np.concatenate(rows), np.concatenate(ys)
        gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=12)
        gbm.fit(Xa, ya)
        # predict every pixel of half B
        pb = np.zeros((len(B), H, W), np.float32)
        for c in range(0, len(B), 64):
            ib = B[c:c + 64]; fb = features(ib, k)
            flat = fb.transpose(0, 2, 3, 1).reshape(-1, fb.shape[1])
            pb[c:c + 64] = gbm.predict_proba(flat)[:, 1].reshape(len(ib), H, W)
        table = fit_calibration(PV[A, k], yv[A, k] > 1.0)
        s_iso = fss_pooled(decode(calibrate(PV[B, k], table), MU[k])[:, None] * 10.0, yv[B, k:k + 1])[0]
        s_gbm = fss_pooled(decode(pb, MU[k])[:, None] * 10.0, yv[B, k:k + 1])[0]
        res["iso"].append(s_iso); res["gbm"].append(s_gbm)
        imp = sorted(zip(gbm.feature_importances_, NAMES), reverse=True)[:5]
        print(f"lead {k + 1} half {hv}: isotonic {s_iso:.4f} | gbm {s_gbm:.4f} | top features {[n for _, n in imp]} ({time.time() - t0:.0f}s)", flush=True)
print("mean isotonic", np.mean(res["iso"]).round(4), "mean gbm", np.mean(res["gbm"]).round(4))
