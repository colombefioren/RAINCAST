"""Evaluate blends of saved validation probability maps and tune per-lead thresholds.
Usage: python src/blend.py --data data --out outputs/blend.json [--dilate 0] prob1_valprob.npy prob2_valprob.npy ...
Each *_valprob.npy must have a sibling *_testprob.npy; the blended test prediction is written next to --out."""
import argparse, json, numpy as np, pandas as pd
from scipy.ndimage import uniform_filter
from fss import fss_pooled, report

p = argparse.ArgumentParser()
p.add_argument("--data", required=True); p.add_argument("--out", required=True)
p.add_argument("--smooth", type=int, default=0, help="average probabilities over k x k window before thresholding")
p.add_argument("--weights", default="", help="comma-separated blend weights, default equal")
p.add_argument("probs", nargs="+")
a = p.parse_args()

import os
YV = f"{a.data}/yval.npy"  # cached validation targets (Dec 2022 - Mar 2023), avoids reloading 4 GB each time
if os.path.exists(YV):
    yv = np.load(YV)
else:
    tr = np.load(f"{a.data}/train.npz"); ts = pd.to_datetime(tr["times"])
    val = np.asarray((ts >= "2022-12-01") & (ts < "2023-04-01")); yv = (tr["y"][val] > 1.0).astype(np.float16) * 10
    np.save(YV, yv)
w = np.array([float(x) for x in a.weights.split(",")]) if a.weights else np.ones(len(a.probs))
w = w / w.sum()
for f in a.probs:
    pv = np.load(f).astype(np.float32); report(f.split("/")[-1][:28], (pv > 0.5) * 10.0, yv)
PV = sum(wi * np.load(f).astype(np.float32) for wi, f in zip(w, a.probs))
PT = sum(wi * np.load(f.replace("_valprob", "_testprob")).astype(np.float32) for wi, f in zip(w, a.probs))
if a.smooth > 1:
    PV = uniform_filter(PV, size=(1, 1, a.smooth, a.smooth), mode="nearest")
    PT = uniform_filter(PT, size=(1, 1, a.smooth, a.smooth), mode="nearest")
thr, best = [], []
for k in range(3):
    cands = np.arange(0.15, 0.76, 0.025)
    sc = [fss_pooled((PV[:, k:k+1] > t) * 10.0, yv[:, k:k+1])[0] for t in cands]
    thr.append(float(cands[int(np.argmax(sc))])); best.append(float(max(sc)))
    print(f"lead {k+1}: thr {thr[-1]:.3f} fss {best[-1]:.4f}")
pred = np.stack([(PV[:, k] > thr[k]) * 10.0 for k in range(3)], 1)
score = report("blend tuned", pred, yv)
json.dump({"thr": thr, "val_fss": score, "per_lead": best, "probs": a.probs, "weights": w.tolist(), "smooth": a.smooth},
          open(a.out, "w"), indent=1)
np.save(a.out.replace(".json", "_testprob.npy"), PT.astype(np.float32))
print("saved", a.out, "and blended test probs")
