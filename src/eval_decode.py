"""Compare per-pixel thresholds with the expected-FSS decoder (src/fss_decode.py) on validation.

Honest estimate: the validation season is split into two halves by alternating weeks. Calibration,
thresholds and mu are tuned on one half and scored on the other, then the halves swap.
Usage: python src/eval_decode.py outputs/kup2/resnet34_s31_last5_valprob.npy
"""
import sys, time, numpy as np, pandas as pd
from fss import fss_pooled
from fss_decode import fit_calibration, calibrate, decode

pv = np.load(sys.argv[1]).astype(np.float32)
y = np.load("../data/yval.npy") if len(sys.argv) < 3 else np.load(sys.argv[2])
tr = np.load("../data/train.npz"); ts = pd.to_datetime(tr["times"])
ts = ts[np.asarray((ts >= "2022-12-01") & (ts < "2023-04-01"))]
week = ((ts - ts[0]).days // 7).values
halves = [week % 2 == 0, week % 2 == 1]
THR = np.arange(0.3, 0.71, 0.025); MU = np.arange(0.20, 0.551, 0.025)

def score(mask, k, idx): return fss_pooled(mask[:, None] * 10.0, y[idx, k:k + 1])[0]

res = {"thr": [], "dec": []}
for h in range(2):
    A, B = halves[h], halves[1 - h]
    for k in range(3):
        t0 = time.time()
        tb = THR[int(np.argmax([score(pv[A, k] > t, k, A) for t in THR]))]
        s_thr = score(pv[B, k] > tb, k, B)
        table = fit_calibration(pv[A, k], y[A, k] > 1.0)
        cA, cB = calibrate(pv[A, k], table), calibrate(pv[B, k], table)
        sc = [score(decode(cA, m), k, A) for m in MU]
        mb = MU[int(np.argmax(sc))]
        s_dec = score(decode(cB, mb), k, B)
        res["thr"].append(s_thr); res["dec"].append(s_dec)
        print(f"half {h} lead {k + 1}: threshold {tb:.3f} -> {s_thr:.4f} | decoder mu {mb:.3f} -> {s_dec:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
print("mean threshold", np.mean(res["thr"]).round(4), "mean decoder", np.mean(res["dec"]).round(4))
